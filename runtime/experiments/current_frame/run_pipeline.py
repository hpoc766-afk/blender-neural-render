"""Bound single-render Blender -> PyTorch -> original compositor workflow."""
import argparse
import hashlib
import json
import subprocess
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import OpenEXR
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from dlss5_pipeline.godot_adapter import GodotStaticAdapter
from frame_contract import CAPTURE_VERSION, digest, sha, validate_cache, write_json

BLENDER = None  # Bound to bpy.app.binary_path by the extension worker.
CHANNELS = ['linear_hdr_r', 'linear_hdr_g', 'linear_hdr_b', 'alpha']
PNG_QUANTIZATION_BUDGET = 1  # One code value in the declared 8-bit display format.
LINEAR_REFERENCE_BUDGET = 1e-6  # Independent of the old failed comparison.


def blender_stage(source, out, script, phase, log):
    bootstrap = HERE.parents[1] / 'blender_stage.py'
    command = [str(BLENDER), '--background', '--python-exit-code', '1',
               '--python', str(bootstrap), '--', str(source), str(HERE / script), str(out), phase]
    with (out / log).open('w', encoding='utf-8') as stream:
        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)


def rgba_part(path, selector=None):
    """Select the recorded part and channel names; never pick the first RGB part."""
    with OpenEXR.File(str(path), separate_channels=True) as file:
        candidates = []
        for part in file.parts:
            header = part.header
            name = str(header.get('name', ''))
            keys = {key.rsplit('.', 1)[-1]: key for key in part.channels}
            if len(keys) != len(part.channels):
                continue  # Ambiguous flat parts with more than one RGB namespace.
            if selector is not None:
                if name != selector['part'] or keys != selector['channels']:
                    continue
            elif name not in {'', 'Combined'} and not name.endswith('.Combined'):
                continue
            if set(keys) != {'R', 'G', 'B', 'A'}:
                continue
            if header.get('colorInteropID') != 'lin_rec709_scene':
                raise RuntimeError('unsupported_capability: model supports declared Linear Rec.709 only')
            pixels = np.stack([part.channels[keys[c]].pixels for c in 'RGBA'], axis=-1)
            if pixels.dtype != np.float32 or not np.isfinite(pixels).all():
                raise RuntimeError('Finite FLOAT32 RGBA required')
            region = {key: [list(map(int, vector)) for vector in header[key]]
                      for key in ('dataWindow', 'displayWindow')}
            candidates.append((pixels.copy(), {'part': name, 'channels': keys,
                                               'colorInteropID': header['colorInteropID'], **region}))
    if len(candidates) != 1:
        raise RuntimeError(f'Expected one explicitly bound Combined part, found {len(candidates)}')
    return candidates[0]


def inspect_capture(path, identity):
    rgba, selection = rgba_part(path)
    h, w = identity['dimensions_hw']
    expected_window = [[0, 0], [w - 1, h - 1]]
    if rgba.shape != (h, w, 4) or any(selection[k] != expected_window for k in ('dataWindow', 'displayWindow')):
        raise RuntimeError('unsupported_capability: EXR extent/window differs from the source view')
    parts = []
    with OpenEXR.File(str(path), separate_channels=True) as file:
        for part in file.parts:
            for key in ('dataWindow', 'displayWindow'):
                window = [list(map(int, item)) for item in part.header[key]]
                if window != expected_window:
                    raise RuntimeError('AOV windows differ from the source view')
            if any(channel.pixels.dtype != np.float32 for channel in part.channels.values()):
                raise RuntimeError('Lossless FLOAT32 source passes required')
            parts.append({'part': part.header.get('name', ''), 'channels': list(part.channels),
                          'colorInteropID': part.header.get('colorInteropID', 'unknown')})
    if selection['part'] != identity['view_layer'] + '.Combined':
        raise RuntimeError('Captured Combined part is not owned by the declared source View Layer')
    cache_layer = identity['view_layer']
    return rgba, selection, parts, cache_layer


def write_hdr(path, rgba, session):
    if rgba.dtype != np.float32 or rgba.ndim != 3 or rgba.shape[2] != 4 or not np.isfinite(rgba).all():
        raise ValueError('Finite native FLOAT32 RGBA required')
    header = {'colorInteropID': 'lin_rec709_scene', 'alpha_mode': 'premultiplied',
              'pipeline_capture_session': session}
    with OpenEXR.File(header, {c: np.ascontiguousarray(rgba[..., i]) for i, c in enumerate('RGBA')}) as file:
        file.write(str(path))


def bind_branch(out, mode, manifest):
    path = out / (mode + '_hdr.exr')
    write_json(out / (mode + '_input.json'), {
        'file': path.name, 'sha256': sha(path),
        'capture_session_id': manifest['capture_session_id'],
        'identity_sha256': manifest['identity_sha256'],
        'dimensions_hw': manifest['identity']['dimensions_hw'],
        'source_alpha_sha256': manifest['alpha_channel_sha256'],
        'colorInteropID': 'lin_rec709_scene', 'alpha_mode': 'premultiplied'})


def reference_check(out, branch):
    png = 'original_rebuilt.png' if branch == 'baseline' else branch + '.png'
    expected = np.asarray(Image.open(out / 'native_reference.png')).astype(np.int16)
    actual = np.asarray(Image.open(out / png)).astype(np.int16)
    if actual.shape != expected.shape:
        raise RuntimeError('Native reference and cached output shapes differ')
    native, _ = rgba_part(out / 'native_linear.exr')
    rebuilt, _ = rgba_part(out / (branch + '_linear.exr'))
    if native.shape != rebuilt.shape:
        raise RuntimeError('Native reference and cached linear output shapes differ')
    stats = {'png_max_code_difference': int(np.abs(actual - expected).max()),
             'png_code_budget': PNG_QUANTIZATION_BUDGET,
             'linear_max_abs_difference': float(np.abs(native - rebuilt).max()),
             'linear_budget': LINEAR_REFERENCE_BUDGET}
    stats['passed'] = (stats['png_max_code_difference'] <= PNG_QUANTIZATION_BUDGET
                       and stats['linear_max_abs_difference'] <= LINEAR_REFERENCE_BUDGET)
    write_json(out / (branch + '_reference_check.json'), stats)
    if not stats['passed']:
        raise RuntimeError(f'Same-render native reference validation failed: {branch}: {stats}')
    return stats


def run(args):
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / 'pipeline_report.json').unlink(missing_ok=True)
    source = args.source.resolve()
    source_before = sha(source)
    started = time.perf_counter()
    write_json(out / 'request.json', {'source': str(source), 'scene': args.scene,
                                    'frame': args.frame, 'view_layer': args.view_layer,
                                    'neural_node': getattr(args, 'neural_node', None)})
    print('PREFLIGHT_START', source, flush=True)
    blender_stage(source, out, 'capture_raw.py', 'inspect', 'preflight.log')
    identity = json.loads((out / 'preflight.json').read_text(encoding='utf-8'))
    if args.reuse_raw:
        manifest_path = out / 'capture_manifest.json'
        if not manifest_path.exists():
            raise RuntimeError('Legacy/unbound cache rejected; render again without --reuse-raw')
        manifest = validate_cache(json.loads(manifest_path.read_text(encoding='utf-8')), identity, out)
        rgba, selection, parts, layer = inspect_capture(out / 'engine_raw.exr', identity)
        if selection != manifest['combined'] or parts != manifest['parts'] or layer != manifest['cache_layer']:
            raise RuntimeError('Raw resource selection or metadata differs from the cache manifest')
        source_renders = 0
    else:
        (out / 'capture_manifest.json').unlink(missing_ok=True)
        print('REUSE_ORIGINAL_RENDERER', source, flush=True)
        blender_stage(source, out, 'capture_raw.py', 'capture', 'raw.log')
        capture = json.loads((out / 'raw_render.json').read_text(encoding='utf-8'))
        if capture['identity'] != identity or capture['source_render_invocations'] != 1:
            raise RuntimeError('Source capture identity or render count mismatch')
        rgba, selection, parts, layer = inspect_capture(out / 'engine_raw.exr', identity)
        manifest = {'version': CAPTURE_VERSION, 'status': 'complete', 'identity': identity,
                    'identity_sha256': digest(identity), 'capture_session_id': str(uuid.uuid4()),
                    'combined': selection, 'parts': parts, 'cache_layer': layer,
                    'alpha_channel_sha256': hashlib.sha256(np.ascontiguousarray(rgba[..., 3], dtype='<f4').tobytes()).hexdigest(),
                    'source_bindings': capture['source_bindings'], 'assets': {
                        key: {'file': filename, 'sha256': sha(out / filename)}
                        for key, filename in [('raw_exr', 'engine_raw.exr'), ('native_png', 'native_reference.png'),
                                               ('native_linear_exr', 'native_linear.exr')]}}
        write_json(out / 'capture_manifest.json', manifest)
        validate_cache(manifest, identity, out)
        source_renders = 1
    if hashlib.sha256(np.ascontiguousarray(rgba[..., 3], dtype='<f4').tobytes()).hexdigest() != manifest['alpha_channel_sha256']:
        raise RuntimeError('Cached source alpha differs from its capture identity')
    print('BOUND_CACHE_VALIDATED', manifest['capture_session_id'], identity['view_layer'], flush=True)
    np.save(out / 'scene_linear.npy', rgba)
    blender_stage(source, out, 'post_stage.py', 'baseline', 'baseline_post.log')
    baseline_check = reference_check(out, 'baseline')
    # Exercise the real adapter restore path with equal model tensors. Negative
    # HDR, >1 highlights and alpha remain the original native values, bit for bit.
    identity_adapter = GodotStaticAdapter.__new__(GodotStaticAdapter)
    identity_adapter.strength, identity_adapter.exposure, identity_adapter.hdr_ceiling = 1.0, 1.0, 64.0
    mapped = np.maximum(rgba[..., :3], 0)
    mapped = mapped / (1 + mapped)
    identity_rgb, identity_stats = identity_adapter.restore_hdr(rgba, CHANNELS, mapped, mapped)
    identity_rgba = rgba.copy()
    identity_rgba[..., :3] = identity_rgb
    if not np.array_equal(identity_rgba, rgba):
        raise RuntimeError('Identity adapter changed native HDR or alpha')
    write_hdr(out / 'identity_hdr.exr', identity_rgba, manifest['capture_session_id'])
    bind_branch(out, 'identity', manifest)
    blender_stage(source, out, 'post_stage.py', 'identity', 'identity_post.log')
    identity_check = reference_check(out, 'identity')
    inference_stats = None
    if not args.identity_only:
        validate_cache(manifest, identity, out)
        if args.strength == 0:
            neural_rgba = rgba.copy()
            inference_stats = {'model_skipped': 'strength_zero', 'hdr_mae': 0.0}
        else:
            checkpoint = getattr(args, 'checkpoint', None) or ROOT / 'assets/dlss5_static.pt'
            adapter = GodotStaticAdapter(checkpoint, cuda_graph=not args.eager, strength=args.strength)
            model_source, model_output, _, inference_stats = adapter.infer(rgba, CHANNELS)
            restored, restoration = adapter.restore_hdr(rgba, CHANNELS, model_source, model_output)
            neural_rgba = rgba.copy()
            neural_rgba[..., :3] = restored
            inference_stats.update(restoration)
            inference_stats['checkpoint_sha256'] = sha(checkpoint)
            inference_stats['checkpoint_path'] = str(checkpoint.resolve())
            inference_stats['model_kind'] = 'static_approximation'
            np.save(out / 'model_input.npy', model_source)
            np.save(out / 'model_output.npy', model_output)
        if not np.array_equal(neural_rgba[..., 3], rgba[..., 3]):
            raise RuntimeError('Neural branch changed source alpha')
        write_hdr(out / 'neural_hdr.exr', neural_rgba, manifest['capture_session_id'])
        bind_branch(out, 'neural', manifest)
        blender_stage(source, out, 'post_stage.py', 'neural', 'neural_post.log')
    reports = {mode: json.loads((out / (mode + '_post.json')).read_text(encoding='utf-8'))
               for mode in ['baseline', 'identity'] + ([] if args.identity_only else ['neural'])}
    if any(report['actual_geometry_render_calls'] != 0 or report['cache_session_id'] != manifest['capture_session_id']
           or report['cache_raw_sha256'] != manifest['assets']['raw_exr']['sha256'] for report in reports.values()):
        raise RuntimeError('Branch source identity or actual renderer guard validation failed')
    if sha(source) != source_before:
        raise RuntimeError('Source Blender file changed')
    report = {'status': 'passed', 'scope': 'identity_only' if args.identity_only else 'static_neural',
              'source': str(source), 'source_sha256': source_before, 'source_file_unchanged': True,
              'identity_sha256': manifest['identity_sha256'], 'capture_session_id': manifest['capture_session_id'],
              'source_geometry_render_invocations_this_run': source_renders,
              'postprocess_actual_geometry_render_calls': 0, 'native_capture_passes': parts,
              'baseline_reference_check': baseline_check, 'identity_reference_check': identity_check,
              'identity_restore': identity_stats, 'inference': inference_stats,
              'wall_seconds': time.perf_counter() - started, 'branches': reports}
    write_json(out / 'pipeline_report.json', report)
    (out / 'error.json').unlink(missing_ok=True)
    print('PIPELINE_COMPLETE', json.dumps(report, ensure_ascii=True), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--blender', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--scene')
    parser.add_argument('--frame', type=int)
    parser.add_argument('--view-layer')
    parser.add_argument('--neural-node')
    parser.add_argument('--reuse-raw', action='store_true')
    parser.add_argument('--eager', action='store_true')
    parser.add_argument('--strength', type=float, default=1.0)
    parser.add_argument('--identity-only', action='store_true')
    args = parser.parse_args()
    global BLENDER
    BLENDER = args.blender
    if not np.isfinite(args.strength) or not 0 <= args.strength <= 1:
        parser.error('strength must be finite and in [0,1]')
    try:
        run(args)
    except Exception as error:
        args.out.mkdir(parents=True, exist_ok=True)
        write_json(args.out / 'error.json', {'status': 'failed', 'type': type(error).__name__, 'error': str(error)})
        raise


if __name__ == '__main__':
    main()
