"""Pure-Python cache identity checks shared with Blender subprocesses."""
import hashlib
import json
from pathlib import Path

CAPTURE_VERSION = 2


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def validate_cache(manifest, identity, directory):
    """Reject legacy, partial, stale or corrupted data before model execution."""
    if (manifest.get('version') != CAPTURE_VERSION or manifest.get('status') != 'complete'
            or not manifest.get('capture_session_id')
            or len(manifest.get('alpha_channel_sha256', '')) != 64):
        raise RuntimeError('Unsupported or incomplete cache; recapture the source frame')
    if manifest.get('identity') != identity:
        raise RuntimeError('Cache identity mismatch: scene, frame, layer, dependencies or settings changed')
    if manifest.get('identity_sha256') != digest(identity):
        raise RuntimeError('Invalid cache identity digest')
    directory = Path(directory).resolve()
    filenames = {'raw_exr': 'engine_raw.exr', 'native_png': 'native_reference.png',
                 'native_linear_exr': 'native_linear.exr'}
    if set(manifest.get('assets', {})) != set(filenames):
        raise RuntimeError('Cache lacks the same-render native reference or raw resources')
    for role, asset in manifest['assets'].items():
        path = (directory / asset['file']).resolve()
        if (asset['file'] != filenames[role] or path.parent != directory
                or not path.is_file() or sha(path) != asset['sha256']):
            raise RuntimeError(f'Missing, redirected or corrupted cache asset: {asset["file"]}')
    return manifest


def validate_branch_input(record, manifest, directory, mode):
    """Bind an injected RGB image to the same immutable native capture."""
    if (record.get('capture_session_id') != manifest['capture_session_id']
            or record.get('identity_sha256') != manifest['identity_sha256']
            or record.get('dimensions_hw') != manifest['identity']['dimensions_hw']
            or record.get('source_alpha_sha256') != manifest['alpha_channel_sha256']
            or record.get('colorInteropID') != 'lin_rec709_scene'
            or record.get('alpha_mode') != 'premultiplied'):
        raise RuntimeError('Injected HDR source identity, alpha or color contract mismatch')
    expected = mode + '_hdr.exr'
    if record.get('file') != expected or sha(Path(directory) / expected) != record.get('sha256'):
        raise RuntimeError('Injected HDR resource is missing, replaced or belongs to another branch')
    return record
