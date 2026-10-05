"""Replay the original compositor from a bound cache; a guard forbids 3D rendering."""
import json
import sys
import time
import uuid
from pathlib import Path
import bpy
sys.path.insert(0, str(Path(__file__).resolve().parent))
from frame_contract import validate_branch_input, validate_cache, write_json
from blender_contract import (load_request, graph_state, scene_settings, link_key,
                              output_node, final_socket, capture_file, save_png,
                              isolate_outputs, restore_outputs)

out = Path(sys.argv[sys.argv.index('--') + 1])
mode = sys.argv[sys.argv.index('--') + 2]
if mode not in {'baseline', 'identity', 'neural'}:
    raise ValueError('Unknown replay branch')
scene, graph, raw, identity = load_request(out)
manifest = json.loads((out / 'capture_manifest.json').read_text(encoding='utf-8'))
validate_cache(manifest, identity, out)
original_graph = graph_state(graph, exclude=(raw.name,))
original_settings = scene_settings(scene)
routes = [(link.from_socket.name, link.to_socket, link_key(link)) for link in graph.links if link.from_node == raw]
source_name = raw.name
image = bpy.data.images.load(str(out / 'engine_raw.exr'), check_existing=False)
image.colorspace_settings.name = 'Linear Rec.709'
image.alpha_mode = 'PREMUL'
proxy = graph.nodes.new('CompositorNodeImage')
proxy.name = 'Bound same-frame renderer cache'
proxy.image = image
proxy.layer = manifest['cache_layer']
missing = []
for name, target, key in routes:
    source = proxy.outputs.get('Combined' if name == 'Image' else name)
    if source is None:
        missing.append(name)
    else:
        graph.links.new(source, target)
if missing:
    raise RuntimeError(f'Unsupported or missing original cached passes: {missing}')
graph.nodes.remove(raw)
injected = None
if mode != 'baseline':
    validate_branch_input(json.loads((out / (mode + '_input.json')).read_text(encoding='utf-8')), manifest, out, mode)
    hdr_path = out / (mode + '_hdr.exr')
    hdr = bpy.data.images.load(str(hdr_path), check_existing=False)
    hdr.colorspace_settings.name = 'Linear Rec.709'
    hdr.alpha_mode = 'PREMUL'
    injected = graph.nodes.new('CompositorNodeImage')
    injected.name = 'Identity HDR' if mode == 'identity' else 'PyTorch HDR before original compositor'
    injected.image = hdr
    for name, target, key in routes:
        if name == 'Image':
            graph.links.new(injected.outputs['Image'], target)
# Verify the actual remaining original node definitions and all non-source links.
excluded = [proxy.name] + ([injected.name] if injected else [])
if graph_state(graph, exclude=excluded) != original_graph:
    raise RuntimeError('Cached replay changed original non-source nodes or links')
settings_before = scene_settings(scene)
if settings_before != original_settings:
    raise RuntimeError('Cached replay changed original scene/render/color settings')

class PostprocessOnlyEngine(bpy.types.RenderEngine):
    bl_idname = 'PIPELINE_POSTPROCESS_ONLY'
    bl_label = 'Pipeline compositor replay guard'
    bl_use_postprocess = True
    actual_render_calls = 0

    def render(self, depsgraph):
        type(self).actual_render_calls += 1
        self.error_set('Cache replay attempted an unexpected 3D render')
        raise RuntimeError('Geometry rendering is forbidden during cached replay')

bpy.utils.register_class(PostprocessOnlyEngine)
# This engine is used only in the process-local cache replay. The file, scene
# inputs, native render and compositor settings are unchanged. The engine cannot
# render geometry, so an unnoticed source cannot trigger a second scene render.
original_engine = scene.render.engine
scene.render.engine = PostprocessOnlyEngine.bl_idname
directory = out / (mode + '_post_work_' + uuid.uuid4().hex[:8])
directory.mkdir(exist_ok=False)
outputs = isolate_outputs(graph, directory)
master = output_node(graph, directory, mode + '_linear', [(final_socket(graph), 'Combined')])
started = time.perf_counter()
print('POSTPROCESS_START', mode, flush=True)
result = bpy.ops.render.render()
if 'FINISHED' not in result or PostprocessOnlyEngine.actual_render_calls:
    raise RuntimeError(f'Cached replay failed or rendered geometry: {result}')
save_png(scene, out / ('original_rebuilt.png' if mode == 'baseline' else mode + '.png'))
capture_file(directory, mode + '_linear').replace(out / (mode + '_linear.exr'))
graph.nodes.remove(master)
restore_outputs(outputs)
scene.render.engine = original_engine
if scene_settings(scene) != original_settings or graph_state(graph, exclude=excluded) != original_graph:
    raise RuntimeError('Postprocess execution changed original nodes or settings')
report = {'mode': mode, 'cache_session_id': manifest['capture_session_id'],
          'identity_sha256': manifest['identity_sha256'], 'cache_raw_sha256': manifest['assets']['raw_exr']['sha256'],
          'original_non_source_nodes_and_links_unchanged': True,
          'original_settings_unchanged': True, 'settings': original_settings,
          'replaced_rgb_links': [key for name, target, key in routes if name == 'Image'] if injected else [],
          'auxiliary_links': [key for name, target, key in routes if name != 'Image'],
          'actual_geometry_render_calls': PostprocessOnlyEngine.actual_render_calls,
          'wall_seconds': time.perf_counter() - started}
write_json(out / (mode + '_post.json'), report)
print('POSTPROCESS_COMPLETE', json.dumps(report, ensure_ascii=True), flush=True)
