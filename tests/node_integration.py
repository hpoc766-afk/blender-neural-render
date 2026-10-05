"""Integration checks for model folder selection in the installed compositor node."""
import bpy
import hashlib
import importlib
import json
import os
import shutil
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ['NRB_TEST_OUTPUT'])
module_name = 'bl_ext.nrb_node_test.neural_render_bridge'
addon = importlib.import_module(module_name)
fixture = Path(os.environ['NRB_TEST_FIXTURE'])
source_hash = hashlib.sha256(fixture.read_bytes()).hexdigest()
bpy.ops.wm.open_mainfile(filepath=str(fixture))
prefs = addon.preferences(bpy.context)
prefs.python_path = os.environ['NRB_TEST_PYTHON']
prefs.model_source = str(OUT / 'invalid-global-source')
prefs.checkpoint = str(OUT / 'invalid-global-checkpoint.pt')
prefs.output_root = str(OUT / 'jobs')
settings = bpy.context.scene.neural_render_bridge
settings.mode = 'NEURAL'
assert bpy.ops.nrb.add_node() == {'FINISHED'}
graph = bpy.context.scene.compositing_node_group
node = next(n for n in graph.nodes if n.bl_idname == 'NRBCompositorNode')
library = OUT / 'model library';library.mkdir(parents=True, exist_ok=True)
selected = library / 'Selected Model.pt'
if not selected.exists():
    try:
        os.link(os.environ['NRB_TEST_CHECKPOINT'], selected)
    except OSError:
        shutil.copyfile(os.environ['NRB_TEST_CHECKPOINT'], selected)
(library / 'Bad Model.pt').write_bytes(b'incompatible checkpoint')
(library / 'ignored.txt').write_text('not a model', encoding='utf-8')
node.model_directory = bpy.path.relpath(str(library))
assert node.model_file == 'Bad Model.pt'
node.model_choice = 'Selected Model.pt'
assert node.model_file == 'Selected Model.pt'
node.model_source = os.environ['NRB_TEST_MODEL_SOURCE']
assert addon.node_checkpoint(node) == selected.resolve()
node.mode = 'IDENTITY'
# Add a raw RGB export bypass: this route must remain uninjected.
raw = next(n for n in graph.nodes if n.bl_idname == 'CompositorNodeRLayers')
sink = graph.nodes.new('CompositorNodeOutputFile')
sink.directory = str(OUT / 'original_bypass')
sink.file_name = 'bypass'
sink.format.media_type = 'IMAGE'
sink.format.file_format = 'PNG'
sink.file_output_items.new('RGBA', 'Bypass')
graph.links.new(raw.outputs['Image'], sink.inputs[0])
raw_link = next(l for l in graph.links if l.to_node == node)
graph.links.remove(raw_link)
try:
    addon.start_job(bpy.context)
    raise AssertionError('Disconnected node was accepted')
except ValueError as exc:
    assert 'directly' in str(exc)
graph.links.new(raw.outputs['Image'], node.inputs['Image'])
# Pixel/geometry edits remain unsaved in the interactive scene.
obj = next(o for o in bpy.context.scene.objects if o.type == 'MESH')
obj.location.x += 0.019
paths = {image.name: image.filepath for image in bpy.data.images}
original_file = bpy.data.filepath
original_node_directory = node.model_directory


def wait():
    deadline = time.monotonic() + 180
    while addon._job.poll() == 'RUNNING':
        if time.monotonic() > deadline:
            addon._job.cancel()
            raise AssertionError('Job timeout')
        time.sleep(0.2)
    assert addon._job.status == 'SUCCEEDED', f'{addon._job.error}: {addon._job.directory}'
    return addon._job.directory


assert bpy.ops.nrb.check_environment() == {'FINISHED'}
wait()
assert bpy.ops.nrb.render() == {'FINISHED'}
identity = wait()
r = json.loads((identity / 'pipeline_report.json').read_text(encoding='utf-8'))
assert r['scope'] == 'identity_only', 'Scene mode overrode node mode'
assert r['baseline_reference_check']['linear_max_abs_difference'] == 0
assert r['identity_reference_check']['linear_max_abs_difference'] == 0
assert r['branches']['identity']['neural_node'] == node.name
assert len(r['branches']['identity']['replaced_rgb_links']) == 1
assert len(r['branches']['identity']['passthrough_rgb_links']) == 1
assert r['source_geometry_render_invocations_this_run'] == 1
assert r['postprocess_actual_geometry_render_calls'] == 0
# Inspect the persisted custom node and selected filename, without changing the current scene.
with bpy.data.libraries.load(str(identity / 'scene_snapshot.blend')) as (available, requested):
    requested.node_groups = [graph.name]
copied = requested.node_groups[0]
copied_node = next(n for n in copied.nodes if n.bl_idname == 'NRBCompositorNode')
assert copied_node.model_file == 'Selected Model.pt'
assert Path(bpy.path.abspath(copied_node.model_directory)).resolve() == library.resolve()
bpy.data.node_groups.remove(copied)
node.mode = 'NEURAL'
node.strength = 1.0
settings.mode = 'IDENTITY'
assert bpy.ops.nrb.render() == {'FINISHED'}
neural = wait()
rn = json.loads((neural / 'pipeline_report.json').read_text(encoding='utf-8'))
assert rn['inference']['checkpoint_path'] == str(selected.resolve())
assert rn['inference']['checkpoint_sha256'] == hashlib.sha256(selected.read_bytes()).hexdigest()
assert rn['source_geometry_render_invocations_this_run'] == 1
assert rn['postprocess_actual_geometry_render_calls'] == 0
assert len(rn['branches']['neural']['replaced_rgb_links']) == 1
assert rn['branches']['neural']['replaced_rgb_links'][0][2] == node.name
assert len(rn['branches']['neural']['passthrough_rgb_links']) == 1
assert bpy.ops.nrb.load_result(branch='neural') == {'FINISHED'}
assert bpy.data.filepath == original_file
assert node.model_directory == original_node_directory
bypass_original = list(neural.glob('baseline_post_work_*/original_exports/**/bypass*.png'))
bypass_neural = list(neural.glob('neural_post_work_*/original_exports/**/bypass*.png'))
assert len(bypass_original) == len(bypass_neural) == 1
assert bypass_original[0].read_bytes() == bypass_neural[0].read_bytes(), 'Raw RGB bypass changed after inference'
assert all(bpy.data.images[name].filepath == value for name, value in paths.items())
assert hashlib.sha256(fixture.read_bytes()).hexdigest() == source_hash
# Missing selected files must fail before launching a worker, without fallback to Preferences.
last_job = addon._job
node.model_file = 'missing.pt'
try:
    addon.start_job(bpy.context)
    raise AssertionError('Missing selected model accepted')
except ValueError as exc:
    assert 'missing' in str(exc)
    assert addon._job is last_job
result = {'status': 'passed', 'identity': str(identity), 'neural': str(neural),
          'node_selected_checkpoint': str(selected), 'selected_checkpoint_sha256': rn['inference']['checkpoint_sha256'],
          'global_preferences_overridden': True, 'relative_model_directory_supported': True,
          'selected_filename_persisted': True, 'disconnected_node_rejected': True,
          'missing_selected_model_rejected': True, 'bypass_rgb_not_injected': True,
          'baseline_linear_max_difference': r['baseline_reference_check']['linear_max_abs_difference'],
          'identity_linear_max_difference': r['identity_reference_check']['linear_max_abs_difference'],
          'source_render_count': 1, 'replay_geometry_count': 0, 'model_gpu_ms': rn['inference']['model_gpu_ms']}
(OUT / 'integration_report.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
print('NODE_MODEL_SELECTION_PASSED', json.dumps(result), flush=True)
