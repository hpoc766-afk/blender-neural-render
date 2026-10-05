"""Blender UI for same-frame HDR processing before the original compositor."""
import json
from pathlib import Path
import sys
import time
import uuid

import bpy
from bpy.props import BoolProperty, EnumProperty, FloatProperty, PointerProperty, StringProperty
from .jobs import Job
from .runtime.bridge_node import NRBCompositorNode, NODE_ID, bridge_node, node_checkpoint

bl_info = {'name': 'Neural Render Bridge', 'author': 'Neural Render Bridge contributors',
           'version': (0, 2, 0), 'blender': (5, 1, 0),
           'location': 'Properties > Render > Neural Render Bridge',
           'description': 'Process scene HDR before the original compositor', 'category': 'Render'}

_job = None
_last_render = None


def preferences(context):
    addon = context.preferences.addons.get(__package__)
    if addon is None:
        raise RuntimeError('Enable Neural Render Bridge in Preferences first')
    return addon.preferences


def redraw():
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            area.tag_redraw()


def poll_job():
    if _job is None:
        return None
    status = _job.poll()
    redraw()
    return 0.5 if status == 'RUNNING' else None


def resolved(value):
    return Path(bpy.path.abspath(value)).expanduser().resolve()


def start_job(context, check=False):
    global _job, _last_render
    if _job is not None and _job.poll() == 'RUNNING':
        raise RuntimeError('A Neural Render Bridge job is already running')
    if bpy.app.version[:2] != (5, 1):
        raise RuntimeError('Blender 5.1 is required')
    prefs = preferences(context)
    graph = context.scene.compositing_node_group
    node = bridge_node(graph) if graph else None
    if (context.space_data and context.space_data.type == 'NODE_EDITOR'
            and context.space_data.edit_tree != graph):
        raise ValueError('Use the active scene compositor for Neural Render Bridge')
    settings = node or context.scene.neural_render_bridge
    source_value = node.model_source if node else prefs.model_source
    if not prefs.python_path or not source_value:
        raise ValueError('Configure Python in Preferences and model source in the node or Preferences')
    python = resolved(prefs.python_path)
    model_source = resolved(source_value)
    if not python.is_file():
        raise ValueError('Python executable not found')
    if not (model_source / 'dlss5' / 'graph.py').is_file():
        raise ValueError('Model source must contain dlss5/graph.py')
    identity = settings.mode == 'IDENTITY'
    needs_model = not identity and settings.strength > 0
    checkpoint = node_checkpoint(node) if node and needs_model else (resolved(prefs.checkpoint) if prefs.checkpoint and not node else None)
    if needs_model and (checkpoint is None or not checkpoint.is_file()):
        raise ValueError('Choose an existing compatible checkpoint in the node or Preferences')
    if prefs.output_root:
        root = resolved(prefs.output_root)
    else:
        root = Path(bpy.utils.extension_path_user(__package__, path='renders', create=True))
    directory = root / (time.strftime('%Y%m%d_%H%M%S_') + uuid.uuid4().hex[:8])
    directory.mkdir(parents=True, exist_ok=False)
    command = [str(python), '-u', str(Path(__file__).parent / 'worker.py'),
               '--blender', bpy.app.binary_path,
               '--blender-ocio', str(Path(bpy.utils.resource_path('LOCAL')) / 'datafiles/colormanagement/config.ocio'),
               '--model-source', str(model_source),
               '--out', str(directory), '--strength', str(settings.strength)]
    if node:
        command += ['--neural-node', node.name]
    if checkpoint:
        command += ['--checkpoint', str(checkpoint)]
    if identity:
        command += ['--identity-only']
    if settings.eager:
        command += ['--eager']
    if check:
        command += ['--check']
    else:
        scene = context.scene
        if not scene.camera or not scene.compositing_node_group or not scene.render.use_compositing:
            raise ValueError('An active camera and compositor are required')
        # File-backed images with unsaved pixel edits would otherwise point at
        # old disk content in the worker; request explicit save/pack instead.
        for image in bpy.data.images:
            if image.is_dirty and image.source == 'FILE' and not image.packed_file:
                raise ValueError(f'Save or pack edited image before rendering: {image.name}')
        if scene.frame_subframe != 0:
            raise ValueError('Subframe renders are not supported; choose an integer frame')
        snapshot = directory / 'scene_snapshot.blend'
        # The library writer recursively includes scene dependencies. ABSOLUTE
        # remapping preserves external resources without saving the open file.
        datablocks = set(bpy.data.scenes) | set(bpy.data.images) | set(bpy.data.texts)
        bpy.data.libraries.write(str(snapshot), datablocks, path_remap='ABSOLUTE')
        command += ['--source', str(snapshot), '--scene', scene.name,
                    '--frame', str(scene.frame_current)]
        (directory / 'launch.json').write_text(json.dumps(
            {'scene': scene.name, 'frame': scene.frame_current, 'subframe': scene.frame_subframe,
             'open_file': bpy.data.filepath, 'snapshot': str(snapshot),
             'neural_node': node.name if node else None,
             'selected_checkpoint': str(checkpoint) if checkpoint else None,
             'model_source': str(model_source),
             'capture_boundary': 'Render Layers Combined HDR before original compositor'},
            indent=2, ensure_ascii=False), encoding='utf-8')
    _job = Job(command, directory, 'check' if check else 'render')
    if not check:
        _last_render = directory
    if not bpy.app.timers.is_registered(poll_job):
        bpy.app.timers.register(poll_job, first_interval=0.5)
    redraw()
    return directory


class NRBPreferences(bpy.types.AddonPreferences):
    bl_idname = __package__
    python_path: StringProperty(name='Python executable', subtype='FILE_PATH',
                               description='External Python with torch, numpy, Pillow and OpenEXR')
    model_source: StringProperty(name='Model source directory', subtype='DIR_PATH',
                                description='Directory containing dlss5/graph.py: vendor/dlss5-onnx/src')
    checkpoint: StringProperty(name='Checkpoint', subtype='FILE_PATH',
                               description='Recovered-static dlss5_static.pt checkpoint')
    output_root: StringProperty(name='Output directory', subtype='DIR_PATH',
                                description='Empty uses the writable extension user directory')

    def draw(self, context):
        layout = self.layout
        for name in ('python_path', 'model_source', 'checkpoint', 'output_root'):
            layout.prop(self, name)
        layout.label(text='PyTorch runs outside Blender; weights are not bundled.')
        layout.operator('nrb.check_environment', icon='CHECKMARK')


class NRBSettings(bpy.types.PropertyGroup):
    mode: EnumProperty(name='Mode', items=[('NEURAL', 'Neural', 'Process HDR RGB with PyTorch'),
                                         ('IDENTITY', 'Identity validation', 'Validate unchanged HDR and compositor replay')])
    strength: FloatProperty(name='Strength', default=1.0, min=0.0, max=1.0,
                            description='Scale the HDR residual; zero skips model inference')
    eager: BoolProperty(name='Eager inference', default=False,
                        description='Use ordinary PyTorch execution instead of CUDA Graph')


class NRB_OT_add_node(bpy.types.Operator):
    bl_idname = 'nrb.add_node'
    bl_label = 'Neural Render Bridge'
    bl_description = 'Insert the bridge between renderer Image and existing postprocessing'
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        graph = context.scene.compositing_node_group
        if not graph:
            self.report({'ERROR'}, 'Enable the active scene compositor first')
            return {'CANCELLED'}
        if any(n.bl_idname == NODE_ID for n in graph.nodes):
            self.report({'ERROR'}, 'This compositor already contains a Neural Render Bridge node')
            return {'CANCELLED'}
        sources = [n for n in graph.nodes if n.bl_idname == 'CompositorNodeRLayers']
        if len(sources) != 1:
            self.report({'ERROR'}, 'Exactly one Render Layers node is required')
            return {'CANCELLED'}
        raw = sources[0]
        targets = [link.to_socket for link in graph.links if link.from_node == raw and link.from_socket.name == 'Image']
        if not targets:
            self.report({'ERROR'}, 'Connect Render Layers Image to the compositor first')
            return {'CANCELLED'}
        node = graph.nodes.new(NODE_ID)
        node.location = (raw.location.x + raw.width + 60, raw.location.y)
        prefs = preferences(context)
        node.model_source = prefs.model_source
        if prefs.checkpoint:
            checkpoint = resolved(prefs.checkpoint)
            node.model_directory = str(checkpoint.parent)
            from .runtime.bridge_node import checkpoint_items
            if checkpoint.name in [item[0] for item in checkpoint_items(node, context)]:
                node.model_choice = checkpoint.name
        settings = context.scene.neural_render_bridge
        node.mode, node.strength, node.eager = settings.mode, settings.strength, settings.eager
        graph.links.new(raw.outputs['Image'], node.inputs['Image'])
        for target in targets:
            graph.links.new(node.outputs['Image'], target)
        for n in graph.nodes:
            n.select = n == node
        graph.nodes.active = node
        self.report({'INFO'}, 'Configure the model folder and checkpoint in the new node')
        return {'FINISHED'}


def node_add_menu(self, context):
    space = context.space_data
    if space and space.type == 'NODE_EDITOR' and space.tree_type == 'CompositorNodeTree':
        self.layout.operator('nrb.add_node', icon='NODETREE')


class NRB_OT_render(bpy.types.Operator):
    bl_idname = 'nrb.render'
    bl_label = 'Render with Neural Pipeline'
    bl_description = 'Capture native HDR, process RGB, then replay the original compositor'

    def execute(self, context):
        try:
            directory = start_job(context)
        except Exception as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        self.report({'INFO'}, f'Render started: {directory}')
        return {'FINISHED'}


class NRB_OT_check(bpy.types.Operator):
    bl_idname = 'nrb.check_environment'
    bl_label = 'Check Environment'

    def execute(self, context):
        try:
            start_job(context, check=True)
        except Exception as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        return {'FINISHED'}


class NRB_OT_cancel(bpy.types.Operator):
    bl_idname = 'nrb.cancel'
    bl_label = 'Cancel Job'

    def execute(self, context):
        try:
            if _job:
                _job.cancel()
            redraw()
        except Exception as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        return {'FINISHED'}


class NRB_OT_open_directory(bpy.types.Operator):
    bl_idname = 'nrb.open_directory'
    bl_label = 'Open Output Folder'

    def execute(self, context):
        if _job is None:
            return {'CANCELLED'}
        bpy.ops.wm.path_open(filepath=str(_job.directory))
        return {'FINISHED'}


class NRB_OT_load_result(bpy.types.Operator):
    bl_idname = 'nrb.load_result'
    bl_label = 'Load Result'
    branch: EnumProperty(items=[('neural', 'Neural', ''), ('original_rebuilt', 'Original', ''),
                                ('identity', 'Identity', '')])

    def execute(self, context):
        if _last_render is None:
            return {'CANCELLED'}
        try:
            report = json.loads((_last_render / 'pipeline_report.json').read_text(encoding='utf-8'))
            if report.get('status') != 'passed':
                raise ValueError('Render has not passed validation')
            image = bpy.data.images.load(str(_last_render / (self.branch + '.png')), check_existing=True)
            image.use_view_as_render = False  # Already encoded using the original display transform.
        except Exception as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        screen = context.screen
        area = next((a for a in screen.areas if a.type == 'IMAGE_EDITOR'), None)
        if area is None:
            area = next((a for a in screen.areas if a.type == 'VIEW_3D'), None)
            if area:
                area.type = 'IMAGE_EDITOR'
        if area:
            area.spaces.active.image = image
        else:
            self.report({'INFO'}, f'Loaded image datablock: {image.name}')
        return {'FINISHED'}


class NRB_PT_render(bpy.types.Panel):
    bl_label = 'Neural Render Bridge'
    bl_idname = 'NRB_PT_render'
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = 'render'

    def draw(self, context):
        layout = self.layout
        settings = context.scene.neural_render_bridge
        running = _job is not None and _job.status == 'RUNNING'
        controls = layout.column()
        controls.enabled = not running
        graph = context.scene.compositing_node_group
        nodes = [n for n in graph.nodes if n.bl_idname == NODE_ID] if graph else []
        if nodes:
            controls.label(text='Model and settings: ' + nodes[0].name)
        else:
            controls.prop(settings, 'mode')
            if settings.mode == 'NEURAL':
                controls.prop(settings, 'strength')
                controls.prop(settings, 'eager')
            controls.operator('nrb.add_node', text='Add Model Selection Node', icon='NODETREE')
        controls.operator('nrb.render', icon='RENDER_STILL')
        controls.operator('nrb.check_environment')
        if _job:
            layout.label(text=_job.stage())
            if running:
                layout.operator('nrb.cancel', icon='CANCEL')
            if _job.error:
                layout.label(text=_job.error[:100], icon='ERROR')
            layout.operator('nrb.open_directory', icon='FILE_FOLDER')
        if _last_render and (_last_render / 'pipeline_report.json').is_file():
            for branch, label in [('original_rebuilt', 'View Original'), ('identity', 'View Identity'), ('neural', 'View Neural')]:
                if (_last_render / (branch + '.png')).exists():
                    layout.operator('nrb.load_result', text=label).branch = branch
        layout.label(text='Native HDR → PyTorch → Original compositor')


CLASSES = (NRBPreferences, NRBSettings, NRBCompositorNode, NRB_OT_add_node, NRB_OT_render, NRB_OT_check, NRB_OT_cancel,
           NRB_OT_open_directory, NRB_OT_load_result, NRB_PT_render)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.neural_render_bridge = PointerProperty(type=NRBSettings)
    bpy.types.NODE_MT_add.append(node_add_menu)


def unregister():
    global _job, _last_render
    if _job is not None and _job.status == 'RUNNING':
        _job.cancel()
    if bpy.app.timers.is_registered(poll_job):
        bpy.app.timers.unregister(poll_job)
    bpy.types.NODE_MT_add.remove(node_add_menu)
    del bpy.types.Scene.neural_render_bridge
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
    _job = _last_render = None
