"""Shared compositor node definition for the UI and background Blender stages."""
import hashlib
from pathlib import Path

import bpy
from bpy.props import BoolProperty, EnumProperty, FloatProperty, StringProperty

NODE_ID = 'NRBCompositorNode'
CHECKPOINT_EXTENSIONS = {'.pt', '.pth', '.ckpt'}
# Keep enum strings alive for Blender's dynamic EnumProperty callbacks.
_enum_cache = {}


def checkpoint_items(node, context):
    try:
        directory = Path(bpy.path.abspath(node.model_directory)).expanduser()
        names = tuple(sorted(p.name for p in directory.iterdir()
                             if p.is_file() and p.suffix.lower() in CHECKPOINT_EXTENSIONS)) if node.model_directory and directory.is_dir() else ()
    except OSError:
        names = ()
    key = (node.model_directory, names)
    if key not in _enum_cache:
        _enum_cache[key] = [(name, name, 'Compatible PyTorch checkpoint',
                             int(hashlib.sha256(name.encode()).hexdigest()[:8], 16) & 0x7fffffff or 1)
                            for name in names] or [('__NONE__', 'No checkpoints found', 'Choose a folder containing compatible .pt, .pth or .ckpt files', 0)]
    return _enum_cache[key]


def selection_changed(node, context):
    selected = node.model_choice
    node.model_file = '' if selected == '__NONE__' else selected


def directory_changed(node, context):
    choices = checkpoint_items(node, context)
    available = [item[0] for item in choices]
    node.model_choice = node.model_file if node.model_file in available else available[0]


class NRBCompositorNode(bpy.types.CompositorNodeCustomGroup):
    bl_idname = NODE_ID
    bl_label = 'Neural Render Bridge'
    bl_description = 'Choose a local model and process renderer HDR before compositing'
    model_directory: StringProperty(name='Model folder', subtype='DIR_PATH', update=directory_changed,
                                    options={'PATH_SUPPORTS_BLEND_RELATIVE'},
                                    description='Folder containing compatible PyTorch checkpoints')
    model_choice: EnumProperty(name='Model', items=checkpoint_items, update=selection_changed)
    model_file: StringProperty(name='Selected checkpoint', description='Persistent filename selected from the model folder')
    model_source: StringProperty(name='Model source', subtype='DIR_PATH', options={'PATH_SUPPORTS_BLEND_RELATIVE'},
                                 description='External model code directory containing dlss5/graph.py')
    mode: EnumProperty(name='Mode', items=[('NEURAL', 'Neural', 'Run the selected PyTorch checkpoint'),
                                          ('IDENTITY', 'Identity validation', 'Validate unchanged HDR replay')])
    strength: FloatProperty(name='Strength', default=1.0, min=0.0, max=1.0)
    eager: BoolProperty(name='Eager inference', default=False)

    @classmethod
    def poll(cls, tree):
        return tree.bl_idname == 'CompositorNodeTree'

    def init(self, context):
        group = bpy.data.node_groups.new('Neural Render Bridge HDR boundary', 'CompositorNodeTree')
        group.interface.new_socket(name='Image', in_out='INPUT', socket_type='NodeSocketColor')
        group.interface.new_socket(name='Image', in_out='OUTPUT', socket_type='NodeSocketColor')
        source = group.nodes.new('NodeGroupInput')
        target = group.nodes.new('NodeGroupOutput')
        group.links.new(source.outputs['Image'], target.inputs['Image'])
        self.node_tree = group
        self.width = 300

    def copy(self, original):
        if self.node_tree:
            self.node_tree = self.node_tree.copy()

    def draw_buttons(self, context, layout):
        layout.prop(self, 'model_directory')
        layout.prop(self, 'model_choice')
        layout.prop(self, 'model_source')
        layout.prop(self, 'mode')
        if self.mode == 'NEURAL':
            layout.prop(self, 'strength')
            layout.prop(self, 'eager')
        layout.operator('nrb.render', text='Render Neural Pipeline', icon='RENDER_STILL')
        layout.operator('nrb.check_environment', icon='CHECKMARK')
        layout.label(text='Renderer HDR → PyTorch → Compositor')


def bridge_node(graph, raw=None, expected=None):
    nodes = [node for node in graph.nodes if node.bl_idname == NODE_ID]
    if len(nodes) > 1:
        raise ValueError('Only one Neural Render Bridge node is supported per compositor')
    node = nodes[0] if nodes else None
    if expected and (node is None or node.name != expected):
        raise ValueError('Requested Neural Render Bridge node is missing or renamed')
    if node is None:
        return None
    if node.mute:
        raise ValueError('Unmute the Neural Render Bridge node before running the neural pipeline')
    sources = [n for n in graph.nodes if n.bl_idname == 'CompositorNodeRLayers']
    if raw is None:
        if len(sources) != 1:
            raise ValueError('Exactly one Render Layers node is required')
        raw = sources[0]
    incoming = [link for link in graph.links if link.to_node == node and link.to_socket.name == 'Image']
    if len(incoming) != 1 or incoming[0].from_node != raw or incoming[0].from_socket.name != 'Image':
        raise ValueError('Connect Render Layers Image directly to Neural Render Bridge Image, before postprocessing')
    if not any(link.from_node == node and link.from_socket.name == 'Image' for link in graph.links):
        raise ValueError('Connect Neural Render Bridge Image output to the original compositor')
    group = node.node_tree
    if not group or len(group.nodes) != 2 or len(group.links) != 1:
        raise ValueError('The Neural Render Bridge internal HDR pass-through group was modified')
    link = group.links[0]
    if (link.from_node.bl_idname != 'NodeGroupInput' or link.to_node.bl_idname != 'NodeGroupOutput'
            or link.from_socket.name != 'Image' or link.to_socket.name != 'Image'):
        raise ValueError('The Neural Render Bridge internal HDR pass-through group was modified')
    return node


def node_checkpoint(node):
    if not node.model_directory or not node.model_file:
        raise ValueError('Choose a model folder and checkpoint in the Neural Render Bridge node')
    name = node.model_file
    if Path(name).name != name or Path(name).suffix.lower() not in CHECKPOINT_EXTENSIONS:
        raise ValueError('Select a compatible checkpoint filename from the model folder')
    directory = Path(bpy.path.abspath(node.model_directory)).expanduser().resolve()
    checkpoint = directory / name
    if not checkpoint.is_file():
        raise ValueError('Selected node checkpoint is missing; choose an existing model')
    return checkpoint
