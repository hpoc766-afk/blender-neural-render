"""Blender 5.1 single-scene contract and compositor preservation checks."""
import json
import os
from pathlib import Path
import bpy
from frame_contract import CAPTURE_VERSION, digest, sha, write_json


def scalar_properties(value):
    result = {}
    for prop in value.bl_rna.properties:
        if prop.is_readonly or prop.type not in {'BOOLEAN', 'INT', 'FLOAT', 'STRING', 'ENUM'}:
            continue
        item = getattr(value, prop.identifier)
        if getattr(prop, 'is_array', False):
            item = list(item)
        elif isinstance(item, set):
            item = sorted(item)
        result[prop.identifier] = item
    return result


def graph_state(graph, exclude=()):
    nodes = {}
    for node in graph.nodes:
        if node.name in exclude:
            continue
        props = scalar_properties(node)
        for key in ('location', 'location_absolute', 'width', 'height', 'select', 'hide'):
            props.pop(key, None)
        sockets = []
        for socket in node.inputs:
            value = getattr(socket, 'default_value', None)
            if value is not None and not isinstance(value, (bool, int, float, str)):
                value = list(value)
            sockets.append([socket.identifier, value])
        state = {'type': node.bl_idname, 'properties': props, 'inputs': sockets}
        if getattr(node, 'node_tree', None):
            state['group'] = graph_state(node.node_tree)
        if getattr(node, 'image', None):
            image = node.image
            state['image'] = [image.name, image.filepath, image.colorspace_settings.name, image.alpha_mode]
        nodes[node.name] = state
    links = sorted(link_key(link) for link in graph.links
                   if link.from_node.name not in exclude and link.to_node.name not in exclude)
    return {'nodes': nodes, 'links': links}


def link_key(link):
    return [link.from_node.name, list(link.from_node.outputs).index(link.from_socket),
            link.to_node.name, list(link.to_node.inputs).index(link.to_socket)]


def scene_settings(scene):
    return {'render': scalar_properties(scene.render),
            'eevee': scalar_properties(scene.eevee), 'cycles': scalar_properties(scene.cycles),
            'view': scalar_properties(scene.view_settings),
            'display': scalar_properties(scene.display_settings),
            'curve': [[list(point.location) for point in curve.points]
                      for curve in scene.view_settings.curve_mapping.curves] if scene.view_settings.use_curve_mapping else None,
            'image_format': scalar_properties(scene.render.image_settings)}


def load_request(directory):
    request = json.loads((directory / 'request.json').read_text(encoding='utf-8'))
    scene = bpy.data.scenes[request['scene']] if request.get('scene') else bpy.context.scene
    bpy.context.window.scene = scene
    if request.get('frame') is not None:
        scene.frame_set(request['frame'])
    if bpy.app.version[:2] != (5, 1):
        raise RuntimeError('unsupported_capability: only Blender 5.1 has been implemented')
    if not scene.camera or not scene.compositing_node_group or not scene.render.use_compositing:
        raise RuntimeError('unsupported_capability: original camera and active compositor required')
    if scene.render.use_multiview or scene.render.use_border:
        raise RuntimeError('unsupported_capability: multiview and border/cropped renders require separate validation')
    if scene.render.use_sequencer and scene.sequence_editor and len(scene.sequence_editor.strips):
        raise RuntimeError('unsupported_capability: sequencer output requires separate routing')
    graph = scene.compositing_node_group
    def check_tree(tree):
        for node in tree.nodes:
            if node.bl_idname.startswith('CompositorNodeCryptomatte'):
                raise RuntimeError('unsupported_capability: Cryptomatte metadata-preserving capture is not implemented')
            if getattr(node, 'node_tree', None):
                if any(n.bl_idname == 'CompositorNodeRLayers' for n in node.node_tree.nodes):
                    raise RuntimeError('unsupported_capability: nested Render Layers sources')
                check_tree(node.node_tree)
    check_tree(graph)
    sources = [n for n in graph.nodes if n.bl_idname == 'CompositorNodeRLayers']
    if len(sources) != 1:
        raise RuntimeError('unsupported_capability: exactly one Render Layers source required')
    raw = sources[0]
    owner = raw.scene or scene
    if owner != scene:
        raise RuntimeError('unsupported_capability: cross-scene Render Layers sources')
    layer = owner.view_layers.get(raw.layer)
    if not layer or not layer.use:
        raise RuntimeError('Render Layers refers to a missing or disabled View Layer')
    if request.get('view_layer') and layer.name != request['view_layer']:
        raise RuntimeError('Requested View Layer differs from the original compositor source')
    if not any(link.from_node == raw and link.from_socket.name == 'Image' for link in graph.links):
        raise RuntimeError('Render Layers Image has no original RGB consumer')
    for image in bpy.data.images:
        if image.source in {'SEQUENCE', 'MOVIE'} or image.source == 'FILE' and image.type == 'MULTILAYER':
            raise RuntimeError('unsupported_capability: animated or external multilayer image dependencies')
    if any(text.use_module for text in bpy.data.texts):
        raise RuntimeError('unsupported_capability: executable embedded Python dependencies')
    if os.environ.get('OCIO'):
        raise RuntimeError('unsupported_capability: custom OCIO configurations require explicit color validation')
    dependencies = {}
    for filename in bpy.utils.blend_paths(absolute=True, packed=False):
        path = Path(filename).resolve()
        if not path.is_file():
            raise RuntimeError(f'unsupported_capability: unresolved external dependency {path}')
        dependencies[str(path)] = sha(path)
    def check_geometry_tree(tree):
        for node in tree.nodes:
            if node.bl_idname in {'GeometryNodeSimulationInput', 'GeometryNodeSimulationOutput', 'GeometryNodeBake'}:
                raise RuntimeError('unsupported_capability: geometry-node simulation/bake caches are not frozen')
            if getattr(node, 'node_tree', None):
                check_geometry_tree(node.node_tree)
    if scene.rigidbody_world and scene.rigidbody_world.enabled:
        raise RuntimeError('unsupported_capability: active rigid-body cache dependencies are not frozen')
    for obj in scene.objects:
        for mod in obj.modifiers:
            if not mod.show_render:
                continue
            if mod.type in {'FLUID', 'CLOTH', 'SOFT_BODY', 'PARTICLE_SYSTEM'}:
                raise RuntimeError('unsupported_capability: active simulation caches are not frozen')
            if mod.type == 'NODES' and mod.node_group:
                check_geometry_tree(mod.node_group)
    height = scene.render.resolution_y * scene.render.resolution_percentage // 100
    width = scene.render.resolution_x * scene.render.resolution_percentage // 100
    code = {p.name: sha(p) for p in Path(__file__).parent.glob('*.py')
            if p.name in {'capture_raw.py', 'post_stage.py', 'blender_contract.py', 'frame_contract.py'}}
    identity = {'version': CAPTURE_VERSION, 'source': str(Path(bpy.data.filepath).resolve()),
                'source_sha256': sha(bpy.data.filepath), 'dependencies': dependencies,
                'scene': scene.name, 'frame': scene.frame_current, 'subframe': scene.frame_subframe,
                'camera': scene.camera.name, 'view_layer': layer.name, 'source_node': raw.name,
                'dimensions_hw': [height, width], 'alpha': 'premultiplied',
                'blender_build': bpy.app.build_hash.decode(), 'blender_version': bpy.app.version_string,
                'settings': scene_settings(scene), 'compositor': graph_state(graph), 'capture_code': code}
    return scene, graph, raw, identity


def output_node(graph, directory, name, sockets):
    node = graph.nodes.new('CompositorNodeOutputFile')
    node.name = 'Pipeline capture ' + name
    node.directory = str(directory)
    node.file_name = name
    node.save_as_render = False
    node.format.media_type = 'MULTI_LAYER_IMAGE'
    node.format.file_format = 'OPEN_EXR_MULTILAYER'
    node.format.color_depth = '32'
    node.format.exr_codec = 'ZIP'
    for socket, label in sockets:
        kind = 'RGBA' if socket.type == 'RGBA' else 'VECTOR' if socket.type == 'VECTOR' else 'FLOAT'
        node.file_output_items.new(kind, label)
        if kind == 'VECTOR':
            node.file_output_items[-1].vector_socket_dimensions = len(socket.default_value)
        # Blender 5.1 adds an empty extend socket after the real file slots.
        # Linking inputs[-1] silently connects that virtual socket and exports nothing.
        graph.links.new(socket, node.inputs[len(node.file_output_items) - 1])
    return node


def isolate_outputs(graph, directory):
    paths = []
    for index, node in enumerate(n for n in graph.nodes if n.bl_idname == 'CompositorNodeOutputFile'):
        paths.append([node, node.directory, node.file_name])
        node.directory = str(directory / 'original_exports' / str(index))
    return paths


def restore_outputs(paths):
    for node, directory, name in paths:
        node.directory, node.file_name = directory, name


def final_socket(graph):
    output = [n for n in graph.nodes if n.bl_idname == 'NodeGroupOutput' and n.is_active_output]
    if len(output) != 1:
        raise RuntimeError('unsupported_capability: one active compositor group output required')
    links = [link for link in graph.links if link.to_node == output[0] and link.to_socket.type == 'RGBA']
    if len(links) != 1:
        raise RuntimeError('unsupported_capability: one connected RGBA final output required')
    return links[0].from_socket


def capture_file(directory, stem):
    candidates = list(directory.glob(stem + '*.exr'))
    if len(candidates) != 1:
        raise RuntimeError(f'Expected one {stem} EXR, found {candidates}')
    destination = directory / (stem + '.exr')
    if candidates[0] != destination:
        candidates[0].replace(destination)
    return destination


def save_png(scene, destination):
    fmt = scalar_properties(scene.render.image_settings)
    try:
        scene.render.image_settings.media_type = 'IMAGE'
        scene.render.image_settings.file_format = 'PNG'
        scene.render.image_settings.color_depth = '8'
        bpy.data.images['Render Result'].save_render(str(destination), scene=scene)
    finally:
        # Set the media first; it constrains valid file format enum values.
        scene.render.image_settings.media_type = fmt.pop('media_type')
        scene.render.image_settings.file_format = fmt.pop('file_format')
        for key, value in fmt.items():
            setattr(scene.render.image_settings, key, set(value) if isinstance(value, list) and key == 'views_format' else value)
