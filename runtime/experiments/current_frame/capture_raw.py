"""Capture native reference and original pre-compositor sockets in one render."""
import json
import sys
import time
import uuid
from pathlib import Path
import bpy
sys.path.insert(0, str(Path(__file__).resolve().parent))
from blender_contract import (load_request, output_node, final_socket, isolate_outputs,
                              restore_outputs, capture_file, save_png, graph_state)
from frame_contract import write_json

out = Path(sys.argv[sys.argv.index('--') + 1])
phase = sys.argv[sys.argv.index('--') + 2]
scene, graph, raw, identity = load_request(out)
if phase == 'inspect':
    write_json(out / 'preflight.json', identity)
    print('PREFLIGHT_COMPLETE', identity['scene'], identity['frame'], identity['view_layer'], flush=True)
else:
    expected = json.loads((out / 'preflight.json').read_text(encoding='utf-8'))
    if identity != expected:
        raise RuntimeError('Source identity changed between preflight and capture')
    directory = out / ('capture_work_' + uuid.uuid4().hex[:8])
    directory.mkdir(exist_ok=False)
    original_graph = graph_state(graph)
    outputs = isolate_outputs(graph, directory)
    sockets = [raw.outputs['Image'], raw.outputs['Alpha']]
    for link in graph.links:
        if link.from_node == raw and link.from_socket not in sockets:
            sockets.append(link.from_socket)
    bindings = [{'source_socket': socket.name, 'cache_socket': 'Combined' if socket.name == 'Image' else socket.name,
                 'type': socket.type} for socket in sockets]
    capture = output_node(graph, directory, 'engine_raw',
                          [(socket, raw.layer + '.' + binding['cache_socket']) for socket, binding in zip(sockets, bindings)])
    reference = output_node(graph, directory, 'native_linear', [(final_socket(graph), 'Combined')])
    graph.update_tag()
    bpy.context.view_layer.update()
    print('CAPTURE_OUTPUTS', [(node.file_name, node.format.media_type, node.format.file_format,
                             [(socket.name, socket.is_linked) for socket in node.inputs])
                            for node in (capture, reference)], flush=True)
    started = time.perf_counter()
    print('RAW_RENDER_START', scene.name, scene.frame_current, raw.layer, flush=True)
    result = bpy.ops.render.render()
    if 'FINISHED' not in result:
        raise RuntimeError(f'Native render did not finish: {result}')
    save_png(scene, out / 'native_reference.png')
    raw_path = capture_file(directory, 'engine_raw')
    linear_path = capture_file(directory, 'native_linear')
    raw_path.replace(out / 'engine_raw.exr')
    linear_path.replace(out / 'native_linear.exr')
    graph.nodes.remove(capture)
    graph.nodes.remove(reference)
    restore_outputs(outputs)
    if graph_state(graph) != original_graph:
        raise RuntimeError('Native capture changed original compositor definitions')
    _, _, _, after = load_request(out)
    if after != identity:
        raise RuntimeError('Source dependencies or settings changed during native render')
    report = {'identity': identity, 'frame': scene.frame_current, 'source_bindings': bindings,
              'source_render_invocations': 1, 'original_graph_unchanged': True,
              'wall_seconds': time.perf_counter() - started}
    write_json(out / 'raw_render.json', report)
    print('RAW_RENDER_COMPLETE', json.dumps(report, ensure_ascii=True), flush=True)
