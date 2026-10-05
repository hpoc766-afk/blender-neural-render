"""Register the bridge node before loading snapshots containing custom nodes."""
from pathlib import Path
import runpy
import sys

import bpy
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bridge_node import NRBCompositorNode

source, stage, directory, phase = sys.argv[sys.argv.index('--') + 1:]
if not hasattr(bpy.types, NRBCompositorNode.bl_idname):
    bpy.utils.register_class(NRBCompositorNode)
bpy.ops.wm.open_mainfile(filepath=source)
sys.argv = [stage, '--', directory, phase]
runpy.run_path(stage, run_name='__main__')
