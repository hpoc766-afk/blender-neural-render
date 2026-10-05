"""External-Python composition root; no bpy or display-image inference."""
import argparse
import json
import os
from pathlib import Path
import sys
import traceback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-source', type=Path, required=True)
    parser.add_argument('--blender', type=Path, required=True)
    parser.add_argument('--blender-ocio', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--scene')
    parser.add_argument('--frame', type=int)
    parser.add_argument('--view-layer')
    parser.add_argument('--neural-node')
    parser.add_argument('--strength', type=float, default=1.0)
    parser.add_argument('--identity-only', action='store_true')
    parser.add_argument('--eager', action='store_true')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    try:
        if not (args.model_source / 'dlss5' / 'graph.py').is_file():
            raise ValueError('Model source must contain dlss5/graph.py (vendor/dlss5-onnx/src)')
        if not args.blender.is_file():
            raise ValueError('Blender executable not found')
        # The separately supplied model source is imported in this worker only.
        # Blender's embedded Python never imports torch or the model modules.
        # Blender exports its bundled OCIO path into the process environment.
        # Drop only that exact inherited path; custom configurations retain the
        # pipeline's explicit rejection. The interactive environment is unchanged.
        inherited_ocio = os.environ.get('OCIO')
        if inherited_ocio and args.blender_ocio and args.blender_ocio.is_file():
            if Path(inherited_ocio).resolve() == args.blender_ocio.resolve():
                os.environ.pop('OCIO')
        sys.path.insert(0, str(args.model_source.resolve()))
        here = Path(__file__).resolve().parent / 'runtime' / 'experiments' / 'current_frame'
        sys.path.insert(0, str(here))
        import run_pipeline as pipeline
        import torch
        import numpy
        import PIL
        import OpenEXR
        needs_cuda = not args.identity_only and args.strength > 0
        if needs_cuda:
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA-enabled PyTorch and a compatible NVIDIA GPU are required')
            if not args.checkpoint or not args.checkpoint.is_file():
                raise ValueError('Choose a compatible dlss5_static.pt checkpoint')
        if args.check:
            result = {'status': 'passed', 'python': sys.executable, 'torch': torch.__version__,
                      'cuda_available': torch.cuda.is_available(), 'numpy': numpy.__version__,
                      'Pillow': PIL.__version__, 'OpenEXR': OpenEXR.__version__,
                      'model_source': str(args.model_source.resolve()),
                      'checkpoint_exists': bool(args.checkpoint and args.checkpoint.is_file())}
            (args.out / 'environment.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
            print('ENVIRONMENT_OK', json.dumps(result), flush=True)
            return
        if not args.source or not args.source.is_file():
            raise ValueError('Scene snapshot not found')
        pipeline.BLENDER = args.blender.resolve()
        args.reuse_raw = False
        pipeline.run(args)
    except Exception as exc:
        (args.out / 'error.json').write_text(json.dumps(
            {'status': 'failed', 'type': type(exc).__name__, 'error': str(exc)}, indent=2), encoding='utf-8')
        traceback.print_exc()
        raise SystemExit(1)


if __name__ == '__main__':
    main()
