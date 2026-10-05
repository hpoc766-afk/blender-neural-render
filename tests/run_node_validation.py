"""Build and install the extension in isolated user folders, then test the node."""
import argparse
from pathlib import Path
import os
import subprocess
import sys
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--blender', required=True)
parser.add_argument('--python', default=sys.executable, help='CUDA-enabled external Python')
parser.add_argument('--model-source', required=True)
parser.add_argument('--checkpoint', required=True)
parser.add_argument('--fixture', required=True, help='Small .blend with a mesh and one connected Render Layers source')
args = parser.parse_args()
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/node_validation'
OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'integration_report.json').unlink(missing_ok=True)
(ROOT / 'dist').mkdir(exist_ok=True)
BLENDER = str(Path(args.blender).resolve())
ISOLATED = OUT / 'isolated' / str(time.time_ns())
env = os.environ.copy()
for key, subdir in [('BLENDER_USER_CONFIG','config'),('BLENDER_USER_SCRIPTS','scripts'),
                    ('BLENDER_USER_EXTENSIONS','extensions'),('BLENDER_USER_DATAFILES','datafiles')]:
    path = ISOLATED / subdir
    path.mkdir(parents=True, exist_ok=True)
    env[key] = str(path)
for key, value in [('OUTPUT',OUT),('PYTHON',args.python),('MODEL_SOURCE',args.model_source),
                   ('CHECKPOINT',args.checkpoint),('FIXTURE',args.fixture)]:
    env['NRB_TEST_' + key] = str(Path(value).resolve())
commands = [
    ('build',['--command','extension','build','--source-dir',str(ROOT),'--output-dir',str(ROOT/'dist')]),
    ('validate',['--command','extension','validate',str(ROOT/'dist/neural_render_bridge-0.2.0.zip')]),
    ('repository',['--command','extension','repo-add','nrb_node_test','--name','NRB node test','--directory',str(ISOLATED/'repository')]),
    ('install',['--command','extension','install-file','--repo','nrb_node_test','--enable',str(ROOT/'dist/neural_render_bridge-0.2.0.zip')]),
    ('integration',['--background','--python-exit-code','1','--python',str(ROOT/'tests/node_integration.py')]),
]
for name, command in commands:
    print('STAGE',name,flush=True)
    with (OUT/(name+'.log')).open('w',encoding='utf-8') as stream:
        result = subprocess.run([BLENDER,*command],env=env,stdout=stream,stderr=subprocess.STDOUT)
    if result.returncode:
        print((OUT/(name+'.log')).read_text(encoding='utf-8',errors='replace'),flush=True)
        raise SystemExit(result.returncode)
print('NODE_VALIDATION_PASSED',flush=True)
