"""One owned process tree at a time; usable without Blender."""
import json
import os
from pathlib import Path
import signal
import subprocess


class Job:
    def __init__(self, command, directory, kind):
        self.directory = Path(directory)
        self.kind = kind
        self.status = 'RUNNING'
        self.error = ''
        self.stream = (self.directory / 'worker.log').open('w', encoding='utf-8')
        options = {'start_new_session': True} if os.name != 'nt' else {
            'creationflags': subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW}
        try:
            self.process = subprocess.Popen(command, stdout=self.stream, stderr=subprocess.STDOUT, **options)
        except Exception:
            self.stream.close()
            raise
        self.pid = self.process.pid

    def poll(self):
        if self.status != 'RUNNING':
            return self.status
        code = self.process.poll()
        if code is None:
            return self.status
        self.stream.close()
        filename = 'environment.json' if self.kind == 'check' else 'pipeline_report.json'
        try:
            report = json.loads((self.directory / filename).read_text(encoding='utf-8'))
            passed = report.get('status') == 'passed'
        except (OSError, ValueError):
            passed = False
        self.status = 'SUCCEEDED' if code == 0 and passed else 'FAILED'
        if self.status == 'FAILED':
            try:
                self.error = json.loads((self.directory / 'error.json').read_text(encoding='utf-8'))['error']
            except (OSError, ValueError, KeyError):
                self.error = f'Worker exited with code {code}; see worker.log and stage logs'
        return self.status

    def cancel(self):
        if self.status != 'RUNNING':
            return
        if os.name == 'nt':
            if self.process.poll() is None:
                result = subprocess.run(['taskkill', '/PID', str(self.pid), '/T', '/F'],
                                        capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
                if result.returncode and self.process.poll() is None:
                    raise RuntimeError('Cannot terminate the owned worker process tree')
        else:
            # The worker is a session leader; Blender children share its group.
            # Also terminate surviving children if the parent has already exited.
            try:
                os.killpg(self.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        self.process.wait(timeout=5)
        self.stream.close()
        self.status = 'CANCELLED'
        (self.directory / 'cancelled.json').write_text(json.dumps(
            {'status': 'cancelled', 'worker_pid': self.pid}), encoding='utf-8')

    def stage(self):
        if self.status != 'RUNNING':
            return self.status
        for filename, label in [('neural_post.log', 'Neural compositor replay'),
                                ('identity_post.log', 'Inference / identity validation'),
                                ('baseline_post.log', 'Original compositor replay'),
                                ('raw.log', 'Native render and HDR capture'),
                                ('preflight.log', 'Scene validation')]:
            if (self.directory / filename).exists():
                return label
        return 'Checking environment' if self.kind == 'check' else 'Starting worker'
