import os
import sys
from types import SimpleNamespace

from ..subprocess import run_command


def test_run_command_stream(tmp_path):
    script = "import sys; print('out1'); print('err1', file=sys.stderr); print('out2')"
    runtime = SimpleNamespace(
        cmdline=f'"{sys.executable}" -c "{script}"',
        environ=dict(os.environ),
        cwd=str(tmp_path),
    )
    runtime = run_command(runtime, output="stream")
    assert runtime.returncode == 0
    assert runtime.stdout.splitlines() == ["out1", "out2"]
    assert runtime.stderr.splitlines() == ["err1"]
    assert len(runtime.merged.splitlines()) == 3
