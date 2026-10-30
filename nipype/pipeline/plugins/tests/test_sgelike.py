import nipype
from nipype.pipeline.plugins.base import SGELikeBatchManagerBase
from nipype.interfaces.utility import Function
import nipype.pipeline.engine as pe
import pytest
from unittest.mock import patch
import os
import subprocess
import sys


def crasher():
    raise ValueError


def submit_batchtask(self, scriptfile, node):
    self._pending[1] = node.output_dir()
    subprocess.call(["bash", scriptfile])
    return 1


def is_pending(self, taskid):
    return False


@patch.object(SGELikeBatchManagerBase, "_submit_batchtask", new=submit_batchtask)
@patch.object(SGELikeBatchManagerBase, "_is_pending", new=is_pending)
def test_crashfile_creation(tmp_path):
    pipe = pe.Workflow(name="pipe", base_dir=str(tmp_path))
    pipe.config["execution"]["crashdump_dir"] = str(tmp_path)
    pipe.add_nodes([pe.Node(interface=Function(function=crasher), name="crasher")])
    sgelike_plugin = SGELikeBatchManagerBase("")
    with pytest.raises(RuntimeError):
        assert pipe.run(plugin=sgelike_plugin)

    crashfiles = list(tmp_path.glob("crash*crasher*.pklz")) + list(
        tmp_path.glob("crash*crasher*.txt")
    )
    assert len(crashfiles) == 1


def test_plugins_import_without_pwd():
    # pwd is POSIX-only; plugins must still import where it is unavailable
    code = "import sys; sys.modules['pwd'] = None; import nipype.pipeline.plugins"
    root = os.path.dirname(os.path.dirname(nipype.__file__))
    subprocess.run([sys.executable, "-c", code], check=True, cwd=root)
