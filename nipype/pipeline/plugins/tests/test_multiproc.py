# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Test the resource management of MultiProc
"""

import sys
import os
import pytest
from nipype.pipeline import engine as pe
from nipype.interfaces import base as nib


class InputSpec(nib.TraitedSpec):
    input1 = nib.traits.Int(desc="a random int")
    input2 = nib.traits.Int(desc="a random int")


class OutputSpec(nib.TraitedSpec):
    output1 = nib.traits.List(nib.traits.Int, desc="outputs")


class MultiprocTestInterface(nib.BaseInterface):
    input_spec = InputSpec
    output_spec = OutputSpec

    def _run_interface(self, runtime):
        runtime.returncode = 0
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["output1"] = [1, self.inputs.input1]
        return outputs


@pytest.mark.skipif(
    sys.version_info >= (3, 8), reason="multiprocessing issues in Python 3.8"
)
def test_run_multiproc(tmpdir):
    tmpdir.chdir()

    pipe = pe.Workflow(name="pipe")
    mod1 = pe.Node(MultiprocTestInterface(), name="mod1")
    mod2 = pe.MapNode(MultiprocTestInterface(), iterfield=["input1"], name="mod2")
    pipe.connect([(mod1, mod2, [("output1", "input1")])])
    pipe.base_dir = os.getcwd()
    mod1.inputs.input1 = 1
    pipe.config["execution"]["poll_sleep_duration"] = 2
    execgraph = pipe.run(plugin="MultiProc")
    names = [node.fullname for node in execgraph.nodes()]
    node = list(execgraph.nodes())[names.index("pipe.mod1")]
    result = node.get_output("output1")
    assert result == [1, 1]


class InputSpecSingleNode(nib.TraitedSpec):
    input1 = nib.traits.Int(desc="a random int")
    input2 = nib.traits.Int(desc="a random int")
    use_gpu = nib.traits.Bool(False, mandatory=False, desc="boolean for GPU nodes")


class OutputSpecSingleNode(nib.TraitedSpec):
    output1 = nib.traits.Int(desc="a random int")


class SingleNodeTestInterface(nib.BaseInterface):
    input_spec = InputSpecSingleNode
    output_spec = OutputSpecSingleNode

    def _run_interface(self, runtime):
        runtime.returncode = 0
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["output1"] = self.inputs.input1
        return outputs


class ErrorInterface(SingleNodeTestInterface):
    def _run_interface(self, runtime):
        raise RuntimeError("This is an error")


def test_no_more_memory_than_specified(tmpdir):
    tmpdir.chdir()
    pipe = pe.Workflow(name="pipe")
    n1 = pe.Node(SingleNodeTestInterface(), name="n1", mem_gb=1)
    n2 = pe.Node(SingleNodeTestInterface(), name="n2", mem_gb=1)
    n3 = pe.Node(SingleNodeTestInterface(), name="n3", mem_gb=1)
    n4 = pe.Node(SingleNodeTestInterface(), name="n4", mem_gb=1)

    pipe.connect(n1, "output1", n2, "input1")
    pipe.connect(n1, "output1", n3, "input1")
    pipe.connect(n2, "output1", n4, "input1")
    pipe.connect(n3, "output1", n4, "input2")
    n1.inputs.input1 = 1

    max_memory = 0.5
    with pytest.raises(RuntimeError):
        pipe.run(
            plugin="MultiProc", plugin_args={"memory_gb": max_memory, "n_procs": 2}
        )


def test_no_more_threads_than_specified(tmpdir):
    tmpdir.chdir()

    pipe = pe.Workflow(name="pipe")
    n1 = pe.Node(SingleNodeTestInterface(), name="n1", n_procs=2)
    n2 = pe.Node(SingleNodeTestInterface(), name="n2", n_procs=2)
    n3 = pe.Node(SingleNodeTestInterface(), name="n3", n_procs=4)
    n4 = pe.Node(SingleNodeTestInterface(), name="n4", n_procs=2)

    pipe.connect(n1, "output1", n2, "input1")
    pipe.connect(n1, "output1", n3, "input1")
    pipe.connect(n2, "output1", n4, "input1")
    pipe.connect(n3, "output1", n4, "input2")
    n1.inputs.input1 = 4

    max_threads = 2
    with pytest.raises(RuntimeError):
        pipe.run(plugin="MultiProc", plugin_args={"n_procs": max_threads})


def test_no_more_gpu_threads_than_specified(tmpdir):
    tmpdir.chdir()

    pipe = pe.Workflow(name="pipe")
    n1 = pe.Node(SingleNodeTestInterface(), name="n1", n_procs=2)
    n1.inputs.use_gpu = True
    n1.inputs.input1 = 4
    pipe.add_nodes([n1])

    max_threads = 2
    max_gpu = 1
    with pytest.raises(RuntimeError):
        pipe.run(
            plugin="MultiProc",
            plugin_args={"n_procs": max_threads, 'n_gpu_procs': max_gpu},
        )


@pytest.mark.skipif(
    sys.version_info >= (3, 8), reason="multiprocessing issues in Python 3.8"
)
def test_hold_job_until_procs_available(tmpdir):
    tmpdir.chdir()

    pipe = pe.Workflow(name="pipe")
    n1 = pe.Node(SingleNodeTestInterface(), name="n1", n_procs=2)
    n2 = pe.Node(SingleNodeTestInterface(), name="n2", n_procs=2)
    n3 = pe.Node(SingleNodeTestInterface(), name="n3", n_procs=2)
    n4 = pe.Node(SingleNodeTestInterface(), name="n4", n_procs=2)

    pipe.connect(n1, "output1", n2, "input1")
    pipe.connect(n1, "output1", n3, "input1")
    pipe.connect(n2, "output1", n4, "input1")
    pipe.connect(n3, "output1", n4, "input2")
    n1.inputs.input1 = 4

    max_threads = 2
    pipe.run(plugin="MultiProc", plugin_args={"n_procs": max_threads})


@pytest.mark.parametrize("plugin", ["MultiProc", "LegacyMultiProc"])
def test_error_run_without_submitting(tmp_path, plugin):
    wf = pe.Workflow(name='rws', base_dir=str(tmp_path))
    n1 = pe.Node(SingleNodeTestInterface(), name='n1')
    n1.inputs.input1 = 1
    n2 = pe.Node(ErrorInterface(), name='n2', run_without_submitting=True)
    n3 = pe.Node(SingleNodeTestInterface(), name='n3')

    wf.connect(
        [
            (n1, n2, [('output1', 'input1')]),
            (n2, n3, [('output1', 'input1')]),
        ]
    ),

    with pytest.raises(RuntimeError):
        wf.run(plugin=plugin)


class _LimitWrapper(nib.DockerContainerWrapper):
    """Docker wrapper stub with a fixed, injectable CPU cap (no docker call)."""

    def __init__(self, image, limit):
        super().__init__(image)
        self._limit = limit

    def cpu_limit(self):
        return self._limit


def _fake_node(container, n_procs, name="wf.node"):
    from types import SimpleNamespace

    inputs = SimpleNamespace(container=container)
    return SimpleNamespace(
        interface=SimpleNamespace(inputs=inputs), n_procs=n_procs, fullname=name
    )


class _FakeGraph:
    def __init__(self, nodes):
        self._nodes = nodes

    def nodes(self):
        return self._nodes


def _plugin(raise_insufficient):
    """A MultiProcPlugin without going through __init__ (no worker pool)."""
    from nipype.pipeline.plugins.multiproc import MultiProcPlugin

    plugin = object.__new__(MultiProcPlugin)
    plugin.raise_insufficient = raise_insufficient
    return plugin


def test_container_cpu_limit_raises_when_exceeded():
    graph = _FakeGraph([_fake_node(_LimitWrapper("img", 2), n_procs=4)])
    with pytest.raises(RuntimeError, match="Insufficient docker CPU"):
        _plugin(raise_insufficient=True)._check_container_cpu_limits(graph)


def test_container_cpu_limit_warns_without_raise(caplog):
    import logging

    graph = _FakeGraph([_fake_node(_LimitWrapper("img", 2), n_procs=4)])
    with caplog.at_level(logging.WARNING, logger="nipype.workflow"):
        _plugin(raise_insufficient=False)._check_container_cpu_limits(graph)
    assert "only has 2 CPUs available" in caplog.text


def test_container_cpu_limit_within_bounds_ok():
    graph = _FakeGraph([_fake_node(_LimitWrapper("img", 4), n_procs=4)])
    # n_procs == limit is fine, must not raise
    _plugin(raise_insufficient=True)._check_container_cpu_limits(graph)


def test_container_cpu_limit_none_is_noop():
    graph = _FakeGraph([_fake_node(_LimitWrapper("img", None), n_procs=999)])
    _plugin(raise_insufficient=True)._check_container_cpu_limits(graph)


def test_container_cpu_limit_skips_non_container_nodes():
    graph = _FakeGraph([_fake_node(nib.Undefined, n_procs=999)])
    _plugin(raise_insufficient=True)._check_container_cpu_limits(graph)
