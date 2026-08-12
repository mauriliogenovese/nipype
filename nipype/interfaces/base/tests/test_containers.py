# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Unit tests for the container-engine wrappers (:mod:`nipype.interfaces.base.containers`)."""

import shlex
from pathlib import Path

import pytest

from ..containers import (
    ContainerRunSpec,
    ContainerWrapper,
    DockerContainerWrapper,
)
from .. import containers as containers_mod


class _Run:
    """Minimal stand-in for a ``subprocess.run`` CompletedProcess result."""

    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _spec(cwd, **kwargs):
    kwargs.setdefault("image", "busybox:latest")
    kwargs.setdefault("inner_cmdline", "true")
    return ContainerRunSpec(cwd=str(cwd), **kwargs)


def _values(tokens, flag):
    """All values following ``flag`` in a token list (e.g. every ``-v`` mount).

    Parsing the command back with ``shlex.split`` makes assertions robust to
    quoting -- the test tmp dirs can contain spaces (see conftest)."""
    return [tokens[i + 1] for i, t in enumerate(tokens[:-1]) if t == flag]


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


def test_docker_wrapper_basics():
    wrapper = DockerContainerWrapper("img:tag")
    assert isinstance(wrapper, ContainerWrapper)
    assert wrapper.image == "img:tag"
    assert wrapper.container_type == "docker"


def test_abstract_cannot_instantiate():
    with pytest.raises(TypeError):
        ContainerWrapper("img")


# ---------------------------------------------------------------------------
# build_command -- Linux/macOS mirror-mount path
# ---------------------------------------------------------------------------


def test_build_command_basic(tmp_path):
    wrapper = DockerContainerWrapper("busybox:latest")
    cmd = wrapper.build_command(_spec(tmp_path, inner_cmdline="ls -la"))
    tokens = shlex.split(cmd)

    assert tokens[:4] == ["docker", "run", "--rm", "--init"]
    # working directory and mirror mount for the (writable) cwd
    assert str(tmp_path) in _values(tokens, "-w")
    assert f"{tmp_path}:{tmp_path}" in _values(tokens, "-v")
    assert "busybox:latest" in tokens
    # inner command runs under sh -c with a permissive umask
    assert tokens[-2] == "-c"
    assert tokens[-3] == "sh"
    assert tokens[-1].startswith("umask 0000;")
    assert tokens[-1].endswith("ls -la")
    # no GPU / CPU flags unless requested
    assert "--gpus" not in tokens
    assert "--cpus" not in tokens


def test_build_command_gpu_and_cpus(tmp_path):
    wrapper = DockerContainerWrapper("img")
    cmd = wrapper.build_command(_spec(tmp_path, use_gpu=True, num_threads=4))
    assert "--gpus all" in cmd
    assert "--cpus 4" in cmd


def test_build_command_environ_and_extra_mounts(tmp_path):
    wrapper = DockerContainerWrapper("img")
    spec = _spec(
        tmp_path,
        environ={"FOO": "bar"},
        extra_mounts=[("/host/lic.txt", "/opt/lic.txt", ":ro")],
    )
    cmd = wrapper.build_command(spec)
    assert "-e FOO=bar" in cmd
    assert "-v /host/lic.txt:/opt/lic.txt:ro" in cmd


def test_build_command_prelude(tmp_path):
    wrapper = DockerContainerWrapper("img")
    cmd = wrapper.build_command(
        _spec(tmp_path, inner_cmdline="MAINCMD", prelude=["setup one", "setup two"])
    )
    sh_arg = shlex.split(cmd)[-1]
    # each prelude entry is best-effort (stderr swallowed) and runs before the cmd
    assert "{ setup one ; } 2>/dev/null" in sh_arg
    assert "{ setup two ; } 2>/dev/null" in sh_arg
    assert sh_arg.index("setup one") < sh_arg.index("MAINCMD")


def test_build_command_collapses_nested_mounts(tmp_path):
    cwd = tmp_path / "work"
    parent = tmp_path / "parent"
    child = parent / "child"
    cwd.mkdir()
    child.mkdir(parents=True)
    wrapper = DockerContainerWrapper("img")
    cmd = wrapper.build_command(_spec(cwd, mount_roots={parent, child}))
    mounts = _values(shlex.split(cmd), "-v")
    # child is covered by parent -> only the outermost root is mounted
    assert f"{parent}:{parent}" in mounts
    assert f"{child}:{child}" not in mounts


def test_build_command_readonly_mount_when_not_writable(tmp_path, monkeypatch):
    # cwd and ro are siblings, so ro is not collapsed into the cwd mount
    cwd = tmp_path / "work"
    ro = tmp_path / "ro"
    cwd.mkdir()
    ro.mkdir()
    # simulate a non-writable host directory
    monkeypatch.setattr(containers_mod.os, "access", lambda p, mode: str(p) != str(ro))
    wrapper = DockerContainerWrapper("img")
    cmd = wrapper.build_command(_spec(cwd, mount_roots={ro}))
    mounts = _values(shlex.split(cmd), "-v")
    assert f"{ro}:{ro}:ro" in mounts
    assert f"{cwd}:{cwd}" in mounts  # writable cwd stays read-write


# ---------------------------------------------------------------------------
# build_command -- Windows path-mapping branch
# ---------------------------------------------------------------------------


def test_build_command_windows_path_map(monkeypatch):
    monkeypatch.setattr(containers_mod, "_is_windows", lambda: True)
    wrapper = DockerContainerWrapper("img")
    spec = ContainerRunSpec(
        image="img",
        cwd="/data/work",
        inner_cmdline="tool /data/work/in.nii",
        mount_roots={Path("/data")},
    )
    cmd = wrapper.build_command(spec)
    # host paths are replaced by canonical POSIX container paths
    assert "/mnt/nipype_vol0" in cmd
    # the working directory maps under the mounted root
    assert "/mnt/nipype_vol0/work" in cmd
    # the original host path no longer appears in the inner command
    assert "/data/work/in.nii" not in cmd


# ---------------------------------------------------------------------------
# Path mechanics (static helpers)
# ---------------------------------------------------------------------------


def test_collapse_mount_roots():
    paths = {Path("/a"), Path("/a/b"), Path("/a/b/c"), Path("/x/y")}
    kept = set(ContainerWrapper._collapse_mount_roots(paths))
    assert kept == {Path("/a"), Path("/x/y")}


def test_build_path_map_deterministic():
    m1 = ContainerWrapper._build_path_map({Path("/b"), Path("/a")}, Path("/a/work"))
    m2 = ContainerWrapper._build_path_map({Path("/a"), Path("/b")}, Path("/a/work"))
    assert m1 == m2  # order-independent, stable across runs
    assert set(m1.values()) == {"/mnt/nipype_vol0", "/mnt/nipype_vol1"}


def test_container_path_for_nested():
    path_map = {Path("/data"): "/mnt/nipype_vol0"}
    assert (
        ContainerWrapper._container_path_for(Path("/data/sub/x"), path_map)
        == "/mnt/nipype_vol0/sub/x"
    )
    # the root itself maps to the bare container root
    assert (
        ContainerWrapper._container_path_for(Path("/data"), path_map)
        == "/mnt/nipype_vol0"
    )
    # a path outside every root is unmapped
    assert ContainerWrapper._container_path_for(Path("/other"), path_map) is None


def test_remap_cmdline_flips_separators():
    path_map = {Path("/data"): "/mnt/nipype_vol0"}
    out = ContainerWrapper._remap_cmdline(r"tool \data\sub\x", path_map)
    # backslashes become POSIX separators
    assert "\\" not in out


def test_mount_mode(tmp_path, monkeypatch):
    assert DockerContainerWrapper._mount_mode(tmp_path) == ""  # writable
    monkeypatch.setattr(containers_mod.os, "access", lambda p, mode: False)
    assert DockerContainerWrapper._mount_mode(tmp_path) == ":ro"


# ---------------------------------------------------------------------------
# cpu_limit
# ---------------------------------------------------------------------------


def test_cpu_limit_default_none_off_windows(monkeypatch):
    monkeypatch.setattr(containers_mod, "_is_windows", lambda: False)
    assert DockerContainerWrapper("img").cpu_limit() is None


def test_cpu_limit_reports_vm_cpus_on_windows(monkeypatch):
    monkeypatch.setattr(containers_mod, "_is_windows", lambda: True)
    monkeypatch.setattr(containers_mod.sp, "run", lambda *a, **k: _Run(stdout="3\n"))
    assert DockerContainerWrapper("img").cpu_limit() == 3


def test_cpu_limit_unknown_returns_none_on_windows(monkeypatch):
    monkeypatch.setattr(containers_mod, "_is_windows", lambda: True)
    monkeypatch.setattr(containers_mod.sp, "run", lambda *a, **k: _Run(stdout=""))
    assert DockerContainerWrapper("img").cpu_limit() is None


# ---------------------------------------------------------------------------
# check_available
# ---------------------------------------------------------------------------


def test_check_available_engine_missing(monkeypatch):
    monkeypatch.setattr(containers_mod, "which", lambda *a, **k: None)
    with pytest.raises(OSError, match="not found on host"):
        DockerContainerWrapper("img").check_available()


def test_check_available_image_not_inspectable(monkeypatch):
    monkeypatch.setattr(containers_mod, "which", lambda *a, **k: "/usr/bin/docker")
    monkeypatch.setattr(
        containers_mod.sp,
        "run",
        lambda *a, **k: _Run(returncode=1, stderr="no such image"),
    )
    with pytest.raises(RuntimeError, match="could not be inspected"):
        DockerContainerWrapper("img").check_available()


def test_check_available_non_root_image(monkeypatch):
    monkeypatch.setattr(containers_mod, "which", lambda *a, **k: "/usr/bin/docker")
    monkeypatch.setattr(containers_mod.sp, "run", lambda *a, **k: _Run(stdout="1000\n"))
    with pytest.raises(RuntimeError, match="does not run as root"):
        DockerContainerWrapper("img").check_available()


@pytest.mark.parametrize("user", ["", "0", "root", "0:0"])
def test_check_available_root_image_ok(monkeypatch, user):
    monkeypatch.setattr(containers_mod, "which", lambda *a, **k: "/usr/bin/docker")
    monkeypatch.setattr(
        containers_mod.sp, "run", lambda *a, **k: _Run(stdout=user + "\n")
    )
    assert DockerContainerWrapper("img").check_available() == "/usr/bin/docker"
