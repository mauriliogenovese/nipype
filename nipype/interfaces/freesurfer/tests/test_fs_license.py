# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Unit tests for FreeSurfer's containerized-license handling (FSLicenseMixin)
and ReconAll's container-aware umask default. Pure Python -- no FreeSurfer
installation required."""

import os
from pathlib import Path

import pytest

from nipype.interfaces.base import DockerContainerWrapper, isdefined
from nipype.interfaces.freesurfer import ReconAll
from nipype.interfaces.freesurfer.base import FSLicenseMixin


def _license(tmp_path, name="license.txt"):
    lic = tmp_path / name
    lic.write_text("dummy license")
    return lic


def test_resolved_license_none(monkeypatch):
    monkeypatch.delenv("FS_LICENSE", raising=False)
    assert ReconAll()._resolved_license_file() is None


def test_resolved_license_from_env(tmp_path, monkeypatch):
    lic = _license(tmp_path)
    monkeypatch.setenv("FS_LICENSE", str(lic))
    assert ReconAll()._resolved_license_file() == str(lic.resolve())


def test_resolved_license_inputs_override_env(tmp_path, monkeypatch):
    env_lic = _license(tmp_path, "env.txt")
    input_lic = _license(tmp_path, "input.txt")
    monkeypatch.setenv("FS_LICENSE", str(env_lic))
    r = ReconAll()
    r.inputs.license_file = str(input_lic)
    # this instance's own license_file wins over the environment default
    assert r._resolved_license_file() == str(input_lic.resolve())


def test_container_extra_hooks_inject_license(tmp_path, monkeypatch):
    monkeypatch.delenv("FS_LICENSE", raising=False)
    lic = _license(tmp_path)
    r = ReconAll()
    r.inputs.container = DockerContainerWrapper("freesurfer/freesurfer:7.4.1")
    r.inputs.license_file = str(lic)

    mounts = r._container_extra_mounts()
    assert (str(lic.resolve()), FSLicenseMixin._CONTAINER_LICENSE_PATH, ":ro") in mounts

    env = r._container_extra_env()
    assert env["FS_LICENSE"] == FSLicenseMixin._CONTAINER_LICENSE_PATH

    prelude = r._container_extra_prelude()
    assert any(FSLicenseMixin._CONTAINER_LICENSE_PATH in cmd for cmd in prelude)
    assert any("license.txt" in cmd for cmd in prelude)


def test_container_extra_hooks_empty_without_license(tmp_path, monkeypatch):
    monkeypatch.delenv("FS_LICENSE", raising=False)
    r = ReconAll()
    r.inputs.container = DockerContainerWrapper("freesurfer/freesurfer:7.4.1")
    assert r._container_extra_mounts() == []
    assert r._container_extra_env() == {}
    assert r._container_extra_prelude() == []


def test_set_default_license_file(tmp_path):
    lic = _license(tmp_path)
    saved = os.environ.get("FS_LICENSE")
    try:
        FSLicenseMixin.set_default_license_file(str(lic))
        assert os.environ["FS_LICENSE"] == os.path.abspath(str(lic))
        with pytest.raises(ValueError):
            FSLicenseMixin.set_default_license_file(str(tmp_path / "missing.txt"))
    finally:
        if saved is None:
            os.environ.pop("FS_LICENSE", None)
        else:
            os.environ["FS_LICENSE"] = saved


def test_reconall_umask_defaults_to_zero_in_container():
    r = ReconAll()
    assert not isdefined(r.inputs.umask)
    r.inputs.container = DockerContainerWrapper("img")
    # containerized ReconAll runs as root; umask 0 keeps outputs host-accessible
    assert r.inputs.umask == 0


def test_reconall_umask_not_overridden_when_set():
    r = ReconAll()
    r.inputs.umask = 2
    r.inputs.container = DockerContainerWrapper("img")
    assert r.inputs.umask == 2
