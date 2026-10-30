# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import sys

import pytest

from nipype.utils import profiler
from nipype.utils.profiler import get_system_total_memory_gb


def test_get_system_total_memory_gb():
    memory_gb = get_system_total_memory_gb()
    assert memory_gb > 0


def test_get_system_total_memory_gb_other_platform(monkeypatch):
    psutil = pytest.importorskip("psutil")
    monkeypatch.setattr(sys, "platform", "win32")
    expected = psutil.virtual_memory().total / (1024.0**3)
    assert get_system_total_memory_gb() == pytest.approx(expected)


def test_get_system_total_memory_gb_no_psutil(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(profiler, "psutil", None)
    with pytest.raises(Exception, match="not supported"):
        get_system_total_memory_gb()
