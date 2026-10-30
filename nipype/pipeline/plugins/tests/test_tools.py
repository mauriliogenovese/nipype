# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Tests for the engine module"""

import os
import re

from unittest import mock

import pytest

from nipype.pipeline.plugins.tools import report_crash


def test_report_crash():
    with mock.patch("pickle.dump", mock.MagicMock()) as mock_pickle_dump:
        with mock.patch(
            "nipype.pipeline.plugins.tools.format_exception", mock.MagicMock()
        ):  # see iss 1517
            mock_pickle_dump.return_value = True
            mock_node = mock.MagicMock(name="mock_node")
            mock_node._id = "an_id"
            mock_node.config = {
                "execution": {"crashdump_dir": ".", "crashfile_format": "pklz"}
            }

            actual_crashfile = report_crash(mock_node)

            expected_crashfile = re.compile(r".*/crash-.*-an_id-[0-9a-f\-]*.pklz")

            assert (
                expected_crashfile.match(actual_crashfile).group() == actual_crashfile
            )
            assert mock_pickle_dump.call_count == 1


@pytest.mark.parametrize("uid", [1234, None])
def test_report_crash_no_login_name(tmp_path, monkeypatch, uid):
    """Fall back gracefully when the login name cannot be determined."""
    monkeypatch.setattr("getpass.getuser", mock.Mock(side_effect=OSError))
    if uid is None:  # e.g., Windows, where os.getuid does not exist
        monkeypatch.delattr(os, "getuid", raising=False)
        expected_login = "unknown"
    else:
        monkeypatch.setattr(os, "getuid", lambda: uid, raising=False)
        expected_login = f"UID{uid}"

    mock_node = mock.MagicMock(name="mock_node")
    mock_node._id = "an_id"
    mock_node.config = {
        "execution": {"crashdump_dir": str(tmp_path), "crashfile_format": "txt"}
    }

    with mock.patch("nipype.pipeline.plugins.tools.crash2txt") as mock_crash2txt:
        crashfile = report_crash(mock_node, traceback=["traceback"])

    assert mock_crash2txt.call_count == 1
    assert re.match(
        rf"crash-\d+-\d+-{expected_login}-an_id-[0-9a-f\-]*\.txt$",
        os.path.basename(crashfile),
    )


"""
Can use the following code to test that a mapnode crash continues successfully
Need to put this into a unit-test with a timeout

import nipype.interfaces.utility as niu
import nipype.pipeline.engine as pe

wf = pe.Workflow(name='test')

def func(arg1):
    if arg1 == 2:
        raise Exception('arg cannot be ' + str(arg1))
    return arg1

funkynode = pe.MapNode(niu.Function(function=func, input_names=['arg1'],
                                                   output_names=['out']),
                       iterfield=['arg1'],
                       name = 'functor')
funkynode.inputs.arg1 = [1,2]

wf.add_nodes([funkynode])
wf.base_dir = '/tmp'

wf.run(plugin='MultiProc')
"""
