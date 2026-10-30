import os

import pytest


from nipype.interfaces import dcm2nii


@pytest.mark.parametrize(
    "fname, extension, search_crop",
    [
        ("output_1", ".txt", False),
        ("output_w_[]_meta_1", ".json", False),
        ("output_w_**^$?_meta_2", ".txt", False),
        ("output_cropped", ".txt", True),
    ],
)
def test_search_files(tmp_path, fname, extension, search_crop):
    tmp_fname = fname + extension
    test_file = tmp_path / tmp_fname
    test_file.touch()
    if search_crop:
        tmp_cropped_fname = fname + "_Crop_1" + extension
        test_cropped_file = tmp_path / tmp_cropped_fname
        test_cropped_file.touch()

    actual_files_list = dcm2nii.search_files(
        str(tmp_path / fname), [extension], search_crop
    )
    for f in actual_files_list:
        if search_crop:
            assert f in (str(test_cropped_file), str(test_file))
        else:
            assert str(test_file) == f


@pytest.mark.parametrize(
    "line, fname",
    [
        ("Convert 1 DICOM as /out/name (64x64x30x1)", "/out/name"),
        ("Convert 30 DICOM as ./name (64x64x30x1)", "./name"),
        ("Convert 1 DICOM as /out dir/my name (64x64x30x1)", "/out dir/my name"),
        ("Convert 1 DICOM as /out/name (1) (64x64x30x1)", "/out/name (1)"),
        (r"Convert 1 DICOM as .\name (64x64x30x1)", r".\name"),
        (r"Convert 1 DICOM as C:\a b\name (64x64x30x1)", r"C:\a b\name"),
        ("Convert 1 DICOM as /out/name (64x64x30x1)\r", "/out/name"),
        ("Convert 1 DICOM as /out/name", "/out/name"),
    ],
)
def test_dcm2niix_parse_stdout(line, fname):
    stdout = "\n".join(
        [
            "Chris Rorden's dcm2niiX version v1.0.20240202",
            "Found 1 DICOM file(s)",
            line,
            "Conversion required 0.012345 seconds (0.010000 for core code).",
        ]
    )
    assert dcm2nii.Dcm2niix()._parse_stdout(stdout) == [os.path.abspath(fname)]
