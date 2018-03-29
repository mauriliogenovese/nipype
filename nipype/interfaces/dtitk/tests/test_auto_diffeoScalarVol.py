# AUTO-GENERATED by tools/checkspecs.py - DO NOT EDIT
from __future__ import unicode_literals
from ..registration import diffeoScalarVol


def test_diffeoScalarVol_inputs():
    input_map = dict(
        args=dict(argstr='%s', ),
        environ=dict(
            nohash=True,
            usedefault=True,
        ),
        flip=dict(
            argstr='-flip %d %d %d',
            exists=True,
        ),
        ignore_exception=dict(
            deprecated='1.0.0',
            nohash=True,
            usedefault=True,
        ),
        in_file=dict(
            argstr='-in %s',
            mandatory=True,
        ),
        interpolation=dict(
            argstr='-interp %s',
            exists=True,
            usedefault=True,
        ),
        out_file=dict(
            argstr='-out %s',
            keep_extension=True,
            name_source='in_file',
            name_template='%s_diffeoxfmd',
        ),
        resampling_type=dict(
            argstr='-type %s',
            exists=True,
        ),
        target=dict(
            argstr='-target %s',
            xor=['voxel_size'],
        ),
        terminal_output=dict(
            deprecated='1.0.0',
            nohash=True,
        ),
        transform=dict(
            argstr='-trans %s',
            mandatory=True,
        ),
        voxel_size=dict(
            argstr='-vsize %g %g %g',
            xor=['target'],
        ),
    )
    inputs = diffeoScalarVol.input_spec()

    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value
def test_diffeoScalarVol_outputs():
    output_map = dict(out_file=dict(), )
    outputs = diffeoScalarVol.output_spec()

    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value
