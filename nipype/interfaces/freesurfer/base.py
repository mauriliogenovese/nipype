# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""The freesurfer module provides basic functions for interfacing with
freesurfer tools.

Currently these tools are supported:

     * Dicom2Nifti: using mri_convert
     * Resample: using mri_convert

Examples
--------
See the docstrings for the individual classes for 'working' examples.

"""

import os
from pathlib import Path

from looseversion import LooseVersion
from traits.trait_base import Undefined

from ...utils.filemanip import fname_presuffix
from ..base import (
    CommandLine,
    Directory,
    CommandLineInputSpec,
    isdefined,
    traits,
    TraitedSpec,
    File,
    PackageInfo,
)

__docformat__ = "restructuredtext"


class Info(PackageInfo):
    """Freesurfer subject directory and version information.

    Examples
    --------

    >>> from nipype.interfaces.freesurfer import Info
    >>> Info.version()  # doctest: +SKIP
    >>> Info.subjectsdir()  # doctest: +SKIP

    """

    if os.getenv("FREESURFER_HOME"):
        version_file = os.path.join(os.getenv("FREESURFER_HOME"), "build-stamp.txt")

    @staticmethod
    def parse_version(raw_info):
        return raw_info.splitlines()[0]

    @classmethod
    def looseversion(cls):
        """Return a comparable version object

        If no version found, use LooseVersion('0.0.0')
        """
        ver = cls.version()
        if ver is None:
            return LooseVersion("0.0.0")

        vinfo = ver.rstrip().split("-")
        try:
            int(vinfo[-1], 16)
        except ValueError:
            githash = ""
        else:
            githash = "." + vinfo[-1]

        # As of FreeSurfer v6.0.0, the final component is a githash
        if githash:
            if vinfo[3] == "dev":
                # This will need updating when v6.0.1 comes out
                vstr = "6.0.0-dev" + githash
            elif vinfo[5][0] == "v":
                vstr = vinfo[5][1:]
            elif len([1 for val in vinfo[3] if val == "."]) == 2:
                "version string: freesurfer-linux-centos7_x86_64-7.1.0-20200511-813297b"
                vstr = vinfo[3]
            else:
                raise RuntimeError("Unknown version string: " + ver)
        # Retain pre-6.0.0 heuristics
        elif "dev" in ver:
            vstr = vinfo[-1] + "-dev"
        else:
            vstr = ver.rstrip().split("-v")[-1]

        return LooseVersion(vstr)

    @classmethod
    def subjectsdir(cls):
        """Check the global SUBJECTS_DIR

        Parameters
        ----------

        subjects_dir :  string
            The system defined subjects directory

        Returns
        -------

        subject_dir : string
            Represents the current environment setting of SUBJECTS_DIR

        """
        if cls.version():
            return os.environ["SUBJECTS_DIR"]
        return None


class FSTraitedSpec(CommandLineInputSpec):
    subjects_dir = Directory(exists=True, desc="subjects directory")
    license_file = File(
        exists=True,
        desc=(
            "Path to the FreeSurfer license file, only used when this interface runs inside a container"
        ),
    )


class FSLicenseMixin:
    """Provides FreeSurfer license resolution/injection for containerized
    execution. Used by FSCommand and by other FreeSurfer CommandLine
    subclasses that, for historical reasons, don't inherit from
    FSCommand directly (see GH nipy/nipype#2655 -- ReconAll is one of
    these; this mixin lets it opt in without depending on a hierarchy
    fix that issue explicitly decided against).
    """

    _CONTAINER_LICENSE_PATH = "/tmp/fs_license.txt"
    _default_license_file = None

    @classmethod
    def set_default_license_file(cls, license_file):
        """Set a FreeSurfer license file used by default by every class
        using this mixin (FSCommand, ReconAll, ...), unless a specific
        node overrides it via inputs.license_file. Sets the attribute
        on FSLicenseMixin itself (not on `cls`), so the default is
        shared across all FreeSurfer interfaces regardless of which one
        this is called on."""
        if not os.path.exists(os.path.abspath(license_file)):
            msg = "License file %s not found. " % license_file
            raise ValueError(msg)
        FSLicenseMixin._default_license_file = os.path.abspath(license_file)

    def _resolved_license_file(self):
        """Resolve which license file to use, in order of precedence:
        1. This instance's own inputs.license_file, if set.
        2. The class-wide default set via set_default_license_file().
        3. FS_LICENSE already defined in the host environment.
        Returns None if no license is configured anywhere.
        """
        if isdefined(getattr(self.inputs, "license_file", Undefined)):
            return str(Path(self.inputs.license_file).resolve())
        if FSLicenseMixin._default_license_file:
            return FSLicenseMixin._default_license_file
        host_license = os.environ.get("FS_LICENSE")
        if host_license:
            return str(Path(host_license).resolve())
        return None

    def _container_extra_mounts(self):
        mounts = super()._container_extra_mounts()
        license_file = self._resolved_license_file()
        if license_file:
            mounts.append((license_file, self._CONTAINER_LICENSE_PATH, ":ro"))
        return mounts

    def _container_extra_env(self):
        env = super()._container_extra_env()
        if self._resolved_license_file():
            env["FS_LICENSE"] = self._CONTAINER_LICENSE_PATH
        return env

class FSCommand(FSLicenseMixin, CommandLine):
    """General support for FreeSurfer commands.

    Every FS command accepts 'subjects_dir' input.
    """

    input_spec = FSTraitedSpec

    _subjects_dir = None

    def __init__(self, **inputs):
        super().__init__(**inputs)
        self.inputs.on_trait_change(self._subjects_dir_update, "subjects_dir")
        if not self._subjects_dir:
            self._subjects_dir = Info.subjectsdir()
        if not isdefined(self.inputs.subjects_dir) and self._subjects_dir:
            self.inputs.subjects_dir = self._subjects_dir
        self._subjects_dir_update()

    def _subjects_dir_update(self):
        if self.inputs.subjects_dir:
            self.inputs.environ.update({"SUBJECTS_DIR": self.inputs.subjects_dir})

    @classmethod
    def set_default_subjects_dir(cls, subjects_dir):
        cls._subjects_dir = subjects_dir

    def run(self, **inputs):
        if "subjects_dir" in inputs:
            self.inputs.subjects_dir = inputs["subjects_dir"]
        self._subjects_dir_update()
        return super().run(**inputs)

    def _gen_fname(self, basename, fname=None, cwd=None, suffix="_fs", use_ext=True):
        """Define a generic mapping for a single outfile

        The filename is potentially autogenerated by suffixing inputs.infile

        Parameters
        ----------
        basename : string (required)
            filename to base the new filename on
        fname : string
            if not None, just use this fname
        cwd : string
            prefix paths with cwd, otherwise os.getcwd()
        suffix : string
            default suffix
        """
        if basename == "":
            msg = "Unable to generate filename for command %s. " % self.cmd
            msg += "basename is not set!"
            raise ValueError(msg)
        if cwd is None:
            cwd = os.getcwd()
        fname = fname_presuffix(basename, suffix=suffix, use_ext=use_ext, newpath=cwd)
        return fname

    @property
    def version(self):
        ver = Info.looseversion()
        if ver > LooseVersion("0.0.0"):
            return ver.vstring


class FSSurfaceCommand(FSCommand):
    """Support for FreeSurfer surface-related functions.
    For some functions, if the output file is not specified starting with 'lh.'
    or 'rh.', FreeSurfer prepends the prefix from the input file to the output
    filename. Output out_file must be adjusted to accommodate this. By
    including the full path in the filename, we can also avoid this behavior.
    """

    @staticmethod
    def _associated_file(in_file, out_name):
        """Based on MRIsBuildFileName in freesurfer/utils/mrisurf.c

        If no path information is provided for out_name, use path and
        hemisphere (if also unspecified) from in_file to determine the path
        of the associated file.
        Use in_file prefix to indicate hemisphere for out_name, rather than
        inspecting the surface data structure.
        """
        path, base = os.path.split(out_name)
        if path == "":
            path, in_file = os.path.split(in_file)
            hemis = ("lh.", "rh.")
            if in_file[:3] in hemis and base[:3] not in hemis:
                base = in_file[:3] + base
        return os.path.join(path, base)


class FSScriptCommand(FSCommand):
    """Support for Freesurfer script commands with log terminal_output"""

    _terminal_output = "file"
    _always_run = False

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["log_file"] = os.path.abspath("output.nipype")
        return outputs


class FSScriptOutputSpec(TraitedSpec):
    log_file = File(
        "output.nipype", usedefault=True, exists=True, desc="The output log"
    )


class FSTraitedSpecOpenMP(FSTraitedSpec):
    num_threads = traits.Int(desc="allows for specifying more threads")


class FSCommandOpenMP(FSCommand):
    """Support for FS commands that utilize OpenMP

    Sets the environment variable 'OMP_NUM_THREADS' to the number
    of threads specified by the input num_threads.
    """

    input_spec = FSTraitedSpecOpenMP

    _num_threads = None

    def __init__(self, **inputs):
        super().__init__(**inputs)
        self.inputs.on_trait_change(self._num_threads_update, "num_threads")
        if not self._num_threads:
            self._num_threads = os.environ.get("OMP_NUM_THREADS", None)
            if not self._num_threads:
                self._num_threads = os.environ.get("NSLOTS", None)
        if not isdefined(self.inputs.num_threads) and self._num_threads:
            self.inputs.num_threads = int(self._num_threads)
        self._num_threads_update()

    def _num_threads_update(self):
        if self.inputs.num_threads:
            self.inputs.environ.update(
                {"OMP_NUM_THREADS": str(self.inputs.num_threads)}
            )

    def run(self, **inputs):
        if "num_threads" in inputs:
            self.inputs.num_threads = inputs["num_threads"]
        self._num_threads_update()
        return super().run(**inputs)


def no_freesurfer():
    """Checks if FreeSurfer is NOT installed
    used with skipif to skip tests that will
    fail if FreeSurfer is not installed"""

    return Info.version() is None
