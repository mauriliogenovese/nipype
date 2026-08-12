.. _containers:

================================
Running interfaces in containers
================================

Any :class:`~nipype.interfaces.base.core.CommandLine` interface can run inside a
container instead of natively on the host, without changing the interface
itself. Set its ``container`` input to a container wrapper carrying the engine
and image to use::

    from nipype.interfaces.fsl import BET
    from nipype.interfaces.base import DockerContainerWrapper

    bet = BET()
    bet.inputs.in_file = "sub-01_T1w.nii.gz"
    bet.inputs.out_file = "sub-01_brain.nii.gz"
    bet.inputs.container = DockerContainerWrapper("my-fsl-image:latest")
    bet.run()

Leave ``container`` unset to run natively (the default, unchanged behavior).

Requirements
============

* The container engine (Docker_) must be installed and on the ``PATH``.
* The image must run as **root** by default. Nipype never changes host file
  permissions to accommodate a non-root container, so a non-root image is
  rejected up front.

Both conditions are checked before execution, so misconfiguration fails fast
with a clear error.

Bind mounts
===========

Every ``File``/``Directory`` input is bind-mounted into the container
automatically -- you do not list volumes yourself. Inputs the process only
reads are mounted read-only; the working directory is mounted read-write so
outputs land back on the host.

Paths that live *inside* the image (e.g. a template shipped with the software)
are not host paths and must not be mounted or rewritten. Wrap them in a
:class:`~nipype.interfaces.base.traits_extension.ContainerPath`::

    from nipype.interfaces.base import ContainerPath

    flirt.inputs.reference = ContainerPath("$FSLDIR/data/standard/MNI152_T1_2mm.nii.gz")

A ``ContainerPath`` is passed through to the containerized command verbatim,
skips host existence checks, and is never bind-mounted.

GPU and CPUs
============

If the node requests a GPU (``inputs.use_gpu`` or ``inputs.use_cuda``), the
container is started with ``--gpus all``. A requested CPU count
(``inputs.num_threads``) is forwarded as ``--cpus``.

FreeSurfer license
==================

FreeSurfer interfaces need a license file inside the container. Point them at a
host license per node::

    from nipype.interfaces.freesurfer import ReconAll

    recon = ReconAll()
    recon.inputs.container = DockerContainerWrapper("my-freesurfer-image:latest")
    recon.inputs.license_file = "/path/to/license.txt"

or set one default for every FreeSurfer node (also honored by the ``FS_LICENSE``
environment variable)::

    from nipype.interfaces.freesurfer.base import FSLicenseMixin

    FSLicenseMixin.set_default_license_file("/path/to/license.txt")

Windows
=======

On Windows, host paths cannot be used as mount targets inside a POSIX
container. Nipype handles this transparently by mapping each mounted directory
to a canonical path inside the container -- no interface-specific changes are
needed.

Other engines
=============

Only Docker_ ships today. Support for another engine (Podman, Singularity_/
Apptainer, ...) is a matter of subclassing
:class:`~nipype.interfaces.base.containers.ContainerWrapper` and assigning an
instance of it to ``inputs.container``; nothing in the interfaces or the
execution plugins needs to change.

.. include:: ../links_names.txt
