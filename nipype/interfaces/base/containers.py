# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Container engine backends for :class:`~nipype.interfaces.base.core.CommandLine`.

This module isolates every container-engine-specific detail behind a single
abstract :class:`ContainerWrapper` interface, so the rest of nipype
(``CommandLine`` and the execution plugins) only ever talks to the generic
methods declared here -- never to ``docker`` directly.

Adding support for a new engine (podman, singularity/apptainer, ...) is a
matter of writing a new :class:`ContainerWrapper` subclass and assigning an
instance of it to a command's ``container`` input; no changes to ``core.py``
or the plugins are required.
"""

import os
import shlex
import platform
import subprocess as sp
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from ...utils.filemanip import which

__all__ = [
    "DockerContainerWrapper",
    "ContainerWrapper",
    "ContainerRunSpec",
]


@dataclass
class ContainerRunSpec:
    """Engine-agnostic description of a single containerized command.

    ``CommandLine`` gathers everything an interface needs to run inside a
    container into one of these, then hands it to a :class:`ContainerWrapper`,
    which turns it into an engine-specific command line. All fields are host
    concepts (real paths, the plain inner command line); translating them to
    the container's view (bind mounts, path remapping) is the wrapper's job.
    """

    #: Image reference to run (e.g. ``'my-image:latest'``).
    image: str
    #: Host working directory the command should run in.
    cwd: str
    #: The command line to execute *inside* the container.
    inner_cmdline: str
    #: Host directories to bind-mount (the wrapper adds ``cwd`` and collapses
    #: nested roots itself).
    mount_roots: set = field(default_factory=set)
    #: Environment variables to export into the container.
    environ: dict = field(default_factory=dict)
    #: Extra ``(host_path, container_path, mode)`` bind mounts, where ``mode``
    #: is ``''`` (read-write) or ``':ro'`` (read-only).
    extra_mounts: list = field(default_factory=list)
    #: Best-effort shell commands to run inside the container before the main
    #: command (failures are swallowed).
    prelude: list = field(default_factory=list)
    #: Whether the node requested GPU execution.
    use_gpu: bool = False
    #: Requested CPU count, or ``None`` to leave the engine default.
    num_threads: int = None


class ContainerWrapper(ABC):
    """Abstract backend for a container engine.

    Subclasses encapsulate all engine-specific behavior; ``CommandLine`` and
    the execution plugins interact only through the generic methods declared
    here -- chiefly :meth:`check_available` (before a run) and
    :meth:`build_command` (at run time). The shared, engine-agnostic mechanics
    (mount-plan collapsing, Windows path remapping, the overall command
    assembly and shell wrapping) live in this base class, so a concrete backend
    only fills in the small pieces that actually differ between engines:

    * ``container_type`` -- the engine's registered name / executable;
    * :meth:`check_available` -- install/image pre-flight check;
    * :meth:`_run_prefix`, :meth:`_mount_flag`, :meth:`_env_flag`,
      :meth:`_workdir_flag`, :meth:`_gpu_flags`, :meth:`_cpu_flags` -- how each
      command fragment is spelled for this engine;
    * optionally :meth:`cpu_limit` -- any CPU cap the engine adds on top of the
      host's (defaults to none).

    See :class:`DockerContainerWrapper` for a complete example.
    """

    #: Engine name, and by convention the name of its executable on ``PATH``
    #: (used to locate the engine and in resource-limit messages). Subclasses
    #: must set it.
    container_type = None

    def __init__(self, image):
        self.image = image

    # -- Engine-specific hooks (must be implemented by subclasses) ----------

    @abstractmethod
    def check_available(self, env=None):
        """Fail fast, before running, if the engine isn't installed or the
        image can't be used. Returns the resolved engine binary path."""

    @abstractmethod
    def _run_prefix(self):
        """Argv that starts an engine invocation, up to but excluding any
        flags (e.g. ``['docker', 'run', '--rm', '--init']``)."""

    @abstractmethod
    def _mount_flag(self, host_path, container_path, mode):
        """Argv that bind-mounts ``host_path`` at ``container_path`` (``mode``
        is ``''`` for read-write or ``':ro'`` for read-only)."""

    @abstractmethod
    def _env_flag(self, key, value):
        """Argv that exports the environment variable ``key=value``."""

    @abstractmethod
    def _workdir_flag(self, container_path):
        """Argv that sets the working directory inside the container."""

    @abstractmethod
    def _gpu_flags(self):
        """Argv that enables GPU access."""

    @abstractmethod
    def _cpu_flags(self, num_threads):
        """Argv that caps the container to ``num_threads`` CPUs."""

    # -- Optional engine hooks (sensible defaults, override as needed) ------

    def cpu_limit(self):
        """Effective CPU cap this engine imposes on a container, on top of the
        host's own CPU count (which the execution plugin already checks against
        ``n_procs``). Return ``None`` when the engine adds no cap of its own.

        This is the single place where an engine decides whether -- and under
        what conditions -- a containerized node can be starved of CPUs it would
        otherwise get natively, so callers (e.g. MultiProc's pre-run check) stay
        free of engine/platform special-casing. The default is ``None`` (no
        extra cap)."""
        return None

    # -- Generic command assembly (shared by all engines) ------------------

    def build_command(self, spec):
        """Turn a :class:`ContainerRunSpec` into an engine command line string.

        Linux/macOS: mirror mounts (host and container paths identical).
        Windows: host paths can't be used as mount targets, so each mount root
        is mapped to a canonical POSIX path and a single text substitution is
        applied over the fully-assembled inner command line.
        """
        cwd = Path(spec.cwd).resolve()
        path_map = self._build_path_map(spec.mount_roots, cwd) if _is_windows() else {}

        cmd = list(self._run_prefix())
        if spec.use_gpu:
            cmd += self._gpu_flags()
        if spec.num_threads is not None:
            cmd += self._cpu_flags(spec.num_threads)

        if path_map:
            container_cwd = self._container_path_for(cwd, path_map)
            if container_cwd is None:
                raise RuntimeError(
                    "Internal error: node working directory missing from "
                    "container path map."
                )
            cmd += self._workdir_flag(container_cwd)
            for host_root, container_root in path_map.items():
                cmd += self._mount_flag(
                    host_root, container_root, self._mount_mode(host_root)
                )
            inner_cmdline = self._remap_cmdline(spec.inner_cmdline, path_map)
        else:
            mount_roots = set(spec.mount_roots)
            mount_roots.add(cwd)
            mount_roots = self._collapse_mount_roots(mount_roots)
            cmd += self._workdir_flag(str(cwd))
            for root in mount_roots:
                cmd += self._mount_flag(root, root, self._mount_mode(root))
            inner_cmdline = spec.inner_cmdline

        for host_path, container_path, mode in spec.extra_mounts:
            cmd += self._mount_flag(host_path, container_path, mode)

        for key, val in spec.environ.items():
            cmd += self._env_flag(key, val)

        cmd.append(spec.image)
        cmd += self._shell_invocation(spec.prelude, inner_cmdline)

        return self._render(cmd)

    @staticmethod
    def _mount_mode(host_path):
        """Bind-mount mode for ``host_path``: read-write if the host grants
        write access, otherwise read-only (``':ro'``)."""
        return "" if os.access(host_path, os.W_OK) else ":ro"

    @staticmethod
    def _shell_invocation(prelude, inner_cmdline):
        """Trailing ``sh -c ...`` that runs, inside the container, an optional
        best-effort prelude followed by the main command, with a permissive
        umask so files created land world-accessible on the (root-owned) host
        mounts. POSIX-shell semantics, shared by every OCI-style engine."""
        prelude_parts = [f"{{ {c} ; }} 2>/dev/null" for c in prelude]
        prelude_prefix = "; ".join(prelude_parts)
        prelude_prefix = f"{prelude_prefix}; " if prelude_prefix else ""
        return ["sh", "-c", f"umask 0000; {prelude_prefix}{inner_cmdline}"]

    @staticmethod
    def _render(cmd):
        """Join the assembled argv into a single command string for the host
        shell that will launch the engine."""
        if _is_windows():
            return sp.list2cmdline(cmd)
        return " ".join(shlex.quote(part) for part in cmd)

    # -- Generic path mechanics (shared by all engines) --------------------

    @staticmethod
    def _collapse_mount_roots(paths):
        """Drop any path already covered by another in the set (keep only the
        outermost root of each nested chain), to avoid redundant or overlapping
        bind mounts."""
        ordered = sorted(paths, key=lambda p: len(p.parts))
        kept = []
        for p in ordered:
            if not any(p == k or k in p.parents for k in kept):
                kept.append(p)
        return kept

    @classmethod
    def _build_path_map(cls, mount_roots, cwd):
        """Windows-only: build a host -> canonical container path mapping.

        Windows host paths (``C:\\...``) can't be used as bind-mount targets
        inside a POSIX container, unlike Linux/macOS where mirror mounts work
        directly. Each mount root gets a stable canonical POSIX path inside the
        container (``/mnt/nipype_volN``), assigned deterministically so
        repeated runs of the same node produce the same map.
        """
        roots = set(mount_roots)
        roots.add(cwd)
        roots = cls._collapse_mount_roots(roots)
        return {
            root: f"/mnt/nipype_vol{i}" for i, root in enumerate(sorted(roots, key=str))
        }

    @staticmethod
    def _container_path_for(host_path, path_map):
        """Find which mount root (from ``path_map``) contains ``host_path`` and
        return the corresponding container path. Needed because
        :meth:`_collapse_mount_roots` keeps only the outermost root, so a nested
        path may not be a literal key in ``path_map`` even though it's covered."""
        for host_root, container_root in path_map.items():
            try:
                rel = host_path.relative_to(host_root)
            except ValueError:
                continue
            rel_posix = rel.as_posix()
            return (
                container_root if rel_posix == "." else f"{container_root}/{rel_posix}"
            )
        return None

    @staticmethod
    def _remap_cmdline(cmdline, path_map):
        """Rewrite a fully-assembled (host-native) cmdline string for Windows:
        replace each mount root's literal host path with its canonical container
        path, then flip remaining OS-native separators to POSIX. Applied once,
        after all interface-specific formatting is done."""
        for host_root, container_root in sorted(
            path_map.items(), key=lambda kv: -len(str(kv[0]))
        ):
            cmdline = cmdline.replace(str(host_root), container_root)
        return cmdline.replace("\\", "/")


def _is_windows():
    return platform.system() == "Windows"


class DockerContainerWrapper(ContainerWrapper):
    """Docker backend: runs commands via ``docker run``."""

    container_type = "docker"

    def check_available(self, env=None):
        """Fail fast if docker isn't installed, or the image can't be
        found/inspected, or it doesn't run as root by default.

        Root inside the container is a hard requirement: running as non-root
        would require chmod/chown on host-owned files to make them accessible
        inside the container, which this implementation deliberately never does
        (see ``CommandLine._containerize_cmdline``). This is verified upfront
        from image metadata only -- no container is started here.
        """
        engine_path = which(self.container_type, env=env or os.environ)
        if engine_path is None:
            raise OSError(
                f'Container engine "{self.container_type}" not found on host. '
                "Please install it to run containerized commands."
            )

        result = sp.run(
            [
                self.container_type,
                "inspect",
                "--format",
                "{{.Config.User}}",
                self.image,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Container image '{self.image}' could not be inspected via "
                f"'{self.container_type} inspect' (is it pulled/built locally?). "
                f"Error:\n{result.stderr.strip()}"
            )

        user = result.stdout.strip()
        if user not in ("", "0", "root", "0:0"):
            raise RuntimeError(
                f"Image '{self.image}' does not run as root by default "
                f"(default user: '{user}'). This interface requires "
                "container images whose default user is root, since file "
                "permissions on the host are never modified to accommodate "
                "non-root containers."
            )

        return engine_path

    def cpu_limit(self):
        """Docker Desktop on Windows runs containers inside a VM with its own,
        separately configured CPU allocation, which may be lower than the
        host's -- so a node's ``n_procs`` can exceed what the container can
        actually use. Report that VM cap here. On Linux/macOS docker uses the
        host CPUs directly (no extra cap beyond the plugin's native check), so
        return ``None``."""
        if not _is_windows():
            return None
        return self._vm_cpus()

    def _vm_cpus(self):
        """Number of CPUs Docker reports for its VM, or ``None`` if it can't be
        determined."""
        result = sp.run(
            [self.container_type, "info", "--format", "{{.NCPU}}"],
            capture_output=True,
            text=True,
            check=False,
        )
        try:
            return int(result.stdout.strip())
        except (ValueError, AttributeError):
            return None

    def _run_prefix(self):
        return [self.container_type, "run", "--rm", "--init"]

    def _mount_flag(self, host_path, container_path, mode):
        return ["-v", f"{host_path}:{container_path}{mode}"]

    def _env_flag(self, key, value):
        return ["-e", f"{key}={value}"]

    def _workdir_flag(self, container_path):
        return ["-w", container_path]

    def _gpu_flags(self):
        return ["--gpus", "all"]

    def _cpu_flags(self, num_threads):
        return ["--cpus", str(num_threads)]
