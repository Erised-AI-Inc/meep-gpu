"""GPU acceleration layer for MEEP — it accelerates MEEP, it does not replace it.

**Read this before adding anything.** The audience is MEEP users first: someone
who has an ``mp.Simulation`` and wants its time-stepping to run on a GPU while
everything else about their workflow stays exactly as it is. Integration into
a host application comes second, on top of that.

MEEP therefore keeps doing everything MEEP already does well, and this package
implements only the per-timestep loop:

===========================================  ===========================================
MEEP keeps                                   this package provides
===========================================  ===========================================
geometry objects and their rasterization     curl updates, boundaries (UPML, Bloch)
subpixel smoothing of geometry               dispersive materials (Lorentz/Drude ADE)
``mp.Medium`` and the material library       instantaneous nonlinearity (chi2/chi3)
sources as ``mp.Source`` specifications      current injection into the field arrays
``mp.Harminv``, near-to-far evaluation       DFT and flux accumulation
mode solving (MPB), adjoint optimisation     mirror-symmetry folding of the stepped grid
===========================================  ===========================================

**The design test for anything proposed here: does it execute inside the
per-timestep loop, over the grid?** If yes it belongs here, because that loop is
what a GPU accelerates. If it runs once at setup, or on a handful of host
numbers afterwards, delegate it to MEEP — reimplementing what MEEP already does
adequately on the CPU adds surface, adds a second thing to keep correct, and buys
no speed. Harmonic inversion is the cautionary example: it costs under 1 % of a
run and is 1-D post-processing, so ``mp.Harminv`` was always the right answer.

"Standalone" in this package means free of **host-application** coupling — no
job, persistence, service or API modules, enforced by
``test_package_boundary.py``. It does **not** mean free of MEEP. MEEP is the
intended companion, and importing it is permitted; the numerical core simply
does not need it, so the modules stay importable and testable on a machine with
neither MEEP nor CuPy installed.

The package provides a Yee grid, D/B-primary fields, UPML, current sources, DFT
and flux monitors, dispersive and nonlinear materials, and the leapfrog step
sequence, in MEEP natural units.

Importing this package never requires CuPy or MEEP. GPU capability is probed
live through :func:`is_available`, :func:`available_gpu` and
:func:`missing_dependencies`: ``prefer_gpu=True`` runs on this host's GPU (CUDA
through CuPy, or Metal on an Apple GPU), and ``prefer_gpu=False`` is the NumPy
reference, which never consults a kernel table.
"""

from __future__ import annotations

from .absorber import AbsorberLayer  # One mp.Absorber face, for FdtdDriver.set_absorber.
from .backends import available_gpu, cupy_available, missing_dependencies, resolve_backend, to_numpy
from .driver import (
    FdtdCancelled,
    FdtdDivergence,
    FdtdDriver,
    FdtdNonlinearityOutOfRange,
)
# The MEEP entry point: lift a constructed mp.Simulation onto the stepper. MEEP is
# imported lazily INSIDE these functions, so this line stays importable on a machine
# that has never had MEEP installed — which every other module here also assumes.
from .from_meep import (
    GpuCompatibility,
    GpuRunResult,
    MeepSimulationNotLiftable,
    # The type of ``GpuRunResult.monitors`` and ``driver.migrated_monitors``.
    # Exported because it is not a plain dict and the difference is the point: it
    # OWNS the MEEP monitor objects, which is what stops a later run's monitor from
    # resolving against a collected earlier one's recycled address.
    MigratedMonitors,
    # Which of the three per-point susceptibility-sigma routes a structured
    # dispersive lift took, and what it measured. Exported because the routes are
    # not interchangeable — the reader is exact, the declared-geometry lookup is
    # exact to MEEP's float32 storage and carries a per-point check, the chi1inv
    # inversion is 4-6 significant digits — and a caller comparing two lifts has no
    # other way to see which one ran.
    StructuredSigmaLift,
    # Raised when a MEEP step function hosted by ``run_on_gpu`` asks the simulation
    # facade for something the driver has no counterpart for. Exported because it is
    # a refusal a caller can act on — host the step function on CPU MEEP, or drop it
    # — and because catching it must not mean catching AttributeError.
    StepFunctionNotHosted,
    gpu_compatibility,
    lift_simulation,
    run_on_gpu,
)
# Only the capitalized names and ``do_harminv`` are re-exported. Binding the
# lowercase ``harminv`` FUNCTION here would shadow the ``meep_gpu.harminv``
# MODULE on the package, so ``import meep_gpu.harminv as hv`` would hand back a
# function. MEEP's own surface is ``mp.Harminv`` and ``mp.py_do_harminv``; the
# convenience call stays reachable as ``meep_gpu.harminv.harminv``.
from .grid import Mirror  # MEEP's mp.Mirror(direction, phase): the symmetry argument's element.
from .harminv import Harminv, Mode, do_harminv
# The float32 subnormal policy: one flag, obeyed by the host FPU, CuPy and Triton
# alike. ``get_subnormal_policy`` and not ``subnormal_policy`` for the same reason
# spelled out for ``harminv`` above — the latter name would shadow the MODULE.
# Importing this costs nothing: the module imports only the standard library and
# ``backends``, never cupy, triton or meep.
from .subnormal_policy import (
    FLUSH,
    KEEP,
    MATCH_MEEP,
    SubnormalPolicyLocked,
    SubnormalPolicyUnattainable,
    default_policy,
    get_subnormal_policy,
    install_subnormal_policy,
    policy_stamp,
    resolve_match_meep,
    set_subnormal_policy,
)


def is_available() -> bool:  # Live GPU capability: prefer_gpu=True resolves on this host.
    return available_gpu() is not None


def _read_version() -> str:
    """The version of the distribution that installed this package.

    Read from the installed package metadata: the distribution is found by the
    import name, and it counts only when it installed THIS copy of the package, so
    a checkout on ``PYTHONPATH`` beside another installed copy does not borrow that
    copy's number. A checkout that is not installed that way (never installed, or
    installed in editable mode, whose metadata lists no package files) reads the
    ``version`` of the ``pyproject.toml`` beside the package. ``"unknown"`` when
    neither exists.
    """
    from importlib import metadata  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    here = Path(__file__).resolve()
    try:
        names = metadata.packages_distributions().get(__name__, [])
    except Exception:  # noqa: BLE001 - unreadable metadata is not a version
        names = []
    for name in names:
        try:
            distribution = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            continue
        for file in distribution.files or ():
            if file.name == "__init__.py" and file.parent.name == __name__ \
                    and Path(distribution.locate_file(file)).resolve() == here:
                return distribution.version
    project = here.parent.parent / "pyproject.toml"
    try:
        text = project.read_text(encoding="utf-8")
    except OSError:
        return "unknown"
    in_project = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("["):
            in_project = stripped == "[project]"
        elif in_project and stripped.startswith("version") and "=" in stripped:
            key, _, value = stripped.partition("=")
            if key.strip() == "version":
                return value.strip().strip("\"'")
    return "unknown"


def __getattr__(name: str):
    # ``__version__`` is read on first use, not at import: finding the distribution
    # walks the installed metadata, which ``import meep_gpu`` should not pay for.
    if name == "__version__":
        value = _read_version()
        globals()["__version__"] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AbsorberLayer",
    "FLUSH",
    "KEEP",
    "MATCH_MEEP",
    "FdtdCancelled",
    "FdtdDivergence",
    "FdtdDriver",
    "FdtdNonlinearityOutOfRange",
    "GpuCompatibility",
    "GpuRunResult",
    "Harminv",
    "MeepSimulationNotLiftable",
    "MigratedMonitors",
    "Mirror",
    "Mode",
    "StepFunctionNotHosted",
    "StructuredSigmaLift",
    "SubnormalPolicyLocked",
    "SubnormalPolicyUnattainable",
    "available_gpu",
    "cupy_available",
    "default_policy",
    "do_harminv",
    "get_subnormal_policy",
    "gpu_compatibility",
    "install_subnormal_policy",
    "is_available",
    "lift_simulation",
    "missing_dependencies",
    "policy_stamp",
    "resolve_backend",
    "resolve_match_meep",
    "run_on_gpu",
    "set_subnormal_policy",
    "to_numpy",
]
