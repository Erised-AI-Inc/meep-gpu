#!/usr/bin/env python3
"""THE DRIVER-ROUTE GATE FOR THE METAL TABLE: fused Metal pairs, through the real seam.

THE TORCH-SIDE SIBLING OF ``gate_dispatch_fused_route.py``, and the measurement the
Metal release rows are cut from. ``meep_gpu.metal_dispatch`` is the second kernel
table's ladder: ``fastpath._decide``'s backend rung routes a NumPy engine with an
MPS device into it, and every rung below is that module's. What licenses a fused
Metal product to reach ``FastPathPlan.dispatch`` is
``metal_dispatch.METAL_RELEASED_FUSED_ARMS`` — a TYPED table naming which arm was
driven on which case — and this gate is what makes those rows a measurement.

"DRIVEN THROUGH THE DRIVER SEAM" MEANS THE SEVEN REAL CONSULTS, exactly as it does
on the Triton track. ``driver.MeepGPUDriver.step`` makes five (``step_B``,
``update_H``, ``step_D``, ``update_E``, ``update_P``) plus the two fill sites, and
``synchronize_magnetic_fields`` makes two more off a plan it deliberately does not
re-freeze. Between them the driver runs work behind NO consult — source withdraw
and inject, ``fill_symmetry_bc_*``, ``zero_metal_*``, ``fill_folded_far_ghosts_*``.
A fused pair takes TWO of those slots: the launch happens at the leading consult and
the absorbed consult answers off a sentinel (``launch.NoopPlan``) or off
``deposit_repair.TrailingRepairPlan``.

WHAT THIS TABLE ADDS THAT THE TRITON ONE DOES NOT HAVE: A RESIDENCY SEAM. Metal
kernels read and write DEVICE mirrors of the engine's host arrays, so between two
launches the host must be authoritative — the driver's unconsulted passes all run
on host arrays. ``metal_kernels.launch.wrap_for_residency`` installs a
``sync_in``/``sync_out`` bracket around every device launch AT PLAN TIME, which is
why ``FastPathPlan.dispatch`` needs no residency branch. That bracket is a
correctness surface no byte comparison can see the absence of on its own — a
missing sync is a stale mirror, and a stale mirror is a wrong answer that looks
like an arithmetic defect — so this gate arms it three ways.

THE PROTOCOL, per case, in ONE process on ONE MPS device. Three legs, all lifted
from identical ``mp.Simulation`` declarations by ``lift_simulation(prefer_gpu=
True)`` -- an Apple GPU driver: NumPy host arrays, ``driver.gpu == "metal"``, checked
after every lift -- and byte-equal before a step is taken, plus armed controls. (Until
2026-09-27 the legs were lifted ``prefer_gpu=False`` under the enable; such a driver is
now the NumPy reference and never plans, so it could not be the fused leg.)::

    fused      MEEP_GPU_DISPATCH=1  (the release admits, or FUSE_ARMS names)  the claim
    unfused    MEEP_GPU_DISPATCH=1  MEEP_GPU_FUSE_ARMS=0            the substitution baseline
    array      MEEP_GPU_DISPATCH=1  MEEP_GPU_FUSED=0                the oracle

FOUR INDEPENDENT ANSWERS TO "DID A KERNEL ACTUALLY RUN", because a leg that quietly
fell back matches trivially — the oracle IS the array path:

1. ``driver.active_step_path == "fused"`` and the record's ``fusion.driven`` names a
   fused arm at BOTH slots of every pair;
2. ``fast_path_report()["launch_counters"]`` shows a nonzero dispatch count on the
   leading slot AND on the absorbed slot. ``programs_per_dispatch`` IS EXPLICITLY
   NOT THE WITNESS HERE and that is a difference from the Triton gate rather than
   an omission: a Metal ``KernelPlan`` spells no launch grid (``plans.py``'s
   ``__slots__`` is ``_args``/``_functions``/``launches``), so the counter reports
   ``None`` by construction and asserting on it would fail every leg. The per-plan
   substitute is ``plan_launches``;
3. the PLANS' OWN ``KernelPlan.launches``, summed through ``launch.declaring_plan``,
   PLUS ``CountingFunction`` wrappers on every compiled callable in the plan's
   ``_functions`` — the package's own mutation seam
   (``gate_metal_fused_hd_pair.py``'s ``count_functions``). Two counts of one event.
   STATED WEAKNESS, honestly: ``torch.mps.compile_shader`` exposes no disassembly
   and there is no package-external MPS launch counter, so unlike Triton's PTX read
   BOTH of these are in-package. ``metal_kernels.subnormal.mps_policy_report``
   already records ``ptx_equivalent_audit: None`` for exactly this reason;
4. THE SUBSTITUTION PROOF: compiled-function calls per complete step must DROP
   against the ``unfused`` leg by exactly one per fused pair per step, counted twice
   over, on IDENTICAL slot sets, while the two legs stay byte-identical.

THE SUBNORMAL POLICY IS THE OTHER HALF OF THE CLAIM. MPS flushes float32 denormals
natively and exposes no lever, so ``flush`` is the only attainable device policy;
this arm64 host's own resolution is ``keep`` (MEEP's ``set_zero_subnormals`` is the
``#if HAVE_IMMINTRIN_H`` no-op here) and ``subnormal_policy`` drives it to flush
through ``fesetenv(_FE_DFL_DISABLE_DENORMS_ENV)``. That is one policy on both
executors, and the ``harness_keep`` leg is what shows the seam refusing rather than
silently splitting: under an installed ``keep`` every dispatch case must be refused
BY NAME at rung 8bM and stay byte-identical to the array path.

Rule 7: one flushed line per leg per chunk, and every row is appended to
``cases.jsonl`` / ``controls.jsonl`` as it lands.

Run (this Mac, ONE MPS device, through the runner that welds the artifact)::

    PYTHONPATH=. python -u \\
        parity/meep_gpu/metal_gate_runner.py \\
        parity/meep_gpu/gate_dispatch_metal_route.py -- \\
        --out parity/meep_gpu/results/dispatch_metal_route_<stamp>/shipped

THE NAME IS LOAD-BEARING AND IS DELIBERATELY OUTSIDE THE ``gate_metal_*`` GLOB.
Three separate mechanisms derive meaning from that stem: ``recut_metal_gates.sh``'s
fleet loop would launch this four-leg campaign with the wrong arguments and count it
as a failed gate; ``test_metal_weld_contract``'s family derivation would invent a
family ``dispatch_metal_route`` and demand a per-family weld for it; and
``test_gate_provenance``'s pattern would sweep it too. The cost of staying outside
is one explicit allowlist entry in ``metal_gate_runner.discover_dispatch_gates``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy
from typing import Any, Dict, List, Optional, Sequence, Tuple

# RESOLVED BY NAME, NEVER BY DEPTH. A ``parents[N]`` walk is correct only while this
# file stays at the depth it was written at; the harness traps this campaign has
# already paid for include exactly that failure in a moved file.
HERE = os.path.dirname(os.path.abspath(__file__))
# The root is the nearest ancestor that holds both the package and the harness
# (``meep_gpu/`` and ``parity/meep_gpu/``): the repository root of this layout,
# and the same directory in any tree the harness is copied into whole.
_API = HERE
while _API != os.path.dirname(_API) and not (
        os.path.isdir(os.path.join(_API, "meep_gpu"))
        and os.path.isdir(os.path.join(_API, "parity", "meep_gpu"))):
    _API = os.path.dirname(_API)
if not os.path.isdir(os.path.join(_API, "parity", "meep_gpu")):  # pragma: no cover
    raise SystemExit(f"cannot locate the repository root above {HERE}: no ancestor "
                     f"holds both meep_gpu/ and parity/meep_gpu/")
for path in (HERE, _API):
    if path not in sys.path:
        sys.path.insert(0, path)

import gate_dispatch_end_to_end as e2e  # noqa: E402  the byte comparator and the cases
import gate_dispatch_fused_route as route  # noqa: E402  the extra cases and the controls
import gate_provenance  # noqa: E402


def say(message: str) -> None:
    e2e.say(message)


# ---------------------------------------------------------------------------
# The launch counter — the Metal analogue of TritonLaunchCounter
# ---------------------------------------------------------------------------

class CountingFunction:
    """An independent launch counter: wraps a compiled function and counts calls.

    Lifted from ``gate_metal_fused_hd_pair.CountingFunction``, which is the seam
    every Metal weld gate's mutation leg already uses. It is a SECOND count of the
    same event as ``KernelPlan.launches``, taken one level lower: the plan books a
    ``run``, this books the compiled callable actually being invoked.
    """

    __slots__ = ("function", "calls")

    def __init__(self, function: Any) -> None:
        self.function = function
        self.calls = 0

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        return self.function(*args, **kwargs)


def _wrap_functions(node: Any, sink: List[CountingFunction]) -> Any:
    """Wrap every compiled callable in a plan's function table, SHAPE PRESERVED.

    TWO TABLE SHAPES, AND ONLY ONE USED TO BE HANDLED. Most plans hold a flat
    ``{mode: function}``; the symmetry plans hold ``{mode: {key: function}}`` and
    index the inner table by the entry's key at launch time (the ``_walk`` method
    of ``metal_kernels/symmetry.py``). Replacing that inner MAPPING with a single
    wrapper left the plan unsubscriptable, so every folded case raised
    ``TypeError: 'CountingFunction' object is not subscriptable`` inside
    ``driver.step`` -- which this gate recorded as a case ERROR, reading as the
    PRODUCT failing when what failed was the instrument measuring it. Recursing
    keeps the mapping shape and still counts each compiled callable exactly once.
    """
    if isinstance(node, dict):
        return {key: _wrap_functions(value, sink) for key, value in node.items()}
    if callable(node):
        wrapper = CountingFunction(node)
        sink.append(wrapper)
        return wrapper
    return node


#: The one leaf type a Metal function table holds: ``device.compile_source`` is a thin
#: wrapper over ``torch.mps.compile_shader``, whose functions are this type. A table
#: leaf of any other type is refused BY NAME rather than counted, so a host helper
#: kept under a ``*functions`` name can never pass for a compiled launch.
_COMPILED_LEAF_TYPE = "_mps_MetalKernel"


def _function_table_names(owner: Any) -> List[str]:
    """Every attribute of ``owner`` that holds a function table, read off the owner.

    THE TABLE IS NOT ALWAYS CALLED ``_functions``, and reading only that name was a
    witness defect for a year of campaigns. ``plans.KernelPlan`` keeps its callables
    in ``_functions``; the real-cylindrical plans are deliberately NOT KernelPlan
    subclasses and keep theirs in ``_prefix_functions``/``_fused_functions``
    (``cylindrical_real_fused_{magnetic,electric}_pair.py``), ``_prefix_functions``/
    ``_curl_functions`` (``cylindrical_real.py``), ``_constitutive_functions``/
    ``_curl_functions`` (``cylindrical_real_fused_hd_pair.py``) and ``_lead_functions``
    (``cylindrical_complex_fused_hd_pair.py``). Measured 2026-09-18: the
    ``cylindrical`` fused leg launched 6396 compiled kernels in steady state and the
    witness saw 0, so the case read VACUOUS-PASS on a product that ran (a mutation of
    its ``_fused_functions`` to a no-op broke byte identity on 12 arrays).

    Walked over the owner's ``__slots__`` along its MRO plus its ``__dict__``, and
    NEVER over slot occupants: ``SyncedPlan`` forwards ``_functions`` through
    ``__getattr__`` (``metal_kernels/launch.py``), so walking it would wrap the same
    table twice.
    """
    names: List[str] = []
    for klass in type(owner).__mro__:
        declared = getattr(klass, "__slots__", ()) or ()
        if isinstance(declared, str):
            declared = (declared,)
        for name in declared:
            if isinstance(name, str) and name.endswith("functions") and name not in names:
                names.append(name)
    for name in (getattr(owner, "__dict__", None) or {}):
        if name.endswith("functions") and name not in names:
            names.append(name)
    return [name for name in names if isinstance(getattr(owner, name, None), dict)]


def _leaf_types(node: Any, sink: List[str]) -> None:
    """The type name of every callable leaf under a (possibly nested) function table."""
    if isinstance(node, dict):
        for value in node.values():
            _leaf_types(value, sink)
    elif isinstance(node, CountingFunction):
        sink.append(type(node.function).__name__)
    elif callable(node):
        sink.append(type(node).__name__)


class MetalLaunchCounter:
    """Two counts of every Metal launch, both taken off the plans a leg runs.

    THE HONEST DIFFERENCE FROM THE TRITON GATE, stated rather than papered over.
    Triton's counter patches ``JITFunction.run`` and ``CompiledKernel.launch_enter
    _hook`` — entry points OUTSIDE anything this package owns — so no defect in the
    dispatcher can make it agree by construction. There is no such external counter
    for ``torch.mps.compile_shader``: it hands back a callable and exposes neither a
    launch hook nor a disassembly. So both counts here are in-package, and what
    separates them is the LEVEL rather than the owner: ``KernelPlan.launches`` is
    incremented by the plan's own ``run``, and :class:`CountingFunction` wraps the
    compiled callable the plan then invokes. A plan that books a launch without
    invoking anything disagrees with its functions; a shim that reports a dispatch
    it never made disagrees with both.

    ONE OWNER IS WRAPPED ONCE. A fused pair occupies BOTH slots of its seam, so
    walking slots reaches the same declaring plan twice and the second wrapper would
    count the first — measured on the sibling gate as exactly twice the launches the
    plans report, which reads as the two witnesses disagreeing.
    """

    def __init__(self) -> None:
        self._owners: List[Any] = []
        self._functions: List[CountingFunction] = []
        self.attached: Dict[str, int] = {}

    def attach(self, label: str, plan_holder: Any) -> Dict[str, Any]:
        """Wrap every compiled function of every plan this leg will run. Idempotent per owner.

        RETURNS WHAT IT COULD NOT SEE, per call, rather than storing it on the
        counter: one counter serves the whole run and leg labels repeat for every
        case, so a list kept here would either never reach a verdict or stick to
        every later case. The caller stores the answer on the LEG.

        An owner is counted as attached only when at least one of its tables was
        wrapped. Counting it regardless is how the witness overstated its own
        coverage while seeing nothing of the cylindrical plans.
        """
        from meep_gpu.metal_kernels import launch  # noqa: PLC0415

        plans = getattr(plan_holder, "plans", None)
        if plans is None:
            step_plan = getattr(plan_holder, "step_plan", None)
            plans = getattr(step_plan, "plans", {}) or {}
        wrapped = 0
        unwitnessed: List[str] = []
        foreign: List[str] = []
        seen = {id(owner) for owner in self._owners}
        for plan in list(plans.values()):
            owner = launch.declaring_plan(plan)
            if owner is None or id(owner) in seen:
                continue
            seen.add(id(owner))
            self._owners.append(owner)
            tables = _function_table_names(owner)
            for name in tables:
                leaves: List[str] = []
                _leaf_types(getattr(owner, name), leaves)
                foreign.extend(f"{type(owner).__name__}.{name}: {leaf}"
                               for leaf in leaves if leaf != _COMPILED_LEAF_TYPE)
                # noqa: SLF001 below - the package's mutation seam, wrapped in place
                setattr(owner, name, _wrap_functions(getattr(owner, name),
                                                     self._functions))
            if tables:
                wrapped += 1
            elif (hasattr(owner, "launches")
                  and getattr(owner, "performs_device_work", True)):
                # ``performs_device_work`` is read for truthiness exactly as
                # ``launch.synced`` reads it: the no-PML null arm declares False and
                # launches nothing, so it has nothing to witness.
                unwitnessed.append(type(owner).__name__)
        self.attached[label] = self.attached.get(label, 0) + wrapped
        return {"wrapped": wrapped, "unwitnessed_owners": unwitnessed,
                "non_kernel_leaves": foreign}

    def snapshot(self) -> Dict[str, Any]:
        return {"total": sum(int(getattr(owner, "launches", 0) or 0)
                             for owner in self._owners),
                "function_calls": sum(counter.calls for counter in self._functions),
                "owners": len(self._owners)}

    @staticmethod
    def delta(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
        return {"total": after["total"] - before["total"],
                "function_calls": after["function_calls"] - before["function_calls"]}


# ---------------------------------------------------------------------------
# The cases
# ---------------------------------------------------------------------------

#: The two dispatch gates' own builders, imported rather than re-declared. A
#: route-gate case is a NAME the release binds, and this gate's cases are the ones
#: ``metal_dispatch.METAL_RELEASED_FUSED_ARMS`` cites by that name.
CASES: Dict[str, Any] = dict(route.CASES)


def case_offdiag_2d(mp, res=20):
    """``pml_2d`` AND an off-diagonal ``chi1inv``, differing in that ONE axis.

    THE SHAPE THE RETIRED ENVELOPE ROW NAMED. When ``off_diagonal_epsilon`` left the
    Metal shared envelope (2026-09-12) the note on the per-arm bound (now
    ``metal_dispatch.METAL_OFF_DIAGONAL_CORNERS``) said what would widen the
    ordinary magnetic pair past 3-D: "a 2-D off-diagonal case with an electric
    source outside the D seam". This is that case, and it is ``case_pml_2d`` with
    its axis-aligned slab replaced by a CYLINDER — the same cell, the same PML, the
    same Ez source and the same two flux planes — because a curved surface is what
    makes MEEP's subpixel averaging write off-diagonal ``chi1inv`` rows (the
    mechanism the end-to-end gate's 3-D sphere already relies on), and every other
    axis staying identical is what makes the single-axis isolation a measurement
    rather than an inference. Measured on the lift 2026-09-12: the two run shapes
    differ in ``off_diagonal_epsilon`` alone, same ``grid_shape`` (200, 120, 1).

    WHY IT LIVES HERE AND NOT IN ``gate_dispatch_fused_route``. That module's bytes
    are bound by the CUDA driver record (``cuda_kernels/fingerprints.json:
    driver_dispatch`` digests ``parity/meep_gpu/gate_dispatch_fused_route.py``), so
    a builder added there costs a CUDA campaign to re-cut; this module is bound by
    the Metal driver record alone, which the Metal campaign re-cuts anyway. The
    Triton round that wants the same shapes (dispatch expansion plan §8) re-gates
    everything and can move them then.

    WHAT IT IS FOR: on Metal the ordinary magnetic pair is admitted off-diagonal
    only inside the corner its own off-diagonal cases sat at, and this case is the
    2-D point of it. The D seam keeps the ``offdiag`` singles here exactly as
    ``pml_3d`` does, so it is a ONE-seam case.
    """
    cell = mp.Vector3(10, 6, 0)
    geometry = [mp.Cylinder(radius=0.8, center=mp.Vector3(),
                            material=mp.Medium(epsilon=12))]
    sources = [mp.Source(e2e._gaussian(mp, 0.25, 0.3), component=mp.Ez,  # noqa: SLF001
                         center=mp.Vector3(-3.5, 0))]
    sim = mp.Simulation(cell_size=cell, boundary_layers=[mp.PML(1.0)],
                        geometry=geometry, sources=sources, resolution=res)
    incident = sim.add_flux(0.25, 0.3, 5, mp.FluxRegion(
        center=mp.Vector3(-2.5, 0), size=mp.Vector3(0, 4)))
    transmitted = sim.add_flux(0.25, 0.3, 5, mp.FluxRegion(
        center=mp.Vector3(3.0, 0), size=mp.Vector3(0, 4)))
    return sim, [incident, transmitted], 60.0


def case_folded_offdiag_2d(mp, res=20):
    """``folded_2d`` AND an off-diagonal ``chi1inv``, differing in that ONE axis.

    ``case_folded_2d`` with its slab replaced by a CYLINDER CENTRED ON THE FOLD
    PLANE: the same cell, PML, mirror, Ez source and flux plane. Centring the
    cylinder on the plane keeps the mirror symmetry exact, and its curved surface
    is what writes the off-diagonal ``chi1inv`` rows (subpixel averaging, as for
    :func:`case_offdiag_2d` above). Measured on the lift against ``folded_2d``
    (2026-09-12): identical ``folded`` ("mirror plane on Y"), ``dimensions`` 2,
    ``pml_active`` True, ``susceptibilities`` 0, ``conductivity`` False,
    ``complex_storage`` False, ``bloch`` False, ``beta`` 0 and the same
    ``grid_shape`` (160, 61, 1); only ``off_diagonal_epsilon`` differs, False ->
    True. Same home as :func:`case_offdiag_2d`, for the same reason.

    WHAT IT IS FOR: the folded magnetic pair on a folded off-diagonal grid — the
    largest single block of served-but-not-dispatched instances on the Metal
    board once the unfolded 2-D block beside it is closed. The folded ELECTRIC
    pair's refusal beside it is half of what the case measures: no D-seam product
    is admitted on an off-diagonal grid, so this is a ONE-seam case.
    """
    cell = mp.Vector3(8, 6, 0)
    geometry = [mp.Cylinder(radius=0.8, center=mp.Vector3(),
                            material=mp.Medium(epsilon=12))]
    sources = [mp.Source(e2e._gaussian(mp, 0.25, 0.3), component=mp.Ez,  # noqa: SLF001
                         center=mp.Vector3(-2.5, 0))]
    sim = mp.Simulation(cell_size=cell, boundary_layers=[mp.PML(1.0)],
                        geometry=geometry, sources=sources, resolution=res,
                        symmetries=[mp.Mirror(mp.Y)])
    incident = sim.add_flux(0.25, 0.3, 5, mp.FluxRegion(
        center=mp.Vector3(2.0, 0), size=mp.Vector3(0, 4)))
    return sim, [incident], 50.0


def case_folded_3d(mp, res=10):
    """``pml_3d_diagonal`` with a mirror plane on Y: the folded pairs in 3-D.

    An axis-aligned block, so ``chi1inv`` stays diagonal and the case is about ONE
    axis of the folded pairs' rows: ``dimensions``. Measured on the lift 2026-09-12:
    ``folded`` "mirror plane on Y", ``dimensions`` 3, ``grid_shape`` (40, 21, 40),
    everything else as ``folded_2d``. The composer selects both folded pairs with
    the mirror fill in both fill slots (unfused 6, fused 4 launches per step) and
    the run is byte-identical to the array path; the same was measured with TWO
    planes (Y, and Z at phase -1), (40, 21, 21), 8 -> 6, which is the fold the five
    served 3-D rows carry (mie_scattering, the two LoadDump 3-D tests, LDOS 3-D,
    grating_3d). One plane is driven here because the fill's plane count is not an
    axis the release reads — ``folded`` is "required" and names no count — and the
    2-D rows served today already span one- and two-plane folds on one-plane
    evidence.
    """
    cell = mp.Vector3(4, 4, 4)
    geometry = [mp.Block(mp.Vector3(mp.inf, 1.0, mp.inf), center=mp.Vector3(),
                         material=mp.Medium(epsilon=9))]
    sources = [mp.Source(e2e._gaussian(mp, 0.4, 0.4), component=mp.Ez,  # noqa: SLF001
                         center=mp.Vector3(-1.2, 0, 0))]
    sim = mp.Simulation(cell_size=cell, boundary_layers=[mp.PML(0.8)],
                        geometry=geometry, sources=sources, resolution=res,
                        symmetries=[mp.Mirror(mp.Y)])
    incident = sim.add_flux(0.4, 0.4, 3, mp.FluxRegion(
        center=mp.Vector3(0.9, 0, 0), size=mp.Vector3(0, 2, 2)))
    return sim, [incident], 35.0


def case_complex_nobloch_2d(mp, res=20):
    """``pml_2d`` under ``force_complex_fields`` with NO k_point: complex, bloch False.

    The shape of ``wvg-src.py`` and ``TestPhysical``/``TestWvgSrc``: complex storage
    asked for directly rather than implied by a k_point. Measured on the lift
    2026-09-12: identical to ``bloch_2d``'s run shape except ``bloch`` False and
    ``k_point`` zero, ``grid_shape`` (200, 120, 1). The composer selects BOTH
    complex pairs when offered (unfused 4, fused 2), byte-identical — so this is
    the case that lets the complex pairs' rows drop their ``bloch`` row: both
    values driven, every other axis of the row single-valued. It needs the complex
    expansion licence exactly as ``bloch_2d`` does.
    """
    cell = mp.Vector3(10, 6, 0)
    geometry = [mp.Block(mp.Vector3(mp.inf, 1.0, mp.inf), center=mp.Vector3(),
                         material=mp.Medium(epsilon=12))]
    sources = [mp.Source(e2e._gaussian(mp, 0.25, 0.3), component=mp.Ez,  # noqa: SLF001
                         center=mp.Vector3(-3.5, 0))]
    sim = mp.Simulation(cell_size=cell, boundary_layers=[mp.PML(1.0)],
                        geometry=geometry, sources=sources, resolution=res,
                        force_complex_fields=True)
    incident = sim.add_flux(0.25, 0.3, 5, mp.FluxRegion(
        center=mp.Vector3(-2.5, 0), size=mp.Vector3(0, 4)))
    return sim, [incident], 60.0


def case_cylindrical_m1(mp, res=20):
    """Dcyl at m = 1 with a flux monitor: the cylindrical COMPLEX electric pair.

    ``case_cylindrical_m0``'s cell, PML, resolution and monitor with ``m=1`` and an
    Er source off-axis (an Ez source at m = 1 is not a mode of the axis). Measured on
    the lift 2026-09-12: ``cylindrical`` True, ``m`` 1, ``complex_storage`` True,
    ``bloch`` False, ``grid_shape`` (80, 1, 80). The composer selects the
    cylindrical complex fused electric D/E pair on the D seam (unfused 4, fused 3),
    byte-identical to the array path. It needs the cylindrical-complex expansion
    licence, so it dispatches on the probe leg.

    THE B SEAM CHANGED ON 2026-09-17 and this docstring said the opposite until then:
    it read that the composer "keeps the ``cylindrical complex`` singles on the B
    seam — that family's magnetic pair carries no absorb row". It has one now, and
    the 2026-09-17 dry run measured the consequence on every Dcyl complex case
    (m = 0, 1, 2, 3 and -1): the cylindrical complex fused MAGNETIC B/H pair takes
    ``step_B``/``update_H`` beside its electric twin, so both seams fuse. The arm was
    certified and welded the whole time; what it lacked was the absorb row.
    """
    cell = mp.Vector3(4, 0, 4)
    sources = [mp.Source(mp.GaussianSource(frequency=0.3, fwidth=0.3),
                         component=mp.Er, center=mp.Vector3(0.6, 0, 0))]
    sim = mp.Simulation(cell_size=cell, boundary_layers=[mp.PML(0.8)],
                        sources=sources, resolution=res,
                        dimensions=mp.CYLINDRICAL, m=1)
    monitor = sim.add_flux(0.3, 0.3, 5, mp.FluxRegion(
        center=mp.Vector3(1.5, 0, 0), size=mp.Vector3(0, 0, 2)))
    return sim, [monitor], 40.0


def case_cylindrical_m0_complex(mp, res=20):
    """Dcyl at m = 0 under ``force_complex_fields``: the cylindrical COMPLEX electric pair at m = 0.

    THE CORNER THE 2026-09-12 VERIFIER NAMED: the cylindrical complex electric D/E
    pair's row pins ``complex_storage`` True and cannot pin ``m`` (the census records
    none), so it admits the ``dipole_in_vacuum_cyl_off_axis`` shape -- m = 0 with
    complex storage forced -- on evidence taken at m = 1 alone. ``case_cylindrical``'s
    cell, PML, resolution and Ez source at r = 0.6 with ``force_complex_fields=True``
    and a flux monitor; every other axis as ``cylindrical_m1``.
    """
    cell = mp.Vector3(4, 0, 4)
    sources = [mp.Source(mp.GaussianSource(frequency=0.3, fwidth=0.3),
                         component=mp.Ez, center=mp.Vector3(0.6, 0, 0))]
    sim = mp.Simulation(cell_size=cell, boundary_layers=[mp.PML(0.8)],
                        sources=sources, resolution=res,
                        dimensions=mp.CYLINDRICAL, m=0, force_complex_fields=True)
    monitor = sim.add_flux(0.3, 0.3, 5, mp.FluxRegion(
        center=mp.Vector3(1.5, 0, 0), size=mp.Vector3(0, 0, 2)))
    return sim, [monitor], 40.0


def case_folded_dispersive_3d(mp, res=10):
    """``folded_3d`` with its block made Lorentzian: the folded B/H pair at 3-D over
    a susceptibility.

    THE CASE THAT KEEPS AN OMITTED AXIS HONEST. ``folded fused B/H pair`` carries no
    ``susceptibilities`` row because ``folded_2d`` and ``folded_dispersive_2d`` drive
    both values — but once ``folded_3d`` opened its ``dimensions`` row to {2, 3},
    both values had been driven only at 2-D, and (folded, 3-D, one pole) would have
    been admitted on nothing. This is exactly the precondition the row-table's note
    states ("driven ACROSS the arm's other axes, not at one corner of them"), so the
    case exists to drive the missing corner rather than to serve a row: the census
    holds no folded 3-D dispersive row today, and the row costs nothing.

    Measured on the lift 2026-09-12: ``folded`` "mirror plane on Y", ``dimensions``
    3, ``susceptibilities`` 1, ``grid_shape`` (40, 21, 40). Offered the released set
    the composer installs the folded B/H pair on the B seam with the mirror fill in
    both fill slots, and the D seam keeps ``folded`` + ``folded dispersive PML E``
    singles with ``ADE update_P`` live — the folded dispersive D/E pair pins
    ``dimensions`` 2 and the folded D/E pair pins ``susceptibilities`` 0, so neither
    is admitted here. 11 -> 10 launches per step, byte-identical.
    """
    lorentz = mp.Medium(epsilon=2.25, E_susceptibilities=[
        mp.LorentzianSusceptibility(frequency=0.4, gamma=0.02, sigma=1.1)])
    cell = mp.Vector3(4, 4, 4)
    geometry = [mp.Block(mp.Vector3(mp.inf, 1.0, mp.inf), center=mp.Vector3(),
                         material=lorentz)]
    sources = [mp.Source(e2e._gaussian(mp, 0.4, 0.4), component=mp.Ez,  # noqa: SLF001
                         center=mp.Vector3(-1.2, 0, 0))]
    sim = mp.Simulation(cell_size=cell, boundary_layers=[mp.PML(0.8)],
                        geometry=geometry, sources=sources, resolution=res,
                        symmetries=[mp.Mirror(mp.Y)])
    incident = sim.add_flux(0.4, 0.4, 3, mp.FluxRegion(
        center=mp.Vector3(0.9, 0, 0), size=mp.Vector3(0, 2, 2)))
    return sim, [incident], 35.0


# ---------------------------------------------------------------------------
# The nine cases added 2026-09-13, and why every one of them is DEFINED HERE
# ---------------------------------------------------------------------------
#
# THE HOME RULE, AND THE ONE PLACE IT IS ALLOWED TO DUPLICATE. A builder belongs in
# ``gate_dispatch_fused_route`` when more than one backend's route gate drives the
# same shape, and Triton and CUDA do drive most of the nine below. It is written
# HERE ANYWAY, for the reason :func:`case_offdiag_2d` states and one more: that
# module's bytes are pinned by the CUDA driver record, the two campaigns run on
# different hardware and on different days, and a Metal DRIVE row whose builder
# lives in a file another lane is editing this hour would make this gate's cases
# depend on that lane's landing order. A name defined in both files resolves HERE —
# ``CASES`` starts as a copy of that module's and these assignments overwrite it —
# so a divergence between the two constructions would be invisible from this side.
# THE TWO COPIES MUST THEREFORE AGREE CELL FOR CELL; every construction below is the
# round's canonical one, and the existing duplicates (``offdiag_2d``,
# ``complex_nobloch_2d``, ``folded_3d``, ``cylindrical_m1``,
# ``cylindrical_m0_complex``) are under the same rule.
#
# WHAT WAS MEASURED BEFORE THESE WERE TYPED, and what was not. Every builder below
# was constructed and handed to ``lift_simulation(prefer_gpu=False)`` on stock MEEP
# 1.33.0 on 2026-09-13, WITH ITS MONITORS, and the run shape quoted in each
# docstring is ``fastpath._run_shape(...)`` read back verbatim; each lifted shape was
# then run through ``metal_dispatch.released_fused_arms_metal`` with this round's
# release rows in place, and that admitted set is what the DRIVE table's ``arms``
# field carries. NOT measured, and it cannot be on this host: the fused pair install
# refuses on a NumPy array module, so which labels the composer SELECTS, how many
# launches each composition costs, and every byte comparison are the campaign's to
# take. ``launches_per_step`` is therefore OMITTED from the nine new DRIVE rows
# rather than guessed — the record reads it as an empty list and says nothing —
# while ``pairs`` and ``seam``, which the ladder checks against, are inherited from
# the named analogous row and must be confirmed by
# ``probe_metal_dispatch_dryrun.py`` before the campaign is trusted. A wrong
# ``pairs`` fails loudly at the slot count; a wrong ``launches_per_step`` would have
# been a typed number nobody measured.


def case_dispersive6_2d(mp, res=15):
    """``dispersive_2d`` at SIX susceptibility states: the pole count as one axis.

    The ordinary magnetic pair's ``susceptibilities`` row was an omitted axis until
    the 2026-09-13 verifier caught what omission means on an INTEGER axis — four
    TestLoadDump rows carrying five Lorentz poles admitted on evidence of zero and
    one — and the row became the enumeration ``{0, 1}``. That enumeration refuses
    every count no case drove, which is correct and which also refuses the three
    ``stochastic_emitter`` rows the composer selects this pair for. This case is the
    six in ``{0, 1, 6}``: an axis-aligned block carrying six explicit Lorentzians,
    everything else as ``case_dispersive_2d``, so the pole COUNT is the only thing
    that moves.

    Measured on the lift 2026-09-13: ``dimensions`` 2, ``grid_shape`` (90, 90, 1),
    ``susceptibilities`` 6, ``complex_storage`` False, ``bloch`` False, ``beta`` 0,
    ``conductivity`` False, ``off_diagonal_epsilon`` False, ``pml_active`` True.

    SIX EXPLICIT LORENTZIANS RATHER THAN ``meep.materials.Ag``, which is the corpus
    rows' own material and reads six states as well (one Drude then five
    Lorentzian). The release reads the COUNT and nothing else about the poles, so
    the two are the same evidence on the axis this case exists for, and an explicit
    medium keeps the case readable and independent of a catalogue that can be
    revised. If the corpus material is preferred, ``Ag`` is the substitution
    and the lifted count must be re-read rather than assumed.

    TWO ARMS, NOT ONE. ``fused dispersive D/E pair`` leaves ``susceptibilities``
    unpinned — the admitted asymmetry recorded on
    ``metal_dispatch.METAL_FUSED_RELEASE_ARM_AXES`` — so the release admits it here
    beside the magnetic pair (measured on the lift), and a DRIVE row naming only the
    magnetic pair would offer a label short of what the release covers.
    """
    medium = mp.Medium(epsilon=2.0, E_susceptibilities=[
        mp.LorentzianSusceptibility(frequency=0.3 + 0.1 * i, gamma=0.05,
                                    sigma=0.4 / (i + 1)) for i in range(6)])
    cell = mp.Vector3(6, 6, 0)
    geometry = [mp.Block(mp.Vector3(mp.inf, 1.0, mp.inf), center=mp.Vector3(),
                         material=medium)]
    sources = [mp.Source(e2e._gaussian(mp, 0.3, 0.3), component=mp.Ez,  # noqa: SLF001
                         center=mp.Vector3(-1.5, 0))]
    sim = mp.Simulation(cell_size=cell, boundary_layers=[mp.PML(1.0)],
                        geometry=geometry, sources=sources, resolution=res)
    monitor = sim.add_flux(0.3, 0.3, 5, mp.FluxRegion(
        center=mp.Vector3(1.5, 0), size=mp.Vector3(0, 4)))
    return sim, [monitor], 45.0


def case_folded_dispersive5_2d(mp, res=20):
    """``folded_dispersive_2d`` at FIVE susceptibility states: the folded pole count.

    The folded magnetic pair's ``susceptibilities`` row carries the same 2026-09-13
    integer-axis correction as the ordinary pair's, and the four rows it costs are
    the folded ``TestLoadDump`` 2-D tests at grid (250, 126, 1), every one of them
    five poles. This is ``case_folded_dispersive_2d`` with five explicit Lorentzians
    in place of its single pole — the same cell, PML, mirror plane, source and
    monitor — so the two cases differ in ``susceptibilities`` alone.

    Measured on the lift 2026-09-13: ``dimensions`` 2, ``grid_shape`` (80, 61, 1),
    ``folded`` "mirror plane on Y", ``susceptibilities`` 5, ``complex_storage``
    False, ``bloch`` False, ``beta`` 0, ``conductivity`` False,
    ``off_diagonal_epsilon`` False, ``pml_active`` True.

    TWO ARMS, for :func:`case_dispersive6_2d`'s reason: ``folded fused dispersive
    D/E pair`` also leaves ``susceptibilities`` unpinned and is admitted here
    (measured on the lift), so the row names it beside the folded magnetic pair.

    THIS SHAPE IS ALSO THE TRITON GATE'S ENVELOPE WITNESS, which is worth knowing
    from this side: on that table the release DECLINES every folded arm here, and
    driving it as a Metal dispatch case does not touch that, because the two gates
    read their own release tables. It is the same asymmetry ``pml_3d_diagonal``
    already has in the other direction — an envelope control there, a dispatch case
    here.
    """
    medium = mp.Medium(epsilon=2.25, E_susceptibilities=[
        mp.LorentzianSusceptibility(frequency=0.3 + 0.1 * i, gamma=0.05,
                                    sigma=0.4 / (i + 1)) for i in range(5)])
    cell = mp.Vector3(4, 6, 0)
    geometry = [mp.Block(mp.Vector3(mp.inf, 1.0, mp.inf), center=mp.Vector3(),
                         material=medium)]
    sources = [mp.Source(e2e._gaussian(mp, 0.25, 0.3), component=mp.Ez,  # noqa: SLF001
                         center=mp.Vector3(-1.2, 0))]
    sim = mp.Simulation(cell_size=cell, boundary_layers=[mp.PML(1.0)],
                        geometry=geometry, sources=sources, resolution=res,
                        symmetries=[mp.Mirror(mp.Y)])
    monitor = sim.add_flux(0.25, 0.3, 5, mp.FluxRegion(
        center=mp.Vector3(0, 1.6), size=mp.Vector3(4, 0)))
    return sim, [monitor], 40.0


def case_complex_1d(mp, res=40):
    """Complex storage on a cell DECLARED 1-D: the 1 in the complex pairs' dimensions.

    ``dimensions=1`` is passed EXPLICITLY rather than left to the cell extents. A
    declared 1 wins outright in the dimensions reader, so stating it makes the
    case's subject the axis the release is widened on instead of a property of the
    geometry — and it is the shape ``refl-angular.py``, ``TestPlanewave1D`` and
    ``TestReflectanceAngular_0_0`` carry.

    Measured on the lift 2026-09-13: ``dimensions`` 1, ``grid_shape`` (1, 1, 480),
    ``complex_storage`` True, ``bloch`` True, ``k_point`` (0, 0, 1.25), ``beta`` 0,
    ``conductivity`` False, ``off_diagonal_epsilon`` False, ``susceptibilities`` 0,
    ``pml_active`` True; both complex pairs admitted.

    IT IS THE FIRST 1-D EVIDENCE EITHER COMPLEX ARM CARRIES, and that is the reason
    it is here rather than a consequence of it. ``gate_metal_tranche7_fused_pairs``
    and ``gate_metal_complex_fused_magnetic_pair`` build their complex fixtures
    through ``matrix.cart``, whose cell is 3-D, and neither family gate has ever
    driven ``dimensions`` 1 on these products. The route gate's byte-identity leg
    through the real driver seam is what supplies it, exactly as ``pml_1d`` does for
    the ordinary pairs.
    """
    cell = mp.Vector3(0, 0, 12)
    sources = [mp.Source(e2e._gaussian(mp, 0.25, 0.3), component=mp.Ex,  # noqa: SLF001
                         center=mp.Vector3(0, 0, -4.0))]
    sim = mp.Simulation(cell_size=cell, boundary_layers=[mp.PML(1.0)],
                        sources=sources, resolution=res, dimensions=1,
                        k_point=mp.Vector3(0, 0, 1.25),
                        force_complex_fields=True)
    monitor = sim.add_flux(0.25, 0.3, 5, mp.FluxRegion(center=mp.Vector3(0, 0, 3)))
    return sim, [monitor], 40.0


def case_complex_3d_thinline(mp, res=25):
    """A ``(0, 0, L)`` cell DECLARED 3-D under an oblique k: a 3 on a (1, 1, N) grid.

    ``cell.z != 0`` with a declared 3, so MEEP builds a grid one cell wide in x and
    y and the dimensions reader keeps the 3 — which is the whole point. Four corpus
    rows are physically one-dimensional and answer ``dimensions`` 3:
    ``antenna_pec_ground_plane_1D.py``, ``dipole_in_vacuum_1D.py``,
    ``TestBoundaries1D.test_boundaries_1D`` and
    ``TestReflectanceAngular.test_reflectance_angular_1_20_6``. The oblique
    ``k_point`` is theirs too.

    Measured on the lift 2026-09-13: ``dimensions`` 3, ``grid_shape`` (1, 1, 275),
    ``complex_storage`` True, ``bloch`` True, ``k_point`` (0.41, 0, 1.8), ``beta``
    0, ``susceptibilities`` 0, ``pml_active`` True; both complex pairs admitted.

    IT IS HALF THE EVIDENCE FOR THE 3 AND NOT ALL OF IT. This grid is degenerate —
    one cell in two directions — so a ``dimensions`` row widened on this case alone
    would license 3-D complex storage on nothing but thin lines.
    :func:`case_complex_3d` is the other half.
    """
    cell = mp.Vector3(0, 0, 11)
    sources = [mp.Source(e2e._gaussian(mp, 0.3, 0.3), component=mp.Ex,  # noqa: SLF001
                         center=mp.Vector3(0, 0, -4.0))]
    sim = mp.Simulation(cell_size=cell, boundary_layers=[mp.PML(1.0)],
                        sources=sources, resolution=res, dimensions=3,
                        k_point=mp.Vector3(0.41, 0, 1.8),
                        force_complex_fields=True)
    monitor = sim.add_flux(0.3, 0.3, 5, mp.FluxRegion(center=mp.Vector3(0, 0, 3)))
    return sim, [monitor], 40.0


def case_complex_3d(mp, res=10):
    """A GENUINE 3-D complex block: the case that keeps the widened 3 from being degenerate.

    THE HONESTY CASE OF THE ROUND, and it is worth stating plainly because it
    credits nothing. The seven corpus rows the complex pairs' ``dimensions``
    widening actually serves are ALL ``(1, 1, N)`` cells, so :func:`case_complex_1d`
    and :func:`case_complex_3d_thinline` between them would have licensed
    ``dimensions`` 3 on evidence from degenerate grids — while the census carries
    genuine 3-D complex rows the widened row admits at RUNTIME:
    ``TestLoadDump.{structure, structure_sharded, chunk_layout_file,
    chunk_layout_sim}_3d`` at (35, 32, 41) and ``TestMaterialGrid.test_matgrid_3d``
    at (25, 25, 25). Measured on the 2026-09-13 census, none of those five is
    CREDITED to these products (their D seam goes to the conductive and
    no-PML-off-diagonal complex families), so the board number is identical with and
    without this case. It is in the set because the alternative is a row that
    admits a shape no case drove, which is the defect class this round exists to
    close.

    An AXIS-ALIGNED block, so ``chi1inv`` stays diagonal and the case is about one
    axis. Measured on the lift 2026-09-13: ``dimensions`` 3, ``grid_shape``
    (40, 40, 40), ``complex_storage`` True, ``bloch`` True, ``k_point``
    (0.3, 0, 0), ``beta`` 0, ``off_diagonal_epsilon`` False, ``susceptibilities`` 0,
    ``pml_active`` True; both complex pairs admitted.

    THE PML IS ON Z ALONE, which is a composition difference from every other
    complex case on this table and is the one thing the dry run must look at before
    the row is trusted: x and y stay periodic under the Bloch phase, so the no-PML
    arms own those directions' work and the slot set may not be the one
    ``complex_nobloch_2d`` measured. ``pairs`` and ``seam`` below are inherited from
    that row and are a prediction, not a reading.
    """
    cell = mp.Vector3(4, 4, 4)
    geometry = [mp.Block(mp.Vector3(mp.inf, 1.0, mp.inf), center=mp.Vector3(),
                         material=mp.Medium(epsilon=12))]
    sources = [mp.Source(e2e._gaussian(mp, 0.4, 0.4), component=mp.Ez,  # noqa: SLF001
                         center=mp.Vector3(-1.2, 0, 0))]
    sim = mp.Simulation(cell_size=cell,
                        boundary_layers=[mp.PML(0.8, direction=mp.Z)],
                        geometry=geometry, sources=sources, resolution=res,
                        k_point=mp.Vector3(0.3, 0, 0),
                        force_complex_fields=True)
    monitor = sim.add_flux(0.4, 0.4, 3, mp.FluxRegion(
        center=mp.Vector3(0.9, 0, 0), size=mp.Vector3(0, 2, 2)))
    return sim, [monitor], 35.0


def case_folded_complex_3d(mp, res=20):
    """A fold, a nonzero k_point and three dimensions at once: the folded complex 3.

    The triangular-lattice-oblique cell — a mirror plane on X, an oblique Bloch
    vector with components in y and z, PML on z alone — which is
    ``TestModeDecomposition.test_triangular_lattice_oblique``'s own shape and the
    row this widening serves.

    Measured on the lift 2026-09-13: ``dimensions`` 3, ``grid_shape`` (8, 21, 126),
    ``folded`` "mirror plane on X", ``complex_storage`` True, ``bloch`` True,
    ``k_point`` (0, -1.7035, 2.4694), ``beta`` 0, ``off_diagonal_epsilon`` False,
    ``susceptibilities`` 0, ``pml_active`` True; both folded complex pairs admitted.

    THE CAVEAT THIS CASE CARRIES, and it is a release decision rather than a defect.
    ``folded complex fused D/E pair`` has NO device gate file of its own — it binds
    to the tranche-7 GROUP weld, whose four folded-complex fixtures are built at
    ``matrix.folded``'s default depth and are all 2-D — so this route case would be
    the FIRST 3-D evidence that arm has ever carried. Its magnetic twin is backed by
    eleven 3-D fixtures on ``gate_metal_folded_complex_fused_magnetic_pair``. If a
    route case may not be an arm's first 3-D evidence, the D/E half is withheld by
    dropping that arm's two case names from ``METAL_RELEASED_FUSED_ARMS`` and
    re-pinning its ``dimensions`` row at 2; the cheaper repair is one line in
    ``gate_metal_tranche7_fused_pairs.py`` adding a 3-D folded-complex fixture,
    which turns this into a coverage extension rather than a first certification.

    THE SOURCE IS IN THE HALF THE MIRROR PLANE KEEPS, which is load-bearing and was
    learned the hard way on this case's sibling. MEEP adds sources with
    ``use_symmetry=false``, so a source in the discarded half lands in no stored
    chunk: the run returns a field whose peak is exactly 0 while reading as
    complete, and the lift refuses by name rather than letting it through. The
    mirror is on X and the source sits at x = 0, on the plane.
    """
    cell = mp.Vector3(0.6, 1.0392304845413263, 6.3)
    sources = [mp.Source(e2e._gaussian(mp, 0.3, 0.3), component=mp.Ez,  # noqa: SLF001
                         center=mp.Vector3(0, 0, -2.0))]
    sim = mp.Simulation(cell_size=cell,
                        boundary_layers=[mp.PML(1.0, direction=mp.Z)],
                        sources=sources, resolution=res,
                        k_point=mp.Vector3(0, -1.7035, 2.4694),
                        symmetries=[mp.Mirror(mp.X)],
                        force_complex_fields=True)
    monitor = sim.add_flux(0.3, 0.3, 5, mp.FluxRegion(
        center=mp.Vector3(0, 0, 2.0), size=mp.Vector3(0.6, 1.0392304845413263, 0)))
    return sim, [monitor], 40.0


def case_folded_complex_kz2d_3d(mp, res=20):
    """A folded zero-thickness cell asked for as 3-D: the second corner of that row.

    ``kz_2d="3d"`` is what makes this a ``dimensions`` 3 case rather than a beta
    one: the cell has zero z extent and a k_point with a nonzero z component, and
    that spelling keeps the z phase out of ``beta`` so the declared 3 stands. MEEP
    reports "Working in 3D dimensions" and the lift agrees. It is
    ``TestEigCoeffs.test_binary_grating_special_kz_2_21_2``'s shape.

    Measured on the lift 2026-09-13: ``dimensions`` 3, ``grid_shape`` (90, 62, 1),
    ``folded`` "mirror plane on Y", ``complex_storage`` True, ``bloch`` True,
    ``k_point`` (2.797, 0, -1.0849), ``beta`` 0, ``susceptibilities`` 0,
    ``pml_active`` True; both folded complex pairs admitted.

    WHY BOTH THIS AND :func:`case_folded_complex_3d`. They are the same axis at two
    very different builds — a (8, 21, 126) volumetric fold and a (90, 62, 1) sheet
    declared 3-D — and the two corpus rows behind the widening are one of each. A
    single case would have left the other's shape admitted on an inference.

    THE SOURCE IS AT ``y = +2.0``, IN THE HALF THE MIRROR KEEPS, and this is the
    case that taught the rule: declared at ``y = -2.0`` the lift refuses by name,
    because MEEP adds sources with ``use_symmetry=false`` and a source in the
    discarded half lands in no stored chunk — the run then returns an exactly-zero
    field while reading as complete. Every folded case on this table puts its
    sources in the kept half.
    """
    cell = mp.Vector3(4.5, 6.0, 0)
    geometry = [mp.Block(mp.Vector3(mp.inf, 1.0, mp.inf), center=mp.Vector3(),
                         material=mp.Medium(epsilon=12))]
    sources = [mp.Source(e2e._gaussian(mp, 0.3, 0.3), component=mp.Ez,  # noqa: SLF001
                         center=mp.Vector3(0, 2.0))]
    sim = mp.Simulation(cell_size=cell,
                        boundary_layers=[mp.PML(1.0, direction=mp.Y)],
                        geometry=geometry, sources=sources, resolution=res,
                        k_point=mp.Vector3(2.7970, 0, -1.0849), kz_2d="3d",
                        symmetries=[mp.Mirror(mp.Y)])
    monitor = sim.add_flux(0.3, 0.3, 5, mp.FluxRegion(
        center=mp.Vector3(0, 1.6), size=mp.Vector3(4.5, 0)))
    return sim, [monitor], 45.0


def case_folded_complex_offdiag_2d(mp, res=20):
    """``folded_complex_2d`` AND an off-diagonal ``chi1inv``, differing in that ONE axis.

    ``folded complex fused B/H pair`` was ABSENT from
    ``metal_dispatch.METAL_OFF_DIAGONAL_CORNERS`` until 2026-09-13, and an arm
    absent from that table is refused on every off-diagonal grid — which is the
    whole of what kept ``TestHoleyWvgBands.test_fields_at_kx`` off the dispatch
    route. This case is the corner entry's Bloch half: the folded complex cell with
    its slab replaced by a CYLINDER CENTRED ON THE FOLD PLANE, so the mirror stays
    exact and the curved surface is what makes MEEP's subpixel averaging write the
    off-diagonal rows — the same mechanism :func:`case_offdiag_2d` and
    :func:`case_folded_offdiag_2d` use.

    Measured on the lift 2026-09-13: ``dimensions`` 2, ``grid_shape`` (80, 62, 1),
    ``folded`` "mirror plane on Y", ``complex_storage`` True, ``bloch`` True,
    ``k_point`` (0.3, 0, 0), ``beta`` 0, ``conductivity`` False,
    ``off_diagonal_epsilon`` True, ``susceptibilities`` 0, ``pml_active`` True.

    EXACTLY ONE ARM IS ADMITTED HERE (measured on the lift): the folded complex
    MAGNETIC pair. The D/E twin has no corner of its own, and no D-seam product is
    admitted on an off-diagonal grid at all, so this is a ONE-seam case and the
    refusal beside it is half of what the case records.

    THE FAMILY EVIDENCE IS AT 3-D AND THIS MOVES IT TO 2-D.
    ``gate_metal_folded_complex_fused_magnetic_pair`` already drives this arm
    off-diagonal on ``y_metallic_offdiag_3d`` and ``xy_mixed_offdiag_wall_z_3d``;
    all three corpus rows behind the corner are 2-D, which is why the corner reads
    ``dimensions`` {2} and why the route case is built at 2-D.
    """
    cell = mp.Vector3(4, 6, 0)
    geometry = [mp.Cylinder(radius=0.8, center=mp.Vector3(),
                            material=mp.Medium(epsilon=12))]
    sources = [mp.Source(e2e._gaussian(mp, 0.25, 0.3), component=mp.Ez,  # noqa: SLF001
                         center=mp.Vector3(-1.2, 0))]
    sim = mp.Simulation(cell_size=cell,
                        boundary_layers=[mp.PML(1.0, direction=mp.Y, side=mp.High)],
                        geometry=geometry, sources=sources, resolution=res,
                        k_point=mp.Vector3(0.3, 0, 0),
                        symmetries=[mp.Mirror(mp.Y)])
    monitor = sim.add_flux(0.25, 0.3, 5, mp.FluxRegion(
        center=mp.Vector3(0, 1.6), size=mp.Vector3(4, 0)))
    return sim, [monitor], 40.0


def case_folded_complex_nobloch_offdiag_2d(mp, res=10):
    """A fold and forced complex storage with NO k_point, on an off-diagonal grid.

    THE UNPHASED HALF of the folded complex magnetic pair's off-diagonal corner, and
    the case that lets that arm's row drop its ``bloch`` pin. Two mirror planes and
    ``force_complex_fields`` with no ``k_point`` at all, a cylinder for the
    off-diagonal ``chi1inv``: it is ``examples:solve-cw.py``'s and
    ``tests:TestArrayMetadata.test_array_metadata``'s shape, and those two rows are
    the reason both halves are needed at once — measured on the 2026-09-13 census
    they are refused TWICE OVER, on ``bloch`` and on ``off_diagonal_epsilon``, so
    the corner entry without the pin drop recovers one instance and the pin drop
    without the corner entry recovers none.

    Measured on the lift 2026-09-13: ``dimensions`` 2, ``grid_shape`` (81, 81, 1),
    ``folded`` "mirror plane on X, Y", ``complex_storage`` True, ``bloch`` False,
    ``k_point`` (0, 0, 0), ``beta`` 0, ``off_diagonal_epsilon`` True,
    ``susceptibilities`` 0, ``pml_active`` True; exactly one arm admitted, the
    folded complex magnetic pair.

    THE SOURCE IS IN THE QUADRANT BOTH PLANES KEEP, at (+4, +4). At (-4, 0) the lift
    refuses, for the reason :func:`case_folded_complex_kz2d_3d` records.

    IT IS EVIDENCE FOR THE MAGNETIC ARM AND FOR NOTHING ELSE, which is why the
    folded complex D/E pair KEEPS its ``bloch`` pin while the magnetic twin drops
    it: the release does not admit the D/E arm on an off-diagonal grid, so this case
    cannot be the "both values driven" licence for it. Writing that drop on the
    twin's evidence would be the over-claim the row tables exist to refuse.
    """
    cell = mp.Vector3(16, 16, 0)
    geometry = [mp.Cylinder(radius=1.2, center=mp.Vector3(),
                            material=mp.Medium(epsilon=12))]
    sources = [mp.Source(e2e._gaussian(mp, 0.25, 0.3), component=mp.Ez,  # noqa: SLF001
                         center=mp.Vector3(4.0, 4.0))]
    sim = mp.Simulation(cell_size=cell, boundary_layers=[mp.PML(1.0)],
                        geometry=geometry, sources=sources, resolution=res,
                        force_complex_fields=True,
                        symmetries=[mp.Mirror(mp.X), mp.Mirror(mp.Y)])
    monitor = sim.add_flux(0.25, 0.3, 5, mp.FluxRegion(
        center=mp.Vector3(6.0, 0), size=mp.Vector3(0, 8)))
    return sim, [monitor], 50.0


CASES["offdiag_2d"] = case_offdiag_2d
CASES["folded_offdiag_2d"] = case_folded_offdiag_2d
CASES["folded_3d"] = case_folded_3d
CASES["folded_dispersive_3d"] = case_folded_dispersive_3d
CASES["complex_nobloch_2d"] = case_complex_nobloch_2d
# THE NINE ADDED 2026-09-13. These assignments OVERWRITE any same-named builder
# inherited from ``gate_dispatch_fused_route`` — see the home-rule note above the
# builders — so where a name exists in both files the two constructions must agree
# cell for cell, because a divergence is invisible from this side.
CASES["dispersive6_2d"] = case_dispersive6_2d
CASES["folded_dispersive5_2d"] = case_folded_dispersive5_2d
CASES["complex_1d"] = case_complex_1d
CASES["complex_3d_thinline"] = case_complex_3d_thinline
CASES["complex_3d"] = case_complex_3d
CASES["folded_complex_3d"] = case_folded_complex_3d
CASES["folded_complex_kz2d_3d"] = case_folded_complex_kz2d_3d
CASES["folded_complex_offdiag_2d"] = case_folded_complex_offdiag_2d
CASES["folded_complex_nobloch_offdiag_2d"] = case_folded_complex_nobloch_offdiag_2d
CASES["cylindrical_m1"] = case_cylindrical_m1
CASES["cylindrical_m0_complex"] = case_cylindrical_m0_complex

CASE_INTENT: Dict[str, str] = dict(route.CASE_INTENT)
CASE_INTENT["cylindrical"] = (
    "the cylindrical m = 0 electric pair on a Dcyl grid. THE BUILDER CARRIES NO "
    "FLUX MONITOR — it is the end-to-end gate's, and the Metal release cites this "
    "name — so the observable comparison and the second consult site are recorded "
    "NOT-APPLICABLE here rather than counted as passes")
CASE_INTENT["special_kz_2d"] = (
    "MEEP's special_kz on real storage: the beta electric pair is selected by the "
    "composer and, from 2026-09-12, RELEASED, so the D seam fuses. It was this "
    "table's only no-fused-arm case until then")
CASE_INTENT["pml_3d"] = (
    "THE OFF-DIAGONAL GRID: a sphere, whose subpixel averaging writes off-diagonal "
    "chi1inv rows. It was the ENVELOPE CONTROL until 2026-09-12; it dispatches the "
    "ordinary magnetic pair now, and carries the refusal of every other arm by "
    "name")
CASE_INTENT["no_pml_2d"] = (
    "THE ENVELOPE CONTROL from 2026-09-12: no absorber, so the SHARED release "
    "envelope refuses on pml_active and the released set is empty. It is the only "
    "case that witnesses the shared table saying no")
CASE_INTENT["offdiag_2d"] = (
    "THE 2-D OFF-DIAGONAL GRID: pml_2d with a cylinder in place of the slab, so "
    "only off_diagonal_epsilon differs. The ordinary magnetic pair is admitted "
    "inside its off-diagonal corner (2-D joined 3-D on 2026-09-12); the D seam "
    "keeps the offdiag singles")
CASE_INTENT["folded_offdiag_2d"] = (
    "THE FOLDED 2-D OFF-DIAGONAL GRID: folded_2d with a cylinder centred on the "
    "fold plane, so only off_diagonal_epsilon differs. The folded magnetic pair is "
    "admitted inside its corner (susceptibilities=0); the folded electric pair is "
    "refused on the off-diagonal grid and the D seam keeps the folded offdiag "
    "singles")
CASE_INTENT["folded_3d"] = (
    "THE FOLDED 3-D GRID: both folded pairs with the mirror fill in both fill "
    "slots, on a block, so the case is about the dimensions row alone")
CASE_INTENT["folded_dispersive_3d"] = (
    "THE FOLDED 3-D GRID OVER A SUSCEPTIBILITY: the folded magnetic pair alone, "
    "driving the (3-D, one pole) corner its omitted susceptibilities row needs")
CASE_INTENT["complex_nobloch_2d"] = (
    "COMPLEX STORAGE WITHOUT A k_point: both complex pairs under the complex "
    "expansion licence, and the case that lets their rows drop bloch")
CASE_INTENT["dispersive6_2d"] = (
    "SIX SUSCEPTIBILITY STATES on an unfolded 2-D grid: the 6 in the ordinary "
    "magnetic pair's enumerated susceptibilities row, with the dispersive electric "
    "pair beside it because that arm leaves the axis unpinned")
CASE_INTENT["folded_dispersive5_2d"] = (
    "FIVE SUSCEPTIBILITY STATES on a fold: the 5 in the folded magnetic pair's "
    "enumerated susceptibilities row, with the folded dispersive electric pair "
    "beside it for the same reason. It is also the Triton gate's envelope witness, "
    "which is a decision on that table and does not reach this one")
CASE_INTENT["complex_1d"] = (
    "COMPLEX STORAGE ON A CELL DECLARED 1-D: the 1 in the complex pairs' dimensions "
    "row, and the first 1-D evidence either arm carries — neither family gate has "
    "driven dimensions=1")
CASE_INTENT["complex_3d_thinline"] = (
    "A (0, 0, L) CELL DECLARED 3-D: the shape four physically one-dimensional "
    "corpus rows carry, lifting to dimensions 3 on a (1, 1, N) grid. Half the "
    "evidence for the 3; complex_3d is the other half")
CASE_INTENT["complex_3d"] = (
    "A GENUINE 3-D COMPLEX BLOCK: the case that stops the widened dimensions row "
    "being evidenced by degenerate grids alone. It credits nothing on the board and "
    "is here for that reason")
CASE_INTENT["folded_complex_3d"] = (
    "A FOLD, A NONZERO k_point AND THREE DIMENSIONS AT ONCE: the folded complex "
    "pairs' dimensions row opened to {2, 3}, on the triangular-lattice-oblique cell")
CASE_INTENT["folded_complex_kz2d_3d"] = (
    "A FOLDED ZERO-THICKNESS CELL ASKED FOR AS 3-D (kz_2d='3d', so the z phase does "
    "not lift as a beta): the second and differently-built corner of the same "
    "dimensions row")
CASE_INTENT["folded_complex_offdiag_2d"] = (
    "THE FOLDED COMPLEX OFF-DIAGONAL GRID, Bloch half: folded_complex_2d with a "
    "cylinder on the fold plane, so only off_diagonal_epsilon differs. Exactly one "
    "arm is admitted — the folded complex magnetic pair, inside the corner this "
    "round gives it — and every other arm is refused by name")
CASE_INTENT["folded_complex_nobloch_offdiag_2d"] = (
    "THE FOLDED COMPLEX OFF-DIAGONAL GRID, unphased half: two mirror planes and "
    "force_complex_fields with no k_point. It is what lets the folded complex "
    "MAGNETIC pair's row drop its bloch pin; the D/E twin keeps that pin, because "
    "no D-seam product is admitted on an off-diagonal grid and this case is "
    "therefore not evidence for it")
CASE_INTENT["cylindrical_m1"] = (
    "Dcyl at m = 1: the cylindrical COMPLEX electric pair on the D seam under the "
    "cylindrical-complex expansion licence; the B seam keeps the singles")
CASE_INTENT["cylindrical_m0_complex"] = (
    "Dcyl at m = 0 with complex storage forced: the cylindrical COMPLEX electric "
    "pair on the corner its row admits without an m pin")


# ---------------------------------------------------------------------------
# The DRIVE table
# ---------------------------------------------------------------------------

#: Which fused arms each case is asked for, what the ladder must find, and how many
#: launches the fusion is expected to SUBSTITUTE per step: ``pairs``, plus N - 1 for
#: each slot a row declares under ``collapsed_single_launches_per_step`` (from
#: 2026-09-19, one row: ``complex_no_pml_3d``; see :func:`expected_drop_per_step`).
#:
#: MEASURED 2026-09-10, NOT GUESSED — FOR THE ROWS THAT CARRY A DATE OF 2026-09-10
#: OR EARLIER. Each of those was read by lifting the case with
#: ``lift_simulation(prefer_gpu=False)``, evaluating
#: ``metal_dispatch.released_fused_arms_metal(fastpath._run_shape(...))`` on it, and
#: composing with ``metal_kernels.launch.plan_step(fuse=True, fuse_labels=<the
#: released set>)`` — which is exactly what rung 5M does. ``launches_per_step`` was
#: then counted off the plans' own ``KernelPlan.launches`` over eight complete steps
#: after the freeze, in both compositions.
#:
#: THE NINE ROWS ADDED 2026-09-13 ARE HALF-MEASURED, AND THE SENTENCE ABOVE WOULD
#: HAVE COVERED THEM FALSELY, so the split is stated here rather than left to a
#: reader comparing dates. On those rows the LIFT and the RELEASE were taken — every
#: builder was constructed with its monitors and handed to
#: ``lift_simulation(prefer_gpu=False)`` on stock MEEP 1.33.0, and the ``arms`` list
#: is the admitted set ``released_fused_arms_metal`` returned on the shape that came
#: back — but the COMPOSITION was not, and could not be: the fused pair install
#: refuses on a NumPy array module, so ``plan_step`` reports no fused label on the
#: host that typed them. ``pairs`` and ``seam`` there are inherited from a named
#: analogous row and are predictions the ladder checks; ``launches_per_step`` is
#: OMITTED, which the record reads as an empty list, rather than carried over from a
#: row whose pole count or dimensionality differs. Run
#: ``probe_metal_dispatch_dryrun.py`` on this Mac's MPS and fill the three in before
#: the campaign's numbers are trusted — that is the same order of operations the
#: table's own note on typed rows prescribes, and the reason the release table can
#: be typed ahead of the run at all is that
#: ``recut_driver_dispatch_record.py --backend metal`` refuses to write a record for
#: any ``(arm, case)`` pair no dispatching leg drove.
#:
#: ``arms`` IS WHAT THE RELEASE ADMITS HERE, not what the composer would select if
#: everything were offered, and on three cases those differ. The composer is handed
#: the admitted label set, so a product the release does not cover is never
#: installed and its seam keeps the separate certified arms:
#:
#: * ``conductive_2d``, ``folded_dispersive_2d`` and ``special_kz_2d`` each USED to
#:   be such a case and stopped being one on 2026-09-12, when the three families
#:   they select were released on their group welds. Their drops deepened by one
#:   launch each (4->2, 11->9, and 4->4 becoming 4->3) and all three are ordinary
#:   dispatching rows now;
#: * ``bloch_2d``, ``complex_nobloch_2d``, ``folded_complex_2d`` and
#:   ``cylindrical_m1`` dispatch ONLY on the probe leg, because their pairs are
#:   licensed by an expansion probe outside the tree; on the shipped leg every one
#:   of them is refused by name. Until later on 2026-09-12 the two complex D/E
#:   pairs were selected-but-withheld on the first two, so those rows drove one
#:   pair; they drive two now. SEVEN OF THE NINE ROWS ADDED 2026-09-13 ARE IN THE
#:   SAME POSITION — ``complex_1d``, ``complex_3d_thinline`` and ``complex_3d``
#:   under the complex probe, ``folded_complex_3d``, ``folded_complex_kz2d_3d``,
#:   ``folded_complex_offdiag_2d`` and ``folded_complex_nobloch_offdiag_2d`` under
#:   the folded-complex one — and the two real-storage additions
#:   (``dispersive6_2d``, ``folded_dispersive5_2d``) need no probe at all;
#: * ``pml_3d``, ``offdiag_2d``, ``folded_offdiag_2d`` and, from 2026-09-13,
#:   ``folded_complex_offdiag_2d`` and ``folded_complex_nobloch_offdiag_2d`` are the
#:   off-diagonal grids, and each admits EXACTLY ONE arm — the ordinary magnetic
#:   pair on the two unfolded ones, the folded magnetic pair on the real fold, the
#:   folded COMPLEX magnetic pair on the two complex folds — because an arm is
#:   admitted off-diagonal only inside the corner its own off-diagonal cases sat at
#:   (``metal_dispatch.METAL_OFF_DIAGONAL_CORNERS``). Every other arm is refused by
#:   name, which is what makes these rows the per-arm half's standing witnesses. No
#:   D-seam product is admitted on an off-diagonal grid, so all three are one-seam
#:   cases.
#:
#: THE TABLE NO LONGER HAS A ``no-fused-arm`` ROW. ``special_kz_2d`` was the last
#: one. That control did not disappear: it moved to :data:`ENVELOPE_CASE`
#: (``no_pml_2d``), where the SHARED table does the refusing — a place no family
#: release can erode, which is exactly what went wrong with resting it here.
#:
#: ``reached_by`` NAMES THE ROUTE. Every dispatching case runs with
#: ``MEEP_GPU_FUSE_ARMS`` UNSET, because that is what SHIPS: the release admits and
#: the switch is not involved. :func:`release_route_leg` reads the record back and
#: requires it to say so, since "the switch was unset in the environment" and "the
#: release is what admitted the arm" are different claims and only the second is the
#: licence.
DRIVE: Dict[str, Dict[str, Any]] = {
    "pml_2d": {
        "arms": ["fused magnetic B/H pair", "fused electric D/E pair"],
        "pairs": 2, "reached_by": "release", "expect": "dispatch",
        "seam": ["B", "D"], "launches_per_step": (4, 2), "monitors": 2,
        "why": ("BOTH ordinary pairs, and both sentinel shapes in one case. The B "
                "seam is CLEAN (no magnetic source) so update_H holds a NoopPlan; "
                "the D seam carries the electric Gaussian, so step_D holds a "
                "LeadingRepairPlan and update_E the TrailingRepairPlan. It is the "
                "case the residency bracket was first measured on: 45 mirrors, two "
                "brackets per step, byte-identical to 384 steps."),
    },
    "magnetic_seam_2d": {
        "arms": ["fused magnetic B/H pair", "fused electric D/E pair"],
        "pairs": 2, "reached_by": "release", "expect": "dispatch",
        "seam": ["B", "D"], "launches_per_step": (4, 2), "monitors": 2,
        "why": ("THE B-SEAM DEPOSIT REPAIR. The magnetic Gaussian is injected "
                "between step_B and update_H, so step_B holds a LeadingRepairPlan "
                "and update_H the TrailingRepairPlan; the D seam is clean here, so "
                "step_D/update_E are the sentinel pair. Exactly the mirror of "
                "pml_2d, and the reason the pair is in the set: a gate that drove "
                "only electric-source cases would leave one of the two installed "
                "pair shapes undriven on the B side."),
    },
    "conductive_2d": {
        "arms": ["fused magnetic B/H pair",
                 "conductive fused electric D/E pair"],
        "pairs": 2, "reached_by": "release", "expect": "dispatch",
        # TWO SEAMS FROM 2026-09-12, and one until then. The conductive product was
        # withheld for want of a per-family weld; it is released now on its group
        # weld, so the D seam it already owned is FUSED rather than separate. The
        # note below is kept because the 2026-09-11 correction it records is still
        # the reason this entry is trusted to say two.
        # ONE SEAM, and the entry said two until 2026-09-11. The `why` below has
        # always said the D seam keeps its separate arms; the `seam` list claimed it
        # anyway, and the in-seam control read that claim and reported a
        # NULL-DID-NOT-DIVERGE for a mutation that had no fused pair to sit beside.
        "seam": ["B", "D"], "launches_per_step": (4, 2), "monitors": 2,
        "why": ("BOTH SEAMS FUSED, AND THE 2026-09-02 LESSON STILL STANDING "
                "BEHIND IT. A D_conductivity is installed, so the conductive "
                "product owns the D curl; from 2026-09-12 that product is RELEASED "
                "on its group weld, so its seam fuses instead of keeping "
                "`conductive PML curl` + `ordinary`, and the launch count drops "
                "from 3 to 2.\n\n"
                "WHAT THIS CASE USED TO CONTROL FOR: before the offer existed, an "
                "un-admitted product took both slots of its seam and clause 8 "
                "refused the WHOLE plan. That refusal was reachable through "
                "`bloch_2d` and `folded_complex_2d` while the two complex D/E "
                "families sat in METAL_PENDING_DEVICE_GATE_ARMS; they were released "
                "later on 2026-09-12 and that table is empty, so the live witness "
                "of the refusal is now the SHIPPED leg, where every probe-licensed "
                "case is refused by name for want of its licence."),
    },
    "dispersive_2d": {
        "arms": ["fused magnetic B/H pair", "fused dispersive D/E pair"],
        "pairs": 2, "reached_by": "release", "expect": "dispatch",
        "seam": ["B", "D"], "launches_per_step": (9, 7), "monitors": 2,
        "why": ("A SECOND FUSED FAMILY BESIDE THE ORDINARY PAIR, and the one case "
                "whose update_P slot is LIVE. On this table that slot holds ONE "
                "MetalAdeUpdatePPlan whose run() takes a contract rather than the "
                "Triton list of per-susceptibility plans taking a drive field — "
                "measured as a raised KeyError on the dispatch path before "
                "FastPathPlan.dispatch learned the difference, which is where "
                "exceptions PROPAGATE."),
    },
    "pml_1d": {
        "arms": ["fused magnetic B/H pair", "fused electric D/E pair"],
        "pairs": 2, "reached_by": "release", "expect": "dispatch",
        "seam": ["B", "D"], "launches_per_step": (4, 2), "monitors": 1,
        "why": ("ONE LIVE AXIS. dimensions is released as {1, 2, 3} on the two "
                "ordinary pairs, and this is the 1 in it: the composer selects the "
                "same two pairs on a (1, 1, N) grid that it selects in 2-D."),
    },
    "pml_3d_diagonal": {
        "arms": ["fused magnetic B/H pair", "fused electric D/E pair"],
        "pairs": 2, "reached_by": "release", "expect": "dispatch",
        "seam": ["B", "D"], "launches_per_step": (4, 2), "monitors": 1,
        "why": ("THE 3 IN dimensions = {1, 2, 3}, and where this gate deliberately "
                "differs from the Triton one: there this shape is the ENVELOPE "
                "CONTROL, here it is a DISPATCH case. An axis-aligned block keeps "
                "the permittivity tensor diagonal, the composer selects both "
                "ordinary pairs exactly as in 2-D, and the envelope control moves "
                "to pml_3d — the SPHERE — whose subpixel averaging writes the "
                "off-diagonal rows the release genuinely does not cover."),
    },
    "folded_2d": {
        "arms": ["folded fused B/H pair", "folded fused D/E pair"],
        "pairs": 2, "reached_by": "release", "expect": "dispatch",
        "seam": ["B", "D"], "fills": ["fill_B", "fill_D"],
        "launches_per_step": (6, 4), "monitors": 1,
        "why": ("BOTH FOLDED PAIRS WITH THE MIRROR FILL IN BOTH FILL SLOTS. Rung "
                "6bM refuses a fold whose fills are on the array path while its "
                "curls are on kernels, so a folded case that does not carry a fill "
                "IN-LAUNCH measures a composition the ladder is supposed to refuse "
                "— and would look identical, byte for byte, to the one it allows. "
                "The fill slots are named in the evidence rather than assumed."),
    },
    "folded_dispersive_2d": {
        "arms": ["folded fused B/H pair",
                 "folded fused dispersive D/E pair"],
        "pairs": 2, "reached_by": "release", "expect": "dispatch",
        # BOTH SEAMS from 2026-09-12. The entry said "THE B SEAM ALONE: this case
        # gives its D seam away" and that was true while the folded dispersive
        # product was unreleased; it owns the D seam and is released now, so the
        # seam fuses. The 2026-09-12 edit added the "B", "D" spelling and left the
        # old single-seam literal on the line above it — a DUPLICATE KEY, which
        # Python resolves silently to the last one, so the right answer was reached
        # by luck with dead evidence sitting beside it. One key now.
        "seam": ["B", "D"], "fills": ["fill_B", "fill_D"],
        "launches_per_step": (11, 9), "monitors": 1,
        "why": ("BOTH FOLDED PAIRS OVER A SUSCEPTIBILITY, update_P live under all "
                "of it. `folded fused D/E pair` pins susceptibilities at 0 because "
                "this case gives its D seam away: the folded DISPERSIVE product "
                "wins that seam, and from 2026-09-12 it is RELEASED on the "
                "tranche-7 group weld, so the seam FUSES rather than running "
                "separate arms. Launches drop from 10 to 9.\n\n"
                "THE ADMITTED ASYMMETRY: the folded dispersive arm leaves "
                "`susceptibilities` unpinned, so it is ADMITTED on folded_2d, which "
                "carries none. Measured 2026-09-12: the composer does not SELECT it "
                "there — folded_2d still composes `folded fused D/E pair` on both D "
                "slots — so admitting it changes nothing that runs. It is the same "
                "shape the ordinary dispersive pair already has on pml_2d."),
    },
    "cylindrical": {
        "arms": ["cylindrical m=0 fused electric D/E pair",
                 "cylindrical m=0 fused magnetic B/H pair"],
        "pairs": 2, "reached_by": "release", "expect": "dispatch",
        # BOTH SEAMS SINCE 2026-09-17. This read `seam: ["D"]` with a comment saying
        # the B seam stays on separate `cylindrical m=0` arms; that was true while
        # cylindrical_real_fused_magnetic_pair held no absorb row, and the all-paths
        # batch gave it one. launches_per_step is dropped rather than re-guessed: the
        # recorded (6, 5) was the one-pair composition's.
        "seam": ["B", "D"], "monitors": 0,
        "why": ("THE Dcyl m = 0 SHAPE, and the reason `cylindrical` is a PER-ARM "
                "axis rather than an implication of dimensionality: a Dcyl lift "
                "reports dimensions 2, so a release keyed on dimensionality alone "
                "would admit a coordinate system no gate ran. The B seam stays on "
                "separate `cylindrical m=0` arms — that family has no "
                "FUSED_PAIR_ARMS row — so one pair, not two."),
    },
    "special_kz_2d": {
        "arms": ["beta fused electric D/E pair"],
        "pairs": 1, "reached_by": "release", "expect": "dispatch",
        "seam": ["D"], "launches_per_step": (4, 3), "monitors": 1,
        "why": ("THE BETA ELECTRIC PAIR, RELEASED 2026-09-12. The composer selects "
                "`beta fused electric D/E pair` on this grid and the release now "
                "admits it: the family cites the tranche-7 GROUP weld, and the arm "
                "pins beta at the ONE value this case drives (0.4) rather than at "
                "any beta. The B seam keeps `special_kz real beta` on both slots, "
                "so this is a one-seam case and the run must still be "
                "byte-identical to the array path.\n\n"
                "WHAT THIS CASE STOPPED BEING, recorded because the gate lost a "
                "control here. Until 2026-09-12 it was the ONLY case whose expect "
                "was `no-fused-arm` — a product selected and nothing admitted, the "
                "cleanest statement of what the offer does. ENVELOPE_CASE is "
                "that control now — `no_pml_2d`, where the SHARED release envelope "
                "refuses on pml_active — and it is a DURABLE one where this case "
                "was not: it rests on a shared row no family release can erode, "
                "while this case rested on a family merely being unreleased, which "
                "is exactly what stopped being true.\n\n"
                "A BFAST CASE WAS BUILT FOR THIS ROLE AND DROPPED, recorded so it "
                "is not rebuilt blind. A pml_2d grid with a nonzero "
                "bfast_scaled_k isolates the refusal to ONE axis (measured: shared "
                "refusals = 1, naming bfast) and would have been the cleaner "
                "witness. It cannot serve: the Metal BFAST fused pairs are refused "
                "by the composer for want of an absorb declaration "
                "(`bfast_fused_magnetic_pair has no absorb declaration`), so "
                "nothing fuses there and the offer direction would have passed "
                "vacuously. Revisit if those pairs ever gain one."),
    },
    "bloch_2d": {
        "arms": ["complex fused magnetic B/H pair",
                 "complex fused electric D/E pair"],
        "pairs": 2, "reached_by": "release", "expect": "dispatch",
        "seam": ["B", "D"], "launches_per_step": (4, 2), "monitors": 1,
        "needs_probe": "MEEP_GPU_METAL_COMPLEX_EXPANSION_PROBE",
        "why": ("COMPLEX STORAGE UNDER A NONZERO k_point, and a case whose answer "
                "depends on an input outside the tree. WITHOUT the complex "
                "expansion licence the complex arms refuse by name and the whole "
                "case falls to the array path; WITH it BOTH complex pairs "
                "dispatch. The electric pair was selected-but-withheld until later "
                "on 2026-09-12 (its cell is bound by the tranche-7 GROUP weld and "
                "the group-key ruling reached it last), so the drop was 4 -> 3 "
                "then and is 4 -> 2 now, measured by the dry-run probe."),
    },
    "complex_nobloch_2d": {
        "arms": ["complex fused magnetic B/H pair",
                 "complex fused electric D/E pair"],
        "pairs": 2, "reached_by": "release", "expect": "dispatch",
        "seam": ["B", "D"], "launches_per_step": (4, 2), "monitors": 1,
        "needs_probe": "MEEP_GPU_METAL_COMPLEX_EXPANSION_PROBE",
        "why": ("COMPLEX STORAGE WITH NO k_point AT ALL (force_complex_fields), the "
                "shape of wvg-src.py and the two waveguide-source tests. It is the "
                "case that lets the complex pairs' rows DROP their bloch row: "
                "bloch_2d drives True and this case drives False, and every other "
                "axis of those rows is pinned to one value, so both values were "
                "driven at one corner rather than at two. Same licence as "
                "bloch_2d, same composition (both pairs, 4 -> 2), byte-identical "
                "by the dry-run probe."),
    },
    "pml_3d": {
        "arms": ["fused magnetic B/H pair"],
        "pairs": 1, "reached_by": "release", "expect": "dispatch",
        "seam": ["B"], "launches_per_step": (4, 3), "monitors": 1,
        "why": ("THE OFF-DIAGONAL GRID, DISPATCHING FROM 2026-09-12. A sphere, "
                "whose subpixel averaging writes off-diagonal chi1inv rows: "
                "measured, this case's run shape reads off_diagonal_epsilon=True. "
                "It was the ONLY case on this table that did until offdiag_2d and "
                "folded_offdiag_2d joined it on 2026-09-12.\n\n"
                "IT WAS THE ENVELOPE CONTROL UNTIL 2026-09-12, and it stopped "
                "being one because the axis moved. While off_diagonal_epsilon was "
                "a SHARED envelope row the released set here was empty and an "
                "opt-in was the only route to a fused label. The row is per-arm "
                "now: the ordinary magnetic pair is admitted inside its "
                "off-diagonal CORNER (metal_dispatch.METAL_OFF_DIAGONAL_CORNERS) and "
                "every other arm is refused BY NAME — an arm with no corner is "
                "refused on any off-diagonal grid, and the folded pair, which has "
                "one, is refused on the sphere by its own dimensions row. Measured "
                "2026-09-12: admitted = exactly ['fused magnetic B/H pair'], every "
                "other arm refused (10 of 11 that morning; the released set has "
                "grown since and the admitted set here has not).\n\n"
                "WHAT DISPATCHES: the magnetic pair takes step_B/update_H; the D "
                "seam keeps `PML` and `offdiag` singles, because no D-seam product "
                "is admitted on an off-diagonal grid. One seam, 4 -> 3.\n\n"
                "WHY IT MATTERS MORE THAN ITS ONE ROW: the off-diagonal refusals "
                "were the biggest single block of served instances the Metal "
                "release did not credit, and the clause that said none COULD be "
                "credited was false — it rested on a specialized-family guard that "
                "exists only in the Triton composer. See "
                "metal_dispatch.METAL_FUSED_RELEASE_ENVELOPE's note."),
    },
    # --- THE ALL-PATHS ROUND'S ELEVEN NEW CASES, 2026-09-17. Every ``arms`` set and
    # every ``pairs`` count below was MEASURED by probe_metal_dispatch_dryrun.py
    # before the row was typed -- the composer's own ``composition.selected``, joined
    # to fastpath._run_shape on the same lift -- rather than inherited from an
    # analogous row and confirmed later. ``launches_per_step`` is still OMITTED: the
    # fused-pair install refuses on a NumPy array module, so the dry run counts
    # dispatches and not launches, and a typed number nobody measured is worse than
    # an absent one.
    #
    # THE BUILDERS ARE INHERITED from gate_dispatch_fused_route: ``CASES`` starts as a
    # copy of that module's, and thirteen of the twenty-nine rows this table already
    # carried resolve that way. The home rule applies to builders this round
    # CONSTRUCTS, and this round constructs none.
    "complex_no_pml_3d": {
        "arms": ["fused complex conductive no-PML curl -> complex stored E"],
        "pairs": 1, "reached_by": "release", "expect": "dispatch",
        "seam": ["D"],
        # THE ONE ROW WHOSE PAIR COLLAPSES A PER-COMPONENT SINGLE. Measured 2026-09-19
        # on dispatch_metal_route_2026-09-19_witness/shipped_expansion_probe, the
        # case's first substitution measurement (it read DID-NOT-FUSE on 09-17): over
        # 600 steps the unfused leg's `complex no-PML stored E` booked 1800 launches on
        # update_E -- ONCE PER COMPONENT, 3 a step -- beside step_D's `complex
        # conductive no-PML curl` at 600, and the fused leg's `fused complex
        # conductive no-PML curl -> complex stored E` booked 600 on step_D with
        # update_E held by the pair. Four launches became one: 8.0 -> 5.0 a step,
        # drop 3.0 with the function-call drop agreeing, the slot sets identical and
        # the final state byte-identical to the array path (44 arrays, 0 differing).
        # The proof expected 1 and read DROPPED-BUT-NOT-EXACT on a correct result.
        # See :func:`expected_drop_per_step` for the rule and what keeps it honest.
        "collapsed_single_launches_per_step": {"update_E": 3},
        "needs_probe": "MEEP_GPU_METAL_COMPLEX_EXPANSION_PROBE",
        "why": ("THE COMPLEX CONDUCTIVE NO-ABSORBER WELD, on the shape the four "
                "TestLoadDump 3-D rows carry. Lifted 2026-09-17: dimensions 3, grid "
                "(23, 21, 27), complex storage, bloch True at k_point "
                "(0.4, -1.3, 0.7), conductivity True, ONE pole, pml_active False. "
                "The family was certified and welded before this round and held no "
                "launch.FUSED_PAIR_ARMS row, so the seam loop could not ask it; the "
                "row landed in the all-paths batch. ONE pair: the B seam keeps "
                "`no-PML null` (nothing to do on a cell with no boundary layer) and "
                "update_P keeps `ADE update_P` beside the weld. Its four corpus rows "
                "are MAGNETIC-sourced, which is the polarity that makes them "
                "reachable at all -- an electric source sits inside this seam."),
    },
    "complex_no_pml_offdiag": {
        "arms": ["complex no-PML off-diagonal fused electric D/E pair"],
        "pairs": 1, "reached_by": "release", "expect": "dispatch",
        "seam": ["D"],
        "needs_probe": "MEEP_GPU_METAL_COMPLEX_EXPANSION_PROBE",
        "why": ("THE COMPLEX UNFOLDED OFF-DIAGONAL WELD. Lifted 2026-09-17: "
                "dimensions 2, grid (25, 25, 1), complex storage, bloch True at "
                "k_point (0.3892, 0.1597, 0), off_diagonal True, lossless, "
                "pml_active False. Both this case and complex_no_pml_3d above read "
                "SELECTED-NOTHING on the first dry-run pass and the composer said "
                "why in both halves' words -- 'no complex-multiply expansion probe "
                "artifact is available ... which arm the numpy reference takes is a "
                "measured platform fact and may not be guessed' -- which is exactly "
                "what the probe leg exists to carry, and why both rows name it."),
    },
    "offdiag_magnetic_2d": {
        "arms": ["fused magnetic B/H pair", "off-diagonal fused electric D/E pair"],
        "pairs": 2, "reached_by": "release", "expect": "dispatch",
        "seam": ["B", "D"],
        "why": ("THE OFF-DIAGONAL WELD'S OWN CASE, and why it is a NEW case rather "
                "than the `offdiag_2d` this table already drives. Measured "
                "2026-09-17: on offdiag_2d and pml_3d the D seam keeps the separate "
                "`PML` and `offdiag` arms, and the composer says why -- 'source 0 "
                "(GaussianPulsedSource) is electric: the driver injects it BETWEEN "
                "step_D and update_E (driver.py:3294-3299) and "
                "deposit_repair.repairable REFUSES an off-diagonal chi1inv "
                "constitutive, so no repair can carry it'. A point repair recomputes "
                "E at the deposit cell from THAT cell's displacement; an off-diagonal "
                "constitutive reads its neighbours'. The magnetic-source variant "
                "moves the injection into the B seam, where this weld does not sit, "
                "and both the Triton and CUDA tables release their twins on exactly "
                "this case for exactly this reason. Lifted 2026-09-17: dimensions 2, "
                "grid (200, 120, 1), off_diagonal True, lossless, no susceptibility, "
                "real storage, PML active. TWO pairs: the ordinary magnetic pair "
                "keeps step_B/update_H beside it."),
    },
    "folded_offdiag_magnetic_2d": {
        "arms": ["folded fused B/H pair",
                 "folded off-diagonal fused electric D/E pair"],
        "pairs": 2, "reached_by": "release", "expect": "dispatch",
        "seam": ["B", "D"], "fills": ["fill_B", "fill_D"],
        "why": ("THE FOLDED TWIN OF THE ROW ABOVE, and the same source polarity is "
                "what makes it reachable. Lifted 2026-09-17: dimensions 2, grid "
                "(160, 61, 1), folded 'mirror plane on Y', off_diagonal True, "
                "lossless, real storage, PML active. Measured selection: the folded "
                "magnetic pair on step_B/update_H and the folded off-diagonal weld on "
                "step_D/update_E, with the mirror-fill arm in the fill slots. This "
                "case also widens `folded fused B/H pair`'s own case list, inside "
                "the corner that arm already declares (dimensions 2, zero poles)."),
    },
    "cylindrical_m0": {
        "arms": ["cylindrical m=0 fused magnetic B/H pair",
                 "cylindrical m=0 fused electric D/E pair"],
        "pairs": 2, "reached_by": "release", "expect": "dispatch",
        "seam": ["B", "D"],
        "why": ("THE REAL Dcyl CELL WITH BOTH SEAMS FUSED. `cylindrical` drives the "
                "electric pair alone and always has; what changed on 2026-09-17 is "
                "that `cylindrical_real_fused_magnetic_pair` gained a "
                "launch.FUSED_PAIR_ARMS row, so the seam loop can ask it. Measured "
                "on this case and not on `cylindrical`: the two cells differ in the "
                "source the B seam sees, and a row typed from the wrong one would "
                "claim a pair the campaign then fails to find. Lifted 2026-09-17: "
                "cylindrical True, dimensions 2, grid (80, 1, 80), real storage, no "
                "k_point, lossless, no susceptibility."),
    },
    "no_pml_dispersive_2d": {
        "arms": ["no-PML fused electric D/E pair"],
        "pairs": 1, "reached_by": "release", "expect": "dispatch",
        "seam": ["D"],
        "why": ("THE NO-ABSORBER STORED-E PAIR. Lifted 2026-09-17: dimensions 2, "
                "grid (120, 120, 1), pml_active False, TWO susceptibility states, "
                "lossless, real storage. ONE pair: the B seam keeps `no-PML curl` "
                "and `no-PML null`, which is what a cell with no boundary layer and "
                "no magnetic constitutive work looks like, and update_P keeps `ADE "
                "update_P` beside the weld."),
    },
    "absorber_1d": {
        "arms": ["conductive no-PML fused electric D/E pair"],
        "pairs": 1, "reached_by": "release", "expect": "dispatch",
        "seam": ["D"],
        "why": ("THE CONDUCTIVE NO-ABSORBER PAIR, on the cell whose mp.Absorber is "
                "the conductivity. Lifted 2026-09-17: dimensions 1, grid "
                "(1, 1, 400), pml_active False, conductivity True, and FIVE "
                "susceptibility states -- MEEP expresses an Absorber's conductivity "
                "profile as susceptibility states, which is why this arm's row pins "
                "five rather than zero and why an omitted count axis here would "
                "admit every pole count."),
    },
    "bfast_1d": {
        "arms": ["BFAST fused electric D/E pair"],
        "pairs": 1, "reached_by": "release", "expect": "dispatch",
        "seam": ["D"],
        "why": ("THE BFAST ELECTRIC PAIR. Lifted 2026-09-17: dimensions 3, grid "
                "(1, 1, 250), bfast True, real storage, lossless, PML active. ONE "
                "pair, and the B seam is the reason: `bfast_fused_magnetic_pair` "
                "holds no launch.FUSED_PAIR_ARMS row, deliberately -- it is one of "
                "the three the all-paths round measured winning its slot and then "
                "left unwired, because routing it empties the population "
                "test_metal_fused_pair_deposit_wiring sweeps for the "
                "refused-by-name property. So the B seam keeps the separate `BFAST` "
                "arms and this case stays a one-seam row."),
    },
    "complex_beta_2d": {
        "arms": ["complex-beta fused magnetic B/H pair",
                 "complex-beta fused electric D/E pair"],
        "pairs": 2, "reached_by": "release", "expect": "dispatch",
        "seam": ["B", "D"],
        "needs_probe": "MEEP_GPU_METAL_EXPANSION_PROBE",
        "why": ("THE COMPLEX BETA PAIRS AT bloch=False. Lifted 2026-09-17: "
                "dimensions 2, grid (120, 120, 1), complex storage, beta 0.4, "
                "bloch False, lossless, PML active. The licence is special_kz's "
                "rather than the base complex one -- both families call "
                "`special_kz.load_expansion_probe()` -- so this case rides the probe "
                "leg on MEEP_GPU_METAL_EXPANSION_PROBE and refuses by name on the "
                "shipped leg."),
    },
    "complex_beta_bloch_2d": {
        "arms": ["complex-beta fused magnetic B/H pair",
                 "complex-beta fused electric D/E pair"],
        "pairs": 2, "reached_by": "release", "expect": "dispatch",
        "seam": ["B", "D"],
        "needs_probe": "MEEP_GPU_METAL_EXPANSION_PROBE",
        "why": ("THE bloch=True VALUE OF THE SAME TWO ARMS, on the same cell. Both "
                "values are driven because both arms' rows pin `bloch` as the "
                "enumeration {False, True}, and an axis that named a value no case "
                "drove would be the widening-without-evidence this table refuses. "
                "Lifted 2026-09-17: the same shape as complex_beta_2d with bloch "
                "True."),
    },
    "folded_complex_beta_2d": {
        "arms": ["folded beta complex fused B/H pair",
                 "folded beta complex fused D/E pair"],
        "pairs": 2, "reached_by": "release", "expect": "dispatch",
        "seam": ["B", "D"], "fills": ["fill_B", "fill_D"],
        "needs_probe": "MEEP_GPU_METAL_FOLDED_COMPLEX_EXPANSION_PROBE",
        "why": ("THE FOLDED BETA-COMPLEX PAIRS AT bloch=False. Lifted 2026-09-17: "
                "dimensions 2, grid (120, 62, 1), folded 'mirror plane on Y', "
                "complex storage, beta 0.4, bloch False, lossless, PML active. This "
                "family reads TWO licences -- folded_complex's and special_kz's "
                "(folded_beta_complex_fused_pair.py:522-528) -- and the folded one is "
                "named here because it is the first the case would fail on; the "
                "probe leg exports all four together."),
    },
    "folded_complex_beta_bloch_2d": {
        "arms": ["folded beta complex fused B/H pair",
                 "folded beta complex fused D/E pair"],
        "pairs": 2, "reached_by": "release", "expect": "dispatch",
        "seam": ["B", "D"], "fills": ["fill_B", "fill_D"],
        "needs_probe": "MEEP_GPU_METAL_FOLDED_COMPLEX_EXPANSION_PROBE",
        "why": ("THE bloch=True VALUE OF THE SAME TWO ARMS, for the reason "
                "complex_beta_bloch_2d exists: both rows pin `bloch` as {False, "
                "True} and both values must be driven for that enumeration to rest "
                "on evidence."),
    },
    "folded_special_kz_2d": {
        "arms": ["folded beta real fused B/H pair",
                 "folded beta real fused D/E pair"],
        "pairs": 2, "reached_by": "release", "expect": "dispatch",
        "seam": ["B", "D"], "fills": ["fill_B", "fill_D"],
        "why": ("THE FOLDED REAL-BETA PAIRS. Lifted 2026-09-17: dimensions 2, grid "
                "(120, 62, 1), folded 'mirror plane on Y', REAL storage, beta 0.4, "
                "bloch False, lossless, PML active. No probe: real storage takes no "
                "expansion licence, which is the axis that separates these two arms "
                "from the folded beta COMPLEX pair on folded_complex_beta_2d. The "
                "run shape reads bloch False although the case carries a kz k_point, "
                "because a kz-only k_point on a 2-D cell folds into beta."),
    },
    "offdiag_2d": {
        "arms": ["fused magnetic B/H pair"],
        "pairs": 1, "reached_by": "release", "expect": "dispatch",
        "seam": ["B"], "launches_per_step": (4, 3), "monitors": 2,
        "why": ("THE 2-D OFF-DIAGONAL GRID, DISPATCHING FROM 2026-09-12. pml_2d's "
                "slab replaced by a cylinder: measured on the lift, the two run "
                "shapes differ in off_diagonal_epsilon and nothing else, same "
                "grid_shape (200, 120, 1). It is the case the retired shared "
                "envelope row named as the closing move — 'a 2-D off-diagonal "
                "case with an electric source outside the D seam' — and the "
                "seventeen served-but-refused instances on the Metal board behind "
                "'driven off-diagonal only at [3]' are its row.\n\n"
                "WHAT DISPATCHES, measured by the dry-run probe before this row "
                "was typed: the magnetic pair takes step_B/update_H; the D seam "
                "keeps `PML` and `offdiag` singles, exactly as pml_3d. Unfused 4, "
                "fused 3, byte-identical to 192 steps in both compositions.\n\n"
                "WHAT IT PROVES ABOUT THE TABLE: the ordinary magnetic pair's "
                "off-diagonal corner is (dimensions in {2, 3}, susceptibilities 0, "
                "conductivity False); this case is the dimensions=2 point of it."),
    },
    "folded_offdiag_2d": {
        "arms": ["folded fused B/H pair"],
        "pairs": 1, "reached_by": "release", "expect": "dispatch",
        "seam": ["B"], "fills": ["fill_B", "fill_D"],
        "launches_per_step": (6, 5), "monitors": 1,
        "why": ("THE FOLDED OFF-DIAGONAL GRID, DISPATCHING FROM 2026-09-12. "
                "folded_2d's slab replaced by a cylinder centred on the fold "
                "plane: measured on the lift, the two run shapes differ in "
                "off_diagonal_epsilon and nothing else, same grid_shape (160, 61, "
                "1). The twenty served-but-refused instances behind 'folded fused "
                "B/H pair: off_diagonal_epsilon=True' were the largest single "
                "block the Metal release did not credit; nineteen are this row's "
                "corner and one (absorbed_power_density.py, one Lorentzian pole) "
                "sits outside it and stays refused by name.\n\n"
                "WHAT DISPATCHES, measured by the dry-run probe: the mirror fill "
                "in both fill slots (so rung 6bM passes), the folded magnetic "
                "pair on step_B/update_H, and the D seam keeps `folded` and "
                "`folded offdiag` singles because the folded electric pair pins "
                "off_diagonal_epsilon False and no D-seam product is admitted on "
                "an off-diagonal grid. Unfused 6, fused 5, byte-identical to 192 "
                "steps in both compositions.\n\n"
                "WHY THE FOLDED ARM NEEDED A CORNER AND NOT AN OMISSION: its row "
                "leaves susceptibilities open because folded_2d and "
                "folded_dispersive_2d drive both values, but off-diagonal it was "
                "driven only at zero — the exact precondition failure the "
                "row-table's note describes. The corner records that; see "
                "metal_dispatch.METAL_OFF_DIAGONAL_CORNERS."),
    },
    "folded_complex_2d": {
        "arms": ["folded complex fused B/H pair",
                 "folded complex fused D/E pair"],
        "pairs": 2, "reached_by": "release", "expect": "dispatch",
        "seam": ["B", "D"], "fills": ["fill_B", "fill_D"],
        "launches_per_step": (8, 6), "monitors": 1,
        "needs_probe": "MEEP_GPU_METAL_FOLDED_COMPLEX_EXPANSION_PROBE",
        "why": ("A FOLD AND A NONZERO k_point AT ONCE: the folded complex fill "
                "covers both fill slots (so rung 6bM passes) and BOTH folded "
                "complex pairs take their seams. The D/E pair was "
                "selected-but-withheld until later on 2026-09-12 (tranche-7 group "
                "weld; the ruling reached it last), so the drop was 8 -> 7 then "
                "and is 8 -> 6 now, measured by the dry-run probe. Without the "
                "folded-complex licence both fill arms refuse and rung 6bM refuses "
                "the whole plan — which is the refusal this case also records."),
    },
    "folded_dispersive_3d": {
        "arms": ["folded fused B/H pair", "folded fused dispersive D/E pair"],
        "pairs": 2, "reached_by": "release", "expect": "dispatch",
        "seam": ["B", "D"], "fills": ["fill_B", "fill_D"], "monitors": 1,
        # TWO PAIRS SINCE 2026-09-17, and this row is the one the confirming pass
        # earned its place on. `folded fused dispersive D/E pair` was RELEASED on
        # folded_dispersive_2d alone while the composer had been holding step_D and
        # update_E here all along; the release row said one case, so this row said one
        # pair, and the two disagreed with the measurement in the same direction.
        # Widening the release row to all three folded dispersive cases -- with the
        # susceptibilities axis it forces, {1, 5} as an enumeration -- is what makes
        # the claim and the composition agree. launches_per_step is dropped: the
        # recorded (11, 10) was the one-pair reading.
        "why": ("THE FOLDED MAGNETIC PAIR AT 3-D OVER ONE LORENTZIAN POLE, update_P "
                "live under it. Its row omits susceptibilities because both values "
                "are driven, and folded_3d opening dimensions to {2, 3} would have "
                "left the (3-D, one pole) corner admitted on nothing — this case "
                "drives it. The D seam keeps `folded` + `folded dispersive PML E` "
                "singles: the folded dispersive D/E pair pins dimensions 2 and the "
                "folded D/E pair pins susceptibilities 0. 11 -> 10, byte-identical "
                "by the dry-run probe, offered exactly the released set."),
    },
    "folded_3d": {
        "arms": ["folded fused B/H pair", "folded fused D/E pair"],
        "pairs": 2, "reached_by": "release", "expect": "dispatch",
        "seam": ["B", "D"], "fills": ["fill_B", "fill_D"],
        "launches_per_step": (6, 4), "monitors": 1,
        "why": ("THE FOLDED PAIRS IN 3-D, on an axis-aligned block under one "
                "mirror plane. Measured by the dry-run probe before this row was "
                "typed: both folded pairs with the mirror fill in both fill slots, "
                "6 -> 4, byte-identical to the array path; the two-plane fold the "
                "five served 3-D rows carry (Y and Z) measured 8 -> 6 in the same "
                "composition. This row is what opens the folded pairs' "
                "dimensions row to {2, 3}; the folded B/H pair's off-diagonal "
                "corner therefore names dimensions=2, because its off-diagonal "
                "case is 2-D only."),
    },
    # -----------------------------------------------------------------------
    # THE NINE ROWS ADDED 2026-09-13, REAL STORAGE FIRST AND THEN COMPLEX, and
    # every one of them BEFORE the two Dcyl complex rows below. That order is a
    # correctness rule and not a style: a Dcyl COMPLEX case stepped earlier in a
    # process makes later real-storage Cartesian cases diverge from the array path
    # at subnormal magnitudes, on both dispatch legs, and the cause is open. Nothing
    # is inserted after `cylindrical_m1`.
    #
    # WHAT IS MEASURED IN THESE ROWS AND WHAT IS NOT. `arms` is READ: every builder
    # was lifted off-device on 2026-09-13 with its monitors and run through
    # `metal_dispatch.released_fused_arms_metal`, and the lists below are those
    # admitted sets verbatim. `pairs` and `seam` are PREDICTED from the named
    # analogous row, because the fused install refuses on a NumPy array module and
    # no composition can be read on the host that typed them; both are checked by
    # the ladder, so a wrong prediction is a loud failure and not a silent claim.
    # `launches_per_step` is OMITTED rather than guessed — the record reads it as an
    # empty list — and `probe_metal_dispatch_dryrun.py` is what supplies all three
    # before the campaign is trusted.
    "dispersive6_2d": {
        "arms": ["fused magnetic B/H pair", "fused dispersive D/E pair"],
        "pairs": 2, "reached_by": "release", "expect": "dispatch",
        "seam": ["B", "D"], "monitors": 1,
        "why": ("SIX SUSCEPTIBILITY STATES, and the pole count as the only axis "
                "that moves. case_dispersive_2d with its single Lorentzian "
                "replaced by six: measured on the lift 2026-09-13 the run shape "
                "reads susceptibilities 6 on a (90, 90, 1) grid, everything else "
                "as dispersive_2d. It is the 6 in the ordinary magnetic pair's "
                "enumerated {0, 1, 6} row, and the three stochastic_emitter rows "
                "are what that 6 serves.\n\n"
                "TWO ARMS BECAUSE THE RELEASE ADMITS TWO. `fused dispersive D/E "
                "pair` leaves susceptibilities UNPINNED — the recorded release decision "
                "on the three dispersive-adjacent rows — so it is admitted here "
                "beside the magnetic pair (measured on the lift). A row naming "
                "only the magnetic pair would hand the composer a label set "
                "narrower than the release, which is not what this table's `arms` "
                "field means.\n\n"
                "COMPOSITION INHERITED FROM dispersive_2d, NOT READ: both seams "
                "fused with update_P live under all of it. At six poles that slot "
                "holds six times the work it holds at one, so the launch counts "
                "are certainly NOT dispersive_2d's (9, 7) and are deliberately "
                "left out of this row rather than carried over. Read them with the "
                "dry-run probe before the campaign."),
    },
    "folded_dispersive5_2d": {
        "arms": ["folded fused B/H pair", "folded fused dispersive D/E pair"],
        "pairs": 2, "reached_by": "release", "expect": "dispatch",
        "seam": ["B", "D"], "fills": ["fill_B", "fill_D"], "monitors": 1,
        "why": ("FIVE SUSCEPTIBILITY STATES ON A FOLD. case_folded_dispersive_2d "
                "with five Lorentzians in place of its one: measured on the lift "
                "2026-09-13, susceptibilities 5 on an (80, 61, 1) grid under a "
                "mirror plane on Y, everything else identical. It is the 5 in the "
                "folded magnetic pair's enumerated {0, 1, 5} row, and the four "
                "folded TestLoadDump 2-D rows at grid (250, 126, 1) are what it "
                "serves.\n\n"
                "TWO ARMS, for dispersive6_2d's reason: `folded fused dispersive "
                "D/E pair` leaves susceptibilities unpinned and the release admits "
                "it here (measured on the lift).\n\n"
                "THE SAME SHAPE IS THE TRITON GATE'S ENVELOPE WITNESS, where the "
                "release DECLINES every folded arm. That is a fact about the "
                "Triton table and it does not reach this one — the two gates read "
                "their own releases — and it is the mirror of pml_3d_diagonal, "
                "which is an envelope control there and a dispatch case here.\n\n"
                "COMPOSITION INHERITED FROM folded_dispersive_2d, NOT READ: both "
                "folded pairs with the mirror fill in both fill slots so rung 6bM "
                "has something to pass, update_P live. The launch counts are again "
                "omitted rather than carried over, because five poles are not "
                "one."),
    },
    "complex_1d": {
        "arms": ["complex fused magnetic B/H pair",
                 "complex fused electric D/E pair"],
        "pairs": 2, "reached_by": "release", "expect": "dispatch",
        "seam": ["B", "D"], "monitors": 1,
        "needs_probe": "MEEP_GPU_METAL_COMPLEX_EXPANSION_PROBE",
        "why": ("COMPLEX STORAGE ON A CELL DECLARED 1-D: the 1 in the complex "
                "pairs' dimensions row, which this round opens from 2 to "
                "{1, 2, 3}. dimensions=1 is passed explicitly, so the case is "
                "about the axis rather than about the cell extents; measured on "
                "the lift 2026-09-13 the run shape reads dimensions 1, grid "
                "(1, 1, 480), complex storage, bloch True at k=(0, 0, 1.25), and "
                "both complex pairs admitted. It is refl-angular.py's shape, and "
                "TestPlanewave1D and TestReflectanceAngular_0_0 are served with "
                "it.\n\n"
                "THIS IS THE FIRST 1-D EVIDENCE EITHER ARM CARRIES, which is the "
                "reason it is a route case and not an inference. Both family gates "
                "build their complex fixtures through matrix.cart on a 3-D cell "
                "and neither has ever driven dimensions=1; the byte-identity leg "
                "through the real driver seam is what supplies it here, exactly as "
                "pml_1d does for the ordinary pairs.\n\n"
                "COMPOSITION INHERITED FROM pml_1d AND bloch_2d, NOT READ: both "
                "pairs on a (1, 1, N) grid, the electric Gaussian inside the D "
                "seam so step_D holds the leading repair. Probe leg only, as every "
                "complex case on this table."),
    },
    "complex_3d_thinline": {
        "arms": ["complex fused magnetic B/H pair",
                 "complex fused electric D/E pair"],
        "pairs": 2, "reached_by": "release", "expect": "dispatch",
        "seam": ["B", "D"], "monitors": 1,
        "needs_probe": "MEEP_GPU_METAL_COMPLEX_EXPANSION_PROBE",
        "why": ("A (0, 0, L) CELL DECLARED 3-D UNDER AN OBLIQUE k, which MEEP does "
                "not collapse: measured on the lift 2026-09-13, dimensions 3 on a "
                "(1, 1, 275) grid, complex storage, k=(0.41, 0, 1.8), both complex "
                "pairs admitted. Four corpus rows are physically one-dimensional "
                "and answer dimensions 3 — antenna_pec_ground_plane_1D.py, "
                "dipole_in_vacuum_1D.py, TestBoundaries1D and "
                "TestReflectanceAngular_1_20_6 — and this is their shape.\n\n"
                "IT IS HALF THE EVIDENCE FOR THE 3. The grid is one cell wide in "
                "two directions, so this case alone would license 3-D complex "
                "storage on nothing but thin lines. complex_3d below is the other "
                "half and is in the campaign for exactly that reason.\n\n"
                "COMPOSITION INHERITED FROM complex_1d, NOT READ."),
    },
    "complex_3d": {
        "arms": ["complex fused magnetic B/H pair",
                 "complex fused electric D/E pair"],
        "pairs": 2, "reached_by": "release", "expect": "dispatch",
        "seam": ["B", "D"], "monitors": 1,
        "needs_probe": "MEEP_GPU_METAL_COMPLEX_EXPANSION_PROBE",
        "why": ("A GENUINE 3-D COMPLEX BLOCK, AND IT CREDITS NOTHING. Measured on "
                "the lift 2026-09-13: dimensions 3, grid (40, 40, 40), complex "
                "storage, bloch True at k=(0.3, 0, 0), off_diagonal_epsilon False "
                "(the block is axis-aligned), both complex pairs admitted. The "
                "seven corpus rows the dimensions widening serves are ALL (1, 1, N) "
                "cells, so complex_1d and complex_3d_thinline between them would "
                "have licensed dimensions 3 on degenerate grids — while the census "
                "carries real 3-D complex rows the widened row admits at runtime "
                "(TestLoadDump's four 3-D shapes at (35, 32, 41) and "
                "TestMaterialGrid.test_matgrid_3d at (25, 25, 25)). Measured on "
                "the 2026-09-13 census none of those five is credited to these "
                "products, so the board number is identical with and without this "
                "case. It is here because a row that admits a shape no case drove "
                "is the defect this round exists to close, not because it moves "
                "the count.\n\n"
                "THE ONE THING THE DRY RUN MUST LOOK AT: the PML is on Z ALONE, so "
                "x and y stay periodic under the Bloch phase and the no-PML arms "
                "may own work the other complex cases give to the PML arms. `pairs` "
                "and `seam` are inherited from complex_nobloch_2d and are a "
                "prediction; if the slot set differs, this row is what the ladder "
                "will say so on."),
    },
    "folded_complex_3d": {
        "arms": ["folded complex fused B/H pair",
                 "folded complex fused D/E pair"],
        "pairs": 2, "reached_by": "release", "expect": "dispatch",
        "seam": ["B", "D"], "fills": ["fill_B", "fill_D"], "monitors": 1,
        "needs_probe": "MEEP_GPU_METAL_FOLDED_COMPLEX_EXPANSION_PROBE",
        "why": ("A FOLD, A NONZERO k_point AND THREE DIMENSIONS AT ONCE — the "
                "triangular-lattice-oblique cell, which is "
                "TestModeDecomposition.test_triangular_lattice_oblique's own "
                "shape. Measured on the lift 2026-09-13: dimensions 3, grid "
                "(8, 21, 126), folded on X, complex storage, k=(0, -1.7035, "
                "2.4694), beta 0, both folded complex pairs admitted. It opens "
                "those arms' dimensions row from 2 to {2, 3}.\n\n"
                "THE D/E HALF IS THIS ARM'S FIRST 3-D EVIDENCE EVER, and it is "
                "flagged rather than buried. `folded complex fused D/E pair` has "
                "no device gate file of its own — it binds to the tranche-7 GROUP "
                "weld, whose four folded-complex fixtures are all 2-D — while its "
                "magnetic twin is backed by eleven 3-D fixtures on its own family "
                "gate. If it is decided that a route case may not be an arm's "
                "first 3-D evidence, drop the D/E half: remove this case and "
                "folded_complex_kz2d_3d from that arm's row in "
                "METAL_RELEASED_FUSED_ARMS, re-pin its dimensions at 2, and the "
                "board loses two instances that no other lever in this batch also "
                "carries. The cheaper repair is one 3-D fixture in "
                "gate_metal_tranche7_fused_pairs.py.\n\n"
                "THE SOURCE SITS ON THE MIRROR PLANE, at x = 0. MEEP adds sources "
                "with use_symmetry=false, so a source in the half the plane "
                "discards lands in no stored chunk and the run returns an "
                "exactly-zero field while reading as complete; the lift refuses "
                "that by name rather than letting it through.\n\n"
                "COMPOSITION INHERITED FROM folded_complex_2d, NOT READ: both "
                "pairs with the folded-complex fill in both fill slots, which is "
                "what rung 6bM checks."),
    },
    "folded_complex_kz2d_3d": {
        "arms": ["folded complex fused B/H pair",
                 "folded complex fused D/E pair"],
        "pairs": 2, "reached_by": "release", "expect": "dispatch",
        "seam": ["B", "D"], "fills": ["fill_B", "fill_D"], "monitors": 1,
        "needs_probe": "MEEP_GPU_METAL_FOLDED_COMPLEX_EXPANSION_PROBE",
        "why": ("A FOLDED ZERO-THICKNESS CELL ASKED FOR AS 3-D. kz_2d='3d' is what "
                "keeps the z component of the k_point from lifting as a beta, so "
                "the declared 3 stands on a cell with no z extent; MEEP reports "
                "'Working in 3D dimensions' and the lift agrees. Measured "
                "2026-09-13: dimensions 3, grid (90, 62, 1), folded on Y, complex "
                "storage, k=(2.797, 0, -1.0849), beta 0, both folded complex pairs "
                "admitted. It is "
                "TestEigCoeffs.test_binary_grating_special_kz_2_21_2's shape.\n\n"
                "WHY BOTH THIS AND folded_complex_3d: they are the same axis at "
                "two very different builds — a volumetric fold and a sheet "
                "declared 3-D — and the two corpus rows behind the widening are "
                "one of each. A single case would leave the other's shape admitted "
                "on an inference.\n\n"
                "THE SOURCE IS AT y = +2.0, IN THE HALF THE MIRROR KEEPS, and this "
                "is the case that taught that rule: at y = -2.0 the lift refuses "
                "by name. Every folded case on this table now puts its sources in "
                "the kept half.\n\n"
                "The D/E half carries folded_complex_3d's caveat; composition "
                "inherited from folded_complex_2d, not read."),
    },
    "folded_complex_offdiag_2d": {
        "arms": ["folded complex fused B/H pair"],
        "pairs": 1, "reached_by": "release", "expect": "dispatch",
        "seam": ["B"], "fills": ["fill_B", "fill_D"], "monitors": 1,
        "needs_probe": "MEEP_GPU_METAL_FOLDED_COMPLEX_EXPANSION_PROBE",
        "why": ("THE FOLDED COMPLEX OFF-DIAGONAL GRID, BLOCH HALF. "
                "folded_complex_2d with its slab replaced by a cylinder centred on "
                "the fold plane — the same cell, PML, k_point, mirror and source — "
                "so the two run shapes differ in off_diagonal_epsilon ALONE. "
                "Measured on the lift 2026-09-13: dimensions 2, grid (80, 62, 1), "
                "folded on Y, complex storage, bloch True, off_diagonal_epsilon "
                "True, and EXACTLY ONE arm admitted — the folded complex magnetic "
                "pair.\n\n"
                "WHAT IT BUYS: `folded complex fused B/H pair` was ABSENT from "
                "metal_dispatch.METAL_OFF_DIAGONAL_CORNERS, and an arm absent from "
                "that table is refused on every off-diagonal grid. This case and "
                "its unphased twin are the corner entry's two halves, and the "
                "entry is the step at which this backend's dispatch count reaches "
                "its target. TestHoleyWvgBands.test_fields_at_kx is the row this "
                "half recovers on its own.\n\n"
                "ONE SEAM, AND THE REFUSAL BESIDE IT IS HALF THE MEASUREMENT: no "
                "D-seam product is admitted on an off-diagonal grid, so the D seam "
                "keeps its separate arms here exactly as on pml_3d, offdiag_2d and "
                "folded_offdiag_2d. Composition inherited from folded_offdiag_2d, "
                "not read."),
    },
    "folded_complex_nobloch_offdiag_2d": {
        "arms": ["folded complex fused B/H pair"],
        "pairs": 1, "reached_by": "release", "expect": "dispatch",
        "seam": ["B"], "fills": ["fill_B", "fill_D"], "monitors": 1,
        "needs_probe": "MEEP_GPU_METAL_FOLDED_COMPLEX_EXPANSION_PROBE",
        "why": ("THE FOLDED COMPLEX OFF-DIAGONAL GRID, UNPHASED HALF: two mirror "
                "planes and force_complex_fields with NO k_point at all, a "
                "cylinder for the off-diagonal chi1inv. Measured on the lift "
                "2026-09-13: dimensions 2, grid (81, 81, 1), folded on X and Y, "
                "complex storage, bloch False, off_diagonal_epsilon True, exactly "
                "one arm admitted. It is examples:solve-cw.py's and "
                "tests:TestArrayMetadata.test_array_metadata's shape.\n\n"
                "THE TWO HALVES PAY ONLY TOGETHER, measured on the 2026-09-13 "
                "census: those two rows are refused TWICE OVER, on bloch AND on "
                "off_diagonal_epsilon, so the corner entry without the bloch drop "
                "recovers one instance (fields_at_kx, the Bloch half's row) and "
                "the bloch drop without the corner entry recovers none. This case "
                "is what lets the folded complex MAGNETIC pair's row drop its "
                "bloch pin: with folded_complex_2d, folded_complex_3d, "
                "folded_complex_kz2d_3d and folded_complex_offdiag_2d driving "
                "True, it drives False.\n\n"
                "THE D/E TWIN KEEPS ITS PIN, and this case is why it must. The "
                "release does not admit that arm on an off-diagonal grid, so this "
                "run is not evidence about it at any k_point; dropping its pin on "
                "the magnetic twin's licence would claim a case that never drove "
                "it.\n\n"
                "THE SOURCE IS IN THE QUADRANT BOTH PLANES KEEP, at (+4, +4); at "
                "(-4, 0) the lift refuses. One seam, composition inherited from "
                "folded_offdiag_2d, not read."),
    },
    "cylindrical_m1": {
        "arms": ["cylindrical complex fused electric D/E pair",
                 "cylindrical complex fused magnetic B/H pair"],
        "pairs": 2, "reached_by": "release", "expect": "dispatch",
        "seam": ["B", "D"], "monitors": 1,
        "needs_probe": "MEEP_GPU_METAL_CYLINDRICAL_COMPLEX_EXPANSION_PROBE",
        "why": ("Dcyl AT m = 1: complex storage on a cylindrical grid, the shape of "
                "all eighteen cylindrical complex corpus rows (every one lossless, "
                "no k_point, no susceptibility). The cylindrical complex fused "
                "electric D/E pair takes the D seam — the Er source is inside it, "
                "so the deposit repair is live — and SINCE 2026-09-17 the B seam is "
                "taken by the cylindrical complex fused MAGNETIC B/H pair, which "
                "gained the absorb row it had been certified and welded without. "
                "This text read `the B seam keeps the cylindrical complex singles` "
                "until that row landed; the dry run measured the change on every "
                "Dcyl complex case. Byte-identical by the dry-run "
                "probe — and the same composition, byte-identical, at m = 2 and "
                "m = -1, which is what stands behind a row that cannot pin m (the "
                "census records no m, so a row would refuse every corpus row as "
                "unread; the arm is bounded on complex storage as its m = 0 twin "
                "is bounded on real). Licensed by the cylindrical-complex "
                "expansion probe, so it dispatches on the probe leg and is refused "
                "by name on the shipped leg."),
    },
    "cylindrical_m0_complex": {
        "arms": ["cylindrical complex fused electric D/E pair",
                 "cylindrical complex fused magnetic B/H pair"],
        "pairs": 2, "reached_by": "release", "expect": "dispatch",
        "seam": ["B", "D"], "monitors": 1,
        "needs_probe": "MEEP_GPU_METAL_CYLINDRICAL_COMPLEX_EXPANSION_PROBE",
        "why": ("Dcyl AT m = 0 UNDER COMPLEX STORAGE: the corner the 2026-09-12 "
                "round's verifier named. The arm's row cannot pin m (the census "
                "records none) and pins complex_storage True, so it admits the "
                "dipole_in_vacuum_cyl_off_axis shape -- m = 0 with "
                "force_complex_fields -- on evidence taken at m = 1 alone. This "
                "case drives that corner: the same composition as cylindrical_m1 "
                "(the complex electric pair on the D seam, the cylindrical complex "
                "singles on the B seam), 4 -> 3, byte-identical by the dry-run "
                "probe 2026-09-13 to 96 steps. Probe leg only, as its m = 1 twin.\n\n"
                "THE B SEAM AND THE PAIR COUNT MOVED ON 2026-09-17, and this row is "
                "the reason a wiring change has to re-confirm rows nobody edited. "
                "`cylindrical_complex_fused_magnetic_pair` gained a "
                "launch.FUSED_PAIR_ARMS row in the all-paths batch -- it was "
                "certified and welded before and the seam loop simply could not ask "
                "it -- so the composer now takes step_B/update_H with the cylindrical "
                "complex MAGNETIC pair and this is a two-seam, two-pair row. Measured "
                "on the 2026-09-17 dry run across every Dcyl complex case (m = 0, 2, "
                "3 and -1). `launches_per_step` is dropped rather than re-guessed: "
                "the recorded (4, 3) was the one-pair composition's."),
    },
}

#: THE TWO CASES :func:`envelope_leg` DRIVES, and why it takes two from 2026-09-12.
#:
#: IT USED TO TAKE ONE. ``pml_3d`` — the SPHERE, whose subpixel averaging writes
#: off-diagonal ``chi1inv`` rows — served both directions while
#: ``off_diagonal_epsilon`` was a SHARED envelope row: nothing was released
#: off-diagonal, so the shared table refused EVERY arm there, and an opt-in was the
#: only way to reach a fused label at all.
#:
#: THAT AXIS IS PER-ARM NOW and the ordinary magnetic pair IS released on the
#: sphere, so the shared table refuses nothing there. One case can no longer witness
#: both directions, and the honest split is:
#:
#: * :data:`ENVELOPE_CASE` — ``no_pml_2d``, where the SHARED table refuses, on
#:   ``pml_active``, and the released set is EMPTY. This is the direction that
#:   matters most and the one a single case would have silently lost: a shared
#:   predicate that has only ever said yes is not known to be able to say no.
#: * :data:`ENVELOPE_OFFER_CASE` — ``pml_3d``, where the shared table now PASSES,
#:   ten of eleven arms are refused by their OWN axes naming ``off_diagonal_epsilon``,
#:   and exactly the one arm driven off-diagonal is admitted. That is a richer
#:   statement than the opt-in it replaces, because the release route reaches it.
#:
#: ``no_pml_2d`` WAS CHOSEN OVER A NEW BFAST CASE, and the alternative is recorded
#: because it was built and measured before being dropped. A ``pml_2d`` grid with a
#: nonzero ``bfast_scaled_k`` isolates the refusal to ONE axis (measured: shared
#: refusals = 1, naming ``bfast``; every other axis a value the gate drove), which is
#: cleaner than ``no_pml_2d``. It was dropped anyway: it adds a case the gate must
#: drive ON DEVICE for a refusal ``no_pml_2d`` already witnesses, and the BFAST fused
#: pairs are refused by the composer for want of an absorb declaration
#: (``bfast_fused_magnetic_pair has no absorb declaration``), so it could not have
#: served the offer direction either. If the BFAST pairs ever gain that declaration,
#: this is the case to revisit.
ENVELOPE_CASE = "no_pml_2d"

#: The axis :data:`ENVELOPE_CASE`'s SHARED refusal must name. Read by
#: :func:`envelope_leg`; it was ``off_diagonal_epsilon`` until 2026-09-12, when that
#: axis stopped being a shared row.
ENVELOPE_AXIS = "pml_active"

#: The case the OFFER direction drives — see :data:`ENVELOPE_CASE` above.
ENVELOPE_OFFER_CASE = "pml_3d"
ENVELOPE_OPT_IN = ["fused magnetic B/H pair"]
ENVELOPE_UNNAMED = ["fused electric D/E pair"]

#: The no-fused-arm control: no absorber at all, so the no-PML arms own every slot
#: and no pair is installed even when every product is offered (measured: 41
#: "offered to run" refusals and zero fused slots).
NO_FUSED_ARM_CASE = "no_pml_2d"

#: How many complete steps every CONTROL runs. The house budget: a control that ran
#: one step could miss an error that needs the next step's neighbour reads.
CONTROL_STEPS = route.CONTROL_STEPS

#: The policy the harness installed, when it is NOT the one this table certifies
#: under. Set by :func:`main` from ``--install-policy``, and it inverts what a
#: dispatch case must DO: nothing may fuse, rung 8bM must refuse BY NAME, and the
#: run must still be byte-identical to the array path.
UNCERTIFIED_POLICY_INSTALLED: Optional[str] = None

#: The phrases rung 8bM's refusal is identified by, matched as a conjunction so a
#: refusal that moved to another rung is not absorbed by a substring that happens to
#: still appear.
POLICY_REFUSAL_NEEDLES: Tuple[str, ...] = ("float32", "subnormal policy")

#: The rungs of the Metal ladder that can refuse, and the phrases that identify
#: each. Read off ``metal_dispatch``'s own refusal text rather than predicted: a
#: refusal that silently moves from one rung to another is a change in what is
#: refused, and the artifact must show it rather than absorb it.
METAL_REFUSAL_RUNGS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("3-backend", ("backend is not CuPy",)),
    ("4M-toolchain", ("torch",)),
    ("6M-completeness", ("the composer filled",
                         "which the driver seam has no consult for")),
    ("6bM-fold", ("mirror-folded", "fill sub-steps")),
    ("7aM-pending", ("has no per-family weld",)),
    ("8M-fused-arm-not-admitted", ("a fused cross-sub-step product won",
                                   "has not been driven through the driver seam")),
    ("8bM-subnormal-policy", ("float32", "subnormal policy")),
    ("9M-no-slot-carries-a-kernel", ("no slot is left carrying a kernel",)),
)


def classify_refusal(reason: str) -> Optional[str]:
    """Which rung answered. ``None`` means a refusal no rung this gate knows produced."""
    for name, needles in METAL_REFUSAL_RUNGS:
        if all(needle in reason for needle in needles):
            return name
    return None


def ladder_for(total: int) -> List[int]:
    return route.ladder_for(total)


# ---------------------------------------------------------------------------
# Leg mechanics
# ---------------------------------------------------------------------------

def _as_list(value: Any) -> Any:
    """``list(...)`` a sequence, and leave anything else exactly as it is.

    THE SENTINEL IS A SENTENCE, NOT AN EMPTY LIST, and ``list()`` over a sentence is
    a list of characters. Before the run shape is reached, ``_base_record`` fills
    ``fusion.released_here`` / ``admitted`` / ``outside_the_released_envelope`` with
    the string "the run shape was not reached" — which is the right record, because
    "no arm was released here" and "the ladder never got far enough to ask" are
    different facts and only the second is true of a run refused at the backend rung.
    Splitting it made a control line read ``released=['t', 'h', 'e', ...]``, which is
    a reader's first impression of a broken gate rather than of a refused one.
    """
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return value


def _fuse_env(arms: Sequence[str],
              reached_by: str = "release") -> Dict[str, Optional[str]]:
    """The environment a fused leg runs under, and WHICH ROUTE reaches the arm.

    ``release`` is the shipped one: the switch UNSET, and
    ``metal_dispatch.METAL_RELEASED_FUSED_ARMS`` narrowed per arm by
    ``METAL_FUSED_RELEASE_ARM_AXES`` does the admitting. ``opt-in`` names the arms
    explicitly and is how a shape the envelope does not cover is reached at all.
    """
    if reached_by == "release":
        return {"MEEP_GPU_DISPATCH": "1", "MEEP_GPU_FUSED": None,
                "MEEP_GPU_FUSE_ARMS": None}
    return {"MEEP_GPU_DISPATCH": "1", "MEEP_GPU_FUSED": None,
            "MEEP_GPU_FUSE_ARMS": ",".join(arms) if arms else None}


ARRAY_ENV: Dict[str, Optional[str]] = {"MEEP_GPU_DISPATCH": "1",
                                       "MEEP_GPU_FUSED": "0",
                                       "MEEP_GPU_FUSE_ARMS": None}


def unfused_env() -> Dict[str, Optional[str]]:
    """The substitution baseline: every certified sub-step arm, NOTHING fused.

    THE VETO TOKEN IS READ FROM THE PACKAGE rather than spelled ``"0"`` here, so a
    gate cannot go on asking for a baseline the product stopped offering. Imported
    LATE, like everything else that touches ``meep_gpu`` in this file: a
    module-scope import would pull MEEP in before the harness has decided the
    subnormal policy, which is the one ordering rung 8bM's evidence depends on.
    """
    from meep_gpu import fastpath as fp  # noqa: PLC0415 - late by design

    return {"MEEP_GPU_DISPATCH": "1", "MEEP_GPU_FUSED": None,
            "MEEP_GPU_FUSE_ARMS": fp.FUSE_ARMS_VETO}


def step_leg(case: str, leg: Dict[str, Any], counter: MetalLaunchCounter,
             num_steps: int) -> None:
    """Advance one leg and keep the PER-CHUNK launch delta, not just the total.

    The per-chunk delta is the substitution proof's instrument. A total accumulated
    from the first chunk also contains everything the configuration freeze paid for,
    and a launches-per-step computed over that is the fusion's ratio plus a constant
    nobody can subtract afterwards. Deltas are per chunk for the second reason too:
    the legs are stepped ALTERNATELY, so a delta from one early origin also counts
    the other legs' launches.

    THE COUNTER IS ATTACHED AFTER THE FIRST CHUNK, and that is why the first chunk
    is excluded from the rate rather than merely noisy: the plan does not exist
    until the driver's first ``step()`` freezes the configuration, so there is
    nothing to wrap before it.
    """
    e2e._apply_env(leg)  # noqa: SLF001
    driver = leg["driver"]
    before = counter.snapshot()
    started = time.time()
    driver.run(num_steps=num_steps)
    elapsed = time.time() - started
    plan = getattr(driver, "_fast_path", None)
    if plan is not None and not leg.get("counter_attached"):
        seen = counter.attach(leg["label"], plan)
        leg["unwitnessed_owners"] = list(seen["unwitnessed_owners"])
        leg["non_kernel_leaves"] = list(seen["non_kernel_leaves"])
        leg["counter_attached"] = True
    chunk = MetalLaunchCounter.delta(before, counter.snapshot())
    running = leg.setdefault("metal_launches", {"total": 0, "function_calls": 0})
    running["total"] += chunk["total"]
    running["function_calls"] += chunk["function_calls"]
    leg.setdefault("chunks", []).append({
        "steps": int(num_steps),
        "launches": chunk["total"], "function_calls": chunk["function_calls"],
        "launches_per_step": (chunk["total"] / num_steps) if num_steps else None,
        "function_calls_per_step": ((chunk["function_calls"] / num_steps)
                                    if num_steps else None),
    })
    leg["active_step_path"] = driver.active_step_path
    leg["plan"] = e2e._plan_summary(driver.fast_path_report())  # noqa: SLF001
    leg["report"] = dict(driver.fast_path_report() or {})
    leg["steps"] = int(driver.step_count)
    leg["wall_s"] = round(leg.get("wall_s", 0.0) + elapsed, 3)
    say(f"{case}/{leg['label']}: path={leg['active_step_path']} steps={leg['steps']} "
        f"+{num_steps} launches+{chunk['total']} "
        f"({chunk['total'] / num_steps if num_steps else 0:.2f}/step, {elapsed:.1f} s)")


def steady_state_rate(leg: Dict[str, Any]) -> Dict[str, Any]:
    """Launches per complete step, over every chunk AFTER the first.

    The first chunk carries the configuration freeze and the counter attachment; a
    rate computed over it is not the run's rate. Every later chunk is steady state.
    """
    chunks = leg.get("chunks") or []
    tail = chunks[1:]
    steps = sum(c["steps"] for c in tail)
    if not steps:
        return {"measured": False, "why": "the leg ran a single chunk"}
    launches = sum(c["launches"] for c in tail)
    functions = sum(c["function_calls"] for c in tail)
    return {"measured": True, "steps": steps, "launches": launches,
            "function_calls": functions,
            "launches_per_step": launches / steps,
            "function_calls_per_step": functions / steps,
            "per_chunk": [c["launches_per_step"] for c in tail]}


def fusion_block(leg: Dict[str, Any]) -> Dict[str, Any]:
    """What the record says the fusion DID, read off the artifact rather than assumed."""
    report = leg.get("report") or {}
    fusion = dict(report.get("fusion") or {})
    slots = report.get("slots") or {}
    driven = dict(fusion.get("driven") or {})
    return {
        "table": report.get("table"),
        "opted_in": fusion.get("opted_in"),
        "driven": driven,
        "driven_slot_count": len(driven),
        "dispatched_slots": [n for n, e in slots.items()
                             if e.get("state") == "dispatched"],
        "arms": {n: e.get("arm") for n, e in slots.items()
                 if e.get("state") == "dispatched"},
        "counters": report.get("launch_counters") or {},
        "residency": report.get("residency") or report.get("residency_syncs"),
        "families": {arm: (report.get("families") or {}).get(arm, {}).get("gate")
                     for arm in set(driven.values())},
        "refused_because": report.get("refused_because"),
        "decision": report.get("decision"),
    }


def fused_leg_is_real(leg: Dict[str, Any], spec: Dict[str, Any]) -> Dict[str, Any]:
    """Every clause below is a FAILURE if false. Non-vacuity, said four ways."""
    failures: List[str] = []
    block = fusion_block(leg)
    driven = block["driven"]
    counters = block["counters"]

    if leg["active_step_path"] != "fused":
        failures.append(f"active_step_path reads {leg['active_step_path']!r}")
    if block["table"] != "metal":
        failures.append(f"the record's table reads {block['table']!r}, not 'metal'; "
                        "this leg is not evidence about the Metal table")
    if not driven:
        failures.append("the record names no driven fused arm")
    expected_slots = 2 * int(spec["pairs"])
    if len(driven) != expected_slots:
        failures.append(f"the record names {len(driven)} fused slots, expected "
                        f"{expected_slots} ({spec['pairs']} pair(s) x 2)")
    for slot, arm in sorted(driven.items()):
        entry = counters.get(slot) or {}
        if not entry.get("dispatches"):
            failures.append(f"{slot} carries {arm!r} but its consult ran "
                            f"{entry.get('dispatches')} times")
    # THE PER-PLAN WITNESS, and it is `plan_launches` rather than
    # `programs_per_dispatch`. A Metal KernelPlan spells no launch grid, so
    # `programs_per_dispatch` is None BY CONSTRUCTION on every Metal slot and
    # asserting on it would fail every leg while measuring nothing. What separates
    # "a kernel launched" from "a consult was answered" here is the plan's own
    # `launches`, which its `run` increments.
    leading = [s for s in driven if s in ("step_B", "step_D")]
    for slot in leading:
        entry = counters.get(slot) or {}
        if not entry.get("plan_launches"):
            failures.append(f"{slot} is the leading slot of a pair and its plan "
                            f"booked {entry.get('plan_launches')} launches; the "
                            "consult was answered by something that did not launch")
    if not leg["metal_launches"]["total"]:
        failures.append("no Metal plan booked a launch at all (KernelPlan.launches)")
    if not leg["metal_launches"]["function_calls"]:
        failures.append("no compiled Metal function was invoked (CountingFunction)")
    # THE TWO COUNTS MUST AGREE, which is what the class docstring always said they
    # did. Checked in STEADY STATE, because the witness attaches after the first chunk
    # and the totals above include that chunk's launches; and checked IN ADDITION to
    # the non-zero clauses rather than instead of them, because steady-state equality
    # alone passes at 0 == 0. Until 2026-09-18 only "non-zero over the whole leg" was
    # asked, and it passed the cylindrical case for as long as one unrelated
    # constitutive plan happened to be visible beside the unwitnessed curls.
    steady = steady_state_rate(leg)
    if not steady.get("measured"):
        failures.append("the leg ran a single chunk, so its steady-state launches and "
                        "compiled calls cannot be compared")
    elif not steady["function_calls"]:
        failures.append("no compiled Metal function was invoked in steady state "
                        "(CountingFunction)")
    elif steady["launches"] != steady["function_calls"]:
        failures.append(f"the plans booked {steady['launches']} launches in steady "
                        f"state and the compiled functions were invoked "
                        f"{steady['function_calls']} times; the two witnesses of one "
                        "event disagree")
    if leg.get("unwitnessed_owners"):
        failures.append(f"HARNESS: plan owner(s) {sorted(leg['unwitnessed_owners'])} "
                        "book launches but hold no function table the witness can "
                        "wrap, so their launches cannot be corroborated")
    if leg.get("non_kernel_leaves"):
        failures.append(f"HARNESS: function-table leaves that are not compiled Metal "
                        f"kernels were found and would be counted as launches: "
                        f"{sorted(leg['non_kernel_leaves'])[:6]}")
    # A FOLDED CASE THAT DOES NOT RUN A FILL IN-LAUNCH IS VACUOUS. Rung 6bM refuses
    # a fold whose fills are on the array path while its curls are not, so a "folded
    # case dispatched" that left both fills on the host would be measuring a
    # composition the ladder is supposed to refuse — and would look identical, byte
    # for byte, to the one it is supposed to allow.
    if spec.get("fills"):
        dispatched = set(block["dispatched_slots"])
        for slot in spec["fills"]:
            if slot not in dispatched:
                failures.append(f"{slot} is not dispatched, so the fold ran its "
                                "ghost fill on the array path while its curls ran "
                                "on kernels")
                continue
            entry = counters.get(slot) or {}
            if not entry.get("dispatches"):
                failures.append(f"{slot} is dispatched but its consult ran "
                                f"{entry.get('dispatches')} times")
        block["far_fill_consults"] = {
            name: (counters.get(name) or {}).get("dispatches")
            for name in ("fill_folded_far_ghosts_B", "fill_folded_far_ghosts_D")}
    return {"real": not failures, "failures": failures, **block}


def array_leg_is_clean(leg: Dict[str, Any]) -> Dict[str, Any]:
    """The oracle leg must BE the array path and must launch nothing.

    THE METAL SPELLING OF ``e2e.killswitch_leg_is_clean``, and it is a separate
    function rather than a call into that one because the launch witness differs:
    the Triton version reads ``leg["triton_launches"]``, a counter installed on
    Triton's own entry points, and this table has none — its two counts are
    ``KernelPlan.launches`` and the ``CountingFunction`` wrappers, both attached to
    the plans a leg holds.

    THE CLAUSE THAT MATTERS IS THE SECOND. "The oracle agreed with the dispatch
    leg" is worthless if the oracle also dispatched, and a kill-switch leg that
    launched a kernel is exactly that failure — the two legs would be the same
    composition compared with itself.
    """
    failures: List[str] = []
    if leg["active_step_path"] != "array":
        failures.append(f"active_step_path reads {leg['active_step_path']!r}")
    launches = leg.get("metal_launches") or {}
    if launches.get("total"):
        failures.append(f"{launches['total']} Metal launches on a leg that must "
                        "launch none")
    if launches.get("function_calls"):
        failures.append(f"{launches['function_calls']} compiled Metal function "
                        "calls on a leg that must make none")
    reason = (leg.get("plan") or {}).get("refused_because") or ""
    if "kill switch" not in reason:
        failures.append(f"the refusal does not name the kill switch: {reason!r}")
    return {"clean": not failures, "failures": failures, "refused_because": reason,
            "metal_launches": launches}


#: The drive-row key naming the replaced singles a fused pair COLLAPSES, as
#: ``{slot: N}``. See :func:`expected_drop_per_step`.
COLLAPSED_SINGLES_KEY = "collapsed_single_launches_per_step"


def expected_drop_per_step(spec: Dict[str, Any]) -> int:
    """How many launches a step a case's fused pairs save, from what the row DECLARES.

    THE RULE: ``pairs + sum over declared slots of (N - 1)``, N being the launches a
    step of the UNFUSED single on that slot, declared under
    :data:`COLLAPSED_SINGLES_KEY`. A slot the row does not declare contributes 0.

    "ONE LAUNCH FEWER PER PAIR" IS THE N = 1 INSTANCE OF IT, and it was the whole rule
    until 2026-09-19. It holds wherever the pair launches its replaced singles at
    the rate they launched unfused, which is every row but one: measured on
    ``dispatch_metal_route_2026-09-19_witness``, dispersive_2d's unfused
    ``update_E`` single books 6600 launches over 2200 steps (3 a step, once per
    component) and so does the fused dispersive D/E pair on step_D, and
    ``cylindrical``'s step_B books 3200 over 1600 on BOTH legs -- each pair still
    saves exactly one launch, and those rows are EXACT undeclared. What
    ``complex_no_pml_3d``'s pair does is different: its unfused ``update_E`` single
    launches 3 a step and the pair folds all three into its ONE launch, so four
    launches become one and the step saves 3. The declaration names that COLLAPSE,
    not merely a per-component single, and a row that declared a per-component
    single the pair does not collapse would expect a drop the run cannot show.

    DECLARED, NOT DETECTED, for the reason the NVIDIA route gate's
    ``expected_substitution`` gives for ``poles`` and ``tail_in_leading``: a clause
    that reads its own expectation off the measurement cannot fail. Two things keep
    the declaration non-vacuous. :func:`substitution_proof` cross-checks each
    declared N against the unfused leg's OWN per-slot counter and refuses a mismatch
    as ``DECLARATION-MISMATCH``; and the measured drop must still equal what this
    function returns, so a wrong declaration -- or a pair that stopped collapsing --
    reads ``DROPPED-BUT-NOT-EXACT``. An UNDECLARED collapse is caught the second way,
    which is exactly how this row was found.
    """
    declared = dict(spec.get(COLLAPSED_SINGLES_KEY) or {})
    return int(spec["pairs"]) + sum(int(n) - 1 for n in declared.values())


def _declaration_mismatches(fused: Dict[str, Any], unfused: Dict[str, Any],
                            declared: Dict[str, int], pairs: int = 1
                            ) -> Tuple[List[str], Dict[str, Any]]:
    """Each declared slot against the unfused leg's own per-slot counter.

    ``plan_launches`` is each plan's own ``KernelPlan.launches``, booked from the
    first step, so it is divided by the leg's TOTAL steps (every chunk, the first
    included) rather than by the steady-state tail: 1800 launches over 600 steps is
    3, and over the tail's 599 it is not an integer at all.

    A DECLARED SLOT NO PAIR REPLACED is a mismatch too: the declaration says a fused
    unit collapses that single, and a slot outside the fused leg's driven set is
    one nothing collapsed.
    """
    steps = sum(c["steps"] for c in (unfused.get("chunks") or []))
    counters = fusion_block(unfused)["counters"]
    driven = fusion_block(fused)["driven"]
    measured: Dict[str, Any] = {}
    failures: List[str] = []
    for slot, n in sorted(declared.items()):
        entry = counters.get(slot) or {}
        launches = entry.get("plan_launches")
        per_step = (launches / steps) if (launches is not None and steps) else None
        measured[slot] = {"declared": int(n), "unfused_plan_launches": launches,
                          "unfused_steps": steps, "unfused_per_step": per_step,
                          "unfused_arm": entry.get("arm"),
                          "replaced_by": driven.get(slot)}
        # N = 1 DECLARES NOTHING AND N < 1 IS FALSE. A declared 0 would lower the
        # expected drop below ``pairs`` and let a step that saved nothing read EXACT --
        # "no data" scoring as the best value -- so a collapse is at least two.
        if int(n) < 2:
            failures.append(f"{slot} is declared collapsed from {n} launches a step; "
                            f"a collapse folds at least two")
        if slot not in driven:
            failures.append(f"{slot} is declared collapsed ({n} a step) and no fused "
                            f"pair replaced it on the fused leg (driven "
                            f"{sorted(driven)})")
        if per_step is None or abs(per_step - int(n)) > 1e-9:
            failures.append(f"{slot} is declared at {n} launches a step and the "
                            f"unfused leg's own counter reads {launches} over "
                            f"{steps} steps ({per_step} a step, arm "
                            f"{entry.get('arm')!r})")
    if declared:
        # THE COLLAPSE IS READ ON THE FUSED SIDE TOO, not inferred from the total. A
        # declared row's expectation rests on two facts the chunk drop alone cannot
        # separate: the pair launched ONCE a step, and nothing outside the pairs moved.
        # A pair that did not collapse (3 a step) beside an unrelated slot that lost 2
        # would otherwise sum to the declared drop. Both are read off each leg's own
        # per-slot counters; on dispatch_metal_route_2026-09-19_witness no undriven slot
        # of any of 97 replayed rows changed between the legs, and complex_no_pml_3d's
        # pair books 600 launches over 600 steps on step_D.
        fused_steps = sum(c["steps"] for c in (fused.get("chunks") or []))
        fused_counters = fusion_block(fused)["counters"]
        pair_launches = [(fused_counters.get(slot) or {}).get("plan_launches")
                         for slot in sorted(driven)]
        booked = [n for n in pair_launches if n is not None]
        pair_per_step = (sum(booked) / fused_steps) if (booked and fused_steps) else None
        measured["fused_pairs_per_step"] = {"booked": booked, "steps": fused_steps,
                                            "per_step": pair_per_step,
                                            "pairs_declared": pairs}
        if pair_per_step is None or abs(pair_per_step - pairs) > 1e-9:
            failures.append(f"the fused leg's pairs book {booked} launches over "
                            f"{fused_steps} steps ({pair_per_step} a step), not the "
                            f"{pairs} a step a collapse into one launch per pair means")
        moved = {}
        for slot in sorted((set(counters) | set(fused_counters)) - set(driven)):
            u = (counters.get(slot) or {}).get("plan_launches")
            f = (fused_counters.get(slot) or {}).get("plan_launches")
            u_rate = (u / steps) if (u is not None and steps) else None
            f_rate = (f / fused_steps) if (f is not None and fused_steps) else None
            if u_rate is None or f_rate is None or abs(u_rate - f_rate) > 1e-9:
                moved[slot] = {"unfused_per_step": u_rate, "fused_per_step": f_rate}
        measured["undriven_slots_moved"] = moved
        if moved:
            failures.append(f"slots outside the pairs changed their launches between "
                            f"the legs ({moved}), so the drop is not the collapse's alone")
    return failures, measured


def substitution_proof(fused: Dict[str, Any], unfused: Dict[str, Any],
                       spec: Dict[str, Any]) -> Dict[str, Any]:
    """The launches the drive row declares a step saves, counted twice over.

    THE CLAIM FUSION MAKES, and the only one no byte comparison can reach: two
    sub-steps became one launch -- one fewer per pair per step, plus N - 1 for each
    single a pair collapses from N launches into its one
    (:func:`expected_drop_per_step`). The comparison is against a THIRD leg that
    dispatched the same slots with fusion off — not against the array path, which
    launches nothing and would make any number look like a saving.

    THE SLOT SETS MUST MATCH before the arithmetic means anything: if the unfused
    leg dispatched a different set, the delta mixes the fusion with a coverage
    difference and the claim is recorded as not-comparable rather than asserted.

    AND THE BASELINE MUST ACTUALLY BE UNFUSED, read out of ITS OWN RECORD rather
    than inferred from the environment it was handed — "the switch said no arms" and
    "the record shows no arm driven" are different claims and only the second is a
    baseline.
    """
    a, b = steady_state_rate(fused), steady_state_rate(unfused)
    # THE EXPECTATION IS THE ROW'S DECLARATION, recorded beside the number it
    # produced so a reader of the artifact can see which rows declared anything.
    declared = {slot: int(n) for slot, n in
                dict(spec.get(COLLAPSED_SINGLES_KEY) or {}).items()}
    expected_drop = expected_drop_per_step(spec)
    out: Dict[str, Any] = {"fused": a, "unfused": b,
                           "expected_drop_per_step": expected_drop,
                           "collapsed_single_launches_per_step": declared,
                           "measured_2026_09_10": list(spec.get("launches_per_step")
                                                       or ())}
    baseline = dict((unfused.get("report") or {}).get("fusion") or {})
    out["baseline_vetoed_fusion"] = baseline.get("vetoed")
    out["baseline_driven"] = dict(baseline.get("driven") or {})
    if not spec["pairs"]:
        # A CASE THAT IS HERE TO BE REFUSED SUBSTITUTES NOTHING, and reporting a
        # zero drop as EXACT would put a green word on a measurement never taken.
        out["verdict"] = "NOT-APPLICABLE"
        out["why"] = "this case expects no fused product, so no launch is substituted"
        return out
    if not (a.get("measured") and b.get("measured")):
        out["verdict"] = "NOT-MEASURED"
        return out
    fused_slots = sorted(fusion_block(fused)["dispatched_slots"])
    unfused_slots = sorted(fusion_block(unfused)["dispatched_slots"])
    out["fused_dispatched_slots"] = fused_slots
    out["unfused_dispatched_slots"] = unfused_slots
    out["same_slot_set"] = fused_slots == unfused_slots
    drop = b["launches_per_step"] - a["launches_per_step"]
    function_drop = b["function_calls_per_step"] - a["function_calls_per_step"]
    out["launch_drop_per_step"] = drop
    out["function_drop_per_step"] = function_drop
    # THE LEVELS AGREE, NOT ONLY THE DROPS. A drop of 2.0 against a function-call
    # drop of 2.0 read EXACT on the cylindrical case while the fused leg's compiled
    # calls stood at 0.0 per step: two blind witnesses agree on a difference.
    levels_agree = (a["launches"] == a["function_calls"]
                    and b["launches"] == b["function_calls"])
    out["levels_agree"] = levels_agree
    out["counts_agree"] = abs(drop - function_drop) < 1e-9 and levels_agree
    # THE DECLARATION IS CHECKED WHERE A SUBSTITUTION EXISTS TO CHECK IT AGAINST: a
    # fused leg that drove a pair. A leg that fused nothing -- a policy-refused leg,
    # a DID-NOT-FUSE -- has no pair to collapse anything and reads NO-DROP below,
    # which says the right thing; relabelling it a declaration error would not.
    #
    # ONLY DECLARED SLOTS ARE CROSS-CHECKED, and "an undeclared replaced single
    # launches once a step" is NOT asserted, because it is false on rows that are
    # correct: measured 2026-09-19, every dispatching leg of
    # dispatch_metal_route_2026-09-19_witness carries nine EXACT rows whose undeclared
    # driven slot launches more than once unfused -- the seven dispersive rows'
    # update_E at 3 a step (dispersive_2d 6600/2200, absorber_1d 9600/3200, ...) and
    # the two cylindrical rows' step_B/step_D at 2 (3200/1600) -- because their pairs
    # launch at that same rate. An undeclared COLLAPSE is still never silently EXACT:
    # it moves the drop off the declared expectation, which is the clause below.
    mismatches: List[str] = []
    if declared and fusion_block(fused)["driven"]:
        mismatches, out["declaration_check"] = _declaration_mismatches(
            fused, unfused, declared, pairs=int(spec["pairs"]))
    if out["baseline_driven"] or out["baseline_vetoed_fusion"] is not True:
        out["verdict"] = "BASELINE-FUSED"
        out["why"] = (f"the unfused leg's own record shows "
                      f"vetoed={out['baseline_vetoed_fusion']!r} and driven="
                      f"{out['baseline_driven']}, so the delta is between two fused "
                      "legs and measures nothing about substitution")
    elif not out["same_slot_set"]:
        out["verdict"] = "NOT-COMPARABLE"
        out["why"] = ("the two legs dispatched different slot sets, so the launch "
                      "delta mixes fusion with a coverage difference")
    elif mismatches:
        out["verdict"] = "DECLARATION-MISMATCH"
        out["why"] = ("the drive row's collapsed-single declaration disagrees with "
                      "the run, so the expected drop it produces is not about this "
                      "composition: " + "; ".join(mismatches))
    elif abs(drop - expected_drop) < 1e-9 and out["counts_agree"]:
        out["verdict"] = "EXACT"
    elif drop > 0:
        out["verdict"] = "DROPPED-BUT-NOT-EXACT"
        out["why"] = (f"the fused leg saved {drop} launches a step (compiled calls "
                      f"{function_drop}, levels_agree={levels_agree}) against the "
                      f"{expected_drop} the drive row declares ({spec['pairs']} "
                      f"pair(s), collapsed singles {declared or 'none'})")
    else:
        out["verdict"] = "NO-DROP"
    return out


# ---------------------------------------------------------------------------
# The residency controls
# ---------------------------------------------------------------------------

def _synced_plans(driver: Any) -> List[Any]:
    """Every ``launch.SyncedPlan`` sitting in this driver's frozen plan, in slot order.

    Walks one level into a wrapper's ``inner``, because ``launch.synced`` brackets
    the INNER launch of a ``LeadingRepairPlan``/``LeadingWithdrawPlan`` so the host
    work that wrapper does first stays first.
    """
    from meep_gpu.metal_kernels import launch  # noqa: PLC0415

    plan = getattr(driver, "_fast_path", None)
    step_plan = getattr(plan, "step_plan", None)
    plans = getattr(step_plan, "plans", None) or {}
    found: List[Any] = []
    for slot in getattr(plan, "slots", ()) or sorted(plans):
        entry = plans.get(slot)
        if isinstance(entry, launch.SyncedPlan):
            found.append(entry)
            continue
        inner = getattr(entry, "inner", None)
        if isinstance(inner, launch.SyncedPlan):
            found.append(inner)
    return found


def _drop_one_sync_in(driver: Any):
    """Suppress the sync_in of the LEADING bracketed launch of every step. MUST diverge.

    THE ARMED NULL FOR THE RESIDENCY SEAM, and the seam has no other witness. In
    per-launch-sync mode the residency VERDICT is not consulted for correctness —
    the bracket is what makes the host authoritative — so "the mirrors agreed" says
    nothing about whether the bracket is load-bearing. Removing one upload per step
    leaves the leading launch reading whatever the device held from the previous
    step, which every unconsulted host pass between the two steps has since
    contradicted. If the run does NOT diverge, either nothing launched or the host
    passes write nothing the kernel reads, and every clean result above it is
    unfalsifiable.

    Patched at the CLASS, keyed on the identity of ONE instance, so exactly one
    bracket per step is disarmed and the rest of the seam runs shipped.
    """
    from meep_gpu.metal_kernels import device as device_module  # noqa: PLC0415
    from meep_gpu.metal_kernels import launch  # noqa: PLC0415

    targets = _synced_plans(driver)
    state: Dict[str, Any] = {"suppressed": 0, "armed": bool(targets),
                             "slot": None, "steps_with_stale_mirrors": 0,
                             "stale_mirrors": {}}
    if not targets:
        return (lambda: None), state

    # UNDER A HOLD, SUPPRESSING A sync_in IS NOT THE NULL -- it can be a genuine
    # no-op. A held ``sync_in`` copies only the mirrors the host OWNS, so if the
    # bracket this control picks happens to have none, removing it changes nothing
    # and the control reads NULL-DID-NOT-DIVERGE on a seam that is perfectly
    # load-bearing. Measured on ``magnetic_seam_2d``, ``dispersive_2d`` and
    # ``pml_3d`` in the first held campaign.
    #
    # What IS load-bearing under a hold is the OWNERSHIP MOVE. So the held null
    # leaves ``acquire_write`` unsealing as usual -- otherwise the host write would
    # merely raise -- and suppresses only the transition to HOST_OWNED. The write
    # then lands on the host and the next ``sync_in`` skips it, because the machine
    # still believes the mirror clean. That is precisely the missed-write defect the
    # ownership machine exists to prevent, so the run MUST diverge.
    residency = getattr(targets[0], "residency", None)
    if residency is not None and getattr(residency, "held", frozenset()):
        real_acquire = type(residency).acquire_write
        state["mode"] = "held: one acquire_write leaves the mirror unclaimed"
        state["armed"] = True

        def patched_acquire(self, target):  # noqa: ANN001
            array = real_acquire(self, target)
            mirrors = self._targets(target)
            if mirrors and state["suppressed"] < 1:
                for mirror in mirrors:
                    mirror.state = device_module.CLEAN_BOTH
                state["suppressed"] += 1
                state["slot"] = mirrors[0].name
            return array

        type(residency).acquire_write = patched_acquire

        def restore_acquire() -> None:
            type(residency).acquire_write = real_acquire

        return restore_acquire, state
    target_id = id(targets[0])
    state["slot"] = getattr(targets[0], "slot", None) or type(
        getattr(targets[0], "inner", None)).__name__
    real = launch.SyncedPlan.run

    def patched(self, *args: Any, **kwargs: Any) -> None:  # noqa: ANN001
        if id(self) == target_id:
            # WAS THERE AN UPLOAD TO SUPPRESS? ``verify()`` reports, per mirror, how
            # many words the device copy differs from the host array by. Read at the
            # instant the upload would have run, an EMPTY verdict means the host has
            # written nothing the device does not already hold -- so skipping the
            # upload changes nothing and a matching pair of legs is a no-op rather
            # than a blind comparator. Counting the steps where it is NON-empty is
            # what makes a null attributable: measured 2026-09-11, four compositions
            # produced NULL-DID-NOT-DIVERGE and the row recorded nothing that could
            # tell "the bracket is not load-bearing" from "the control did not arm".
            try:
                stale = {name: count for name, count
                         in (self.residency.verify() or {}).items() if count}
            except Exception:  # noqa: BLE001 - an unreadable verdict is not a failure
                stale = {}
            if stale:
                state["steps_with_stale_mirrors"] += 1
                for name, count in stale.items():
                    state["stale_mirrors"][name] = max(
                        state["stale_mirrors"].get(name, 0), int(count))
            state["suppressed"] += 1
            self.inner.run(*args, **kwargs)
            self.residency.sync_out()
            return
        real(self, *args, **kwargs)

    launch.SyncedPlan.run = patched

    def restore() -> None:
        launch.SyncedPlan.run = real

    return restore, state


def _drop_the_inner_bracket(driver: Any):
    """Move a wrapper's bracket from its INNER launch to the whole SLOT. MUST diverge.

    THE PLACEMENT CONTROL. Two wrappers in this composition do HOST work before
    their launch — ``deposit_repair.LeadingRepairPlan`` saves the deposit points and
    ``withdraw_hoist.LeadingWithdrawPlan`` performs the seam's electric withdraw —
    and ``launch.synced`` therefore brackets the inner launch rather than the slot,
    so the launch reads the host state AFTER that work. Bracketing the whole slot
    instead uploads BEFORE the host work, so the launch reads a ``D`` still holding
    the previous step's standing dipole. That is the exact defect the hoist exists
    to prevent, and a shim that synchronised around the slot would report success.

    Returns ``armed: False`` when the case's plan carries no such wrapper, which is
    the honest answer for a clean seam rather than a pass.
    """
    from meep_gpu.metal_kernels import launch  # noqa: PLC0415

    plan = getattr(driver, "_fast_path", None)
    step_plan = getattr(plan, "step_plan", None)
    plans = getattr(step_plan, "plans", None) or {}
    state: Dict[str, Any] = {"armed": False, "slots": [], "read_only_slots": [],
                             "wrapper_kinds": {}}
    undo: List[Tuple[Any, str, Any]] = []
    for slot, entry in list(plans.items()):
        inner = getattr(entry, "inner", None)
        if not isinstance(inner, launch.SyncedPlan):
            continue
        kind = type(entry).__name__
        state["wrapper_kinds"][slot] = kind
        # A WRAPPER WHOSE PRE-LAUNCH HOST WORK WRITES NOTHING CANNOT BE MEASURED BY
        # THIS CONTROL, and saying so is the difference between a null and a result.
        # Moving an upload ACROSS a read-only operation cannot change a byte on
        # either side: ``deposit_repair.save`` gathers the deposit cells OUT -- through
        # an advanced index, or the plan's cached linear index into a C-order view,
        # each of which returns a NEW array -- and writes only into the dict it
        # returns and the plan's own ``_PreparedCells`` cache, never into a field
        # array, so a leading REPAIR slot yields a provable no-op. (Until 2026-09-19
        # this cited ``copy=True``; ``save`` no longer makes that second copy except
        # for a scalar-part index, ``deposit_repair._gathered``, and the contract never
        # rested on it.) The shape this control was written for is
        # ``withdraw_hoist.LeadingWithdrawPlan``, whose pre-launch work WRITES
        # (``source.withdraw(fields)``) -- and it installs on ``update_H``, which no
        # composition in this drive table dispatches. Measured 2026-09-11: 9 nulls
        # and 0 fires across the whole corpus, every armed slot a LeadingRepairPlan.
        if _pre_launch_host_work_is_read_only(entry):
            state["read_only_slots"].append(slot)
            continue
        residency = inner.residency
        raw = inner.inner
        undo.append((entry, slot, inner))
        entry.inner = raw                    # the host work loses its bracket...
        plans[slot] = launch.SyncedPlan(entry, residency)   # ...and the SLOT gains one
        state["armed"] = True
        state["slots"].append(slot)

    def restore() -> None:
        for wrapper, slot, bracketed in undo:
            wrapper.inner = bracketed
            plans[slot] = wrapper

    return restore, state


#: Wrapper classes whose pre-launch host work only READS the fields. Named by class
#: rather than probed, because "did this write" is not observable from outside the
#: call and a wrong guess here would silently disarm a control that can fire.
#: ``LeadingRepairPlan`` gathers the deposit cells into new arrays and writes only the
#: dict ``deposit_repair.save`` returns and the plan's own ``_PreparedCells`` cache --
#: never a field array; ``LeadingWithdrawPlan`` calls
#: ``source.withdraw(fields)``, which does write, and is the shape this control
#: exists for.
_READ_ONLY_PRE_LAUNCH_WRAPPERS = ("LeadingRepairPlan",)


def _pre_launch_host_work_is_read_only(wrapper: Any) -> bool:
    """True when moving an upload across this wrapper's host work cannot change bytes."""
    return type(wrapper).__name__ in _READ_ONLY_PRE_LAUNCH_WRAPPERS


def _verify_after_every_sync_out(driver: Any):
    """Read ``Residency.verify()`` IMMEDIATELY after each sync_out. The vacuity floor.

    THE INSTANT MATTERS AND THE MEASUREMENT PROVES IT. Read at the END of a step,
    ``verify()`` is NOT empty on a leg carrying a trailing deposit repair — measured
    2026-09-10 on ``bloch_2d``: ``{'Ez': 2, 'Dz': 2, 'f_w_Ez': 2}`` — because the
    repair is HOST work that runs after the last launch's copy-back and the
    invariant is host-authoritative, not mirror-equal. So a control asserting an
    empty verdict at the step boundary would be red on a correct seam, and a control
    that dropped the assertion would measure nothing.

    Read immediately after ``sync_out``, nothing has run since the copy-back and the
    two sides must agree exactly. That is the residency claim, at the one moment it
    is a claim.

    THE FLOOR IS THE REGISTRY: an empty ``verify()`` from an empty registry proves
    nothing, so ``residency.names`` is counted too.

    UNDER A HOLD THE CLAIM CHANGES AND SO DOES THE CHECK. The device owns the
    written set between launches, so the host is stale for those mirrors by design
    and blanket equality would be red on correct code. The held form asserts a SET
    equality instead -- the mirrors that differ are exactly the ones the bookkeeping
    calls device-owned -- which fails in both directions and so also catches a
    bracket that is still copying what the hold said it would keep. Do NOT narrow
    the check to the held names: the untransported mirrors are precisely the ones
    new code can corrupt.
    """
    from meep_gpu.metal_kernels import launch  # noqa: PLC0415

    real = launch.SyncedPlan.run
    state: Dict[str, Any] = {"reads": 0, "disagreements": [], "mirrors": 0,
                             "held_reads": 0, "over_synced": [], "under_synced": []}

    def patched(self, *args: Any, **kwargs: Any) -> None:  # noqa: ANN001
        real(self, *args, **kwargs)
        residency = self.residency
        state["mirrors"] = len(residency.names)
        verdict = residency.verify()
        state["reads"] += 1
        held = getattr(residency, "held", frozenset())
        if not held:
            # UNHELD: the claim is mirror-equality and blanket equality is the
            # right form. Unchanged from the control that has been running since
            # 2026-09-10.
            if verdict and len(state["disagreements"]) < 8:
                state["disagreements"].append(dict(verdict))
            return
        # HELD: THE SAME ASSERTION WOULD BE RED ON CORRECT CODE. The device owns the
        # written set between launches, so the host is stale for exactly those
        # mirrors BY DESIGN and a blanket "verify() is empty" check measures the
        # design rather than the implementation.
        #
        # The falsifiable form is a SET comparison, and it is strictly stronger than
        # equality because it fails in both directions: the mirrors that differ must
        # be exactly the ones the bookkeeping calls device-owned. A mirror that
        # differs while the machine believes it clean is a lost device write
        # (under_synced); a mirror the machine calls device-owned that does NOT
        # differ means the bracket copied something the hold said it would keep, so
        # the saving is not being taken (over_synced). Equality can see neither.
        state["held_reads"] += 1
        grouped = residency.by_state()
        expected = set(grouped.get("device_owned", ()))
        differing = set(verdict or ())
        under = sorted(differing - expected)
        over = sorted(expected - differing)
        if under and len(state["under_synced"]) < 8:
            state["under_synced"].append(
                {name: verdict[name] for name in under[:8]})
        if over and len(state["over_synced"]) < 8:
            state["over_synced"].append(over[:8])

    launch.SyncedPlan.run = patched

    def restore() -> None:
        launch.SyncedPlan.run = real

    return restore, state


# ---------------------------------------------------------------------------
# The subnormal census
# ---------------------------------------------------------------------------

def _census_state(driver: Any) -> Dict[str, int]:
    """How many float32 words of every stored array are in the subnormal band."""
    from meep_gpu.metal_kernels import subnormal  # noqa: PLC0415

    out: Dict[str, int] = {}
    for name, array in e2e.collect_state(driver).items():
        try:
            count = subnormal.census(array)
        except Exception:  # noqa: BLE001 - a dtype with no float32 word view is skipped
            continue
        if count:
            out[name] = count
    return out


def band_witness(case: str, res: Optional[int], steps: int) -> Dict[str, Any]:
    """Does the ORACLE's own trajectory enter the subnormal band? The vacuity floor.

    THE CLAIM UNDER TEST IS AN EQUIVALENCE — that flushing float32 denormals on both
    executors changes no byte of the answer — and an equivalence over a value class
    the run never constructs is vacuous. So the array leg is re-run with the host
    KEEPING and every stored array is censused every step: at least one dispatching
    case's window must be NON-EMPTY, or the flush-equivalence claim this gate
    licenses is about nothing.

    Run in the ``band_witness`` leg, which installs ``keep`` and dispatches nothing.

    "DISPATCHES NOTHING" IS CHECKED HERE, NOT ASSUMED. The driver is an Apple GPU
    driver like every other this gate lifts (:func:`_lift`), so its array path is the
    environment's doing: ``run_metal_dispatch_campaign.sh`` hands this leg
    ``MEEP_GPU_DISPATCH=0``. A witness that fused would census a trajectory stepped
    under the Metal table's installed flush, where the band is empty by
    construction, so a freeze that reads anything but ``"array"`` is refused by name.
    """
    driver, _monitors, _until = _lift(case, res)
    window: List[Dict[str, Any]] = []
    total = 0
    for step in range(1, steps + 1):
        driver.run(num_steps=1)
        if driver.active_step_path != "array":
            path = driver.active_step_path
            driver.close()
            raise RuntimeError(
                f"{case}: the band witness stepped {path!r} at step {step}; it "
                "censuses the ARRAY path with the host keeping, so run this leg "
                "with MEEP_GPU_DISPATCH=0 (run_metal_dispatch_campaign.sh hands it "
                "that value)")
        counts = _census_state(driver)
        if counts:
            total += sum(counts.values())
            if len(window) < 12:
                window.append({"step": step, "arrays": len(counts),
                               "subnormal_words": sum(counts.values()),
                               "worst": sorted(counts.items(),
                                               key=lambda kv: -kv[1])[:3]})
    lifted_as = driver.gpu
    driver.close()
    row = {"control": "band-witness", "case": case, "steps": steps,
           "entered_the_band": bool(window), "subnormal_words_total": total,
           "step_path": "array", "driver_gpu": lifted_as,
           "window": window}
    row["verdict"] = "BAND-NON-EMPTY" if window else "BAND-EMPTY"
    say(f"{case}: band witness -> {row['verdict']} "
        f"({total} subnormal words over {steps} steps)")
    return row


def _seam_census(driver: Any):
    """Census every operand a plan is about to read, immediately before each launch.

    THE ONE MOMENT AT WHICH THE KERNEL'S OPERANDS ARE EXACTLY THE ARRAYS ON THE
    OBJECT — after the seam's deposits, before the sub-step overwrites them. Recorded
    as PROVENANCE and not as a verdict: under the installed flush both executors
    flush, so the byte comparison is the arbiter and this says which value class the
    comparison was made over.
    """
    from meep_gpu.metal_kernels import launch  # noqa: PLC0415

    real = launch.SyncedPlan.run
    state: Dict[str, Any] = {"launches": 0, "launches_with_subnormals": 0,
                             "worst": 0, "first": None}

    def patched(self, *args: Any, **kwargs: Any) -> None:  # noqa: ANN001
        counts = _census_state(driver)
        state["launches"] += 1
        total = sum(counts.values())
        if total:
            state["launches_with_subnormals"] += 1
            state["worst"] = max(state["worst"], total)
            if state["first"] is None:
                state["first"] = {"launch": state["launches"],
                                  "subnormal_words": total,
                                  "arrays": sorted(counts)[:6]}
        real(self, *args, **kwargs)

    launch.SyncedPlan.run = patched

    def restore() -> None:
        launch.SyncedPlan.run = real

    return restore, state


#: The smallest positive NORMAL float32. Anything strictly between zero and this is
#: what a flushing executor turns into zero, and it is the whole surface the cliff
#: control measures.
_SMALLEST_NORMAL_FLOAT32 = 1.1754943508222875e-38


#: THE THREE BOUNDS THE CLIFF PREDICATE READS. Two are absolute, because "the
#: disagreement is in the subnormal band" is an absolute statement about float32 and
#: has no relative form; the third is normalised against float32's own resolution.
#:
#: WHY THE OLD FIXED RATIO WAS WRONG, measured 2026-09-11 on ``pml_3d_diagonal``. The
#: predicate required ``max_rel_diff < 1e-30``, and ``max_rel_diff`` is ``max|a-b|``
#: over ``max|a|`` -- the disagreement divided by THE ARRAY'S OWN PEAK. Five of that
#: case's six judged arrays peak near 1e-04 and passed; ``fields.Bz`` peaks at
#: 1.97e-08, four orders lower, because Bz carries only numerical residue in a
#: diagonal 3D PML. Its worst disagreement is 1.55e-36 -- the second SMALLEST of the
#: six -- yet divided by that tiny peak it read 7.85e-29 and failed by 78x. The clause
#: was grading the array's AMPLITUDE, not the disagreement.
#:
#: WHAT THE REPLACEMENT IS AND IS NOT, stated without flattery because this is a
#: RELEASE bound. Write P for the array peak and d for the disagreement. The old
#: predicate admitted ``d < P*1e-30``; the new one admits ``d < min(1e-28, P*R)`` with
#: R = ``_BAND_RATIO_CEILING``. The ratio of the two is ``min(1e2/P, R*1e30)``, so the
#: new predicate is LOOSER for every array peaking below 1e+02 -- which is every array
#: in this corpus -- and tighter only above it. An earlier revision of this comment
#: claimed the opposite ("the only shape the old clause refused and this one admits is
#: an array whose peak is below 1e-16"); that had the inequality inverted and it is
#: recorded here rather than quietly corrected, because a wrong justification under a
#: release bound is the defect, not the typo.
#:
#: SO THE LOOSENING IS DELIBERATE AND IS PAID FOR ELSEWHERE. It is deliberate because
#: ``P*1e-30`` is unsatisfiable on a quiet field component for reasons that have
#: nothing to do with the arithmetic: at P = 1.97e-08 it demanded d < 1.97e-38, which is
#: 1.7 times the smallest normal float32 (1.18e-38) -- so it admitted only a handful of
#: subnormal values and nothing above them, on an array whose disagreements are measured
#: at 1.55e-36. (An earlier revision of this sentence said that bound was BELOW the
#: smallest normal, i.e. exact equality. It is not; it is just above it, and the
#: correction is recorded rather than quietly applied.) What replaces the lost strictness is
#: not a tighter constant but two things a constant cannot do:
#:   * EVERY differing array is now judged. Until 2026-09-11 ``cliff_control`` handed
#:     this predicate ``differences[:6]``, so 6 of 24 arrays decided
#:     ``pml_3d_diagonal`` and 6 of 12 decided three other cases. That truncation was
#:     worth more than any ceiling.
#:   * R is tied to float32 RESOLUTION rather than chosen. float32 eps is 1.19e-07, so
#:     a one-ULP arithmetic error anywhere in an array shows up at max_rel_diff ~1e-07.
#:     R = 1e-20 sits thirteen orders BELOW that -- no arithmetic defect in a float32
#:     kernel can hide under it -- and eight orders ABOVE the worst value this control
#:     has ever measured (7.85e-29). It is not fitted to that measurement; 1e-20 is
#:     where "invisible at float32 resolution" stops being arguable.
#:
#: WHAT THIS PREDICATE STILL CANNOT SAY, recorded so the next reader does not have to
#: rediscover it. The ratio clause binds only where ``P*R < 1e-28``, i.e. on an array
#: peaking below **1e-08** -- that crossover moves with R and an earlier revision of
#: this comment left the 1e-16 figure from the R = 1e-12 era standing, which was stale
#: by eight orders. The quietest array this corpus produces peaks at 1.97e-08, just
#: above it, so the clause is unreachable on every array measured to date.
#: ``max_abs_diff`` is implied by the value clause within a factor of two, since
#: |a-b| <= 2*max(|a|,|b|). So on this corpus the effective constraint is the VALUE
#: bound plus the requirement that some element be exactly zero on one side; the other
#: two are cheap, honest about what they bound, and would bite on a louder corpus.
_BAND_VALUE_CEILING = 1e-28
_BAND_DIFFERENCE_CEILING = 1e-28
_BAND_RATIO_CEILING = 1e-20

#: THE STORED SPLIT-FIELD PML INTERMEDIATES, and their own value ceiling.
#:
#: WHY THEY ARE NOT FIELDS. ``fields.py`` says it outright at the declaration: "MEEP
#: step_generic.cpp: fu holds the INTERMEDIATE RESULT for the dsigu PML direction,
#: needed when two PML directions are active (3D) in step_curl". The recurrence is
#: ``_apply_pml_update`` (stepping.py:1975):
#:
#:     fu_previous = fu.copy()
#:     fu *= kms ; fu -= curl ; fu *= sinv                      # stage 1 -> fu_D*
#:     field *= kms_u ; field += fu ; field -= fu_previous ; field *= sinv_u
#:
#: Stage 1 writes the stored intermediate; stage 2 consumes it as a DIFFERENCE of
#: successive values. So a flush perturbation in ``fu`` reaches the field only through
#: ``fu - fu_previous``, where it largely cancels -- and the stored intermediate carries
#: it UNCANCELLED. Judging the two against one ceiling compares quantities one stage
#: apart.
#:
#: MEASURED 2026-09-11 on ``pml_3d_diagonal``, which is the case with two active PML
#: directions and therefore the only one that allocates these arrays at all. Every
#: other component's intermediate and consumer carry the SAME differing magnitude
#: (fu_Dx/Dx 4.568e-30, fu_Dy/Dy 1.439e-29, fu_Bx/Bx 2.678e-29, fu_Bz/Bz 6.666e-32 --
#: ratio 1.00 on each), and exactly one diverges: fu_Dz at 1.922e-28 against Dz at
#: 6.421e-32, a ratio of 2994, with the SAME 180 elements exactly zero on one side and
#: the same 176 of subnormal magnitude in both. The 1e-28 field ceiling refused the
#: case on that one array.
#:
#: AND THE 1.922e-28 IS THE FLUSH, PROVEN ON THE HOST rather than argued.
#: ``parity/meep_gpu/probe_pml_intermediate_flush.py`` lifts the case twice on the
#: ARRAY path in NumPy with no device, runs one leg as it ships and the other with
#: stage 1's output flushed wherever it is subnormal -- a VALUE test, not a bit mask --
#: and reproduces the device's numbers exactly: fu_Dz 1.922e-28 / max_abs_diff
#: 1.204e-35 / 180 zero-on-one-side / 176 subnormal, and ten of ten other arrays'
#: magnitudes to every recorded digit. 2260 subnormal intermediates flushed over 144
#: stage-1 calls. A flush of the PML intermediate, and nothing else, produces that
#: number.
#:
#: SO THE CEILING IS SEPARATE AND NAMED, not raised for everything. 1e-26 is two orders
#: above the measured value -- not fitted to its last digit -- and twelve orders below
#: fu_Dz's own peak of 2.008e-03, so an arithmetic defect at the field's scale is still
#: refused on these arrays as loudly as on any other. THE STRONGER MOVE, recorded as
#: owed rather than done: make the host replay a LEG of this control, the way
#: ``band_witness`` decides what a threshold could not, and let an array over the field
#: ceiling pass on reproduced evidence instead of on a second constant.
_BAND_INTERMEDIATE_PREFIX = "fu_"
_BAND_INTERMEDIATE_VALUE_CEILING = 1e-26


def _is_pml_intermediate(array: Any) -> bool:
    """Is this stored array a split-field PML stage-1 intermediate rather than a field?

    Decided on the ``fu_`` prefix, which is MEEP's own name for it and the name
    ``fields.Fields`` declares (``fu_Bx``..``fu_Dz``, allocated only by
    ``enable_pml_storage``). A prefix is a weak instrument in general; here it is the
    RIGHT one, because the prefix IS the structural fact -- there are exactly six such
    arrays, they are the only stored values that are an intermediate of a two-stage
    recurrence, and a seventh could not appear without a change to ``fields.py``'s
    declarations. ``_band_ceiling_names_every_intermediate`` in
    ``parity/meep_gpu/test_metal_route_cliff_bound.py`` pins that correspondence
    against ``fields.py`` so the two cannot drift.
    """
    name = str(array or "")
    return name.rsplit(".", 1)[-1].startswith(_BAND_INTERMEDIATE_PREFIX)


def _pml_intermediate_of(array: Any) -> Optional[str]:
    """``fields.Dy`` -> ``fields.fu_Dy``: the stage-1 intermediate that WRITES it.

    Returns None for an array that is itself an intermediate, and a name that simply
    will not be found for the components that have none — ``E``/``H`` carry ``f_w_``
    arrays rather than ``fu_`` ones, and this deliberately does not claim them (see
    :func:`band_confinement` for why that case is left alone).
    """
    name = str(array)
    if _is_pml_intermediate(name):
        return None
    head, dot, leaf = name.rpartition(".")
    if not dot:
        return None
    return f"{head}.{_BAND_INTERMEDIATE_PREFIX}{leaf}"



def _band_census(state: Dict[str, Any]) -> Dict[str, Any]:
    """How many stored words sit IN the subnormal band, per array. The vacuity floor.

    WHY THE CLIFF CONTROL CANNOT CONCLUDE WITHOUT IT. When a keeping host and a
    flushing device produce the same answer, the row named two hypotheses and could
    separate neither: "either the run never enters the subnormal band or the device
    did not flush". Only the first is a statement about the CASE, and it is directly
    measurable on the leg the control already has in hand -- if no stored word is in
    the band, a flush has nothing to bite on and the sameness is arithmetic rather
    than a missing refusal.

    The census reads the KEEPING leg, because that is the one whose values survive:
    on a flushing leg the band is empty by construction, which is why the gate's
    ``seam_census`` reads zero on cases that did produce a cliff and cannot be used
    for this.
    """
    counts: Dict[str, int] = {}
    total = 0
    for name, array in sorted(state.items()):
        values = numpy.asarray(array)
        if values.dtype not in (numpy.float32, numpy.complex64):
            continue
        magnitude = numpy.abs(values.view(numpy.float32)
                              if values.dtype == numpy.complex64 else values)
        in_band = int(numpy.count_nonzero(
            (magnitude > 0.0) & (magnitude < _SMALLEST_NORMAL_FLOAT32)))
        if in_band:
            counts[name] = in_band
            total += in_band
    return {"words_in_the_band": total, "arrays": dict(sorted(
        counts.items(), key=lambda item: -item[1])[:8]),
        "smallest_normal": _SMALLEST_NORMAL_FLOAT32}


#: Where the campaign leaves the band witness's verdict for the legs that follow it.
#: An ARTIFACT PATH and not a recomputation: the witness is its own leg, run under
#: keep with nothing dispatched, and re-deriving its answer inside another leg would
#: be a second instrument to keep in step with the first.
BAND_WITNESS_ENVIRONMENT = "MEEP_GPU_METAL_BAND_WITNESS"


def band_witness_verdicts() -> Dict[str, str]:
    """Per case, what the band-witness leg measured. Empty when it has not run.

    WHY THE CLIFF CONTROL CANNOT CONCLUDE WITHOUT IT, and why nothing inside that
    control can substitute. When a keeping host and a flushing device agree, the row
    could only name two hypotheses -- "the run never enters the subnormal band" or
    "the device did not flush" -- and separating them needs a census of the ORACLE's
    own trajectory, every step, under keep, with nothing dispatched. That is a
    different leg, and a census taken at the end of the cliff control's own short run
    is provably not a substitute: on pml_3d_diagonal the stored state at the end
    holds ZERO band words while the two legs differ on 496 elements, because MPS
    flushes exact subnormal INTERMEDIATES that leave no subnormal in storage.

    MEASURED PAIRING, 2026-09-11: the five cases that read NO-CLIFF -- conductive_2d,
    dispersive_2d, pml_1d, folded_dispersive_2d, cylindrical -- are EXACTLY the five
    the witness reports BAND-EMPTY, and every case that did cliff is BAND-NON-EMPTY.
    """
    path = os.environ.get(BAND_WITNESS_ENVIRONMENT)
    if not path or not os.path.isfile(path):
        return {}
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:  # noqa: BLE001 - an unreadable witness decides nothing
        return {}
    # THE SUMMARY KEYS ITS CONTROLS "<case>:<control>", so the case is the part
    # before the colon. Read from the summary's own shape rather than from a
    # remembered one: this is the only consumer, and a shape that moved would
    # otherwise hand every case ``None`` and silently restore the old NO-CLIFF.
    verdicts: Dict[str, str] = {}
    controls = payload.get("controls")
    if isinstance(controls, dict):
        for key, verdict in controls.items():
            case, _, control = str(key).partition(":")
            if control == "band-witness" and case:
                verdicts[case] = verdict
    elif isinstance(controls, list):
        for row in controls:
            if isinstance(row, dict) and (row.get("control") or "") == "band-witness":
                if row.get("case"):
                    verdicts[row["case"]] = row.get("verdict")
    return verdicts


def band_confinement(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Is every array's disagreement at the subnormal band's scale? With the reason.

    SEPARATED FROM :func:`cliff_control` SO IT CAN BE ARMED WITHOUT A DEVICE. The
    predicate is the licensing half of the cliff control -- it is what decides
    ``CLIFF-CONFINED-TO-THE-BAND`` against ``CLIFF-OUTSIDE-THE-BAND``, which is the
    difference between a policy split and "an arithmetic defect in a dispatched
    kernel" -- and a bound that can only be exercised by a three-hour campaign on a
    Metal host is a bound nothing checks. It takes the comparator's own per-array
    forensics (:func:`gate_dispatch_end_to_end.compare_state`'s ``differences``) and
    nothing else, so the test suite can hand it rows.

    An entry missing a number it is judged on FAILS: the ``or 1.0`` substitutes a
    value that no ceiling admits, so a forensics shape that stopped reporting one of
    these would refuse rather than quietly confine.
    """
    # THE MAGNITUDE EACH ARRAY DIFFERED AT, so a field can be judged against the
    # intermediate that WROTE it. See the inheritance note below.
    magnitudes = {str(entry.get("array")):
                  (entry.get("largest_differing_magnitude") or 0.0)
                  for entry in entries}
    per_array = []
    for entry in entries:
        value = entry.get("largest_differing_magnitude") or 1.0
        difference = entry.get("max_abs_diff") or 1.0
        ratio = entry.get("max_rel_diff") or 1.0
        own = _is_pml_intermediate(entry.get("array"))

        # A FIELD INHERITS ITS OWN INTERMEDIATE'S CEILING, and the reason is the
        # recurrence rather than a preference. stepping._apply_pml_update's stage 2
        # is `field += fu; field -= fu_previous`, so it consumes the stage-1
        # intermediate as a DIFFERENCE OF SUCCESSIVE VALUES and the intermediate's
        # differing magnitude passes straight through into the field. Granting
        # `fu_Dy` a 1e-26 ceiling while holding `Dy` to 1e-28 is therefore
        # incoherent: every intermediate flush between those two values refuses its
        # own consumer, and the consumer's magnitude was never independent evidence.
        #
        # MEASURED, 2026-09-12, and this is the bug that found it. The beta electric
        # pair's release put special_kz_2d's `fields.Dy` at 1.894e-28 — over the
        # field ceiling — while its own `fields.fu_Dy` sat at 1.896e-28, UNDER the
        # intermediate ceiling. The host-only replay in
        # probe_pml_intermediate_flush.py (FU_DZ_CASE=special_kz_2d) flushes ONLY
        # stage-1 outputs, on NumPy, with no device present, and reproduces both:
        # fu_Dy 1.896e-28, Dy 1.894e-28, max_abs_diff 1.204e-35, 24 arrays. Every
        # other pair in that run matches to four figures too — fu_Dx/Dx 1.052e-30,
        # fu_Dz/Dz 4.731e-32, fu_Bz/Bz 3.363e-32, fu_Bx/Bx 2.003e-34 — which is the
        # pass-through stated as a measurement rather than read off the source.
        #
        # IT IS BOUNDED BY THE PRODUCER, not granted outright: the field must differ
        # at NO MORE than its intermediate did. A field carrying a LARGER magnitude
        # than the intermediate that wrote it is not explained by this mechanism and
        # is refused, which is the case this clause must not swallow.
        #
        # THE TIGHT GUARDS ARE UNTOUCHED. `max_abs_diff` (1e-28) and `max_rel_diff`
        # (1e-20) still judge every array at the field ceiling; only band MEMBERSHIP
        # — the magnitude at which elements differ — inherits. On the case that
        # found this, those two sat at 1.2e-35 and 2.3e-31, orders below their own
        # ceilings.
        #
        # `E`/`H` ARE NOT CLAIMED. Their companions are `f_w_` arrays, and whether
        # that relationship is the same pass-through has not been measured, so they
        # keep the field ceiling and this clause says nothing about them.
        producer = _pml_intermediate_of(entry.get("array"))
        produced_at = magnitudes.get(producer) if producer else None
        inherits = produced_at is not None and value <= produced_at
        intermediate = own or inherits
        value_ceiling = (_BAND_INTERMEDIATE_VALUE_CEILING if intermediate
                         else _BAND_VALUE_CEILING)
        failed = [name for name, seen, ceiling in (
            ("largest_differing_magnitude", value, value_ceiling),
            ("max_abs_diff", difference, _BAND_DIFFERENCE_CEILING),
            ("max_rel_diff", ratio, _BAND_RATIO_CEILING)) if not seen < ceiling]
        per_array.append({"array": entry.get("array"), "over_ceiling": failed,
                          # WHICH CEILING JUDGED IT, in the record. A reader of the
                          # artifact can otherwise not tell why two arrays with the
                          # same magnitude got different verdicts.
                          "judged_as": ("pml_intermediate" if own else
                                        "field_downstream_of_its_pml_intermediate"
                                        if inherits else "field"),
                          # NAMED, so the artifact says which array lent the
                          # ceiling rather than leaving a reader to infer it.
                          "ceiling_inherited_from": producer if inherits else None,
                          "value_ceiling": value_ceiling})
    return {"confined": bool(entries) and not any(row["over_ceiling"]
                                                  for row in per_array),
                "ceilings": {"largest_differing_magnitude": _BAND_VALUE_CEILING,
                         "largest_differing_magnitude_pml_intermediate":
                             _BAND_INTERMEDIATE_VALUE_CEILING,
                         "max_abs_diff": _BAND_DIFFERENCE_CEILING,
                         "max_rel_diff": _BAND_RATIO_CEILING},
            "per_array": per_array}


def cliff_control(case: str, spec: Dict[str, Any], res: Optional[int]) -> Dict[str, Any]:
    """Drive a released pair with the HOST KEEPING and require divergence IN THE BAND.

    THE MEASUREMENT THAT MAKES THE POLICY RUNG A RULE RATHER THAN A PRECAUTION, and
    it has to go round the ladder in TWO places rather than one. Rung 8bM refuses to
    dispatch under an installed ``keep``, so there is no composition to compare on
    that leg; and the composer's own backend clause
    (``metal_kernels.subnormal.mps_policy_reasons``) refuses every arm under a
    resolved ``keep`` as well, so an installed ``keep`` means nothing is even
    SELECTED. Measured 2026-09-10 on the ``harness_keep`` leg: the composer filled
    zero slots and the control reported "nothing to arm" — a control that cannot
    fire is a control that measures nothing.

    SO THE SPLIT IS BUILT IN THE ORDER IT WOULD HAPPEN, on a leg where the composer
    works: the plans are composed under a DECLARED flush and seated at the driver's
    own freeze slot (exactly as ``probe_metal_dispatch_dryrun`` does), and only then
    is the HOST FPU driven to keep for the duration of the run. The device half
    flushes natively and cannot be moved at all, which is the whole reason ``keep``
    is unattainable there — so what steps is a genuine two-policy process, which is
    precisely what rung 8bM exists to refuse.

    THE PREDICATE IS THE COMPARATOR'S OWN BAND BOUND, and the obvious stronger
    assertion is FALSE. Measured 2026-09-10 at step 24: ``fields.Ez`` has
    ``elements_differing`` 111 against ``elements_subnormal_magnitude`` 53, because
    results derived FROM a flushed intermediate are normal-magnitude one step later
    — the subnormal-intermediate instrument gap. So requiring
    ``elements_subnormal_magnitude == elements_differing`` would turn a correct
    licensing control red. What is true, and what is asserted, is that the
    divergence is CONFINED to the band's scale: the largest differing magnitude is
    within a few ULP of the subnormal range against a smallest normal of 1.18e-38,
    the disagreement ITSELF is bounded at that same scale, it is invisible at
    float32 resolution against the array's own peak, and at least one side of some
    differing element is exactly zero — which is what a flush produces and nothing
    else does.
    """
    import meep_gpu  # noqa: PLC0415
    from meep_gpu import subnormal_policy  # noqa: PLC0415

    row: Dict[str, Any] = {"control": "subnormal-cliff", "case": case,
                           "steps": CONTROL_STEPS * 2,
                           "how": ("composed under a declared flush, then the HOST "
                                   "FPU driven to keep for the run; the MPS half "
                                   "flushes natively and has no lever")}
    fused_driver, _m, _u = _lift(case, res)
    array_driver, _am, _au = _lift(case, res)
    pre = e2e.compare_state(e2e.collect_state(fused_driver),
                            e2e.collect_state(array_driver))
    row["precondition_identical"] = pre["identical"]
    if not pre["identical"]:
        row["verdict"] = "HARNESS-FAILURE"
        fused_driver.close()
        array_driver.close()
        return row
    shim, detail = _dryrun().install_shim(
        fused_driver, fused_driver._sources,  # noqa: SLF001
        fuse=True, fuse_labels=tuple(spec["arms"]))
    row["composition"] = {"selected": detail["selected"], "mirrors": detail["mirrors"]}
    if shim is None:
        row["verdict"] = "NOT-ARMED"
        row["why"] = ("the composer filled no slot under a keeping host, so there is "
                      "no launch beside which the cliff could be measured")
        fused_driver.close()
        array_driver.close()
        return row
    # THE HOST GOES TO KEEP HERE AND NOWHERE EARLIER: the composition above needed
    # the declared flush to exist at all, and the split is what the STEPPING does.
    flushing_at_entry = subnormal_policy.backends.subnormals_flushed()
    subnormal_policy._drive_host_fpu(False)  # noqa: SLF001 - the rollback path's own call
    row["host_kept"] = not subnormal_policy.backends.subnormals_flushed()
    at_twelve: Optional[bool] = None
    try:
        for point, chunk in ((CONTROL_STEPS, CONTROL_STEPS),
                             (CONTROL_STEPS * 2, CONTROL_STEPS)):
            fused_driver.run(num_steps=chunk)
            array_driver.run(num_steps=chunk)
            verdict = e2e.compare_state(e2e.collect_state(fused_driver),
                                        e2e.collect_state(array_driver))
            if point == CONTROL_STEPS:
                at_twelve = verdict["identical"]
            # EVERY DIFFERING ARRAY, NOT THE FIRST SIX. This was
            # ``verdict["differences"][:6]`` until 2026-09-11, and the predicate below
            # reads exactly this list -- so the control that decides whether a Metal
            # release is a policy split or an arithmetic defect was judging 6 of the
            # 24 arrays that differed on ``pml_3d_diagonal``, and 6 of 12 on
            # ``pml_2d``, ``magnetic_seam_2d`` and ``folded_2d``, then calling the
            # whole case CONFINED. A truncation for readability sat upstream of a
            # verdict. The list stays whole here; ``compare_state``'s own cap of 40 is
            # the only one left, and ``judged_every_differing_array`` below refuses
            # rather than trusting it.
            row[f"at_{point}_steps"] = {
                "identical": verdict["identical"],
                "arrays_differing": verdict["arrays_differing"],
                "differences": verdict["differences"]}
    finally:
        # UNCONDITIONAL, exception included. A control that left the process in the
        # policy it was arming would contaminate every case after it, which is the
        # one failure a green isolated run cannot see.
        subnormal_policy._drive_host_fpu(flushing_at_entry)  # noqa: SLF001
    row["host_restored"] = (subnormal_policy.backends.subnormals_flushed()
                            == flushing_at_entry)
    # THE BAND CENSUS IS TAKEN BEFORE THE DRIVERS CLOSE, on the ARRAY leg -- the one
    # whose host kept, so its subnormals are still there to count. Taken here rather
    # than only on the NO-CLIFF branch, so a run that DID cliff also records how much
    # band it had.
    keeping_state = e2e.collect_state(array_driver)
    band_census = _band_census(keeping_state)
    band_arrays = len(keeping_state)
    row["band_census"] = band_census
    fused_driver.close()
    array_driver.close()
    row["policy_in_force"] = dict(meep_gpu.policy_stamp())
    final = row[f"at_{CONTROL_STEPS * 2}_steps"]
    row["identical_at_the_house_budget"] = at_twelve
    if not row["host_kept"]:
        row["verdict"] = "NOT-ARMED"
        row["why"] = ("the host FPU could not be driven to keep, so both executors "
                      "were flushing and no split was built")
        say(f"{case}: cliff control -> NOT-ARMED (the host would not keep)")
        return row
    if final["identical"]:
        # WHICH OF THE TWO HYPOTHESES, decided rather than listed. The census is taken
        # on the KEEPING leg, whose values a flush has not already zeroed.
        # AN EMPTY STORED BAND IS NOT A PROOF THAT THE RUN STAYED OUT OF THE BAND,
        # and this branch used to draw that conclusion. Measured on this same
        # campaign, pml_3d_diagonal: the keeping leg's stored census reads ZERO words
        # in the band at the end of the run and the two legs nonetheless differ on
        # 496 elements of fields.Bx -- because MPS flushes exact subnormal
        # INTERMEDIATES, which no stored-state census can see, and the results
        # derived from them are normal-magnitude one step later. So the census is
        # recorded as DISCLOSURE beside the verdict and decides nothing: turning it
        # into a pass would have passed exactly the cases where the evidence is
        # weakest, which is the wrong direction for a control to be wrong in.
        # Separating "never entered the band" from "the device did not flush" needs
        # the band-witness leg, which the campaign runs last.
        witness = band_witness_verdicts().get(case)
        row["band_witness"] = witness
        if witness == "BAND-EMPTY":
            row["verdict"] = "NOT-APPLICABLE"
            row["why"] = (
                "a released pair ran under a keeping host and the answer did not "
                "move BECAUSE this case's trajectory never enters the subnormal "
                "band: the band-witness leg censused the oracle's own run, every "
                "step, under keep with nothing dispatched, and reported BAND-EMPTY. "
                "A flush has nothing to bite on here, so there is no cliff for this "
                "control to measure and rung 8bM's refusal is licensed by the cases "
                "whose band IS non-empty")
            say(f"{case}: cliff control -> NOT-APPLICABLE (band witness: BAND-EMPTY)")
            return row
        row["verdict"] = "NO-CLIFF"
        row["why"] = (
            "a released pair ran under a keeping host and the answer did not move "
            + (f"while the band-witness leg reports {witness!r} for this case, so "
               "the run DOES enter the band and the device did not flush"
               if witness else
               "and no band-witness verdict was available to say whether the run "
               f"enters the band at all (set {BAND_WITNESS_ENVIRONMENT} to the "
               "witness leg's artifact; the campaign runs that leg first)"))
        say(f"{case}: cliff control -> NO-CLIFF "
            f"(stored band {band_census['words_in_the_band']} words, which decides "
            "nothing)")
        return row
    # THE BAND BOUND, read off the comparator's own forensics rather than restated.
    worst = [entry for entry in final["differences"] if "largest_differing_magnitude" in entry]
    row["band"] = [{"array": entry["array"],
                    "elements_differing": entry.get("elements_differing"),
                    "elements_subnormal_magnitude": entry.get("elements_subnormal_magnitude"),
                    "elements_one_side_exactly_zero": entry.get("elements_one_side_exactly_zero"),
                    "largest_differing_magnitude": entry.get("largest_differing_magnitude"),
                    "max_abs_diff": entry.get("max_abs_diff"),
                    "max_rel_diff": entry.get("max_rel_diff"),
                    "reads_as": entry.get("reads_as")}
                   for entry in worst]
    confinement = band_confinement(worst)
    confined = confinement["confined"]
    row["band_bounds"] = confinement
    # FAIL CLOSED ON A SHORT LIST. The predicate is only a statement about the case if
    # it saw every array that differed; a comparator cap (``compare_state`` keeps 40)
    # or a re-introduced slice would otherwise quietly narrow what CONFINED means.
    judged = len(confinement["per_array"])
    row["judged_every_differing_array"] = {
        "arrays_differing": final["arrays_differing"],
        "arrays_judged": judged,
        "agrees": judged == final["arrays_differing"]}
    if judged != final["arrays_differing"]:
        confined = False
        row["band_bounds"]["confined"] = False
        row["band_bounds"]["short_list"] = (
            f"{judged} of {final['arrays_differing']} differing arrays reached the "
            f"predicate; a confinement verdict over a subset is not a verdict about "
            f"the case")
    one_side_zero = any((entry.get("elements_one_side_exactly_zero") or 0) > 0
                        for entry in worst)
    row["confined_to_the_band"] = confined
    row["some_element_is_exactly_zero_on_one_side"] = one_side_zero
    row["verdict"] = ("CLIFF-CONFINED-TO-THE-BAND" if (confined and one_side_zero)
                      else "CLIFF-OUTSIDE-THE-BAND")
    if not (confined and one_side_zero):
        row["why"] = ("the two executors disagreed at magnitudes a flush-vs-keep "
                      "split cannot produce on its own; that is an arithmetic "
                      "defect in a dispatched kernel, not a policy split")
    say(f"{case}: cliff control -> {row['verdict']} "
        f"(identical at {CONTROL_STEPS} steps: {at_twelve})")
    return row


# ---------------------------------------------------------------------------
# Case-level helpers
# ---------------------------------------------------------------------------

def _dryrun():
    """The dry-run probe's composer + shim, imported lazily.

    SHARED RATHER THAN COPIED. The cliff control needs a composition seated at the
    driver's freeze slot without the ladder, which is precisely what
    ``probe_metal_dispatch_dryrun`` is; two spellings of "install the shipped
    composer's plans in the engine's own seam" would be two chances to place the
    residency bracket differently from the way the ladder places it.
    """
    import probe_metal_dispatch_dryrun  # noqa: PLC0415

    return probe_metal_dispatch_dryrun


def metal_host_refusal(gate: str) -> Optional[str]:
    """Why this host cannot run a Metal-table harness, or ``None`` on an Apple GPU.

    ``prefer_gpu=True`` is THIS HOST'S GPU. On a CUDA host it lifts CuPy drivers,
    which dispatch the NVIDIA tables and would be graded here under Metal labels; on
    a host with no GPU the first lift raises. Both are refused up front, by name,
    before a leg directory is written.
    """
    from meep_gpu import backends  # noqa: PLC0415

    gpu = backends.available_gpu()
    if gpu == "metal":
        return None
    return (f"REFUSING: {gate} drives the Metal kernel table and "
            f"meep_gpu.backends.available_gpu() is {gpu!r} on this host, not "
            f"'metal'. It runs only on an Apple GPU host.")


def require_metal_driver(driver: Any, what: str) -> None:
    """Refuse a leg whose driver is not an Apple GPU driver.

    A ``prefer_gpu=False`` lift is the NumPy reference: it never consults a kernel
    table, so as a fused or unfused leg it would time and compare the array path
    against itself. A CuPy driver is the other tables'. Either is a harness defect,
    and it stops the gate at the lift rather than surfacing as a vacuous leg.
    """
    gpu = getattr(driver, "gpu", None)
    if gpu != "metal":
        raise SystemExit(f"the Metal route gate lifted a {gpu!r} driver for {what}; "
                         "every leg is an Apple GPU driver "
                         "(lift_simulation(prefer_gpu=True) on an Apple GPU host)")


def _lift(case: str, res: Optional[int]) -> Tuple[Any, Any, float]:
    import meep as mp  # noqa: PLC0415
    import meep_gpu  # noqa: PLC0415

    try:
        mp.verbosity(0)
    except Exception:  # noqa: BLE001
        pass
    builder = CASES[case]
    sim, monitors, until = builder(mp) if res is None else builder(mp, res)
    driver = meep_gpu.lift_simulation(sim, prefer_gpu=True)
    require_metal_driver(driver, f"{case}")
    return driver, monitors, until


def _pair_of_lifted(case: str, spec: Dict[str, Any], res: Optional[int],
                    counter: MetalLaunchCounter,
                    label_a: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """A fused leg and an array leg, lifted the way the case's own route reaches it.

    An armed control that reached its fused launch through the opt-in while the case
    reached it through the release would be arming a different composition than the
    one it is a control for — the same pair, admitted by a different clause.
    """
    builder = CASES[case]
    a = e2e.run_leg(case, builder, label_a,
                    _fuse_env(spec["arms"], spec.get("reached_by", "release")),
                    counter, 0, res)
    b = e2e.run_leg(case, builder, "array", ARRAY_ENV, counter, 0, res)
    for leg in (a, b):
        require_metal_driver(leg["driver"], f"{case}/{leg['label']}")
    return a, b


def _one_step_report(case: str, env: Dict[str, Optional[str]],
                     res: Optional[int]) -> Tuple[Dict[str, Any], str]:
    """Lift ``case`` under ``env``, take ONE driver step, return its record + path.

    One step is the whole instrument for a leg that reads a DECISION: the plan is
    frozen on the first step and the artifact is complete from then on. Legs that
    compare BYTES are the ones that need the full budget, and they are elsewhere.
    """
    for name, value in env.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
    driver, _monitors, _until = _lift(case, res)
    driver.step()
    report = dict(driver.fast_path_report() or {})
    path = driver.active_step_path
    driver.close()
    return report, path


# ---------------------------------------------------------------------------
# The armed controls
# ---------------------------------------------------------------------------

def withheld_control(case: str, spec: Dict[str, Any], res: Optional[int],
                     counter: MetalLaunchCounter) -> Dict[str, Any]:
    """The absorbed consult forced False. MUST diverge.

    With the second consult answering False the driver runs the array
    ``update_H``/``update_E`` ON TOP of the constitutive half the fused launch
    already computed, and ``stepping._apply_constitutive_pml`` ACCUMULATES — so that
    is a wrong answer wherever the PML is graded, not a redundant recomputation. If
    this control does NOT diverge, the byte comparison cannot see whether the fused
    kernel's second half ran at all.

    THE LEADING REPAIR SLOT IS EXCLUDED, structurally rather than by slot name:
    ``deposit_repair.LeadingRepairPlan`` carries ``absorbed_by is self.inner`` and it
    LAUNCHES, so withholding it suppresses the launch itself and makes a divergence
    ambiguous about which mechanism produced it. ``route._withhold_absorbed_consult``
    is that discriminator, reused rather than re-derived.

    ON AN INACTIVE LAYER THE MECHANISM DOES NOT EXIST, and the ladder says so the way
    the fused route gate's does (``gate_dispatch_fused_route.withheld_control``),
    reusing its helpers unchanged. ``stepping.update_E`` forks on ``_pml_is_active``:
    the inactive arm is a pure overwrite ``E = constitutive`` and ``update_H`` returns
    without a write, so the array call withheld here ERASES whatever the kernel wrote
    and recomputes E from the same D and P the array leg holds -- array E against
    array E, identical whatever the kernel did. Measured 2026-09-18 on this device:
    with the kernel's E write suppressed AND the consult withheld, 0 of 65 arrays
    differ on ``absorber_1d`` and 0 of 29 on ``no_pml_dispersive_2d``. It read
    NOT-ARMED until the envelope round admitted those arms on ``pml_active=False``,
    and NULL-DID-NOT-DIVERGE after, on the same blindness. NOT-APPLICABLE is granted
    only where every guard below holds, and never on an unreadable layer.
    """
    say(f"{case}: ARMED NULL — {CONTROL_STEPS} steps with the ABSORBED consult "
        "forced False")
    a, b = _pair_of_lifted(case, spec, res, counter, "withheld")
    pre = e2e.compare_state(e2e.collect_state(a["driver"]),
                            e2e.collect_state(b["driver"]))
    row: Dict[str, Any] = {"control": "absorbed-consult-withheld",
                           "precondition_identical": pre["identical"]}
    if not pre["identical"]:
        row["verdict"] = "HARNESS-FAILURE"
        a["driver"].close()
        b["driver"].close()
        return row
    restore, state = route._withhold_absorbed_consult()  # noqa: SLF001
    try:
        step_leg(case, a, counter, CONTROL_STEPS)
    finally:
        restore()
    step_leg(case, b, counter, CONTROL_STEPS)
    block = fusion_block(a)
    verdict = e2e.compare_state(e2e.collect_state(a["driver"]),
                                e2e.collect_state(b["driver"]))
    skipped_launching = dict(state.get("skipped_launching") or {})
    launching = route._launching_slots_ran(a, skipped_launching,  # noqa: SLF001
                                           CONTROL_STEPS)
    constitutive_only = bool(state["slots"]) and all(
        slot in route._CONSTITUTIVE_SLOTS for slot in state["slots"])  # noqa: SLF001
    overwrite_branch = route._constitutive_is_an_overwrite(a)  # noqa: SLF001
    sides = sorted(side for side, slots in _SEAM_SLOTS.items()
                   if any(slot in slots for slot in state["slots"]))
    unarmed_sides = [side for side in sides
                     if side not in (spec.get("seam") or ())
                     or not _side_is_fused(block["driven"], side)]
    row.update({
        "steps": CONTROL_STEPS,
        "consults_withheld": state["withheld"],
        "withheld_slots": state["slots"],
        "leading_repair_slots_left_running": state["skipped_leading"],
        "launching_slots_left_running": skipped_launching,
        "launching_slots": launching,
        "every_withheld_slot_is_constitutive": constitutive_only,
        "constitutive_is_a_pure_overwrite": overwrite_branch,
        "withheld_sides": sides,
        "withheld_sides_without_an_armed_in_seam_control": unarmed_sides,
        "fused_arms_driven": block["driven"],
        "diverged": not verdict["identical"],
        "arrays_differing": verdict["arrays_differing"],
        "arrays_compared": verdict["arrays"],
        "first_differences": verdict["differences"][:6],
    })
    if not state["withheld"]:
        row["verdict"] = "NOT-ARMED"
        row["why"] = ("no absorbed consult was reached, so nothing was withheld and "
                      "this control measured nothing")
    elif launching["failures"]:
        # PORTED FROM THE FUSED ROUTE GATE: a device group this control left running
        # must show it ran. It only tightens; no Metal composition measured so far
        # leaves one running.
        row["verdict"] = "LAUNCHING-SLOT-DID-NOT-LAUNCH"
        row["why"] = "; ".join(launching["failures"])
    elif not verdict["identical"]:
        row["verdict"] = "DIVERGED-AS-REQUIRED"
    elif (overwrite_branch is True and constitutive_only and sides
          and not unarmed_sides and not skipped_launching):
        # THE MECHANISM DOES NOT EXIST ON THIS BRANCH (docstring). What carries the
        # claim is named in the order it carries it: the MAIN byte comparison first
        # -- the fused leg answered the update_E consult itself on every step, so no
        # array update_E ran there, and a dead second half parts it from the array
        # leg (measured 13 of 65 arrays on absorber_1d, 8 of 29 on
        # no_pml_dispersive_2d, in 12 steps) -- and the in-seam mutation only as
        # corroboration that no array pass recomputed E after the wipe. The in-seam
        # row alone does NOT carry it: it diverges identically with the second half
        # suppressed.
        row["verdict"] = "NOT-APPLICABLE"
        row["why"] = (
            f"the layer is INACTIVE, so the array arm of every withheld slot "
            f"{sorted(state['slots'])} is a pure overwrite (update_E, "
            f"stepping.py:1019-1022) or a return without a write (update_H, "
            f"stepping.py:944-945): the withheld call erases what the kernel wrote "
            f"and this comparison is array against array whatever the kernel did. "
            f"What carries the claim instead: the case row's own fused-vs-array byte "
            f"comparison over the full run, on a leg whose update_E consult the fused "
            f"product answered every step (a dead second half fails it); corroborated "
            f"by in-seam-pass-mutated-{'/'.join(sides)}, which shows no array pass "
            f"recomputed E after the wall wipe")
    elif verdict["identical"]:
        row["verdict"] = "NULL-DID-NOT-DIVERGE"
        row["why"] = ("the array constitutive call ran ON TOP of the fused launch's "
                      "own constitutive half and the answer did not change, so this "
                      "comparison cannot see whether the fused kernel's second half "
                      "ran at all")
    else:
        row["verdict"] = "DIVERGED-AS-REQUIRED"
    say(f"{case}: ARMED NULL -> {row['verdict']} "
        f"({row.get('consults_withheld')} consults withheld, "
        f"{row.get('arrays_differing')} arrays differ)")
    a["driver"].close()
    b["driver"].close()
    return row


#: Which driver slots each seam's fused pair occupies. A pair "spans" a seam when it
#: holds the seam's CURL slot -- the slot the in-seam wipe runs beside.
_SEAM_SLOTS = {"B": ("step_B", "update_H"), "D": ("step_D", "update_E", "update_P")}


def _side_is_fused(driven: Dict[str, str], side: str) -> bool:
    """Does a fused pair span this seam on this leg? Read off the driven slots.

    Read from the RECORD rather than from the drive table's ``seam`` declaration,
    because the declaration is what was wrong: three cases declared both seams while
    their own prose said one seam had been given away to separate arms.
    """
    slots = _SEAM_SLOTS.get(side, ())
    return any(driven.get(slot) for slot in slots)


def seam_mutation_control(case: str, spec: Dict[str, Any], side: str,
                          res: Optional[int],
                          counter: MetalLaunchCounter) -> Dict[str, Any]:
    """``zero_metal_{side}`` made non-idempotent. MUST diverge.

    Byte identity on a metallically walled grid rests on a composition no gate had
    run: the fused kernel performs the wall wipe IN-LAUNCH and the driver then runs
    the ARRAY wipe on top, unconditionally, because ``driver.step`` puts it outside
    every consult. "That re-run changes nothing" is a NULL. Replacing the wipe with
    a write the launch cannot have produced turns the null into a positive.
    """
    say(f"{case}: SEAM MUTATION on zero_metal_{side} — {CONTROL_STEPS} steps")
    restore, state = route._mutate_in_seam_pass(side)  # noqa: SLF001
    row: Dict[str, Any] = {
        "control": f"in-seam-pass-mutated-{side}",
        "mutation": (f"zero_metal_{side} writes {route.SEAM_MUTATION_VALUE} into the "
                     "wall plane instead of leaving the wipe idempotent")}
    try:
        a, b = _pair_of_lifted(case, spec, res, counter, "mutated")
        pre = e2e.compare_state(e2e.collect_state(a["driver"]),
                                e2e.collect_state(b["driver"]))
        row["precondition_identical"] = pre["identical"]
        if not pre["identical"]:
            row["verdict"] = "HARNESS-FAILURE"
            a["driver"].close()
            b["driver"].close()
            return row
        step_leg(case, a, counter, CONTROL_STEPS)
        step_leg(case, b, counter, CONTROL_STEPS)
        verdict = e2e.compare_state(e2e.collect_state(a["driver"]),
                                    e2e.collect_state(b["driver"]))
        block = fusion_block(a)
        row.update({
            "steps": CONTROL_STEPS,
            "pass_calls": state["calls"],
            "components_written": state["written"],
            "fused_arms_driven": block["driven"],
            "diverged": not verdict["identical"],
            "arrays_differing": verdict["arrays_differing"],
            "first_differences": verdict["differences"][:6],
        })
        if not state["calls"]:
            row["verdict"] = "NOT-ARMED"
            row["why"] = f"zero_metal_{side} was never called"
        elif not block["driven"]:
            row["verdict"] = "NOT-ARMED"
            row["why"] = "the leg did not fuse, so the mutation had no seam to sit in"
        elif not _side_is_fused(block["driven"], side):
            # THE GUARD IS PER SIDE, because the mutation is. ``zero_metal_B`` only
            # sits in a seam the MAGNETIC pair spans; on a case whose B seam runs
            # separate certified arms, both legs run the same mutated wipe followed
            # by the same array sub-step and the mutation cancels -- which is a true
            # statement about that case and NOT evidence that the comparator is
            # blind. Asking "did anything fuse" instead of "did THIS side fuse"
            # reported three such cases as NULL-DID-NOT-DIVERGE on 2026-09-11:
            # conductive_2d and folded_dispersive_2d fuse only the B seam, cylindrical
            # only the D seam, and each case's own `why` text in DRIVE says so.
            row["verdict"] = "NOT-APPLICABLE"
            row["why"] = (f"this case fuses {sorted(set(block['driven'].values()))} "
                          f"and no fused pair spans the {side} seam, so the mutated "
                          f"wipe has no in-launch twin to be compared against")
        elif verdict["identical"]:
            row["verdict"] = "NULL-DID-NOT-DIVERGE"
            row["why"] = ("the in-seam pass was changed to write a value the fused "
                          "launch cannot have produced and the two legs still "
                          "matched, so the byte comparison is blind to what runs "
                          "between the two consults")
        else:
            row["verdict"] = "DIVERGED-AS-REQUIRED"
        a["driver"].close()
        b["driver"].close()
    finally:
        restore()
    say(f"{case}: SEAM MUTATION {side} -> {row.get('verdict')} "
        f"({row.get('arrays_differing')} arrays differ)")
    return row


def residency_controls(case: str, spec: Dict[str, Any], res: Optional[int],
                       counter: MetalLaunchCounter) -> List[Dict[str, Any]]:
    """The three residency controls, each a separate row. Two MUST diverge; one is a floor.

    (a) DROP ONE sync_in PER STEP on the leading bracketed launch — the seam's own
        armed null;
    (b) DROP THE INNER BRACKET of every wrapper that does host work first, so the
        upload lands BEFORE that work;
    (c) VERIFY IMMEDIATELY AFTER EVERY sync_out — the vacuity floor, which must be
        clean on a shipped seam and whose registry must be non-empty.
    """
    rows: List[Dict[str, Any]] = []
    for name, arm, must_diverge in (
            ("residency-sync_in-dropped", _drop_one_sync_in, True),
            ("residency-inner-bracket-dropped", _drop_the_inner_bracket, True),
            ("residency-verify-after-every-sync_out", _verify_after_every_sync_out,
             False)):
        say(f"{case}: RESIDENCY CONTROL {name} — {CONTROL_STEPS} steps")
        row: Dict[str, Any] = {"control": name}
        a, b = _pair_of_lifted(case, spec, res, counter, name)
        pre = e2e.compare_state(e2e.collect_state(a["driver"]),
                                e2e.collect_state(b["driver"]))
        row["precondition_identical"] = pre["identical"]
        if not pre["identical"]:
            row["verdict"] = "HARNESS-FAILURE"
            a["driver"].close()
            b["driver"].close()
            rows.append(row)
            continue
        # ONE STEP FIRST: the plan does not exist until the driver's freeze, and
        # every arm below reaches into it.
        step_leg(case, a, counter, 1)
        step_leg(case, b, counter, 1)
        restore, state = arm(a["driver"])
        try:
            step_leg(case, a, counter, CONTROL_STEPS)
        finally:
            restore()
        step_leg(case, b, counter, CONTROL_STEPS)
        verdict = e2e.compare_state(e2e.collect_state(a["driver"]),
                                    e2e.collect_state(b["driver"]))
        row.update({"steps": CONTROL_STEPS, "state": state,
                    "diverged": not verdict["identical"],
                    "arrays_differing": verdict["arrays_differing"],
                    "first_differences": verdict["differences"][:4],
                    "fused_arms_driven": fusion_block(a)["driven"]})
        if must_diverge:
            if not state.get("armed"):
                read_only = state.get("read_only_slots") or []
                # A PROVEN NO-OP IS NOT AN UNARMED CONTROL. ``NOT-ARMED`` says the
                # control could not arm and does not say why, and the leg fails on
                # it; ``NOT-APPLICABLE`` carries the measurement that this
                # composition cannot exercise the control at all, which is a
                # statement and not a gap.
                # NEITHER A WRITING WRAPPER NOR ANY WRAPPER AT ALL is a property of
                # the COMPOSITION, measured here and recorded with the wrapper kinds
                # this plan actually holds. The control disarms a bracket around
                # pre-launch host work that WRITES; a plan with no such wrapper
                # cannot exercise it, and calling that a gap would fail a leg for a
                # shape the control was never about.
                row["verdict"] = "NOT-APPLICABLE"
                row["why"] = (
                    (f"this composition carries no bracketed wrapper at all "
                     f"({state.get('wrapper_kinds') or 'none'}), so there is no "
                     "pre-launch host work for this control to move an upload "
                     "across")
                    if not read_only else
                    (f"the only bracketed wrappers here are {read_only} "
                     f"({state.get('wrapper_kinds')}), whose pre-launch host work "
                     "only READS the fields; moving an upload across a read-only "
                     "operation cannot change a byte, so there is nothing this "
                     "control could have measured on this composition"))
            elif verdict["identical"] and not state.get("steps_with_stale_mirrors",
                                                        1):
                # THE UPLOAD HAD NOTHING TO CARRY. Measured at the instant it was
                # suppressed: every mirror already held what the host held, on every
                # step. Skipping a copy of bytes that are already equal cannot change
                # an answer, so this is a statement about the composition -- nothing
                # writes a mirrored array between the previous step's last copy-back
                # and this launch -- and not about the comparator.
                row["verdict"] = "NOT-APPLICABLE"
                row["why"] = (
                    f"the suppressed upload was a no-op on all "
                    f"{state.get('suppressed')} steps: residency.verify() read EMPTY "
                    "at the instant of every skipped sync_in, so the device already "
                    "held the host's bytes and there was nothing for the bracket to "
                    "carry")
            elif verdict["identical"]:
                row["verdict"] = "NULL-DID-NOT-DIVERGE"
                row["why"] = (
                    f"the residency bracket was disarmed on "
                    f"{state.get('steps_with_stale_mirrors')} step(s) where the "
                    f"device did NOT already hold the host's bytes "
                    f"({state.get('stale_mirrors')}) and the answer still did not "
                    "change, so the sync is not load-bearing here and the byte "
                    "identity above it is not evidence about the seam")
            else:
                row["verdict"] = "DIVERGED-AS-REQUIRED"
        else:
            reads = state.get("reads", 0)
            mirrors = state.get("mirrors", 0)
            if not reads or not mirrors:
                row["verdict"] = "NOT-ARMED"
                row["why"] = (f"{reads} verdicts read over {mirrors} registered "
                              "mirrors; an empty verdict from an empty registry "
                              "proves nothing")
            elif state.get("disagreements"):
                row["verdict"] = "RESIDENCY-DISAGREED"
                row["why"] = ("a device mirror disagreed with its host array "
                              "immediately after the copy-back, when nothing had "
                              "run in between")
            elif state.get("under_synced"):
                # THE HELD FORM'S FIRST FAILURE DIRECTION: a mirror differs that
                # the bookkeeping believes CLEAN. That is a device write the state
                # machine did not record, so nothing will ever sync it out and the
                # host reads pre-launch bytes for the rest of the run.
                row["verdict"] = "RESIDENCY-UNDER-SYNCED"
                row["why"] = ("a mirror the ownership machine calls clean disagreed "
                              "with its host array: a device write it did not "
                              "record, which no later sync will repair")
                row["under_synced"] = state["under_synced"][:4]
            else:
                row["verdict"] = "RESIDENCY-CLEAN"
            # THE OVER-CLAIM IS MEASURED, NOT FAILED, and the asymmetry is the whole
            # reason the two directions are separated. An UNDER-sync is a device
            # write the machine did not record: nothing will ever copy it out, the
            # host reads pre-launch bytes for the rest of the run, and the answer is
            # wrong. An OVER-claim is the opposite -- a mirror marked device-owned
            # that the launch did not actually change. Every consumer still behaves
            # correctly: ``sync_in`` skips it because both sides are equal, and an
            # ``acquire_read`` copies it out and gets the right words. It costs a
            # redundant copy and it cannot produce a wrong number.
            #
            # It is over-claimed here BY CONSTRUCTION: a held composition brackets
            # with ``writes=None``, so ``sync_out`` marks every registered mirror
            # device-owned although the launch wrote six. Making the claim exact
            # needs the per-slot write declarations -- which is the SCOPED mode's
            # work, and is measured rather than derived because ``volumes`` is not a
            # cover of the written set on 28 of the 49 families. Until that lands,
            # the honest thing is to publish the size of the over-claim rather than
            # either failing a correct run or pretending the claim is tight.
            row["over_claimed"] = state.get("over_synced", [])[:4]
            row["over_claimed_reads"] = len(state.get("over_synced", []))
            row["held_reads"] = state.get("held_reads", 0)
        say(f"{case}: {name} -> {row['verdict']}")
        a["driver"].close()
        b["driver"].close()
        rows.append(row)
    return rows


def release_route_leg(case: str, spec: Dict[str, Any],
                      res: Optional[int]) -> Dict[str, Any]:
    """The SHIPPED route, asserted: the switch UNSET, and the release does the admitting.

    "The switch was unset in the environment" and "the release is what admitted the
    arm" are different claims and only the second is the licence. If a future edit
    made the main leg set the switch again, every byte comparison would still pass
    and the gate would go on calling the result evidence about the shipped path. The
    three assertions below are what stop that.
    """
    if spec.get("reached_by") != "release":
        return {"leg": "release-route", "case": case, "verdict": "NOT-APPLICABLE",
                "why": (f"{case} is reached by {spec.get('reached_by')!r}; there is "
                        "no release to assert here")}
    report, path = _one_step_report(case, _fuse_env(spec["arms"], "release"), res)
    fusion = dict(report.get("fusion") or {})
    driven = sorted(set((fusion.get("driven") or {}).values()))
    released = _as_list(fusion.get("released_here") or [])
    row = {
        "leg": "release-route", "case": case,
        "table": report.get("table"),
        "switch_value_in_the_record": fusion.get("value"),
        "opted_in": _as_list(fusion.get("opted_in") or []),
        "released_here": released,
        "outside_the_released_envelope": _as_list(
            fusion.get("outside_the_released_envelope") or []),
        "driven": dict(fusion.get("driven") or {}),
        "driver_route_gate": fusion.get("driver_route_gate"),
        "decision": report.get("decision"),
        "active_step_path": path,
        "the_switch_was_not_used": (fusion.get("value") is None
                                    and not (fusion.get("opted_in") or [])),
        "the_release_covers_every_driven_arm": (
            bool(driven) and isinstance(released, list)
            and set(driven) <= set(released)),
    }
    row["verdict"] = ("RELEASE-ADMITTED"
                      if (row["the_switch_was_not_used"]
                          and row["the_release_covers_every_driven_arm"]
                          and report.get("decision") == "dispatched"
                          and path == "fused")
                      else "RELEASE-DID-NOT-ADMIT")
    say(f"{case}: release-route leg -> {row['verdict']} :: switch="
        f"{row['switch_value_in_the_record']!r} released={released} driven={driven}")
    return row


def envelope_leg(res: Optional[int]) -> Dict[str, Any]:
    """The ENVELOPE, both directions, on a shape the release does not cover.

    IT DRIVES TWO CASES FROM 2026-09-12, one per direction, and
    :data:`ENVELOPE_CASE` records why the single case it used to drive can no
    longer witness both.

    DIRECTION 1 — THE SHARED ENVELOPE REFUSES THE REQUEST. With the switch unset on
    :data:`ENVELOPE_CASE` (``no_pml_2d``), ``released_fused_arms_metal`` is empty,
    so the composer is asked with ``fuse=False``, nothing fuses, and the record must
    NAME :data:`ENVELOPE_AXIS` (``pml_active``). Without this the envelope is a
    null: a predicate that has only ever said yes is not known to be able to say no.
    The axis is read from the constant rather than hardcoded because it has already
    changed once — it was ``off_diagonal_epsilon`` until that axis went per-arm.

    WHICH CONJUNCT CARRIES THE WEIGHT HERE, said plainly because it changed with the
    case. Direction 1 is ``names_the_axis and nothing_fused and not released_here``.
    On ``no_pml_2d`` the ``nothing_fused`` conjunct is NON-DISCRIMINATING: no fused
    product is a candidate on an absorber-free grid at all, so it would read true
    against a broken envelope too. The weight is on the other two — the released set
    must be EMPTY and the record must NAME the axis — and those are what a broken
    shared table would fail. ``nothing_fused`` is kept because it costs nothing and
    would catch the different bug of something fusing anyway, not because it is
    evidence here.

    THE AXIS IS READ FROM BOTH HALVES OF THE RELEASE. The shared envelope refuses
    every arm at once; each arm's own axes refuse only it, and the record publishes
    the second as ``outside_this_arms_own_cases``. A leg reading only the first
    could find it empty on a shape refused per-arm and report the envelope broken
    while it was working.

    DIRECTION 2 — WHAT AN UN-ADMITTED PRODUCT DOES, on
    :data:`ENVELOPE_OFFER_CASE` (``pml_3d``). The composer is handed the admitted
    LABEL SET, so a product this run cannot admit is never installed, its seam keeps
    the separate certified arms, and the plan DISPATCHES the rest. MEASURED on this
    case 2026-09-10: opting into ``fused magnetic B/H pair`` alone installs it on
    ``step_B``/``update_H`` while ``step_D``/``update_E`` keep ``PML`` and
    ``offdiag``, with 42 ``offered to run`` refusals recorded for the products that
    were not offered. The same shape holds on the RELEASE route there from
    2026-09-12 — that arm is admitted and every other arm is refused by its own
    axes or by the corner table — so this direction is now witnessed twice, by two
    routes.

    DIRECTION 3 — clause 8M is a BACKSTOP and the artifact says so: on both legs
    every DRIVEN fused label must be one the ladder admitted, which is the property
    that makes the clause unreachable from the shipped composer.

    All three must hold. Any one alone is satisfiable by a broken predicate.
    """
    unset_report, unset_path = _one_step_report(
        ENVELOPE_CASE, _fuse_env([], "release"), res)
    unset_fusion = dict(unset_report.get("fusion") or {})
    shared = _as_list(unset_fusion.get("outside_the_released_envelope") or [])
    if not isinstance(shared, list):
        # The ladder refused before the run shape was read, so the envelope was
        # never consulted. Recorded as the sentence it is rather than split.
        shared = [str(shared)]
    per_arm = dict(unset_fusion.get("outside_this_arms_own_cases") or {})
    reasons = shared + [f"{arm}: {why}" for arm, whys in sorted(per_arm.items())
                        for why in whys]

    opt_report, opt_path = _one_step_report(
        ENVELOPE_OFFER_CASE, _fuse_env(ENVELOPE_OPT_IN, "opt-in"), res)
    opt_fusion = dict(opt_report.get("fusion") or {})
    opt_driven = sorted(set((opt_fusion.get("driven") or {}).values()))
    opt_slots = {name: entry.get("arm")
                 for name, entry in (opt_report.get("slots") or {}).items()
                 if entry.get("state") == "dispatched"}
    opt_reasons = dict(opt_fusion.get("not_installed") or {})
    offer_reasons = [f"{product}: {text}" for product, text
                     in sorted(opt_reasons.items())
                     if "offered to run" in str(text)]

    row: Dict[str, Any] = {
        "leg": "envelope", "case": ENVELOPE_CASE,
        "offer_case": ENVELOPE_OFFER_CASE,
        "switch_unset": {
            "released_here": _as_list(unset_fusion.get("released_here") or []),
            "outside_the_released_envelope": shared,
            "outside_this_arms_own_cases": per_arm,
            "driven": dict(unset_fusion.get("driven") or {}),
            "decision": unset_report.get("decision"),
            "active_step_path": unset_path,
            "names_the_axis": any(ENVELOPE_AXIS in reason
                                  for reason in reasons),
            "the_axis_it_must_name": ENVELOPE_AXIS,
            "nothing_fused": not (unset_fusion.get("driven") or {}),
        },
        "opted_into_a_subset": {
            "opted_in": ENVELOPE_OPT_IN,
            "not_admitted": ENVELOPE_UNNAMED,
            "admitted": _as_list(opt_fusion.get("admitted") or []),
            "driven": opt_driven,
            "decision": opt_report.get("decision"),
            "active_step_path": opt_path,
            "dispatched_slots": opt_slots,
            "the_named_arm_was_driven": set(opt_driven) == set(ENVELOPE_OPT_IN),
            "the_unnamed_arm_was_not": not (set(ENVELOPE_UNNAMED) & set(opt_driven)),
            # The un-offered seam must be SERVED, not emptied: a plan that merely
            # refused to fuse step_D/update_E and left them on the array path would
            # satisfy the two clauses above and still be the regression this
            # direction exists to measure.
            "the_unoffered_seam_kept_its_arms": all(
                opt_slots.get(slot) for slot in ("step_D", "update_E")),
            "the_composer_named_the_offer": bool(offer_reasons),
            "offer_reasons": offer_reasons[:2],
            "refused_because": opt_report.get("refused_because") or "",
        },
    }
    one = (row["switch_unset"]["names_the_axis"]
           and row["switch_unset"]["nothing_fused"]
           and not row["switch_unset"]["released_here"])
    row["the_ladder_reached_the_run_shape"] = isinstance(
        unset_fusion.get("released_here"), (list, tuple))
    subset = row["opted_into_a_subset"]
    two = (opt_report.get("decision") == "dispatched"
           and opt_path == "fused"
           and subset["the_named_arm_was_driven"]
           and subset["the_unnamed_arm_was_not"]
           and subset["the_unoffered_seam_kept_its_arms"]
           and subset["the_composer_named_the_offer"])
    three = True
    backstop = []
    for name, report in (("switch_unset", unset_report),
                         ("opted_into_a_subset", opt_report)):
        fusion = dict(report.get("fusion") or {})
        driven = set((fusion.get("driven") or {}).values())
        admitted_value = fusion.get("admitted") or ()
        admitted = set(admitted_value) if isinstance(admitted_value,
                                                     (list, tuple, set)) else set()
        backstop.append({"leg": name, "driven": sorted(driven),
                         "admitted": sorted(admitted),
                         "driven_is_a_subset_of_admitted": driven <= admitted})
        three = three and driven <= admitted
    row["clause_8_backstop"] = backstop
    row["direction_1_envelope_refuses_the_request"] = one
    row["direction_2_the_offer_keeps_the_unadmitted_product_out"] = two
    row["direction_3_no_fused_label_reaches_clause_8_unadmitted"] = three
    row["verdict"] = ("ENVELOPE-HOLDS" if (one and two and three)
                      else "ENVELOPE-DID-NOT-HOLD")
    say(f"{ENVELOPE_CASE}: envelope leg -> {row['verdict']} :: unset "
        f"reasons={reasons[:1]} | subset driven={opt_driven} path={opt_path} "
        f"offer_named={subset['the_composer_named_the_offer']}")
    return row


def second_consult_site(case: str, legs: Dict[str, Dict[str, Any]],
                        counter: MetalLaunchCounter,
                        spec: Dict[str, Any]) -> Dict[str, Any]:
    """Exercise ``synchronize_magnetic_fields`` on every leg and compare.

    THE SEAM HAS SEVEN CONSULTS, NOT FIVE. ``synchronize_magnetic_fields`` runs
    ``step_B`` and ``update_H`` again off a plan it deliberately does NOT re-freeze,
    with the same unconsulted in-seam passes between them; any flux or energy
    monitor reaches it. A gate that drove only ``step()`` has not driven the seam.

    ON THIS TABLE IT IS ALSO THE RESIDENCY'S HARDEST SITE: the call backs the
    magnetic arrays up on the HOST, steps them, and restores them in place, all
    outside any bracket. Per-launch sync is what makes that correct with no driver
    change, and this leg is the measurement of it.

    A CASE WITH NO MONITOR IS NOT-APPLICABLE rather than a pass: the driver's own
    ``synchronize_magnetic_fields`` can be called anyway, but on a case with no
    observable the comparison that follows it has nothing to be about, and calling
    that a pass would put a green word on a measurement never taken.
    """
    out: Dict[str, Any] = {"site": "synchronize_magnetic_fields", "case": case}
    if not spec.get("monitors"):
        out["verdict"] = "NOT-APPLICABLE"
        out["why"] = (f"{case} carries no flux monitor, so the second consult site "
                      "has no observable to be measured against")
        return out
    before: Dict[str, Any] = {}
    after: Dict[str, Any] = {}
    launches: Dict[str, Any] = {}
    for label, leg in legs.items():
        driver = leg["driver"]
        before[label] = e2e.collect_state(driver)
        e2e._apply_env(leg)  # noqa: SLF001
        snap = counter.snapshot()
        driver.synchronize_magnetic_fields()
        launches[label] = MetalLaunchCounter.delta(snap, counter.snapshot())
        after[label] = e2e.collect_state(driver)
    out["launches_during_sync"] = launches
    from meep_gpu import fastpath as _fastpath  # noqa: PLC0415 - late, as above

    out["sync_pass_name"] = _fastpath.SYNC_UPDATE_H_PASS
    out["sync_refusals"] = {
        label: dict(getattr(getattr(leg["driver"], "_fast_path", None),
                            "sync_refusals", {}) or {})
        for label, leg in legs.items()}
    out["no_pair_was_declined_at_the_second_site"] = not any(
        out["sync_refusals"].values())
    out["synchronized_state"] = {
        "fused_vs_array": e2e.compare_state(after["fused"],
                                            after["array"])["identical"],
        "unfused_vs_array": e2e.compare_state(after["unfused"],
                                              after["array"])["identical"],
        "sync_changed_the_state": {
            label: not e2e.compare_state(before[label], after[label])["identical"]
            for label in legs},
    }
    restored: Dict[str, Any] = {}
    for label, leg in legs.items():
        leg["driver"].restore_magnetic_fields()
        restored[label] = e2e.collect_state(leg["driver"])
    out["restore_returns_the_state"] = {
        label: e2e.compare_state(before[label], restored[label])["identical"]
        for label in legs}
    out["restored_fused_vs_array"] = e2e.compare_state(
        restored["fused"], restored["array"])["identical"]
    out["launch_drop_in_sync"] = (launches["unfused"]["total"]
                                  - launches["fused"]["total"])
    out["array_leg_launched"] = launches["array"]["total"]
    out["verdict"] = (
        "PASS" if (out["synchronized_state"]["fused_vs_array"]
                   and out["restored_fused_vs_array"]
                   and launches["array"]["total"] == 0
                   and out["no_pair_was_declined_at_the_second_site"])
        else "FAIL")
    say(f"{case}: second consult site -> {out['verdict']} (sync launches "
        f"fused={launches['fused']['total']} unfused={launches['unfused']['total']} "
        f"array={launches['array']['total']})")
    return out


# ---------------------------------------------------------------------------
# One case
# ---------------------------------------------------------------------------

def run_case(case: str, counter: MetalLaunchCounter, res: Optional[int],
             max_steps: Optional[int]) -> Dict[str, Any]:
    spec = DRIVE[case]
    builder = CASES[case]
    row: Dict[str, Any] = {
        "case": case, "intent": CASE_INTENT.get(case, "?"),
        "why_this_case": spec["why"], "arms_requested": spec["arms"],
        "pairs_expected": spec["pairs"], "expect": spec["expect"],
        "needs_probe": spec.get("needs_probe"),
        "probe_path": (os.environ.get(spec["needs_probe"])
                       if spec.get("needs_probe") else None),
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    # A CASE WHOSE LICENCE THIS LEG DOES NOT CARRY IS NOT THIS LEG'S EVIDENCE.
    # The complex arms take their expansion licence from a probe record named by an
    # environment variable, and the probe leg is the one that exports it. Run
    # anywhere else the arm refuses by name, the case falls to the array path, and
    # the leg used to record that as DID-NOT-FUSE -- a FAILING verdict that says
    # "this product did not dispatch" when the honest statement is "this leg was
    # never the one that could dispatch it". Every leg of the campaign then failed
    # on two cases that only one leg was ever meant to drive.
    #
    # THE TEST IS THE ENVIRONMENT, NOT THE LEG'S NAME, so a probe leg that forgot to
    # export the variable is still a failure rather than a skip.
    needs = spec.get("needs_probe")
    if needs and not os.environ.get(needs):
        row["verdict"] = "PASS-NOT-THIS-LEG"
        row["why"] = (f"this case's expansion licence comes from the record "
                      f"{needs} names, and this leg exports no such path; the "
                      f"arms {spec['arms']} cannot be admitted here and the "
                      f"campaign drives them on shipped_expansion_probe")
        say(f"{case}: {needs} unset -> PASS-NOT-THIS-LEG (the probe leg drives it)")
        return row

    say(f"{case}: lifting three legs (fused / unfused / array)")
    fused = e2e.run_leg(case, builder, "fused",
                        _fuse_env(spec["arms"], spec.get("reached_by", "release")),
                        counter, 0, res)
    unfused = e2e.run_leg(case, builder, "unfused", unfused_env(), counter, 0, res)
    array = e2e.run_leg(case, builder, "array", ARRAY_ENV, counter, 0, res)
    legs = {"fused": fused, "unfused": unfused, "array": array}
    for label, leg in legs.items():
        require_metal_driver(leg["driver"], f"{case}/{label}")
    row["grid_shape"] = fused["grid_shape"]

    states = {label: e2e.collect_state(leg["driver"]) for label, leg in legs.items()}
    pre_fa = e2e.compare_state(states["fused"], states["array"])
    pre_ua = e2e.compare_state(states["unfused"], states["array"])
    row["precondition"] = {"fused_vs_array": pre_fa["identical"],
                           "unfused_vs_array": pre_ua["identical"],
                           "arrays": pre_fa["arrays"],
                           "words": pre_fa["words_compared"]}
    if not (pre_fa["identical"] and pre_ua["identical"]):
        row["verdict"] = "HARNESS-FAILURE"
        row["why"] = ("the legs were lifted to DIFFERENT initial states, so no "
                      "downstream comparison is about the Metal route")
        for leg in legs.values():
            leg["driver"].close()
        return row
    say(f"{case}: precondition OK — {pre_fa['arrays']} arrays, "
        f"{pre_fa['words_compared']} words identical on all three legs")

    dt = float(fused["driver"].grid.dt)
    total = max(1, int(round(fused["until"] / dt)))
    if max_steps is not None:
        total = min(total, max_steps)
    ladder = ladder_for(total)
    row["total_steps"] = total
    row["checkpoints_planned"] = ladder
    row["checkpoints"] = []
    row["first_divergent_checkpoint"] = None

    # THE SEAM CENSUS rides the fused leg for its whole run, recorded as provenance.
    census_restore, census_state = _seam_census(fused["driver"])
    taken = 0
    try:
        for point in ladder:
            for leg in (fused, unfused, array):
                step_leg(case, leg, counter, point - taken)
            taken = point
            snap = {label: e2e.collect_state(leg["driver"])
                    for label, leg in legs.items()}
            fa = e2e.compare_state(snap["fused"], snap["array"])
            ua = e2e.compare_state(snap["unfused"], snap["array"])
            entry = {"steps": point,
                     "fused_vs_array_identical": fa["identical"],
                     "unfused_vs_array_identical": ua["identical"],
                     "arrays": fa["arrays"], "words": fa["words_compared"],
                     "fused_arrays_differing": fa["arrays_differing"],
                     "unfused_arrays_differing": ua["arrays_differing"],
                     "differences": fa["differences"][:6]}
            row["checkpoints"].append(entry)
            say(f"{case}: checkpoint {point}/{total} -> fused"
                f"{'==' if fa['identical'] else '!='}array, unfused"
                f"{'==' if ua['identical'] else '!='}array")
            if not fa["identical"] and row["first_divergent_checkpoint"] is None:
                row["first_divergent_checkpoint"] = point
                row["first_divergence"] = entry
    finally:
        census_restore()
    row["seam_census"] = dict(census_state)

    row["steps_taken"] = {label: leg["steps"] for label, leg in legs.items()}
    row["wall_s"] = {label: leg["wall_s"] for label, leg in legs.items()}
    finals = {label: e2e.collect_state(leg["driver"]) for label, leg in legs.items()}
    spectra = {label: e2e.read_spectra(leg["driver"]) for label, leg in legs.items()}
    row["final_state_fused_vs_array"] = e2e.compare_state(finals["fused"],
                                                          finals["array"])
    row["final_state_unfused_vs_array"] = e2e.compare_state(finals["unfused"],
                                                            finals["array"])
    row["observables"] = e2e.compare_spectra(spectra["fused"], spectra["array"])
    row["comparator_control"] = e2e.comparator_negative_control(finals["fused"])
    row["array_leg_control"] = array_leg_is_clean(array)
    row["legs"] = {label: {"plan": leg["plan"], "fusion": fusion_block(leg),
                           "metal_launches": leg["metal_launches"],
                           "chunks": leg["chunks"],
                           "steady_state": steady_state_rate(leg)}
                   for label, leg in legs.items()}
    row["substitution"] = substitution_proof(fused, unfused, spec)
    row["evidence"] = fused_leg_is_real(fused, spec)
    row["second_consult_site"] = second_consult_site(case, legs, counter, spec)
    row["policy"] = e2e._policy_stamp()  # noqa: SLF001

    for leg in legs.values():
        leg["driver"].close()

    # ------------------------------------------------------------------ verdict
    agree = (row["final_state_fused_vs_array"]["identical"]
             and row["observables"]["identical"]
             and row["first_divergent_checkpoint"] is None)
    control = row["comparator_control"]
    dispatched = (fused["active_step_path"] == "fused"
                  and bool(row["evidence"]["driven"]))

    if UNCERTIFIED_POLICY_INSTALLED is not None and spec["expect"] == "dispatch":
        # THE SEAM SUPPORTS ONE POLICY, AND THIS IS THE LEG THAT SAYS SO. The
        # harness installed a policy this table was not certified under, so the
        # required outcome inverts: rung 8bM must refuse BY NAME, nothing may fuse,
        # and the run must still be byte-identical to the array path, because a
        # refusal that also changed the answer would be two failures wearing one
        # word.
        report = fused.get("report") or {}
        reason = report.get("refused_because") or ""
        # THE REFUSAL IS WHEREVER THE LADDER PUT IT, AND UNDER AN INSTALLED KEEP IT
        # IS PER SLOT. The composer's own backend clause
        # (``metal_kernels.subnormal.mps_policy_reasons``) refuses every ARM under a
        # resolved keep, so the plan is empty before rung 8bM is reached and the
        # top-level string is the generic "no slot is left carrying a kernel". Each
        # slot's own reason names the policy in full -- "the resolved float32
        # subnormal policy is 'keep' (from the installed policy), and the MPS
        # executor cannot honour it" -- so reading only the top-level string scored a
        # correctly named refusal as POLICY-REFUSAL-MISSING on all nine dispatching
        # cases (measured 2026-09-11). Both places are read and the artifact records
        # WHICH one named it, because a refusal that moves between rungs is a change
        # in what is refused.
        slot_reasons = dict((report.get("slots") or {}))
        per_slot = {name: str((entry or {}).get("reason") or "")
                    for name, entry in slot_reasons.items()}
        if not per_slot:
            per_slot = {name: str(text) for name, text
                        in ((report.get("array_slot_reasons") or {}) or {}).items()}
        named_in_top = all(needle in reason for needle in POLICY_REFUSAL_NEEDLES)
        naming_slots = sorted(
            name for name, text in per_slot.items()
            if all(needle in text for needle in POLICY_REFUSAL_NEEDLES))
        named = named_in_top or bool(naming_slots)
        row["policy_refusal"] = {
            "harness_installed": UNCERTIFIED_POLICY_INSTALLED,
            "named_by": ("the plan's own refusal" if named_in_top
                         else f"{len(naming_slots)} slot reason(s)" if naming_slots
                         else None),
            "slots_naming_the_policy": naming_slots,
            "refused_because": reason,
            "refusing_rung": classify_refusal(reason) if reason else None,
            "names_the_policy_rung": named,
            "names_the_installed_policy": (
                repr(UNCERTIFIED_POLICY_INSTALLED) in reason
                or any(repr(UNCERTIFIED_POLICY_INSTALLED) in per_slot[name]
                       for name in naming_slots)),
            "did_not_dispatch": not dispatched,
            "byte_identical_to_array": agree,
        }
        if dispatched:
            row["verdict"] = "DISPATCHED-UNDER-AN-UNCERTIFIED-POLICY"
            row["why"] = (f"the harness installed {UNCERTIFIED_POLICY_INSTALLED!r} "
                          "and a fused pair dispatched anyway; every Metal family "
                          "was certified under flush and MPS cannot keep")
        elif not named:
            row["verdict"] = "POLICY-REFUSAL-MISSING"
            row["why"] = (f"nothing dispatched, but the refusal does not name the "
                          f"subnormal policy: {reason!r}")
        elif not agree:
            row["verdict"] = "DIVERGENCE"
            row["why"] = ("the policy rung refused as required and the run still "
                          "differed from the array path")
        else:
            row["verdict"] = "PASS-POLICY-REFUSED"
        return row

    if spec["expect"] == "no-fused-arm":
        # A PRODUCT THE COMPOSER SELECTS AND THE RELEASE DOES NOT ADMIT. What must
        # hold is that no FUSED product is driven here and the run is byte-identical
        # to the array path whichever way the plan goes. When the plan IS refused
        # the rung is classified and recorded, so a refusal that moves between rungs
        # is visible rather than absorbed.
        report = fused.get("report") or {}
        reason = report.get("refused_because") or ""
        driven = ((report.get("fusion") or {}).get("driven") or {})
        row["no_fused_arm"] = {
            "refused_because": reason,
            "refusing_rung": classify_refusal(reason) if reason else None,
            "the_plan_was_refused": bool(reason),
            "fused_arms_driven": dict(driven),
            "dispatched_slots": fusion_block(fused)["dispatched_slots"],
            "unfused_leg_dispatched": unfused["active_step_path"] == "fused",
            "byte_identical_to_array": agree,
        }
        if driven:
            row["verdict"] = "FUSED-WITHOUT-A-RELEASE"
            row["why"] = (f"a fused product was driven here "
                          f"({sorted(set(driven.values()))}); the release does not "
                          "admit one and the composer is not offered one")
        elif reason and classify_refusal(reason) is None:
            row["verdict"] = "REFUSAL-CHANGED"
            row["why"] = (f"the plan was refused and the refusal matches no rung "
                          f"this gate knows: {reason!r}")
        elif not agree:
            row["verdict"] = "DIVERGENCE"
        else:
            row["verdict"] = "PASS-NO-FUSED-ARM"
        return row

    if not agree:
        row["verdict"] = "DIVERGENCE"
        if (row["final_state_fused_vs_array"]["identical"]
                and row["observables"]["identical"]):
            row["why"] = ("the FINAL state and the observables agree, but the fused "
                          f"and array paths differed at "
                          f"{row['first_divergent_checkpoint']} steps and the "
                          "difference was later absorbed; they computed different bits")
    elif not (control.get("armed") and control.get("comparator_saw_it")
              and control.get("words_mismatched") == 1):
        row["verdict"] = "COMPARATOR-BLIND"
    elif not dispatched:
        row["verdict"] = "DID-NOT-FUSE"
        row["why"] = ((fused.get("report") or {}).get("refused_because")
                      or "the fused route was asked for and did not dispatch")
        row["refusing_rung"] = classify_refusal(row["why"]) if row["why"] else None
    elif not row["evidence"]["real"]:
        row["verdict"] = "VACUOUS-PASS"
        row["why"] = "; ".join(row["evidence"]["failures"])
    elif not row["array_leg_control"]["clean"]:
        row["verdict"] = "CONTROL-FAILURE"
    elif row["substitution"]["verdict"] != "EXACT":
        # EVERY VERDICT BUT EXACT REFUSES HERE, DECLARATION-MISMATCH included: a
        # drive row whose collapsed-single declaration the unfused leg's own counter
        # contradicts has no expected drop to be exact against.
        row["verdict"] = "NO-SUBSTITUTION"
        row["why"] = (row["substitution"].get("why")
                      or "the fused leg did not save the launches a step the drive "
                         "row declares against the unfused leg, so the byte identity "
                         "is about two spellings of the same launches")
    elif row["second_consult_site"]["verdict"] not in ("PASS", "NOT-APPLICABLE"):
        row["verdict"] = "SECOND-CONSULT-FAILURE"
    else:
        row["verdict"] = "PASS-FUSED"
    return row


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

#: ``PASS-NOT-THIS-LEG`` is a SKIP, not a pass about the product: the case declares
#: an expansion-probe record this leg does not export, so no arm of it could be
#: admitted here. It keeps a leg from failing on cases a different leg drives, and
#: it cannot mask a real refusal because the probe leg DOES export the variable and
#: so never reaches it.
PASS_VERDICTS = ("PASS-FUSED", "PASS-POLICY-REFUSED", "PASS-NO-FUSED-ARM",
                 "PASS-NOT-THIS-LEG")
#: Every verdict a control may carry and still let the leg release. ``NO-CLIFF`` is
#: NOT among them, deliberately: a released pair that ran under a keeping host and
#: did not move is either a run that never entered the band or a device that did not
#: flush, and both make the policy rung's refusal a precaution rather than a rule.
CONTROL_PASS = ("DIVERGED-AS-REQUIRED", "NOT-APPLICABLE", "REFUSED-BY-NAME",
                "RELEASE-ADMITTED", "ENVELOPE-HOLDS", "RESIDENCY-CLEAN",
                "BAND-NON-EMPTY", "CLIFF-CONFINED-TO-THE-BAND", "PASS")


def _provenance() -> Dict[str, Any]:
    """The bytes, the host, the toolchain, and the policy this run measured under."""
    import hashlib  # noqa: PLC0415
    import platform  # noqa: PLC0415

    package = os.path.join(_API, "meep_gpu")
    digests: Dict[str, str] = {}
    # ``deposit_repair.py`` IS THE BRACKET, and it joined this list 2026-09-19. Its
    # ``LeadingRepairPlan`` / ``TrailingRepairPlan`` wrap every bracketed seam's launch
    # (``_drop_the_inner_bracket`` reads them slot by slot), yet the five legs of
    # ``dispatch_metal_route_2026-09-17_allpaths`` name it only in the top-level
    # ``imported_source_sha256`` (``c1d55704``): this block -- 11 keys on each leg, the
    # one ``recut_driver_dispatch_record.py --backend metal`` compares -- did not.
    # Package-relative like its neighbours, so ``_leg_key`` finds it if the record ever
    # binds ``meep_gpu/deposit_repair.py``. Recording it binds nothing by itself: the
    # recut binds the key set of the record it cuts
    # (``recut_driver_dispatch_record.py:310``) and ignores extra leg keys.
    for name in ("fastpath.py", "driver.py", "fields.py", "metal_dispatch.py",
                 "subnormal_policy.py", "deposit_repair.py", "metal_kernels/launch.py",
                 "metal_kernels/device.py", "metal_kernels/subnormal.py",
                 "metal_kernels/arms.py", "metal_kernels/fingerprints.json"):
        path = os.path.join(package, name)
        try:
            with open(path, "rb") as handle:
                digests[name] = hashlib.sha256(handle.read()).hexdigest()
        except OSError as exc:
            digests[name] = f"unreadable: {exc!r}"
    # THE GATE ITSELF, keyed the way the LEDGER spells a parity path rather than the
    # way this file's siblings are keyed. `mint_metal_weld` requires every pinned
    # path to appear in the artifact's own record — a weld may only bind bytes the
    # gate is recorded as having imported — and
    # `test_every_metal_weld_pins_the_script_that_gated_it` requires one of those
    # pins to be a parity/ path. Recording it under the repo-relative key is what
    # lets `recut_driver_dispatch_record.py --backend metal` compare leg, leg and
    # tree on it the same way it compares the package files.
    _this = "parity/meep_gpu/gate_dispatch_metal_route.py"
    with open(os.path.abspath(__file__), "rb") as handle:
        digests[_this] = hashlib.sha256(handle.read()).hexdigest()
    record: Dict[str, Any] = {
        "gate": "dispatch_metal_route",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": platform.node(), "machine": platform.machine(),
        "python": sys.version.split()[0],
        "source_sha256": digests,
        "this_script_sha256": digests[_this],
        # HOW EVERY LEG WAS LIFTED, stamped since 2026-09-27 so the driver_dispatch
        # record reads it off the legs. An artifact without this block was lifted
        # ``prefer_gpu=False`` under the enable, which planned Metal before that date.
        "lift": {"prefer_gpu": True, "driver_gpu": "metal",
                 "checked": "driver.gpu == 'metal' after every lift "
                            "(require_metal_driver)"},
        "expansion_probes": {
            name: os.environ.get(name) for name in (
                "MEEP_GPU_METAL_COMPLEX_EXPANSION_PROBE",
                "MEEP_GPU_METAL_EXPANSION_PROBE",
                "MEEP_GPU_METAL_FOLDED_COMPLEX_EXPANSION_PROBE",
                "MEEP_GPU_METAL_CYLINDRICAL_COMPLEX_EXPANSION_PROBE")},
    }
    for module in ("meep", "numpy", "torch"):
        try:
            record[module] = __import__(module).__version__
        except Exception as exc:  # noqa: BLE001
            record[module] = f"unavailable: {exc!r}"
    try:
        from meep_gpu import metal_dispatch  # noqa: PLC0415
        from meep_gpu.metal_kernels import device, subnormal  # noqa: PLC0415

        record["metal_frontend"] = device.metal_frontend_version()
        # THE STAMP THE ARTIFACT CARRIES, and it is not decoration: the claim is only
        # as good as the precondition it was certified under, and this arm's answer
        # can differ from `policy_resolution`'s because it also honours an install
        # and a dispatch's declaration.
        record["mps_policy_report"] = subnormal.mps_policy_report()
        record["table"] = {
            "driver_route_gate": metal_dispatch.METAL_DRIVER_ROUTE_GATE,
            "subnormal_policy": metal_dispatch.certification_policy(),
            "governed_executors": list(metal_dispatch.GOVERNED_EXECUTORS),
            "released_fused_arms": {
                arm: list(cases) for arm, cases
                in metal_dispatch.METAL_RELEASED_FUSED_ARMS.items()},
            "pending_device_gate_arms": sorted(
                metal_dispatch.METAL_PENDING_DEVICE_GATE_ARMS),
            "validated_toolchains": [list(pair) for pair
                                     in metal_dispatch.validated_toolchains()],
            "unresolved_certification_rows": list(
                metal_dispatch.unresolved_certification_rows()),
            "release_rows_without_a_weld": list(
                metal_dispatch.release_rows_without_a_weld()),
        }
    except Exception as exc:  # noqa: BLE001 - an unreadable table is recorded, not fatal
        record["table"] = {"unreadable": repr(exc)}
    return record


def main(argv: Optional[Sequence[str]] = None) -> int:
    global UNCERTIFIED_POLICY_INSTALLED  # noqa: PLW0603

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True,
                        help="LEG DIRECTORY; the artifact is <out>/gate.json")
    parser.add_argument("--cases", default="all")
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None,
                        help="cap the ladder; the full run is the default")
    parser.add_argument("--install-policy", default=None,
                        help="install this float32 subnormal policy before anything "
                             "else. UNSET IS THE SHIPPED LEG: it leaves the install "
                             "to rung 8bM, which is precisely the evidence "
                             "DISPATCH_BY_DEFAULT would wait on.")
    parser.add_argument("--band-witness", action="store_true",
                        help="the array leg only, censusing every stored array every "
                             "step with the host KEEPING: the vacuity floor under "
                             "the flush-equivalence claim")
    parser.add_argument("--no-controls", action="store_true",
                        help="skip the armed controls (diagnostic only; a run made "
                             "with this is not a gate run)")
    arguments = parser.parse_args(argv)

    # A METAL GATE, REFUSED OFF AN APPLE GPU before its leg directory exists.
    refusal = metal_host_refusal("gate_dispatch_metal_route.py")
    if refusal:
        print(refusal, file=sys.stderr)
        return 2

    os.makedirs(arguments.out, exist_ok=True)
    # A GATE MUST NOT WRITE INTO ANOTHER RUN'S ARTIFACT. The JSONL files are opened
    # in APPEND mode because rule 7 wants every row on disk the moment it lands; the
    # cost is that pointing --out at a directory a previous run wrote silently
    # INTERLEAVES two runs' rows in one file while summary.json is rewritten from
    # this process's own list, so the two disagree and only the JSONL carries the
    # contamination. Refusing is the fix; truncating would destroy the earlier run.
    existing = sorted(name for name in ("cases.jsonl", "controls.jsonl",
                                        "summary.json", "gate.json")
                      if os.path.exists(os.path.join(arguments.out, name)))
    if existing:
        print(f"REFUSING: {arguments.out} already holds a previous run's "
              f"{', '.join(existing)}. This gate appends its rows, so writing here "
              f"would interleave two runs in one artifact. Point --out at a fresh "
              f"directory.", file=sys.stderr)
        return 5

    e2e._PROGRESS_PATH = os.path.join(arguments.out, "progress.log")  # noqa: SLF001
    # LIFTED AS AN APPLE GPU DRIVER, ALWAYS. `prefer_gpu=True` resolves this host's
    # GPU, which on the host `metal_host_refusal` admitted is Metal: NumPy host arrays
    # with device mirrors beside them. A `prefer_gpu=False` lift is the NumPy
    # reference and never plans, whatever MEEP_GPU_DISPATCH says, so it cannot be a
    # fused or unfused leg; the array oracle is the same GPU driver under
    # MEEP_GPU_FUSED=0.
    e2e.PREFER_GPU = True
    rows_path = os.path.join(arguments.out, "cases.jsonl")

    names = (list(DRIVE) if arguments.cases == "all"
             else [n.strip() for n in arguments.cases.split(",") if n.strip()])
    unknown = [n for n in names if n not in DRIVE]
    if unknown:
        print(f"unknown cases: {unknown}; known: {sorted(DRIVE)}", file=sys.stderr)
        return 2

    # MEEP FIRST, and not as a formality: `install_host_policy` drives the host FPU
    # through MEEP's own `set_zero_subnormals` before falling back to the fenv lever,
    # and a request it cannot verify is reported UNATTAINABLE.
    import meep  # noqa: PLC0415,F401
    import meep_gpu  # noqa: PLC0415

    provenance = _provenance()
    provenance["install_policy_requested"] = arguments.install_policy
    provenance["harness_installs_nothing"] = arguments.install_policy is None
    provenance["band_witness_leg"] = bool(arguments.band_witness)
    if arguments.install_policy:
        try:
            provenance["installed_policy"] = dict(
                meep_gpu.install_subnormal_policy(arguments.install_policy,
                                                  executors=("host", "mps"),
                                                  strict=False))
            say(f"installed subnormal policy {arguments.install_policy!r} on "
                "host + mps")
        except BaseException as exc:  # noqa: BLE001
            say(f"REFUSING: install_subnormal_policy raised {exc!r}")
            return 4
        # WHICH LEG THIS IS, decided against the TABLE's own constant rather than a
        # literal here, so a change to the certified policy moves this leg with it
        # instead of leaving a gate asserting a policy nobody certifies any more.
        from meep_gpu import metal_dispatch  # noqa: PLC0415

        certified = metal_dispatch.certification_policy()
        provenance["certification_policy"] = certified
        if arguments.install_policy != certified:
            UNCERTIFIED_POLICY_INSTALLED = arguments.install_policy
            say(f"the harness installed {arguments.install_policy!r} and every "
                f"Metal family was certified under {certified!r}: this leg REQUIRES "
                "rung 8bM to refuse by name on every dispatch case, with the run "
                "still byte-identical to the array path")
    provenance["uncertified_policy_installed"] = UNCERTIFIED_POLICY_INSTALLED
    provenance["policy_at_start"] = e2e._policy_stamp()  # noqa: SLF001
    with open(os.path.join(arguments.out, "provenance.json"), "w",
              encoding="utf-8") as handle:
        json.dump(provenance, handle, indent=1, default=str)
    say(f"host={provenance['host']} torch={provenance.get('torch')} "
        f"meep={provenance.get('meep')} frontend={provenance.get('metal_frontend')} "
        f"policy={provenance['policy_at_start']}")

    counter = MetalLaunchCounter()
    rows: List[Dict[str, Any]] = []
    controls: List[Dict[str, Any]] = []

    def record(entry: Dict[str, Any], case_name: Optional[str] = None) -> None:
        """Append one control row and FLUSH IT, one row per control (rule 7)."""
        if case_name is not None:
            entry.setdefault("case", case_name)
        controls.append(entry)
        with open(os.path.join(arguments.out, "controls.jsonl"), "a",
                  encoding="utf-8") as handle:
            handle.write(json.dumps(entry, default=str) + "\n")
            handle.flush()

    # ---------------------------------------------------------------- band witness
    if arguments.band_witness:
        # THE ARRAY PATH AGAINST ITS OWN ARITHMETIC, and nothing else. This leg
        # dispatches nothing by construction: it is the measurement that the value
        # class the flush-equivalence claim is ABOUT is a class this corpus's own
        # trajectories construct.
        for index, name in enumerate(names, 1):
            say(f"=== band witness {index}/{len(names)}: {name} ===")
            try:
                entry = band_witness(name, arguments.resolution, CONTROL_STEPS * 2)
            except BaseException as exc:  # noqa: BLE001
                import traceback  # noqa: PLC0415
                entry = {"control": "band-witness", "case": name, "verdict": "ERROR",
                         "error": f"{type(exc).__name__}: {exc}"[:2000],
                         "traceback": traceback.format_exc()[-4000:]}
            record(entry, name)
        entered = [c["case"] for c in controls if c.get("entered_the_band")]
        released = bool(entered)
        reasons = ([] if released else
                   ["no case's own trajectory entered the subnormal band, so the "
                    "flush-equivalence claim this campaign licenses is vacuous"])
        summary = {
            "gate": "dispatch_metal_route", "leg": "band_witness",
            "purpose": ("census the ORACLE's own trajectory for float32 subnormals, "
                        "so the flush-equivalence claim is known to be about a value "
                        "class this corpus constructs"),
            "provenance": provenance,
            "cases_entering_the_band": entered,
            "controls": {f"{c.get('case')}:band-witness": c.get("verdict")
                         for c in controls},
            "release": {"released": released, "reasons": reasons},
            "what_this_licenses": (
                "that the subnormal band is non-empty on the named cases. It "
                "licenses NOTHING about dispatch: this leg installs keep and runs "
                "the array path only."),
            "control_rows": controls,
        }
        gate_provenance.stamp(summary)
        _write(arguments.out, summary)
        say(f"BAND WITNESS entered_the_band={entered}")
        return 0 if released else 1

    # ---------------------------------------------------------------------- cases
    for index, name in enumerate(names, 1):
        say(f"=== case {index}/{len(names)}: {name} ===")
        started = time.time()
        try:
            row = run_case(name, counter, arguments.resolution, arguments.max_steps)
        except BaseException as exc:  # noqa: BLE001 - a raising case is a recorded row
            import traceback  # noqa: PLC0415
            row = {"case": name, "verdict": "ERROR",
                   "error": f"{type(exc).__name__}: {exc}"[:2000],
                   "traceback": traceback.format_exc()[-4000:]}
        row["case_wall_s"] = round(time.time() - started, 2)
        rows.append(row)
        with open(rows_path, "a", encoding="utf-8") as handle:  # rule 7, incremental
            handle.write(json.dumps(row, default=str) + "\n")
            handle.flush()
        say(f"=== case {index}/{len(names)}: {name} -> {row.get('verdict')} "
            f"({row['case_wall_s']} s) ===")

        # THE CONTROLS FOLLOW THE CASE. A case skipped because this leg carries no
        # probe for it has no fused launch to arm anything against, and running the
        # armed controls anyway produced eight NOT-ARMED rows per case -- each of
        # which the release reads as a control that failed.
        if (arguments.no_controls or DRIVE[name]["expect"] != "dispatch"
                or row.get("verdict") == "PASS-NOT-THIS-LEG"):
            continue
        if UNCERTIFIED_POLICY_INSTALLED is not None:
            # THE ARMED CONTROLS CANNOT ARM ON THIS LEG, and the honest record says
            # so rather than reporting NOT-ARMED rows as failures. Each needs a
            # fused launch to sit beside, and rung 8bM refused the plan whole. What
            # this leg measures instead is the CLIFF: a released pair driven under a
            # keeping host, round the ladder, to show the split the rung forbids.
            record({"control": "armed-controls", "verdict": "NOT-APPLICABLE",
                    "why": (f"the harness installed {UNCERTIFIED_POLICY_INSTALLED!r}, "
                            "so rung 8bM refused the plan and no fused launch exists "
                            "to withhold or to put a mutated in-seam pass beside")},
                   name)
            record({"control": "subnormal-cliff", "verdict": "NOT-APPLICABLE",
                    "why": ("with keep INSTALLED the composer's own backend clause "
                            "refuses every Metal arm, so no plan exists to build the "
                            "split beside — measured 2026-09-10, zero slots filled. "
                            "The cliff is armed on the SHIPPED leg instead, where "
                            "the composition exists and the HOST alone is driven to "
                            "keep for the run")},
                   name)
            continue

        for maker in (
            lambda: withheld_control(name, DRIVE[name], arguments.resolution,
                                     counter),
            lambda: release_route_leg(name, DRIVE[name], arguments.resolution),
        ):
            try:
                entry = maker()
            except BaseException as exc:  # noqa: BLE001
                import traceback  # noqa: PLC0415
                entry = {"control": "raised", "verdict": "ERROR",
                         "error": f"{type(exc).__name__}: {exc}"[:2000],
                         "traceback": traceback.format_exc()[-4000:]}
            record(entry, name)
        for side in DRIVE[name]["seam"]:
            try:
                entry = seam_mutation_control(name, DRIVE[name], side,
                                              arguments.resolution, counter)
            except BaseException as exc:  # noqa: BLE001
                import traceback  # noqa: PLC0415
                entry = {"control": f"in-seam-pass-mutated-{side}",
                         "verdict": "ERROR",
                         "error": f"{type(exc).__name__}: {exc}"[:2000],
                         "traceback": traceback.format_exc()[-4000:]}
            record(entry, name)
        try:
            entry = cliff_control(name, DRIVE[name], arguments.resolution)
        except BaseException as exc:  # noqa: BLE001
            import traceback  # noqa: PLC0415
            entry = {"control": "subnormal-cliff", "verdict": "ERROR",
                     "error": f"{type(exc).__name__}: {exc}"[:2000],
                     "traceback": traceback.format_exc()[-4000:]}
        record(entry, name)
        try:
            residency_rows = residency_controls(name, DRIVE[name],
                                                arguments.resolution, counter)
        except BaseException as exc:  # noqa: BLE001
            import traceback  # noqa: PLC0415
            residency_rows = [{"control": "residency", "verdict": "ERROR",
                               "error": f"{type(exc).__name__}: {exc}"[:2000],
                               "traceback": traceback.format_exc()[-4000:]}]
        for entry in residency_rows:
            record(entry, name)

    # ------------------------------------------------------------------- envelope
    # ONCE PER RUN AND NOT PER CASE. It is a property of the RELEASE, not of any
    # case here: it drives its own out-of-envelope shape and asks the two questions
    # no in-envelope case can reach.
    if not arguments.no_controls:
        if UNCERTIFIED_POLICY_INSTALLED is not None:
            entry = {"control": "envelope", "verdict": "NOT-APPLICABLE",
                     "why": (f"the harness installed {UNCERTIFIED_POLICY_INSTALLED!r}, "
                             "so rung 8bM refuses before clause 8M is reached and "
                             "neither direction of the envelope is observable")}
        else:
            try:
                entry = envelope_leg(arguments.resolution)
            except BaseException as exc:  # noqa: BLE001
                import traceback  # noqa: PLC0415
                entry = {"control": "envelope", "verdict": "ERROR",
                         "error": f"{type(exc).__name__}: {exc}"[:2000],
                         "traceback": traceback.format_exc()[-4000:]}
        entry.setdefault("case", ENVELOPE_CASE)
        entry.setdefault("control", "envelope")
        record(entry)

    # -------------------------------------------------------------------- release
    failures = [r["case"] for r in rows if r.get("verdict") not in PASS_VERDICTS]
    control_failures = [f"{c.get('case')}:{c.get('control') or c.get('leg')}"
                        for c in controls if c.get("verdict") not in CONTROL_PASS]
    dispatched = [r["case"] for r in rows if r.get("verdict") == "PASS-FUSED"]
    policy_refused = [r["case"] for r in rows
                      if r.get("verdict") == "PASS-POLICY-REFUSED"]
    demonstrated = policy_refused if UNCERTIFIED_POLICY_INSTALLED else dispatched
    released = bool(demonstrated) and not failures and not control_failures
    reasons: List[str] = []
    if failures:
        reasons.append(f"cases not passing: {failures}")
    if control_failures:
        reasons.append(f"controls not passing: {control_failures}")
    if not demonstrated and UNCERTIFIED_POLICY_INSTALLED:
        reasons.append(f"the harness installed {UNCERTIFIED_POLICY_INSTALLED!r} and "
                       "no case was refused by name at the subnormal-policy rung, "
                       "so nothing about the seam's single-policy rule was measured")
    elif not demonstrated:
        reasons.append("no case dispatched a fused Metal pair through the driver "
                       "seam, so nothing about the Metal route was measured")
    # A LEG THAT MEASURES ZERO MUST NOT RELEASE. `control_failures` catches a control
    # that ran and failed; it cannot catch one that never ran, and the two controls
    # below are what this release rests on — the shipped ROUTE and the ENVELOPE.
    if not arguments.no_controls and UNCERTIFIED_POLICY_INSTALLED is None:
        verdicts = [c.get("verdict") for c in controls]
        if "ENVELOPE-HOLDS" not in verdicts:
            reasons.append("the envelope leg did not run or did not hold, so nothing "
                           "measured that the Metal release can decline a "
                           "configuration or that clause 8M still refuses an "
                           "unadmitted arm by name")
            released = False
        if "RELEASE-ADMITTED" not in verdicts:
            reasons.append("no case was shown reaching its fused arm through "
                           "METAL_RELEASED_FUSED_ARMS with MEEP_GPU_FUSE_ARMS unset, "
                           "so this run is not evidence about the SHIPPED route")
            released = False
    if arguments.no_controls:
        reasons.append("--no-controls: the armed nulls did not run, so this is a "
                       "diagnostic run and not a gate run")
        released = False

    summary = {
        "gate": "dispatch_metal_route",
        "purpose": ("drive a cross-sub-step fused METAL pair through the driver's "
                    "own consults, over the residency seam, and byte-compare the "
                    "whole run against the array path"),
        "provenance": provenance,
        "verdicts": {r["case"]: r.get("verdict") for r in rows},
        "first_divergent_checkpoint": {r["case"]: r.get("first_divergent_checkpoint")
                                       for r in rows},
        "substitution": {r["case"]: (r.get("substitution") or {}).get("verdict")
                         for r in rows},
        "arms_driven": {r["case"]: (r.get("evidence") or {}).get("driven")
                        for r in rows},
        "reached_by": {name: DRIVE[name].get("reached_by", "release")
                       for name in (r["case"] for r in rows) if name in DRIVE},
        "residency": {r["case"]: ((r.get("evidence") or {}).get("residency"))
                      for r in rows},
        "seam_census": {r["case"]: r.get("seam_census") for r in rows},
        "controls": {f"{c.get('case')}:{c.get('control') or c.get('leg')}":
                     c.get("verdict") for c in controls},
        "dispatched_cases": dispatched,
        "policy_refused_cases": policy_refused,
        "uncertified_policy_installed": UNCERTIFIED_POLICY_INSTALLED,
        "failures": failures,
        "control_failures": control_failures,
        "release": {"released": released, "reasons": reasons},
        "what_this_licenses": (
            "the arms listed under arms_driven, on the exact configurations listed "
            "under verdicts, with the policy recorded in provenance, under the "
            "residency mode each row records (record.residency.mode: held, the "
            "default, or shipped, the per-launch bracket). It licenses NOTHING "
            "about an arm it did not drive, a shape it did not run, a residency "
            "mode no row ran, the value of DISPATCH_BY_DEFAULT, or any timing "
            "claim whatsoever — the route is a correctness surface here."),
        "rows": rows,
        "control_rows": controls,
    }
    gate_provenance.stamp(summary)
    _write(arguments.out, summary)
    say(f"SUMMARY {json.dumps(summary['verdicts'])}")
    say(f"CONTROLS {json.dumps(summary['controls'])}")
    say(f"SUBSTITUTION {json.dumps(summary['substitution'])}")
    say(f"RELEASE released={released} reasons={reasons}")
    return 0 if released else 1


def _write(out: str, summary: Dict[str, Any]) -> None:
    """``summary.json`` keeps the rows; ``gate.json`` is what the recut and the weld read."""
    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=1, default=str)
    with open(os.path.join(out, "gate.json"), "w", encoding="utf-8") as handle:
        json.dump({k: v for k, v in summary.items()
                   if k not in ("rows", "control_rows")}, handle, indent=1,
                  default=str)


if __name__ == "__main__":
    raise SystemExit(main())
