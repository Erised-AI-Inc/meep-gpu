"""The Dcyl m = 0 REAL H->D product: ``update_H`` into scratch, then the certified
cylindrical ``step_D`` sub-step -- the radial scan over the NEW Hp and the curl --
over the rotated field.

THE FOURTH SEAM ON A CYLINDRICAL ROW. The driver runs

    ``update_H`` -> the magnetic ``f_minus_p`` restore -> ``step_D``

and on a Dcyl grid ``step_D`` opens by summing the magnetic field ``update_H`` has
just written along r: MEEP's per-chunk ``f_rderiv_int`` of ``Hp = Hy``
(``stepping.cylindrical_rderiv_difference``, step_db.cpp:104-118), fed to the ``Dz``
term only. So the seam's second half depends NON-LOCALLY on the first half's output --
every row of a radial column -- and the one-launch weld the Cartesian product
:mod:`.fused_hd_pair` ships cannot be written here at all.

=============================================================================
THE SHAPE: THREE LAUNCHES OWNING BOTH SLOTS
=============================================================================

Per timestep, on an admitted row:

* **launch 1** (slot ``update_H``, :func:`cyl_real_update_H_scratch`): every program
  computes its OWN cell's ``H`` and ``f_w_H`` through :func:`.fused_hd_pair._h_cell`
  -- ``kernels.constitutive_step``'s certified side-H body as a function of a cell,
  the chunk record read at the cell, MEEP's start of H on the run's first launch --
  into launch-local SCRATCH, never in place. Its statements ARE the Cartesian weld's
  constitutive block (:func:`certified_weld_block`, checked statement for statement),
  so the auxiliary contract on the scratch history is the one that product certifies:
  B on the dsigw cells, the pre-launch word everywhere else;
* **launches 2 and 3** (slot ``step_D``): :class:`.cylindrical_triton.CylindricalCurlPlan`
  -- the certified plan, built by the certified harness builder over the SCRATCH ``H``
  and launched by its own ``run``: the column-serial radial scan
  (``cyl_radial_difference``) over the stepped ``Hy``, then ``cyl_pml_curl_step``
  reading the stepped ``H`` and the scan's difference. Nothing in either launch is new
  or retyped; the plan binds the certified host bindings (the chunk record, the scan's
  starts line, the Courant words) exactly as the singles route binds them;
* the ``H``/``f_w_H`` references ROTATE onto the freshly written scratch
  (:class:`.offdiag_scratch_weld.ScratchWeldPairPlan`'s certified choreography).

The rotation alternates the live ``H`` between exactly two allocations per name, so
the plan builds the certified curl plan TWICE at plan time, once over each, and
launches the one whose sources are this launch's scratch. Both are allocation-free per
step, as the certified plan is.

=============================================================================
THE LAUNCH ALGEBRA, AND WHY THIS PRODUCT SAVES NO LAUNCH
=============================================================================

Under MEEP's per-chunk radial sum the D side's own baseline is three launches --
``update_H``, the scan, ``step_D`` -- because the scan forms its increments from Hp
itself, row by row. The 0.9.2 product fused the elementwise pre-``cumsum`` increment
into launch 1 and saved four launches against a seven-launch baseline (the prefix's
multiply, row-0 fill, subtract, divide and ``cumsum``); the column-serial scan removed
those elementwise passes, and the saving went with them. This product therefore
performs the seam in THREE launches against THREE. It is kept, not dispatched, for the
form alignment its gate measures. A two-launch shape (``update_H`` row by row inside
the column-serial scan) exists and is a design ruling, not this module's to make.

WHY THE SCAN READS THE SCRATCH AND NOTHING ELSE. ``update_H`` is POINTWISE, so the
scratch ``Hy`` is the field ``update_H`` writes, at every cell, before the scan runs:
the scan reads exactly what the array path's ``step_D`` reads. Reading the PRE-LAUNCH
``Hy`` instead would sum the PREVIOUS step's field (the armed host mutation
``m_scan_reads_the_pre_launch_Hy``).

=============================================================================
WHAT SITS IN THE SEAM, AND THE ROWS THIS PRODUCT REFUSES
=============================================================================

:data:`CARRIES_DEPOSIT_REPAIR` is False and it is a FACT about the driver: nothing is
injected between the two consults. :data:`HOISTS_THE_WITHDRAW` is False for the reason
:mod:`.fused_hd_pair` gives, so a row whose electric withdraw does work is refused BY
NAME.

=============================================================================
IT IS NOT INSTALLED
=============================================================================

:data:`INSTALLABLE` is False and :data:`INSTALLABLE_REASON` carries both halves: the
label boundary the Cartesian product names, and the launch algebra above -- three
launches against three, so displacing either released neighbouring cylindrical pair
LOSES a launch.

SERVED ON A FUSION BOARD IS PREDICATE ADMISSION, by that board's own definition. No
timing exists for this shape and none is licensed by anything in this module.

Import contract: importable WITHOUT Triton -- the predicate, the lift checks and the
plan builder (to ``None``) must answer on the laptop that is the merge bar.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import coverage as _coverage
from . import cylindrical_triton as _cyl
from . import fused_hd_pair as _cartesian
from .. import withdraw_hoist as _withdraw_hoist
from .offdiag_scratch_weld import ScratchWeldPairPlan, twin_table

#: The family name, as the board, the battery and the weld record spell it.
FAMILY = "cylindrical_real_fused_hd_pair"

#: The sub-step slot this product STARTS at -- the first half of the seam.
SLOT = "update_H"

#: The driver call sites ONE run of this plan performs, in driver order. Nothing
#: between them is carried: the electric withdraw is declined this round.
REPLACES: Tuple[str, ...] = ("update_H", "step_D")

#: The seam name ``launch.CERTIFIED_FUSED_PAIR_SEAMS`` files this row under.
SEAM: str = _withdraw_hoist.SEAM

#: The constitutive side and the curl sub-step, in driver order.
CONSTITUTIVE_SIDE = "H"
CURL_SUB_STEP = "step_D"

#: The two arms this product implements on ``(update_H, step_D)`` -- the labels
#: ``launch.plan_step`` writes for the two predicates this module conjoins
#: (``cylindrical_constitutive_coverage('H')`` and ``cylindrical_curl_coverage``).
ARMS: Tuple[str, str] = ("cylindrical", "cylindrical PML")

#: The six volumes one run ROTATES, in the launch's argument order.
ROTATED: Tuple[str, ...] = ("Hx", "Hy", "Hz", "f_w_Hx", "f_w_Hy", "f_w_Hz")

#: The volumes the certified curl steps IN PLACE -- it reads and writes them at the
#: program's own cell only.
IN_PLACE: Tuple[str, ...] = ("Dx", "Dy", "Dz", "fu_Dx", "fu_Dy", "fu_Dz")

#: The constitutive half's source volumes, read const for the whole of launch 1.
CONSTITUTIVE_SOURCES: Tuple[str, ...] = ("Bx", "By", "Bz")

#: The radial sum's source component and its ``ir0``, READ off the certified
#: cylindrical family's own table rather than restated
#: (``cylindrical_triton.SUB_STEPS['step_D']``): the scan the certified plan launches.
PREFIX_COMPONENT: str = _cyl.SUB_STEPS[CURL_SUB_STEP]["prefix_component"]
PREFIX_IR0: float = float(_cyl.SUB_STEPS[CURL_SUB_STEP]["prefix_ir0"])

#: ``SUB_STEPS['step_D']['backward']``, restated as a declaration a test pins.
BACKWARD = 1

#: Elements per program -- restated from ``kernels.DEFAULT_BLOCK`` for the import
#: reason every family gives, and pinned against that file's source by a test.
DEFAULT_BLOCK = 256

#: Device launches ONE run performs: launch 1, the certified scan, the certified curl.
LAUNCHES_PER_RUN = 3

#: Of those, kernel launches: all three (the scan is a Triton kernel since the
#: column-serial ``cyl_radial_difference``; no array-module operation remains).
KERNEL_LAUNCHES_PER_RUN = 3

#: Nothing is injected in this seam, so no deposit repair.
CARRIES_DEPOSIT_REPAIR = False
REPAIR_PATHS: Tuple[str, ...] = ()

#: The seam's electric withdraw is NOT performed by this product this round.
HOISTS_THE_WITHDRAW = False

#: May the composer install this product? NO, for the two reasons below.
INSTALLABLE = False

INSTALLABLE_REASON = (
    "THE COMPOSER IS OFFERED THIS PRODUCT AND REFUSES IT BY THIS FLAG, and that is "
    "the first half of the reason: since the 2026-09-07 wiring round it is routed "
    "(launch.CERTIFIED_FUSED_PRODUCTS, CERTIFIED_FUSED_PAIR_ARMS, SUPPORT_MODULES), "
    "its label is declared in the DISPATCHER'S own tables outside this package, by "
    "the one-way rule it keeps (the constituent map, and the pending-gate map that "
    "names this flag beside the missing ledger entry), and joined for the board in "
    "parity/meep_gpu/dispatch_reachability.PRODUCT_ARM_LABELS -- so "
    "launch._declared_uninstallable records this refusal "
    "by name on every row BEFORE the predicate is asked, and no composer writes the "
    "label into a slot. THE SECOND HALF IS THE ARBITRATION, and under MEEP's "
    "per-chunk radial sum it does not favour this product: the D-side seam costs "
    "THREE device launches on the singles route (update_H 1; the column-serial "
    "radial scan 1; step_D 1) and this product performs it in THREE (launch 1, the "
    "certified scan, the certified curl), while each released neighbouring "
    "cylindrical pair saves ONE -- so displacing a neighbour LOSES a launch per step "
    "per row. The composer's rule (launch._neighbouring_seam_claimant, "
    "_pair_may_absorb) counts launches as 4 - (installed pairs); the launch counts "
    "per arrangement are recorded by "
    "parity/meep_gpu/gate_triton_cylindrical_real_fused_hd_pair.py's launch_structure "
    "leg. A two-launch shape (update_H row by row inside the column-serial scan) is "
    "a design ruling outside this module, not a flag edit")

__all__ = [
    "ARMS", "BACKWARD", "CARRIES_DEPOSIT_REPAIR", "CONSTITUTIVE_SIDE",
    "CONSTITUTIVE_SOURCES", "CURL_SUB_STEP", "DEFAULT_BLOCK", "FAMILY",
    "HOISTS_THE_WITHDRAW", "INSTALLABLE", "INSTALLABLE_REASON", "IN_PLACE",
    "KERNEL_LAUNCHES_PER_RUN", "LAUNCHES_PER_RUN", "PREFIX_COMPONENT", "PREFIX_IR0",
    "REPAIR_PATHS", "REPLACES", "ROTATED", "SEAM", "SLOT",
    "CylindricalRealFusedHdPairPlan",
    "certified_weld_block", "cyl_real_update_H_scratch_kernel",
    "cylindrical_real_fused_hd_pair_coverage",
    "explain_cylindrical_real_fused_hd_pair", "lifted_weld_block",
    "plan_cylindrical_real_fused_hd_pair",
    "plan_cylindrical_real_fused_hd_pair_from_arrays",
]


# ---------------------------------------------------------------------------
# The machine-checked lift -- host text, no Triton
# ---------------------------------------------------------------------------

#: Where both kernels finish their index decode; every statement below it is the
#: constitutive block.
WELD_DECODE_END = "        i = plane // ny\n"

#: This module's launch-1 kernel, by name.
SCRATCH_KERNEL = "cyl_real_update_H_scratch"


def _statements(text: str) -> List[str]:
    """The code statements of a body: blank lines and comments carry no arithmetic."""
    return [line.rstrip() for line in text.splitlines()
            if line.strip() and not line.strip().startswith("#")]


def certified_weld_block() -> List[str]:
    """The Cartesian H->D weld's constitutive block, statement by statement.

    :func:`.fused_hd_pair.fused_constitutive_curl_H_to_D` between its index decode and
    its curl anchor: ``update_H`` at the program's own cell through ``_h_cell`` into
    the scratch, the history's auxiliary-contract store included. Cut off that file,
    so a re-cut of the certified weld moves this product's expected text with it.
    """
    source = _cartesian._source_of(  # noqa: SLF001 - the shared cutter, by design
        "fused_constitutive_curl_H_to_D", Path(_cartesian.__file__))
    return _statements(_cartesian._cut(  # noqa: SLF001
        source, WELD_DECODE_END, stop=_cartesian.CURL_BODY_ANCHOR))


def lifted_weld_block() -> List[str]:
    """This module's launch-1 kernel below its decode, statement by statement."""
    source = _cartesian._source_of(SCRATCH_KERNEL, Path(__file__))  # noqa: SLF001
    return _statements(_cartesian._cut(source, WELD_DECODE_END))  # noqa: SLF001


# ---------------------------------------------------------------------------
# The kernel
# ---------------------------------------------------------------------------

try:  # pragma: no cover - the absent branch is exercised by the absence test
    import triton
    import triton.language as tl
    _TRITON_IMPORT_ERROR: Optional[BaseException] = None
except ImportError as _exc:  # pragma: no cover - the laptop path
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _TRITON_IMPORT_ERROR = _exc


if triton is not None:  # pragma: no cover - device code, certified by the gate

    # THE CERTIFIED CONSTITUTIVE, imported as a device function rather than copied:
    # `_h_cell` is `kernels.constitutive_step`'s side-H body as a function of a cell
    # (its lift is machine-checked in fused_hd_pair and re-checked by this family's
    # gate). A second copy would be a second place for the lift to drift.
    from .fused_hd_pair import _h_cell  # noqa: PLC0415

    @triton.jit(do_not_specialize=["start_h"])
    def cyl_real_update_H_scratch(
        ho0, ho1, ho2,                  # SCRATCH out: the stepped Hx, Hy, Hz
        wo0, wo1, wo2,                  # SCRATCH out: the stepped f_w_Hx, f_w_Hy, f_w_Hz
        hi0, hi1, hi2,                  # PRE-LAUNCH Hx, Hy, Hz            (read-only)
        wi0, wi1, wi2,                  # PRE-LAUNCH f_w_Hx, f_w_Hy, f_w_Hz (read-only)
        b0, b1, b2,                     # Bx, By, Bz                       (read-only)
        kp0, kp1, kp2, kmx, kmy, kmz,   # kps/kms on each component's own axis, INTEGER lattice
        cb,                             # chunk record: the constitutive's dsigw lines
        nx, ny, nz, n_elem, start_h,    # start_h: MEEP's start of H
        BLOCK: tl.constexpr,
    ):
        """``update_H`` into scratch: the Cartesian weld's constitutive block, alone.

        ``hi``/``wi``/``b`` are const for the whole dispatch and ``ho``/``wo`` are
        write-only scratch, so nothing written by this launch is read by it. The scan
        and the curl that follow read the scratch after this launch has returned.
        """
        idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        live = idx < n_elem
        nyz = ny * nz
        k = idx % nz
        plane = idx // nz
        j = plane % ny
        i = plane // ny

        # update_H at the program's own cell, to SCRATCH (fused_hd_pair's block)
        own0, own1, own2, src0, src1, src2, dw0, dw1, dw2 = _h_cell(
            i, j, k, live, hi0, hi1, hi2, wi0, wi1, wi2, b0, b1, b2,
            kp0, kp1, kp2, kmx, kmy, kmz, cb, start_h, ny, nz)
        tl.store(ho0 + idx, own0, mask=live)
        tl.store(ho1 + idx, own1, mask=live)
        tl.store(ho2 + idx, own2, mask=live)
        # The history takes B on the accumulation's cells only (the auxiliary
        # contract); every other cell of the twin takes the pre-launch word, so the
        # rotation hands the engine an f_w_H that kept its word there.
        tl.store(wo0 + idx, tl.where(dw0, src0, tl.load(
            wi0 + idx, mask=live & (dw0 == 0), other=0.0)), mask=live)
        tl.store(wo1 + idx, tl.where(dw1, src1, tl.load(
            wi1 + idx, mask=live & (dw1 == 0), other=0.0)), mask=live)
        tl.store(wo2 + idx, tl.where(dw2, src2, tl.load(
            wi2 + idx, mask=live & (dw2 == 0), other=0.0)), mask=live)

else:  # pragma: no cover - the laptop path
    cyl_real_update_H_scratch = None  # type: ignore[assignment]


def cyl_real_update_H_scratch_kernel() -> Any:
    """The shipped launch-1 kernel object, or a refusal naming the missing import."""
    if triton is None:  # pragma: no cover - the laptop path
        raise ImportError(
            "the cylindrical real H->D pair needs Triton to launch; the predicate, the "
            f"lift checks and the plan builder answer without it ({_TRITON_IMPORT_ERROR})")
    return cyl_real_update_H_scratch


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def cylindrical_real_fused_hd_pair_coverage(fields: Any, pml: Any,
                                            sources: Any = None) -> "_coverage.Coverage":
    """May ONE run span ``update_H`` -> the electric withdraw -> ``step_D`` on this
    real m = 0 Dcyl run?

    A conjunction of the two certified halves' OWN predicates
    (:func:`.cylindrical_triton.cylindrical_constitutive_coverage` for side H, then
    :func:`.cylindrical_triton.cylindrical_curl_coverage`) plus the seam clauses.
    Nothing is weakened: a configuration either half refuses is refused here with that
    half's reasons, prefixed so a reader can tell which side said it. THE ORDER IS THE
    DRIVER'S: the constitutive half runs first and speaks first.
    """
    grid = getattr(fields, "grid", None)
    if grid is None:
        return _coverage.Coverage(False, ("fields carries no grid",))

    reasons: List[str] = []
    constitutive = _cyl.cylindrical_constitutive_coverage(fields, pml, CONSTITUTIVE_SIDE)
    if not constitutive.covered:
        reasons.extend(f"cylindrical constitutive half: {reason}"
                       for reason in constitutive.reasons)
    curl = _cyl.cylindrical_curl_coverage(fields, pml)
    if not curl.covered:
        reasons.extend(f"cylindrical curl half: {reason}" for reason in curl.reasons)

    # THE SEAM'S ONE PASS -- the electric integrated-source withdraw. IGNORANCE IS
    # NEVER AN EMPTY SET: `Fields` does not hold the source list.
    reasons.extend(_withdraw_hoist.seam_withdraw_reasons(
        fields, sources,
        undeclared=(
            "the source set was not declared: this predicate cannot infer from Fields "
            "that no electric withdraw stands between update_H and step_D"),
        refusal=lambda index, source: (
            f"source {index} ({type(source).__name__}) has a standing integrated "
            f"electric withdraw, which the driver runs BETWEEN the update_H and "
            f"step_D consults (driver.py:3315-3316); this product declares "
            f"HOISTS_THE_WITHDRAW = False because it declares INSTALLABLE = False, "
            f"so _install_fused_pair's withdraw-hoist branch is unreachable for it "
            f"and nothing would perform the withdraw before the launch"),
        hoists_the_withdraw=HOISTS_THE_WITHDRAW,
        span=REPLACES))

    # THE ROTATION IS THE PRODUCT'S OWN INVARIANT.
    for name in ROTATED:
        if getattr(fields, name, None) is None:
            reasons.append(
                f"{name} is not allocated; this weld rotates it against a plan-owned "
                f"scratch twin after every run")
    for name in IN_PLACE + CONSTITUTIVE_SOURCES:
        if getattr(fields, name, None) is None:
            reasons.append(f"{name} is not allocated")
    return _coverage.Coverage(not reasons, tuple(dict.fromkeys(reasons)))


def explain_cylindrical_real_fused_hd_pair(fields: Any, pml: Any,
                                           sources: Any = None) -> "_coverage.Coverage":
    """The predicate under the name a report reads."""
    return cylindrical_real_fused_hd_pair_coverage(fields, pml, sources)


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

def _address(array: Any) -> Optional[int]:
    """The allocation an array names, on either array module."""
    data = getattr(array, "data", None)
    pointer = getattr(data, "ptr", None)
    if pointer is not None:
        return int(pointer)
    interface = getattr(array, "__array_interface__", None)
    if isinstance(interface, dict):
        return int(interface["data"][0])
    return None


class CylindricalRealFusedHdPairPlan(_cartesian.StartsH, ScratchWeldPairPlan):
    """Launch 1 into scratch, then the certified cylindrical ``step_D`` plan over it.

    THE SIX ROTATING VOLUMES ARE RESOLVED PER RUN, never cached. The certified curl
    plan binds its sources at plan time, so it is built ONCE OVER EACH of the two
    allocations the rotation alternates between, and :meth:`_launch` runs the one
    whose ``Hy`` is this run's scratch. ``__init__`` REFUSES ALIASING between anything
    a run writes (the six twins, ``D``, ``fu_D``, the two scan outputs) and anything
    it reads (``B``, the coefficient vectors), and among the written volumes.
    """

    __slots__ = ("dtdx", "courant", "backward", "_b", "_kps", "_kms", "_kernel",
                 "_pointer", "_flat", "_curl_plans", "kernel_launches", "_h_state")

    replaces = REPLACES

    #: How many device launches ONE run performs, and how many of them are kernels --
    #: declared, so a launch witness can hold them to it.
    launches_per_run = LAUNCHES_PER_RUN
    kernel_launches_per_run = KERNEL_LAUNCHES_PER_RUN

    def __init__(self, shape: Sequence[int], dtdx: float, courant: float, block: int,
                 fields: Any, twins: Dict[str, Any],
                 flux: Sequence[Any], targets: Sequence[Any],
                 auxiliaries: Sequence[Any], curl_coefficients: Sequence[Any],
                 constitutive_coefficients: Sequence[Any], xp: Any, *,
                 chunk_bounds: Any, scan_starts: Any, kernel: Any = None,
                 curl_kernel: Any = None, scan_kernel: Any = None,
                 num_warps: Optional[int] = 1, h_state: Any = None) -> None:
        from .launch import CupyPointer, _flat  # noqa: PLC0415

        super().__init__(FAMILY, fields, twins, ROTATED, shape, block, REPLACES,
                         num_warps=num_warps, chunk_bounds=chunk_bounds)
        if int(self.shape[1]) != 1:
            raise ValueError(
                f"a Dcyl grid stores one phi cell; shape {self.shape} does not")
        self.dtdx = float(dtdx)
        self.courant = float(courant)
        self.backward = BACKWARD
        self._pointer = CupyPointer
        self._flat = _flat
        self._b = tuple(CupyPointer(a) for a in flux)
        kps = list(constitutive_coefficients[:3])
        kms = list(constitutive_coefficients[3:])
        if len(curl_coefficients) != 6 or len(kps) != 3 or len(kms) != 3:
            raise ValueError(
                "this plan binds six curl coefficient vectors (kms/sinv per axis) and "
                "six constitutive ones (kps, then kms, per axis), all on the integer "
                "sub-lattice")
        self._kps = tuple(CupyPointer(_flat(a)) for a in kps)
        self._kms = tuple(CupyPointer(_flat(a)) for a in kms)
        # THE CERTIFIED step_D PLAN, once over each allocation the rotation alternates
        # between, built by the certified harness builder: the same scan, curl, chunk
        # record, starts line and Courant words the singles route launches.
        live = {name: getattr(fields, name) for name in ROTATED}
        curl_flat = dict(zip((f"{stem}_{axis}" for axis in "xyz"
                              for stem in ("kms", "sinv")), curl_coefficients))
        plans: Dict[int, Any] = {}
        for magnetic in (live, twins):
            arrays = {name: magnetic[name] for name in ("Hx", "Hy", "Hz")}
            arrays.update(zip(IN_PLACE[:3], targets))
            arrays.update(("fu_" + name, aux) for name, aux in zip(IN_PLACE[:3],
                                                                  auxiliaries))
            plan = _cyl.plan_cylindrical_curl_from_arrays(
                CURL_SUB_STEP, arrays, curl_flat, dtdx, courant, xp,
                chunk_bounds=chunk_bounds, scan_starts=scan_starts, block=block,
                kernel=curl_kernel, scan_kernel=scan_kernel)
            plans[_address(magnetic[PREFIX_COMPONENT])] = plan
        if len(plans) != 2:
            raise ValueError(
                "the live Hy and its scratch twin are one allocation; the certified "
                "step_D plan cannot be built over two distinct sets")
        self._curl_plans = plans
        self._check_aliases(twins, flux, targets, auxiliaries, curl_coefficients,
                            constitutive_coefficients)
        self._kernel = kernel
        self.kernel_launches = 0
        # MEEP's start of H, on the Cartesian weld's terms (fused_hd_pair.StartsH).
        self._h_state = h_state

    @property
    def step_d_plans(self) -> Tuple[Any, ...]:
        """The two certified ``step_D`` plans, one over each rotating allocation."""
        return tuple(self._curl_plans.values())

    def _check_aliases(self, twins, flux, targets, auxiliaries,
                       curl_coefficients, constitutive_coefficients) -> None:
        """No written volume may share an allocation with a read one, or another."""
        outputs: Dict[int, str] = {}
        named = [(name, twins[name]) for name in ROTATED]
        named += [(IN_PLACE[index], array) for index, array in enumerate(targets)]
        named += [(IN_PLACE[3 + index], array)
                  for index, array in enumerate(auxiliaries)]
        named += [(f"scan_out_{index}", plan.scan.out)
                  for index, plan in enumerate(self._curl_plans.values())]
        for label, array in named:
            key = _address(array)
            if key is None:
                continue
            if key in outputs:
                raise ValueError(
                    f"outputs {outputs[key]} and {label} are the same allocation; "
                    f"one run would write both")
            outputs[key] = label
        inputs: List[Tuple[str, Any]] = [
            (CONSTITUTIVE_SOURCES[index], array) for index, array in enumerate(flux)]
        inputs += [(f"curl_coefficient{index}", array)
                   for index, array in enumerate(curl_coefficients)]
        inputs += [(f"constitutive_coefficient{index}", array)
                   for index, array in enumerate(constitutive_coefficients)]
        for label, array in inputs:
            key = _address(array)
            if key is not None and key in outputs:
                raise ValueError(
                    f"input {label} aliases output {outputs[key]}: the run would read a "
                    f"volume it is writing")

    def _curl_plan_for(self, writes: Sequence[Any]) -> Any:
        """The certified plan whose sources are this run's scratch ``H``."""
        plan = self._curl_plans.get(_address(writes[ROTATED.index(PREFIX_COMPONENT)]))
        if plan is None:
            raise RuntimeError(
                "this run's scratch Hy is neither allocation the certified step_D plans "
                "were built over; the rotation lost a buffer")
        return plan

    # -- the three launches ------------------------------------------------
    def _launch(self, writes: Sequence[Any], reads: Sequence[Any],
                guard: Optional[bool]) -> None:
        from .kernels import ENABLE_FP_FUSION  # noqa: PLC0415

        nx, ny, nz = self.shape
        extra = {} if self.num_warps is None else {"num_warps": self.num_warps}
        pointer = self._pointer
        scratch = [pointer(array) for array in writes]
        prior = [pointer(array) for array in reads]

        # LAUNCH 1 -- update_H to scratch.
        kernel = (self._kernel if self._kernel is not None
                  else cyl_real_update_H_scratch_kernel())
        kernel[self._grid](
            *scratch, *prior, *self._b, *self._kps, *self._kms, self._cb,
            nx, ny, nz, self.n_elem, self.start_h(),
            BLOCK=self.block,
            enable_fp_fusion=ENABLE_FP_FUSION if guard is None else bool(guard),
            **extra,
        )
        # LAUNCHES 2 AND 3 -- the certified step_D plan over the scratch H: its scan,
        # then its curl, by its own run. A warm (an empty grid here) compiles both at
        # zero programs, exactly as the certified plan would be warmed.
        plan = self._curl_plan_for(writes)
        if self._grid[0]:
            plan.run(guard)
            self.kernel_launches += KERNEL_LAUNCHES_PER_RUN
            return
        saved = (plan._grid, plan.scan._grid)  # noqa: SLF001
        plan._grid = plan.scan._grid = (0,)  # noqa: SLF001
        try:
            plan.run(guard)
        finally:
            plan._grid, plan.scan._grid = saved  # noqa: SLF001

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (f"CylindricalRealFusedHdPairPlan(shape={self.shape}, block={self.block}, "
                f"num_warps={self.num_warps})")


def _sub_lattice_suffixes() -> Tuple[str, str]:
    """``(curl suffix, constitutive suffix)``, READ from the shipped tables and asserted
    equal: both halves take the INTEGER sub-lattice on this seam."""
    from .launch import SUB_STEPS  # noqa: PLC0415

    curl = _cyl.SUB_STEPS[CURL_SUB_STEP]["suffix"]
    constitutive = ("_h" if _coverage.CONSTITUTIVE_SIDES[CONSTITUTIVE_SIDE]
                    ["half_integer"] else "")
    if curl != constitutive:
        raise AssertionError(
            f"step_D reads the {curl or 'integer'!r} PML sub-lattice and update_H the "
            f"{constitutive or 'integer'!r} one; both halves of this product take one "
            f"lattice and that is only correct while they agree")
    if int(_cyl.SUB_STEPS[CURL_SUB_STEP]["backward"]) != BACKWARD or \
            int(SUB_STEPS[CURL_SUB_STEP]["backward"]) != BACKWARD:
        raise AssertionError(
            f"the tables say step_D differences {_cyl.SUB_STEPS[CURL_SUB_STEP]['backward']} "
            f"/ {SUB_STEPS[CURL_SUB_STEP]['backward']}, and this product declares "
            f"BACKWARD = {BACKWARD}")
    return curl, constitutive


def plan_cylindrical_real_fused_hd_pair(
        fields: Any, pml: Any, sources: Any = None, block: Optional[int] = None,
        num_warps: Optional[int] = 1, kernel: Any = None, curl_kernel: Any = None,
        scan_kernel: Any = None) -> Optional[CylindricalRealFusedHdPairPlan]:
    """Build the plan from the engine's own objects, or ``None`` when refused.

    ``None`` is the only refusal. ``kernel``, ``curl_kernel`` and ``scan_kernel`` are
    the mutation seams; dropping them is not a slowdown, it is a silent DISARMING of
    every mutation leg that hands a broken copy through them. The chunk record and the
    scan's starts line are the certified host bindings, called
    (:func:`.launch.chunk_bounds_for`, :func:`.cylindrical_triton.radial_scan_starts`).
    """
    from .launch import chunk_bounds_for  # noqa: PLC0415
    if not cylindrical_real_fused_hd_pair_coverage(fields, pml, sources).covered:
        return None
    suffix, constitutive_suffix = _sub_lattice_suffixes()
    grid = fields.grid
    return CylindricalRealFusedHdPairPlan(
        grid.shape, grid.dt / grid.dx, float(grid.courant),
        DEFAULT_BLOCK if block is None else block,
        fields,
        twin_table(fields, ROTATED),
        [getattr(fields, name) for name in CONSTITUTIVE_SOURCES],
        [getattr(fields, name) for name in IN_PLACE[:3]],
        [getattr(fields, name) for name in IN_PLACE[3:]],
        [getattr(pml, f"{stem}_{axis}{suffix}")
         for axis in "xyz" for stem in ("kms", "sinv")],
        ([getattr(pml, f"kps_{axis}{constitutive_suffix}") for axis in "xyz"]
         + [getattr(pml, f"kms_{axis}{constitutive_suffix}") for axis in "xyz"]),
        grid.xp,
        chunk_bounds=chunk_bounds_for(fields, pml, None),
        scan_starts=_cyl.radial_scan_starts(fields, pml, CURL_SUB_STEP, None),
        kernel=kernel, curl_kernel=curl_kernel, scan_kernel=scan_kernel,
        num_warps=num_warps, h_state=fields,
    )


def plan_cylindrical_real_fused_hd_pair_from_arrays(
        arrays: Dict[str, Any], flat: Dict[str, Any], dtdx: float, courant: float,
        fields: Any, xp: Any, block: Optional[int] = None,
        kernel: Any = None, curl_kernel: Any = None, scan_kernel: Any = None,
        num_warps: Optional[int] = 1, *,
        chunk_bounds: Any, scan_starts: Any) -> CylindricalRealFusedHdPairPlan:
    """Build from bare device arrays -- the gate's route. No predicate runs.

    ``arrays`` supplies the twins under ``scratch_Hx`` ...; ``flat`` supplies
    ``kms_x``/``sinv_x``/``kps_x`` ... on the sub-lattice the caller ALREADY SELECTED,
    so a swapped pair can be handed over and measured. ``chunk_bounds`` and
    ``scan_starts`` are REQUIRED, as the certified curl plan requires them. No start of
    H: a harness state has already started.
    """
    shape = tuple(int(n) for n in arrays[IN_PLACE[0]].shape)
    twins = {name: arrays["scratch_" + name] for name in ROTATED}
    return CylindricalRealFusedHdPairPlan(
        shape, dtdx, courant, DEFAULT_BLOCK if block is None else block,
        fields, twins,
        [arrays[name] for name in CONSTITUTIVE_SOURCES],
        [arrays[name] for name in IN_PLACE[:3]],
        [arrays[name] for name in IN_PLACE[3:]],
        [flat[f"{stem}_{axis}"] for axis in "xyz" for stem in ("kms", "sinv")],
        ([flat[f"kps_{axis}"] for axis in "xyz"] + [flat[f"kms_{axis}"] for axis in "xyz"]),
        xp, chunk_bounds=chunk_bounds, scan_starts=scan_starts, kernel=kernel,
        curl_kernel=curl_kernel, scan_kernel=scan_kernel, num_warps=num_warps,
    )
