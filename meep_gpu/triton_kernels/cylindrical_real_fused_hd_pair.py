"""The Dcyl m = 0 REAL H->D product: ``update_H`` + the pre-scan stage of the radial
prefix in ONE launch, the array path's ``cumsum`` untouched, the certified cylindrical
``step_D`` curl in a second launch.

THE FOURTH SEAM ON A CYLINDRICAL ROW, and the first product on this backend that
spans it there. The driver runs

    ``update_H`` (driver.py:3313) -> the electric integrated-source withdraw
    (:3315-3316) -> ``step_D`` (:3317)

with exactly one statement between the two consults, the withdraw loop, and on a Dcyl
grid ``step_D`` opens by prefixing the magnetic field ``update_H`` has just written:
``prefixed["Hy"] = cylindrical_rderiv_prefix(grid.xp, magnetic["Hy"], 0.5, scratch=
scratch)`` (stepping.py:447), fed to the ``Dz`` term only (:461-462; ``Dx`` reads the
raw ``Hy``). So the seam's second half depends NON-LOCALLY on the first half's
output -- every cell of a radial column -- and the one-launch weld the Cartesian
product :mod:`.fused_hd_pair` ships cannot be written here at all.

=============================================================================
THE SHAPE: TWO LAUNCHES OWNING BOTH SLOTS, THE ORACLE LEFT WHERE IT IS
=============================================================================

``cylindrical_rderiv_prefix`` (stepping.py:1313-1333, the pooled branch) is four
elementwise passes and one scan::

    xp.multiply(f_p, weights, out=weighted)               # field LEFT of the weight
    increment[_face(0, 0)] = 0                            # row 0 is the sum's start
    xp.subtract(weighted[1:], weighted[:-1], out=increment[1:])
    increment[1:] /= divisor                              # IEEE '/', row-constant
    xp.cumsum(increment, axis=0, out=scratch.take("cyl_prefix", ...))

The scan's summation order is the array module's own and is NOT a sequential
accumulation on CuPy (cylindrical_triton.py, "THE SCAN STAYS ON THE ARRAY PATH";
re-measured 2026-09-06, a column-serial device scan differs from ``cupy.cumsum`` on
every case and equals ``numpy.cumsum`` on every case). So this product does not scan.
Per timestep, on an admitted row:

* **launch 1** (slot ``update_H``): every program computes its OWN cell's ``H`` and
  ``f_w_H`` through :func:`.fused_hd_pair._h_cell` -- ``kernels.constitutive_step``'s
  certified side-H body as a function of a cell, the SAME lifted text the Cartesian
  H->D weld certifies -- into launch-local SCRATCH, never in place; then it forms the
  radial INCREMENT for its cell: ``Hy`` at the one-cell backward radial neighbour is
  RECOMPUTED through the same body from the same unwritten state (no read of another
  program's store), and ``inc[i] = (Hy_new[i]*w[i] - Hy_new[i-1]*w[i-1]) / d[i-1]``
  with ``inc[0] = +0.0``, ``w`` and ``d`` the very row vectors the array path caches
  (``scratch.constant(("cyl_rderiv", nr, 0.5, float32), ...)``, stepping.py:1320);
* **the scan**: ``xp.cumsum(increment, axis=0, out=scratch.take("cyl_prefix", ...))``
  -- stepping.py:1332-1333 verbatim, on the pooled buffers the array path itself uses,
  so the prefix IS the oracle's own statement over an increment measured equal to the
  oracle's own increment;
* the ``H``/``f_w_H`` references ROTATE onto the freshly written scratch
  (:class:`.offdiag_scratch_weld.ScratchWeldPairPlan`'s certified choreography);
* **launch 2** (slot ``step_D``): :func:`.cylindrical_triton.cyl_pml_curl_step`,
  VERBATIM -- the certified m = 0 curl with ``BACKWARD = 1`` -- reading the rotated
  ``H`` (raw ``Hy`` for ``Dx``; the new ``Hy`` again for the on-axis ``Dz`` add,
  which is what ``fields.get_H("Hy")`` returns after ``update_H``) and the cumsum
  output as ``pfx``. Nothing in it is new.

Device launches on the D-side seam per step per row: today ``update_H`` 1 + the
prefix 5 (multiply, row-0 fill, subtract, divide, cumsum) + ``step_D`` 1 = **7**;
here **3** (launch 1, the CuPy scan, launch 2). :data:`LAUNCHES_PER_RUN` counts the
scan; :data:`KERNEL_LAUNCHES_PER_RUN` is the two this module compiles.

=============================================================================
THE ARITHMETIC OF THE INCREMENT IS MEASURED, NOT CHOSEN
=============================================================================

The increment stage was measured on the device before this module was written
(lane record ``cupy_probe/FINDINGS.md``, NVRTC, both float32 subnormal policies,
13 corpus radial extents x 2 ir0 x 4 value classes): the spelling below is 0 of
7,193,552 differing float32 words against the array path's four passes, and its
``cumsum`` equals the SHIPPED prefix on 0 words. The controls that bite, and are
therefore this product's armed mutations: a reciprocal multiply
``diff * (1/d)`` (1,816,293 / 1,802,148 words keep / flush), the divide distributed
over the subtraction (2,114,294 / 2,065,183), and the same source built with
floating-point contraction on (1,168,184 / 1,149,775 -- the subtract contracts one
product). The swapped multiply operand order does NOT bite (0 / 0): IEEE-754
multiplication commutes bitwise, and "field on the left" is documentation, not a
rounding fact.

THE DIVIDE IS ``tl.math.div_rn``, NOT ``/``. Those NVRTC numbers were taken with the
C ``/`` under ``--fmad=false``, which is an IEEE round-to-nearest division; Triton's
``/`` on fp32 lowers to ``div.full.f32`` (approximately 2 ulp, not correctly rounded
-- the reason :mod:`.cylindrical_complex` binds its i*m/r rows host-built rather than
dividing in-kernel). The array path's ``increment[1:] /= divisor`` is CuPy's
``true_divide``, correctly rounded. So the correctly-rounded intrinsic is what
reproduces it, and ``/`` is an armed mutation here rather than the spelling. The
Triton-compiled increment's own bytes are measured by the gate's ``increment_stage``
leg; the NVRTC record above does not transfer to them and is not cited as if it did.

WHY THE HALO RECOMPUTE IS EXACT. ``update_H`` is POINTWISE (stepping.py:907-923
through ``_apply_constitutive_pml``), so ``Hy_new[i-1]`` is a function of state this
launch does not write, and evaluating it at the neighbour's coordinates through the
same ``_h_cell`` gives the same bits the neighbour's own program stores. Reading the
STORED ``Hy[i-1]`` instead would read the PREVIOUS step's field (the armed mutation
``m_halo_reads_the_stored_Hy``); reading the neighbour program's scratch store would
put the block schedule back into the answer.

=============================================================================
WHAT SITS IN THE SEAM, AND THE TWO ROWS THIS PRODUCT REFUSES
=============================================================================

:data:`CARRIES_DEPOSIT_REPAIR` is False and it is a FACT about the driver: nothing is
injected between the two consults. :data:`HOISTS_THE_WITHDRAW` is False IN THIS ROUND
for the reason :mod:`.fused_hd_pair` gives -- the only wiring that performs the hoist
is ``launch._install_fused_pair``'s ``withdraw_hoist.SEAM`` branch, unreachable for a
product that declares :data:`INSTALLABLE` False -- so a row whose electric withdraw
does work is refused BY NAME. On the three boards' cylindrical cells that is
``examples:cylinder_cross_section.py`` and ``examples:zone_plate.py``, both in the
COMPLEX cell (this real family's three rows carry none): filed ``withdraw_seam`` and
untouched by this round.

=============================================================================
IT IS NOT INSTALLED, AND THE CYLINDRICAL ALGEBRA IS THE OPPOSITE OF THE CARTESIAN
=============================================================================

:data:`INSTALLABLE` is False and :data:`INSTALLABLE_REASON` carries both halves. The
first is the label boundary the Cartesian product names. The second is the
arbitration, and on THESE rows it favours this product: the D-side baseline is 7
device launches, this product saves 4, and each released neighbouring cylindrical
pair (``fused pair B (cylindrical)``, ``fused pair D (cylindrical)``) saves 1 -- so
displacing both is +4 - 2 = **+2 launches per step per row**. The composer's rule
counts launches as ``4 - (installed pairs)`` and would refuse or lose the tie; the
flip is a MEASURED step (a composition gate counting device launches with this
product installed against the neighbours installed -- the gate's
``launch_structure`` leg records the counts per arrangement -- and a per-family
weight in the arbitration, a rule change in a file this round does not own), not a
flag edit.

SERVED ON A FUSION BOARD IS PREDICATE ADMISSION, by that board's own definition. No
timing exists for this shape and none is licensed by anything in this module.

Import contract: importable WITHOUT Triton -- the predicate, the lift checks and the
plan builder (to ``None``) must answer on the laptop that is the merge bar.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

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

#: The volumes launch 2 steps IN PLACE -- the certified curl reads and writes them at
#: the program's own cell only.
IN_PLACE: Tuple[str, ...] = ("Dx", "Dy", "Dz", "fu_Dx", "fu_Dy", "fu_Dz")

#: The constitutive half's source volumes, read const for the whole of launch 1.
CONSTITUTIVE_SOURCES: Tuple[str, ...] = ("Bx", "By", "Bz")

#: The prefix's source component and its ``ir0``, READ off the certified cylindrical
#: family's own table rather than restated (``cylindrical_triton.SUB_STEPS['step_D']``).
PREFIX_COMPONENT: str = _cyl.SUB_STEPS[CURL_SUB_STEP]["prefix_component"]
PREFIX_IR0: float = float(_cyl.SUB_STEPS[CURL_SUB_STEP]["prefix_ir0"])

#: The array path's own pooled-buffer tags for the increment and the scan output
#: (stepping.py:1326, :1332-1333). This plan takes the SAME slots, so the scan writes
#: where the oracle writes; a laptop test pins both against ``stepping.py``'s text.
INCREMENT_TAG = "cyl_increment"
PREFIX_TAG = "cyl_prefix"

#: The ``scratch.constant`` key of the two invariant row vectors (stepping.py:1320),
#: restated as a callable so the plan binds the array path's OWN cached arrays.
def rderiv_constant_key(rows: int, real_dtype: Any) -> Tuple[Any, ...]:
    return ("cyl_rderiv", int(rows), float(PREFIX_IR0), real_dtype)


#: ``SUB_STEPS['step_D']['backward']``, restated as a declaration a test pins.
BACKWARD = 1

#: Elements per program -- restated from ``kernels.DEFAULT_BLOCK`` for the import
#: reason every family gives, and pinned against that file's source by a test.
DEFAULT_BLOCK = 256

#: Device launches ONE run performs: launch 1, the array module's scan, launch 2.
LAUNCHES_PER_RUN = 3

#: Of those, the kernels this module compiles: launch 1 and the certified curl.
KERNEL_LAUNCHES_PER_RUN = 2

#: Nothing is injected in this seam (driver.py:3313-3317), so no deposit repair.
CARRIES_DEPOSIT_REPAIR = False
REPAIR_PATHS: Tuple[str, ...] = ()

#: The seam's electric withdraw is NOT performed by this product this round.
HOISTS_THE_WITHDRAW = False

#: May the composer install this product? NO, for the two measured reasons below.
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
    "label into a slot. THE SECOND HALF IS THE ARBITRATION, and on a cylindrical row its "
    "algebra is the OPPOSITE of the Cartesian cell's: the D-side seam costs SEVEN "
    "device launches today (update_H 1; the prefix's multiply, row-0 fill, subtract, "
    "divide and cumsum 5; step_D 1) and this product performs it in THREE, while each "
    "released neighbouring cylindrical pair saves ONE -- so displacing both "
    "neighbours is +4 - 2 = +2 launches per step per row in this product's favour. "
    "The composer's rule (launch._neighbouring_seam_claimant, _pair_may_absorb) "
    "counts launches as 4 - (installed pairs) and would refuse or lose the tie on "
    "every row; the flip is a MEASURED step -- a composition gate counting device "
    "launches per step with this product installed against the two neighbours "
    "installed (parity/meep_gpu/gate_triton_cylindrical_real_fused_hd_pair.py, "
    "launch_structure, records the counts per arrangement) and a per-family "
    "launches_saved weight in the arbitration, which is a rule change in a file this "
    "round does not own -- and not a flag edit")

__all__ = [
    "ARMS", "BACKWARD", "CARRIES_DEPOSIT_REPAIR", "CONSTITUTIVE_SIDE",
    "CONSTITUTIVE_SOURCES", "CURL_SUB_STEP", "DEFAULT_BLOCK", "FAMILY",
    "HOISTS_THE_WITHDRAW", "INCREMENT_SPELLING", "INCREMENT_TAG", "INSTALLABLE",
    "INSTALLABLE_REASON", "IN_PLACE", "KERNEL_LAUNCHES_PER_RUN", "LAUNCHES_PER_RUN",
    "PREFIX_COMPONENT", "PREFIX_IR0", "PREFIX_TAG", "REPAIR_PATHS", "REPLACES",
    "ROTATED", "SEAM", "SLOT",
    "CylindricalRealFusedHdPairPlan",
    "cyl_real_update_H_increment_kernel", "cylindrical_real_fused_hd_pair_coverage",
    "explain_cylindrical_real_fused_hd_pair", "increment_statements",
    "plan_cylindrical_real_fused_hd_pair",
    "plan_cylindrical_real_fused_hd_pair_from_arrays", "rderiv_constant_key",
    "rderiv_vectors",
]


# ---------------------------------------------------------------------------
# The increment spelling, as DATA a test and the gate pin against the kernel text
# ---------------------------------------------------------------------------

#: Every statement of the increment stage, in order, exactly as the kernel spells it.
#: The laptop test asserts each appears once in the kernel body; the gate's
#: ``increment_stage`` leg measures each against the array path with its controls.
INCREMENT_SPELLING: Tuple[Tuple[str, str], ...] = (
    ("halo", "halo = _h_tap(1, i - 1, j, k, inner, hi0, hi1, hi2, wi0, wi1, wi2, "
             "b0, b1, b2,"),
    ("weight_here", "weighted_here = own1 * w_i"),
    ("weight_below", "weighted_below = halo * w_im1"),
    ("subtract", "diff = weighted_here - weighted_below"),
    ("divide", "value = tl.math.div_rn(diff, d_im1)"),
    ("row0", "value = tl.where(inner, value, 0.0)"),
)


def increment_statements() -> List[str]:
    """The increment statements as they stand in the kernel SOURCE, for the tests."""
    import ast  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    text = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "cyl_real_update_H_increment":
            segment = ast.get_source_segment(text, node) or ""
            return [line.strip() for line in segment.splitlines()
                    if line.strip() and not line.strip().startswith("#")]
    raise AssertionError("the module no longer defines cyl_real_update_H_increment")


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

    # THE CERTIFIED CONSTITUTIVE, imported as device functions rather than copied:
    # `_h_cell` is `kernels.constitutive_step`'s side-H body as a function of a cell
    # (its lift is machine-checked in fused_hd_pair and re-checked by this family's
    # gate), and `_h_tap` is one component of it under the caller's validity flag. A
    # second copy would be a second place for the lift to drift.
    from .fused_hd_pair import _h_cell, _h_tap  # noqa: PLC0415

    @triton.jit
    def cyl_real_update_H_increment(
        ho0, ho1, ho2,                  # SCRATCH out: the stepped Hx, Hy, Hz
        wo0, wo1, wo2,                  # SCRATCH out: the stepped f_w_Hx, f_w_Hy, f_w_Hz
        hi0, hi1, hi2,                  # PRE-LAUNCH Hx, Hy, Hz            (read-only)
        wi0, wi1, wi2,                  # PRE-LAUNCH f_w_Hx, f_w_Hy, f_w_Hz (read-only)
        b0, b1, b2,                     # Bx, By, Bz                       (read-only)
        inc,                            # SCRATCH out: the radial increment (nr, ny, nz)
        wgt, dvs,                       # the array path's cached weights (nr), divisor (nr-1)
        kp0, kp1, kp2, kmx, kmy, kmz,   # kps/kms on each component's own axis, INTEGER lattice
        nx, ny, nz, n_elem,
        BLOCK: tl.constexpr,
    ):
        """``update_H`` into scratch, plus the pre-scan stage of the radial prefix.

        ``hi``/``wi``/``b`` are const for the whole dispatch and ``ho``/``wo``/``inc``
        are write-only scratch, so nothing written by this launch is read by it. The
        one foreign value -- ``Hy`` one row down -- is a RECOMPUTE from the unwritten
        state, never a load of another program's store.
        """
        idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        live = idx < n_elem
        nyz = ny * nz
        k = idx % nz
        plane = idx // nz
        j = plane % ny
        i = plane // ny

        # ============ THE CONSTITUTIVE: update_H at the program's own cell, to SCRATCH
        own0, own1, own2, src0, src1, src2 = _h_cell(
            i, j, k, live, hi0, hi1, hi2, wi0, wi1, wi2, b0, b1, b2,
            kp0, kp1, kp2, kmx, kmy, kmz, ny, nz)
        tl.store(ho0 + idx, own0, mask=live)
        tl.store(ho1 + idx, own1, mask=live)
        tl.store(ho2 + idx, own2, mask=live)
        tl.store(wo0 + idx, src0, mask=live)
        tl.store(wo1 + idx, src1, mask=live)
        tl.store(wo2 + idx, src2, mask=live)

        # ============ THE INCREMENT: stepping.cylindrical_rderiv_prefix's four passes
        # `inner` is every row but the axis row: the array path assigns the integer 0
        # to row 0 (a +0.0 word) and forms the difference on rows 1..nr-1. The halo
        # is Hy one radial row DOWN, recomputed through the certified body under the
        # `inner` guard, so no out-of-range coordinate is ever dereferenced.
        inner = live & (i >= 1)
        halo = _h_tap(1, i - 1, j, k, inner, hi0, hi1, hi2, wi0, wi1, wi2, b0, b1, b2,
                      kp0, kp1, kp2, kmx, kmy, kmz, ny, nz)
        w_i = tl.load(wgt + i, mask=live, other=0.0)
        w_im1 = tl.load(wgt + i - 1, mask=inner, other=0.0)
        d_im1 = tl.load(dvs + i - 1, mask=inner, other=1.0)
        # xp.multiply(f_p, weights): the FIELD on the left, on both rows.
        weighted_here = own1 * w_i
        weighted_below = halo * w_im1
        # xp.subtract(weighted[1:], weighted[:-1]).
        diff = weighted_here - weighted_below
        # increment[1:] /= divisor: CORRECTLY ROUNDED. `tl.math.div_rn`, never `/`
        # (div.full.f32) and never a reciprocal multiply -- see the module docstring.
        value = tl.math.div_rn(diff, d_im1)
        # increment[_face(0, 0)] = 0: the axis row is an exact +0.0.
        value = tl.where(inner, value, 0.0)
        tl.store(inc + idx, value, mask=live)

else:  # pragma: no cover - the laptop path
    cyl_real_update_H_increment = None  # type: ignore[assignment]


def cyl_real_update_H_increment_kernel() -> Any:
    """The shipped launch-1 kernel object, or a refusal naming the missing import."""
    if triton is None:  # pragma: no cover - the laptop path
        raise ImportError(
            "the cylindrical real H->D pair needs Triton to launch; the predicate, the "
            f"lift checks and the plan builder answer without it ({_TRITON_IMPORT_ERROR})")
    return cyl_real_update_H_increment


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

    # THE ROTATION AND THE POOL ARE THE PRODUCT'S OWN INVARIANTS.
    for name in ROTATED:
        if getattr(fields, name, None) is None:
            reasons.append(
                f"{name} is not allocated; this weld rotates it against a plan-owned "
                f"scratch twin after every run")
    for name in IN_PLACE + CONSTITUTIVE_SOURCES:
        if getattr(fields, name, None) is None:
            reasons.append(f"{name} is not allocated")
    if getattr(fields, "scratch", None) is None:
        reasons.append(
            "fields carries no StepScratch: the increment and the scan output are the "
            "array path's own pooled slots (cyl_increment, cyl_prefix) and the two "
            "invariant row vectors are its cached constants; this product takes them "
            "from the same pool rather than allocating twins the oracle never sees")
    return _coverage.Coverage(not reasons, tuple(dict.fromkeys(reasons)))


def explain_cylindrical_real_fused_hd_pair(fields: Any, pml: Any,
                                           sources: Any = None) -> "_coverage.Coverage":
    """The predicate under the name a report reads."""
    return cylindrical_real_fused_hd_pair_coverage(fields, pml, sources)


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

def rderiv_vectors(xp: Any, scratch: Any, rows: int, real_dtype: Any) -> Tuple[Any, Any]:
    """The array path's OWN cached ``(weights, divisor)`` for this radial extent.

    Bound through the same ``scratch.constant`` key ``stepping.cylindrical_rderiv_prefix``
    uses (stepping.py:1320-1322), so the kernel multiplies and divides by the very
    device arrays the oracle's four passes use. Rebuilding them here -- even by the
    same arithmetic -- would be a second copy of an invariant the pool already holds.
    """
    from ..stepping import _cylindrical_rderiv_weights  # noqa: PLC0415

    return scratch.constant(
        rderiv_constant_key(rows, real_dtype),
        lambda: _cylindrical_rderiv_weights(xp, int(rows), PREFIX_IR0, real_dtype))


class CylindricalRealFusedHdPairPlan(ScratchWeldPairPlan):
    """TWO launches and ONE array-module scan for ``update_H`` and ``step_D``.

    THE SIX ROTATING VOLUMES ARE RESOLVED PER RUN, never cached. ``__init__`` REFUSES
    ALIASING between anything this run writes (the six twins, ``D``, ``fu_D``, the
    increment, the prefix) and anything it reads (``B``, the two row vectors, the
    coefficient vectors), and among the written volumes themselves.

    ``_launch`` performs the three stages in order and ``run`` (the base class) rotates
    afterwards; launch 2 reads the freshly written twins by pointer, so the rotation's
    placement after it is a bookkeeping choice, not an ordering one.
    """

    __slots__ = ("dtdx", "four_dtdx", "backward", "xp", "scratch", "_b", "_targets",
                 "_aux", "_curl_coeff", "_kps", "_kms", "_weights", "_divisor",
                 "_kernel", "_curl_kernel", "_pointer", "_flat", "cumsum_calls",
                 "kernel_launches", "_dtype")

    replaces = REPLACES

    #: How many device launches ONE run performs, and how many of them are kernels
    #: this module compiles -- declared, so a launch witness can hold them to it.
    launches_per_run = LAUNCHES_PER_RUN
    kernel_launches_per_run = KERNEL_LAUNCHES_PER_RUN

    def __init__(self, shape: Sequence[int], dtdx: float, block: int,
                 fields: Any, twins: Dict[str, Any],
                 flux: Sequence[Any], targets: Sequence[Any],
                 auxiliaries: Sequence[Any], curl_coefficients: Sequence[Any],
                 constitutive_coefficients: Sequence[Any], xp: Any, scratch: Any,
                 kernel: Any = None, curl_kernel: Any = None,
                 num_warps: Optional[int] = 1) -> None:
        from .launch import CupyPointer, _flat  # noqa: PLC0415

        super().__init__(FAMILY, fields, twins, ROTATED, shape, block, REPLACES,
                         num_warps=num_warps)
        if int(self.shape[1]) != 1:
            raise ValueError(
                f"a Dcyl grid stores one phi cell; shape {self.shape} does not")
        if scratch is None:
            raise ValueError(
                "this plan takes the increment and the scan output from the engine's "
                "StepScratch and binds its cached row vectors; a None pool is refused")
        # float(): the array path multiplies a float32 volume by a Python float, which
        # NumPy/CuPy cast to float32 before the multiply, and Triton types a Python
        # float argument as fp32 -- the same bits.
        self.dtdx = float(dtdx)
        # The 4*Courant of the m = 0 on-axis Dz add, rounded as the array path rounds
        # it (stepping.py:619, NEP-50) -- `cylindrical_triton.CylindricalCurlPlan`'s
        # own argument, formed the same way.
        self.four_dtdx = float(_cyl._float32(4.0 * float(dtdx)))  # noqa: SLF001
        self.backward = BACKWARD
        self.xp = xp
        self.scratch = scratch
        self._pointer = CupyPointer
        self._flat = _flat
        self._b = tuple(CupyPointer(a) for a in flux)
        self._targets = tuple(CupyPointer(a) for a in targets)
        self._aux = tuple(CupyPointer(a) for a in auxiliaries)
        self._curl_coeff = tuple(CupyPointer(_flat(a)) for a in curl_coefficients)
        kps = list(constitutive_coefficients[:3])
        kms = list(constitutive_coefficients[3:])
        if len(self._curl_coeff) != 6 or len(kps) != 3 or len(kms) != 3:
            raise ValueError(
                "this plan binds six curl coefficient vectors (kms/sinv per axis) and "
                "six constitutive ones (kps, then kms, per axis), all on the integer "
                "sub-lattice")
        self._kps = tuple(CupyPointer(_flat(a)) for a in kps)
        self._kms = tuple(CupyPointer(_flat(a)) for a in kms)
        self._dtype = targets[0].dtype
        weights, divisor = rderiv_vectors(xp, scratch, self.shape[0],
                                          getattr(targets[0], "real", targets[0]).dtype)
        if int(weights.shape[0]) != self.shape[0] or int(divisor.shape[0]) != self.shape[0] - 1:
            raise ValueError(
                f"the cached rderiv vectors are {weights.shape} / {divisor.shape} for a "
                f"{self.shape[0]}-row grid; the pool holds another extent's constants")
        self._weights = weights
        self._divisor = divisor
        self._check_aliases(twins, flux, targets, auxiliaries, curl_coefficients,
                            constitutive_coefficients)
        self._kernel = kernel
        self._curl_kernel = curl_kernel
        self.cumsum_calls = 0
        self.kernel_launches = 0

    # -- the pooled buffers -------------------------------------------------
    def _increment(self) -> Any:
        return self.scratch.take(INCREMENT_TAG, self.shape, self._dtype)

    def _prefix_out(self) -> Any:
        return self.scratch.take(PREFIX_TAG, self.shape, self._dtype)

    def _check_aliases(self, twins, flux, targets, auxiliaries,
                       curl_coefficients, constitutive_coefficients) -> None:
        """No written volume may share an allocation with a read one, or another."""
        def address(array: Any) -> Optional[int]:
            data = getattr(array, "data", None)
            pointer = getattr(data, "ptr", None)
            if pointer is not None:
                return int(pointer)
            interface = getattr(array, "__array_interface__", None)
            if isinstance(interface, dict):
                return int(interface["data"][0])
            return None

        outputs: Dict[int, str] = {}
        named = [(name, twins[name]) for name in ROTATED]
        named += [(IN_PLACE[index], array) for index, array in enumerate(targets)]
        named += [(IN_PLACE[3 + index], array)
                  for index, array in enumerate(auxiliaries)]
        named += [(INCREMENT_TAG, self._increment()), (PREFIX_TAG, self._prefix_out())]
        for label, array in named:
            key = address(array)
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
        inputs += [("rderiv_weights", self._weights), ("rderiv_divisor", self._divisor)]
        for label, array in inputs:
            key = address(array)
            if key is not None and key in outputs:
                raise ValueError(
                    f"input {label} aliases output {outputs[key]}: the run would read a "
                    f"volume it is writing")

    # -- the three stages --------------------------------------------------
    def _launch(self, writes: Sequence[Any], reads: Sequence[Any],
                guard: Optional[bool]) -> None:
        from .kernels import ENABLE_FP_FUSION  # noqa: PLC0415

        nx, ny, nz = self.shape
        extra = {} if self.num_warps is None else {"num_warps": self.num_warps}
        pointer = self._pointer
        scratch = [pointer(array) for array in writes]
        prior = [pointer(array) for array in reads]

        # LAUNCH 1 -- update_H to scratch, plus the increment.
        kernel = (self._kernel if self._kernel is not None
                  else cyl_real_update_H_increment_kernel())
        increment = self._increment()
        kernel[self._grid](
            *scratch, *prior, *self._b, pointer(increment),
            pointer(self._flat(self._weights)), pointer(self._flat(self._divisor)),
            *self._kps, *self._kms,
            nx, ny, nz, self.n_elem,
            BLOCK=self.block,
            enable_fp_fusion=ENABLE_FP_FUSION if guard is None else bool(guard),
            **extra,
        )
        self.kernel_launches += 1

        # THE SCAN -- stepping.py:1332-1333, on the pool's own slot. The oracle.
        prefix = self.xp.cumsum(increment, axis=0, out=self._prefix_out())
        self.cumsum_calls += 1

        # LAUNCH 2 -- the certified cylindrical curl, verbatim, over the NEW H, launched
        # exactly as `cylindrical_triton.CylindricalCurlPlan.run` launches it (no
        # num_warps override: that plan takes Triton's default and so does this call).
        curl = (self._curl_kernel if self._curl_kernel is not None
                else _cyl.cylindrical_curl_kernel())
        curl[self._grid](
            *self._targets, *self._aux,
            scratch[0], scratch[1], scratch[2],       # the stepped Hx, Hy, Hz
            pointer(prefix), scratch[1],              # pfx, then Hp for the axis add
            *self._curl_coeff,
            nx, ny, nz, self.n_elem, self.dtdx, self.four_dtdx,
            BACKWARD=self.backward,
            BLOCK=self.block,
            enable_fp_fusion=ENABLE_FP_FUSION if guard is None else bool(guard),
        )
        self.kernel_launches += 1

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
        ) -> Optional[CylindricalRealFusedHdPairPlan]:
    """Build the plan from the engine's own objects, or ``None`` when refused.

    ``None`` is the only refusal. ``kernel`` and ``curl_kernel`` are the mutation
    seams; dropping them is not a slowdown, it is a silent DISARMING of every mutation
    leg that hands a broken copy through them.
    """
    if not cylindrical_real_fused_hd_pair_coverage(fields, pml, sources).covered:
        return None
    suffix, constitutive_suffix = _sub_lattice_suffixes()
    grid = fields.grid
    return CylindricalRealFusedHdPairPlan(
        grid.shape, grid.dt / grid.dx,
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
        grid.xp, fields.scratch,
        kernel=kernel, curl_kernel=curl_kernel, num_warps=num_warps,
    )


def plan_cylindrical_real_fused_hd_pair_from_arrays(
        arrays: Dict[str, Any], flat: Dict[str, Any], dtdx: float,
        fields: Any, xp: Any, scratch: Any, block: Optional[int] = None,
        kernel: Any = None, curl_kernel: Any = None,
        num_warps: Optional[int] = 1) -> CylindricalRealFusedHdPairPlan:
    """Build from bare device arrays -- the gate's route. No predicate runs.

    ``arrays`` supplies the twins under ``scratch_Hx`` ...; ``flat`` supplies
    ``kms_x``/``sinv_x``/``kps_x`` ... on the sub-lattice the caller ALREADY SELECTED,
    so a swapped pair can be handed over and measured.
    """
    shape = tuple(int(n) for n in arrays[IN_PLACE[0]].shape)
    twins = {name: arrays["scratch_" + name] for name in ROTATED}
    return CylindricalRealFusedHdPairPlan(
        shape, dtdx, DEFAULT_BLOCK if block is None else block,
        fields, twins,
        [arrays[name] for name in CONSTITUTIVE_SOURCES],
        [arrays[name] for name in IN_PLACE[:3]],
        [arrays[name] for name in IN_PLACE[3:]],
        [flat[f"{stem}_{axis}"] for axis in "xyz" for stem in ("kms", "sinv")],
        ([flat[f"kps_{axis}"] for axis in "xyz"] + [flat[f"kms_{axis}"] for axis in "xyz"]),
        xp, scratch, kernel=kernel, curl_kernel=curl_kernel, num_warps=num_warps,
    )
