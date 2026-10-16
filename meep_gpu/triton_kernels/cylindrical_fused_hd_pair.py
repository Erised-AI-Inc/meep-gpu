"""The Dcyl COMPLEX H->D product: complex ``update_H`` into scratch, then the certified
complex cylindrical ``step_D`` sub-step -- the radial scan over the NEW Hp and the
curl -- over the rotated field.

The complex-storage twin of :mod:`.cylindrical_real_fused_hd_pair`, on the buildable
rows of the ``(cylindrical complex -> cylindrical complex PML)`` H->D cell (every m the
complex family carries -- m = 0 under complex storage, |m| = 1, |m| >= 2 -- and both
z terminations). Read that module first: the shape, the seam, the withdraw clause, the
launch algebra (three launches against three; kept, not dispatched) and the
arbitration are identical and are not restated here.

=============================================================================
WHAT THE COMPLEX STORAGE CHANGES, AND WHAT IT DOES NOT
=============================================================================

**Launch 1** (:func:`cyl_complex_update_H_scratch`) is the complex Cartesian weld's
constitutive block, statement for statement (:func:`certified_weld_block`): every
program computes its OWN cell's complex ``H`` and ``f_w_H`` word pairs through
:func:`.complex_fused_hd_pair._h_cell_complex` -- ``complex_fields.bloch_constitutive_step``'s
side-H body as a function of a cell, the root's ``constitutive_form`` on each stored
part, the chunk record read at the cell, MEEP's start of H on the run's first launch
-- into launch-local SCRATCH. That body is the one Dcyl's complex ``update_H`` arm
launches (:func:`.cylindrical_complex.plan_cylindrical_complex_constitutive` builds
the certified ``ComplexConstitutivePlan``), so it is IMPORTED, not copied: no complex
multiply, no multiply-add and no divide is spelled here.

**Launches 2 and 3** are :class:`.cylindrical_complex.CylindricalComplexCurlPlan`
-- the certified plan, built by the certified harness builder over the SCRATCH ``H``
and launched by its own ``run``: the column-serial radial scan over the stepped ``Hy``'s
two stored parts, then ``cyl_complex_pml_curl_step`` with its i*m/r rows, axis rules
and Courant words, all bound by that module's own host functions. As on the real
family, the plan is built once over each allocation the rotation alternates between.

Import contract: importable WITHOUT Triton.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import complex_fused_hd_pair as _complex
from . import coverage as _coverage
from . import cylindrical_complex as _cx
from . import cylindrical_real_fused_hd_pair as _real
from . import fused_hd_pair as _cartesian
from .. import withdraw_hoist as _withdraw_hoist
from .complex_fields import _word_view
from .offdiag_scratch_weld import ScratchWeldPairPlan, twin_table

FAMILY = "cylindrical_fused_hd_pair"
SLOT = "update_H"
REPLACES: Tuple[str, ...] = ("update_H", "step_D")
SEAM: str = _withdraw_hoist.SEAM
CONSTITUTIVE_SIDE = "H"
CURL_SUB_STEP = "step_D"

#: The two arms this product implements on ``(update_H, step_D)`` -- the labels the
#: composer writes for ``cylindrical_complex_constitutive_coverage('H')`` and
#: ``cylindrical_complex_curl_coverage('step_D')``.
ARMS: Tuple[str, str] = ("cylindrical complex", "cylindrical complex PML")

ROTATED: Tuple[str, ...] = _real.ROTATED
IN_PLACE: Tuple[str, ...] = _real.IN_PLACE
CONSTITUTIVE_SOURCES: Tuple[str, ...] = _real.CONSTITUTIVE_SOURCES

#: The radial sum's source and ir0, read off the complex family's own table.
PREFIX_COMPONENT: str = _cx.PREFIX[CURL_SUB_STEP]["component"]
PREFIX_IR0: float = float(_cx.PREFIX[CURL_SUB_STEP]["ir0"])

BACKWARD = 1
DEFAULT_BLOCK = 256
LAUNCHES_PER_RUN = _real.LAUNCHES_PER_RUN
KERNEL_LAUNCHES_PER_RUN = _real.KERNEL_LAUNCHES_PER_RUN
CARRIES_DEPOSIT_REPAIR = False
REPAIR_PATHS: Tuple[str, ...] = ()
HOISTS_THE_WITHDRAW = False
INSTALLABLE = False
INSTALLABLE_REASON = _real.INSTALLABLE_REASON.replace(
    "gate_triton_cylindrical_real_fused_hd_pair.py",
    "gate_triton_cylindrical_fused_hd_pair.py")

#: The helpers this product's own kernel reaches, and NO OTHER: none. The complex
#: constitutive is imported, the curl and the scan are the certified plan's.
MULTIPLY_HELPERS: Tuple[str, ...] = ()

__all__ = [
    "ARMS", "BACKWARD", "CARRIES_DEPOSIT_REPAIR", "CONSTITUTIVE_SIDE",
    "CONSTITUTIVE_SOURCES", "CURL_SUB_STEP", "DEFAULT_BLOCK", "FAMILY",
    "HOISTS_THE_WITHDRAW", "INSTALLABLE", "INSTALLABLE_REASON", "IN_PLACE",
    "KERNEL_LAUNCHES_PER_RUN", "LAUNCHES_PER_RUN", "MULTIPLY_HELPERS",
    "PREFIX_COMPONENT", "PREFIX_IR0", "REPAIR_PATHS", "REPLACES", "ROTATED", "SEAM",
    "SLOT",
    "CylindricalFusedHdPairPlan",
    "certified_weld_block", "cyl_complex_update_H_scratch_kernel",
    "cylindrical_fused_hd_pair_coverage", "explain_cylindrical_fused_hd_pair",
    "lifted_weld_block", "plan_cylindrical_fused_hd_pair",
    "plan_cylindrical_fused_hd_pair_from_arrays",
]


# ---------------------------------------------------------------------------
# The machine-checked lift -- host text, no Triton
# ---------------------------------------------------------------------------

#: This module's launch-1 kernel, by name.
SCRATCH_KERNEL = "cyl_complex_update_H_scratch"


def certified_weld_block() -> List[str]:
    """The complex Cartesian H->D weld's constitutive block, statement by statement.

    :func:`.complex_fused_hd_pair.fused_complex_constitutive_curl_H_to_D` between its
    index decode and its curl anchor, cut off that file.
    """
    source = _cartesian._source_of(  # noqa: SLF001 - the shared cutter, by design
        "fused_complex_constitutive_curl_H_to_D", Path(_complex.__file__))
    return _real._statements(_cartesian._cut(  # noqa: SLF001
        source, _real.WELD_DECODE_END, stop=_complex.CURL_BODY_ANCHOR))


def lifted_weld_block() -> List[str]:
    """This module's launch-1 kernel below its decode, statement by statement."""
    source = _cartesian._source_of(SCRATCH_KERNEL, Path(__file__))  # noqa: SLF001
    return _real._statements(_cartesian._cut(source, _real.WELD_DECODE_END))  # noqa: SLF001


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

    # THE CERTIFIED COMPLEX CONSTITUTIVE, imported as a device function rather than
    # copied: its lift is machine-checked in complex_fused_hd_pair and re-checked by
    # this family's gate.
    from .complex_fused_hd_pair import _h_cell_complex  # noqa: PLC0415

    @triton.jit(do_not_specialize=["start_h"])
    def cyl_complex_update_H_scratch(
        ho0, ho1, ho2,                  # SCRATCH out: stepped Hx,Hy,Hz    (c8 as words)
        wo0, wo1, wo2,                  # SCRATCH out: stepped f_w_H*      (c8 as words)
        hi0, hi1, hi2,                  # PRE-LAUNCH Hx, Hy, Hz              (read-only)
        wi0, wi1, wi2,                  # PRE-LAUNCH f_w_Hx, f_w_Hy, f_w_Hz (read-only)
        b0, b1, b2,                     # Bx, By, Bz                         (read-only)
        kp0, kp1, kp2, kmx, kmy, kmz,   # kps/kms on each component's own axis, INTEGER
        cb,                             # chunk record: the constitutive's dsigw lines
        nx, ny, nz, n_elem, start_h,    # n_elem = COMPLEX cells; MEEP's start of H
        BLOCK: tl.constexpr,
    ):
        """complex ``update_H`` into scratch: the complex weld's constitutive block.

        ``hi``/``wi``/``b`` are const for the whole dispatch and ``ho``/``wo`` are
        write-only scratch, so nothing written by this launch is read by it.
        """
        idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        live = idx < n_elem
        nyz = ny * nz
        k = idx % nz
        plane = idx // nz
        j = plane % ny
        i = plane // ny

        # complex update_H at the program's own cell, to SCRATCH (the complex weld's
        # block)
        (own0_re, own0_im, own1_re, own1_im, own2_re, own2_im,
         src0_re, src0_im, src1_re, src1_im, src2_re, src2_im,
         dw0, dw1, dw2) = _h_cell_complex(
            i, j, k, live, hi0, hi1, hi2, wi0, wi1, wi2, b0, b1, b2,
            kp0, kp1, kp2, kmx, kmy, kmz, cb, start_h, ny, nz)
        tl.store(ho0 + 2 * idx, own0_re, mask=live)
        tl.store(ho0 + 2 * idx + 1, own0_im, mask=live)
        tl.store(ho1 + 2 * idx, own1_re, mask=live)
        tl.store(ho1 + 2 * idx + 1, own1_im, mask=live)
        tl.store(ho2 + 2 * idx, own2_re, mask=live)
        tl.store(ho2 + 2 * idx + 1, own2_im, mask=live)
        # The history takes B on the accumulation's cells only (the auxiliary
        # contract); every other cell of the twin takes the pre-launch word pair.
        tl.store(wo0 + 2 * idx, tl.where(dw0, src0_re, tl.load(
            wi0 + 2 * idx, mask=live & (dw0 == 0), other=0.0)), mask=live)
        tl.store(wo0 + 2 * idx + 1, tl.where(dw0, src0_im, tl.load(
            wi0 + 2 * idx + 1, mask=live & (dw0 == 0), other=0.0)), mask=live)
        tl.store(wo1 + 2 * idx, tl.where(dw1, src1_re, tl.load(
            wi1 + 2 * idx, mask=live & (dw1 == 0), other=0.0)), mask=live)
        tl.store(wo1 + 2 * idx + 1, tl.where(dw1, src1_im, tl.load(
            wi1 + 2 * idx + 1, mask=live & (dw1 == 0), other=0.0)), mask=live)
        tl.store(wo2 + 2 * idx, tl.where(dw2, src2_re, tl.load(
            wi2 + 2 * idx, mask=live & (dw2 == 0), other=0.0)), mask=live)
        tl.store(wo2 + 2 * idx + 1, tl.where(dw2, src2_im, tl.load(
            wi2 + 2 * idx + 1, mask=live & (dw2 == 0), other=0.0)), mask=live)

else:  # pragma: no cover - the laptop path
    cyl_complex_update_H_scratch = None  # type: ignore[assignment]


def cyl_complex_update_H_scratch_kernel() -> Any:
    """The shipped launch-1 kernel object, or a refusal naming the missing import."""
    if triton is None:  # pragma: no cover - the laptop path
        raise ImportError(
            "the cylindrical complex H->D pair needs Triton to launch; the predicate, "
            "the lift checks and the plan builder answer without it "
            f"({_TRITON_IMPORT_ERROR})")
    return cyl_complex_update_H_scratch


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def cylindrical_fused_hd_pair_coverage(fields: Any, pml: Any, sources: Any = None
                                       ) -> "_coverage.Coverage":
    """May ONE run span ``update_H`` -> the electric withdraw -> ``step_D`` on this
    complex Dcyl run? The two certified halves' OWN predicates, in the driver's order,
    plus the seam clauses -- nothing weakened, every reason prefixed by its half."""
    grid = getattr(fields, "grid", None)
    if grid is None:
        return _coverage.Coverage(False, ("fields carries no grid",))

    reasons: List[str] = []
    constitutive = _cx.cylindrical_complex_constitutive_coverage(
        fields, pml, CONSTITUTIVE_SIDE)
    if not constitutive.covered:
        reasons.extend(f"cylindrical complex constitutive half: {reason}"
                       for reason in constitutive.reasons)
    curl = _cx.cylindrical_complex_curl_coverage(fields, pml, CURL_SUB_STEP)
    if not curl.covered:
        reasons.extend(f"cylindrical complex curl half: {reason}"
                       for reason in curl.reasons)

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

    for name in ROTATED:
        if getattr(fields, name, None) is None:
            reasons.append(
                f"{name} is not allocated; this weld rotates it against a plan-owned "
                f"scratch twin after every run")
    for name in IN_PLACE + CONSTITUTIVE_SOURCES:
        if getattr(fields, name, None) is None:
            reasons.append(f"{name} is not allocated")
    return _coverage.Coverage(not reasons, tuple(dict.fromkeys(reasons)))


def explain_cylindrical_fused_hd_pair(fields: Any, pml: Any, sources: Any = None
                                      ) -> "_coverage.Coverage":
    return cylindrical_fused_hd_pair_coverage(fields, pml, sources)


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

class CylindricalFusedHdPairPlan(_cartesian.StartsH, ScratchWeldPairPlan):
    """Launch 1 into scratch, then the certified complex cylindrical ``step_D`` plan.

    The real family's shape (:class:`.cylindrical_real_fused_hd_pair.CylindricalRealFusedHdPairPlan`)
    on complex64 word pairs: the certified curl plan is built once over each rotating
    allocation, launch 1 word-views this run's scratch and pre-launch volumes at launch
    time, and ``__init__`` refuses aliasing among everything written and between
    written and read.
    """

    __slots__ = ("dtdx", "courant", "m", "bcz", "backward", "_b", "_kps", "_kms",
                 "_kernel", "_pointer", "_curl_plans", "kernel_launches", "_dtype",
                 "_h_state")

    replaces = REPLACES
    launches_per_run = LAUNCHES_PER_RUN
    kernel_launches_per_run = KERNEL_LAUNCHES_PER_RUN

    def __init__(self, shape: Sequence[int], dtdx: float, courant: float, m: int,
                 accurate_fields_near_cylorigin: bool, bcz: int,
                 block: int, fields: Any, twins: Dict[str, Any],
                 flux: Sequence[Any], targets: Sequence[Any],
                 auxiliaries: Sequence[Any], curl_coefficients: Sequence[Any],
                 constitutive_coefficients: Sequence[Any], xp: Any, *,
                 chunk_bounds: Any, scan_starts: Any, kernel: Any = None,
                 curl_kernel: Any = None, scan_kernel: Any = None,
                 num_warps: Optional[int] = 1, curl_num_warps: Optional[int] = None,
                 h_state: Any = None) -> None:
        from .launch import CupyPointer, _flat  # noqa: PLC0415

        super().__init__(FAMILY, fields, twins, ROTATED, shape, block, REPLACES,
                         num_warps=num_warps, chunk_bounds=chunk_bounds)
        if int(self.shape[1]) != 1:
            raise ValueError(
                f"a Dcyl grid stores one phi cell; shape {self.shape} does not")
        self._dtype = getattr(targets[0], "dtype", None)
        if str(self._dtype) != "complex64":
            raise ValueError(f"this plan steps complex64 word pairs, not {self._dtype}")
        if 2 * self.n_elem >= 2 ** 31:
            raise ValueError(
                f"{self.n_elem} complex cells needs {2 * self.n_elem} int32 word "
                f"indices, which overflows launch 1's 2 * idx addressing")
        self.dtdx = float(dtdx)
        self.courant = float(courant)
        self.m = int(m)
        self.bcz = int(bcz)
        self.backward = BACKWARD
        self._pointer = CupyPointer
        self._b = tuple(CupyPointer(_word_view(a)) for a in flux)
        kps = list(constitutive_coefficients[:3])
        kms = list(constitutive_coefficients[3:])
        if len(curl_coefficients) != 6 or len(kps) != 3 or len(kms) != 3:
            raise ValueError(
                "this plan binds six curl coefficient vectors (kms/sinv per axis) and "
                "six constitutive ones (kps, then kms, per axis), all on the integer "
                "sub-lattice")
        self._kps = tuple(CupyPointer(_flat(a)) for a in kps)
        self._kms = tuple(CupyPointer(_flat(a)) for a in kms)
        # THE CERTIFIED COMPLEX step_D PLAN, once over each rotating allocation, built
        # by the certified harness builder.
        live = {name: getattr(fields, name) for name in ROTATED}
        curl_flat = dict(zip((f"{stem}_{axis}" for axis in "xyz"
                              for stem in ("kms", "sinv")), curl_coefficients))
        plans: Dict[int, Any] = {}
        for magnetic in (live, twins):
            arrays = {name: magnetic[name] for name in ("Hx", "Hy", "Hz")}
            arrays.update(zip(IN_PLACE[:3], targets))
            arrays.update(("fu_" + name, aux) for name, aux in zip(IN_PLACE[:3],
                                                                  auxiliaries))
            plan = _cx.plan_cylindrical_complex_curl_from_arrays(
                CURL_SUB_STEP, arrays, curl_flat, dtdx, courant, m,
                accurate_fields_near_cylorigin, bcz, xp,
                chunk_bounds=chunk_bounds, scan_starts=scan_starts, block=block,
                kernel=curl_kernel, scan_kernel=scan_kernel, num_warps=curl_num_warps)
            plans[_real._address(magnetic[PREFIX_COMPONENT])] = plan  # noqa: SLF001
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
        _real.CylindricalRealFusedHdPairPlan._check_aliases(  # the real family's rule
            self, twins, flux, targets, auxiliaries, curl_coefficients,
            constitutive_coefficients)

    def _curl_plan_for(self, writes: Sequence[Any]) -> Any:
        """The certified plan whose sources are this run's scratch ``H``."""
        return _real.CylindricalRealFusedHdPairPlan._curl_plan_for(self, writes)

    # -- the three launches ------------------------------------------------
    def _launch(self, writes: Sequence[Any], reads: Sequence[Any],
                guard: Optional[bool]) -> None:
        from .kernels import ENABLE_FP_FUSION  # noqa: PLC0415

        nx, ny, nz = self.shape
        extra = {} if self.num_warps is None else {"num_warps": self.num_warps}
        pointer = self._pointer
        # WORD-VIEWED HERE, not at plan time: the rotation moves which allocation is
        # live, so the view has to be taken of whatever the engine names now.
        scratch = [pointer(_word_view(array)) for array in writes]
        prior = [pointer(_word_view(array)) for array in reads]

        # LAUNCH 1 -- complex update_H to scratch.
        kernel = (self._kernel if self._kernel is not None
                  else cyl_complex_update_H_scratch_kernel())
        kernel[self._grid](
            *scratch, *prior, *self._b, *self._kps, *self._kms, self._cb,
            nx, ny, nz, self.n_elem, self.start_h(),
            BLOCK=self.block,
            enable_fp_fusion=ENABLE_FP_FUSION if guard is None else bool(guard),
            **extra,
        )
        # LAUNCHES 2 AND 3 -- the certified complex step_D plan over the scratch H.
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
        return (f"CylindricalFusedHdPairPlan(shape={self.shape}, m={self.m}, "
                f"bcz={self.bcz}, block={self.block}, num_warps={self.num_warps})")


def _sub_lattice_suffixes() -> Tuple[str, str]:
    """``(curl suffix, constitutive suffix)``, READ from the shipped tables and asserted
    equal: both halves take the INTEGER sub-lattice on this seam."""
    curl = _cx.SUB_STEPS[CURL_SUB_STEP]["suffix"]
    constitutive = ("_h" if _coverage.CONSTITUTIVE_SIDES[CONSTITUTIVE_SIDE]
                    ["half_integer"] else "")
    if curl != constitutive:
        raise AssertionError(
            f"step_D reads the {curl or 'integer'!r} PML sub-lattice and update_H the "
            f"{constitutive or 'integer'!r} one; both halves of this product take one "
            f"lattice and that is only correct while they agree")
    if int(_cx.SUB_STEPS[CURL_SUB_STEP]["backward"]) != BACKWARD:
        raise AssertionError(
            f"the table says step_D differences {_cx.SUB_STEPS[CURL_SUB_STEP]['backward']}, "
            f"and this product declares BACKWARD = {BACKWARD}")
    return curl, constitutive


def plan_cylindrical_fused_hd_pair(
        fields: Any, pml: Any, sources: Any = None, block: Optional[int] = None,
        num_warps: Optional[int] = 1, kernel: Any = None, curl_kernel: Any = None,
        scan_kernel: Any = None
        ) -> Optional[CylindricalFusedHdPairPlan]:
    """Build the plan from the engine's own objects, or ``None`` when refused.

    ``kernel``, ``curl_kernel`` and ``scan_kernel`` are the mutation seams. The chunk
    record and the scan's starts line are the certified host bindings, called.
    """
    from .launch import chunk_bounds_for  # noqa: PLC0415
    if not cylindrical_fused_hd_pair_coverage(fields, pml, sources).covered:
        return None
    suffix, constitutive_suffix = _sub_lattice_suffixes()
    from ..stepping import _boundary_kinds as resolve  # noqa: PLC0415

    grid = fields.grid
    kinds = resolve(grid, pml)
    return CylindricalFusedHdPairPlan(
        grid.shape, grid.dt / grid.dx, float(grid.courant), int(grid.m),
        bool(grid.accurate_fields_near_cylorigin),
        1 if kinds[2] == "metallic" else 0,
        DEFAULT_BLOCK if block is None else block,
        fields, twin_table(fields, ROTATED),
        [getattr(fields, name) for name in CONSTITUTIVE_SOURCES],
        [getattr(fields, name) for name in IN_PLACE[:3]],
        [getattr(fields, name) for name in IN_PLACE[3:]],
        [getattr(pml, f"{stem}_{axis}{suffix}")
         for axis in "xyz" for stem in ("kms", "sinv")],
        ([getattr(pml, f"kps_{axis}{constitutive_suffix}") for axis in "xyz"]
         + [getattr(pml, f"kms_{axis}{constitutive_suffix}") for axis in "xyz"]),
        grid.xp,
        chunk_bounds=chunk_bounds_for(fields, pml, None),
        scan_starts=_cx.radial_scan_starts(fields, pml, CURL_SUB_STEP, None),
        kernel=kernel, curl_kernel=curl_kernel, scan_kernel=scan_kernel,
        num_warps=num_warps, h_state=fields,
    )


def plan_cylindrical_fused_hd_pair_from_arrays(
        arrays: Dict[str, Any], curl_flat: Dict[str, Any],
        constitutive_flat: Dict[str, Any], dtdx: float, courant: float, m: int,
        accurate_fields_near_cylorigin: bool, bcz: int,
        fields: Any, xp: Any, block: Optional[int] = None,
        kernel: Any = None, curl_kernel: Any = None, scan_kernel: Any = None,
        num_warps: Optional[int] = 1, *, chunk_bounds: Any,
        scan_starts: Any) -> CylindricalFusedHdPairPlan:
    """Build from bare device arrays -- the gate's route. No predicate runs, and
    ``chunk_bounds`` and ``scan_starts`` are REQUIRED, as the certified curl plan
    requires them. No start of H: a harness state has already started."""
    shape = tuple(int(n) for n in arrays[IN_PLACE[0]].shape)
    twins = {name: arrays["scratch_" + name] for name in ROTATED}
    return CylindricalFusedHdPairPlan(
        shape, dtdx, courant, m, accurate_fields_near_cylorigin, bcz,
        DEFAULT_BLOCK if block is None else block,
        fields, twins,
        [arrays[name] for name in CONSTITUTIVE_SOURCES],
        [arrays[name] for name in IN_PLACE[:3]],
        [arrays[name] for name in IN_PLACE[3:]],
        [curl_flat[f"{stem}_{axis}"] for axis in "xyz" for stem in ("kms", "sinv")],
        ([constitutive_flat[f"kps_{axis}"] for axis in "xyz"]
         + [constitutive_flat[f"kms_{axis}"] for axis in "xyz"]),
        xp, chunk_bounds=chunk_bounds, scan_starts=scan_starts, kernel=kernel,
        curl_kernel=curl_kernel, scan_kernel=scan_kernel, num_warps=num_warps,
    )
