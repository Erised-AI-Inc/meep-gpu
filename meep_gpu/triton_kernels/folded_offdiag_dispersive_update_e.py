"""Triton ``update_E`` for the fold + tensor epsilon + dispersion intersection.

This is the one residual product exercised by MEEP's
``absorbed_power_density.py`` configuration.  It is not a host-side composition
of the folded-offdiagonal and dispersive kernels: the off-diagonal Yee row reads
the *other* components at four addresses, and every one of those reads must be
``D - P0 - P1 - ...`` in registered-polarization order.  Materialising three
temporary volumes would preserve the array path but add the full-volume CuPy
passes this kernel exists to remove.  The kernel below instead performs the
ordered subtraction at every gathered address and then uses the certified
folded ghost, row-mask, wall-mask and PML recurrence shapes.

The coverage predicate is deliberately fail closed.  It requires real float32
CuPy storage, an active PML, at least one mirror fold, at least one surviving
off-diagonal row and at least one registered supported electric susceptibility.
It also validates both ``P`` and ``P_prev`` because the ADE update rotates those
buffers after every step.  The plan caches pole *order* but resolves the live
``P`` pointers on every launch through :class:`LivePoleBinding`.

This module is intentionally not wired into ``launch.py`` here.  Its exact
central integration contract is recorded at the bottom so the shared planner
can be edited once after independent families return.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import coverage as _coverage
from . import folded_offdiag_update_e as _folded
from . import offdiag_update_e as _offdiag
from . import symmetry as _symmetry
from .dispersive_update_e import (
    E_TERMS,
    MAX_POLES,
    LivePoleBinding,
    StaticPoleBinding,
    poles_per_component,
)

DEFAULT_BLOCK = _offdiag.DEFAULT_BLOCK
ROW_SLOTS = _offdiag.ROW_SLOTS
HALF_INTEGER = True

CODE_PERIODIC = _folded.CODE_PERIODIC
CODE_METALLIC = _folded.CODE_METALLIC
CODE_MIRROR_METALLIC = _folded.CODE_MIRROR_METALLIC
CODE_MIRROR_PERIODIC = _folded.CODE_MIRROR_PERIODIC
MIRROR_CODES = _folded.MIRROR_CODES
MIRROR_SOURCE_INDEX = _folded.MIRROR_SOURCE_INDEX

SHARED_CLAUSES = (
    "coverage._susceptibility_reasons",
    "coverage._layout_reasons",
    "coverage._inverse_epsilon_reasons",
    "coverage._coefficient_reasons",
    "coverage._volume_reasons",
    "folded_offdiag_update_e._folded_offdiag_grid_reasons",
    "folded_offdiag_update_e.mirror_ghost_weights",
    "folded_offdiag_update_e.FoldedOffdiagConstitutivePlan",
    "offdiag_update_e._row_reasons",
    "offdiag_update_e.row_volumes_for",
    "offdiag_update_e.wall_mask_axes",
    "symmetry._has_real_fold",
    "dispersive_update_e.poles_per_component",
    "dispersive_update_e.LivePoleBinding",
)


try:  # pragma: no cover - the absent branch is the laptop path
    import triton
    import triton.language as tl
    _TRITON_IMPORT_ERROR: Optional[BaseException] = None
except ImportError as _exc:  # pragma: no cover
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _TRITON_IMPORT_ERROR = _exc


if triton is not None:
    PERIODIC = tl.constexpr(CODE_PERIODIC)
    METALLIC = tl.constexpr(CODE_METALLIC)
    MIRROR_ROW = tl.constexpr(MIRROR_SOURCE_INDEX)
    # A direct global binding, rather than an attribute lookup through
    # ``_folded`` inside a JIT body. Triton resolves called JIT functions from
    # the defining module's globals; module-attribute calls are not a frontend
    # contract we ask it to carry.
    _masked_row_sum = _folded._masked_row_sum

    @triton.jit
    def _load_d_minus_p(g, p0, p1, p2, p3, p4, p5, p6, p7,
                        offset, valid, NP: tl.constexpr):
        """Load ``D - P0 - ...`` left-to-right, exactly as ``subtract_into``.

        The sequential form is load-bearing: pre-accumulating ``sum(P)`` changes
        float32 rounding for two or more poles.  ``NP`` is a constexpr, so a
        component with fewer poles emits no loads from its dead pointer slots.
        """
        value = tl.load(g + offset, mask=valid, other=0.0)
        if NP > 0:
            value = value - tl.load(p0 + offset, mask=valid, other=0.0)
        if NP > 1:
            value = value - tl.load(p1 + offset, mask=valid, other=0.0)
        if NP > 2:
            value = value - tl.load(p2 + offset, mask=valid, other=0.0)
        if NP > 3:
            value = value - tl.load(p3 + offset, mask=valid, other=0.0)
        if NP > 4:
            value = value - tl.load(p4 + offset, mask=valid, other=0.0)
        if NP > 5:
            value = value - tl.load(p5 + offset, mask=valid, other=0.0)
        if NP > 6:
            value = value - tl.load(p6 + offset, mask=valid, other=0.0)
        if NP > 7:
            value = value - tl.load(p7 + offset, mask=valid, other=0.0)
        return value

    @triton.jit
    def _folded_dispersive_term(g, p0, p1, p2, p3, p4, p5, p6, p7,
                                u, o_c, o_d, o_u, o_ud,
                                v_c, v_d, v_u, v_ud, w_d,
                                NP: tl.constexpr, MG: tl.constexpr):
        """One tensor-row partner term whose four field reads are ``D-sum(P)``."""
        if MG:
            near = (_load_d_minus_p(
                g, p0, p1, p2, p3, p4, p5, p6, p7, o_c, v_c, NP)
                + w_d * _load_d_minus_p(
                    g, p0, p1, p2, p3, p4, p5, p6, p7, o_d, v_d, NP))
            far = (_load_d_minus_p(
                g, p0, p1, p2, p3, p4, p5, p6, p7, o_u, v_u, NP)
                + w_d * _load_d_minus_p(
                    g, p0, p1, p2, p3, p4, p5, p6, p7, o_ud, v_ud, NP))
        else:
            near = (_load_d_minus_p(
                g, p0, p1, p2, p3, p4, p5, p6, p7, o_c, v_c, NP)
                + _load_d_minus_p(
                    g, p0, p1, p2, p3, p4, p5, p6, p7, o_d, v_d, NP))
            far = (_load_d_minus_p(
                g, p0, p1, p2, p3, p4, p5, p6, p7, o_u, v_u, NP)
                + _load_d_minus_p(
                    g, p0, p1, p2, p3, p4, p5, p6, p7, o_ud, v_ud, NP))
        return 0.25 * ((near * tl.load(u + o_c, mask=v_c, other=0.0))
                       + (far * tl.load(u + o_u, mask=v_u, other=0.0)))

    @triton.jit
    def folded_offdiag_dispersive_constitutive_step(
        f0, f1, f2,
        w0, w1, w2,
        g0, g1, g2,
        e0, e1, e2,
        a0, a1, a2, a3, a4, a5, a6, a7,
        b0, b1, b2, b3, b4, b5, b6, b7,
        c0, c1, c2, c3, c4, c5, c6, c7,
        u01, u02, u11, u12, u21, u22,
        kp0, km0, kp1, km1, kp2, km2,
        gwx, gwy, gwz,
        nx, ny, nz, n_elem,
        NP0: tl.constexpr, NP1: tl.constexpr, NP2: tl.constexpr,
        R01: tl.constexpr, R02: tl.constexpr,
        R11: tl.constexpr, R12: tl.constexpr,
        R21: tl.constexpr, R22: tl.constexpr,
        BCX: tl.constexpr, BCY: tl.constexpr, BCZ: tl.constexpr,
        MG_X: tl.constexpr, MG_Y: tl.constexpr, MG_Z: tl.constexpr,
        WM_X: tl.constexpr, WM_Y: tl.constexpr, WM_Z: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """Fused ``D-sum(P)`` plus folded tensor-row plus PML ``update_E``."""
        idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        live = idx < n_elem
        nyz = ny * nz
        k = idx % nz
        plane = idx // nz
        j = plane % ny
        i = plane // ny
        at_x, at_y, at_z = i == 0, j == 0, k == 0

        di, dj, dk = i - 1, j - 1, k - 1
        ui, uj, uk = i + 1, j + 1, k + 1
        dvx, dvy, dvz = live, live, live
        uvx, uvy, uvz = live, live, live
        wx, wy, wz = 1.0, 1.0, 1.0
        if BCX == PERIODIC:
            di = tl.where(di < 0, nx - 1, di)
            ui = tl.where(ui == nx, 0, ui)
        elif BCX == METALLIC:
            dvx = live & (di >= 0)
            uvx = live & (ui < nx)
        else:
            di = tl.where(at_x, MIRROR_ROW, di)
            wx = tl.where(at_x, gwx, 1.0)
            uvx = live & (ui < nx)
        if BCY == PERIODIC:
            dj = tl.where(dj < 0, ny - 1, dj)
            uj = tl.where(uj == ny, 0, uj)
        elif BCY == METALLIC:
            dvy = live & (dj >= 0)
            uvy = live & (uj < ny)
        else:
            dj = tl.where(at_y, MIRROR_ROW, dj)
            wy = tl.where(at_y, gwy, 1.0)
            uvy = live & (uj < ny)
        if BCZ == PERIODIC:
            dk = tl.where(dk < 0, nz - 1, dk)
            uk = tl.where(uk == nz, 0, uk)
        elif BCZ == METALLIC:
            dvz = live & (dk >= 0)
            uvz = live & (uk < nz)
        else:
            dk = tl.where(at_z, MIRROR_ROW, dk)
            wz = tl.where(at_z, gwz, 1.0)
            uvz = live & (uk < nz)

        kp_0 = tl.load(kp0 + i, mask=live, other=0.0)
        km_0 = tl.load(km0 + i, mask=live, other=0.0)
        kp_1 = tl.load(kp1 + j, mask=live, other=0.0)
        km_1 = tl.load(km1 + j, mask=live, other=0.0)
        kp_2 = tl.load(kp2 + k, mask=live, other=0.0)
        km_2 = tl.load(km2 + k, mask=live, other=0.0)

        # Ex: own x, partners y then z.
        prev0 = tl.load(w0 + idx, mask=live, other=0.0)
        gs0 = _load_d_minus_p(
            g0, a0, a1, a2, a3, a4, a5, a6, a7, idx, live, NP0)
        us0 = tl.load(e0 + idx, mask=live, other=0.0)
        if R01:
            total0 = _folded_dispersive_term(
                g1, b0, b1, b2, b3, b4, b5, b6, b7, u01,
                idx, i * nyz + dj * nz + k, ui * nyz + j * nz + k,
                ui * nyz + dj * nz + k, live, dvy, uvx, uvx & dvy,
                wy, NP1, MG_Y)
            if R02:
                total0 = total0 + _folded_dispersive_term(
                    g2, c0, c1, c2, c3, c4, c5, c6, c7, u02,
                    idx, i * nyz + j * nz + dk, ui * nyz + j * nz + k,
                    ui * nyz + j * nz + dk, live, dvz, uvx, uvx & dvz,
                    wz, NP2, MG_Z)
            src0 = _masked_row_sum(
                gs0 * us0, total0, at_y, at_z, WM_Y, WM_Z)
        else:
            if R02:
                total0 = _folded_dispersive_term(
                    g2, c0, c1, c2, c3, c4, c5, c6, c7, u02,
                    idx, i * nyz + j * nz + dk, ui * nyz + j * nz + k,
                    ui * nyz + j * nz + dk, live, dvz, uvx, uvx & dvz,
                    wz, NP2, MG_Z)
                src0 = _masked_row_sum(
                    gs0 * us0, total0, at_y, at_z, WM_Y, WM_Z)
            else:
                src0 = gs0 * us0
        tl.store(w0 + idx, src0, mask=live)
        value0 = tl.load(f0 + idx, mask=live, other=0.0)
        value0 = value0 + kp_0 * src0
        value0 = value0 - km_0 * prev0
        tl.store(f0 + idx, value0, mask=live)

        # Ey: own y, partners z then x.
        prev1 = tl.load(w1 + idx, mask=live, other=0.0)
        gs1 = _load_d_minus_p(
            g1, b0, b1, b2, b3, b4, b5, b6, b7, idx, live, NP1)
        us1 = tl.load(e1 + idx, mask=live, other=0.0)
        if R11:
            total1 = _folded_dispersive_term(
                g2, c0, c1, c2, c3, c4, c5, c6, c7, u11,
                idx, i * nyz + j * nz + dk, i * nyz + uj * nz + k,
                i * nyz + uj * nz + dk, live, dvz, uvy, uvy & dvz,
                wz, NP2, MG_Z)
            if R12:
                total1 = total1 + _folded_dispersive_term(
                    g0, a0, a1, a2, a3, a4, a5, a6, a7, u12,
                    idx, di * nyz + j * nz + k, i * nyz + uj * nz + k,
                    di * nyz + uj * nz + k, live, dvx, uvy, uvy & dvx,
                    wx, NP0, MG_X)
            src1 = _masked_row_sum(
                gs1 * us1, total1, at_x, at_z, WM_X, WM_Z)
        else:
            if R12:
                total1 = _folded_dispersive_term(
                    g0, a0, a1, a2, a3, a4, a5, a6, a7, u12,
                    idx, di * nyz + j * nz + k, i * nyz + uj * nz + k,
                    di * nyz + uj * nz + k, live, dvx, uvy, uvy & dvx,
                    wx, NP0, MG_X)
                src1 = _masked_row_sum(
                    gs1 * us1, total1, at_x, at_z, WM_X, WM_Z)
            else:
                src1 = gs1 * us1
        tl.store(w1 + idx, src1, mask=live)
        value1 = tl.load(f1 + idx, mask=live, other=0.0)
        value1 = value1 + kp_1 * src1
        value1 = value1 - km_1 * prev1
        tl.store(f1 + idx, value1, mask=live)

        # Ez: own z, partners x then y.
        prev2 = tl.load(w2 + idx, mask=live, other=0.0)
        gs2 = _load_d_minus_p(
            g2, c0, c1, c2, c3, c4, c5, c6, c7, idx, live, NP2)
        us2 = tl.load(e2 + idx, mask=live, other=0.0)
        if R21:
            total2 = _folded_dispersive_term(
                g0, a0, a1, a2, a3, a4, a5, a6, a7, u21,
                idx, di * nyz + j * nz + k, i * nyz + j * nz + uk,
                di * nyz + j * nz + uk, live, dvx, uvz, uvz & dvx,
                wx, NP0, MG_X)
            if R22:
                total2 = total2 + _folded_dispersive_term(
                    g1, b0, b1, b2, b3, b4, b5, b6, b7, u22,
                    idx, i * nyz + dj * nz + k, i * nyz + j * nz + uk,
                    i * nyz + dj * nz + uk, live, dvy, uvz, uvz & dvy,
                    wy, NP1, MG_Y)
            src2 = _masked_row_sum(
                gs2 * us2, total2, at_x, at_y, WM_X, WM_Y)
        else:
            if R22:
                total2 = _folded_dispersive_term(
                    g1, b0, b1, b2, b3, b4, b5, b6, b7, u22,
                    idx, i * nyz + dj * nz + k, i * nyz + j * nz + uk,
                    i * nyz + dj * nz + uk, live, dvy, uvz, uvz & dvy,
                    wy, NP1, MG_Y)
                src2 = _masked_row_sum(
                    gs2 * us2, total2, at_x, at_y, WM_X, WM_Y)
            else:
                src2 = gs2 * us2
        tl.store(w2 + idx, src2, mask=live)
        value2 = tl.load(f2 + idx, mask=live, other=0.0)
        value2 = value2 + kp_2 * src2
        value2 = value2 - km_2 * prev2
        tl.store(f2 + idx, value2, mask=live)

else:  # pragma: no cover - the laptop path
    _masked_row_sum = None  # type: ignore[assignment]
    folded_offdiag_dispersive_constitutive_step = None  # type: ignore[assignment]


def folded_offdiag_dispersive_constitutive_step_kernel() -> Any:
    """Return the compiled kernel, or a diagnosable optional-dependency error."""
    if folded_offdiag_dispersive_constitutive_step is None:
        raise ImportError(
            "the folded off-diagonal dispersive update_E kernel needs the "
            "optional `triton` package. Original error: "
            f"{_TRITON_IMPORT_ERROR}")
    return folded_offdiag_dispersive_constitutive_step


def folded_offdiag_dispersive_constitutive_coverage(
        fields: Any, pml: Any) -> _coverage.Coverage:
    """Whether the exact fold + offdiag + pole ``update_E`` product is covered."""
    grid = getattr(fields, "grid", None)
    if grid is None:
        return _coverage.Coverage(False, ("fields carries no grid",))

    reasons: List[str] = list(
        _folded._folded_offdiag_grid_reasons(fields, pml, grid))
    reasons.extend(_coverage._susceptibility_reasons(fields))

    if not _symmetry._has_real_fold(grid):
        reasons.append(
            "no mirror plane is active: this family is the folded intersection, "
            "not an ordering-dependent replacement for an unfolded product")

    states = tuple(getattr(fields, "polarizations", ()) or ())
    if not states:
        reasons.append(
            "no susceptibility is registered: the folded off-diagonal "
            "non-dispersive family owns that product")

    if not any(value is not None for value in _offdiag.row_volumes_for(fields)):
        reasons.append(
            "no off-diagonal chi1inv row survived installation: the folded "
            "dispersive diagonal family owns that product")

    if not getattr(fields, "stores_E", False):
        reasons.append("E is recomputed from D rather than stored")

    shape = tuple(getattr(grid, "shape", ()))
    reasons.extend(_offdiag._row_reasons(fields, shape))

    counts: Dict[str, int] = {name: 0 for name, _, _ in E_TERMS}
    for index, state in enumerate(states):
        driven = _coverage._call(state, "driven", default=None)
        if driven is None:
            continue  # _susceptibility_reasons reports it.
        for name in tuple(driven):
            if name in counts:
                counts[name] += 1
            for slot in ("P", "P_prev"):
                array = (getattr(state, slot, {}) or {}).get(name)
                if array is None:
                    reasons.append(
                        f"polarization {index} {slot}[{name}] is not allocated")
                elif len(shape) == 3:
                    reasons.extend(_coverage._volume_reasons(
                        f"polarization {index} {slot}[{name}]", array, shape))
    for name, count in counts.items():
        if count > MAX_POLES:
            reasons.append(
                f"{name} is driven by {count} poles, more than the kernel's "
                f"MAX_POLES={MAX_POLES} compiled slots")

    names = tuple(term[0] for term in E_TERMS)
    names += tuple("f_w_" + term[0] for term in E_TERMS)
    names += tuple(term[1] for term in E_TERMS)
    for name in names:
        if getattr(fields, name, None) is None:
            reasons.append(f"{name} is not allocated")
    reasons.extend(_coverage._layout_reasons(fields, shape, names))
    if len(shape) == 3:
        reasons.extend(_coverage._inverse_epsilon_reasons(fields, shape))
        if pml is not None and getattr(pml, "is_active", False):
            reasons.extend(_coverage._coefficient_reasons(
                pml, shape, ("kps", "kms"), ("_h",)))

    for axis, weight in enumerate(_folded.mirror_ghost_weights(grid)):
        if weight != weight or weight not in (1.0, -1.0):
            reasons.append(
                f"axis {axis} mirror ghost weight is {weight!r}, not +1.0 or "
                "-1.0")

    return _coverage.Coverage(not reasons, tuple(reasons))


class FoldedOffdiagDispersiveConstitutivePlan:
    """Launch binding with live pole-pointer resolution on every call."""

    __slots__ = ("_base", "_poles", "_pointer", "counts", "_output_addresses",
                 "_kernel")

    def __init__(self, shape, block: int, targets, auxiliaries, sources,
                 inverse_epsilon, poles, rows, coefficients, boundary_codes,
                 wall_axes, ghost_weights, kernel: Any = None) -> None:
        from .launch import CupyPointer  # noqa: PLC0415

        targets = tuple(targets)
        auxiliaries = tuple(auxiliaries)
        self._base = _folded.FoldedOffdiagConstitutivePlan(
            shape, block, targets, auxiliaries, sources, inverse_epsilon, rows,
            coefficients, boundary_codes, wall_axes, ghost_weights)
        self._poles = poles
        self.counts = tuple(int(count) for count in poles.counts)
        if len(self.counts) != 3:
            raise ValueError("a pole binding carries one count per E component")
        for term, count in zip(E_TERMS, self.counts):
            if count > MAX_POLES:
                raise ValueError(
                    f"{term[0]} is driven by {count} poles; MAX_POLES={MAX_POLES}")
        self._pointer = CupyPointer
        self._output_addresses = frozenset(
            address for array in targets + auxiliaries
            if (address := _offdiag._base_address(array)) is not None)
        self._kernel = kernel

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self._base.shape

    @property
    def n_elem(self) -> int:
        return self._base.n_elem

    @property
    def block(self) -> int:
        return self._base.block

    @property
    def launch_grid(self) -> Tuple[int]:
        return self._base.launch_grid

    @property
    def row_mask(self) -> Tuple[int, ...]:
        return self._base.row_mask

    def _live_pole_slots(self) -> List[Any]:
        groups = self._poles.arrays()
        slots: List[Any] = []
        for index, group in enumerate(groups):
            if len(group) != self.counts[index]:
                raise RuntimeError(
                    f"component {index} resolved {len(group)} poles, not the "
                    f"compiled {self.counts[index]}")
            for array in group:
                address = _offdiag._base_address(array)
                if address is not None and address in self._output_addresses:
                    raise ValueError(
                        "a live P volume aliases an E/f_w output; neighbour "
                        "gathers make the result launch-order dependent")
                slots.append(self._pointer(array))
            slots.extend([self._base._sources[index]] * (MAX_POLES - len(group)))
        return slots

    def run(self, guard: Optional[bool] = None) -> None:
        from .kernels import ENABLE_FP_FUSION  # noqa: PLC0415

        kernel = (self._kernel if self._kernel is not None
                  else folded_offdiag_dispersive_constitutive_step_kernel())
        base = self._base
        nx, ny, nz = base.shape
        slots = self._live_pole_slots()
        kernel[base.launch_grid](
            *base._targets, *base._aux, *base._sources, *base._inv_eps,
            *slots, *base._rows, *base._coefficients, *base.ghost_weights,
            nx, ny, nz, base.n_elem,
            NP0=self.counts[0], NP1=self.counts[1], NP2=self.counts[2],
            R01=base.row_mask[0], R02=base.row_mask[1],
            R11=base.row_mask[2], R12=base.row_mask[3],
            R21=base.row_mask[4], R22=base.row_mask[5],
            BCX=base.boundary_codes[0], BCY=base.boundary_codes[1],
            BCZ=base.boundary_codes[2],
            MG_X=base.ghost_axes[0], MG_Y=base.ghost_axes[1],
            MG_Z=base.ghost_axes[2],
            WM_X=base.wall_axes[0], WM_Y=base.wall_axes[1],
            WM_Z=base.wall_axes[2], BLOCK=base.block,
            enable_fp_fusion=ENABLE_FP_FUSION if guard is None else bool(guard),
        )

    def __repr__(self) -> str:
        return ("FoldedOffdiagDispersiveConstitutivePlan("
                f"shape={self.shape}, rows={self.row_mask}, "
                f"poles={self.counts}, block={self.block})")


def plan_folded_offdiag_dispersive_constitutive(
        fields: Any, pml: Any, block: Optional[int] = None
        ) -> Optional[FoldedOffdiagDispersiveConstitutivePlan]:
    """Build from live engine objects, returning ``None`` on every refusal."""
    if not folded_offdiag_dispersive_constitutive_coverage(fields, pml).covered:
        return None
    codes, _reasons = _symmetry.folded_axis_kinds(fields.grid, pml)
    if codes is None:  # pragma: no cover - predicate already refuses
        return None
    targets = tuple(term[0] for term in E_TERMS)
    order = poles_per_component(fields)
    return FoldedOffdiagDispersiveConstitutivePlan(
        fields.grid.shape, DEFAULT_BLOCK if block is None else block,
        [getattr(fields, name) for name in targets],
        [getattr(fields, "f_w_" + name) for name in targets],
        [getattr(fields, term[1]) for term in E_TERMS],
        [fields.inverse_epsilon_for(name) for name in targets],
        LivePoleBinding(fields, order),
        _offdiag.row_volumes_for(fields),
        [getattr(pml, f"{stem}_{axis}_h")
         for axis in "xyz" for stem in ("kps", "kms")],
        codes, _offdiag.wall_mask_axes(fields.grid),
        _folded.mirror_ghost_weights(fields.grid),
    )


def plan_folded_offdiag_dispersive_constitutive_from_arrays(
        arrays: Dict[str, Any], flat: Dict[str, Any],
        poles: Dict[str, Sequence[Any]], rows: Dict[str, Dict[str, Any]],
        boundary_codes: Sequence[int], wall_axes: Sequence[int],
        ghost_weights: Sequence[float], block: Optional[int] = None,
        kernel: Any = None) -> FoldedOffdiagDispersiveConstitutivePlan:
    """Bare-array builder for the CUDA byte gate and mutation controls."""
    targets = tuple(term[0] for term in E_TERMS)
    shape = tuple(int(n) for n in arrays[targets[0]].shape)
    return FoldedOffdiagDispersiveConstitutivePlan(
        shape, DEFAULT_BLOCK if block is None else block,
        [arrays[name] for name in targets],
        [arrays["f_w_" + name] for name in targets],
        [arrays[term[1]] for term in E_TERMS],
        [arrays["inv_eps_" + name] for name in targets],
        StaticPoleBinding([poles[name] for name in targets]),
        [(rows.get(row) or {}).get(partner) for row, partner in ROW_SLOTS],
        [flat[f"{stem}_{axis}"] for axis in "xyz" for stem in ("kps", "kms")],
        boundary_codes, wall_axes, ghost_weights, kernel=kernel,
    )


def explain_folded_offdiag_dispersive(fields: Any, pml: Any) -> _coverage.Coverage:
    """Return the fail-closed verdict without importing Triton on the caller."""
    return folded_offdiag_dispersive_constitutive_coverage(fields, pml)


# CENTRAL INTEGRATION SPEC (shared files deliberately untouched here):
#
# 1. Add this module to ``launch.FAMILY_MODULES``.
# 2. Add lazy forwarders for
#    ``folded_offdiag_dispersive_constitutive_coverage`` and
#    ``plan_folded_offdiag_dispersive_constitutive``.
# 3. In ``plan_step``'s ``update_E`` candidates, place an arm labelled
#    ``"folded off-diagonal dispersive"`` before the separate folded-offdiag
#    and folded-dispersive arms.  Its predicate is already disjoint because it
#    requires all three features and an actual fold.
# 4. Add the label to ``ROW_PRODUCT_ARMS`` so
#    ``_veto_dropped_offdiagonal_coupling`` recognizes that this body carries
#    the tensor row instead of silently vetoing it.
# 5. Export the predicate, builder and plan from ``triton_kernels.__init__``.
# 6. Do not change the incumbent predicates: their separate refusals are the
#    disjointness seams that route this exact intersection here.
