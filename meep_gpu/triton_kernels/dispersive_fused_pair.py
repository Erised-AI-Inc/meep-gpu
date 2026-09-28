"""Fused dispersive electric pair: ``step_D`` through pole-aware ``update_E``.

This specialization closes exactly one dependency-local seam.  It combines the
already-gated integer-lattice D curl, ``zero_metal_D``, ordered ``D - P``
subtractions, and half-lattice electric constitutive update.  ``update_P`` remains
the separately gated ADE sub-step and consumes the ``f_w_E`` written here.

Electric source injection occurs inside this seam and is CARRIED, not refused:
``launch._install_fused_pair`` brackets this product's launch with the deposit repair
(see :data:`CARRIES_DEPOSIT_REPAIR` below).  Magnetic source injection occurs earlier,
in the B/H seam, and does not disqualify this pair either way.  The source inventory
must be declared; ignorance is never treated as an empty set.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from . import coverage as _coverage
from .dispersive_update_e import (
    E_TERMS,
    MAX_POLES,
    LivePoleBinding,
    StaticPoleBinding,
    dispersive_constitutive_coverage,
    poles_per_component,
)
from .. import deposit_repair as _deposit_repair

#: Does this product bracket its fused launch with the deposit repair? YES, and the
#: wiring this claims is ``launch._install_fused_pair`` (``launch.py:1639``), which
#: installs THIS product at ``launch.py:2988``: a ``LeadingRepairPlan`` in the
#: ``step_D`` slot saves ``Ex/Ey/Ez`` and ``f_w_E*`` at every deposit point immediately
#: before the launch, and a ``TrailingRepairPlan`` in the ``update_E`` slot recomputes
#: them once the driver has injected, filled symmetry and cleared walls.
#:
#: THE PRECONDITION THIS PRODUCT MEETS is that its pole-aware constitutive half is
#: POINTWISE. ``E = (D - sum_n P_n) * inv_eps`` reads D, every driving P and the inverse
#: epsilon at the SAME cell (``stepping.py:983-986``, ``fields.py:1079-1105``); the
#: subtraction order the kernel fuses is the array path's own, which is what the
#: certified byte gate holds. ``deposit_repair.apply`` recomputes exactly that
#: expression through ``Fields.displacement_minus_polarization``, and it is entitled to:
#: ``update_P`` runs AFTER ``update_E`` (``driver.py:3313``), so the polarization arrays
#: it reads at the repair are the ones the launch read, unchanged.
#:
#: IT IS NOT A HINT. A product that passes True without those wrappers computes the
#: constitutive half against a pre-injection field and reports success, which is the
#: exact failure ``deposit_repair`` exists to prevent. Flag and wiring move together.
CARRIES_DEPOSIT_REPAIR = True

try:  # pragma: no cover - CUDA host only
    import triton
    import triton.language as tl
    _TRITON_IMPORT_ERROR: Optional[BaseException] = None
except ImportError as _exc:  # pragma: no cover - laptop path
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _TRITON_IMPORT_ERROR = _exc


if triton is not None:
    METALLIC = tl.constexpr(1)

    @triton.jit
    def fused_curl_dispersive_E(
        f0, f1, f2,
        u0, u1, u2,
        g0, g1, g2,
        kmx, sinvx, kmy, sinvy, kmz, sinvz,
        e0, e1, e2,
        w0, w1, w2,
        ie0, ie1, ie2,
        a0, a1, a2, a3, a4, a5, a6, a7,
        b0, b1, b2, b3, b4, b5, b6, b7,
        c0, c1, c2, c3, c4, c5, c6, c7,
        kp0, km0, kp1, km1, kp2, km2,
        nx, ny, nz, n_elem, dtdx,
        NP0: tl.constexpr, NP1: tl.constexpr, NP2: tl.constexpr,
        BACKWARD: tl.constexpr,
        BCX: tl.constexpr, BCY: tl.constexpr, BCZ: tl.constexpr,
        ZM_X: tl.constexpr, ZM_Y: tl.constexpr, ZM_Z: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """D curl plus ordered dispersive E update, with ADE left separate."""
        idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        live = idx < n_elem
        nyz = ny * nz
        k = idx % nz
        plane = idx // nz
        j = plane % ny
        i = plane // ny

        # D curl: verbatim from fused_curl_constitutive_D.
        if BACKWARD:
            si, sj, sk = i - 1, j - 1, k - 1
        else:
            si, sj, sk = i + 1, j + 1, k + 1
        vx, vy, vz = live, live, live
        if BCX == METALLIC:
            vx = live & (si >= 0) & (si < nx)
        else:
            si = tl.where(si < 0, nx - 1, tl.where(si == nx, 0, si))
        if BCY == METALLIC:
            vy = live & (sj >= 0) & (sj < ny)
        else:
            sj = tl.where(sj < 0, ny - 1, tl.where(sj == ny, 0, sj))
        if BCZ == METALLIC:
            vz = live & (sk >= 0) & (sk < nz)
        else:
            sk = tl.where(sk < 0, nz - 1, tl.where(sk == nz, 0, sk))

        ox = si * nyz + j * nz + k
        oy = i * nyz + sj * nz + k
        oz = i * nyz + j * nz + sk

        a = tl.load(g0 + idx, mask=live, other=0.0)
        b = tl.load(g1 + idx, mask=live, other=0.0)
        c = tl.load(g2 + idx, mask=live, other=0.0)
        a_y = tl.load(g0 + oy, mask=vy, other=0.0)
        a_z = tl.load(g0 + oz, mask=vz, other=0.0)
        b_x = tl.load(g1 + ox, mask=vx, other=0.0)
        b_z = tl.load(g1 + oz, mask=vz, other=0.0)
        c_x = tl.load(g2 + ox, mask=vx, other=0.0)
        c_y = tl.load(g2 + oy, mask=vy, other=0.0)

        curl0 = dtdx * ((c_y - c) + (b - b_z))
        curl1 = dtdx * ((a_z - a) + (c - c_x))
        curl2 = dtdx * ((b_x - b) + (a - a_y))

        at_x, at_y, at_z = i == 0, j == 0, k == 0
        if BACKWARD:
            if BCY == METALLIC:
                curl0 = tl.where(at_y, 0.0, curl0)
            if BCZ == METALLIC:
                curl0 = tl.where(at_z, 0.0, curl0)
            if BCX == METALLIC:
                curl1 = tl.where(at_x, 0.0, curl1)
            if BCZ == METALLIC:
                curl1 = tl.where(at_z, 0.0, curl1)
            if BCX == METALLIC:
                curl2 = tl.where(at_x, 0.0, curl2)
            if BCY == METALLIC:
                curl2 = tl.where(at_y, 0.0, curl2)
        else:
            if BCX == METALLIC:
                curl0 = tl.where(at_x, 0.0, curl0)
            if BCY == METALLIC:
                curl1 = tl.where(at_y, 0.0, curl1)
            if BCZ == METALLIC:
                curl2 = tl.where(at_z, 0.0, curl2)

        km_x = tl.load(kmx + i, mask=live, other=0.0)
        si_x = tl.load(sinvx + i, mask=live, other=0.0)
        km_y = tl.load(kmy + j, mask=live, other=0.0)
        si_y = tl.load(sinvy + j, mask=live, other=0.0)
        km_z = tl.load(kmz + k, mask=live, other=0.0)
        si_z = tl.load(sinvz + k, mask=live, other=0.0)

        p0 = tl.load(u0 + idx, mask=live, other=0.0)
        n0 = ((p0 * km_y) - curl0) * si_y
        v0 = (((tl.load(f0 + idx, mask=live, other=0.0) * km_z) + n0) - p0) * si_z

        p1 = tl.load(u1 + idx, mask=live, other=0.0)
        n1 = ((p1 * km_z) - curl1) * si_z
        v1 = (((tl.load(f1 + idx, mask=live, other=0.0) * km_x) + n1) - p1) * si_x

        p2 = tl.load(u2 + idx, mask=live, other=0.0)
        n2 = ((p2 * km_x) - curl2) * si_x
        v2 = (((tl.load(f2 + idx, mask=live, other=0.0) * km_y) + n2) - p2) * si_y

        if ZM_X:
            v1 = tl.where(at_x, 0.0, v1)
            v2 = tl.where(at_x, 0.0, v2)
        if ZM_Y:
            v0 = tl.where(at_y, 0.0, v0)
            v2 = tl.where(at_y, 0.0, v2)
        if ZM_Z:
            v0 = tl.where(at_z, 0.0, v0)
            v1 = tl.where(at_z, 0.0, v1)

        tl.store(u0 + idx, n0, mask=live)
        tl.store(u1 + idx, n1, mask=live)
        tl.store(u2 + idx, n2, mask=live)
        tl.store(f0 + idx, v0, mask=live)
        tl.store(f1 + idx, v1, mask=live)
        tl.store(f2 + idx, v2, mask=live)

        # Pole-aware E update.  Start from the live D registers and subtract each
        # component's poles left to right in the fields.polarizations order.
        kp_0 = tl.load(kp0 + i, mask=live, other=0.0)
        km_0 = tl.load(km0 + i, mask=live, other=0.0)
        kp_1 = tl.load(kp1 + j, mask=live, other=0.0)
        km_1 = tl.load(km1 + j, mask=live, other=0.0)
        kp_2 = tl.load(kp2 + k, mask=live, other=0.0)
        km_2 = tl.load(km2 + k, mask=live, other=0.0)

        prev0 = tl.load(w0 + idx, mask=live, other=0.0)
        s0 = v0
        if NP0 > 0:
            s0 = s0 - tl.load(a0 + idx, mask=live, other=0.0)
        if NP0 > 1:
            s0 = s0 - tl.load(a1 + idx, mask=live, other=0.0)
        if NP0 > 2:
            s0 = s0 - tl.load(a2 + idx, mask=live, other=0.0)
        if NP0 > 3:
            s0 = s0 - tl.load(a3 + idx, mask=live, other=0.0)
        if NP0 > 4:
            s0 = s0 - tl.load(a4 + idx, mask=live, other=0.0)
        if NP0 > 5:
            s0 = s0 - tl.load(a5 + idx, mask=live, other=0.0)
        if NP0 > 6:
            s0 = s0 - tl.load(a6 + idx, mask=live, other=0.0)
        if NP0 > 7:
            s0 = s0 - tl.load(a7 + idx, mask=live, other=0.0)
        src0 = s0 * tl.load(ie0 + idx, mask=live, other=0.0)
        tl.store(w0 + idx, src0, mask=live)
        a0v = tl.load(e0 + idx, mask=live, other=0.0)
        a0v = a0v + kp_0 * src0
        a0v = a0v - km_0 * prev0
        tl.store(e0 + idx, a0v, mask=live)

        prev1 = tl.load(w1 + idx, mask=live, other=0.0)
        s1 = v1
        if NP1 > 0:
            s1 = s1 - tl.load(b0 + idx, mask=live, other=0.0)
        if NP1 > 1:
            s1 = s1 - tl.load(b1 + idx, mask=live, other=0.0)
        if NP1 > 2:
            s1 = s1 - tl.load(b2 + idx, mask=live, other=0.0)
        if NP1 > 3:
            s1 = s1 - tl.load(b3 + idx, mask=live, other=0.0)
        if NP1 > 4:
            s1 = s1 - tl.load(b4 + idx, mask=live, other=0.0)
        if NP1 > 5:
            s1 = s1 - tl.load(b5 + idx, mask=live, other=0.0)
        if NP1 > 6:
            s1 = s1 - tl.load(b6 + idx, mask=live, other=0.0)
        if NP1 > 7:
            s1 = s1 - tl.load(b7 + idx, mask=live, other=0.0)
        src1 = s1 * tl.load(ie1 + idx, mask=live, other=0.0)
        tl.store(w1 + idx, src1, mask=live)
        a1v = tl.load(e1 + idx, mask=live, other=0.0)
        a1v = a1v + kp_1 * src1
        a1v = a1v - km_1 * prev1
        tl.store(e1 + idx, a1v, mask=live)

        prev2 = tl.load(w2 + idx, mask=live, other=0.0)
        s2 = v2
        if NP2 > 0:
            s2 = s2 - tl.load(c0 + idx, mask=live, other=0.0)
        if NP2 > 1:
            s2 = s2 - tl.load(c1 + idx, mask=live, other=0.0)
        if NP2 > 2:
            s2 = s2 - tl.load(c2 + idx, mask=live, other=0.0)
        if NP2 > 3:
            s2 = s2 - tl.load(c3 + idx, mask=live, other=0.0)
        if NP2 > 4:
            s2 = s2 - tl.load(c4 + idx, mask=live, other=0.0)
        if NP2 > 5:
            s2 = s2 - tl.load(c5 + idx, mask=live, other=0.0)
        if NP2 > 6:
            s2 = s2 - tl.load(c6 + idx, mask=live, other=0.0)
        if NP2 > 7:
            s2 = s2 - tl.load(c7 + idx, mask=live, other=0.0)
        src2 = s2 * tl.load(ie2 + idx, mask=live, other=0.0)
        tl.store(w2 + idx, src2, mask=live)
        a2v = tl.load(e2 + idx, mask=live, other=0.0)
        a2v = a2v + kp_2 * src2
        a2v = a2v - km_2 * prev2
        tl.store(e2 + idx, a2v, mask=live)

else:  # pragma: no cover - laptop path
    fused_curl_dispersive_E = None  # type: ignore[assignment]


def fused_curl_dispersive_E_kernel() -> Any:
    if fused_curl_dispersive_E is None:
        raise ImportError(
            "the fused dispersive D/E kernel needs the optional `triton` package "
            f"(pip install triton). Original error: {_TRITON_IMPORT_ERROR}")
    return fused_curl_dispersive_E


def dispersive_fused_pair_coverage(fields: Any, pml: Any,
                                   sources: Any = None) -> _coverage.Coverage:
    """May the specialized D curl plus pole-aware E update span this seam?"""
    reasons: List[str] = []
    curl = _coverage.pml_curl_coverage(fields, pml, "step_D")
    if not curl.covered:
        reasons.extend(f"curl half: {reason}" for reason in curl.reasons)
    constitutive = dispersive_constitutive_coverage(fields, pml)
    if not constitutive.covered:
        reasons.extend(f"dispersive constitutive half: {reason}"
                       for reason in constitutive.reasons)

    reasons.extend(_deposit_repair.seam_source_reasons(
        fields, sources, "D",
        undeclared=(
            "the source set was not declared: this predicate cannot infer an empty "
            "electric source seam from Fields"),
        refusal=lambda index, source: (
            f"source {index} ({type(source).__name__}) is electric: the "
            "driver injects it BETWEEN step_D and update_E"),
        carries_repair=CARRIES_DEPOSIT_REPAIR))

    grid = getattr(fields, "grid", None)
    if grid is None:
        reasons.append("fields carries no grid")
    else:
        for name in ("has_metallic", "is_metallic", "is_mirrored"):
            if getattr(grid, name, None) is None:
                reasons.append(
                    f"grid does not expose {name}; zero_metal_D cannot be carried inline")
    return _coverage.Coverage(not reasons, tuple(reasons))


class DispersiveFusedPairPlan:
    """One allocation-free launch for D curl and pole-aware E constitutive work."""

    __slots__ = (
        "shape", "n_elem", "dtdx", "bc", "zero_metal", "block", "num_warps",
        "counts", "_targets", "_aux", "_sources", "_curl_coefficients",
        "_e_targets", "_e_aux", "_inverse_epsilon", "_e_coefficients", "_poles",
        "_grid", "_kernel", "_pointer",
    )

    def __init__(self, shape, dtdx: float, bc, zero_metal, block: int,
                 targets, auxiliaries, sources, curl_coefficients,
                 e_targets, e_aux, inverse_epsilon, e_coefficients, poles,
                 kernel: Any = None, num_warps: Optional[int] = 1) -> None:
        from .launch import CupyPointer, _flat  # noqa: PLC0415

        self.shape = tuple(int(n) for n in shape)
        self.n_elem = self.shape[0] * self.shape[1] * self.shape[2]
        self.dtdx = float(dtdx)
        self.bc = tuple(int(value) for value in bc)
        self.zero_metal = tuple(bool(value) for value in zero_metal)
        self.block = int(block)
        self.num_warps = num_warps
        self._pointer = CupyPointer
        self._targets = tuple(CupyPointer(array) for array in targets)
        self._aux = tuple(CupyPointer(array) for array in auxiliaries)
        self._sources = tuple(CupyPointer(array) for array in sources)
        self._curl_coefficients = tuple(
            CupyPointer(_flat(array)) for array in curl_coefficients)
        self._e_targets = tuple(CupyPointer(array) for array in e_targets)
        self._e_aux = tuple(CupyPointer(array) for array in e_aux)
        self._inverse_epsilon = tuple(
            CupyPointer(array) for array in inverse_epsilon)
        self._e_coefficients = tuple(
            CupyPointer(_flat(array)) for array in e_coefficients)
        self._poles = poles
        self.counts = tuple(int(value) for value in poles.counts)
        if any(value > MAX_POLES for value in self.counts):
            raise ValueError(
                f"pole counts {self.counts} exceed MAX_POLES={MAX_POLES}")
        self._grid = ((self.n_elem + self.block - 1) // self.block,)
        self._kernel = kernel

    def run(self, guard: Optional[bool] = None) -> None:
        from .kernels import ENABLE_FP_FUSION  # noqa: PLC0415

        kernel = (self._kernel if self._kernel is not None
                  else fused_curl_dispersive_E_kernel())
        groups = self._poles.arrays()
        slots: List[Any] = []
        for index, group in enumerate(groups):
            if len(group) != self.counts[index]:
                raise RuntimeError(
                    f"component {index} resolved {len(group)} poles, not "
                    f"the compiled count {self.counts[index]}")
            slots.extend(self._pointer(array) for array in group)
            slots.extend([self._targets[index]] * (MAX_POLES - len(group)))
        nx, ny, nz = self.shape
        extra = {} if self.num_warps is None else {"num_warps": self.num_warps}
        kernel[self._grid](
            *self._targets, *self._aux, *self._sources, *self._curl_coefficients,
            *self._e_targets, *self._e_aux, *self._inverse_epsilon,
            *slots, *self._e_coefficients,
            nx, ny, nz, self.n_elem, self.dtdx,
            NP0=self.counts[0], NP1=self.counts[1], NP2=self.counts[2],
            BACKWARD=1,
            BCX=self.bc[0], BCY=self.bc[1], BCZ=self.bc[2],
            ZM_X=self.zero_metal[0], ZM_Y=self.zero_metal[1],
            ZM_Z=self.zero_metal[2], BLOCK=self.block,
            enable_fp_fusion=ENABLE_FP_FUSION if guard is None else bool(guard),
            **extra,
        )

    def __repr__(self) -> str:
        return (f"DispersiveFusedPairPlan(shape={self.shape}, poles={self.counts}, "
                f"bc={self.bc}, block={self.block}, num_warps={self.num_warps})")


def plan_dispersive_fused_pair(fields: Any, pml: Any, sources: Any = None,
                               block: Optional[int] = None,
                               num_warps: Optional[int] = 1,
                               ) -> Optional[DispersiveFusedPairPlan]:
    """Build the specialized live-engine plan, or ``None`` when refused."""
    if not dispersive_fused_pair_coverage(fields, pml, sources).covered:
        return None
    from ..stepping import _boundary_kinds  # noqa: PLC0415
    from .kernels import DEFAULT_BLOCK  # noqa: PLC0415

    grid = fields.grid
    order = poles_per_component(fields)
    return DispersiveFusedPairPlan(
        grid.shape, grid.dt / grid.dx,
        [1 if kind == "metallic" else 0 for kind in _boundary_kinds(grid, pml)],
        _coverage.zero_metal_axes(grid),
        DEFAULT_BLOCK if block is None else block,
        [getattr(fields, name) for name in ("Dx", "Dy", "Dz")],
        [getattr(fields, name) for name in ("fu_Dx", "fu_Dy", "fu_Dz")],
        [getattr(fields, name) for name in ("Hx", "Hy", "Hz")],
        [getattr(pml, f"{stem}_{axis}")
         for axis in "xyz" for stem in ("kms", "sinv")],
        [getattr(fields, name) for name in ("Ex", "Ey", "Ez")],
        [getattr(fields, name) for name in ("f_w_Ex", "f_w_Ey", "f_w_Ez")],
        [fields.inverse_epsilon_for(name) for name in ("Ex", "Ey", "Ez")],
        [getattr(pml, f"{stem}_{axis}_h")
         for axis in "xyz" for stem in ("kps", "kms")],
        LivePoleBinding(fields, order), num_warps=num_warps,
    )


def plan_dispersive_fused_pair_from_arrays(
        arrays: Dict[str, Any], curl_flat: Dict[str, Any],
        constitutive_flat: Dict[str, Any], poles: Dict[str, Sequence[Any]],
        codes, zero_metal, dtdx: float, block: Optional[int] = None,
        kernel: Any = None, num_warps: Optional[int] = 1,
        ) -> DispersiveFusedPairPlan:
    """Build the bare-array gate route with caller-selected coefficients."""
    from .kernels import DEFAULT_BLOCK  # noqa: PLC0415

    shape = tuple(int(n) for n in arrays["Dx"].shape)
    return DispersiveFusedPairPlan(
        shape, dtdx, codes, zero_metal,
        DEFAULT_BLOCK if block is None else block,
        [arrays[name] for name in ("Dx", "Dy", "Dz")],
        [arrays[name] for name in ("fu_Dx", "fu_Dy", "fu_Dz")],
        [arrays[name] for name in ("Hx", "Hy", "Hz")],
        [curl_flat[f"{stem}_{axis}"]
         for axis in "xyz" for stem in ("kms", "sinv")],
        [arrays[name] for name in ("Ex", "Ey", "Ez")],
        [arrays[name] for name in ("f_w_Ex", "f_w_Ey", "f_w_Ez")],
        [arrays["inv_eps_" + name] for name in ("Ex", "Ey", "Ez")],
        [constitutive_flat[f"{stem}_{axis}"]
         for axis in "xyz" for stem in ("kps", "kms")],
        StaticPoleBinding([poles[name] for name in ("Ex", "Ey", "Ez")]),
        kernel=kernel, num_warps=num_warps,
    )
