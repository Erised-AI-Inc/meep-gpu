"""Complex64/Bloch conductive curls with no active PML layer.

This is the curl product required by residual Group I: complex64 storage, a
Bloch phase, material conductivity, registered electric Lorentz/Drude poles and
no active absorber.  The poles are deliberately absent from the kernel
signature.  ``stepping.step_B`` and ``step_D`` never read a polarization; they
only difference the stored E/H values written by the surrounding sub-steps.

The body is the established complex Yee stencil and Bloch-wrap transcription
from :mod:`complex_fields`, followed by the fourth no-PML branch of
``stepping._apply_curl``::

    field *= condfac
    field -= curl
    field *= condinv

Conductivity is selected per TARGET, exactly where ``_apply_curl`` reads it.
Admission is intentionally wider: a conductivity on any of Bx..Dz assigns both
curl slots to this family, because the incumbent complex no-PML predicate
refuses both slots on that same run-wide question.  Consequently a D-only sigma
selects this family for ``step_B`` while all three ``COND`` flags remain false;
the D sigma cannot leak into B arithmetic.

The full complex multiplication with a zero-imaginary real coefficient is not
replaceable by plane-wise scaling: its signed-zero words and its FMA expansion
are platform facts.  Both conductive passes therefore call
``complex_fields._mul_field_left`` and use the same measured EXPANSION licence
as the complex curl and PML products.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from .complex_fields import (
    DEFAULT_BLOCK,
    METALLIC,
    PROBE_PATTERNS,
    _complex_grid_reasons,
    _complex_layout_reasons,
    _complex_volume_reasons,
    _mul_coefficient_left,
    _mul_field_left,
    _phase_arguments,
    _resolve_expansion,
    _rotate_field_left,
    _word_view,
    bloch_phase_table,
    tl,
    triton,
)
from .coverage import CURL_SUB_STEPS, Coverage, _susceptibility_reasons, _volume_reasons
from .launch import SUB_STEPS, CupyPointer, _flat
from .no_pml_conductive import _any_curl_conductivity, conductive_no_pml_targets
from .complex_no_pml_curl import SOURCE_ACCESSOR, source_arrays

__all__ = [
    "COMPLEX_CONDUCTIVE_PROBE_PATTERNS",
    "ComplexConductiveNoPmlCurlPlan",
    "complex_conductive_no_pml_curl_coverage",
    "complex_conductive_no_pml_curl_step",
    "plan_complex_conductive_no_pml_curl",
    "plan_complex_conductive_no_pml_curl_from_arrays",
]

COMPLEX_CONDUCTIVE_PROBE_PATTERNS: Tuple[str, ...] = PROBE_PATTERNS


class _CurlScopeView:
    """Delegate to Fields while masking facts irrelevant to curl arithmetic.

    ``_complex_grid_reasons`` is the single home for the complex storage, phase,
    boundary, no-PML, nonlinearity, beta, BFAST, stored-E and expansion clauses.
    Its original family also refuses conductivity and dispersion because its
    *whole* product did not carry them.  This view masks only those two facts so
    the shared clauses can be reused verbatim.  The real fields are then checked
    positively below: conductivity is required, and each susceptibility must be
    an electric Lorentz/Drude state.
    """

    __slots__ = ("_fields",)

    def __init__(self, fields: Any) -> None:
        object.__setattr__(self, "_fields", fields)

    def __getattr__(self, name: str) -> Any:
        if name == "condfac_for":
            return lambda _target: None
        if name == "has_magnetic_conductivity":
            return False
        if name == "polarizations":
            return ()
        if name == "has_polarizations":
            return False
        return getattr(object.__getattribute__(self, "_fields"), name)


@triton.jit
def complex_conductive_no_pml_curl_step(
    f0, f1, f2,                       # Bx,By,Bz or Dx,Dy,Dz, complex words
    g0, g1, g2,                       # stored Ex,Ey,Ez or Bx,By,Bz, complex words
    cf0, cf1, cf2,                    # condfac, float32 volumes or dead placeholders
    ci0, ci1, ci2,                    # condinv, float32 volumes or dead placeholders
    nx, ny, nz, n_elem, dtdx,
    pxr, pxi, pyr, pyi, pzr, pzi,
    BACKWARD: tl.constexpr,
    BCX: tl.constexpr, BCY: tl.constexpr, BCZ: tl.constexpr,
    PHX: tl.constexpr, PHY: tl.constexpr, PHZ: tl.constexpr,
    COND0: tl.constexpr, COND1: tl.constexpr, COND2: tl.constexpr,
    EXPANSION: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One complex/Bloch curl plus the per-target no-PML conductive tail.

    Everything through the ownership mask is line-for-line the shared
    ``bloch_pml_curl_step`` body.  Only its split-field PML tail is replaced by
    ``_apply_conductive_update``'s three sequential passes.  Do not flatten the
    passes: each in-place complex64 operation rounds before the next begins.
    """
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = idx < n_elem
    nyz = ny * nz
    k = idx % nz
    plane = idx // nz
    j = plane % ny
    i = plane // ny

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

    if BACKWARD:
        wx, wy, wz = i == 0, j == 0, k == 0
    else:
        wx, wy, wz = i == nx - 1, j == ny - 1, k == nz - 1

    a_re = tl.load(g0 + 2 * idx, mask=live, other=0.0)
    a_im = tl.load(g0 + 2 * idx + 1, mask=live, other=0.0)
    b_re = tl.load(g1 + 2 * idx, mask=live, other=0.0)
    b_im = tl.load(g1 + 2 * idx + 1, mask=live, other=0.0)
    c_re = tl.load(g2 + 2 * idx, mask=live, other=0.0)
    c_im = tl.load(g2 + 2 * idx + 1, mask=live, other=0.0)
    a_y_re = tl.load(g0 + 2 * oy, mask=vy, other=0.0)
    a_y_im = tl.load(g0 + 2 * oy + 1, mask=vy, other=0.0)
    a_z_re = tl.load(g0 + 2 * oz, mask=vz, other=0.0)
    a_z_im = tl.load(g0 + 2 * oz + 1, mask=vz, other=0.0)
    b_x_re = tl.load(g1 + 2 * ox, mask=vx, other=0.0)
    b_x_im = tl.load(g1 + 2 * ox + 1, mask=vx, other=0.0)
    b_z_re = tl.load(g1 + 2 * oz, mask=vz, other=0.0)
    b_z_im = tl.load(g1 + 2 * oz + 1, mask=vz, other=0.0)
    c_x_re = tl.load(g2 + 2 * ox, mask=vx, other=0.0)
    c_x_im = tl.load(g2 + 2 * ox + 1, mask=vx, other=0.0)
    c_y_re = tl.load(g2 + 2 * oy, mask=vy, other=0.0)
    c_y_im = tl.load(g2 + 2 * oy + 1, mask=vy, other=0.0)

    if PHX:
        rot_re, rot_im = _rotate_field_left(b_x_re, b_x_im, pxr, pxi, EXPANSION)
        b_x_re = tl.where(wx, rot_re, b_x_re)
        b_x_im = tl.where(wx, rot_im, b_x_im)
        rot_re, rot_im = _rotate_field_left(c_x_re, c_x_im, pxr, pxi, EXPANSION)
        c_x_re = tl.where(wx, rot_re, c_x_re)
        c_x_im = tl.where(wx, rot_im, c_x_im)
    if PHY:
        rot_re, rot_im = _rotate_field_left(a_y_re, a_y_im, pyr, pyi, EXPANSION)
        a_y_re = tl.where(wy, rot_re, a_y_re)
        a_y_im = tl.where(wy, rot_im, a_y_im)
        rot_re, rot_im = _rotate_field_left(c_y_re, c_y_im, pyr, pyi, EXPANSION)
        c_y_re = tl.where(wy, rot_re, c_y_re)
        c_y_im = tl.where(wy, rot_im, c_y_im)
    if PHZ:
        rot_re, rot_im = _rotate_field_left(a_z_re, a_z_im, pzr, pzi, EXPANSION)
        a_z_re = tl.where(wz, rot_re, a_z_re)
        a_z_im = tl.where(wz, rot_im, a_z_im)
        rot_re, rot_im = _rotate_field_left(b_z_re, b_z_im, pzr, pzi, EXPANSION)
        b_z_re = tl.where(wz, rot_re, b_z_re)
        b_z_im = tl.where(wz, rot_im, b_z_im)

    # stepping._curl_from_operands, including its load-bearing parentheses.
    t0_re = ((c_y_re - c_re) + (b_re - b_z_re))
    t0_im = ((c_y_im - c_im) + (b_im - b_z_im))
    t1_re = ((a_z_re - a_re) + (c_re - c_x_re))
    t1_im = ((a_z_im - a_im) + (c_im - c_x_im))
    t2_re = ((b_x_re - b_re) + (a_re - a_y_re))
    t2_im = ((b_x_im - b_im) + (a_im - a_y_im))
    curl0_re, curl0_im = _mul_coefficient_left(dtdx, t0_re, t0_im, EXPANSION)
    curl1_re, curl1_im = _mul_coefficient_left(dtdx, t1_re, t1_im, EXPANSION)
    curl2_re, curl2_im = _mul_coefficient_left(dtdx, t2_re, t2_im, EXPANSION)

    # stepping._mask_non_owned_cells.  A complex zero assigns +0 to both words.
    at_x, at_y, at_z = i == 0, j == 0, k == 0
    if BACKWARD:
        if BCY == METALLIC:
            curl0_re = tl.where(at_y, 0.0, curl0_re)
            curl0_im = tl.where(at_y, 0.0, curl0_im)
        if BCZ == METALLIC:
            curl0_re = tl.where(at_z, 0.0, curl0_re)
            curl0_im = tl.where(at_z, 0.0, curl0_im)
        if BCX == METALLIC:
            curl1_re = tl.where(at_x, 0.0, curl1_re)
            curl1_im = tl.where(at_x, 0.0, curl1_im)
        if BCZ == METALLIC:
            curl1_re = tl.where(at_z, 0.0, curl1_re)
            curl1_im = tl.where(at_z, 0.0, curl1_im)
        if BCX == METALLIC:
            curl2_re = tl.where(at_x, 0.0, curl2_re)
            curl2_im = tl.where(at_x, 0.0, curl2_im)
        if BCY == METALLIC:
            curl2_re = tl.where(at_y, 0.0, curl2_re)
            curl2_im = tl.where(at_y, 0.0, curl2_im)
    else:
        if BCX == METALLIC:
            curl0_re = tl.where(at_x, 0.0, curl0_re)
            curl0_im = tl.where(at_x, 0.0, curl0_im)
        if BCY == METALLIC:
            curl1_re = tl.where(at_y, 0.0, curl1_re)
            curl1_im = tl.where(at_y, 0.0, curl1_im)
        if BCZ == METALLIC:
            curl2_re = tl.where(at_z, 0.0, curl2_re)
            curl2_im = tl.where(at_z, 0.0, curl2_im)

    # _apply_conductive_update: three separate complex64 in-place passes.
    v0_re = tl.load(f0 + 2 * idx, mask=live, other=0.0)
    v0_im = tl.load(f0 + 2 * idx + 1, mask=live, other=0.0)
    if COND0:
        factor = tl.load(cf0 + idx, mask=live, other=0.0)
        v0_re, v0_im = _mul_field_left(v0_re, v0_im, factor, EXPANSION)
    v0_re = v0_re - curl0_re
    v0_im = v0_im - curl0_im
    if COND0:
        inverse = tl.load(ci0 + idx, mask=live, other=0.0)
        v0_re, v0_im = _mul_field_left(v0_re, v0_im, inverse, EXPANSION)

    v1_re = tl.load(f1 + 2 * idx, mask=live, other=0.0)
    v1_im = tl.load(f1 + 2 * idx + 1, mask=live, other=0.0)
    if COND1:
        factor = tl.load(cf1 + idx, mask=live, other=0.0)
        v1_re, v1_im = _mul_field_left(v1_re, v1_im, factor, EXPANSION)
    v1_re = v1_re - curl1_re
    v1_im = v1_im - curl1_im
    if COND1:
        inverse = tl.load(ci1 + idx, mask=live, other=0.0)
        v1_re, v1_im = _mul_field_left(v1_re, v1_im, inverse, EXPANSION)

    v2_re = tl.load(f2 + 2 * idx, mask=live, other=0.0)
    v2_im = tl.load(f2 + 2 * idx + 1, mask=live, other=0.0)
    if COND2:
        factor = tl.load(cf2 + idx, mask=live, other=0.0)
        v2_re, v2_im = _mul_field_left(v2_re, v2_im, factor, EXPANSION)
    v2_re = v2_re - curl2_re
    v2_im = v2_im - curl2_im
    if COND2:
        inverse = tl.load(ci2 + idx, mask=live, other=0.0)
        v2_re, v2_im = _mul_field_left(v2_re, v2_im, inverse, EXPANSION)

    tl.store(f0 + 2 * idx, v0_re, mask=live)
    tl.store(f0 + 2 * idx + 1, v0_im, mask=live)
    tl.store(f1 + 2 * idx, v1_re, mask=live)
    tl.store(f1 + 2 * idx + 1, v1_im, mask=live)
    tl.store(f2 + 2 * idx, v2_re, mask=live)
    tl.store(f2 + 2 * idx + 1, v2_im, mask=live)


def complex_conductive_no_pml_curl_coverage(
    fields: Any, pml: Any, sub_step: str, probe: Any = None,
) -> Coverage:
    """Whether this family may replace one complex conductive curl sub-step."""
    if sub_step not in CURL_SUB_STEPS:
        raise ValueError(
            f"sub_step must be one of {tuple(CURL_SUB_STEPS)}, got {sub_step!r}")
    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False, ("fields carries no grid",))

    # Reuse every complex clause except the two facts this family adds.
    reasons = _complex_grid_reasons(
        _CurlScopeView(fields), pml, grid, probe, require_active_pml=False)
    if not _any_curl_conductivity(fields):
        reasons.append(
            "no curl target carries a conductivity; the lossless complex no-PML "
            "family owns this slot")
    reasons.extend(_susceptibility_reasons(fields))

    spec = SUB_STEPS[sub_step]
    targets = tuple(spec["targets"])
    for name in targets:
        if getattr(fields, name, None) is None:
            reasons.append(f"{name} is not allocated")

    accessor = getattr(fields, SOURCE_ACCESSOR[sub_step], None)
    sources: List[Any] = []
    if not callable(accessor):
        reasons.append(f"fields.{SOURCE_ACCESSOR[sub_step]} is missing or not callable")
    else:
        for name in spec["sources"]:
            try:
                source = accessor(name)
            except Exception as exc:  # noqa: BLE001 - unreadable is not coverage
                reasons.append(
                    f"fields.{SOURCE_ACCESSOR[sub_step]}({name!r}) raised {exc!r}")
                source = None
            if source is None:
                reasons.append(
                    f"{name} is not served by fields.{SOURCE_ACCESSOR[sub_step]}")
            sources.append(source)

    shape = tuple(getattr(grid, "shape", ()))
    reasons.extend(_complex_layout_reasons(fields, shape, targets))
    if len(shape) == 3:
        for name, source in zip(spec["sources"], sources):
            if source is not None:
                reasons.extend(_complex_volume_reasons(name, source, shape))

    condfac_for = getattr(fields, "condfac_for", None)
    condinv_for = getattr(fields, "condinv_for", None)
    if not callable(condfac_for) or not callable(condinv_for):
        reasons.append(
            "fields does not expose callable condfac_for/condinv_for readers")
    else:
        flags = conductive_no_pml_targets(fields, sub_step)
        for index, target in enumerate(targets):
            try:
                condfac = condfac_for(target)
                condinv = condinv_for(target)
            except Exception as exc:  # noqa: BLE001 - unreadable is not coverage
                reasons.append(
                    f"condfac_for/condinv_for({target!r}) raised {type(exc).__name__}")
                continue
            if (condfac is None) != (condinv is None):
                reasons.append(
                    f"{target} has only one of condfac/condinv; both are required")
            elif flags[index] and len(shape) == 3:
                reasons.extend(_volume_reasons(f"condfac[{target}]", condfac, shape))
                reasons.extend(_volume_reasons(f"condinv[{target}]", condinv, shape))

    return Coverage(not reasons, tuple(reasons))


class ComplexConductiveNoPmlCurlPlan:
    """A launchable complex/Bloch no-PML conductive curl plan."""

    __slots__ = (
        "sub_step", "shape", "n_elem", "dtdx", "backward", "bc", "cond",
        "phased", "phase_values", "expansion", "block", "num_warps",
        "_targets", "_sources", "_condfac", "_condinv", "_grid", "_kernel",
    )

    def __init__(
        self, sub_step: str, shape: Sequence[int], dtdx: float,
        codes: Sequence[int], phases: Sequence[Optional[complex]], expansion: int,
        cond: Sequence[bool], targets: Sequence[Any], sources: Sequence[Any],
        condfac: Sequence[Any], condinv: Sequence[Any], *,
        block: int = DEFAULT_BLOCK, kernel: Any = None,
        num_warps: Optional[int] = None,
    ) -> None:
        if sub_step not in SUB_STEPS:
            raise ValueError(f"unknown curl sub-step {sub_step!r}")
        self.sub_step = sub_step
        self.shape = tuple(int(n) for n in shape)
        self.n_elem = self.shape[0] * self.shape[1] * self.shape[2]
        self.dtdx = float(dtdx)
        self.backward = int(SUB_STEPS[sub_step]["backward"])
        self.bc = tuple(int(code) for code in codes)
        self.cond = tuple(int(bool(flag)) for flag in cond)
        if len(self.cond) != 3:
            raise ValueError("a curl plan needs three per-target conductivity flags")
        self.phased, self.phase_values = _phase_arguments(
            tuple(phases), backward=bool(self.backward))
        self.expansion = int(expansion)
        self.block = int(block)
        self.num_warps = None if num_warps is None else int(num_warps)
        target_words = tuple(_word_view(array) for array in targets)
        self._targets = tuple(CupyPointer(array) for array in target_words)
        self._sources = tuple(CupyPointer(_word_view(array)) for array in sources)
        # A dead coefficient slot binds its target's float32 word view.  The
        # pointer is correctly typed and COND=0 compiles every load away.
        self._condfac = tuple(
            CupyPointer(_flat(array)) if array is not None
            else CupyPointer(target_words[index])
            for index, array in enumerate(condfac))
        self._condinv = tuple(
            CupyPointer(_flat(array)) if array is not None
            else CupyPointer(target_words[index])
            for index, array in enumerate(condinv))
        self._grid = ((self.n_elem + self.block - 1) // self.block,)
        self._kernel = kernel

    def run(self, guard: Optional[bool] = None) -> None:
        from .kernels import ENABLE_FP_FUSION  # noqa: PLC0415

        nx, ny, nz = self.shape
        kernel = self._kernel or complex_conductive_no_pml_curl_step
        extra = {} if self.num_warps is None else {"num_warps": self.num_warps}
        kernel[self._grid](
            *self._targets, *self._sources, *self._condfac, *self._condinv,
            nx, ny, nz, self.n_elem, self.dtdx, *self.phase_values,
            BACKWARD=self.backward,
            BCX=self.bc[0], BCY=self.bc[1], BCZ=self.bc[2],
            PHX=self.phased[0], PHY=self.phased[1], PHZ=self.phased[2],
            COND0=self.cond[0], COND1=self.cond[1], COND2=self.cond[2],
            EXPANSION=self.expansion, BLOCK=self.block,
            enable_fp_fusion=ENABLE_FP_FUSION if guard is None else bool(guard),
            **extra,
        )

    def __repr__(self) -> str:
        return (
            f"ComplexConductiveNoPmlCurlPlan({self.sub_step}, shape={self.shape}, "
            f"bc={self.bc}, phased={self.phased}, cond={self.cond}, "
            f"expansion={self.expansion}, block={self.block})")


def plan_complex_conductive_no_pml_curl(
    fields: Any, pml: Any, sub_step: str, block: Optional[int] = None,
    num_warps: Optional[int] = None, probe: Any = None,
) -> Optional[ComplexConductiveNoPmlCurlPlan]:
    """Build from engine objects, returning None for every refusal."""
    verdict = complex_conductive_no_pml_curl_coverage(
        fields, pml, sub_step, probe=probe)
    if not verdict.covered:
        return None
    expansion = _resolve_expansion(probe)
    if expansion is None:  # pragma: no cover - predicate already refused
        return None
    from ..stepping import _boundary_kinds  # noqa: PLC0415

    grid = fields.grid
    spec = SUB_STEPS[sub_step]
    targets = tuple(spec["targets"])
    kinds = _boundary_kinds(grid, None)
    phases = bloch_phase_table(grid, kinds)
    flags = conductive_no_pml_targets(fields, sub_step)
    return ComplexConductiveNoPmlCurlPlan(
        sub_step, grid.shape, grid.dt / grid.dx,
        [1 if kind == "metallic" else 0 for kind in kinds],
        phases, expansion, flags,
        [getattr(fields, name) for name in targets],
        source_arrays(fields, sub_step),
        [fields.condfac_for(name) if flags[index] else None
         for index, name in enumerate(targets)],
        [fields.condinv_for(name) if flags[index] else None
         for index, name in enumerate(targets)],
        block=DEFAULT_BLOCK if block is None else block,
        num_warps=num_warps,
    )


def plan_complex_conductive_no_pml_curl_from_arrays(
    sub_step: str, arrays: Dict[str, Any], phases: Sequence[Optional[complex]],
    dtdx: float, codes: Sequence[int], expansion: int,
    cond: Sequence[bool] = (True, True, True), *,
    block: Optional[int] = None, kernel: Any = None,
    num_warps: Optional[int] = None,
) -> ComplexConductiveNoPmlCurlPlan:
    """Build from deliberate bare arrays for the device gate and mutations."""
    spec = SUB_STEPS[sub_step]
    targets = tuple(spec["targets"])
    flags = tuple(bool(flag) for flag in cond)
    shape = tuple(int(n) for n in arrays[targets[0]].shape)
    return ComplexConductiveNoPmlCurlPlan(
        sub_step, shape, dtdx, codes, phases, expansion, flags,
        [arrays[name] for name in targets],
        [arrays[name] for name in spec["sources"]],
        [arrays[f"condfac_{name}"] if flags[index] else None
         for index, name in enumerate(targets)],
        [arrays[f"condinv_{name}"] if flags[index] else None
         for index, name in enumerate(targets)],
        block=DEFAULT_BLOCK if block is None else block,
        kernel=kernel, num_warps=num_warps,
    )


# The central planner consults this family through lazy launch/package
# forwarders. Driver/fastpath certification remains separate: the family must
# pass its CUDA byte gate and the central seam must be recertified before this
# arm is eligible for an opted-in driver route.
