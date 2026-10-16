"""Complex64/Bloch conductive curls with no active PML layer.

This is the curl product required by residual Group I: complex64 storage, a
Bloch phase, material conductivity, registered electric Lorentz/Drude poles and
no active absorber.  The poles are deliberately absent from the kernel
signature.  ``stepping.step_B`` and ``step_D`` never read a polarization; they
only difference the stored E/H values written by the surrounding sub-steps.

THE BODY. Through the ownership mask it is :func:`.complex_fields.bloch_pml_curl_step`
statement for statement (the test file compares the two parse trees): the ghost rule,
MEEP's Bloch connection of the UNSHIFTED partner on plane 0 (a copy, a per-part
negation or the unfused product, as :func:`.complex_fields.connection_arguments`
classifies each axis), and MEEP's curl stencil per stored part,
``dtdx * (((g1s - g1) + g2) - g2s)`` with the one-partner forms where the chunk
record's presence word marks a partner's derivative axis invariant
(``stepping._curl_from_operands``, ``forms.curl_stencil``). Then the fourth no-PML
branch of ``stepping._apply_curl`` (``_apply_conductive_update``,
step_generic.cpp:87-97), on each part::

    field *= condfac     # part_products.scale
    field -= curl
    field *= condinv     # part_products.scale

THE CHUNK RECORD. The kernel reads only its presence word: a run with no absorber has
no membership run, and the conductivity is a whole-run volume on this branch (the
array path applies it wherever ``condfac_for`` returns one). The plans bind the
record of the run's division (``launch.chunk_bounds_for``), which with no absorber is
the one-rank record's words on every division.

Conductivity is selected per TARGET, exactly where ``_apply_curl`` reads it.
Admission is intentionally wider: a conductivity on any of Bx..Dz assigns both
curl slots to this family, because the incumbent complex no-PML predicate
refuses both slots on that same run-wide question.  Consequently a D-only sigma
selects this family for ``step_B`` while all three ``COND`` flags remain false;
the D sigma cannot leak into B arithmetic.

Every product has a real operand and is formed one float32 product per part
(``complex_fields._mul_field_left``, ``_connect``, ``forms.curl_stencil``), as MEEP's
``DOCMP`` loop forms it; no complex-by-complex product and no multiply-add is
emitted.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from .complex_fields import (
    CONNECTION_KINDS,
    DEFAULT_BLOCK,
    METALLIC,
    PERIODIC,
    _TRITON_IMPORT_ERROR,
    _complex_grid_reasons,
    _complex_layout_reasons,
    _complex_volume_reasons,
    _connect,
    _mul_field_left,
    _word_view,
    bloch_phase_table,
    connection_arguments,
    tl,
    triton,
)
from .coverage import CURL_SUB_STEPS, Coverage, _susceptibility_reasons, _volume_reasons
from .launch import SUB_STEPS, CupyPointer, _flat, chunk_bounds_for
from .no_pml_conductive import _any_curl_conductivity, conductive_no_pml_targets
from .complex_no_pml_curl import SOURCE_ACCESSOR, source_arrays

__all__ = [
    "ComplexConductiveNoPmlCurlPlan",
    "complex_conductive_no_pml_curl_coverage",
    "complex_conductive_no_pml_curl_step",
    "plan_complex_conductive_no_pml_curl",
    "plan_complex_conductive_no_pml_curl_from_arrays",
]

# The curl forms and the record's presence word, read by the kernel below. Imported
# only when Triton is (both modules import it at module scope); the predicates and the
# builders to None answer without it.
if _TRITON_IMPORT_ERROR is None:
    from .forms import axis_present, curl_stencil
    from .kernels import RECORD_PRESENCE


class _CurlScopeView:
    """Delegate to Fields while masking facts irrelevant to curl arithmetic.

    ``_complex_grid_reasons`` is the single home for the complex storage, phase,
    boundary, no-PML, nonlinearity, beta, BFAST and stored-E clauses.
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
    cb,                               # chunk record: the curl's partner presence
    nx, ny, nz, n_elem, dtdx,         # n_elem = COMPLEX cells; dtdx pre-rounded to f32
    cxr, cxi, cyr, cyi, czr, czi,     # per-axis connection factor, float32 parts
    BACKWARD: tl.constexpr,           # 0 = B (forward differences), 1 = D
    BCX: tl.constexpr, BCY: tl.constexpr, BCZ: tl.constexpr,
    CNX: tl.constexpr, CNY: tl.constexpr, CNZ: tl.constexpr,   # connection kind per axis
    COND0: tl.constexpr, COND1: tl.constexpr, COND2: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One complex/Bloch curl plus the per-target no-PML conductive tail.

    Everything through the ownership mask is line for line the shared
    :func:`.complex_fields.bloch_pml_curl_step` body: the ghost rule, the Bloch
    connection of the unshifted partners on plane 0, MEEP's stencil per part with the
    record's partner presence, and the ownership mask. Only its split-field PML tail
    is replaced by ``_apply_conductive_update``'s three sequential passes, each a
    real product or difference per part. Do not flatten the passes: each rounds to
    float32 before the next begins.
    """
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = idx < n_elem
    nyz = ny * nz
    k = idx % nz
    plane = idx // nz
    j = plane % ny
    i = plane // ny

    # --- the ghost rule, per axis (stepping._shift_up / _shift_down) ------------
    # PERIODIC wraps (a plain roll, phased or not); METALLIC serves an exact 0.0 past
    # the wall -- a (+0.0, +0.0) word pair from `other=` IS the complex metallic ghost.
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

    # --- loads: two words per operand ------------------------------------------
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

    # --- the connection on the UNSHIFTED partners, plane 0 of each partner's axis
    # Target 0 reads c along y and b along z, target 1 a along z and c along x,
    # target 2 b along x and a along y (stepping's term table). `tl.where` is a
    # bitwise select, so every cell off plane 0 keeps its loaded words.
    at_x, at_y, at_z = i == 0, j == 0, k == 0
    c0_re, c0_im = c_re, c_im          # target 0's g1, connected on y
    b0_re, b0_im = b_re, b_im          # target 0's g2, connected on z
    a1_re, a1_im = a_re, a_im          # target 1's g1, connected on z
    c1_re, c1_im = c_re, c_im          # target 1's g2, connected on x
    b2_re, b2_im = b_re, b_im          # target 2's g1, connected on x
    a2_re, a2_im = a_re, a_im          # target 2's g2, connected on y
    if CNX:
        r_re, r_im = _connect(c_re, c_im, cxr, cxi, CNX)
        c1_re = tl.where(at_x, r_re, c_re)
        c1_im = tl.where(at_x, r_im, c_im)
        r_re, r_im = _connect(b_re, b_im, cxr, cxi, CNX)
        b2_re = tl.where(at_x, r_re, b_re)
        b2_im = tl.where(at_x, r_im, b_im)
    if CNY:
        r_re, r_im = _connect(c_re, c_im, cyr, cyi, CNY)
        c0_re = tl.where(at_y, r_re, c_re)
        c0_im = tl.where(at_y, r_im, c_im)
        r_re, r_im = _connect(a_re, a_im, cyr, cyi, CNY)
        a2_re = tl.where(at_y, r_re, a_re)
        a2_im = tl.where(at_y, r_im, a_im)
    if CNZ:
        r_re, r_im = _connect(b_re, b_im, czr, czi, CNZ)
        b0_re = tl.where(at_z, r_re, b_re)
        b0_im = tl.where(at_z, r_im, b_im)
        r_re, r_im = _connect(a_re, a_im, czr, czi, CNZ)
        a1_re = tl.where(at_z, r_re, a_re)
        a1_im = tl.where(at_z, r_im, a_im)

    # --- the curl (stepping._curl_from_operands), per part ---------------------
    # kernels.pml_curl_step's three calls on each part: MEEP's left-to-right
    # association with both partners, the one-partner form where the record says the
    # other partner's derivative axis is invariant, an exact +0 with neither. The
    # dtdx scale is the real scalar times each part (part_products.scale).
    presence = tl.load(cb + RECORD_PRESENCE)
    px = axis_present(presence, 0)
    py = axis_present(presence, 1)
    pz = axis_present(presence, 2)
    curl0_re = curl_stencil(c_y_re, c0_re, b0_re, b_z_re, dtdx, py, pz)
    curl0_im = curl_stencil(c_y_im, c0_im, b0_im, b_z_im, dtdx, py, pz)
    curl1_re = curl_stencil(a_z_re, a1_re, c1_re, c_x_re, dtdx, pz, px)
    curl1_im = curl_stencil(a_z_im, a1_im, c1_im, c_x_im, dtdx, pz, px)
    curl2_re = curl_stencil(b_x_re, b2_re, a2_re, a_y_re, dtdx, px, py)
    curl2_im = curl_stencil(b_x_im, b2_im, a2_im, a_y_im, dtdx, px, py)

    # --- ownership mask (stepping._mask_non_owned_cells) -----------------------
    # Writes +0.0 to BOTH parts -- the array path assigns complex zero.
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
        v0_re, v0_im = _mul_field_left(v0_re, v0_im, factor)
    v0_re = v0_re - curl0_re
    v0_im = v0_im - curl0_im
    if COND0:
        inverse = tl.load(ci0 + idx, mask=live, other=0.0)
        v0_re, v0_im = _mul_field_left(v0_re, v0_im, inverse)

    v1_re = tl.load(f1 + 2 * idx, mask=live, other=0.0)
    v1_im = tl.load(f1 + 2 * idx + 1, mask=live, other=0.0)
    if COND1:
        factor = tl.load(cf1 + idx, mask=live, other=0.0)
        v1_re, v1_im = _mul_field_left(v1_re, v1_im, factor)
    v1_re = v1_re - curl1_re
    v1_im = v1_im - curl1_im
    if COND1:
        inverse = tl.load(ci1 + idx, mask=live, other=0.0)
        v1_re, v1_im = _mul_field_left(v1_re, v1_im, inverse)

    v2_re = tl.load(f2 + 2 * idx, mask=live, other=0.0)
    v2_im = tl.load(f2 + 2 * idx + 1, mask=live, other=0.0)
    if COND2:
        factor = tl.load(cf2 + idx, mask=live, other=0.0)
        v2_re, v2_im = _mul_field_left(v2_re, v2_im, factor)
    v2_re = v2_re - curl2_re
    v2_im = v2_im - curl2_im
    if COND2:
        inverse = tl.load(ci2 + idx, mask=live, other=0.0)
        v2_re, v2_im = _mul_field_left(v2_re, v2_im, inverse)

    tl.store(f0 + 2 * idx, v0_re, mask=live)
    tl.store(f0 + 2 * idx + 1, v0_im, mask=live)
    tl.store(f1 + 2 * idx, v1_re, mask=live)
    tl.store(f1 + 2 * idx + 1, v1_im, mask=live)
    tl.store(f2 + 2 * idx, v2_re, mask=live)
    tl.store(f2 + 2 * idx + 1, v2_im, mask=live)


def complex_conductive_no_pml_curl_coverage(
    fields: Any, pml: Any, sub_step: str,
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
        _CurlScopeView(fields), pml, grid, require_active_pml=False)
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
    """A launchable complex/Bloch no-PML conductive curl plan.

    ``connections`` / ``connection_values`` are
    :func:`.complex_fields.connection_arguments`' kinds and float32 parts for this
    sub-step's difference direction, refused on an axis that does not wrap (the array
    path connects only a wrapping axis). ``chunk_bounds`` is the chunk record,
    required and bound as :class:`.complex_fields.ComplexPmlCurlPlan` binds it
    (``launch.bind_chunk_bounds``); the kernel reads its presence word.
    """

    __slots__ = (
        "sub_step", "shape", "n_elem", "dtdx", "backward", "bc", "cond",
        "connections", "connection_values", "block", "num_warps",
        "chunk_bounds", "_targets", "_sources", "_condfac", "_condinv", "_cb",
        "_grid", "_kernel",
    )

    def __init__(
        self, sub_step: str, shape: Sequence[int], dtdx: float,
        codes: Sequence[int], connections: Sequence[int],
        connection_values: Sequence[float],
        cond: Sequence[bool], targets: Sequence[Any], sources: Sequence[Any],
        condfac: Sequence[Any], condinv: Sequence[Any], *,
        chunk_bounds: Any, block: int = DEFAULT_BLOCK, kernel: Any = None,
        num_warps: Optional[int] = None,
    ) -> None:
        if sub_step not in SUB_STEPS:
            raise ValueError(f"unknown curl sub-step {sub_step!r}")
        from .launch import bind_chunk_bounds  # noqa: PLC0415

        self.sub_step = sub_step
        self.shape = tuple(int(n) for n in shape)
        self.n_elem = self.shape[0] * self.shape[1] * self.shape[2]
        # float(): the array path scales each part by the float32-rounded scalar
        # (part_products.scale); Triton types a Python float argument as fp32.
        self.dtdx = float(dtdx)
        self.backward = int(SUB_STEPS[sub_step]["backward"])
        self.bc = tuple(int(code) for code in codes)
        self.cond = tuple(int(bool(flag)) for flag in cond)
        if len(self.cond) != 3:
            raise ValueError("a curl plan needs three per-target conductivity flags")
        self.connections = tuple(int(kind) for kind in connections)
        if len(self.connections) != 3 or any(
                kind not in CONNECTION_KINDS.values() for kind in self.connections):
            raise ValueError(f"connections must be three kinds in "
                             f"{sorted(CONNECTION_KINDS.values())}, got {connections!r}")
        for axis, (kind, code) in enumerate(zip(self.connections, self.bc)):
            # stepping._bloch_ghost_operand connects a wrapping axis only: on a wall
            # the kernel would connect a stored plane the array path reads unconnected.
            if kind and code != getattr(PERIODIC, "value", PERIODIC):
                raise ValueError(f"axis {axis} carries a connection but its boundary "
                                 f"code {code} does not wrap")
        self.connection_values = tuple(float(value) for value in connection_values)
        if len(self.connection_values) != 6:
            raise ValueError("a curl plan needs six connection parts (re, im per axis)")
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
        self.chunk_bounds, self._cb = bind_chunk_bounds(
            chunk_bounds, self.shape, targets[0], "ComplexConductiveNoPmlCurlPlan")
        self._grid = ((self.n_elem + self.block - 1) // self.block,)
        self._kernel = kernel

    def run(self, guard: Optional[bool] = None) -> None:
        from .kernels import ENABLE_FP_FUSION  # noqa: PLC0415

        nx, ny, nz = self.shape
        kernel = self._kernel or complex_conductive_no_pml_curl_step
        extra = {} if self.num_warps is None else {"num_warps": self.num_warps}
        kernel[self._grid](
            *self._targets, *self._sources, *self._condfac, *self._condinv, self._cb,
            nx, ny, nz, self.n_elem, self.dtdx, *self.connection_values,
            BACKWARD=self.backward,
            BCX=self.bc[0], BCY=self.bc[1], BCZ=self.bc[2],
            CNX=self.connections[0], CNY=self.connections[1], CNZ=self.connections[2],
            COND0=self.cond[0], COND1=self.cond[1], COND2=self.cond[2],
            BLOCK=self.block,
            enable_fp_fusion=ENABLE_FP_FUSION if guard is None else bool(guard),
            **extra,
        )

    def __repr__(self) -> str:
        return (
            f"ComplexConductiveNoPmlCurlPlan({self.sub_step}, shape={self.shape}, "
            f"bc={self.bc}, connections={self.connections}, cond={self.cond}, "
            f"block={self.block}, "
            f"chunk_bounds={self.chunk_bounds.tolist()})")


def plan_complex_conductive_no_pml_curl(
    fields: Any, pml: Any, sub_step: str, block: Optional[int] = None,
    num_warps: Optional[int] = None, chunk_layout: Any = None,
) -> Optional[ComplexConductiveNoPmlCurlPlan]:
    """Build from engine objects, returning None for every refusal.

    ``chunk_layout`` is the run's chunk division, ``None`` for the one-rank default.
    With no absorber every division binds the same words: no membership run, and the
    grid's presence word.
    """
    verdict = complex_conductive_no_pml_curl_coverage(
        fields, pml, sub_step)
    if not verdict.covered:
        return None
    from ..stepping import _boundary_kinds  # noqa: PLC0415

    grid = fields.grid
    spec = SUB_STEPS[sub_step]
    targets = tuple(spec["targets"])
    kinds = _boundary_kinds(grid, None)
    phases = bloch_phase_table(grid, kinds)
    connections, values = connection_arguments(phases, backward=bool(spec["backward"]))
    flags = conductive_no_pml_targets(fields, sub_step)
    return ComplexConductiveNoPmlCurlPlan(
        sub_step, grid.shape, grid.dt / grid.dx,
        [1 if kind == "metallic" else 0 for kind in kinds],
        connections, values, flags,
        [getattr(fields, name) for name in targets],
        source_arrays(fields, sub_step),
        [fields.condfac_for(name) if flags[index] else None
         for index, name in enumerate(targets)],
        [fields.condinv_for(name) if flags[index] else None
         for index, name in enumerate(targets)],
        # None, not the inert layer object: the absorber-free division, whose record
        # has no run and the grid's partner presence.
        chunk_bounds=chunk_bounds_for(fields, None, chunk_layout),
        block=DEFAULT_BLOCK if block is None else block,
        num_warps=num_warps,
    )


def plan_complex_conductive_no_pml_curl_from_arrays(
    sub_step: str, arrays: Dict[str, Any], phases: Sequence[Optional[complex]],
    dtdx: float, codes: Sequence[int],
    cond: Sequence[bool] = (True, True, True), *,
    chunk_bounds: Any, block: Optional[int] = None, kernel: Any = None,
    num_warps: Optional[int] = None,
) -> ComplexConductiveNoPmlCurlPlan:
    """Build from deliberate bare arrays for the device gate and mutations.

    ``chunk_bounds`` has no default: a harness states the record it binds, and a plan
    built without one is refused by name rather than defaulted (``kernel=`` carries a
    gate's mutation override; dropping either would disarm the legs that need it).
    """
    spec = SUB_STEPS[sub_step]
    targets = tuple(spec["targets"])
    flags = tuple(bool(flag) for flag in cond)
    shape = tuple(int(n) for n in arrays[targets[0]].shape)
    connections, values = connection_arguments(tuple(phases),
                                               backward=bool(spec["backward"]))
    return ComplexConductiveNoPmlCurlPlan(
        sub_step, shape, dtdx, codes, connections, values, flags,
        [arrays[name] for name in targets],
        [arrays[name] for name in spec["sources"]],
        [arrays[f"condfac_{name}"] if flags[index] else None
         for index, name in enumerate(targets)],
        [arrays[f"condinv_{name}"] if flags[index] else None
         for index, name in enumerate(targets)],
        chunk_bounds=chunk_bounds,
        block=DEFAULT_BLOCK if block is None else block,
        kernel=kernel, num_warps=num_warps,
    )


# The central planner consults this family through lazy launch/package
# forwarders. Driver/fastpath certification remains separate: the family must
# pass its CUDA byte gate and the central seam must be recertified before this
# arm is eligible for an opted-in driver route.
