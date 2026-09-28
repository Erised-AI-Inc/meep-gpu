"""Complex64, pole-aware ``update_E`` with stored E and no active PML.

This family is the exact intersection left by the existing constitutive arms for
the four Group-I load/dump corpus rows: complex field storage, a registered
Lorentz/Drude pole, stored E, conductivity, Bloch-periodic boundaries, and an
inert absorber.  The real no-PML stored-E body refuses complex storage;
``complex_fields`` and ``dispersive_update_e`` require an active PML.  This file
does not widen any of those predicates.

The array path is ``stepping.update_E`` followed through
``Fields.displacement_minus_polarization``.  With neither nonlinearity nor an
off-diagonal inverse permittivity it is, per component, exactly::

    source = ((D - P0) - P1) - ...     # registration order, left to right
    E[...] = source * inverse_epsilon  # complex field on the left

The poles may not be pre-summed: that changes float32 rounding at two or more
poles.  The plan caches pole *order* but resolves every live ``state.P`` pointer
at launch, because ``PolarizationState.update`` rotates P/P_prev/scratch.

Conductivity is deliberately admitted without a coefficient or branch.  It is
read in the B/D curl recurrences, not in ``update_E``.  A nonzero Bloch vector is
also admitted: this sub-step is pointwise and neither shifts a field nor applies
a boundary phase.  The only complex multiplication here is the already-probed
``c8_mul_f4_field_left`` orientation, so the shared four-pattern expansion
license is necessary and sufficient; the fifth complex-ADE scalar pattern is
not executed by this body.

The central planner reaches this family through lazy launch/package forwarders.
Driver/fastpath certification remains separate: the family must pass its CUDA
byte gate and the central seam must be recertified before this arm is eligible
for an opted-in driver route.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import coverage as _coverage
from .complex_fields import (
    DEFAULT_BLOCK,
    PROBE_PATTERNS,
    _UnavailableKernel,
    _complex_volume_reasons,
    _expansion_reasons,
    _mul_field_left,
    _resolve_expansion,
    _word_view,
)

try:  # The laptop predicate remains importable without optional Triton.
    import triton
    import triton.language as tl
    _TRITON_IMPORT_ERROR: Optional[BaseException] = None
except ImportError as _exc:  # pragma: no cover - exercised by laptop tests
    _TRITON_IMPORT_ERROR = _exc

    class _MissingTriton:
        @staticmethod
        def jit(function):
            return _UnavailableKernel(function.__name__, _TRITON_IMPORT_ERROR)

    class _MissingLanguage:
        @staticmethod
        def constexpr(value):
            return value

    triton = _MissingTriton()  # type: ignore[assignment]
    tl = _MissingLanguage()  # type: ignore[assignment]


MAX_POLES = 8
E_TERMS: Tuple[Tuple[str, str], ...] = (
    ("Ex", "Dx"), ("Ey", "Dy"), ("Ez", "Dz"),
)

__all__ = [
    "E_TERMS",
    "MAX_POLES",
    "PROBE_PATTERNS",
    "ComplexStoredEPlan",
    "LiveComplexPoleBinding",
    "StaticComplexPoleBinding",
    "complex_stored_e_coverage",
    "complex_stored_e_step",
    "plan_complex_stored_e",
    "plan_complex_stored_e_from_arrays",
    "poles_per_component",
]


@triton.jit
def _subtract_complex_poles(
    source, p0, p1, p2, p3, p4, p5, p6, p7,
    word, live,
    NP: tl.constexpr,
):
    """Load ``D`` and subtract up to eight complex P arrays in exact order."""
    real = tl.load(source + word, mask=live, other=0.0)
    imag = tl.load(source + word + 1, mask=live, other=0.0)
    if NP > 0:
        real = real - tl.load(p0 + word, mask=live, other=0.0)
        imag = imag - tl.load(p0 + word + 1, mask=live, other=0.0)
    if NP > 1:
        real = real - tl.load(p1 + word, mask=live, other=0.0)
        imag = imag - tl.load(p1 + word + 1, mask=live, other=0.0)
    if NP > 2:
        real = real - tl.load(p2 + word, mask=live, other=0.0)
        imag = imag - tl.load(p2 + word + 1, mask=live, other=0.0)
    if NP > 3:
        real = real - tl.load(p3 + word, mask=live, other=0.0)
        imag = imag - tl.load(p3 + word + 1, mask=live, other=0.0)
    if NP > 4:
        real = real - tl.load(p4 + word, mask=live, other=0.0)
        imag = imag - tl.load(p4 + word + 1, mask=live, other=0.0)
    if NP > 5:
        real = real - tl.load(p5 + word, mask=live, other=0.0)
        imag = imag - tl.load(p5 + word + 1, mask=live, other=0.0)
    if NP > 6:
        real = real - tl.load(p6 + word, mask=live, other=0.0)
        imag = imag - tl.load(p6 + word + 1, mask=live, other=0.0)
    if NP > 7:
        real = real - tl.load(p7 + word, mask=live, other=0.0)
        imag = imag - tl.load(p7 + word + 1, mask=live, other=0.0)
    return real, imag


@triton.jit
def complex_stored_e_step(
    f0, f1, f2,                         # Ex/Ey/Ez word pointers (out)
    g0, g1, g2,                         # Dx/Dy/Dz word pointers (in)
    e0, e1, e2,                         # float32 inverse epsilon (in)
    a0, a1, a2, a3, a4, a5, a6, a7,   # Ex poles, registration order
    b0, b1, b2, b3, b4, b5, b6, b7,   # Ey poles, registration order
    c0, c1, c2, c3, c4, c5, c6, c7,   # Ez poles, registration order
    n_elem,
    NP0: tl.constexpr, NP1: tl.constexpr, NP2: tl.constexpr,
    EXPANSION: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Store complex ``(D - P0 - P1 ...) * inv_eps`` for all E components."""
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = idx < n_elem
    word = 2 * idx

    s0_re, s0_im = _subtract_complex_poles(
        g0, a0, a1, a2, a3, a4, a5, a6, a7, word, live, NP0)
    o0_re, o0_im = _mul_field_left(
        s0_re, s0_im, tl.load(e0 + idx, mask=live, other=0.0), EXPANSION)
    tl.store(f0 + word, o0_re, mask=live)
    tl.store(f0 + word + 1, o0_im, mask=live)

    s1_re, s1_im = _subtract_complex_poles(
        g1, b0, b1, b2, b3, b4, b5, b6, b7, word, live, NP1)
    o1_re, o1_im = _mul_field_left(
        s1_re, s1_im, tl.load(e1 + idx, mask=live, other=0.0), EXPANSION)
    tl.store(f1 + word, o1_re, mask=live)
    tl.store(f1 + word + 1, o1_im, mask=live)

    s2_re, s2_im = _subtract_complex_poles(
        g2, c0, c1, c2, c3, c4, c5, c6, c7, word, live, NP2)
    o2_re, o2_im = _mul_field_left(
        s2_re, s2_im, tl.load(e2 + idx, mask=live, other=0.0), EXPANSION)
    tl.store(f2 + word, o2_re, mask=live)
    tl.store(f2 + word + 1, o2_im, mask=live)


def poles_per_component(fields: Any) -> Dict[str, Tuple[Any, ...]]:
    """Driving states in exactly ``fields.polarizations`` registration order."""
    states = tuple(getattr(fields, "polarizations", ()) or ())
    return {
        component: tuple(
            state for state in states
            if _coverage._call(state, "drives", component, default=False)
        )
        for component in _coverage.ELECTRIC_COMPONENTS
    }


def complex_stored_e_coverage(fields: Any, pml: Any,
                              probe: Any = None) -> _coverage.Coverage:
    """May the complex, pole-aware, no-PML stored-E body own ``update_E``?"""
    reasons: List[str] = []
    grid = getattr(fields, "grid", None)
    if grid is None:
        return _coverage.Coverage(False, ("fields carries no grid",))

    xp = getattr(grid, "xp", None)
    if getattr(xp, "__name__", "") != "cupy":
        reasons.append(
            f"array module is {getattr(xp, '__name__', xp)!r}, not cupy")
    if not getattr(fields, "force_complex_fields", False):
        reasons.append(
            "force_complex_fields=False: this family requires complex64 storage; "
            "the real no-PML stored-E family owns float32")

    if pml is not None and bool(getattr(pml, "is_active", False)):
        reasons.append(
            "an active PML layer is installed: update_E uses the split-field "
            "constitutive recurrence and writes f_w there")
    if bool(getattr(fields, "_pml_active", False)):
        reasons.append(
            "Fields is in PML storage mode: this no-absorber body neither reads "
            "PML coefficients nor writes f_w")
    if not getattr(fields, "stores_E", False):
        reasons.append(
            "E is not stored: update_E returns without writing on the no-PML path")

    if getattr(fields, "has_offdiagonal_epsilon", False):
        reasons.append(
            "an off-diagonal inverse permittivity is installed: update_E becomes "
            "a three-component row product")
    if getattr(fields, "has_nonlinearity", False):
        reasons.append(
            "chi2/chi3 is installed: update_E applies the nonlinear Pade factor")

    # Conservative products not present in the four target rows.  Bloch is NOT
    # among them: this pointwise sub-step never consults a boundary phase.
    if _coverage._call(grid, "has_symmetry", default=False):
        reasons.append("a mirror plane is active (not certified for this family)")
    for axis in range(3):
        if _coverage._call(grid, "is_mirrored", axis, default=False):
            reasons.append(
                f"axis {axis} is mirror-folded (not certified for this family)")
    if getattr(grid, "cylindrical", False):
        reasons.append("cylindrical coordinates are not certified for this family")
    for axis in range(3):
        if _coverage._call(grid, "is_axis", axis, default=False):
            reasons.append(
                f"axis {axis} is the cylindrical r=0 axis (not certified)")
    if getattr(grid, "bfast_active", False):
        reasons.append("BFAST is active (not certified for this family)")
    if float(getattr(grid, "beta", 0.0)) != 0.0:
        reasons.append(
            f"special_kz beta={getattr(grid, 'beta', 0.0)!r} is nonzero "
            "(not certified for this family)")

    reasons.extend(_coverage._susceptibility_reasons(fields))
    order = poles_per_component(fields)
    counts = {component: len(order[component]) for component, _ in E_TERMS}
    if not any(counts.values()):
        reasons.append(
            "no registered susceptibility drives E: this narrowly scoped family "
            "requires at least one live pole")
    for component, count in counts.items():
        if count > MAX_POLES:
            reasons.append(
                f"{component} is driven by {count} poles, more than the kernel's "
                f"MAX_POLES={MAX_POLES} compiled slots")

    shape = tuple(getattr(grid, "shape", ()))
    if len(shape) != 3:
        reasons.append(f"grid shape {shape!r} is not three-dimensional")
        return _coverage.Coverage(False, tuple(reasons))
    total = int(shape[0]) * int(shape[1]) * int(shape[2])
    if 2 * total >= 2 ** 31:
        reasons.append(
            f"{total} complex cells exceed the int32 word-offset range")

    for component, states in order.items():
        for state_index, state in enumerate(states):
            for slot in ("P", "P_prev"):
                array = (getattr(state, slot, {}) or {}).get(component)
                if array is None:
                    reasons.append(
                        f"polarization {state_index} {slot}[{component}] is not allocated")
                else:
                    reasons.extend(_complex_volume_reasons(
                        f"polarization {state_index} {slot}[{component}]",
                        array, shape))

    for component, displacement in E_TERMS:
        for name in (component, displacement):
            array = getattr(fields, name, None)
            if array is None:
                reasons.append(f"{name} is not allocated")
            else:
                reasons.extend(_complex_volume_reasons(name, array, shape))
    reasons.extend(_coverage._inverse_epsilon_reasons(fields, shape))
    reasons.extend(_expansion_reasons(probe))
    return _coverage.Coverage(not reasons, tuple(reasons))


class LiveComplexPoleBinding:
    """Cache pole identity/order, but resolve each current P array per launch."""

    __slots__ = ("fields", "order", "counts")

    def __init__(self, fields: Any,
                 order: Dict[str, Sequence[Any]]) -> None:
        self.fields = fields
        self.order = {
            component: tuple(states) for component, states in order.items()
        }
        self.counts = tuple(
            len(self.order[component]) for component, _ in E_TERMS)

    def arrays(self) -> Tuple[Tuple[Any, ...], ...]:
        live = poles_per_component(self.fields)
        groups: List[Tuple[Any, ...]] = []
        for component, _ in E_TERMS:
            expected = self.order[component]
            found = live[component]
            if len(found) != len(expected) or any(
                    current is not planned
                    for current, planned in zip(found, expected)):
                raise RuntimeError(
                    f"the polarization set driving {component} changed after "
                    "planning; pole order is bit-load-bearing")
            groups.append(tuple(state.P[component] for state in expected))
        return tuple(groups)


class StaticComplexPoleBinding:
    """Fixed pole arrays for isolated gates and explicit mutation legs."""

    __slots__ = ("_arrays", "counts")

    def __init__(self, arrays: Sequence[Sequence[Any]]) -> None:
        self._arrays = tuple(tuple(group) for group in arrays)
        if len(self._arrays) != len(E_TERMS):
            raise ValueError("a pole binding requires one group per E component")
        self.counts = tuple(len(group) for group in self._arrays)

    def arrays(self) -> Tuple[Tuple[Any, ...], ...]:
        return self._arrays


class ComplexStoredEPlan:
    """One launchable complex64 stored-E ``update_E`` plan."""

    __slots__ = (
        "shape", "n_elem", "block", "counts", "expansion", "_targets",
        "_sources", "_inv_eps", "_poles", "_grid", "_kernel", "_pointer",
    )

    def __init__(self, shape: Sequence[int], targets: Sequence[Any],
                 sources: Sequence[Any], inverse_epsilon: Sequence[Any],
                 poles: Any, expansion: int, block: int = DEFAULT_BLOCK,
                 kernel: Any = None, pointer: Any = None) -> None:
        if pointer is None:
            from .launch import CupyPointer  # noqa: PLC0415
            pointer = CupyPointer
        self.shape = tuple(int(value) for value in shape)
        self.n_elem = self.shape[0] * self.shape[1] * self.shape[2]
        self.block = int(block)
        self.counts = tuple(int(value) for value in poles.counts)
        if len(self.counts) != len(E_TERMS):
            raise ValueError("a plan requires one pole count per E component")
        for (component, _), count in zip(E_TERMS, self.counts):
            if count > MAX_POLES:
                raise ValueError(
                    f"{component} has {count} poles; MAX_POLES={MAX_POLES}")
        self.expansion = int(expansion)
        self._pointer = pointer
        self._targets = tuple(pointer(_word_view(array)) for array in targets)
        self._sources = tuple(pointer(_word_view(array)) for array in sources)
        self._inv_eps = tuple(pointer(array) for array in inverse_epsilon)
        self._poles = poles
        self._grid = ((self.n_elem + self.block - 1) // self.block,)
        self._kernel = kernel

    def run(self, guard: Optional[bool] = None) -> None:
        """Resolve current P pointers and launch the pointwise sub-step in place."""
        if guard is None:
            from .kernels import ENABLE_FP_FUSION  # noqa: PLC0415
            fusion = ENABLE_FP_FUSION
        else:
            fusion = bool(guard)
        groups = self._poles.arrays()
        slots: List[Any] = []
        for index, group in enumerate(groups):
            if len(group) != self.counts[index]:
                raise RuntimeError(
                    f"component {index} resolved {len(group)} poles, not the "
                    f"planned {self.counts[index]}")
            slots.extend(self._pointer(_word_view(array)) for array in group)
            slots.extend(
                [self._sources[index]] * (MAX_POLES - len(group)))
        kernel = self._kernel if self._kernel is not None else complex_stored_e_step
        kernel[self._grid](
            *self._targets, *self._sources, *self._inv_eps, *slots,
            self.n_elem,
            NP0=self.counts[0], NP1=self.counts[1], NP2=self.counts[2],
            EXPANSION=self.expansion, BLOCK=self.block,
            enable_fp_fusion=fusion,
        )

    def __repr__(self) -> str:
        return (
            f"ComplexStoredEPlan(shape={self.shape}, poles={self.counts}, "
            f"expansion={self.expansion}, block={self.block})")


def plan_complex_stored_e(fields: Any, pml: Any, block: Optional[int] = None,
                          probe: Any = None) -> Optional[ComplexStoredEPlan]:
    """Build the engine-object plan, or ``None`` with predicate reasons."""
    if not complex_stored_e_coverage(fields, pml, probe=probe).covered:
        return None
    expansion = _resolve_expansion(probe)
    if expansion is None:  # pragma: no cover - coverage refused the same record
        return None
    order = poles_per_component(fields)
    return ComplexStoredEPlan(
        fields.grid.shape,
        [getattr(fields, component) for component, _ in E_TERMS],
        [getattr(fields, displacement) for _, displacement in E_TERMS],
        [fields.inverse_epsilon_for(component) for component, _ in E_TERMS],
        LiveComplexPoleBinding(fields, order), expansion,
        DEFAULT_BLOCK if block is None else block,
    )


def plan_complex_stored_e_from_arrays(
    arrays: Dict[str, Any], poles: Dict[str, Sequence[Any]], expansion: int,
    block: Optional[int] = None, kernel: Any = None,
) -> ComplexStoredEPlan:
    """Build from deliberate gate arrays, retaining an optional mutated kernel."""
    shape = tuple(int(value) for value in arrays["Ex"].shape)
    return ComplexStoredEPlan(
        shape,
        [arrays[component] for component, _ in E_TERMS],
        [arrays[displacement] for _, displacement in E_TERMS],
        [arrays["inv_eps_" + component] for component, _ in E_TERMS],
        StaticComplexPoleBinding(
            [poles[component] for component, _ in E_TERMS]),
        expansion, DEFAULT_BLOCK if block is None else block, kernel=kernel,
    )
