"""Complex64 Lorentz/Drude ``update_P`` for an inert absorber.

This family closes the ADE sub-step of the four Group-I corpus rows: complex
field storage, a conductivity, one registered susceptibility, and no active
split-field PML.  Conductivity is deliberately not a predicate clause because
it enters the B/D curl recurrence and never ``PolarizationState.update``
(``dispersion.py:658-691``).

The kernel steps complex64 arrays as interleaved float32 word pairs.  Every
product has a real operand and is formed one float32 product per stored part
(``complex_fields._mul_field_left``), MEEP's ``DOCMP2`` loop: ``P * two_minus_od``
(field left), ``gamma1 * P_prev`` and the scalar forms of
``sigma``/``omega0dtsqr``/``gamma1inv`` (coefficient left), and volume
``sigma * W``.  The grouping is MEEP's isotropic ``update_P``
(susceptibility.cpp:251-256), per part, on the four realnum words
``PolarizationState.constants``::

    gamma1inv * (((P*two_minus_od) - (gamma1*P_prev)) + (omega0dtsqr*(sigma*W)))

The plan resolves P, P_prev and the shared scratch pointer before every launch,
then applies the reference three-buffer rotation only after a successful
launch.  Caching any of those pointers is a silent wrong answer after the first
component.

This module is intentionally not wired into ``launch.plan_step`` here.  Its
predicate, builder, gate, and tests are independent; central wiring is a
separate composition edit so concurrent corpus-family work cannot overlap in
``launch.py``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from .complex_fields import (
    DEFAULT_BLOCK,
    _UnavailableKernel,
    _complex_volume_reasons,
    _mul_coefficient_left,
    _mul_field_left,
    _word_view,
)
from .coverage import (
    COVERED_SUSCEPTIBILITY_KINDS,
    ELECTRIC_COMPONENTS,
    Coverage,
    _call,
    _volume_reasons,
    polarization_constant_reasons,
)

try:  # The host predicate remains usable without the optional Triton package.
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


__all__ = [
    "ComplexAdeUpdatePPlan",
    "complex_ade_update_p",
    "complex_ade_update_p_coverage",
    "plan_complex_ade_update_p",
]


@triton.jit
def complex_ade_update_p(
    p_out, p_now, p_prev, sigma, drive,
    two_minus_od, gamma1, omega0dtsqr, gamma1inv, n_elem,
    SIGMA_IS_VOLUME: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One complex64 Lorentz/Drude recurrence, addressed as float32 words.

    :func:`kernels.ade_update_p`'s expression, MEEP's isotropic ``update_P``
    (susceptibility.cpp:251-256) on the four realnum words, stepped PART BY PART:
    MEEP's ``DOCMP2`` loop runs the real recurrence once per stored part, and the
    array path's ``PolarizationState.update`` forms every product through
    ``part_products.scale``. Per part::

        gamma1inv * (((p * two_minus_od) - (gamma1 * pp)) + (omega0dtsqr * (s * w)))

    every operation one float32 rounding; the complex sum and difference are per
    part already.
    """
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = idx < n_elem
    word = 2 * idx

    p_re = tl.load(p_now + word, mask=live, other=0.0)
    p_im = tl.load(p_now + word + 1, mask=live, other=0.0)
    q_re = tl.load(p_prev + word, mask=live, other=0.0)
    q_im = tl.load(p_prev + word + 1, mask=live, other=0.0)
    w_re = tl.load(drive + word, mask=live, other=0.0)
    w_im = tl.load(drive + word + 1, mask=live, other=0.0)

    # PolarizationState.update, per part: p * (2 - od), minus gamma1 * pp.
    a_re, a_im = _mul_field_left(p_re, p_im, two_minus_od)
    b_re, b_im = _mul_coefficient_left(gamma1, q_re, q_im)
    out_re = a_re - b_re
    out_im = a_im - b_im

    if SIGMA_IS_VOLUME:
        s = tl.load(sigma + idx, mask=live, other=0.0)
    else:
        s = sigma
    sw_re, sw_im = _mul_coefficient_left(s, w_re, w_im)
    d_re, d_im = _mul_coefficient_left(omega0dtsqr, sw_re, sw_im)
    out_re = out_re + d_re
    out_im = out_im + d_im
    out_re, out_im = _mul_coefficient_left(gamma1inv, out_re, out_im)

    tl.store(p_out + word, out_re, mask=live)
    tl.store(p_out + word + 1, out_im, mask=live)


def _inert_drive_reasons(fields: Any, pml: Any, component: str) -> List[str]:
    """Require the no-PML stored-E drive that Group I actually presents."""
    reasons: List[str] = []
    if pml is not None and bool(getattr(pml, "is_active", False)):
        reasons.append(
            "an active PML layer is installed: this family carries the no-absorber "
            "stored-E drive, not f_w")
    if bool(getattr(fields, "_pml_active", False)):
        reasons.append(
            "Fields is in PML storage mode: drive_field would return f_w even "
            "though this family requires the no-absorber stored E")
    if not getattr(fields, "stores_E", False):
        reasons.append(
            "E is recomputed instead of stored: update_E would not refresh the "
            "complex ADE drive")

    reader = getattr(fields, "drive_field", None)
    if not callable(reader):
        reasons.append("fields does not expose drive_field()")
        return reasons
    try:
        drive = reader(component)
    except Exception as exc:  # noqa: BLE001 - an unreadable drive is refused
        reasons.append(f"drive_field({component!r}) raised {exc!r}")
        return reasons
    stored = getattr(fields, component, None)
    if drive is None:
        reasons.append(f"drive_field({component!r}) is None")
    elif drive is not stored:
        reasons.append(
            f"drive_field({component!r}) is not the stored {component} array")
    return reasons


def complex_ade_update_p_coverage(fields: Any, pml: Any, state: Any,
                                  component: str) -> Coverage:
    """May the complex word-pair kernel advance this pole/component?"""
    reasons: List[str] = []
    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False, ("fields carries no grid",))

    xp = getattr(grid, "xp", None)
    if getattr(xp, "__name__", "") != "cupy":
        reasons.append(f"array module is {getattr(xp, '__name__', xp)!r}, not cupy")
    if not getattr(fields, "force_complex_fields", False):
        reasons.append(
            "force_complex_fields=False: complex64 storage is required; the real "
            "ADE families own float32")

    if component not in ELECTRIC_COMPONENTS:
        reasons.append(f"component {component!r} is outside {ELECTRIC_COMPONENTS}")
        return Coverage(False, tuple(reasons))
    if not _call(state, "drives", component, default=False):
        reasons.append(f"this susceptibility does not drive {component}")

    kind = getattr(getattr(state, "susceptibility", None), "kind", None)
    if kind not in COVERED_SUSCEPTIBILITY_KINDS:
        reasons.append(
            f"kind {kind!r} is outside {COVERED_SUSCEPTIBILITY_KINDS}")

    reasons.extend(polarization_constant_reasons(state))

    dt = getattr(grid, "dt", None)
    if dt is None:
        reasons.append("grid carries no dt")
    elif getattr(getattr(state, "grid", None), "dt", dt) != dt:
        reasons.append("the state's coefficients were built for a different dt")

    reasons.extend(_inert_drive_reasons(fields, pml, component))
    shape = tuple(getattr(grid, "shape", ()))
    if len(shape) != 3:
        reasons.append(f"grid shape {shape!r} is not three-dimensional")
        return Coverage(False, tuple(reasons))
    total = int(shape[0]) * int(shape[1]) * int(shape[2])
    if 2 * total >= 2 ** 31:
        reasons.append(
            f"{total} complex cells exceed the word-pair kernel's int32 index range")

    buffers: Dict[str, Any] = {
        "P": (getattr(state, "P", {}) or {}).get(component),
        "P_prev": (getattr(state, "P_prev", {}) or {}).get(component),
        "_scratch": getattr(state, "_scratch", None),
    }
    for name, array in buffers.items():
        if array is None:
            reasons.append(f"{name}[{component}] is not allocated")
        else:
            reasons.extend(_complex_volume_reasons(
                f"{name}[{component}]", array, shape))

    reader = getattr(fields, "drive_field", None)
    drive = None
    if callable(reader):
        try:
            drive = reader(component)
        except Exception:  # already reported above
            drive = None
    if drive is not None:
        reasons.extend(_complex_volume_reasons(
            f"drive_field({component!r})", drive, shape))

    sigma = (getattr(state, "sigma", {}) or {}).get(component)
    if sigma is None:
        reasons.append(f"sigma[{component}] is missing")
    elif getattr(sigma, "shape", ()):
        reasons.extend(_volume_reasons(f"sigma[{component}]", sigma, shape))
    else:
        try:
            float(sigma)
        except Exception:  # noqa: BLE001 - neither scalar nor volume
            reasons.append(
                f"sigma[{component}]={sigma!r} is neither a scalar nor a volume")

    return Coverage(not reasons, tuple(reasons))


class ComplexAdeUpdatePPlan:
    """One complex susceptibility plan with live three-buffer resolution."""

    __slots__ = (
        "state", "components", "shape", "n_elem", "block",
        "_sigma_is_volume", "_grid", "_kernel", "_pointer",
    )

    def __init__(self, state: Any, components: Sequence[str], shape: Sequence[int],
                 block: int = DEFAULT_BLOCK, kernel: Any = None,
                 pointer: Any = None) -> None:
        if pointer is None:
            from .launch import CupyPointer  # noqa: PLC0415
            pointer = CupyPointer
        self.state = state
        self.components = tuple(components)
        self.shape = tuple(int(value) for value in shape)
        self.n_elem = self.shape[0] * self.shape[1] * self.shape[2]
        self.block = int(block)
        self._sigma_is_volume = {
            component: bool(getattr(state.sigma[component], "shape", ()))
            for component in self.components
        }
        self._grid = ((self.n_elem + self.block - 1) // self.block,)
        self._kernel = kernel
        self._pointer = pointer

    def run(self, drive, guard: Optional[bool] = None) -> None:
        """Launch each component, rotating ownership only after each success."""
        if guard is None:
            from .kernels import ENABLE_FP_FUSION  # noqa: PLC0415
            fusion = ENABLE_FP_FUSION
        else:
            # The explicit gate/test route must remain usable without importing
            # kernels.py on the no-Triton merge-bar host.
            fusion = bool(guard)

        state = self.state
        kernel = self._kernel if self._kernel is not None else complex_ade_update_p
        # MEEP's four realnum words; each is a float32 word, passed exactly.
        two_minus_od, gamma1, omega0dtsqr, gamma1inv = state.constants
        for component in self.components:
            p = state.P[component]
            p_prev = state.P_prev[component]
            scratch = state._scratch
            w = drive(component)
            sigma = state.sigma[component]
            volume = self._sigma_is_volume[component]
            kernel[self._grid](
                self._pointer(_word_view(scratch)),
                self._pointer(_word_view(p)),
                self._pointer(_word_view(p_prev)),
                self._pointer(sigma) if volume else float(sigma),
                self._pointer(_word_view(w)),
                float(two_minus_od), float(gamma1), float(omega0dtsqr),
                float(gamma1inv), self.n_elem,
                SIGMA_IS_VOLUME=1 if volume else 0,
                BLOCK=self.block,
                enable_fp_fusion=fusion,
            )
            state.P[component] = scratch
            state.P_prev[component] = p
            state._scratch = p_prev

    def __repr__(self) -> str:
        return (f"ComplexAdeUpdatePPlan({self.components}, shape={self.shape}, "
                f"block={self.block})")


def plan_complex_ade_update_p(fields: Any, pml: Any, state: Any,
                              block: Optional[int] = None) -> Optional[ComplexAdeUpdatePPlan]:
    """Build one all-or-nothing complex ADE plan, or ``None`` on refusal."""
    driven = getattr(state, "driven", None)
    components = tuple(driven()) if callable(driven) else ()
    if not components:
        return None
    for component in components:
        if not complex_ade_update_p_coverage(
                fields, pml, state, component).covered:
            return None
    return ComplexAdeUpdatePPlan(
        state, components, fields.grid.shape,
        DEFAULT_BLOCK if block is None else block)
