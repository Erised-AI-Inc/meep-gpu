"""Complex64 Lorentz/Drude ``update_P`` for an inert absorber.

This family closes the ADE sub-step of the four Group-I corpus rows: complex
field storage, a conductivity, one registered susceptibility, and no active
split-field PML.  Conductivity is deliberately not a predicate clause because
it enters the B/D curl recurrence and never ``PolarizationState.update``
(``dispersion.py:658-691``).

The kernel steps complex64 arrays as interleaved float32 word pairs.  Its three
complex-by-real operand orientations are licensed by the shared complex
expansion probe.  Two were already named there; the first recurrence term adds
one genuinely new orientation:

``c8_mul_python_float_field_left``
    ``P * c_now`` from ``xp.multiply(P, c_now, out=scratch)``.  The complex
    field is left and the Python coefficient is right.  It is not the existing
    ``c8_mul_f4_field_left`` (whose coefficient is an array), nor the existing
    ``python_float_left`` (whose scalar is left).  A probe record missing this
    fifth pattern refuses by name; no expansion arm is inferred.

Every other term maps to a previously probed orientation: ``c_prev * P_prev``
and the scalar forms of ``sigma``/``c_drive`` are ``python_float_left``;
volume ``sigma * W`` is ``f4_mul_c8_coefficient_left``.  The grouping is the
array path's two in-place additions, exactly::

    ((P*c_now) + (c_prev*P_prev)) + (c_drive*(sigma*W))

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

from typing import Any, Dict, List, Optional, Sequence, Tuple

from .complex_fields import (
    DEFAULT_BLOCK,
    PROBE_PATTERNS,
    _UnavailableKernel,
    _complex_volume_reasons,
    _mul_coefficient_left,
    _mul_field_left,
    _policy_in_force,
    _word_view,
    expansion_certification_reasons,
    expansion_from_probe,
    expansion_license,
    expansion_policy_reasons,
    load_expansion_probe,
)
from .coverage import (
    COVERED_SUSCEPTIBILITY_KINDS,
    ELECTRIC_COMPONENTS,
    Coverage,
    _call,
    _volume_reasons,
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


COMPLEX_ADE_PROBE_PATTERN = "c8_mul_python_float_field_left"
COMPLEX_ADE_PROBE_PATTERNS: Tuple[str, ...] = (
    *PROBE_PATTERNS,
    COMPLEX_ADE_PROBE_PATTERN,
)

__all__ = [
    "COMPLEX_ADE_PROBE_PATTERN",
    "COMPLEX_ADE_PROBE_PATTERNS",
    "ComplexAdeUpdatePPlan",
    "complex_ade_update_p",
    "complex_ade_update_p_coverage",
    "plan_complex_ade_update_p",
]


@triton.jit
def complex_ade_update_p(
    p_out, p_now, p_prev, sigma, drive,
    c_now, c_prev, c_drive, n_elem,
    SIGMA_IS_VOLUME: tl.constexpr,
    EXPANSION: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One complex64 Lorentz/Drude recurrence, addressed as float32 words."""
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = idx < n_elem
    word = 2 * idx

    p_re = tl.load(p_now + word, mask=live, other=0.0)
    p_im = tl.load(p_now + word + 1, mask=live, other=0.0)
    q_re = tl.load(p_prev + word, mask=live, other=0.0)
    q_im = tl.load(p_prev + word + 1, mask=live, other=0.0)
    w_re = tl.load(drive + word, mask=live, other=0.0)
    w_im = tl.load(drive + word + 1, mask=live, other=0.0)

    # dispersion.py:686-688, including operand orientation and grouping.
    a_re, a_im = _mul_field_left(p_re, p_im, c_now, EXPANSION)
    b_re, b_im = _mul_coefficient_left(c_prev, q_re, q_im, EXPANSION)
    out_re = a_re + b_re
    out_im = a_im + b_im

    if SIGMA_IS_VOLUME:
        s = tl.load(sigma + idx, mask=live, other=0.0)
    else:
        s = sigma
    sw_re, sw_im = _mul_coefficient_left(s, w_re, w_im, EXPANSION)
    d_re, d_im = _mul_coefficient_left(c_drive, sw_re, sw_im, EXPANSION)
    out_re = out_re + d_re
    out_im = out_im + d_im

    tl.store(p_out + word, out_re, mask=live)
    tl.store(p_out + word + 1, out_im, mask=live)


def _probe_record(probe: Any) -> Any:
    return probe if probe is not None else load_expansion_probe()


def _complex_ade_expansion_reasons(probe: Any = None) -> List[str]:
    """Judge the shared probe over this family's five-pattern superset."""
    record = _probe_record(probe)
    from ..expansion_refusal import RefusedExpansionProbe  # noqa: PLC0415

    if isinstance(record, RefusedExpansionProbe):
        return [
            f"the expansion probe artifact offered through {record.key} was "
            "REFUSED by this dispatch and may not license the complex ADE arm: "
            + "; ".join(record.reasons)
        ]
    if record is None:
        return [
            "no complex-multiply expansion probe artifact is available; the "
            f"complex ADE kernel requires {COMPLEX_ADE_PROBE_PATTERNS!r} and "
            "may not guess an EXPANSION constexpr"
        ]
    verdict = expansion_license(record, COMPLEX_ADE_PROBE_PATTERNS)
    if verdict["expansion"] is None:
        return [
            "the expansion probe artifact does not license the complex ADE "
            f"orientations {COMPLEX_ADE_PROBE_PATTERNS!r}: "
            + "; ".join(verdict["refusals"])
        ]
    policy = _policy_in_force()
    return (expansion_policy_reasons(record, policy)
            + expansion_certification_reasons(policy))


def _resolve_complex_ade_expansion(probe: Any = None) -> Optional[int]:
    if _complex_ade_expansion_reasons(probe):
        return None
    record = _probe_record(probe)
    if not isinstance(record, dict):
        return None
    return expansion_from_probe(record, COMPLEX_ADE_PROBE_PATTERNS)


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
                                  component: str,
                                  probe: Any = None) -> Coverage:
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

    coefficients = getattr(state, "_coefficients", None)
    if coefficients is None or len(tuple(coefficients)) != 3:
        reasons.append("the (c_now, c_prev, c_drive) coefficient triple is missing")
    else:
        for name, value in zip(("c_now", "c_prev", "c_drive"), coefficients):
            try:
                number = float(value)
            except Exception:  # noqa: BLE001 - non-numeric is a refusal
                reasons.append(f"{name}={value!r} is not a float")
                continue
            if number != number or number in (float("inf"), float("-inf")):
                reasons.append(f"{name}={number!r} is not finite")

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

    reasons.extend(_complex_ade_expansion_reasons(probe))
    return Coverage(not reasons, tuple(reasons))


class ComplexAdeUpdatePPlan:
    """One complex susceptibility plan with live three-buffer resolution."""

    __slots__ = (
        "state", "components", "shape", "n_elem", "block", "expansion",
        "_sigma_is_volume", "_grid", "_kernel", "_pointer",
    )

    def __init__(self, state: Any, components: Sequence[str], shape: Sequence[int],
                 expansion: int, block: int = DEFAULT_BLOCK, kernel: Any = None,
                 pointer: Any = None) -> None:
        if pointer is None:
            from .launch import CupyPointer  # noqa: PLC0415
            pointer = CupyPointer
        self.state = state
        self.components = tuple(components)
        self.shape = tuple(int(value) for value in shape)
        self.n_elem = self.shape[0] * self.shape[1] * self.shape[2]
        self.block = int(block)
        self.expansion = int(expansion)
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
        c_now, c_prev, c_drive = state._coefficients
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
                float(c_now), float(c_prev), float(c_drive), self.n_elem,
                SIGMA_IS_VOLUME=1 if volume else 0,
                EXPANSION=self.expansion,
                BLOCK=self.block,
                enable_fp_fusion=fusion,
            )
            state.P[component] = scratch
            state.P_prev[component] = p
            state._scratch = p_prev

    def __repr__(self) -> str:
        return (f"ComplexAdeUpdatePPlan({self.components}, shape={self.shape}, "
                f"expansion={self.expansion}, block={self.block})")


def plan_complex_ade_update_p(fields: Any, pml: Any, state: Any,
                              block: Optional[int] = None,
                              probe: Any = None) -> Optional[ComplexAdeUpdatePPlan]:
    """Build one all-or-nothing complex ADE plan, or ``None`` on refusal."""
    driven = getattr(state, "driven", None)
    components = tuple(driven()) if callable(driven) else ()
    if not components:
        return None
    for component in components:
        if not complex_ade_update_p_coverage(
                fields, pml, state, component, probe=probe).covered:
            return None
    expansion = _resolve_complex_ade_expansion(probe)
    if expansion is None:  # pragma: no cover - the predicate already refused
        return None
    return ComplexAdeUpdatePPlan(
        state, components, fields.grid.shape, expansion,
        DEFAULT_BLOCK if block is None else block)
