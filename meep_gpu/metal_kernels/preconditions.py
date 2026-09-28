"""The checked precondition every Metal claim is made subject to.

THE CLAIM SHAPE, and it is the whole reason this module exists: **byte-identity to
``stepping.py`` subject to a CHECKED subnormal-free precondition** — never a stated
tolerance and never an ASSUMED precondition. On MPS the float32 subnormal flush is
NATIVE and has no lever (:mod:`.subnormal`), so anything whose operand, result or
intermediate lands in the band returns zero where NumPy returns the subnormal. The
measured cliff, from scaling a whole field state and re-running one kernel:

    1e+00  physical        in_subnormal=0     -> IDENTICAL (3188 words moved)
    1e-20  small normal    in_subnormal=0     -> IDENTICAL (3195 moved)
    1e-30  nearer          in_subnormal=0     -> IDENTICAL (3195 moved)
    1e-38  subnormal band  in_subnormal=3233  -> DIVERGES, 6480 words
    1e-40  deep            in_subnormal=3240  -> DIVERGES, 6480 words

Three decades of headroom, then total divergence. A CLIFF, not a tolerance.

REACHABILITY WAS MEASURED, NOT ARGUED: a 3-D pulsed Gaussian in a PML box, 1,201
steps, censused every 25 across all twelve field arrays and 1.5M nonzero cells,
produced ZERO subnormal cells; the smallest nonzero magnitude bottoms at ~3e-19 and
HOLDS — the tail decays to the float32 round-off floor and stops rather than
descending into the band. A CW control agreed. That measurement is why the
precondition is expected to hold; it is NOT why it is trusted. It is checked.

WHY THIS IS A MODULE AND NOT A GATE LEG. The census lived in ``subnormal.py`` and
the WINDOW logic lived inline in one gate's leg 6, with no window concept at all.
Seven gates asking the same question seven different ways is exactly how a family
arrives green having censused nothing, so the window, its floors and its refusal are
one implementation here and every gate imports it.

THE INTERMEDIATE HOOK IS NOT SPECULATIVE. The census must cover RESULTS, not only
operands — measured: on the 1e-38 control the curl's OUTPUT subnormal count (6432)
is nearly double its INPUT count (3233), so the kernel PRODUCES subnormals it was
not given. And one queued family reaches the band by construction rather than by
decay: chi2/chi3 multiplies field by field by field, so ``E^3`` with ``E ~ 1e-13``
lands at 1e-39, INSIDE the band. :func:`nonlinear_intermediate_reasons` is the
plan-time bound that family needs; it is written here so the family inherits the
question rather than inventing an answer.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from .subnormal import census, signed_zero_census  # noqa: F401 - one census, re-exported

#: The largest positive float32 subnormal — the absolute bound on any single cell's
#: error under a flush the reference did not perform. Recorded so an artifact can
#: state the bound it is NOT relying on: the claim is identity under a checked
#: precondition, not "the error is at most 1.18e-38".
MAX_SUBNORMAL = 1.1754942106924411e-38

#: The plan-time bound the nonlinear family's intermediates must satisfy. Measured
#: on the Triton track by emulated flush in NumPy; on Metal the flush is NATIVE, so
#: the crossing must be RE-ESTABLISHED on the device before this number is trusted
#: for this executor. Until that measurement lands the bound is carried with its
#: provenance rather than silently reused.
NONLINEAR_INTERMEDIATE_BOUND = 1e29
NONLINEAR_BOUND_PROVENANCE = (
    "measured on the Triton track in NumPy with an EMULATED flush; the Metal flush "
    "is native, so this crossing must be re-measured through the shader before the "
    "bound is trusted on this executor")


class SubnormalWindow:
    """A census over one run's operands, results and named intermediates.

    ``first_step``/``last_step`` bound the window the claim is made over, because a
    census taken at step 0 of a 1,200-step run certifies step 0. The floors are the
    vacuity discipline: a window that observed NOTHING is not clean, it is empty,
    and :meth:`report` says which.

    ``per_array_words`` and ``per_intermediate_words`` are the counts a caller must
    have observed for the window to be non-vacuous — not thresholds on the subnormal
    count itself, which must be zero.
    """

    __slots__ = ("first_step", "last_step", "per_array_words",
                 "per_intermediate_words", "_arrays", "_intermediates", "_steps")

    def __init__(self, first_step: int = 0, last_step: int = 0,
                 per_array_words: int = 1,
                 per_intermediate_words: int = 0) -> None:
        if last_step < first_step:
            raise ValueError(f"window {first_step}..{last_step} runs backwards")
        self.first_step = int(first_step)
        self.last_step = int(last_step)
        self.per_array_words = int(per_array_words)
        self.per_intermediate_words = int(per_intermediate_words)
        self._arrays: Dict[str, Tuple[int, int]] = {}
        self._intermediates: Dict[str, Tuple[int, int]] = {}
        self._steps: List[int] = []

    def observe(self, label: str, array: Any, step: Optional[int] = None) -> int:
        """Census one operand or result volume. Returns its subnormal word count."""
        import numpy as np  # noqa: PLC0415

        words = int(np.asarray(array).size) * (
            2 if str(getattr(array, "dtype", "")) == "complex64" else 1)
        found = census(array)
        previous = self._arrays.get(label, (0, 0))
        self._arrays[label] = (previous[0] + found, previous[1] + words)
        if step is not None:
            self._steps.append(int(step))
        return found

    def observe_intermediate(self, label: str, array: Any,
                             step: Optional[int] = None) -> int:
        """Census a value that is never stored — a product formed inside the kernel.

        The kernel's own intermediates are not readable from the host, so a caller
        RECONSTRUCTS them on the host (the same expression, same operand order) and
        censuses that. That is weaker than reading the device's registers and the
        artifact must say so; it is strictly stronger than censusing only what was
        stored, which is what a family that multiplies field by field by field needs.
        """
        import numpy as np  # noqa: PLC0415

        words = int(np.asarray(array).size) * (
            2 if str(getattr(array, "dtype", "")) == "complex64" else 1)
        found = census(array)
        previous = self._intermediates.get(label, (0, 0))
        self._intermediates[label] = (previous[0] + found, previous[1] + words)
        if step is not None:
            self._steps.append(int(step))
        return found

    @property
    def subnormal_words(self) -> int:
        return (sum(found for found, _ in self._arrays.values())
                + sum(found for found, _ in self._intermediates.values()))

    @property
    def observed_words(self) -> int:
        return (sum(total for _, total in self._arrays.values())
                + sum(total for _, total in self._intermediates.values()))

    @property
    def clean(self) -> bool:
        return self.subnormal_words == 0

    def vacuity_reasons(self) -> List[str]:
        """Why this window certifies nothing, even at a zero census."""
        reasons: List[str] = []
        if not self._arrays:
            reasons.append("no operand or result volume was censused: an empty "
                           "window reports clean and proves nothing")
        for label, (_, total) in sorted(self._arrays.items()):
            if total < self.per_array_words:
                reasons.append(f"{label} contributed {total} words, below the "
                               f"{self.per_array_words}-word floor")
        if self.per_intermediate_words and not self._intermediates:
            reasons.append(
                "this family declared an intermediate floor and censused no "
                "intermediate; a family whose products can land in the band must "
                "census them, not only what it stored")
        for label, (_, total) in sorted(self._intermediates.items()):
            if total < self.per_intermediate_words:
                reasons.append(f"intermediate {label} contributed {total} words, "
                               f"below the {self.per_intermediate_words}-word floor")
        return reasons

    def report(self) -> Dict[str, Any]:
        """What an artifact records — including the vacuity verdict, always."""
        vacuity = self.vacuity_reasons()
        return {
            "window": [self.first_step, self.last_step],
            "steps_censused": sorted(set(self._steps)),
            "arrays": {k: {"subnormal_words": v[0], "words": v[1]}
                       for k, v in sorted(self._arrays.items())},
            "intermediates": {k: {"subnormal_words": v[0], "words": v[1]}
                              for k, v in sorted(self._intermediates.items())},
            "subnormal_words": self.subnormal_words,
            "observed_words": self.observed_words,
            "clean": self.clean,
            "vacuous": bool(vacuity),
            "vacuity_reasons": vacuity,
            "max_subnormal_magnitude": MAX_SUBNORMAL,
        }


def assert_clean_or_refuse(window: SubnormalWindow, context: str) -> Dict[str, Any]:
    """The precondition, enforced. Returns the report; raises on a failure.

    TWO WAYS TO FAIL AND BOTH ARE FAILURES:

    * the census FIRED — an operand, result or intermediate is in the band, so the
      byte-identity claim does not hold for this case and the gate REFUSES it. Not a
      tolerance, not a warning;
    * the window is VACUOUS — it censused nothing, or less than its own floor. A
      precondition that observed nothing reports clean, and a gate that accepted
      that would be certifying the absence of a measurement.
    """
    report = window.report()
    if report["vacuous"]:
        raise AssertionError(
            f"{context}: the subnormal window is VACUOUS — {report['vacuity_reasons']}. "
            f"A window that censused nothing reports clean and certifies nothing")
    if not report["clean"]:
        raise AssertionError(
            f"{context}: REFUSED by the subnormal precondition — "
            f"{report['subnormal_words']} of {report['observed_words']} censused "
            f"words are in the float32 subnormal band. On MPS the flush is native "
            f"and has no lever, so byte-identity is not claimable here; the honest "
            f"outcomes are a stated tolerance or a refusal, and this is the refusal")
    return report


def demonstrate_firing(window: SubnormalWindow, context: str) -> Dict[str, Any]:
    """The other half: a control that MUST put the window in the band.

    A precondition never demonstrated to fire is decorative. Every gate that carries
    :func:`assert_clean_or_refuse` must also carry a scaled control through this,
    or its precondition leg proves only that the physical band is clean — which was
    never in doubt.
    """
    report = window.report()
    if report["vacuous"]:
        raise AssertionError(f"{context}: the control window is VACUOUS — "
                             f"{report['vacuity_reasons']}")
    if report["clean"]:
        raise AssertionError(
            f"{context}: the census DID NOT FIRE on the control. A precondition "
            f"never demonstrated to fire is decorative, and this control exists to "
            f"demonstrate it")
    return report


def nonlinear_intermediate_reasons(chi_volumes: Sequence[Any], chi1inv: Sequence[Any],
                                   order: int,
                                   bound: float = NONLINEAR_INTERMEDIATE_BOUND,
                                   ) -> List[str]:
    """Plan-time bound on the nonlinear family's un-storable intermediates.

    ``chi2 * |chi1inv|^2 <= bound`` and ``chi3 * |chi1inv|^3 <= bound``, checked from
    the INSTALLED volumes rather than from a nominal material constant, because the
    product that can cross is the one the kernel forms. Returns refusal reasons, so
    a plan builder can fold them into its coverage verdict by name.

    THE BOUND IS CARRIED WITH ITS PROVENANCE (:data:`NONLINEAR_BOUND_PROVENANCE`)
    and is not yet a Metal measurement. A family that uses this must re-measure the
    crossing THROUGH THE SHADER before its gate is cut.
    """
    import numpy as np  # noqa: PLC0415

    if order not in (2, 3):
        raise ValueError(f"order must be 2 (chi2) or 3 (chi3), got {order!r}")
    reasons: List[str] = []
    peak_inverse = 0.0
    for volume in chi1inv:
        if volume is None:
            continue
        peak_inverse = max(peak_inverse, float(np.max(np.abs(np.asarray(volume)))))
    for index, volume in enumerate(chi_volumes):
        if volume is None:
            continue
        peak = float(np.max(np.abs(np.asarray(volume))))
        product = peak * (peak_inverse ** order)
        if not (product <= bound):
            reasons.append(
                f"chi{order} component {index} reaches {peak:.3e} and |chi1inv| "
                f"reaches {peak_inverse:.3e}, so the kernel forms an intermediate up "
                f"to {product:.3e}, above the {bound:.3e} bound "
                f"({NONLINEAR_BOUND_PROVENANCE})")
    return reasons
