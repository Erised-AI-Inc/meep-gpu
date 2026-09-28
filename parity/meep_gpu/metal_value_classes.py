"""The float32 VALUE CLASSES the Metal fused-product gates seed and score.

WHY THIS MODULE EXISTS, and it is a gap in the record rather than a refactor.
Every Metal fused-product gate seeds one value class — a physical-band
``standard_normal * 0.37`` — and then reports ``reference_subnormals: 0`` and
``subnormal_free: true`` under policy ``flush`` on every case it runs. Read from
the artifact alone that is a SILENCE, not a measurement: a reader cannot tell
whether the flush policy was exercised and found harmless, or whether the band
was never entered so the policy was never asked anything. Every Triton and CUDA
gate in this tree carries a second value class; the Metal fused half did not.

WHAT THIS MODULE SUPPLIES, and what it deliberately does not. It supplies the two
things that are a property of the BACKEND rather than of any one family:

* :func:`pm_zero_lattice` — the +-0 lattice, and :func:`zero_sign_census`, the
  floor that makes comparing it discriminating rather than decorative;
* :data:`PRECONDITION_SCALES` and :func:`scaled_normal` — the ladder that walks a
  seeded state from the physical band down through the subnormal one, so the
  census a gate already reports is shown to FIRE somewhere.

It does NOT supply the seeding of any family's own arrays, the compared set, the
case matrix, or any verdict. Those are the family's claim; a kit that supplied
them would let a family inherit a claim it never made (``metal_gate_kit`` states
the same boundary and this module keeps it).

THE MEASUREMENT THIS MODULE IS BUILT ON, taken on this host rather than recalled.
On the fused magnetic pair (``wall_xyz_3d``, four complete driver steps, the
shipped kernel against the array-path oracle, uint32 word compare):

    scale    label            census (ref)   differing words, step 1..4
    1e+00    physical                    0   0, 0, 0, 0
    1e-25    small_normal                0   0, 0, 0, 0
    1e-30    band_edge                   0   0, 0, 0, 0
    1e-34    subnormal_band             27   1036, 7638, 19077, 30688
    1e-41    deep_subnormal         115575   112031, 112042, 112032, 111964
    (+-0 lattice)                        0   0, 0, 0, 0   [45156 words moved,
                                             15324 negative zeros in the output]

So on MPS the subnormal band is NOT a second byte-identity comparison and cannot
be made into one: the executor's flush is native and has no lever
(``meep_gpu/metal_kernels/subnormal.py``), the NumPy oracle keeps, and the two
diverge as soon as a subnormal exists. That is a REFUSAL, and this module's job is
to make each gate state it WITH the numbers rather than leave the empty census to
be read as a passed policy. The +-0 lattice is the value class that IS reachable,
and it is a real second comparison: the state moves, and the reference output
carries both signs of zero, so a uint32 compare there is discriminating a sign a
float compare would call equal.

RULE 4 OF THE FIVE PAID-FOR TRAPS, RESTATED FOR THIS FILE: a zero state is a fixed
point of some seams and not of others. :func:`pm_zero_lattice` is therefore never
exempt from a floor — it swaps the MOVEMENT floor for the SIGN floor, and a gate
that finds neither must fail rather than report a vacuous pass.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "PRECONDITION_SCALES",
    "UNIFORM",
    "PM_ZERO_LATTICE",
    "SUBNORMAL_BAND",
    "VALUE_CLASSES",
    "pm_zero_lattice",
    "pm_zero_lattice_complex",
    "scaled_normal",
    "zero_sign_census",
    "band_refusal_note",
    "precondition_verdict",
    "census_ladder",
]

#: The physical band every gate already seeded. Named so an artifact row can say
#: which class it ran rather than leaving it implied by absence.
UNIFORM = "uniform"

#: The second COMPARED class: a checkerboard of +0.0 and -0.0.
PM_ZERO_LATTICE = "pm_zero_lattice"

#: The third class, and it is REFUSED rather than compared. Kept as a name so the
#: refusal appears in the record beside the two that ran.
SUBNORMAL_BAND = "subnormal_band"

#: The classes a fused-product gate scores, in the order it should score them.
VALUE_CLASSES: Tuple[str, ...] = (UNIFORM, PM_ZERO_LATTICE)

#: Scale, and what the per-step subnormal census MUST do at it: ``True`` stay
#: clean, ``False`` fire, ``None`` either (the cliff edge, which is a fixture
#: property and not a claim). The CLEAN rows are what make the firing rows mean
#: something — a census that fired on everything would fire on the physical band
#: too and would certify nothing.
#:
#: The factors are the ones the cliff was measured at; the values are transcribed
#: from that measurement (see this module's docstring) rather than chosen.
PRECONDITION_SCALES: Tuple[Tuple[str, float, Optional[bool]], ...] = (
    ("physical", 1.0, True),
    ("small_normal", 1e-25, True),
    ("band_edge", 1e-30, None),
    ("subnormal_band", 1e-34, False),
    ("deep_subnormal", 1e-41, False),
)


def scaled_normal(rng: Any, shape: Sequence[int], amplitude: float,
                  scale: float) -> np.ndarray:
    """The gate's own seeding, times a scale. ONE multiply and nothing else.

    The scale is applied to the float64 draw and the result cast once, so the
    ladder walks the SAME state down the exponent range rather than drawing a
    different one at each rung — which is what makes "the census fired at 1e-34
    and not at 1e-30" a statement about the band and not about two samples.
    """
    return (rng.standard_normal(tuple(shape)) * amplitude * scale).astype(np.float32)


def pm_zero_lattice(shape: Sequence[int]) -> np.ndarray:
    """A checkerboard of ``+0.0`` and ``-0.0`` over ``shape``.

    Even flat index positive, odd negative. The alternation is over the FLAT index
    so every axis of every stride pattern sees both signs, including the ghost
    planes a fold or a wall reaches, which a per-axis alternation would leave
    single-signed on whichever axis it did not alternate along.
    """
    shape = tuple(int(value) for value in shape)
    flat = np.arange(int(np.prod(shape)), dtype=np.int64).reshape(shape)
    return np.where(flat % 2 == 0, np.float32(0.0), np.float32(-0.0)
                    ).astype(np.float32)


def pm_zero_lattice_complex(shape: Sequence[int]) -> np.ndarray:
    """The +-0 lattice for a complex64 state, with the two planes OPPOSED.

    The real plane gets the lattice and the imaginary plane its negation, so at
    every cell the two planes hold OPPOSITE signs of zero. That matters on the
    complex families for the reason their own gates give: a helper that folded
    ``z_im * 0.0f`` to a literal is only observable where the imaginary plane's
    zero has a sign to lose, and a lattice that gave both planes the same sign at
    each cell would leave half the cross terms unexercised.

    The planes are assigned through the array's own ``.real`` / ``.imag`` views
    rather than built with ``a + 1j * b``: the multiply-and-add form goes through
    float arithmetic that does not reliably carry a negative zero through, and the
    whole content of this class is the sign.
    """
    lattice = pm_zero_lattice(shape)
    out = np.zeros(tuple(int(value) for value in shape), dtype=np.complex64)
    out.real = lattice
    out.imag = -lattice
    return out


def zero_sign_census(arrays: Iterable[Any]) -> Dict[str, int]:
    """How many ``+0.0`` and ``-0.0`` WORDS a state holds, by uint32 bit pattern.

    THE FLOOR THE LATTICE CLASS ANSWERS TO. ``0.0 == -0.0`` in float, so a value
    class made of zeros is only worth comparing if the comparison can see the sign
    — and this gate family compares uint32 words, which can. Requiring BOTH counts
    nonzero in the REFERENCE OUTPUT is what turns "we also ran a zero case" into a
    measurement: it says the oracle produced both bit patterns and the device
    reproduced each one where it stood.
    """
    positive = negative = other = 0
    for array in arrays:
        words = np.frombuffer(np.ascontiguousarray(array).tobytes(),
                              dtype=np.uint32)
        positive += int(np.count_nonzero(words == np.uint32(0x00000000)))
        negative += int(np.count_nonzero(words == np.uint32(0x80000000)))
        other += int(words.size)
    return {"positive_zero_words": positive, "negative_zero_words": negative,
            "total_words": other}


def band_refusal_note() -> str:
    """The one-line reason the band is refused, for the artifact row."""
    return ("MPS flushes float32 subnormals natively with no source-level or "
            "environment lever (meep_gpu/metal_kernels/subnormal.py:6-13) while "
            "the NumPy oracle keeps them, so a seeded subnormal band is a "
            "MEASURED divergence rather than a second byte-identity comparison; "
            "this family's claim is byte identity under a CHECKED subnormal-free "
            "precondition, and the ladder below is that precondition firing")


def census_ladder(build_at_scale: Any, step_once: Any, census_of: Any,
                  budget: int = 4, log: Any = None) -> Dict[str, Any]:
    """Walk ONE family's oracle down the exponent range and score the census.

    THE FAMILY SUPPLIES THE THREE THINGS THAT ARE ITS OWN and this function
    supplies nothing but the ladder:

    * ``build_at_scale(scale)`` -> whatever ``step_once`` takes. The gate's own
      builder with its own seeding, times the scale — so the rungs are the SAME
      state walked down, not five different draws;
    * ``step_once(engine)`` -> one complete driver step on the ARRAY PATH. The
      oracle alone; nothing on the device runs here, because a banded rung is a
      refusal and not a comparison;
    * ``census_of(engine)`` -> the subnormal word count the gate's product rows
      already report. The same function, so this leg scores the detector those
      rows depend on rather than a second one that might agree by accident.

    Returns :func:`precondition_verdict`'s shape.
    """
    rows: List[Dict[str, Any]] = []
    for label, scale, expect_clean in PRECONDITION_SCALES:
        engine = build_at_scale(scale)
        per_step: List[int] = []
        for _ in range(budget):
            step_once(engine)
            per_step.append(int(census_of(engine)))
        fired = [index for index, count in enumerate(per_step) if count]
        agrees = (expect_clean is None
                  or (expect_clean and not fired)
                  or (not expect_clean and bool(fired)))
        rows.append({"scale": label, "factor": scale,
                     "subnormal_words_per_step": per_step,
                     "census_fired": bool(fired),
                     "first_step": fired[0] + 1 if fired else None,
                     "expected_clean": expect_clean, "agrees": agrees})
        if log is not None:
            log(f"    ladder {label} factor={scale:g} fired={bool(fired)} "
                f"census={per_step} expected_clean={expect_clean}")
    return precondition_verdict(rows)


def precondition_verdict(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Score a census ladder: every rung must do what its scale says it must.

    A ladder on which nothing fired is a census that has never been shown to work,
    and a ladder on which everything fired is a census that cannot discriminate.
    Both are failures here, and both are named separately so the artifact says
    which one happened.
    """
    rows = list(rows)
    agrees = all(bool(row.get("agrees")) for row in rows)
    fired = [row["scale"] for row in rows if row.get("census_fired")]
    clean = [row["scale"] for row in rows if not row.get("census_fired")]
    return {
        "passed": bool(rows and agrees and fired and clean),
        "rows": [dict(row) for row in rows],
        "scales_where_the_census_fired": fired,
        "scales_where_the_census_stayed_clean": clean,
        "detector_is_discriminating": bool(fired and clean),
        "banded_rows_were_byte_compared": False,
        "note": band_refusal_note(),
    }
