"""THE BYTE GATE for the Metal chi2/chi3 (Pade) ``update_E`` family.

The kernel steps the array path's forms: the four corners left to right, ``Dsqr``'s
``0.0625`` term and the addition after it in software binary64 rounded once (ruling
R7: soft-fp64, ``stepping._nonlinear_dsqr``), and the constitutive form per structure
chunk from the chunk record every plan here binds (``layer_record``, the one-rank
division the reference ``Fields`` steps with). The reference runs with the host's FPU
in the flush policy, the only one the executor runs (:func:`flushed_host`).

WHAT IT CERTIFIES: that the shipped kernel reproduces ``stepping.update_E``
(stepping.py:954, the ``if nonlinear:`` branch at :999-1000) WORD FOR WORD as
uint32, over the configuration space the family's predicate admits, and that the
transcription choices which decide those words are byte-VISIBLE rather than
merely believed — each one carried as an armed mutation that must be CAUGHT.

WHAT IT DOES NOT CERTIFY, stated rather than papered over:

* **it is behavioural, not generated-code evidence.** ``torch.mps.compile_shader``
  exposes no AIR, no GPU ISA and no optimisation report (``device.py:46-62``), so
  this gate catches a wrong ANSWER, never a wrong INSTRUCTION. That is this
  backend's standing certification gap against the Triton and CUDA tracks and it
  stays stated on every claim built on top of it;
* **one sub-step, not a complete step.** The whole-step arbiter is
  ``gate_metal_whole_step.py``. A stale mirror, a seam and an accumulating
  auxiliary are all invisible to a single-launch gate by construction — that is
  precisely why the off-diagonal family's source-alias defect was byte-identical
  for one sub-step and 19,620 words wrong over two. THE CENSUS LEG IS THE ONE
  EXCEPTION and is multi-step deliberately, because band entry is a RUN fact;
* **it runs under the FLUSH policy and nothing else is offerable.** Metal flushes
  subnormals and exposes no lever, so every claim here rides on the CHECKED
  subnormal-free precondition;
* **a wholesale transverse-partner swap is INVISIBLE to any byte gate**, and that
  is measured here rather than left to be discovered — see the
  ``partner_squares_order_swapped`` null control.

THE LEGS:

1. ``identity`` — the shipped plan against the array path over a swept matrix:
   3 boundary configurations x 3 component masks x 2 chi arms x 3 value classes.
   Words, never ``allclose``; a vacuity floor on every case, because zero-init is a
   FIXED POINT of the constitutive sub-step and a no-op agreeing with a no-op is
   trivially identical.
2. ``linear_reduction`` — a partly nonlinear run's LINEAR components against the
   CERTIFIED plain constitutive kernel, byte for byte. This is what makes MEEP's
   ``else if (u)`` seam (stepping.py:1100-1102) a measurement instead of a claim.
3. ``mutations`` — the transcription choices, each planted and each required to be
   caught, plus NULL controls whose whole job is to fail to be caught. Every
   mutation carries the boundary set it is ARMED on; on the complement it is run
   as a PREDICTED NULL and required NOT to fire, so "this defect cannot show under
   a periodic ghost" is a measurement rather than an excuse. The previous package
   forms are must-catch mutations here (KERNELS.md §7.1): the paired corners
   (``four_corner_pair_association``), the two-step accumulation
   (``previous_two_step_accumulation``) and the float32 ``Dsqr``
   (``dsqr_in_float32``, armed on the ``dsqr_band`` class); the §7.2 B battery of the
   per-chunk tail rides with them, two of its entries planted in the RECORD rather
   than the source.
4. ``ceiling`` — the plan-time magnitude clause at 1e29, pinned in both
   directions: a configuration under it is admitted AND byte-identical, one over
   it is REFUSED BY NAME. Both sides carry a non-vacuity check on the MEASURED
   per-cell magnitude, because a nominal chi says nothing about the product the
   kernel forms.
5. ``census`` — the subnormal precondition, over a REAL multi-step array-path run,
   covering the un-storable INTERMEDIATES and not only what was stored. Reported
   as a WINDOW ``[first_step, last_step]``. A case that reaches the band is
   REFUSED BY NAME and its comparisons are withheld. Carries the firing control
   ``preconditions`` requires, because a precondition never demonstrated to fire
   is decorative.
6. ``dsqr`` — the binary64 helper alone, launched on its band (``gs*gs`` normal,
   ``0.0625*(g1s*g1s + g2s*g2s)`` below float32's smallest normal) and at field
   scale, against ``stepping._nonlinear_dsqr`` under the flush policy, with the
   float32 form as the control that must differ on the band and nowhere else; then
   the shipped kernel on a ``dsqr_band`` state per boundary configuration, with the
   number of source words the float32 form would have moved as its vacuity floor.
7. ``spine`` — the three sub-steps that do **not** read chi2/chi3,
   ``step_B -> update_H -> step_D``.  They deliberately reuse the separately
   certified ordinary PML shaders under nonlinear-only admissions.  This leg
   proves that the new admissions build, launch, retain their shared mirrors, and
   reproduce the engine in a nonlinear material context; it does not pretend to
   re-certify their already-gated shader arithmetic.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
API_ROOT = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
if API_ROOT not in sys.path:
    sys.path.insert(0, API_ROOT)

# The gate runs under the only policy this executor can honour. Set BEFORE the
# package is imported, because the predicate reads it.
os.environ.setdefault("MEEP_GPU_SUBNORMAL_POLICY", "flush")

import metal_gate_kit as kit  # noqa: E402
from gate_chunk_records import layer_record  # noqa: E402

from meep_gpu import backends, stepping, subnormal_policy  # noqa: E402
from meep_gpu.chunk_map import record_index  # noqa: E402
from meep_gpu.fields import Fields  # noqa: E402
from meep_gpu.grid import Grid  # noqa: E402
from meep_gpu.metal_kernels import (  # noqa: E402
    launch,
    nonlinear_update_e as nonlinear,
    preconditions,
    shaders,
    softfp64,
    subnormal,
)
from meep_gpu.metal_kernels.device import Residency, compile_source  # noqa: E402
from meep_gpu.pml import PML  # noqa: E402

WATCHED = ("Ex", "Ey", "Ez", "f_w_Ex", "f_w_Ey", "f_w_Ez")
SEEDED = ("Ex", "Ey", "Ez", "Dx", "Dy", "Dz", "f_w_Ex", "f_w_Ey", "f_w_Ez")

#: The three boundary configurations. ``mixed`` is what catches a per-axis ghost
#: rule applied on the wrong axis — an all-metallic and an all-periodic case
#: cannot, because every axis answers the same way in both.
BOUNDARIES: Tuple[Tuple[str, Any], ...] = (
    ("periodic", "periodic"),
    ("metallic", "metallic"),
    ("mixed", ("periodic", "metallic", "periodic")),
)

#: Component masks. ``one`` is the corpus row's shape (Ez alone); ``all`` exercises
#: every partner pairing at once; ``two`` is the partly nonlinear seam.
MASKS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("one", ("Ez",)),
    ("two", ("Ex", "Ez")),
    ("all", ("Ex", "Ey", "Ez")),
)

#: How many complete array-path cycles the census window walks. Band entry was
#: measured to be a RUN-and-WINDOW fact rather than a family fact, so a census at
#: step 0 certifies step 0 and nothing else.
CENSUS_STEPS = 8

#: The ``dsqr_band`` class: ``|Dz| ~ 2**BAND_GS_EXPONENT`` and the partner volumes
#: Dx, Dy ~ ``2**BAND_PARTNER_EXPONENT`` and same-signed, so each four-corner sum is
#: ``2**-63 .. 2**-62``, its square a normal float, and ``0.0625 * (g1s*g1s +
#: g2s*g2s)`` below float32's smallest normal: the band where the float32 and the
#: binary64 ``Dsqr`` differ under the flush policy. ``chi3`` is solved per cell for
#: ``chi3 * |chi1inv|^3 = BAND_CHI3_MAGNITUDE``, under the 1e29 ceiling, which is
#: what carries the ``Dsqr`` change into ``c3`` and the ``1 + ...`` sums.
BAND_GS_EXPONENT = -57
BAND_PARTNER_EXPONENT = -65
BAND_CHI3_MAGNITUDE = 9e28


class flushed_host:
    """``with flushed_host():`` drives this thread's FPU to the flush policy and puts
    back the mode it found. The reference runs inside it: the Metal executor flushes,
    so the array path it must equal is the one a flushing host computes, and a Metal
    plan's freeze leaves the thread flushing, so the mode is set here rather than
    inherited from whichever leg ran last."""

    def __enter__(self) -> "flushed_host":
        self.before = backends.subnormals_flushed()
        subnormal_policy._drive_host_fpu(True)  # noqa: SLF001
        assert backends.subnormals_flushed(), "the host FPU could not be driven to flush"
        return self

    def __exit__(self, *exc: Any) -> None:
        subnormal_policy._drive_host_fpu(bool(self.before))  # noqa: SLF001


# ---------------------------------------------------------------------------
# uint32 word accounting
# ---------------------------------------------------------------------------

class Tally:
    """Words compared and words disagreeing, for the artifact's headline.

    THE COUNT THIS GATE REPORTS IS WORDS, NOT ROWS. A row count says how many
    configurations ran; it says nothing about how much of each state was actually
    looked at, and two gates quoting "78" can mean quantities three orders of
    magnitude apart. Every comparison this class mediates is uint32 equality on the
    whole volume, so the total is the number of individual words that had to agree.
    """

    __slots__ = ("compared", "differing", "withheld")

    def __init__(self) -> None:
        self.compared = 0
        self.differing = 0
        self.withheld = 0

    def compare(self, left: Any, right: Any) -> int:
        """One uint32 word comparison over a whole volume. Returns words differing."""
        count = int(kit.words(left).size)
        differing = kit.differing(left, right)
        self.compared += count
        self.differing += differing
        return differing

    def compare_state(self, state: Dict[str, Any], reference: Dict[str, Any],
                      names: Sequence[str] = WATCHED) -> int:
        return sum(self.compare(state[name], reference[name]) for name in names)

    def report(self) -> Dict[str, int]:
        return {"uint32_words_compared": self.compared,
                "uint32_words_differing": self.differing,
                "uint32_words_withheld": self.withheld}


TALLY = Tally()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def build(boundaries: Any, components: Sequence[str], scalar: bool,
          value_class: str, seed: int = 17,
          chi2_magnitude: float = 0.011,
          chi3_magnitude: float = 0.023) -> Tuple[Any, Any]:
    """A real ``Fields``/``PML`` pair, seeded in a named value class.

    ZERO INIT IS A FIXED POINT of this sub-step, so every class fills the volumes
    with values that actually move state; ``assert_moved`` is what turns that from
    an intention into a check.
    """
    grid = Grid(resolution=10.0, cell_size=(1.2, 1.0, 0.9), boundaries=boundaries,
                dimensions=3, courant=0.35, k_point=(0.0, 0.0, 0.0), xp=np)
    fields = Fields(grid=grid, force_complex_fields=False)
    shape = tuple(grid.shape)
    rng = np.random.default_rng(seed)

    inverse = (0.35 + 0.5 * rng.random(shape)).astype(np.float32)
    epsilon = (1.0 / inverse).astype(np.float32)
    fields.set_epsilon_volumes({c: epsilon for c in ("Ex", "Ey", "Ez")},
                               {c: inverse for c in ("Ex", "Ey", "Ez")})
    fields.enable_pml_storage()

    for name in SEEDED:
        if value_class == "random":
            values = rng.standard_normal(shape)
        elif value_class == "signed_zero":
            # Half the lanes exactly +/- 0.0. The wall plane a metallic run clears
            # is exactly this, and c2 carries D LINEARLY, so a signed zero reaches
            # the Pade numerator rather than being an abstract worry.
            values = rng.standard_normal(shape)
            picks = rng.random(shape) < 0.5
            values[picks] = 0.0
            flip = picks & (rng.random(shape) < 0.5)
            values[flip] = -0.0
        elif value_class == "large_u":
            # Amplitudes that drive u well away from 1, so the quotient's own
            # arithmetic is exercised rather than only the 1 + epsilon band.
            values = 12.0 * rng.standard_normal(shape)
        elif value_class == "dsqr_band":
            # See BAND_GS_EXPONENT. E and f_w stay at field scale.
            if name == "Dz":
                values = (np.ldexp(rng.uniform(1.0, 2.0, shape), BAND_GS_EXPONENT)
                          * rng.choice((-1.0, 1.0), shape))
            elif name in ("Dx", "Dy"):
                values = np.ldexp(rng.uniform(1.0, 2.0, shape), BAND_PARTNER_EXPONENT)
            else:
                values = rng.standard_normal(shape)
        else:
            raise ValueError(value_class)
        getattr(fields, name)[...] = values.astype(np.float32)

    def chi(magnitude: float) -> Any:
        if scalar:
            return np.float32(magnitude)
        return (magnitude * (0.5 + rng.random(shape))).astype(np.float32)

    if value_class == "dsqr_band":
        # Solved in float64 against the installed epsilon and rounded once, as
        # build_at_magnitude does; a scalar arm takes the smallest |chi1inv|'s value.
        solved = BAND_CHI3_MAGNITUDE / np.abs(inverse.astype(np.float64)) ** 3
        third = (np.float32(BAND_CHI3_MAGNITUDE / float(np.max(np.abs(inverse))) ** 3)
                 if scalar else solved.astype(np.float32))
        fields.set_nonlinear_volumes({c: chi(chi2_magnitude) for c in components},
                                     {c: third for c in components})
    else:
        fields.set_nonlinear_volumes({c: chi(chi2_magnitude) for c in components},
                                     {c: chi(chi3_magnitude) for c in components})
    # A SEEDED STATE IS A MID-RUN ONE: H was started from B long ago
    # (stepping.separate_h_from_b), so the reference's update_H does not start it.
    fields.h_separated = True
    return fields, PML(grid=grid, thickness=tuple((2, 2) for _ in range(3)))


def build_at_magnitude(boundaries: Any, components: Sequence[str],
                       target: float, order: int, seed: int = 17
                       ) -> Tuple[Any, Any]:
    """A pair whose MEASURED per-cell ``|chi| * |chi1inv|^order`` is exactly ``target``.

    THE CEILING LEG CANNOT USE A NOMINAL CHI, and that is not a stylistic
    preference — it is the trap the previous round fell into. A composition row
    built at ``chi3 = 1e30`` with ``chi1inv = 1/3`` forms ``1e30 * (1/3)^3 =
    3.7e28``, which is UNDER the 1e29 ceiling: the row meant to exercise a refusal
    exercised an admission instead, and only a sweep that recomputed the product
    caught it. So the chi volume is SOLVED FOR from the epsilon volume,
    ``chi = target / |chi1inv|^order`` per cell, which makes the quantity the clause
    actually reads exactly ``target`` everywhere rather than approximately it
    somewhere.

    The OTHER order is held at a physical magnitude so exactly one clause is under
    test: a case where both cross tells you a refusal happened, not which clause
    refused.
    """
    grid = Grid(resolution=10.0, cell_size=(1.2, 1.0, 0.9), boundaries=boundaries,
                dimensions=3, courant=0.35, k_point=(0.0, 0.0, 0.0), xp=np)
    fields = Fields(grid=grid, force_complex_fields=False)
    shape = tuple(grid.shape)
    rng = np.random.default_rng(seed)

    inverse = (0.35 + 0.5 * rng.random(shape)).astype(np.float32)
    epsilon = (1.0 / inverse).astype(np.float32)
    fields.set_epsilon_volumes({c: epsilon for c in ("Ex", "Ey", "Ez")},
                               {c: inverse for c in ("Ex", "Ey", "Ez")})
    fields.enable_pml_storage()
    for name in SEEDED:
        getattr(fields, name)[...] = rng.standard_normal(shape).astype(np.float32)
    fields.h_separated = True  # A seeded state is a mid-run one.

    # Solved in float64 and rounded once: the clause reads float64 (the family's
    # _magnitude is float64 deliberately, because chi3 * |chi1inv|^3 at the ceiling
    # would overflow float32 while being computed).
    solved = (target / (np.abs(inverse.astype(np.float64)) ** order)).astype(np.float32)
    physical = (0.017 * (0.5 + rng.random(shape))).astype(np.float32)
    if order == 2:
        chi2, chi3 = solved, physical
    else:
        chi2, chi3 = physical, solved
    fields.set_nonlinear_volumes({c: chi2 for c in components},
                                 {c: chi3 for c in components})
    return fields, PML(grid=grid, thickness=tuple((2, 2) for _ in range(3)))


def measured_magnitude(fields: Any, component: str, order: int) -> float:
    """What the clause itself computes, read back through the family's own helper.

    Read through ``nonlinear._magnitude`` rather than recomputed here, so a case
    claiming to sit above the ceiling is measured by the SAME expression the
    predicate will use. A gate that computed its own magnitude could build a case
    it believed was over the line while the clause saw it under.
    """
    chi2, chi3 = nonlinear.chi_pair_for(fields, component)
    chi = chi2 if order == 2 else chi3
    value = nonlinear._magnitude(chi, fields.inverse_epsilon_for(component), order)
    assert value is not None, (
        f"the family's own magnitude helper could not evaluate {component} at "
        f"order {order}; a case whose magnitude is unreadable pins nothing")
    return value


def array_path_reference(fields: Any, pml: Any) -> Tuple[Dict[str, Any],
                                                         Dict[str, Any], int]:
    """Run ``stepping.update_E`` on a flushing host and return (before, after, moved)."""
    before = {n: getattr(fields, n).copy() for n in WATCHED}
    with flushed_host():
        stepping.update_E(fields, pml)
    after = {n: getattr(fields, n).copy() for n in WATCHED}
    moved = sum(kit.differing(after[n], before[n]) for n in WATCHED)
    for name, values in before.items():
        getattr(fields, name)[...] = values
    return before, after, moved


def drive_one_cycle(fields: Any, pml: Any) -> None:
    """ONE COMPLETE array-path step, in the driver's own order.

    Transcribed from ``driver.py:3281-3302``: the ten passes of a step, minus the
    source injections this fixture has none of and ``update_P``, which no pole is
    registered for. The fill and wall passes are no-ops on an unfolded Cartesian
    grid, and they are called anyway — the census is a statement about the run the
    family would be used in, and dropping the passes because they are believed
    inert would make it a statement about a shorter step instead.

    THE CENSUS NEEDS THIS AND THE IDENTITY LEG MUST NOT USE IT. Repeating
    ``update_E`` alone with D held fixed does not evolve the system: ``src`` depends
    only on D and epsilon, so ``f_w`` is constant after the first call and E
    accumulates a fixed increment. A window taken over that would sweep a state the
    engine never visits.
    """
    stepping.step_B(fields, pml)
    stepping.fill_symmetry_bc_B(fields)
    stepping.zero_metal_B(fields)
    stepping.fill_folded_far_ghosts_B(fields)
    stepping.update_H(fields, pml)
    stepping.step_D(fields, pml)
    stepping.fill_symmetry_bc_D(fields)
    stepping.zero_metal_D(fields)
    stepping.fill_folded_far_ghosts_D(fields)
    stepping.update_E(fields, pml)


def nonlinear_intermediates(fields: Any, pml: Any) -> Dict[str, Any]:
    """Every un-storable value the Pade body forms, reconstructed on the host.

    DRIVEN THROUGH ``stepping``'S OWN HELPERS wherever one exists —
    ``_nonlinear_displacement`` (:1030), ``_nonlinear_transverse_sums`` (:1121) and
    ``_nonlinear_dsqr`` (:1100) — rather than re-transcribed here. A gate that
    re-transcribed the oracle would be censusing its own second opinion, and a
    disagreement between the two transcriptions would show up as a clean census
    rather than as a failure.

    The four values INSIDE ``calc_nonlinear_u`` have no helper of their own, so they
    are transcribed from its body (stepping.py:1145-1147) term for term, with the
    association Python gives them::

        c2  = (di * chi2) * (chi1inv * chi1inv)
        c3  = (dsqr * chi3) * (chi1inv * chi1inv * chi1inv)
        u   = ((1 + c2) + 2*c3) / ((1 + 2*c2) + 3*c3)

    THIS IS THE LEG THE FAMILY EXISTS TO HAVE. ``preconditions``' own docstring
    names chi2/chi3 as the one queued family that reaches the subnormal band BY
    CONSTRUCTION rather than by decay — it multiplies field by field by field, so
    ``E^3`` at ``E ~ 1e-13`` lands at 1e-39, inside the band — and a census that
    covered only the stored E and f_w would look at the one part of this family
    that cannot get there.
    """
    displacement = stepping._nonlinear_displacement(fields, pml)
    out: Dict[str, Any] = {}
    for component, _source, _axis in nonlinear.E_TERMS:
        if not fields.is_nonlinear(component):
            continue
        gs = displacement["volumes"][component]
        us = fields.inverse_epsilon_for(component)
        chi2 = fields.chi2_for(component)
        chi3 = fields.chi3_for(component)
        first, second = stepping._nonlinear_transverse_sums(fields, component,
                                                            displacement)
        dsqr = stepping._nonlinear_dsqr(gs, (first, second), stepping._WHOLE)
        us_sq = us * us
        us_cu = (us * us) * us
        c2 = (gs * chi2) * us_sq
        c3 = (dsqr * chi3) * us_cu
        num = (1 + c2) + 2 * c3
        den = (1 + 2 * c2) + 3 * c3
        u = num / den
        row = gs * us
        out.update({
            f"g1s:{component}": first, f"g2s:{component}": second,
            f"dsqr:{component}": dsqr,
            f"us_sq:{component}": us_sq, f"us_cu:{component}": us_cu,
            f"c2:{component}": c2, f"c3:{component}": c3,
            f"numerator:{component}": num, f"denominator:{component}": den,
            f"u:{component}": u,
            f"row:{component}": row, f"src:{component}": row * u,
        })
    return out


def harness_arrays(fields: Any, pml: Any) -> Tuple[Dict[str, Any],
                                                   Dict[str, Any],
                                                   Dict[str, Dict[str, Any]]]:
    """The bare-array view ``plan_..._from_arrays`` takes — the mutation route."""
    arrays: Dict[str, Any] = {}
    for component, source, _axis in nonlinear.E_TERMS:
        arrays[component] = getattr(fields, component)
        arrays["f_w_" + component] = getattr(fields, "f_w_" + component)
        arrays[source] = fields.displacement_minus_polarization_volumes()[component]
        arrays["inv_eps_" + component] = fields.inverse_epsilon_for(component)
    flat = {f"{stem}_{axis}": getattr(pml, f"{stem}_{axis}_h")
            for axis in "xyz" for stem in ("kps", "kms")}
    chi: Dict[str, Dict[str, Any]] = {}
    for component, _source, _axis in nonlinear.E_TERMS:
        if fields.is_nonlinear(component):
            chi[component] = {"chi2": fields.chi2_for(component),
                              "chi3": fields.chi3_for(component)}
    return arrays, flat, chi


def run_plan(plan: Any, residency: Any, fields: Any) -> None:
    with flushed_host():
        residency.sync_in()
        plan.run()
        residency.sync_out()


def record_of(fields: Any, pml: Any) -> Any:
    """The chunk record a plan here binds: the reference's own one-rank division."""
    return layer_record(fields.grid, pml)


# ---------------------------------------------------------------------------
# Leg 1: identity
# ---------------------------------------------------------------------------

def leg_identity(payload: Dict[str, Any], out: str) -> None:
    rows: List[Dict[str, Any]] = []
    payload["legs"]["identity"] = rows
    compared = 0
    for boundary_label, boundaries in BOUNDARIES:
        for mask_label, components in MASKS:
            for scalar in (False, True):
                for value_class in ("random", "signed_zero", "large_u"):
                    fields, pml = build(boundaries, components, scalar,
                                        value_class)
                    verdict = nonlinear.nonlinear_constitutive_coverage(
                        fields, pml, Residency())
                    assert verdict.covered, (boundary_label, mask_label,
                                             verdict.reasons)
                    _before, reference, moved = array_path_reference(fields, pml)
                    kit.assert_moved(
                        moved, f"{boundary_label}/{mask_label}/{value_class}")

                    residency = Residency()
                    plan = nonlinear.plan_nonlinear_constitutive(
                        fields, pml, residency, chunk_bounds=record_of(fields, pml))
                    assert plan is not None
                    run_plan(plan, residency, fields)
                    assert plan.launches == 1

                    state = {n: getattr(fields, n) for n in WATCHED}
                    differing = TALLY.compare_state(state, reference)
                    row = {
                        "boundaries": boundary_label, "mask": mask_label,
                        "chi_arm": "scalar" if scalar else "volume",
                        "value_class": value_class,
                        "nonlinear": list(plan.nonlinear),
                        "boundary_codes": list(plan.boundary_codes),
                        "moved_words": moved, "differing": differing,
                        "words_compared": int(sum(kit.words(state[n]).size
                                                  for n in WATCHED)),
                        "digest": kit.state_digest(reference),
                    }
                    rows.append(row)
                    compared += 1
                    payload["legs"]["identity"] = rows
                    kit.save(payload, out)
                    kit.log(f"[identity] {boundary_label:<9} {mask_label:<4} "
                            f"{'scalar' if scalar else 'volume':<7} "
                            f"{value_class:<12} moved={moved:<7} "
                            f"differing={differing}")
                    assert differing == 0, row
    payload["identity_cases"] = compared


# ---------------------------------------------------------------------------
# Leg 2: the linear-component reduction
# ---------------------------------------------------------------------------

def leg_linear_reduction(payload: Dict[str, Any], out: str) -> None:
    """A LINEAR component in a partly nonlinear run must be byte-identical to the
    CERTIFIED plain constitutive kernel's output for that component.

    MEEP's ``else if (u)`` branch (stepping.py:1100-1102) is ``gs * us``, and the
    emitter compiles exactly that. If it did not, the divergence would be confined
    to components nobody was looking at and every nonlinear-component check would
    still pass — which is why this is its own leg rather than a corollary.
    """
    from meep_gpu.metal_kernels.launch import plan_constitutive

    rows: List[Dict[str, Any]] = []
    payload["legs"]["linear_reduction"] = rows
    for boundary_label, boundaries in BOUNDARIES:
        fields, pml = build(boundaries, ("Ez",), False, "random")
        _before, reference, moved = array_path_reference(fields, pml)
        kit.assert_moved(moved, f"linear_reduction/{boundary_label}")

        residency = Residency()
        plan = nonlinear.plan_nonlinear_constitutive(
            fields, pml, residency, chunk_bounds=record_of(fields, pml))
        run_plan(plan, residency, fields)
        nonlinear_state = {n: getattr(fields, n).copy() for n in WATCHED}

        # The certified plain kernel on the SAME seeds, with the nonlinearity
        # removed so its own predicate admits the run.
        linear_fields, linear_pml = build(boundaries, (), False, "random")
        for name in SEEDED:
            getattr(linear_fields, name)[...] = getattr(fields, name) * 0 + \
                getattr(linear_fields, name)
        linear_residency = Residency()
        linear_plan = plan_constitutive(linear_fields, linear_pml, "E",
                                        linear_residency,
                                        chunk_bounds=record_of(linear_fields,
                                                               linear_pml))
        assert linear_plan is not None, "the certified plain kernel refused"
        run_plan(linear_plan, linear_residency, linear_fields)

        # Ex and Ey are LINEAR in the nonlinear run and are the comparison; Ez is
        # not and must NOT match, which is the control that stops this leg from
        # passing because the two kernels agree everywhere for a trivial reason.
        linear_diff = sum(
            TALLY.compare(nonlinear_state[n], getattr(linear_fields, n))
            for n in ("Ex", "Ey", "f_w_Ex", "f_w_Ey"))
        pade_diff = kit.differing(nonlinear_state["f_w_Ez"],
                                  linear_fields.f_w_Ez)
        row = {"boundaries": boundary_label, "linear_differing": linear_diff,
               "pade_differing_control": pade_diff, "moved_words": moved}
        rows.append(row)
        payload["legs"]["linear_reduction"] = rows
        kit.save(payload, out)
        kit.log(f"[linear] {boundary_label:<9} linear_differing={linear_diff} "
                f"pade_control={pade_diff}")
        assert linear_diff == 0, row
        assert pade_diff > 0, (
            "the nonlinear component agreed with the LINEAR kernel: the Pade "
            "factor is then not being applied and this leg would pass vacuously")


# ---------------------------------------------------------------------------
# Leg 3: mutations
# ---------------------------------------------------------------------------

ALL_BOUNDARIES = ("periodic", "metallic", "mixed")

#: (label, old, new, must_catch, armed_on, scalar_arm, value_class, why).
#:
#: ``value_class`` IS PART OF THE ARMING AND THAT WAS MEASURED, NOT DESIGNED IN.
#: A defect must be planted where it can be SEEN, and for this family that is not
#: the same seeding for every defect. The four-corner sum reaches the answer only
#: through ``u = num/den``, and at a physical ``chi3`` with unit-amplitude fields
#: the Pade correction is ``c3 ~ 7e-3``: a one-ulp change in the four-corner sum
#: moves ``u`` by ~1e-9 relative, which is a hundredfold BELOW float32's 1.2e-7
#: resolution and is absorbed before it can change a stored word. Measured on this
#: host: the corrected association needle diverges by 0 words under ``random`` and
#: ``signed_zero`` on all three boundaries, and by 34 / 32 / 39 words under
#: ``large_u``. So it is armed on ``large_u``, and the invisibility elsewhere is
#: recorded rather than hidden — it is a real statement about how much of this
#: kernel a byte gate at physical amplitudes can resolve.
#:
#: ``armed_on`` IS NOT A CONVENIENCE AND IT IS NOT AN EXCUSE. A defect that lives
#: in the metallic ghost rule cannot alter a single word under an all-periodic
#: boundary, because the periodic spelling rewrites the INDEX and leaves the
#: validity flags at their initialised ``true`` — the mutated ternary is then
#: provably dead. Scoping such a mutation to the boundaries where it can fire and
#: reporting ``CAUGHT 2/2`` would be honest; running it on the third and reporting
#: ``CAUGHT 2/3`` would be misleading in the other direction. So this gate does
#: BOTH: the armed set must catch, and the complement is RUN AS A PREDICTED NULL
#: and required NOT to fire. If a complement case does fire, the exclusion reason
#: was wrong and the gate says so.
#:
#: ``must_catch=False`` is a NULL CONTROL: a spelling asserted to be genuinely
#: equivalent rather than merely untested.
MUTATIONS: Tuple[Tuple[str, str, str, Any, Tuple[str, ...], bool, str], ...] = (
    ("divide_fast",
     "float u2 = num2 / den2;", "float u2 = fast::divide(num2, den2);", True,
     ALL_BOUNDARIES, False, "random",
     "the measured-divergent divide spelling: fast::divide misses by up to 2 ulp "
     "on 102,578 words. This is the mutation the whole divide probe exists for"),
    ("four_corner_pair_association",
     "float sum_20 = ((g0[ii] + (uvz ? g0[i * nyz + j * nzi + uk] : 0.0f)) "
     "+ (dvx ? g0[di * nyz + j * nzi + k] : 0.0f)) "
     "+ ((uvz && dvx) ? g0[di * nyz + j * nzi + uk] : 0.0f);",
     "float sum_20 = (g0[ii] + (dvx ? g0[di * nyz + j * nzi + k] : 0.0f)) "
     "+ ((uvz ? g0[i * nyz + j * nzi + uk] : 0.0f) "
     "+ ((uvz && dvx) ? g0[di * nyz + j * nzi + uk] : 0.0f));", True,
     ALL_BOUNDARIES, False, "large_u",
     "THE PACKAGE'S PREVIOUS ASSOCIATION, put back (KERNELS.md 7.1): the near pair "
     "(g[i] + g[i-s1]) plus the far pair (g[i+s] + g[i+(s-s1)]), where MEEP and "
     "stepping._nonlinear_transverse_sums add the four corners left to right, "
     "((g[i] + g[i+s]) + g[i-s1]) + g[i+(s-s1)] (step_generic.cpp:646-648). "
     "ARMED ON large_u, as its mirror image was: at a physical chi3, c3 ~ 7e-3, and "
     "a one-ulp change in the four-corner sum moves u by ~1e-9 relative, a "
     "hundredfold under float32's resolution; large_u is where the association "
     "reaches a stored word"),
    ("four_corner_index_axis_mispair",
     "float sum_20 = ((g0[ii] + (uvz ? g0[i * nyz + j * nzi + uk] : 0.0f)) "
     "+ (dvx ? g0[di * nyz + j * nzi + k] : 0.0f)) "
     "+ ((uvz && dvx) ? g0[di * nyz + j * nzi + uk] : 0.0f);",
     "float sum_20 = ((g0[ii] + (uvz ? g0[i * nyz + j * nzi + uk] : 0.0f)) "
     "+ (dvy ? g0[i * nyz + dj * nzi + k] : 0.0f)) "
     "+ ((uvz && dvy) ? g0[i * nyz + dj * nzi + uk] : 0.0f);", True,
     ALL_BOUNDARIES, False, "random",
     "the offset-1 sum shifted down the Y axis when its partner axis is X — the "
     "exact defect `_index`/`_STRIDE` exist to prevent, since the module assembles "
     "the six index expressions from data precisely because 'six hand-written "
     "index expressions is six chances to write dj where dk belongs'"),
    ("dsqr_quarter_not_sixteenth",
     "sd_scale_pow2(sd_widen_f(transverse, true), -4, true)",
     "sd_scale_pow2(sd_widen_f(transverse, true), -2, true)", True,
     ALL_BOUNDARIES, False, "random",
     "0.0625 = (1/4)^2 = 2**-4 renormalises each UNNORMALIZED four-point sum before "
     "it is squared (stepping._nonlinear_dsqr); 2**-2 is the un-squared factor"),
    ("dsqr_in_float32",
     "    return sd_narrow_f(sd_add(sd_widen_f(own, true),\n"
     "                              sd_scale_pow2(sd_widen_f(transverse, true), -4, true),\n"
     "                              true), true);",
     "    return own + 0.0625f * transverse;", True,
     ALL_BOUNDARIES, False, "dsqr_band",
     "THE PACKAGE'S PREVIOUS Dsqr, put back (ruling R7 replaced it): the scale and "
     "the addition in float32, where MEEP's 0.0625 is a double literal and "
     "stepping._nonlinear_dsqr forms both in binary64 and rounds once. The two "
     "forms differ only where 0.0625 * t lies below float32's smallest normal, so "
     "it is ARMED ON THE dsqr_band CLASS, where chi3 * |chi1inv|^3 = 9e28 (under the "
     "ceiling) carries the change into c3; on a physical class it is invisible"),
    ("pade_numerator_coefficient",
     "float num2 = (1.0f + c2_2) + 2.0f * c3_2;",
     "float num2 = (1.0f + c2_2) + 3.0f * c3_2;", True,
     ALL_BOUNDARIES, False, "random",
     "the Pade numerator's 2*c3 and denominator's 3*c3 are not interchangeable "
     "(stepping.py:1056)"),
    ("pade_flattened_numerator",
     "float num2 = (1.0f + c2_2) + 2.0f * c3_2;",
     "float num2 = 1.0f + (c2_2 + 2.0f * c3_2);", True,
     ALL_BOUNDARIES, False, "random",
     "regrouping the numerator is a different float32 number: the association is "
     "Python's left-to-right in calc_nonlinear_u, not an algebraic identity"),
    ("chi_powers_flattened",
     "float c3_2 = (dsqr2 * chi3_2) * us_cu2;",
     "float c3_2 = dsqr2 * (chi3_2 * us_cu2);", True,
     ALL_BOUNDARIES, False, "random",
     "stepping.py:1026 groups (dsqr*chi3) first; re-associating rounds differently"),  # stepping.py live lines for the frozen device-text citation(s) in this string: 1026->1055
    ("row_scaled_before_formed",
     "float src2 = (gs2 * us2) * u2;", "float src2 = gs2 * (us2 * u2);", True,
     ALL_BOUNDARIES, False, "random",
     "stepping.py:1078-1080 forms the ROW and then scales it; folding u into the "  # stepping.py live lines for the frozen device-text citation(s) in this string: 1078-1080->1107-1109
     "epsilon multiply is a different rounding"),
    ("prev_used_after_store",
     "if (dw2) { a2 = f2[ii] + ((kp_2 * src2) - (km_2 * prev2)); }",
     "if (dw2) { a2 = f2[ii] + ((kp_2 * src2) - (km_2 * src2)); }", True,
     ALL_BOUNDARIES, False, "random",
     "the dsigw tail must subtract the PREVIOUS f_w, not the one just stored "
     "(stepping._apply_constitutive_pml). Wrong only where kms != 0 — INSIDE THE "
     "PML ONLY — so it looks like a slightly worse absorber rather than like a bug"),
    ("partner_volume_mispair",
     "float sum_20 = ((g0[ii] +", "float sum_20 = ((g1[ii] +", True,
     ALL_BOUNDARIES, False, "random",
     "Ez's offset-1 partner is Dx and its offset-2 partner is Dy "
     "(cycle_direction, stepping.py:1183-1185). Reading the wrong partner volume "
     "on one corner is a smooth wrong answer, and Dsqr's g1s^2 + g2s^2 is "
     "SYMMETRIC in the two sums — so a mispairing that swapped them wholesale "
     "would cancel; this one perturbs a single corner and does not"),

    # ---- ARMED THIS ROUND: the structural choices the module calls load-bearing
    # ---- and the previous cut left unmeasured.
    ("shifts_same_direction",
     "float sum_20 = ((g0[ii] + (uvz ? g0[i * nyz + j * nzi + uk] : 0.0f)) "
     "+ (dvx ? g0[di * nyz + j * nzi + k] : 0.0f)) "
     "+ ((uvz && dvx) ? g0[di * nyz + j * nzi + uk] : 0.0f);",
     "float sum_20 = ((g0[ii] + (dvz ? g0[i * nyz + j * nzi + dk] : 0.0f)) "
     "+ (dvx ? g0[di * nyz + j * nzi + k] : 0.0f)) "
     "+ ((dvz && dvx) ? g0[di * nyz + j * nzi + dk] : 0.0f);", True,
     ALL_BOUNDARIES, False, "random",
     "THE HALF-CELL REGISTRATION ERROR stepping.py:1131-1142 names explicitly: the "  # stepping.py live lines for the frozen device-text citation(s) in this string: 1131-1142->1160-1171
     "two shifts genuinely go in OPPOSITE directions — half a cell DOWN the "
     "partner's axis, then the formed pair half a cell UP the component's own — "
     "and taking both the same way round 'survives every scalar test'. The module "
     "docstring calls this choice load-bearing (point 5); before this round nothing "
     "measured it"),
    ("corner_drops_partner_flag",
     "((uvz && dvy) ? g1[i * nyz + dj * nzi + uk] : 0.0f)",
     "(uvz ? g1[i * nyz + dj * nzi + uk] : 0.0f)", True,
     ("metallic", "mixed"), False, "random",
     "the doubly-shifted corner takes BOTH axes' ghost rules and the load site ANDs "
     "them; dropping the partner-axis flag reads across a PEC wall. ARMED ON THE "
     "METALLIC AXES ONLY: under an all-periodic ghost the flags are never assigned "
     "— the periodic spelling rewrites the index instead — so dvy is provably true "
     "and the mutated ternary is dead code. Run on periodic anyway, as a predicted "
     "null that must not fire"),
    ("metallic_ghost_wraps_instead_of_zeroing",
     "    dvy = (dj >= 0);\n    uvy = (uj < nyi);",
     "    dj = (dj < 0) ? (nyi - 1) : dj;\n    uj = (uj == nyi) ? 0 : uj;", True,
     ("metallic", "mixed"), False, "random",
     "serving the WRAPPED plane where a PEC wall must serve an exact 0.0 "
     "(stepping._shift_down:1827-1829, _shift_up:1781-1783). The needle is the "
     "metallic spelling itself, so on an all-periodic grid it is ABSENT rather "
     "than dead — recorded n/a there, which is a different fact from a null"),
    ("coefficient_table_axis_cycled",
     "float kp_2 = kp2[k], km_2 = km2[k];",
     "float kp_2 = kp0[k], km_2 = km0[k];", True,
     ALL_BOUNDARIES, False, "random",
     "the constitutive coefficient index is on the component's OWN axis — MEEP's "
     "dsigw — NOT the dsig/dsigu cycle the curl recurrence uses. Cycling the table "
     "keeps every index in bounds and every magnitude plausible"),
    ("linear_arm_on_nonlinear_component",
     "float src2 = (gs2 * us2) * u2;", "float src2 = gs2 * us2;", True,
     ALL_BOUNDARIES, False, "random",
     "the per-component split compiled the LINEAR arm (MEEP's else-if-(u) branch) "
     "on a component the medium makes nonlinear. This is the specialisation "
     "choosing wrongly rather than the arithmetic rounding wrongly, and the two "
     "fail in different places"),
    ("chi2_squared_chi3_cubed_swapped",
     "float c2_2 = (gs2 * chi2_2) * us_sq2;",
     "float c2_2 = (gs2 * chi2_2) * us_cu2;", True,
     ALL_BOUNDARIES, False, "random",
     "chi2 carries |chi1inv|^2 and chi3 carries |chi1inv|^3 (stepping.py:1116-1117). "  # stepping.py live lines for the frozen device-text citation(s) in this string: 1116-1117->1145-1146
     "The two powers are both formed in float32 from the same volume, so swapping "
     "them is dimensionally wrong and numerically smooth"),
    ("chi_scalar_slots_swapped",
     "float chi2_2 = chis[2];", "float chi2_2 = chis[5];", True,
     ALL_BOUNDARIES, True, "random",
     "THE PACKED SCALAR BUFFER'S SLOT MAP. Six chi scalars share one binding "
     "because 34 bindings would not compile at all, so the component->slot "
     "arithmetic (chi2 at 0..2, chi3 at 3..5) is carried in the emitter rather than "
     "in the signature, where a mispairing is invisible to the compiler. Armed on "
     "the SCALAR arm, which is the only arm that reads the buffer"),

    # ---- THE PER-CHUNK TAIL (KERNELS.md 7.1 and the 7.2 B battery) -----------
    ("previous_two_step_accumulation",
     "if (dw2) { a2 = f2[ii] + ((kp_2 * src2) - (km_2 * prev2)); }",
     "if (dw2) { a2 = (f2[ii] + kp_2 * src2) - km_2 * prev2; }", True,
     ALL_BOUNDARIES, False, "random",
     "THE PACKAGE'S PREVIOUS ACCUMULATION, put back (KERNELS.md 7.1): "
     "((f + kps*src) - kms*prev), where MEEP's step_update_EDHB and "
     "stepping._apply_constitutive_pml form the right-hand side first, "
     "f + ((kps*src) - (kms*prev))"),
    ("select_removed_accumulates_everywhere",
     "const bool dw0 = in_h_x, dw1 = in_h_y, dw2 = in_h_z;",
     "const bool dw0 = true, dw1 = true, dw2 = true;", True,
     ALL_BOUNDARIES, False, "random",
     "the per-chunk select removed: every cell takes the accumulation and touches "
     "f_w, the whole-grid form the kernel stepped before the record (7.2 b1). On a "
     "plain cell the array path stores src itself and leaves f_w alone"),
    ("predicate_on_coefficient_identity",
     "const bool dw0 = in_h_x, dw1 = in_h_y, dw2 = in_h_z;",
     "const bool dw0 = (km_0 != 1.0f), dw1 = (km_1 != 1.0f), dw2 = (km_2 != 1.0f);",
     True, ALL_BOUNDARIES, False, "random",
     "the form keyed on the coefficient (kms != 1) instead of chunk membership "
     "(7.2 b2): a cell inside its layer's chunk whose sigma is 0 (kps = kms = 1 "
     "there) takes the output select instead of the accumulation"),
    ("dsig_flag_on_the_wrong_axis",
     "const bool dw0 = in_h_x, dw1 = in_h_y, dw2 = in_h_z;",
     "const bool dw0 = in_h_x, dw1 = in_h_y, dw2 = in_h_x;", True,
     ALL_BOUNDARIES, False, "random",
     "Ez's membership read on the x line rather than its own z line: the "
     "constitutive form follows the component's OWN axis (MEEP's dsigw), as the "
     "coefficient index does"),
    ("aux_stored_on_a_plain_cell",
     "if (dw2) { w2[ii] = src2; }", "w2[ii] = src2;", True,
     ALL_BOUNDARIES, False, "random",
     "f_w written outside its layer's chunks (7.2 b6): the auxiliary contract "
     "leaves it untouched there, and the watched f_w volumes compare every cell"),
    ("plain_cell_as_unit_coefficients",
     "    float a2 = src2;\n"
     "    if (dw2) { a2 = f2[ii] + ((kp_2 * src2) - (km_2 * prev2)); }",
     "    float a2 = 0.0f + ((1.0f * src2) - (0.0f * prev2));\n"
     "    if (dw2) { a2 = f2[ii] + ((kp_2 * src2) - (km_2 * prev2)); }", True,
     ALL_BOUNDARIES, False, "signed_zero",
     "the plain constitutive spelled as the accumulation with kps = 1, kms = 0 and "
     "zeroed inputs instead of the output select (7.2 b8, C4). It differs only on a "
     "-0 source: MEEP stores -0, the accumulation +0, so it is ARMED ON signed_zero, "
     "whose half-zero D lanes plant -0 on plain cells"),
    # ---- NULL CONTROLS -----------------------------------------------------
    ("aux_load_unconditional",
     "float prev2 = dw2 ? w2[ii] : 0.0f;", "float prev2 = w2[ii];", False,
     ALL_BOUNDARIES, False, "random",
     "DECLARED NULL (7.2): the auxiliary load made unconditional while its store "
     "stays guarded. prev2 is read only inside the guarded accumulation, so no word "
     "changes; the guard is pinned by the module test that reads the tail's lines"),
    ("dsqr_add_operands_commuted",
     "    return sd_narrow_f(sd_add(sd_widen_f(own, true),\n"
     "                              sd_scale_pow2(sd_widen_f(transverse, true), -4, true),\n"
     "                              true), true);",
     "    return sd_narrow_f(sd_add(sd_scale_pow2(sd_widen_f(transverse, true), -4, true),\n"
     "                              sd_widen_f(own, true),\n"
     "                              true), true);", False,
     ALL_BOUNDARIES, False, "dsqr_band",
     "NULL CONTROL: the binary64 addition's operands commuted. IEEE addition is "
     "commutative on every non-NaN operand, signed zero included, and the software "
     "helper is exact, so no word may change, even on the band where the two Dsqr "
     "forms differ. A CAUGHT verdict would mean the helper is not IEEE addition"),
    ("us_cube_operands_commuted",
     "float us_cu2 = (us2 * us2) * us2;", "float us_cu2 = us2 * (us2 * us2);",
     False, ALL_BOUNDARIES, False, "random",
     "NULL CONTROL, and it is a COMMUTATION rather than a reassociation: both "
     "spellings round us*us first and then multiply by us, and float32 "
     "multiplication IS commutative. Carried to pin that the gate does not report "
     "spurious catches on operand order — the genuine association hazards are "
     "`chi_powers_flattened` and `pade_flattened_numerator`, which ARE caught"),
    ("partner_squares_order_swapped",
     "float transverse = (g1s * g1s) + (g2s * g2s);",
     "float transverse = (g2s * g2s) + (g1s * g1s);", False,
     ALL_BOUNDARIES, False, "random",
     "NULL CONTROL THAT RECORDS A REAL BLIND SPOT rather than a safe spelling. A "
     "WHOLESALE swap of the two transverse partners — Ez taking Dy then Dx instead "
     "of Dx then Dy — reaches the answer only through Dsqr's g1s^2 + g2s^2, which "
     "is SYMMETRIC in the two sums, so the defect is exactly this reordering and "
     "float32 addition is commutative. No byte gate can see it, here or on any "
     "other backend. That is why `partner_volume_mispair` perturbs a SINGLE corner "
     "instead, and why the partner order is additionally pinned against stepping's "
     "own (own_axis + offset) % 3 by a sibling unit test rather than by this gate"),
)


def leg_mutations(payload: Dict[str, Any], out: str) -> None:
    harness = kit.MutationHarness(payload, out)
    armed_total = caught_total = 0
    for (label, old, new, must_catch, armed_on, scalar_arm, value_class,
         why) in MUTATIONS:
        ran = caught = 0
        null_ran = null_caught = 0
        not_applicable: List[str] = []
        missed = False
        launches = 0
        for boundary_label, boundaries in BOUNDARIES:
            is_armed = boundary_label in armed_on
            # The band class makes Ez alone nonlinear: its partners are the tiny
            # Dx, Dy volumes, whose own squares would underflow as Ex's and Ey's gs.
            components = (("Ez",) if value_class == "dsqr_band"
                          else ("Ex", "Ey", "Ez"))
            fields, pml = build(boundaries, components, scalar_arm, value_class)
            _before, reference, moved = array_path_reference(fields, pml)
            kit.assert_moved(moved, f"{label}/{boundary_label}")
            arrays, flat, chi = harness_arrays(fields, pml)
            codes = tuple(nonlinear.BOUNDARY_CODES[kind] for kind in
                          nonlinear._boundary_kinds(fields.grid, pml))
            live = nonlinear.nonlinear_components(fields)
            chi2_volume, chi3_volume = nonlinear.chi_volume_arms(fields)
            shipped = nonlinear.nonlinear_source(live, chi2_volume, chi3_volume,
                                                 codes)
            try:
                mutated = kit.needle(shipped, old, new)
            except LookupError:
                # An ARMED boundary whose needle is absent is a real miss. On the
                # complement it means the spelling this defect lives in is not
                # emitted there at all, which is a third outcome and is recorded
                # as such rather than folded into either count.
                if is_armed:
                    missed = True
                else:
                    not_applicable.append(boundary_label)
                continue
            function = kit.Counter(
                compile_source(mutated).nonlinear_constitutive_step)
            residency = Residency()
            plan = nonlinear.plan_nonlinear_constitutive_from_arrays(
                arrays, flat, chi, codes, residency,
                functions={shaders.CONTRACT_OFF: function},
                chunk_bounds=record_of(fields, pml))
            run_plan(plan, residency, fields)
            launches += function.launches
            differing = sum(kit.differing(getattr(fields, n), reference[n])
                            for n in WATCHED)
            if is_armed:
                ran += 1
                caught += int(differing > 0)
            else:
                null_ran += 1
                null_caught += int(differing > 0)
        row_extra = {
            "armed_on": list(armed_on),
            "chi_arm": "scalar" if scalar_arm else "volume",
            "value_class": value_class,
            "predicted_null_ran": null_ran,
            "predicted_null_fired": null_caught,
            "not_applicable_on": not_applicable,
        }
        harness.record(label, kit.MutationHarness.verdict(missed, ran, launches,
                                                          caught),
                       launches, caught, ran, must_catch, why, extra=row_extra)
        # The complement is a MEASUREMENT, not an assumption: a mutation excluded
        # from a boundary because it "cannot fire there" is required to actually
        # not fire there. A firing complement means the exclusion reason is wrong.
        assert null_caught == 0, (
            f"{label} fired on a boundary it was NOT armed on ({null_caught} of "
            f"{null_ran} predicted-null cases diverged). The exclusion reason "
            f"recorded for this mutation is wrong: {why}")
        if must_catch is True:
            armed_total += ran
            caught_total += caught

    payload["mutation_totals"] = {
        "must_catch_entries": sum(1 for m in MUTATIONS if m[3] is True),
        "null_control_entries": sum(1 for m in MUTATIONS if m[3] is False),
        "armed_cases": armed_total,
        "caught_cases": caught_total,
    }
    kit.save(payload, out)

    # THE SUB-LATTICE SWAP IS A HOST DEFECT AND NO SOURCE NEEDLE CAN REACH IT.
    # The kernel takes coefficient POINTERS and never asks which Yee lattice they
    # came from, so binding the INTEGER tables where the E side wants the
    # HALF-INTEGER ones (stepping.py:1015) is a half-cell absorber error that
    # compiles, launches and looks like a slightly different PML. It is planted in
    # the PLAN rather than in the source, which is why it sits outside the needle
    # loop above.
    ran = caught = launches = 0
    for boundary_label, boundaries in BOUNDARIES:
        fields, pml = build(boundaries, ("Ex", "Ey", "Ez"), False, "random")
        _before, reference, moved = array_path_reference(fields, pml)
        kit.assert_moved(moved, f"sub_lattice_swap/{boundary_label}")
        arrays, flat, chi = harness_arrays(fields, pml)
        swapped = {f"{stem}_{axis}": getattr(pml, f"{stem}_{axis}")
                   for axis in "xyz" for stem in ("kps", "kms")}
        assert any(kit.differing(swapped[k], flat[k]) for k in flat), (
            "the integer and half-integer coefficient tables are identical on "
            "this grid, so this mutation could not diverge for a reason that has "
            "nothing to do with the kernel")
        codes = tuple(nonlinear.BOUNDARY_CODES[kind] for kind in
                      nonlinear._boundary_kinds(fields.grid, pml))
        residency = Residency()
        plan = nonlinear.plan_nonlinear_constitutive_from_arrays(
            arrays, swapped, chi, codes, residency,
            chunk_bounds=record_of(fields, pml))
        run_plan(plan, residency, fields)
        launches += plan.launches
        differing = sum(kit.differing(getattr(fields, n), reference[n])
                        for n in WATCHED)
        ran += 1
        caught += int(differing > 0)
    harness.record("sub_lattice_swap",
                   kit.MutationHarness.verdict(False, ran, launches, caught),
                   launches, caught, ran, True,
                   "binding the INTEGER kps/kms tables where update_E wants the "
                   "HALF-INTEGER ones (stepping.py:1015): a half-cell absorber "
                   "error that compiles and looks plausible. Planted in the PLAN, "
                   "because the kernel cannot see which lattice a pointer came "
                   "from")
    payload["mutation_totals"]["armed_cases"] += ran
    payload["mutation_totals"]["caught_cases"] += caught
    payload["mutation_totals"]["must_catch_entries"] += 1
    kit.save(payload, out)

    # THE RECORD MUTATIONS (KERNELS.md 7.2 b3, b4): planted in the chunk record the
    # plan binds rather than in the source, because the kernel reads bounds and
    # cannot see which division they describe.
    for label, mutate, why in RECORD_MUTATIONS:
        ran = caught = launches = 0
        for boundary_label, boundaries in BOUNDARIES:
            fields, pml = build(boundaries, ("Ex", "Ey", "Ez"), False, "random")
            _before, reference, moved = array_path_reference(fields, pml)
            kit.assert_moved(moved, f"{label}/{boundary_label}")
            arrays, flat, chi = harness_arrays(fields, pml)
            codes = tuple(nonlinear.BOUNDARY_CODES[kind] for kind in
                          nonlinear._boundary_kinds(fields.grid, pml))
            shipped = np.array(record_of(fields, pml), dtype=np.int32, copy=True)
            planted = mutate(shipped.copy())
            assert int(np.count_nonzero(planted != shipped)), (
                f"{label}: the planted record equals the shipped one on this grid, so "
                f"the mutation could not diverge for a reason that is the kernel's")
            residency = Residency()
            plan = nonlinear.plan_nonlinear_constitutive_from_arrays(
                arrays, flat, chi, codes, residency, chunk_bounds=planted)
            run_plan(plan, residency, fields)
            launches += plan.launches
            differing = sum(kit.differing(getattr(fields, n), reference[n])
                            for n in WATCHED)
            ran += 1
            caught += int(differing > 0)
        harness.record(label, kit.MutationHarness.verdict(False, ran, launches, caught),
                       launches, caught, ran, True, why)
        payload["mutation_totals"]["armed_cases"] += ran
        payload["mutation_totals"]["caught_cases"] += caught
        payload["mutation_totals"]["must_catch_entries"] += 1
        kit.save(payload, out)
    kit.log(f"[mutations] armed {payload['mutation_totals']['caught_cases']}"
            f"/{payload['mutation_totals']['armed_cases']} cases caught across "
            f"{payload['mutation_totals']['must_catch_entries']} must-catch "
            f"entries; {payload['mutation_totals']['null_control_entries']} null "
            f"controls held")


def _swap_lattices(record: Any) -> Any:
    """The integer and half-integer entries exchanged on every axis."""
    out = record.copy()
    for axis in range(3):
        for word in range(4):
            out[record_index(True, axis, word)] = record[record_index(False, axis, word)]
            out[record_index(False, axis, word)] = record[record_index(True, axis, word)]
    return out


def _low_bound_off_by_one(record: Any) -> Any:
    """The half-integer x line's low run one cell longer: ``a1 + 1``."""
    out = record.copy()
    out[record_index(True, 0, 1)] += 1
    return out


#: (label, record mutation, why). Each must be caught.
RECORD_MUTATIONS: Tuple[Tuple[str, Any, str], ...] = (
    ("record_lattices_swapped", _swap_lattices,
     "the integer-lattice runs bound where update_E reads the half-integer ones "
     "(7.2 b4): an E component sits half a cell off on its own axis, so the "
     "boundary plane of every layer moves by one cell"),
    ("record_bound_off_by_one", _low_bound_off_by_one,
     "a1 + 1 on the half-integer x line (7.2 b3): the first interior plane takes "
     "the accumulation"),
)


# ---------------------------------------------------------------------------
# Leg 4: the plan-time magnitude ceiling
# ---------------------------------------------------------------------------

def leg_ceiling(payload: Dict[str, Any], out: str) -> None:
    """The 1e29 clause, pinned in BOTH directions and non-vacuously.

    THE SCIENCE IS NOT RE-DERIVED HERE and the 20,000-step census is NOT re-run.
    That measurement — five real lifted rows, 218,841,786 state words — established
    that the Pade tree DOES reach the subnormal band and that the band is PROVABLY
    INVISIBLE, because it cannot propagate past the ``1 + ...`` additions, with the
    crossing located at ``chi3 * |chi1inv|^3 = 1e31``. What this leg pins is the
    CLAUSE built on that result: the constant, the admission below it, the refusal
    above it, and the by-name reason the refusal carries.

    NON-VACUITY IS THE POINT OF THE LEG, not a decoration on it. The previous
    round's composition row for this clause was VACUOUS and nobody noticed by
    reading it: ``chi3 = 1e30`` with ``chi1inv = 1/3`` forms ``3.7e28``, UNDER the
    ceiling, so the row that was supposed to demonstrate a refusal demonstrated an
    admission. Both sides here are therefore measured through the family's OWN
    magnitude helper and asserted to land where the case claims:

    * the ADMITTED case must measure above ``ceiling / 10`` — otherwise "below the
      ceiling" is satisfied by any physical material and pins nothing;
    * the REFUSED case must measure above the ceiling — otherwise the refusal came
      from some other clause and this leg would be reading the wrong reason.
    """
    rows: List[Dict[str, Any]] = []
    payload["legs"]["ceiling"] = rows

    ceiling = nonlinear.CHI_MAGNITUDE_CEILING
    assert ceiling == 1e29, (
        f"the family's ceiling moved to {ceiling!r}. The settled crossing is "
        f"chi3 * |chi1inv|^3 = 1e31 and the clause sits two decades under it; a "
        f"change to this constant is a change to what the family refuses and must "
        f"be re-measured, not re-pinned")
    # ONE CONSTANT, TWO HOMES — pinned because they can drift apart silently.
    # preconditions.NONLINEAR_INTERMEDIATE_BOUND is the shared bound written for
    # this family before it existed; the family carries its own because its clause
    # is PER-CELL (max_i |chi_i| * |chi1inv_i|^p) while the shared helper takes a
    # product of independent maxima, which is strictly looser. Different
    # expressions, same number, and nothing else checks that.
    assert preconditions.NONLINEAR_INTERMEDIATE_BOUND == ceiling, (
        f"preconditions.NONLINEAR_INTERMEDIATE_BOUND is "
        f"{preconditions.NONLINEAR_INTERMEDIATE_BOUND!r} but the family refuses at "
        f"{ceiling!r}: two spellings of one clause have drifted apart")
    payload["ceiling"] = {
        "constant": ceiling,
        "shared_bound": preconditions.NONLINEAR_INTERMEDIATE_BOUND,
        "measured_crossing": 1e31,
        "decades_of_margin": 2,
    }

    for order, label in ((2, "chi2"), (3, "chi3")):
        # ---- BELOW: admitted, AND byte-identical at the clause's own edge -----
        target = ceiling / 2.0
        fields, pml = build_at_magnitude("metallic", ("Ez",), target, order)
        measured = measured_magnitude(fields, "Ez", order)
        assert measured > ceiling / 10.0, (
            f"VACUOUS admission case: {label} measured {measured:.3e}, which is "
            f"more than a decade under the {ceiling:.0e} ceiling. A case this far "
            f"below the line is admitted for reasons that have nothing to do with "
            f"the clause")
        assert measured <= ceiling, (
            f"the admission case measured {measured:.3e}, ABOVE the ceiling")
        verdict = nonlinear.nonlinear_constitutive_coverage(fields, pml,
                                                            Residency())
        assert verdict.covered, (
            f"{label} at {measured:.3e} is under the {ceiling:.0e} ceiling and was "
            f"refused anyway: {verdict.reasons}")
        _before, reference, moved = array_path_reference(fields, pml)
        kit.assert_moved(moved, f"ceiling/{label}/below")
        residency = Residency()
        plan = nonlinear.plan_nonlinear_constitutive(
            fields, pml, residency, chunk_bounds=record_of(fields, pml))
        assert plan is not None
        run_plan(plan, residency, fields)
        state = {n: getattr(fields, n) for n in WATCHED}
        differing = TALLY.compare_state(state, reference)
        row = {"clause": label, "order": order, "side": "below",
               "target": target, "measured_magnitude": measured,
               "covered": True, "moved_words": moved, "differing": differing}
        rows.append(row)
        payload["legs"]["ceiling"] = rows
        kit.save(payload, out)
        kit.log(f"[ceiling] {label} below  measured={measured:.3e} "
                f"covered=True moved={moved} differing={differing}")
        assert differing == 0, row

        # ---- ABOVE: refused, BY NAME -----------------------------------------
        target = ceiling * 10.0
        fields, pml = build_at_magnitude("metallic", ("Ez",), target, order)
        measured = measured_magnitude(fields, "Ez", order)
        assert measured > ceiling, (
            f"VACUOUS refusal case: {label} measured {measured:.3e}, which is NOT "
            f"above the {ceiling:.0e} ceiling. This is exactly the trap the "
            f"previous round's composition row fell into — a nominal chi says "
            f"nothing about the product the clause reads")
        verdict = nonlinear.nonlinear_constitutive_coverage(fields, pml,
                                                            Residency())
        assert not verdict.covered, (
            f"{label} at {measured:.3e} is ABOVE the {ceiling:.0e} ceiling and was "
            f"admitted anyway")
        # BY NAME: the reason must name the clause, the component, the measured
        # magnitude and the ceiling. A bare "not covered" would be satisfied by a
        # refusal from any of the other eighteen clauses.
        named = [reason for reason in verdict.reasons
                 if reason.startswith(f"{label}[Ez] * |chi1inv|^{order}")]
        assert len(named) == 1, (
            f"the refusal is not BY NAME: no single reason names "
            f"{label}[Ez] * |chi1inv|^{order}. Reasons were {verdict.reasons}")
        assert f"{ceiling:.0e}" in named[0], (
            f"the named refusal does not carry the ceiling it refused against: "
            f"{named[0]!r}")
        assert "1e31" in named[0], (
            f"the named refusal does not carry the measured crossing it is "
            f"derived from: {named[0]!r}")
        # And the refusal must actually stop a plan, not merely be recorded.
        assert nonlinear.plan_nonlinear_constitutive(
            fields, pml, Residency(), chunk_bounds=record_of(fields, pml)) is None, (
            "the predicate refused but the plan builder still returned a plan: a "
            "coverage refusal that does not stop the build is decorative")
        row = {"clause": label, "order": order, "side": "above",
               "target": target, "measured_magnitude": measured,
               "covered": False, "refusal": named[0]}
        rows.append(row)
        payload["legs"]["ceiling"] = rows
        kit.save(payload, out)
        kit.log(f"[ceiling] {label} above  measured={measured:.3e} "
                f"covered=False REFUSED BY NAME")


# ---------------------------------------------------------------------------
# Leg 5: the subnormal census, as a WINDOW over a real run
# ---------------------------------------------------------------------------

def leg_census(payload: Dict[str, Any], out: str) -> None:
    """The precondition every other leg leans on, AS A NUMBER AND AS A WINDOW.

    THREE THINGS THE PREVIOUS CUT DID NOT DO, each of which it needed to:

    1. **it censused only what was STORED.** E and f_w cannot reach the band on a
       physical fill; the values that can are the ones the Pade body forms and
       never writes down — ``dsqr``, ``c2``, ``c3``, ``us^3``. ``preconditions``'
       own docstring names chi2/chi3 as the one family that gets there BY
       CONSTRUCTION, and the census looked past it;
    2. **it reported a scalar.** Band entry was measured to be a RUN-and-WINDOW
       fact, so a census after one ``update_E`` certifies one ``update_E``. This
       walks ``CENSUS_STEPS`` complete array-path cycles and reports
       ``[first_step, last_step]``;
    3. **it never demonstrated the census could fire.** A precondition that has
       only ever been run on clean data is decorative — ``preconditions``
       says so in as many words — so the scaled control at the end is required to
       put the window in the band.

    A case that reaches the band is REFUSED BY NAME: a coverage refusal, not a
    failure, with its comparisons withheld from the certified total.
    """
    rows: List[Dict[str, Any]] = []
    payload["legs"]["census"] = rows
    zero_population = 0
    refused: List[str] = []

    for boundary_label, boundaries in BOUNDARIES:
        for value_class in ("random", "signed_zero", "large_u"):
            name = f"{boundary_label}/{value_class}"
            fields, pml = build(boundaries, ("Ex", "Ey", "Ez"), False,
                                value_class)
            shape_words = int(np.asarray(fields.Ez).size)
            window = preconditions.SubnormalWindow(
                first_step=0, last_step=CENSUS_STEPS - 1,
                per_array_words=shape_words,
                per_intermediate_words=shape_words)
            first_fired: Optional[int] = None
            last_fired: Optional[int] = None
            moved_total = 0
            signed = 0

            for step in range(CENSUS_STEPS):
                fired = 0
                # OPERANDS: what update_E reads this step.
                volumes = fields.displacement_minus_polarization_volumes()
                for component, _source, _axis in nonlinear.E_TERMS:
                    fired += window.observe(f"D:{component}", volumes[component],
                                            step=step)
                    fired += window.observe(
                        f"inv_eps:{component}",
                        fields.inverse_epsilon_for(component), step=step)
                # INTERMEDIATES: the values the kernel forms and never stores.
                for label, array in nonlinear_intermediates(fields, pml).items():
                    fired += window.observe_intermediate(label, array, step=step)
                # RESULTS: what it wrote last step.
                for watched in WATCHED:
                    fired += window.observe(f"out:{watched}",
                                            getattr(fields, watched), step=step)
                    signed += subnormal.signed_zero_census(
                        getattr(fields, watched)).get("negative_zero", 0)
                if fired:
                    first_fired = step if first_fired is None else first_fired
                    last_fired = step

                before = {n: getattr(fields, n).copy() for n in WATCHED}
                drive_one_cycle(fields, pml)
                moved_total += sum(kit.differing(getattr(fields, n), before[n])
                                   for n in WATCHED)

            report = window.report()
            zero_population += signed
            row = {
                "boundaries": boundary_label, "value_class": value_class,
                "window": [window.first_step, window.last_step],
                "subnormal_window": [first_fired, last_fired],
                "subnormal_words": report["subnormal_words"],
                "observed_words": report["observed_words"],
                "intermediates_censused": len(report["intermediates"]),
                "negative_zero_words": signed,
                "moved_words": moved_total,
                "vacuous": report["vacuous"],
                "vacuity_reasons": report["vacuity_reasons"],
            }
            rows.append(row)
            payload["legs"]["census"] = rows
            kit.save(payload, out)
            kit.log(f"[census] {boundary_label:<9} {value_class:<12} "
                    f"steps=[{window.first_step},{window.last_step}] "
                    f"observed={report['observed_words']:<8} "
                    f"subnormal={report['subnormal_words']:<6} "
                    f"band_window=[{first_fired},{last_fired}] "
                    f"neg_zero={signed} moved={moved_total}")
            kit.assert_moved(moved_total, f"census/{name}")

            try:
                preconditions.assert_clean_or_refuse(window, f"nonlinear/{name}")
            except AssertionError as exc:
                # A COVERAGE REFUSAL, NOT A FAILURE. Metal flushes and has no
                # lever, so a case that reaches the band is one this family cannot
                # claim byte-identity for. Its comparisons are withheld.
                row["refused"] = str(exc)
                refused.append(name)
                TALLY.withheld += report["observed_words"]
                payload["legs"]["census"] = rows
                kit.save(payload, out)
                kit.log(f"[census] REFUSED BY NAME: {name} — band reached over "
                        f"steps [{first_fired}, {last_fired}]")

    kit.assert_census_floor(zero_population, "negative zero words reaching the "
                                             "Pade numerator")
    payload["census_refused"] = refused

    # ---- THE FIRING CONTROL, AS A MEASURED LADDER --------------------------
    # Without this the empty windows above prove only that the physical band is
    # clean, which was never in doubt.
    #
    # THE SCALE IS MEASURED, NOT ASSUMED, AND THE FIRST ATTEMPT AT IT WAS WRONG.
    # ``preconditions``' docstring motivates the intermediate hook with "E^3 with
    # E ~ 1e-13 lands at 1e-39", and a control built at 1e-13 DID NOT FIRE — 0 of
    # 414,720 words. The figure is right about a cube of the field and wrong about
    # THIS tree, which forms ``c3 = (Dsqr * chi3) * chi1inv^3`` — a SQUARE of D
    # carrying two small coefficients, not a cube of E. At D ~ 1e-13 that is
    # 1e-26 * 0.023 * 0.3 ~ 7e-29, three decades of headroom still in hand.
    # So the ladder below LOCATES the entry instead of asserting it, which is the
    # same shape as the cliff ``preconditions`` records for the curl and is the
    # number this family did not have.
    ladder: List[Dict[str, Any]] = []
    fired_scales: List[float] = []
    for scale in (1e0, 1e-6, 1e-13, 1e-17, 1e-18, 1e-19, 1e-20):
        fields, pml = build("metallic", ("Ex", "Ey", "Ez"), False, "random")
        for seeded in SEEDED:
            getattr(fields, seeded)[...] = (getattr(fields, seeded) * scale
                                            ).astype(np.float32)
        control = preconditions.SubnormalWindow(
            first_step=0, last_step=0,
            per_array_words=int(np.asarray(fields.Ez).size))
        for component, _source, _axis in nonlinear.E_TERMS:
            control.observe(
                f"D:{component}",
                fields.displacement_minus_polarization_volumes()[component],
                step=0)
        for label, array in nonlinear_intermediates(fields, pml).items():
            control.observe_intermediate(label, array, step=0)
        report = control.report()
        band_labels = sorted(
            label for label, counts in report["intermediates"].items()
            if counts["subnormal_words"])
        rung = {"scale": scale,
                "subnormal_words": report["subnormal_words"],
                "observed_words": report["observed_words"],
                "intermediates_in_band": band_labels}
        ladder.append(rung)
        payload["legs"]["census_firing_control"] = {"ladder": ladder}
        kit.save(payload, out)
        kit.log(f"[control] scale={scale:<8.0e} "
                f"subnormal={report['subnormal_words']:<7} of "
                f"{report['observed_words']} reached={band_labels}")
        if band_labels:
            fired_scales.append(scale)
            # The precondition is enforced through the shared helper on the rung
            # that fires, so the control exercises the same code path the clean
            # cases are judged by rather than a private re-implementation.
            preconditions.demonstrate_firing(control,
                                             f"nonlinear/control@{scale:.0e}")

    payload["legs"]["census_firing_control"] = {
        "ladder": ladder,
        "clean_at": [rung["scale"] for rung in ladder
                     if not rung["intermediates_in_band"]],
        "fired_at": fired_scales,
        "note": ("the entry is located by measurement; the 1e-13 figure in "
                 "preconditions.py describes a cube of the field and does not "
                 "describe this tree, which squares D and carries two small "
                 "coefficients"),
    }
    kit.save(payload, out)
    # The control must fire THROUGH AN INTERMEDIATE, which is the whole claim
    # about this family: its subnormal exposure is the un-storable products. A
    # control that only put an OPERAND in the band would demonstrate the census
    # works on inputs and say nothing about the tree.
    assert fired_scales, (
        "no rung of the ladder put an INTERMEDIATE in the band, so the census "
        "was never demonstrated to fire on the values this family's exposure "
        "actually lives in. A precondition only ever run on clean data is "
        "decorative")
    assert not ladder[0]["intermediates_in_band"], (
        "the UNSCALED physical fill already reaches the band, which would mean "
        "the clean cases above are clean for some other reason")


# ---------------------------------------------------------------------------
# Leg 6: the binary64 Dsqr, alone and through the kernel
# ---------------------------------------------------------------------------

#: The helper alone, launched over three float32 volumes: the bytes the kernel
#: inlines (``nonlinear_update_e.DSQR_HELPER``), behind the same header, directive
#: and software binary64 fragment.
_DSQR_PROBE_KERNEL = """
kernel void nonlinear_dsqr_probe(
    device float*       out     [[buffer(0)]],
    device const float* gs      [[buffer(1)]],
    device const float* g1s     [[buffer(2)]],
    device const float* g2s     [[buffer(3)]],
    constant uint&      n       [[buffer(4)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx >= n) { return; }
    out[idx] = nonlinear_dsqr(gs[idx], g1s[idx], g2s[idx]);
}
"""

#: Operands per class of the helper leg.
DSQR_OPERANDS = 1 << 20


def dsqr_probe_source(helper: str = None) -> str:
    """The probe library: the kernel's own prologue pieces and ``helper``."""
    return softfp64.library_source(
        "\n" + (nonlinear.DSQR_HELPER if helper is None else helper) + "\n"
        + _DSQR_PROBE_KERNEL)


def _dsqr_operands(band: bool, n: int, seed: int) -> Tuple[Any, Any, Any]:
    rng = np.random.default_rng(seed)
    if band:
        gs = (np.ldexp(rng.uniform(1.0, 2.0, n), BAND_GS_EXPONENT)
              * rng.choice((-1.0, 1.0), n)).astype(np.float32)
        g1s = np.ldexp(rng.uniform(1.0, 2.0, n), BAND_PARTNER_EXPONENT + 2).astype(np.float32)
        g2s = np.ldexp(rng.uniform(1.0, 2.0, n), BAND_PARTNER_EXPONENT + 2).astype(np.float32)
    else:
        gs, g1s, g2s = (rng.standard_normal(n).astype(np.float32) for _ in range(3))
    return gs, g1s, g2s


def launch_dsqr(library: Any, gs: Any, g1s: Any, g2s: Any) -> Any:
    import torch  # noqa: PLC0415

    n = int(gs.size)
    out = torch.empty(n, dtype=torch.float32, device="mps")
    library.nonlinear_dsqr_probe(
        out, *(torch.from_numpy(np.ascontiguousarray(a)).to("mps")
               for a in (gs, g1s, g2s)), n, threads=n)
    torch.mps.synchronize()
    return out.cpu().numpy()


def leg_dsqr(payload: Dict[str, Any], out: str) -> None:
    """The binary64 ``Dsqr`` as words: the helper on its band and at field scale, then
    the shipped kernel on the ``dsqr_band`` state.

    The float32 form is the CONTROL on both halves: it must differ from
    ``stepping._nonlinear_dsqr`` on the band and nowhere at field scale, and on the
    kernel half the number of ``src`` words it would have moved is the vacuity floor
    (a band that moved no stored word would certify the helper and say nothing
    about E).
    """
    result: Dict[str, Any] = {"helper": [], "kernel": []}
    payload["legs"]["dsqr"] = result
    library = compile_source(dsqr_probe_source())
    float32_library = compile_source(dsqr_probe_source(
        nonlinear.DSQR_HELPER.replace(
            "    return sd_narrow_f(sd_add(sd_widen_f(own, true),\n"
            "                              sd_scale_pow2(sd_widen_f(transverse, true), -4, true),\n"
            "                              true), true);",
            "    return own + 0.0625f * transverse;")))
    for band, seed in ((True, 71), (False, 73)):
        gs, g1s, g2s = _dsqr_operands(band, DSQR_OPERANDS, seed)
        with flushed_host():
            reference = stepping._nonlinear_dsqr(gs, (g1s, g2s), stepping._WHOLE)
            shipped = launch_dsqr(library, gs, g1s, g2s)
            float32_form = launch_dsqr(float32_library, gs, g1s, g2s)
        row = {"class": "band" if band else "field_scale", "operands": int(gs.size),
               "shipped_differing": TALLY.compare(shipped, reference),
               "float32_form_differing": kit.differing(float32_form, reference)}
        result["helper"].append(row)
        kit.save(payload, out)
        kit.log(f"[dsqr] helper {row['class']:<11} shipped "
                f"{row['shipped_differing']}/{row['operands']} differ, float32 form "
                f"{row['float32_form_differing']}/{row['operands']}")
        assert row["shipped_differing"] == 0, row
        if band:
            assert row["float32_form_differing"] > row["operands"] // 10, (
                "VACUOUS: the band does not separate the two Dsqr forms", row)
        else:
            assert row["float32_form_differing"] == 0, (
                "the float32 Dsqr differs at field scale, where the two forms are "
                "the same word: the reference or the control is not what it says", row)

    for boundary_label, boundaries in BOUNDARIES:
        fields, pml = build(boundaries, ("Ez",), False, "dsqr_band", seed=79)
        verdict = nonlinear.nonlinear_constitutive_coverage(fields, pml, Residency())
        assert verdict.covered, (boundary_label, verdict.reasons)
        displacement = stepping._nonlinear_displacement(fields, pml)
        g1s, g2s = stepping._nonlinear_transverse_sums(fields, "Ez", displacement)
        gs, us = fields.Dz, fields.inverse_epsilon_for("Ez")
        chi2, chi3 = fields.chi2_for("Ez"), fields.chi3_for("Ez")
        with flushed_host():
            exact = stepping._nonlinear_dsqr(gs, (g1s, g2s), stepping._WHOLE)
            float32_form = gs * gs + np.float32(0.0625) * (g1s * g1s + g2s * g2s)
            rows = [(gs * us) * stepping.calc_nonlinear_u(d, gs, us, chi2, chi3)
                    for d in (exact, float32_form)]
        dsqr_moved = kit.differing(exact, float32_form)
        src_moved = kit.differing(rows[0], rows[1])
        _before, reference, moved = array_path_reference(fields, pml)
        kit.assert_moved(moved, f"dsqr/{boundary_label}")
        residency = Residency()
        plan = nonlinear.plan_nonlinear_constitutive(
            fields, pml, residency, chunk_bounds=record_of(fields, pml))
        assert plan is not None
        run_plan(plan, residency, fields)
        state = {n: getattr(fields, n) for n in WATCHED}
        differing = TALLY.compare_state(state, reference)
        row = {"boundaries": boundary_label, "dsqr_words_the_forms_split": dsqr_moved,
               "src_words_the_float32_form_moves": src_moved,
               "moved_words": moved, "differing": differing,
               "words_compared": int(sum(kit.words(state[n]).size for n in WATCHED))}
        result["kernel"].append(row)
        kit.save(payload, out)
        kit.log(f"[dsqr] kernel {boundary_label:<9} Dsqr split {dsqr_moved}, src "
                f"the float32 form moves {src_moved}, differing={differing}")
        assert differing == 0, row
        assert src_moved > 0, (
            "VACUOUS: on this state the float32 Dsqr moves no source word", row)


# ---------------------------------------------------------------------------
# Leg 7: nonlinear PML spine (B -> H -> D)
# ---------------------------------------------------------------------------

_SPINE_STEPS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("step_B", ("Bx", "By", "Bz", "fu_Bx", "fu_By", "fu_Bz")),
    ("update_H", ("Hx", "Hy", "Hz", "f_w_Hx", "f_w_Hy", "f_w_Hz")),
    ("step_D", ("Dx", "Dy", "Dz", "fu_Dx", "fu_Dy", "fu_Dz")),
)


def _spine_plan(fields: Any, pml: Any, slot: str, residency: Residency) -> Any:
    """Build one of the three nonlinear-only admissions through its public route.

    Each binds the one-rank record of the case's layer, the division the reference
    ``Fields`` steps with (no driver installs another).
    """
    record = layer_record(fields.grid, pml)
    if slot == "update_H":
        return nonlinear.plan_nonlinear_run_constitutive(fields, pml, "H",
                                                          residency,
                                                          chunk_bounds=record)
    return nonlinear.plan_nonlinear_run_pml_curl(fields, pml, slot, residency,
                                                 chunk_bounds=record)


def _spine_reference(fields: Any, pml: Any, slot: str) -> None:
    """The engine's corresponding sub-step; never a second transcription."""
    if slot == "update_H":
        stepping.update_H(fields, pml)
    else:
        getattr(stepping, slot)(fields, pml)


def _spine_case(fields: Any, pml: Any, label: str) -> Dict[str, Any]:
    """Run the three ordinary kernels in the nonlinear context, in driver order.

    A separate residency object is intentionally shared by all three plans.  The
    relevant dependencies cross every seam in this short sequence: ``update_H``
    reads B after ``step_B``, and ``step_D`` reads H after ``update_H``.  Thus this
    is stronger than three independent one-launch checks while still making the
    narrow claim this added admission needs.
    """
    reference, reference_pml = build(
        tuple(fields.grid.boundaries), ("Ex", "Ey", "Ez"), False, "random",
        seed=0,
    )
    # ``build`` above only gives us a compatible container.  Copy every state that
    # can reach this B/H/D chain so the reference is the exact same nonlinear run,
    # including PML coefficients and both chi volumes.
    for name in (
            "Bx", "By", "Bz", "Dx", "Dy", "Dz", "Ex", "Ey", "Ez",
            "Hx", "Hy", "Hz", "fu_Bx", "fu_By", "fu_Bz", "fu_Dx",
            "fu_Dy", "fu_Dz", "f_w_Hx", "f_w_Hy", "f_w_Hz"):
        getattr(reference, name)[...] = getattr(fields, name)
    reference.set_nonlinear_volumes(
        {component: fields.chi2_for(component)
         for component in ("Ex", "Ey", "Ez")},
        {component: fields.chi3_for(component)
         for component in ("Ex", "Ey", "Ez")})
    reference.h_separated = True

    residency = Residency()
    plans = [(slot, watched, _spine_plan(fields, pml, slot, residency))
             for slot, watched in _SPINE_STEPS]
    assert all(plan is not None for _slot, _watched, plan in plans), (
        label, [(slot, None if plan is None else type(plan).__name__)
                for slot, _watched, plan in plans])

    # The original ordinary builders are the deliberately wrong admission under a
    # nonlinear field.  They MUST refuse: accepting them would make the result
    # depend on table order rather than the separate, inverted spine predicate.
    record = layer_record(fields.grid, pml)
    assert launch.plan_pml_curl(fields, pml, "step_B", Residency(),
                                chunk_bounds=record) is None
    assert launch.plan_pml_curl(fields, pml, "step_D", Residency(),
                                chunk_bounds=record) is None
    assert launch.plan_constitutive(fields, pml, "H", Residency(),
                                    chunk_bounds=record) is None

    residency.sync_in()
    rows: List[Dict[str, Any]] = []
    for slot, watched, plan in plans:
        before = {name: np.array(getattr(fields, name), copy=True) for name in watched}
        plan.run()
        import torch  # noqa: PLC0415

        torch.mps.synchronize()
        _spine_reference(reference, reference_pml, slot)
        residency.sync_out(watched)
        moved = sum(kit.differing(before[name], getattr(reference, name))
                    for name in watched)
        kit.assert_moved(moved, f"spine/{label}/{slot}")
        differing = sum(TALLY.compare(getattr(fields, name), getattr(reference, name))
                        for name in watched)
        row = {
            "slot": slot,
            "moved_words": moved,
            "differing_words": differing,
            "words_compared": int(sum(kit.words(getattr(fields, name)).size
                                      for name in watched)),
            "launches": plan.launches,
            "plan": type(plan).__name__,
        }
        rows.append(row)
        assert plan.launches == 1, row
        assert differing == 0, row
    residency.verify()
    return {"label": label, "steps": rows, "residency_mirrors": len(residency.names)}


def leg_spine(payload: Dict[str, Any], out: str) -> None:
    """The newly admitted nonlinear B/D/H surface, including its E-only ceiling.

    The first three products sweep the per-axis ghost rules.  The fourth installs
    chi3 above the Pade ``update_E`` ceiling: E must refuse by name while this
    spine remains available, because the engine never reads chi2/chi3 in its B/H/D
    arithmetic.  That asymmetric case is the reason this is a separate admission
    rather than a relaxation of the ordinary predicate.
    """
    rows: List[Dict[str, Any]] = []
    payload["legs"]["spine"] = rows
    products: Tuple[Tuple[str, Any, bool], ...] = (
        ("periodic", "periodic", False),
        ("metallic", "metallic", False),
        ("mixed", ("periodic", "metallic", "periodic"), False),
        ("above_update_e_ceiling", "metallic", True),
    )
    for ordinal, (label, boundaries, above_ceiling) in enumerate(products, 1):
        if above_ceiling:
            fields, pml = build_at_magnitude(
                boundaries, ("Ex", "Ey", "Ez"),
                2.0 * nonlinear.CHI_MAGNITUDE_CEILING, order=3, seed=61)
            e_coverage = nonlinear.nonlinear_constitutive_coverage(
                fields, pml, Residency())
            assert not e_coverage.covered
            assert any("ceiling" in reason for reason in e_coverage.reasons), \
                e_coverage.reasons
        else:
            fields, pml = build(boundaries, ("Ex", "Ey", "Ez"), False,
                                "random", seed=41 + ordinal)
        row = _spine_case(fields, pml, label)
        row["above_update_e_ceiling"] = above_ceiling
        rows.append(row)
        payload["legs"]["spine"] = rows
        kit.save(payload, out)
        compared = sum(step["words_compared"] for step in row["steps"])
        differing = sum(step["differing_words"] for step in row["steps"])
        moved = sum(step["moved_words"] for step in row["steps"])
        kit.log(f"[spine] {ordinal}/{len(products)} {label:<22} "
                f"moved={moved:<7} differing={differing:<4} words={compared}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main(argv: Sequence[str]) -> int:
    parser = kit.argument_parser(__doc__ or "")
    args = parser.parse_args(list(argv))
    started = time.time()
    out = os.path.abspath(args.out)

    payload: Dict[str, Any] = {
        "gate": "metal_nonlinear_update_e",
        "family": nonlinear.FAMILY,
        "slot": nonlinear.SLOT,
        "environment": kit.environment_stamp(),
        "subnormal_policy": subnormal.mps_policy_report(),
        "corpus_digest": nonlinear.corpus_digest(),
        "legs": {},
    }
    kit.save(payload, out)

    reasons: List[str] = []
    if not payload["environment"].get("mps_available"):
        reasons.append("no MPS device available")
    policy_reasons = subnormal.mps_policy_reasons()
    if policy_reasons:
        reasons.extend(policy_reasons)
    if reasons:
        return kit.cannot_certify(payload, out, reasons)

    out_dir = os.path.dirname(out)
    kit.provenance(out_dir, {
        "nonlinear_update_e.py": os.path.join(
            API_ROOT, "meep_gpu/metal_kernels/nonlinear_update_e.py"),
        "softfp64.py": os.path.join(API_ROOT, "meep_gpu/metal_kernels/softfp64.py"),
        "stepping.py": os.path.join(API_ROOT, "meep_gpu/stepping.py"),
        "preconditions.py": os.path.join(
            API_ROOT, "meep_gpu/metal_kernels/preconditions.py"),
        "gate": os.path.abspath(__file__),
    }, kernel_sources={
        f"{live}|{q2}|{q3}|{codes}": nonlinear.nonlinear_source(live, q2, q3,
                                                                codes)
        for live, q2, q3, codes in nonlinear.specialisations()[::97]
    }, name="provenance_nonlinear.json")

    legs = (
        ("identity", leg_identity),
        ("linear_reduction", leg_linear_reduction),
        ("mutations", leg_mutations),
        ("ceiling", leg_ceiling),
        ("census", leg_census),
        ("dsqr", leg_dsqr),
        ("spine", leg_spine),
    )
    ran = kit.run_legs(legs, payload, out, kit.wanted_legs(args.legs))

    payload["totals"] = TALLY.report()
    payload["totals"]["cases"] = sum(
        len(rows) for rows in payload["legs"].values() if isinstance(rows, list))
    kit.save(payload, out)
    kit.log(f"  UINT32 WORDS COMPARED    : {TALLY.compared:,}")
    kit.log(f"  UINT32 WORDS DIFFERING   : {TALLY.differing:,}")
    kit.log(f"  WORDS WITHHELD (refused) : {TALLY.withheld:,}")

    return kit.summarize(
        payload, out,
        claim=("the Metal chi2/chi3 Pade update_E kernel, bound to the chunk record, "
               "reproduces stepping.update_E word for word over the swept "
               "configuration space, its binary64 Dsqr included; the transcription "
               "choices that decide those words are byte-visible; the 1e29 "
               "magnitude clause admits below it and refuses by name above it; and "
               "the nonlinear-only B/H/D spine reuses the base PML shaders with "
               "exact bytes and persistent mirror continuity"),
        scope=("ONE sub-step, real float32 storage, active PML, k = 0, unfolded, "
               "Cartesian, no poles and no off-diagonal row; FLUSH policy, which "
               "is the only one this executor can honour. The census leg alone is "
               f"multi-step ({CENSUS_STEPS} complete array-path cycles), because "
               "band entry is a run fact rather than a launch fact. The spine leg "
               "is B -> H -> D only: it intentionally excludes the Pade E body, "
               "which has the identity and mutation evidence above"),
        stated_weakness=(
            "BEHAVIOURAL ONLY: torch.mps.compile_shader exposes no AIR, no GPU "
            "ISA and no optimisation report, so this catches a wrong ANSWER and "
            "never a wrong INSTRUCTION. Single-launch for every leg but the "
            "census — a stale mirror, a seam and an accumulating auxiliary are "
            "invisible here by construction and belong to the whole-step gate. "
            "And a WHOLESALE transverse-partner swap is invisible to any byte "
            "gate, measured here as the partner_squares_order_swapped null: Dsqr "
            "is symmetric in the two sums, so the defect reduces to a commutation."),
        started=started, legs_run=ran,
        compared=TALLY.compared,
        certified=TALLY.differing == 0,
        extra={"cases": payload["totals"]["cases"],
               "census_refused": payload.get("census_refused", [])})


if __name__ == "__main__":
    from metal_gate_runner import run_current_measurement

    raise SystemExit(run_current_measurement(__file__, sys.argv[1:]))
