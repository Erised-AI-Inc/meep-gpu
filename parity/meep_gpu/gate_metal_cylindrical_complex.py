"""BYTE GATE — the COMPLEX cylindrical family (Dcyl, complex64 storage at every m:
m = 0 since 2026-09-04, |m| >= 1 before) on Metal.

WHAT THIS CERTIFIES: that ``metal_kernels.cylindrical_complex``'s four products
reproduce ``stepping.py`` WORD FOR WORD as uint32 on a complex64 Dcyl grid — the
new curl on both sub-steps and the CERTIFIED complex constitutive pair re-admitted
under this family's restated predicate — per SUB-STEP **and per COMPLETE STEP**,
over a swept case matrix, with the transcription choices that decide those words
carried as ARMED MUTATIONS rather than as comments.

WHY THE COMPLETE STEP IS THE ARBITER HERE AND NOT A BONUS. Six green per-sub-step
comparisons say nothing about the object the engine would run. This family's
complete step is SIX passes on a walled run (``driver.py:3281-3287`` / ``:3292-3302``
with the two fold passes inert on a Dcyl grid):

    step_B -> zero_metal_B -> update_H -> step_D -> zero_metal_D -> update_E

and three failure classes live ONLY there:

* **a STALE MIRROR** — and this family is the one that makes that class concrete
  rather than theoretical. Its radial prefix is a SEQUENTIAL HOST SCAN, so every
  launch pulls the prefix SOURCE off the device, scans it on the host and pushes the
  prefix back (``CylindricalComplexCurlPlan.refresh_prefix``). Drop the pull and the
  SUB-STEP route is byte-perfect at ANY launch count — measured, 0 differing words at
  one launch and at six, on both sub-steps, over every case — because a curl-only
  loop never writes its own prefix source. In a complete step the same defect is
  wrong from step 0, because ``update_H`` writes ``Hy`` on the device and ``step_D``
  scans it. Leg ``mutations`` arms it and reports both numbers;
* **a SEAM** — ``zero_metal_B``/``zero_metal_D`` clear stored cell 0 of the walled
  axis BETWEEN the curl and the constitutive pass, on the HOST, while the device
  mirror is authoritative. Three arms: the pass dropped, the pass run without its
  sync bracket, and the pass moved to the wrong side of the constitutive pass;
* **an ACCUMULATING AUXILIARY** — ``fu_*`` and ``f_w_*`` are STATE. The |m| >= 2 rule
  zeroes the auxiliaries AFTER the recurrence, which is precisely the kind of write a
  single launch cannot separate from a no-op.

WHAT THIS GATE DOES NOT CERTIFY, each stated rather than papered over:

* **it is BEHAVIOURAL, not generated-code evidence.** ``torch.mps.compile_shader``
  exposes no AIR, no GPU ISA and no optimisation report (``device.py:46-62``), so
  every leg catches a wrong ANSWER, never a wrong INSTRUCTION. That is this
  backend's standing certification gap against the Triton and CUDA tracks and it
  stays stated on every claim built on top of it;
* **it runs under the FLUSH policy and nothing else is offerable.** MPS flushes
  float32 subnormals natively and exposes no lever, so ``keep`` is NOT OFFERABLE and
  the honest third policy value is REFUSE. Every claim rides a CHECKED
  subnormal-free precondition, reported as a WINDOW — ``[first_step, last_step]`` —
  because band entry was measured to be a RUN-and-WINDOW fact, not a family fact. A
  case whose census fires is REFUSED BY NAME and its comparisons are withheld from
  the certified total: a coverage refusal, not a failure;
* **it makes NO throughput claim.** The host prefix scan hands back part of what the
  residency layer buys, and this family's number may not be inherited from the
  certified families' benchmark;
* **it does not re-cut the 759-slot coverage number**, and the Metal-vs-Triton
  admission diff is REFUSED ON THIS HOST rather than fudged — see leg
  ``containment`` for the measurement that establishes why.

THE LEGS:

1.  ``execution``     provenance, the probe-bound arm, and every specialisation
                      COMPILED in both contraction modes.
2.  ``curl``          per sub-step, the new kernel against ``stepping.step_B`` /
                      ``step_D``. Words, never ``allclose``; a vacuity floor on
                      every case.
3.  ``constitutive``  per side, the CERTIFIED complex body under this family's
                      admission, plus the assertion that this family emits NO
                      constitutive source of its own.
4.  ``axis``          the r = 0 seam: the near ghost MEASURED on the array path
                      (a nonzero count is a legitimate outcome and a refusal), the
                      far ghost asserted as the source identity it is, and a floor
                      on how many ROW-0 words the reference actually moved.
5.  ``whole_step``    the complete driver step, twelve cycles, per-slot launch
                      counters and a per-step subnormal window.
6.  ``precondition``  the same per-step census driven over a state scaled INTO the
                      band, REQUIRED to fire. A precondition never demonstrated to
                      fire is decoration.
7.  ``split``         the m = 0 versus |m| >= 1 boundary, as REFUSALS: clause 14 by
                      name, ``m_class(0)`` raising, the plan returning None and the
                      composer selecting nothing from this family.
8.  ``containment``   Metal's admitted set beside Triton's, every slot named.
9.  ``mutations``     armed, launch-counted, caught-of-armed reported honestly.

    python -u gate_metal_cylindrical_complex.py --out results/.../gate.json
"""

from __future__ import annotations

import inspect
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
API_ROOT = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
if API_ROOT not in sys.path:
    sys.path.insert(0, API_ROOT)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# THE POLICY IS SET BEFORE ANY `meep_gpu` MODULE IS REACHED, because the predicates
# read it at import. On MPS the subnormal flush is native and has no lever, so the
# resolved default is not offerable and every Metal predicate refuses by name.
# Setting `flush` is what puts the claim under a CHECKED precondition rather than
# under a pretence.
os.environ.setdefault("MEEP_GPU_SUBNORMAL_POLICY", "flush")

import metal_composition_matrix as matrix  # noqa: E402

ENVIRONMENT = matrix.prepare_environment()

from meep_gpu import stepping  # noqa: E402
from meep_gpu.metal_kernels import (  # noqa: E402
    complex_fields,
    coverage as metal_coverage,
    cylindrical_complex as cyl,
    launch,
    preconditions,
    shaders,
    subnormal,
    templates,
)
from meep_gpu.metal_kernels.device import Residency, compile_source  # noqa: E402
from meep_gpu.triton_kernels.coverage import CONSTITUTIVE_SIDES  # noqa: E402

import metal_gate_kit as kit  # noqa: E402

log, save, differing, words = kit.log, kit.save, kit.differing, kit.words
needle = kit.needle

PROBE = cyl.load_expansion_probe()
EXPANSION = cyl.expansion_from_probe(PROBE) if PROBE else None

#: Every volume one complete step can touch. ``fu_*`` and ``f_w_*`` are STATE: a
#: kernel right for one launch and wrong forever after diverges only once they
#: accumulate, which is why the whole-step leg compares all of it every step.
STATE: Tuple[str, ...] = (
    "Bx", "By", "Bz", "Dx", "Dy", "Dz", "Ex", "Ey", "Ez", "Hx", "Hy", "Hz",
    "fu_Bx", "fu_By", "fu_Bz", "fu_Dx", "fu_Dy", "fu_Dz",
    "f_w_Hx", "f_w_Hy", "f_w_Hz", "f_w_Ex", "f_w_Ey", "f_w_Ez")

#: The seam order one complete driver step runs in, taken from the composer's OWN
#: residency table rather than re-listed, so the walk and the residency model cannot
#: disagree about what a step is. ``update_P`` is dropped: no Metal product carries
#: it and no case here registers a polarization.
SEAM_ORDER: Tuple[str, ...] = tuple(
    name for name in metal_coverage.RESIDENCY_ORDER if name != "update_P")

#: Which array-path function each seam slot is.
ARRAY_PATH: Dict[str, Callable[[Any, Any], None]] = {
    "step_B": lambda f, p: stepping.step_B(f, p),
    "update_H": lambda f, p: stepping.update_H(f, p),
    "step_D": lambda f, p: stepping.step_D(f, p),
    "update_E": lambda f, p: stepping.update_E(f, p),
    "zero_metal_B": lambda f, p: stepping.zero_metal_B(f),
    "zero_metal_D": lambda f, p: stepping.zero_metal_D(f),
    "fill_B": lambda f, p: stepping.fill_symmetry_bc_B(f),
    "fill_D": lambda f, p: stepping.fill_symmetry_bc_D(f),
    "fill_folded_far_ghosts_B": lambda f, p: stepping.fill_folded_far_ghosts_B(f),
    "fill_folded_far_ghosts_D": lambda f, p: stepping.fill_folded_far_ghosts_D(f),
}

#: The budget every whole-step case runs. Twelve rather than four for the reason the
#: certified whole-step gate uses twelve: the classes this leg exists for COMPOUND,
#: and a defect that needs three steps to reach the low bits is exactly the kind a
#: short budget reports as green. The comparison is per COMPLETE STEP, so the budget
#: also bounds ``first_divergent``.
CYCLES = 12

#: THE CASE MATRIX. Every axis is here because some mutation is REACHABLE on one
#: value and a structural NULL on the other:
#:
#:   M CLASS     |m| = 1 compiles the axis-row REPLACEMENT and |m| >= 2 compiles the
#:               near-axis zeroing of six volumes — different bodies, which is why
#:               ``M_CLASS`` stays a compile-time branch where ``zero_rows`` became a
#:               runtime uniform;
#:   SIGN OF m   the axis-increment scalar ``1j * (m*dtdx)`` carries a ``-0.0`` REAL
#:               WORD at m < 0 and a ``+0.0`` at m > 0, and the i*m/r row launders the
#:               opposite way. A single-sign matrix measures one of the two;
#:   z KIND      METALLIC drives the other ``BCZ`` arm AND makes the ``zero_metal_*``
#:               SEAM live; PERIODIC has no wall pass at all, so every seam mutation
#:               is a structural null there and is scoped rather than counted;
#:   ACCURATE    ``accurate_fields_near_cylorigin`` binds a DIFFERENT zero-row uniform
#:               at the SAME m (1 instead of |m|). If the count were a compile-time
#:               constant this would be a different kernel;
#:   ZERO ROWS   |m| = 3 and |m| = 5 put the near-axis threshold at 3 and 5, so a
#:               literal baked at 2 is right at m = 2 and wrong elsewhere;
#:   COURANT     a non-power-of-two dtdx, so nothing passes because every scale factor
#:               happened to be exact;
#:   SIGNED ZERO the class the c_mul zero cross terms and the REPLACE-versus-ACCUMULATE
#:               axis rule live in. Random-normal state does not contain it, and a
#:               defect invisible on the state a gate happens to build is reported as
#:               uncaught — which is the most dangerous row a mutation leg can print.
#:   m = 0       the THIRD BODY (2026-09-04): no i*m/r block, no increment, and the
#:               m = 0 axis rules on the stored registers. Its rows cover both z
#:               terminations, the corpus row's own power-of-two Courant (0.5) and a
#:               non-power-of-two one (the `4*Courant` post-add scalar is exact only
#:               at the former), planted signed zeros (the axis clears and the
#:               post-add on a `-0.0` word), and the ACCURATE flag — which has NO
#:               effect at m = 0 (`_cylindrical_axis_rows` is consulted for |m| >= 2
#:               only) and is driven to show that rather than to assume it.
CASES: Tuple[Tuple[str, Dict[str, Any]], ...] = (
    ("m1_metallic", dict(m=1, z_kind="metallic")),
    ("m1_periodic", dict(m=1, z_kind="periodic")),
    ("m1_metallic_signed_zeros", dict(m=1, z_kind="metallic", plant=True)),
    ("m_minus1_metallic", dict(m=-1, z_kind="metallic")),
    ("m1_odd_courant", dict(m=1, z_kind="metallic", courant=0.2718281828)),
    ("m2_metallic", dict(m=2, z_kind="metallic")),
    ("m_minus2_periodic", dict(m=-2, z_kind="periodic")),
    ("m3_metallic", dict(m=3, z_kind="metallic")),
    ("m3_metallic_accurate", dict(m=3, z_kind="metallic", accurate=True,
                                  courant=0.25)),
    ("m5_periodic_signed_zeros", dict(m=5, z_kind="periodic", plant=True)),
    ("m1_metallic_axis_seam", dict(m=1, z_kind="metallic", axis_seam=True)),
    ("m0_metallic", dict(m=0, z_kind="metallic")),
    ("m0_periodic", dict(m=0, z_kind="periodic")),
    ("m0_metallic_half_courant", dict(m=0, z_kind="metallic", courant=0.5)),
    ("m0_odd_courant", dict(m=0, z_kind="metallic", courant=0.2718281828)),
    ("m0_metallic_signed_zeros", dict(m=0, z_kind="metallic", plant=True)),
    ("m0_metallic_accurate", dict(m=0, z_kind="metallic", accurate=True)),
)

#: The four signed-zero sign combinations, plus two mixed words. Scattered ON TOP of
#: the random state rather than replacing it, so the class is reachable without
#: flattening the row-to-row variation the index needles need.
SIGN_ZERO_PLANTS = np.array(
    [complex(0.0, 0.0), complex(-0.0, -0.0), complex(0.0, -0.0), complex(-0.0, 0.0),
     complex(1.5, -0.0), complex(-0.0, 1.5)], dtype=np.complex64)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def plant_signed_zeros(fields: Any) -> int:
    """Scatter every signed-zero sign combination through the stored state.

    ON TOP of the random fill and SPARSE, deliberately. The class must be present —
    ``c_mul``'s literal zero cross terms and the |m| = 1 REPLACE-versus-ACCUMULATE
    rule are both signed-zero facts and a random-normal state contains no zero at all
    — and the rows must still differ from one another, because half this gate's
    needles are index errors ("read one row over") whose divergence disappears if a
    plant makes neighbouring rows identical.

    THE AXIS ROW AND ITS FIRST OFF-AXIS NEIGHBOUR ARE PLANTED EXPLICITLY: they are
    what the |m| = 1 increment reads (``g2[nyz + j*nzi + k]``) and what the per-|m|
    rules write, so a scatter that missed them would leave this family's own rows the
    only ones without the class.
    """
    planted = 0
    for offset, name in enumerate(STATE):
        volume = getattr(fields, name, None)
        if volume is None or volume.dtype.kind != "c":
            continue
        nr, ny, nz = volume.shape
        for index, value in enumerate(SIGN_ZERO_PLANTS):
            for row in range(min(2, nr)):          # the axis row and its neighbour
                volume[row, 0, (index * 3 + offset) % nz] = value
                planted += 1
            volume[(index * 5 + offset) % nr, 0, (index * 7 + offset) % nz] = value
            planted += 1
    return planted


#: The z lane the axis-seam plant is written on. Any lane with a z-UP neighbour will
#: do; it is a constant so the plant and its tag cannot drift apart.
AXIS_SEAM_LANE = 5


def plant_axis_seam(fields: Any) -> Dict[str, Any]:
    """CONSTRUCT the one state in which REPLACE and ACCUMULATE differ.

    THIS IS THE PLANT THAT MADE A DEAD CLAIM MEASURABLE, and the arithmetic is worth
    stating because nothing about it is reachable by accident.

    The |m| = 1 axis-row increment REPLACES curl row 0 rather than accumulating onto
    it. The masked row is exactly ``+0.0`` and ``+0.0 + x == x`` for every x EXCEPT
    ``-0.0``, so the two spellings differ ONLY where the increment's real word is an
    exact ``+0.0`` AND the recurrence operand it meets is an exact ``-0.0``. Measured
    on the matrix's random and signed-zero-scattered states: 0 of 5 — the defect was
    ARMED AND UNREACHED, which is a leg that certifies nothing about the claim it
    names.

    So the pair is built:

    * ``Ey[0,0,k] = -0.0 + 1.0j`` and ``Ey[0,0,k+1] = +0.0 + 0.5j`` make the z
      difference's real word exactly ``-0.0``, which ``(-dtdx) * d`` turns into
      ``+0.0``;
    * ``Ez[1,0,k] = 0`` makes the ``(1j*m*dtdx) * Ez[r+1]`` product's real word
      exactly ``+0.0``, so the increment's real word is ``+0.0 - +0.0 = +0.0``;
    * ``fu_Bx[0,0,k] = -0.0 + 1.0j`` makes the recurrence's minuend real word
      ``-0.0``, which is the only operand that can tell ``-(+0.0)`` from
      ``+0.0 - (+0.0)``.

    Measured with this plant: the shipped kernel stays byte-identical to the array
    path (0 differing of 3,801 moved) and the accumulating mutant misses by exactly
    ONE WORD — the real half of ``fu_Bx[0,0,k]``. One word is the whole class.
    """
    k = AXIS_SEAM_LANE
    fields.Ey[0, 0, k] = complex(-0.0, 1.0)
    fields.Ey[0, 0, k + 1] = complex(0.0, 0.5)
    fields.Ez[1, 0, k] = complex(0.0, 0.0)
    fields.fu_Bx[0, 0, k] = complex(-0.0, 1.0)
    fields.Bx[0, 0, k] = complex(0.25, -0.5)
    return {"lane": k, "words_planted": 10}


def build(label: str, seed: int = 11) -> Tuple[Any, Any]:
    """A Dcyl ``(Fields, PML)`` pair seeded in the PHYSICAL BAND, everywhere.

    NOT COSMETIC, and not the composition matrix's fill either. ``matrix.cylindrical``
    seeds the twelve field volumes; this seeds the ``fu_*`` and ``f_w_*`` recurrence
    state as well, because zero init is a FIXED POINT of half of what this family
    does — the near-axis rule writes zeros over zeros, the recurrence carries a zero
    ``fprev`` — and a no-op agreeing with a no-op is trivially identical.
    """
    options = dict(CASES)[label]
    plant = bool(options.get("plant"))
    seam = bool(options.get("axis_seam"))
    kwargs = {k: v for k, v in options.items() if k not in ("plant", "axis_seam")}
    fields, pml = matrix.cylindrical(**kwargs)
    rng = np.random.default_rng(seed)
    for name in STATE:
        array = getattr(fields, name, None)
        if array is None:
            continue
        if array.dtype.kind == "c":
            array.real[...] = rng.standard_normal(array.shape).astype(np.float32)
            array.imag[...] = rng.standard_normal(array.shape).astype(np.float32)
        else:
            array[...] = rng.standard_normal(array.shape).astype(array.dtype)
    if plant:
        plant_signed_zeros(fields)
    if seam:
        plant_axis_seam(fields)
    return fields, pml


def tags(fields: Any, pml: Any) -> Tuple[str, ...]:
    """What a case IS, derived from the ENGINE rather than typed beside the row.

    Every mutation scopes itself with these. Typing them into the matrix would put
    this family's two structural splits — the m class and whether a wall pass exists
    — in a second place, and a scoping that disagreed with the grid would silently
    turn a reachable defect into a NEEDLE-MISSED row.
    """
    grid = fields.grid
    m = int(grid.m)
    accurate = bool(grid.accurate_fields_near_cylorigin)
    rows = cyl.zero_rows(m, accurate)
    found = [{cyl.M_ZERO: "m_zero", cyl.M_ONE: "m_one",
              cyl.M_MANY: "m_many"}[cyl.m_class(m)],
             "m_nonzero" if m != 0 else "m_is_zero",
             "negative_m" if m < 0 else "positive_m",
             "walled" if bool(grid.has_metallic) else "unwalled"]
    if accurate:
        found.append("accurate")
    # THE TAG THE "BAKED LITERAL" NEEDLE NEEDS, and it is not the same as `m_many`.
    # A count baked at 2 is RIGHT wherever the true count IS 2, so scoping that
    # mutation to the whole |m| >= 2 arm would report a structural null as a partial
    # catch — measured: 2/4, with the two misses exactly the |m| = 2 rows.
    if abs(m) >= 2 and rows != 2:
        found.append("zero_rows_not_two")
    if carries_axis_seam(fields):
        found.append("axis_seam")
    return tuple(found)


def carries_axis_seam(fields: Any) -> bool:
    """Whether :func:`plant_axis_seam` was written into this state.

    The seam plant is a STATE fact and not a grid fact, so it cannot be derived the
    way every other tag is. It is recovered from the plant's OWN SIGNATURE — the two
    ``Ey`` real words, ``-0.0`` beside ``+0.0`` — which keeps the tag a property of
    the object in hand rather than of a label the caller carried along beside it. A
    tag that disagreed with the state would turn a reachable defect into a
    NEEDLE-MISSED row, which is the failure scoping exists to avoid.
    """
    ey = getattr(fields, "Ey", None)
    if ey is None or ey.shape[0] < 2 or ey.shape[2] <= AXIS_SEAM_LANE + 1:
        return False
    return bool(words(ey[0, 0, AXIS_SEAM_LANE])[0] == np.uint32(0x80000000)
                and words(ey[0, 0, AXIS_SEAM_LANE + 1])[0] == np.uint32(0))


def snapshot(fields: Any, names: Sequence[str] = STATE) -> Dict[str, Any]:
    return {name: np.array(getattr(fields, name), copy=True) for name in names
            if getattr(fields, name, None) is not None}


def restore(fields: Any, state: Dict[str, Any]) -> None:
    for name, array in state.items():
        getattr(fields, name)[...] = array


def curl_names(sub_step: str) -> Tuple[str, ...]:
    spec = cyl.SUB_STEPS[sub_step]
    return tuple(spec["targets"]) + tuple("fu_" + n for n in spec["targets"])


def constitutive_names(side: str) -> Tuple[str, ...]:
    spec = CONSTITUTIVE_SIDES[side]
    return tuple(spec["targets"]) + tuple(spec["aux"])


def oracle(fields: Any, apply: Callable[[], None],
           names: Sequence[str]) -> Tuple[Dict[str, Any], Dict[str, Any], int]:
    """Run the ARRAY PATH, capture what it wrote, and put the state back.

    Returns ``(before, after, moved)``. ``moved`` is the vacuity floor every leg
    asserts: a step that wrote nothing agrees with a kernel that wrote nothing.
    """
    before = snapshot(fields, names)
    apply()
    after = snapshot(fields, names)
    restore(fields, before)
    moved = sum(differing(before[n], after[n]) for n in names)
    return before, after, moved


def compared_words(state: Dict[str, Any]) -> int:
    return sum(int(words(array).size) for array in state.values())


def divergence(fields: Any, after: Dict[str, Any]) -> Dict[str, int]:
    return {name: differing(getattr(fields, name), array)
            for name, array in after.items()}


def state_census(fields: Any) -> int:
    """How many stored words sit in the subnormal band, read off the BITS.

    A value comparison would be answered by the very flushing this counts.
    """
    return sum(subnormal.census(getattr(fields, name))
               for name in STATE if getattr(fields, name, None) is not None)


# ---------------------------------------------------------------------------
# Device runners — ONE plan-assembly route, so the bytes the gate certifies are
# the bytes the engine would launch
# ---------------------------------------------------------------------------

def run_curl(fields: Any, pml: Any, sub_step: str,
             functions: Optional[Dict[str, Any]] = None,
             launches: int = 1) -> Any:
    """Launch the cylindrical complex curl through the ENGINE route (or a mutant).

    ``functions`` is the MUTATION SEAM and it forces the bare-array route, exactly as
    the shipped module documents. Dropping the argument would not be a slowdown but a
    silent DISARMING: every mutation leg would launch the shipped kernel and report
    its defect as uncaught.
    """
    residency = Residency()
    if functions is None:
        plan = cyl.plan_cylindrical_complex_pml_curl(fields, pml, sub_step,
                                                     residency, probe=PROBE)
        if plan is None:
            raise AssertionError(cyl.cylindrical_complex_pml_curl_coverage(
                fields, pml, sub_step, residency, PROBE).reasons)
    else:
        spec = cyl.SUB_STEPS[sub_step]
        grid = fields.grid
        kinds = stepping._boundary_kinds(grid, pml)
        bcz = templates.METALLIC if kinds[2] == "metallic" else templates.PERIODIC
        names = (tuple(spec["targets"]) + tuple("fu_" + n for n in spec["targets"])
                 + tuple(spec["sources"]))
        plan = cyl.plan_cylindrical_complex_pml_curl_from_arrays(
            sub_step, {n: getattr(fields, n) for n in names},
            {f"{stem}_{axis}": getattr(pml, f"{stem}_{axis}{spec['suffix']}")
             for axis in "xyz" for stem in ("kms", "sinv")},
            bcz, int(grid.m), bool(grid.accurate_fields_near_cylorigin),
            grid.dt / grid.dx, EXPANSION, residency, functions=functions)
    residency.sync_in()
    for _ in range(launches):
        plan.run()
    residency.sync_out()
    return plan


def run_constitutive(fields: Any, pml: Any, side: str) -> Any:
    residency = Residency()
    plan = cyl.plan_cylindrical_complex_constitutive(fields, pml, side, residency,
                                                     probe=PROBE)
    if plan is None:
        raise AssertionError(cyl.cylindrical_complex_constitutive_coverage(
            fields, pml, side, residency, PROBE).reasons)
    residency.sync_in()
    plan.run()
    residency.sync_out()
    return plan


def live_passes(fields: Any, pml: Any) -> Tuple[str, ...]:
    """Which seam slots this run actually executes — the COMPOSER's own model.

    An unreadable live set is a refusal rather than an empty tuple: a walk that
    stepped nothing would compare a no-op with a no-op and pass.
    """
    live = launch.live_sub_steps(fields, pml, ())
    assert live is not None, (
        "the live sub-step set is unreadable for this configuration; the walk would "
        "silently step a subset and the comparison would certify it")
    assert "update_P" not in live, (
        "this case registers a polarization: no Metal product carries update_P and "
        "this gate has no array-path entry for it, so the walked step would not be "
        "a complete one")
    return tuple(name for name in SEAM_ORDER if name in set(live))


def compose(fields: Any, pml: Any) -> Tuple[Any, Any, Tuple[str, ...],
                                            Tuple[str, ...]]:
    """Plan the step TWICE, and the second time is not redundant.

    The residency verdict needs the SYNCED set — which array-path passes the caller
    brackets with an explicit sync out and back — and no caller knows that set until
    it knows which slots the composer filled. So the first call is a throwaway that
    answers "which slots are mine", and the second is the composition this gate runs.
    """
    live = live_passes(fields, pml)
    scout = launch.plan_step(fields, pml, residency=Residency(), sources=(),
                             cylindrical_complex_probe=PROBE)
    on_device = set(scout.plans)
    synced = tuple(name for name in live if name not in on_device)

    residency = Residency()
    plan = launch.plan_step(fields, pml, residency=residency, sources=(),
                            synced=synced, cylindrical_complex_probe=PROBE)
    return plan, residency, live, synced


def walk(fields: Any, pml: Any, plan: Any, residency: Any, live: Sequence[str],
         skip: Sequence[str] = (), unsynced: Sequence[str] = (),
         order: Optional[Sequence[str]] = None) -> None:
    """One complete driver step, Metal where the composer filled the slot.

    A slot the composer did NOT fill runs on the ARRAY PATH and is bracketed with an
    explicit ``sync_out`` / ``sync_in``. That bracket is the residency clause made
    operational; ``unsynced`` removes it for one named pass, which is how leg
    ``mutations`` measures that it is load-bearing rather than ceremonial.
    """
    for slot in (live if order is None else order):
        if slot in skip:
            continue
        if slot in plan.plans:
            plan.plans[slot].run()
            continue
        if slot in unsynced:
            ARRAY_PATH[slot](fields, pml)       # THE MUTATION: no sync bracket
            continue
        residency.sync_out()
        ARRAY_PATH[slot](fields, pml)
        residency.sync_in()


def reference_step(fields: Any, pml: Any, live: Sequence[str],
                   skip: Sequence[str] = (),
                   order: Optional[Sequence[str]] = None) -> None:
    for slot in (live if order is None else order):
        if slot in skip:
            continue
        ARRAY_PATH[slot](fields, pml)


# ---------------------------------------------------------------------------
# LEG execution
# ---------------------------------------------------------------------------

def leg_execution(payload: Dict[str, Any], out: str) -> None:
    """What was hashed, which arm was bound, and does every specialisation BUILD.

    A specialisation that fails to COMPILE is a crash at plan time on a configuration
    nobody swept, so the whole enumeration is built here — in BOTH contraction modes,
    because the contraction pragma is the one directive this backend's byte identity
    rests on and a mode that failed to compile would silently never be exercised.
    """
    sources = cyl.enumerate_cylindrical_sources(EXPANSION)
    kit.provenance(os.path.dirname(os.path.abspath(out)), {
        "cylindrical_complex.py": os.path.join(
            API_ROOT, "meep_gpu/metal_kernels/cylindrical_complex.py"),
        "complex_fields.py": os.path.join(
            API_ROOT, "meep_gpu/metal_kernels/complex_fields.py"),
        "templates.py": os.path.join(API_ROOT, "meep_gpu/metal_kernels/templates.py"),
        "shaders.py": os.path.join(API_ROOT, "meep_gpu/metal_kernels/shaders.py"),
        "launch.py": os.path.join(API_ROOT, "meep_gpu/metal_kernels/launch.py"),
        "stepping.py": os.path.join(API_ROOT, "meep_gpu/stepping.py"),
        "gate": os.path.abspath(__file__),
    }, kernel_sources=sources, name="provenance_gate.json")

    built = 0
    for mode in shaders.CONTRACT_MODES:
        for source in cyl.enumerate_cylindrical_sources(EXPANSION, mode).values():
            compile_source(source)
            built += 1
    kit.assert_moved(built, "no specialisation compiled", floor=2 * len(sources))
    assert len(sources) == 12, (
        f"the enumeration is {len(sources)} sources, not 12. The whole point of the "
        f"runtime zero-rows uniform is that BACKWARD x BCZ x M_CLASS is CLOSED "
        f"(three m classes since 2026-09-04); a different count means an axis "
        f"became a specialisation again")

    payload["legs"]["execution"] = {
        "expansion": EXPANSION,
        "probe_patterns": (PROBE or {}).get("patterns"),
        "probe_backend": (PROBE or {}).get("backend"),
        "specialisations": len(sources),
        "compiled": built,
        "contract_modes": list(shaders.CONTRACT_MODES),
        "environment": ENVIRONMENT,
        "cases": [label for label, _ in CASES],
        "cycles": CYCLES,
        "no_generated_code_audit": (
            "torch.mps.compile_shader exposes no AIR, no GPU ISA and no optimisation "
            "report, so this gate cannot refuse a compile whose emitted code violates "
            "the policy nor establish that the contraction guard was obeyed. Every "
            "leg here is BEHAVIOURAL: it catches a wrong answer, never a wrong "
            "instruction."),
    }
    save(payload, out)
    log(f"[execution] arm={EXPANSION} specialisations={len(sources)} "
        f"compiled={built}")


# ---------------------------------------------------------------------------
# LEG curl — per sub-step, which is where the four cylindrical additions live
# ---------------------------------------------------------------------------

def leg_curl(payload: Dict[str, Any], out: str) -> None:
    rows: List[Dict[str, Any]] = []
    for label, _ in CASES:
        fields, pml = build(label)
        for sub_step in ("step_B", "step_D"):
            names = curl_names(sub_step)
            before, after, moved = oracle(
                fields, lambda: getattr(stepping, sub_step)(fields, pml), names)
            kit.assert_moved(moved, f"{label}/{sub_step} reference barely moved",
                             floor=64)
            plan = run_curl(fields, pml, sub_step)
            assert plan.launches == 1, (label, sub_step, plan.launches)
            assert plan.prefix_syncs == 1, (label, sub_step, plan.prefix_syncs)
            per = divergence(fields, after)
            row = {"case": label, "sub_step": sub_step, "tags": tags(fields, pml),
                   "m": int(fields.grid.m), "bcz": plan.bcz, "m_arm": plan.m_arm,
                   "zero_rows": plan.zero_rows, "moved": moved,
                   "compared": compared_words(after),
                   "differing": sum(per.values()), "per_target": per,
                   "prefix_syncs": plan.prefix_syncs, "launches": plan.launches,
                   "digest": kit.state_digest(after)}
            rows.append(row)
            payload["legs"]["curl"] = rows
            save(payload, out)
            log(f"[curl] {label:<26} {sub_step} m_arm={plan.m_arm} "
                f"zrows={plan.zero_rows} moved={moved} "
                f"differing={sum(per.values())}")
            restore(fields, before)
    payload["legs"]["curl"] = rows
    save(payload, out)


# ---------------------------------------------------------------------------
# LEG constitutive — no new device code, so the claim is the ADMISSION's
# ---------------------------------------------------------------------------

def leg_constitutive(payload: Dict[str, Any], out: str) -> None:
    """The CERTIFIED complex body under this family's restated predicate.

    That this family emits NO constitutive source is ASSERTED rather than described:
    the enumeration carries no constitutive label, so an edit that added a body here
    would show up in the count the execution leg records.
    """
    assert not any("constitutive" in label for label in
                   cyl.enumerate_cylindrical_sources(EXPANSION)), (
        "this family emitted a constitutive source: it is supposed to re-admit "
        "complex_fields.bloch_constitutive_step, not replace it")
    rows: List[Dict[str, Any]] = []
    for label, _ in CASES:
        fields, pml = build(label)
        for side, sub_step in (("H", "update_H"), ("E", "update_E")):
            names = constitutive_names(side)
            before, after, moved = oracle(
                fields, lambda: getattr(stepping, sub_step)(fields, pml), names)
            kit.assert_moved(moved, f"{label}/{side} reference barely moved",
                             floor=64)
            plan = run_constitutive(fields, pml, side)
            assert plan.launches == 1, (label, side, plan.launches)
            assert isinstance(plan, complex_fields.ComplexConstitutivePlan), type(plan)
            per = divergence(fields, after)
            rows.append({"case": label, "side": side, "tags": tags(fields, pml),
                         "moved": moved, "compared": compared_words(after),
                         "differing": sum(per.values()), "per_target": per,
                         "plan_class": type(plan).__name__})
            payload["legs"]["constitutive"] = rows
            save(payload, out)
            log(f"[constitutive] {label:<26} {side} moved={moved} "
                f"differing={sum(per.values())}")
            restore(fields, before)
    payload["legs"]["constitutive"] = rows
    save(payload, out)


# ---------------------------------------------------------------------------
# LEG axis — the r = 0 seam, measured rather than assumed
# ---------------------------------------------------------------------------

def leg_axis(payload: Dict[str, Any], out: str) -> None:
    """The two halves of the kernel's ``BCX = METALLIC`` choice, established DIFFERENTLY.

    ``stepping._boundary_kinds`` returns ``CYL_AXIS`` on r and the kernel compiles
    METALLIC anyway. That is two claims, not one:

    * the FAR ghost is a SOURCE IDENTITY — ``_shift_up`` is ONE branch over
      ``(MIRROR, METALLIC, CYL_AXIS)`` writing the same hard zero for all three.
      Nothing to measure; the branch TEXT is asserted, so a future split of it fails
      here rather than in a field;
    * the NEAR ghost is NOT. ``_shift_down``'s ``CYL_AXIS`` branch writes the
      ``r_to_minus_r`` image where METALLIC writes zero, and the kernel's claim is
      that the ownership mask zeroes the only row that can consume it. That is a
      claim about ``stepping.py``, so it is MEASURED: the real sub-step against the
      same sub-step with ``boundaries[0]`` forced metallic. **A NONZERO COUNT IS A
      LEGITIMATE OUTCOME** and would mean this family must carry the near ghost; the
      leg refuses rather than relaxing.

    The third row is a VACUITY FLOOR nobody else carries: how many words the
    reference moved ON ROW 0. Every cylindrical addition except the prefix acts on
    that row, and a matrix whose reference never touched it would certify the r = 0
    machinery by never exercising it.
    """
    source = inspect.getsource(stepping._shift_up)
    far_identity = "if boundary in (MIRROR, METALLIC, CYL_AXIS):" in source
    assert far_identity, (
        "stepping._shift_up no longer serves CYL_AXIS from the METALLIC branch; the "
        "cylindrical kernel's BCX = METALLIC specialisation rests on that identity "
        "and this gate's whole r-axis argument is void")

    original = stepping._boundary_kinds
    rows: List[Dict[str, Any]] = []
    for label, _ in CASES:
        for sub_step in ("step_B", "step_D"):
            fields_a, pml_a = build(label, seed=23)
            fields_b, pml_b = build(label, seed=23)
            names = curl_names(sub_step)
            before = snapshot(fields_a, names)
            getattr(stepping, sub_step)(fields_a, pml_a)

            def forced(grid, pml, _original=original):
                return (stepping.METALLIC,) + tuple(_original(grid, pml)[1:])

            stepping._boundary_kinds = forced
            try:
                getattr(stepping, sub_step)(fields_b, pml_b)
            finally:
                stepping._boundary_kinds = original

            moved = sum(differing(before[n], getattr(fields_a, n)) for n in names)
            near_ghost = sum(differing(getattr(fields_a, n), getattr(fields_b, n))
                             for n in names)
            row_zero = sum(differing(before[n][0], getattr(fields_a, n)[0])
                           for n in names)
            kit.assert_moved(moved, f"{label}/{sub_step} axis leg reference")
            kit.assert_census_floor(
                row_zero, f"{label}/{sub_step} ROW-0 words the reference moved")
            rows.append({"case": label, "sub_step": sub_step,
                         "tags": tags(fields_a, pml_a),
                         "moved": moved, "row_zero_words_moved": row_zero,
                         "near_ghost_differing": near_ghost,
                         "far_ghost_source_identity": far_identity})
            payload["legs"]["axis"] = rows
            save(payload, out)
            log(f"[axis] {label:<26} {sub_step} row0_moved={row_zero} "
                f"near_ghost_differing={near_ghost}")
            assert near_ghost == 0, (
                f"{label}/{sub_step}: the r = 0 NEAR ghost is OBSERVABLE "
                f"({near_ghost} words). The kernel compiles BCX = METALLIC and this "
                f"family would have to carry the r_to_minus_r image — a legitimate "
                f"outcome, and a refusal rather than a test to relax")
    payload["legs"]["axis"] = rows
    save(payload, out)


# ---------------------------------------------------------------------------
# LEG whole_step — the real arbiter
# ---------------------------------------------------------------------------

def leg_whole_step(payload: Dict[str, Any], out: str, budget: int = CYCLES) -> None:
    """Per COMPLETE STEP, reporting the FIRST DIVERGENT STEP and a census WINDOW.

    Every slot's launch counter is asserted against what the plan OWES per cycle: a
    slot that passes by NOT EXECUTING is the hollow pass this discipline exists to
    prevent. ``prefix_syncs`` is asserted equal to ``launches`` on both curl plans,
    because this family's prefix is refreshed per launch and a plan that refreshed it
    once would be byte-perfect on step 0 and wrong forever after.

    THE SUBNORMAL CENSUS IS TAKEN PER STEP AND REPORTED AS A WINDOW. Band entry is a
    RUN-and-WINDOW fact: a row recording a scalar count cannot be read for what it
    covers. A case whose census fires is REFUSED BY NAME and its comparisons are
    withheld from the certified total — a COVERAGE REFUSAL, not a failure — and
    :func:`leg_precondition` is what proves the detector fires at all.
    """
    rows: List[Dict[str, Any]] = []
    for label, _ in CASES:
        fields, pml = build(label)
        reference_fields, reference_pml = build(label)
        plan, residency, live, synced = compose(fields, pml)
        assert set(plan.plans) == {"step_B", "update_H", "step_D", "update_E"}, (
            label, sorted(plan.plans))
        residency.sync_in()

        first_divergent: Optional[Dict[str, Any]] = None
        per_step: List[int] = []
        census_per_step: List[int] = []
        fired: List[int] = []
        for step in range(budget):
            reference_step(reference_fields, reference_pml, live)
            walk(fields, pml, plan, residency, live)
            residency.sync_out()
            per = {name: differing(getattr(fields, name),
                                   getattr(reference_fields, name))
                   for name in STATE if getattr(fields, name, None) is not None}
            total = sum(per.values())
            per_step.append(total)
            count = state_census(reference_fields)
            census_per_step.append(count)
            if count:
                fired.append(step)
            if total and first_divergent is None:
                first_divergent = {"step": step,
                                   "targets": {k: v for k, v in per.items() if v}}

        evolved = sum(differing(np.zeros_like(getattr(reference_fields, n)),
                                getattr(reference_fields, n))
                      for n in ("Bx", "By", "Bz", "Ex", "Ey", "Ez"))
        kit.assert_moved(evolved, f"{label} whole-step state never evolved", floor=64)
        for slot in ("step_B", "update_H", "step_D", "update_E"):
            assert plan.plans[slot].launches == budget, (
                label, slot, plan.plans[slot].launches)
        for slot in ("step_B", "step_D"):
            product = plan.plans[slot]
            assert product.prefix_syncs == budget, (
                label, slot, product.prefix_syncs,
                "the prefix was not refreshed once per launch: a stale prefix is "
                "byte-perfect on step 0 and wrong on every step after it")
        refused = bool(fired)
        if not refused:
            assert first_divergent is None, (label, first_divergent, per_step)
        window = {"first_step": fired[0] if fired else None,
                  "last_step": fired[-1] if fired else None,
                  "steps_censused": budget,
                  "subnormal_words_per_step": census_per_step}
        rows.append({
            "case": label, "tags": tags(fields, pml), "budget": budget,
            "live_passes": list(live), "synced_passes": list(synced),
            "device_slots": sorted(plan.plans),
            "selected": dict(plan.selected),
            "per_step_differing": per_step,
            "first_divergent": first_divergent,
            "evolved_words": evolved,
            "subnormal_window": window,
            "refused": refused,
            "refusal": (f"{label}: REFUSED (subnormal precondition) — the census "
                        f"fired on steps {fired[0]}..{fired[-1]} of {budget}; every "
                        f"arithmetic claim on this backend rides a CHECKED "
                        f"subnormal-free precondition and this case does not meet it"
                        ) if refused else None,
            "compared": 0 if refused else
                        compared_words(snapshot(reference_fields)) * budget,
            "launches": {k: v.launches for k, v in plan.plans.items()},
            "prefix_syncs": {k: plan.plans[k].prefix_syncs
                             for k in ("step_B", "step_D")},
            "residency_verify": residency.verify(),
        })
        payload["legs"]["whole_step"] = rows
        save(payload, out)
        log(f"[whole_step] {label:<26} {budget} steps, passes={len(live)}, "
            f"first_divergent="
            f"{None if first_divergent is None else first_divergent['step']}, "
            f"evolved={evolved} window=[{window['first_step']}, "
            f"{window['last_step']}]{' REFUSED' if refused else ''}")
    payload["legs"]["whole_step"] = rows
    save(payload, out)


# ---------------------------------------------------------------------------
# LEG precondition — the control that makes the per-step census FIRE
# ---------------------------------------------------------------------------

#: Scale, and what the per-step census MUST do at it: ``True`` stay clean, ``False``
#: fire, ``None`` record only. The clean rows are what make the firing rows mean
#: something — a detector that fired on everything would refuse the physical band too
#: and certify nothing.
#:
#: THE ``band_edge`` ROW IS RECORDED RATHER THAN ASSERTED, AND IT IS THE MOST USEFUL
#: ROW HERE. Measured on this family over six steps of the live pass list: clean at
#: 1e-25, and at 1e-30 the census FIRST FIRES AT STEP 5 with 2 words. Nothing is in
#: the band at step 0 — the state is scaled uniformly and 1e-30 is a normal float32 —
#: so the band is reached by the RUN, through cancellation in the split-field
#: recurrence, and not by the seeding. That is the whole reason every census in this
#: artifact is per-step and reported as a WINDOW: a scalar count taken at step 0 here
#: would have said "clean" about a run that is not.
PRECONDITION_SCALES: Tuple[Tuple[str, float, Optional[bool]], ...] = (
    ("physical", 1.0, True),
    ("small_normal", 1e-25, True),
    ("band_edge", 1e-30, None),
    ("subnormal_band", 1e-34, False),
    ("deep_subnormal", 1e-41, False),
)


def leg_precondition(payload: Dict[str, Any], out: str, budget: int = 6) -> None:
    """A precondition never demonstrated to FIRE is decoration.

    :func:`leg_whole_step` censuses every case per step and reports a window; on the
    physical band that window is empty, and an empty window is exactly what a broken
    census also produces. This drives the SAME per-step census over a state scaled
    INTO the band and requires it to fire, so the empty windows next door are a
    measurement rather than a silence. The scaled rows are REFUSED BY NAME, never
    compared.

    The ``SubnormalWindow`` control at the end is the second half: it pins that the
    shipped detector — the one the sibling tests use — refuses a banded volume and
    refuses an EMPTY window as vacuous, which is the failure mode a census-of-zero
    would otherwise hide.
    """
    rows: List[Dict[str, Any]] = []
    for label, scale, expect_clean in PRECONDITION_SCALES:
        reference_fields, reference_pml = build("m1_metallic")
        if scale != 1.0:
            for name in STATE:
                array = getattr(reference_fields, name, None)
                if array is not None:
                    array *= np.array(scale, dtype=np.float32)
        live = live_passes(reference_fields, reference_pml)
        fired: List[int] = []
        per_step: List[int] = []
        for step in range(budget):
            reference_step(reference_fields, reference_pml, live)
            count = state_census(reference_fields)
            per_step.append(count)
            if count:
                fired.append(step)
        row = {"scale": label, "factor": scale, "steps_censused": budget,
               "subnormal_words_per_step": per_step,
               "first_step": fired[0] if fired else None,
               "last_step": fired[-1] if fired else None,
               "census_fired": bool(fired),
               "expectation": {True: "must stay clean", False: "must fire",
                               None: "recorded only"}[expect_clean],
               "verdict": ("REFUSED (subnormal precondition)" if fired
                           else "precondition holds")}
        rows.append(row)
        payload["legs"]["precondition"] = rows
        save(payload, out)
        log(f"[precondition] {label:<16} scale={scale:g} window="
            f"[{row['first_step']}, {row['last_step']}] fired={bool(fired)} "
            f"words_per_step={per_step}")
        if expect_clean is True:
            assert not fired, (
                f"{label}: the census fired on a band it must not — a detector that "
                f"refuses the physical band certifies nothing")
        elif expect_clean is False:
            assert fired, (
                f"{label}: the census DID NOT FIRE on the scaled control. Every "
                f"empty window this gate reports would then be a silence rather "
                f"than a measurement")
            kit.assert_census_floor(max(per_step), f"{label} control")

    # THE SHIPPED DETECTOR, driven the way the sibling tests drive it: clean, fired,
    # and VACUOUS are three different outcomes and the third is not a pass.
    fields, _pml = build("m1_metallic", seed=41)
    clean = preconditions.SubnormalWindow(0, 1, per_array_words=64)
    for name in ("Ey", "Hy", "Ez"):
        clean.observe(name, getattr(fields, name), step=1)
    clean_report = preconditions.assert_clean_or_refuse(clean, "control/clean")

    banded = preconditions.SubnormalWindow(0, 1, per_array_words=64)
    banded.observe("Ey_scaled",
                   (fields.Ey * np.float32(1e-40)).astype(np.complex64), step=1)
    banded_report = preconditions.demonstrate_firing(banded, "control/banded")

    vacuous = False
    try:
        preconditions.assert_clean_or_refuse(
            preconditions.SubnormalWindow(0, 1, per_array_words=64),
            "control/empty")
    except AssertionError as exc:
        vacuous = "VACUOUS" in str(exc)
    assert vacuous, (
        "an EMPTY subnormal window did not report vacuous: a census that observed "
        "nothing reports clean, and every window in this artifact would then be "
        "unreadable")
    payload["legs"]["precondition_detector"] = {
        "clean": clean_report, "banded_control": banded_report,
        "empty_window_reports_vacuous": vacuous,
        "policy": subnormal.mps_policy_report(),
        "why": ("MPS flushes float32 subnormals natively and exposes no lever, so "
                "`keep` is NOT OFFERABLE on this backend and the honest third policy "
                "value is REFUSE. Every byte claim in this artifact rides this "
                "checked precondition"),
    }
    save(payload, out)


# ---------------------------------------------------------------------------
# LEG split — the m = 0 versus |m| >= 1 boundary, as REFUSALS
# ---------------------------------------------------------------------------

def leg_split(payload: Dict[str, Any], out: str) -> None:
    """The partition against the cylindrical REAL product is STORAGE, checked both
    ways in four places — at the helper, the predicate, the plan and the composer.

    Until 2026-09-04 this leg pinned the OPPOSITE: clause 14 refused m = 0 here by
    name, on the reading that ``stepping.py:731-736`` (complex storage mandatory at
    |m| >= 1) made clause 2 a mere engine coupling. That reading was wrong about the
    partition, because the real product refuses complex64 storage by ITS clause 2 —
    so a complex64 m = 0 run, a constructible one and the corpus row
    ``dipole_in_vacuum_cyl_off_axis.py``, was refused by BOTH families and stepped
    by the array path on all four slots. The ``M_ZERO`` arm carries it here, where
    the storage already is, and this leg now measures the four-place admission, the
    real family's refusal of the same run, and this family's refusal of the real
    run — every one by name.
    """
    record: Dict[str, Any] = {}
    from meep_gpu.metal_kernels import cylindrical_real  # noqa: PLC0415

    assert cyl.m_class(0) == cyl.M_ZERO
    record["m_class_zero"] = cyl.m_class(0)

    zero_fields, zero_pml = matrix.cylindrical(m=0, complex_storage=True)
    residency = Residency()
    admissions: Dict[str, Any] = {}
    real_refusals: Dict[str, Any] = {}
    for sub_step in ("step_B", "step_D"):
        verdict = cyl.cylindrical_complex_pml_curl_coverage(
            zero_fields, zero_pml, sub_step, residency, PROBE)
        assert verdict.covered, verdict.reasons
        plan = cyl.plan_cylindrical_complex_pml_curl(
            zero_fields, zero_pml, sub_step, residency, probe=PROBE)
        assert plan is not None and plan.m_arm == cyl.M_ZERO, plan
        assert plan.zero_rows == 0
        assert plan.axis_coef == cyl.axis_coefficient(
            zero_fields.grid.dt / zero_fields.grid.dx)
        admissions[sub_step] = {"covered": True, "m_arm": plan.m_arm,
                                "zero_rows": plan.zero_rows,
                                "axis_coef": plan.axis_coef}
        real = cylindrical_real.cylindrical_real_curl_coverage(
            zero_fields, zero_pml, sub_step, residency)
        named = [r for r in real.reasons if "force_complex_fields=True" in r]
        assert not real.covered and named, real.reasons
        real_refusals[sub_step] = named
    for side in ("H", "E"):
        verdict = cyl.cylindrical_complex_constitutive_coverage(
            zero_fields, zero_pml, side, residency, PROBE)
        assert verdict.covered, verdict.reasons
        admissions[side] = {"covered": True}
        real = cylindrical_real.cylindrical_real_constitutive_coverage(
            zero_fields, zero_pml, side, residency)
        named = [r for r in real.reasons if "force_complex_fields=True" in r]
        assert not real.covered and named, real.reasons
        real_refusals[side] = named
    record["m_zero_complex_admitted"] = admissions
    record["m_zero_complex_refused_by_the_real_family"] = real_refusals

    # AND AT THE COMPOSER: all four arithmetic slots come back this family's label,
    # and no sibling co-admits (the composer would refuse the slot rather than pick).
    zero_plan = launch.plan_step(zero_fields, zero_pml, residency=Residency(),
                                 sources=(), cylindrical_complex_probe=PROBE)
    selected = dict(zero_plan.selected)
    assert {selected.get(s) for s in ("step_B", "step_D", "update_H", "update_E")} \
        == {"cylindrical complex"}, selected
    record["m_zero_composer_selection"] = selected

    # THE OTHER DIRECTION: real storage. At m = 0 that is the real product's run and
    # at |m| = 1 it cannot exist on the engine; clause 2 refuses BOTH by name here.
    refusals: Dict[str, Any] = {}
    for m in (0, 1):
        real_fields, real_pml = matrix.cylindrical(m=m, complex_storage=False)
        verdict = cyl.cylindrical_complex_pml_curl_coverage(
            real_fields, real_pml, "step_B", residency, PROBE)
        named = [r for r in verdict.reasons
                 if "force_complex_fields is not set" in r and "REAL product" in r]
        assert not verdict.covered and named, verdict.reasons
        assert cyl.plan_cylindrical_complex_pml_curl(
            real_fields, real_pml, "step_B", residency, probe=PROBE) is None
        refusals[f"m{m}"] = named
    record["clause_2_refusals"] = refusals

    payload["legs"]["split"] = record
    save(payload, out)
    log(f"[split] complex64 m = 0 admitted on 4 slots (M_ZERO), refused by the real "
        f"family by name; real storage refused here at m = 0 and m = 1; composer "
        f"selection {selected}")


# ---------------------------------------------------------------------------
# LEG containment — Metal's admitted set beside Triton's, every slot named
# ---------------------------------------------------------------------------

#: The two reasons Triton's predicate gives on THIS host that have nothing to do with
#: the cylindrical family. A row whose ONLY refusals are these is NOT COMPARABLE here:
#: the Triton product requires CuPy, and the engine on this machine holds NumPy.
BACKEND_ONLY_FRAGMENTS: Tuple[str, ...] = (
    "array module is",
    "expansion probe artifact",
)


def leg_containment(payload: Dict[str, Any], out: str) -> None:
    """Metal-only / Triton-only / both / neither, per slot — and an honest refusal.

    The rule the round before this one set: a Metal-only slot is not automatically a
    defect, but it is NEVER accepted on argument. So both predicates are asked, per
    case per slot, and the answer is recorded.

    WHAT IS ACTUALLY MEASURABLE HERE IS LESS THAN THAT, and saying so is the point.
    The Triton family's predicate carries a BACKEND clause — it requires the engine's
    array module to be CuPy, and this host's engine holds NumPy — so on this machine
    it refuses every row for a reason that has nothing to do with cylindrical
    geometry. A row whose only Triton refusals are the backend and probe clauses is
    therefore recorded NOT COMPARABLE rather than counted as Metal-only, and the
    exact refusal strings are kept so a reader can check that classification. A
    like-for-like admission diff needs a CuPy host and this gate does not claim one.
    """
    from meep_gpu.triton_kernels import cylindrical_complex as triton_cyl

    rows: List[Dict[str, Any]] = []
    counts = {"both": 0, "metal_only": 0, "triton_only": 0, "neither": 0,
              "not_comparable": 0}
    for label, _ in CASES:
        fields, pml = build(label)
        residency = Residency()
        for slot in ("step_B", "step_D", "update_H", "update_E"):
            if slot in ("step_B", "step_D"):
                mine = cyl.cylindrical_complex_pml_curl_coverage(
                    fields, pml, slot, residency, PROBE)
                theirs = triton_cyl.cylindrical_complex_curl_coverage(
                    fields, pml, slot)
            else:
                side = "H" if slot == "update_H" else "E"
                mine = cyl.cylindrical_complex_constitutive_coverage(
                    fields, pml, side, residency, PROBE)
                theirs = triton_cyl.cylindrical_complex_constitutive_coverage(
                    fields, pml, side)
            backend_only = bool(theirs.reasons) and all(
                any(fragment in reason for fragment in BACKEND_ONLY_FRAGMENTS)
                for reason in theirs.reasons)
            if backend_only:
                verdict = "not_comparable_on_this_host"
                counts["not_comparable"] += 1
            elif mine.covered and theirs.covered:
                verdict = "both"
                counts["both"] += 1
            elif mine.covered:
                verdict = "metal_only"
                counts["metal_only"] += 1
            elif theirs.covered:
                verdict = "triton_only"
                counts["triton_only"] += 1
            else:
                verdict = "neither"
                counts["neither"] += 1
            rows.append({"case": label, "slot": slot, "verdict": verdict,
                         "metal_admitted": bool(mine.covered),
                         "triton_admitted": bool(theirs.covered),
                         "triton_reasons": list(theirs.reasons),
                         "metal_reasons": list(mine.reasons)})
    assert all(row["metal_admitted"] for row in rows), (
        "a case in this gate's own matrix is not admitted by the family under test")
    # THE m = 0 ROWS (2026-09-04) ARE RECORDED SEPARATELY. On this host they land in
    # the same NOT COMPARABLE bucket as every other row: the Triton predicate's
    # backend and probe clauses fire first and it says nothing about m, so whether
    # the Triton cylindrical complex family admits complex64 at m = 0 is that
    # track's question and is not answered here.
    m_zero_verdicts = sorted({(row["case"], row["verdict"]) for row in rows
                              if row["case"].startswith("m0_")})
    payload["legs"]["containment"] = {
        "per_slot": rows,
        "counts": counts,
        "m_zero_rows": [list(pair) for pair in m_zero_verdicts],
        "slots_examined": len(rows),
        "verdict": (
            "NOT MEASURABLE ON THIS HOST. Triton's cylindrical complex predicate "
            "refuses every slot here on its BACKEND clause (the engine holds NumPy "
            "and that product requires CuPy), so no row carries a comparable Triton "
            "verdict and this gate claims no containment relation in either "
            "direction. Triton's full corpus result is 759/759; the older 702 "
            "figure was a nine-certified-family subtotal cut on a CuPy host and "
            "is not re-derived here."
            if counts["not_comparable"] == len(rows) else
            "COMPARABLE ROWS EXIST — read counts"),
        "m_zero_note": (
            "the m = 0 rows (2026-09-04) are NOT COMPARABLE on this host for the "
            "same backend reason as every other row; whether the Triton "
            "cylindrical complex family admits complex64 storage at m = 0 is that "
            "track's question and this gate does not answer it"),
        "backend_only_fragments": list(BACKEND_ONLY_FRAGMENTS),
    }
    save(payload, out)
    log(f"[containment] {len(rows)} slot-cases: {counts}")


# ---------------------------------------------------------------------------
# LEG mutations — a leg that cannot fail certifies nothing
# ---------------------------------------------------------------------------

#: ``(label, sub_step, scope, needle, replacement, must_catch, why)``.
#:
#: ``scope`` is a tag from :func:`tags` or ``None`` for the whole matrix. A defect
#: that is STRUCTURALLY ABSENT on some cases is scoped rather than reported as a
#: partial catch: rounding a structural null into a weakness is as dishonest as
#: rounding a miss into a pass.
#:
#: ``must_catch=False`` is a NULL CONTROL — a spelling asserted to be genuinely
#: equivalent rather than merely untested.
SOURCE_MUTATIONS: Tuple[Tuple[str, str, Optional[str], str, str, Optional[bool],
                              str], ...] = (
    # ---- THE m PHASE: the i*m/r coupling -----------------------------------
    # SCOPED TO m != 0 since 2026-09-04: the M_ZERO body carries no coupling block
    # at all (stepping :348/:430 branch on grid.m != 0), so these needles are absent
    # there by construction — a structural absence, not a partial catch.
    ("imr_sign_flipped", "step_B", "m_nonzero",
     "curl0 = curl0 - m0;", "curl0 = curl0 + m0;", True,
     "the i*m/r term enters the curl NEGATED (stepping.py:724 returns the term in "  # stepping.py live lines for the frozen device-text citation(s) in this string: 724->751
     "curl-sign convention). Flipping it is a smooth wrong answer that looks like a "
     "different azimuthal order"),
    ("imr_partner_register_swapped", "step_B", "m_nonzero",
     "float2 m0 = c_mul(q0, c);", "float2 m0 = c_mul(q0, a);", True,
     "target 0 couples to the CENTER load g2 (Ez/Hz) and target 2 to g0 (Ex/Hx) "
     "(stepping :348-355 B, :430-437 D). Reading the other register is the mistake a "
     "reader makes once and it costs no pointer, so nothing else would notice"),
    ("imr_row_indexed_by_z", "step_D", "m_nonzero",
     "float2 q0 = c0[i];", "float2 q0 = c0[k];", True,
     "the coefficient row is a function of r ALONE (an (nr,1,1) broadcast). Indexing "
     "it by the z lane compiles, stays in bounds while nz >= nr, and produces a "
     "plausible field with the wrong radial profile"),
    ("imr_orientation_swapped", "step_B", "m_nonzero",
     "float2 m0 = c_mul(q0, c);", "float2 m0 = c_mul(c, q0);", False,
     "MEASURED NULL, 0/10, and it is pinned that way rather than left as an "
     "expectation: the probe classified this family's row orientation "
     "AMBIGUOUS_BOTH with `real_words_all_plus_zero: true` at every m, and under "
     "FMA_V1 with the coefficient's imaginary part the only asymmetric operand the "
     "two spellings round identically. Carried because a coefficient row that ever "
     "stopped laundering its real word to +0.0 would flip this to CAUGHT"),
    ("imr_subtract_spelled_as_add_negation", "step_B", "m_nonzero",
     "curl0 = curl0 - m0;",
     "curl0 = curl0 + float2(-m0.x, -m0.y);", False,
     "NULL CONTROL. IEEE-754 defines x - y as x + (-y) exactly, signed zeros "
     "included, and negation on this backend was re-measured as a sign-bit operation "
     "(0/1024 against numpy.negative). Carried so the gate is shown not to report "
     "spurious catches on a spelling that genuinely cannot differ"),

    # ---- THE ZERO-ROW COUNT: the runtime uniform ---------------------------
    ("zero_rows_off_by_one", "step_D", "m_many",
     "bool near = (i < int(zrows));", "bool near = (i <= int(zrows));", True,
     "stepping._cylindrical_axis_rows is slice(0, |m|) — a half-open range. One extra "
     "row of forced zeros is a boundary condition applied one cell too far"),
    ("zero_rows_baked_as_a_literal", "step_B", "zero_rows_not_two",
     "bool near = (i < int(zrows));", "bool near = (i < 2);", True,
     "the whole reason the count is a RUNTIME UNIFORM: baked as a constant it is "
     "right at |m| = 2 and wrong at 3, at 5 and on the accurate branch, and the "
     "enumeration would be 8 x UNBOUNDED. SCOPED TO THE ROWS WHERE THE TRUE COUNT IS "
     "NOT 2, because where it is 2 the literal is right and the row would report a "
     "structural null as a partial catch — measured unscoped: 2/4"),
    ("zero_rows_unsigned_compare", "step_B", "m_many",
     "bool near = (i < int(zrows));", "bool near = (uint(i) < zrows);", False,
     "NULL CONTROL, and it is the module's own claim: i is never negative here so "
     "the signed and unsigned comparisons cannot differ in a bit — the whole path is "
     "integer and exact"),
    ("near_axis_auxiliary_not_zeroed", "step_B", "m_many",
     "        n0 = near ? float2(0.0f, 0.0f) : n0;", "        // n0 kept", True,
     "the |m| >= 2 rule holds the fu_ AUXILIARIES at zero as well as the fields "
     "(stepping :591-599, :663-672). fu_ is STATE, so dropping it is invisible for "
     "one launch and compounds after"),
    ("near_axis_field_not_zeroed", "step_D", "m_many",
     "        v0 = near ? float2(0.0f, 0.0f) : v0;", "        // v0 kept", True,
     "the same rule's FIELD half. Both halves are needled because they are separate "
     "statements and a transcription can lose either"),

    # ---- THE r = 0 SEAM: the ownership mask and the axis-row replacement ----
    ("r_ownership_mask_dropped", "step_D", "m_one",
     "    curl2 = at_x ? float2(0.0f, 0.0f) : curl2;",
     "    // MUTANT: the r ownership mask on target 2 dropped", True,
     "MEEP's little_owned_corner0 deliberately excludes r = 0 (vec.hpp:1100-1104) — "
     "the axis row belongs to the per-m rules. Scoped to |m| = 1 because at |m| >= 2 "
     "the near-axis rule zeroes the STORED value on that row anyway and the defect "
     "would be hidden; at |m| = 1 the D-side rule zeroes the field ONLY, so the "
     "auxiliary carries it out"),
    ("r_mask_dropped_under_the_axis_increment", "step_B", "m_one",
     "    curl0 = at_x ? float2(0.0f, 0.0f) : curl0;",
     "    // MUTANT: the r ownership mask on target 0 dropped", False,
     "MEASURED NULL, and it is what makes the axis increment a REPLACEMENT rather "
     "than an accumulation visible: the |m| = 1 B-side increment overwrites curl0 at "
     "at_x unconditionally, so masking it first cannot change a word. Carried "
     "because if the increment ever became an accumulation this row flips to CAUGHT"),
    ("axis_increment_not_negated", "step_B", "m_one",
     "curl0 = at_x ? -inc : curl0;", "curl0 = at_x ? inc : curl0;", True,
     "the axis row enters the curl negated (stepping :370-372). Negation is `-x` "
     "here, re-measured on this backend rather than inherited from the Triton "
     "track's opposite finding"),
    ("axis_increment_accumulates", "step_B", "axis_seam",
     "curl0 = at_x ? -inc : curl0;", "curl0 = at_x ? (curl0 - inc) : curl0;", True,
     "REPLACE versus ACCUMULATE, and this row is the reason the matrix carries a "
     "CONSTRUCTED case. The masked row is exactly +0.0 and +0.0 + x == x for every x "
     "EXCEPT -0.0, so the two spellings differ only where the increment's real word "
     "is an exact +0.0 AND the recurrence operand it meets is an exact -0.0. "
     "MEASURED ON RANDOM AND SIGNED-ZERO-SCATTERED STATE: 0 of 5 — armed and "
     "UNREACHED, which certifies nothing. Scoped to the case that builds the pair "
     "(see plant_axis_seam), where it misses by exactly ONE WORD"),
    ("axis_increment_reads_the_axis_row", "step_B", "m_one",
     "float2 e1 = g2[nyz + j * nzi + k];", "float2 e1 = g2[j * nzi + k];", True,
     "stepping.py:642 takes the FIRST OFF-AXIS row (xp.take(Ez, 1, axis=0)). Reading "  # stepping.py live lines for the frozen device-text citation(s) in this string: 642->671
     "row 0 instead is in bounds, compiles, and is the read the nr >= 2 clause exists "
     "to keep legal"),
    ("m_one_D_zeroes_the_auxiliary_too", "step_D", "m_one",
     "    v2 = at_x ? float2(0.0f, 0.0f) : v2;   // Dz FIELD only, never fu_Dz.",
     "    v2 = at_x ? float2(0.0f, 0.0f) : v2;\n"
     "    n2 = at_x ? float2(0.0f, 0.0f) : n2;", True,
     "stepping :588-589 zeroes the Dz FIELD on the axis row and NOT fu_Dz. The "
     "|m| >= 2 rule zeroes both, so the two arms differ in exactly this and a "
     "transcription that copied the wrong one is invisible at |m| >= 2"),
    ("m_one_B_zeroes_the_axis_row", "step_B", "m_one",
     "    // |m| = 1, B side: NOTHING. stepping._cylindrical_axis_zero_B (:648-671)",
     "    v0 = at_x ? float2(0.0f, 0.0f) : v0;   // MUTANT: a B-side axis zero", True,
     "the |m| = 1 B side does NOTHING (stepping :648-671 branches on m == 0 and "
     "abs(m) > 1 only). The absence is TRANSCRIBED, and this is what makes it a "
     "measurement rather than an omission a reader has to trust"),

    # ---- THE PREFIX --------------------------------------------------------
    ("prefix_row_stride_wrong", "step_B", None,
     "float2 pu = pfx[ii + nyz];", "float2 pu = pfx[ii + 1];", True,
     "the extended prefix is (nr+1, ny, nz) and the forward difference steps one ROW "
     "— a stride of nyz CELLS. A stride of one is the z neighbour and stays in "
     "bounds, so nothing crashes"),
    ("prefix_difference_reversed", "step_B", None,
     "curl2 = c_mul_coefficient_left(dtdx, pu - pd);",
     "curl2 = c_mul_coefficient_left(dtdx, pd - pu);", True,
     "stepping :343-347 is prefix_ext[1:] - prefix_ext[:-1]; reversing it negates "
     "Bz's whole curl, which is a sign error the four-operand form cannot make"),

    # ---- THE m = 0 ARM (2026-09-04): the axis rules on the STORED registers ---
    ("m_zero_B_axis_clear_dropped", "step_B", "m_zero",
     "    v0 = at_x ? float2(0.0f, 0.0f) : v0;", "    // MUTANT: Bx[r=0] kept", True,
     "stepping._cylindrical_axis_zero_B (:661-662): Bx on the axis is identically "
     "zero, written AFTER the recurrence. Without it the recurrence's own output "
     "survives on row 0 — a plausible field, and wrong on every lane of the axis"),
    ("m_zero_B_axis_clear_also_masks_the_auxiliary", "step_B", "m_zero",
     "    v0 = at_x ? float2(0.0f, 0.0f) : v0;",
     "    v0 = at_x ? float2(0.0f, 0.0f) : v0;\n"
     "    n0 = at_x ? float2(0.0f, 0.0f) : n0;", True,
     "the m = 0 rule zeroes the FIELD only (:661-662 writes fields.Bx and nothing "
     "else); the |m| >= 2 hold zeroes fu_ too, and a transcription that borrowed "
     "that component set would be right for one launch and wrong after"),
    ("m_zero_D_post_add_dropped", "step_D", "m_zero",
     "        v2 = at_x ? (v2 + inc0) : v2;", "        // MUTANT: no post-add", True,
     "stepping :586: Dz[r=0] += (4*Courant)*Hp[r=0], the analytic limit of "
     "(1/r) d(r Hp)/dr on the axis. Dropping it leaves the masked row's recurrence "
     "output — which is what a reader who trusts the ownership mask would ship"),
    ("m_zero_D_post_add_lands_on_the_wrong_component", "step_D", "m_zero",
     "        v2 = at_x ? (v2 + inc0) : v2;",
     "        v0 = at_x ? (v0 + inc0) : v0;", True,
     "the increment belongs to Dz (target 2), not Dr (target 0)"),
    ("m_zero_D_post_add_also_writes_the_auxiliary", "step_D", "m_zero",
     "        v2 = at_x ? (v2 + inc0) : v2;",
     "        v2 = at_x ? (v2 + inc0) : v2;\n"
     "        n2 = at_x ? (n2 + inc0) : n2;", True,
     "the post-add touches the stored FIELD (:586) and never fu_Dz; fu_ is STATE, "
     "so the over-carry is invisible for one launch and compounds after"),
    ("m_zero_D_post_add_folded_into_the_curl", "step_D", "m_zero",
     (("        v2 = at_x ? (v2 + inc0) : v2;", "        // MUTANT: folded instead"),
      ("    bool at_x = (i == 0), at_y = (j == 0), at_z = (k == 0);\n",
       "    bool at_x = (i == 0), at_y = (j == 0), at_z = (k == 0);\n"
       "    curl2 = at_x ? -(c_mul_coefficient_left(axis_coef, b)) : curl2;\n")),
     None, True,
     "THE DEFECT stepping.py:546-551 RECORDS AS MEASURED WRONG: routing the m = 0 "  # stepping.py live lines for the frozen device-text citation(s) in this string: 546-551->575-580
     "increment through the split-field recurrence (Er 2.7e-01 / Hp 4.5e-01 under "
     "PML against the post-add's 3.6e-07). The |m| = 1 increment IS a curl fold, so "
     "this is exactly the slip a reader copying that arm would make. Two edits, "
     "applied in sequence: the post-add removed and the fold planted before the "
     "ownership mask"),
    ("m_zero_D_Dy_axis_clear_dropped", "step_D", "m_zero",
     "        v1 = at_x ? float2(0.0f, 0.0f) : v1;", "        // MUTANT: Dy[r=0] kept",
     True,
     "stepping :587: Dp is identically zero on the axis (an azimuthal vector at "
     "r = 0 points nowhere), the FIELD only, written after the post-add"),
    ("m_zero_axis_coef_formed_in_kernel", "step_D", "m_zero",
     "float2 inc0 = c_mul_coefficient_left(axis_coef, b);",
     "float2 inc0 = c_mul_coefficient_left(4.0f * dtdx, b);", False,
     "NULL CONTROL, and it pins the module's own claim about the bound scalar: "
     "`4.0 * (dt/dx)` formed in float64 and rounded once at the binding is the same "
     "word as `4.0f * dtdx` in-kernel, because scaling by four is exact. A "
     "divergence here would mean the binding and the array path disagree about a "
     "word this family ships"),
)

#: Host-side mutations: defects the KERNEL SOURCE cannot express because the value is
#: computed on the host and bound. ``(label, patch, scope, sub_step, must_catch, why)``
#: where ``patch`` is a context manager factory applied around the plan build.
HOST_MUTATIONS: Tuple[Tuple[str, str, Optional[str], str, Optional[bool], str], ...] = (
    # THE TWO ROW MUTATIONS ARE SCOPED TO m != 0 (2026-09-04): at m = 0 the rows are
    # exactly zero whatever shift or clamp built them, and the M_ZERO body never
    # reads them — a structural null on those cases, not a partial catch.
    ("imr_yee_shift_from_the_other_target", "imr_shift", "m_nonzero", "step_B", True,
     "the i*m/r denominator is 2*ir + the target's OWN radial Yee shift "
     "(stepping :713-718). Borrowing the partner's shift is a half-cell error in the "
     "coefficient row that the kernel cannot see, because it takes a pointer"),
    ("imr_clamp_removed", "imr_clamp", "m_nonzero", "step_B", False,
     "MEASURED NULL, 0/11, and it PINS THE MODULE'S OWN CLAIM rather than testing "
     "the kernel: xp.maximum(r_doubled, 1.0) bites only on row 0 of a SHIFT-0 target "
     "(Bx on B, Dz on D), and that target's curl is exactly what the r ownership "
     "mask selects away — so removing the clamp puts an INFINITY in the coefficient "
     "row and not one word of it survives. If a future edit dropped that mask this "
     "row flips to CAUGHT, which is what makes it worth carrying"),
    ("prefix_source_component_swapped", "prefix_component", None, "step_B", True,
     "the B-side prefix scans Ey specifically (stepping :299-347; the D side scans "
     "Hy, :414-426). Swapped for a SIBLING SOURCE OF THE SAME SUB-STEP — Ex, which "
     "the plan already mirrors — so the defect is a silent wrong field rather than "
     "an unregistered-mirror crash, which would be measuring the harness"),
    ("prefix_ir0_swapped", "prefix_ir0", None, "step_B", True,
     "ir0 is 0.0 on the B side and 0.5 on the D side — the two sub-lattices. The scan "
     "runs either way and the profile is smoothly wrong"),
    ("m_class_arm_swapped", "m_arm", None, "step_B", True,
     "compile the NEXT M_CLASS body for this m (m = 0 -> |m| = 1 -> |m| >= 2 -> "
     "|m| = 1). The three arms are DIFFERENT ARITHMETIC — no coupling and the m = 0 "
     "axis rules, an axis-row replacement, a near-axis zeroing of six volumes — "
     "which is why M_CLASS stays a compile-time branch where zero_rows became a "
     "runtime uniform. This is that claim, armed"),
    ("m_class_zero_body_at_nonzero_m", "m_arm_zero", "m_nonzero", "step_D", True,
     "compile the m = 0 body for an |m| >= 1 run — the OTHER direction of the "
     "swap above, on the D side: no coupling, no increment, and the m = 0 axis "
     "rules where the |m| >= 1 rules belong. What a family that admitted m = 0 by "
     "evaluating the |m| >= 1 arithmetic at m = 0 would have to be wrong about, "
     "seen from the other side"),
)

#: Where each arm's swap goes under ``m_class_arm_swapped``. A cycle over the three
#: bodies so every arm is compiled AGAINST a different one.
_ARM_SWAP: Dict[int, int] = {cyl.M_ZERO: cyl.M_ONE, cyl.M_ONE: cyl.M_MANY,
                             cyl.M_MANY: cyl.M_ONE}


def apply_edit(source: str, old: Any, new: Any) -> str:
    """One needle or a SEQUENCE of ``(old, new)`` pairs, each refusing a no-op edit.

    A mutation that needs two sites — the m = 0 fold removes the post-add AND plants
    the fold before the mask — is spelled as a tuple of pairs and applied in order,
    so the table stays one row per defect.
    """
    if isinstance(old, tuple):
        for pair_old, pair_new in old:
            source = needle(source, pair_old, pair_new)
        return source
    return needle(source, old, new)


def compile_mutant(source: str) -> Any:
    return kit.Counter(compile_source(source).cyl_complex_pml_curl_step)


def leg_mutations(payload: Dict[str, Any], out: str) -> None:
    harness = kit.MutationHarness(payload, out)

    # ---- source needles ---------------------------------------------------
    for (label, sub_step, scope, old, new, must_catch, why) in SOURCE_MUTATIONS:
        missed = False
        ran = caught = launches = 0
        skipped: List[str] = []
        for case, _ in CASES:
            fields, pml = build(case)
            if scope is not None and scope not in tags(fields, pml):
                skipped.append(case)
                continue
            names = curl_names(sub_step)
            before, after, moved = oracle(
                fields, lambda: getattr(stepping, sub_step)(fields, pml), names)
            kit.assert_moved(moved, f"{label}/{case} reference barely moved",
                             floor=64)
            grid = fields.grid
            kinds = stepping._boundary_kinds(grid, pml)
            bcz = (templates.METALLIC if kinds[2] == "metallic"
                   else templates.PERIODIC)
            shipped = cyl.cylindrical_curl_source(
                bcz, bool(cyl.SUB_STEPS[sub_step]["backward"]),
                cyl.m_class(int(grid.m)), EXPANSION)
            try:
                mutant = apply_edit(shipped, old, new)
            except LookupError:
                missed = True
                restore(fields, before)
                continue
            counter = compile_mutant(mutant)
            run_curl(fields, pml, sub_step,
                     functions={shaders.CONTRACT_OFF: counter})
            launches += counter.launches
            ran += 1
            if sum(divergence(fields, after).values()):
                caught += 1
            restore(fields, before)
        harness.record(label, harness.verdict(missed, ran, launches, caught),
                       launches, caught, ran, must_catch, why,
                       extra={"kind": "source", "sub_step": sub_step,
                              "scope": scope, "skipped_cases": skipped})

    # ---- host mutations ---------------------------------------------------
    for (label, kind, scope, sub_step, must_catch, why) in HOST_MUTATIONS:
        ran = caught = launches = 0
        skipped: List[str] = []
        for case, _ in CASES:
            fields, pml = build(case)
            if scope is not None and scope not in tags(fields, pml):
                skipped.append(case)
                continue
            names = curl_names(sub_step)
            before, after, moved = oracle(
                fields, lambda: getattr(stepping, sub_step)(fields, pml), names)
            kit.assert_moved(moved, f"{label}/{case} reference barely moved",
                             floor=64)
            plan = apply_host_mutation(kind, fields, pml, sub_step)
            launches += plan.launches
            ran += 1
            if sum(divergence(fields, after).values()):
                caught += 1
            restore(fields, before)
        harness.record(label, harness.verdict(False, ran, launches, caught),
                       launches, caught, ran, must_catch, why,
                       extra={"kind": "host", "sub_step": sub_step,
                              "scope": scope, "skipped_cases": skipped})

    # ---- whole-step mutations: the classes a single launch CANNOT see ------
    leg_whole_step_mutations(harness)


def apply_host_mutation(kind: str, fields: Any, pml: Any, sub_step: str) -> Any:
    """Plant one HOST-side defect, launch through the shipped route, restore.

    Every patch is applied to the module attribute the plan builder reads by GLOBAL
    NAME and removed in a ``finally``, so a mutation that raised cannot leak into the
    next row. The plan is otherwise built exactly as the engine builds it: these are
    defects in what is COMPUTED and BOUND, and the kernel takes a pointer and cannot
    see any of them.
    """
    spec = cyl.SUB_STEPS[sub_step]
    other = "step_D" if sub_step == "step_B" else "step_B"
    saved: Dict[str, Any] = {}
    try:
        if kind == "imr_shift":
            original = cyl.imr_coefficient_row
            saved["fn"] = original

            def wrong_shift(xp, target, sign, m, dtdx, rows, dtype,
                            _original=original):
                partner = {"Bx": "Bz", "Bz": "Bx", "Dx": "Dz", "Dz": "Dx"}[target]
                return _original(xp, partner, sign, m, dtdx, rows, dtype)

            cyl.imr_coefficient_row = wrong_shift
        elif kind == "imr_clamp":
            original = cyl.imr_coefficient_row
            saved["fn"] = original

            def unclamped(xp, target, sign, m, dtdx, rows, dtype):
                from meep_gpu.fields import IYEE_SHIFTS

                iyee_r = IYEE_SHIFTS[target][0]
                doubled = 2 * xp.arange(int(rows), dtype=xp.float64) + iyee_r
                with np.errstate(divide="ignore", invalid="ignore"):
                    return ((-1j) * (float(sign) * 2.0 * int(m) * float(dtdx))
                            / doubled.reshape(-1, 1, 1)).astype(dtype)

            cyl.imr_coefficient_row = unclamped
        elif kind == "prefix_component":
            saved["component"] = cyl.PREFIX[sub_step]["component"]
            sibling = next(name for name in spec["sources"]
                           if name != saved["component"])
            cyl.PREFIX[sub_step]["component"] = sibling
        elif kind == "prefix_ir0":
            saved["ir0"] = cyl.PREFIX[sub_step]["ir0"]
            cyl.PREFIX[sub_step]["ir0"] = cyl.PREFIX[other]["ir0"]
        elif kind in ("m_arm", "m_arm_zero"):
            grid = fields.grid
            kinds = stepping._boundary_kinds(grid, pml)
            bcz = (templates.METALLIC if kinds[2] == "metallic"
                   else templates.PERIODIC)
            shipped_arm = cyl.m_class(int(grid.m))
            other_arm = (_ARM_SWAP[shipped_arm] if kind == "m_arm" else cyl.M_ZERO)
            assert other_arm != shipped_arm, (kind, shipped_arm)
            counter = compile_mutant(cyl.cylindrical_curl_source(
                bcz, bool(spec["backward"]), other_arm, EXPANSION))
            plan = run_curl(fields, pml, sub_step,
                            functions={shaders.CONTRACT_OFF: counter})
            plan.launches = counter.launches
            return plan
        else:  # pragma: no cover - a typo in the table is a hard failure
            raise ValueError(f"unknown host mutation {kind!r}")
        return run_curl(fields, pml, sub_step)
    finally:
        if "fn" in saved:
            cyl.imr_coefficient_row = saved["fn"]
        if "component" in saved:
            cyl.PREFIX[sub_step]["component"] = saved["component"]
        if "ir0" in saved:
            cyl.PREFIX[sub_step]["ir0"] = saved["ir0"]


def curl_only_probe(case: str, sub_step: str, launches: int) -> int:
    """Differing words after N launches of ONE sub-step, array path against Metal.

    The instrument that turns "a single-launch gate cannot see this" into a number.
    Called under a patched plan so the defect is live; a nonzero result would mean
    the defect is NOT a whole-step-only class and the leg's claim about it is wrong.
    """
    fields_a, pml_a = build(case)
    fields_b, pml_b = build(case)
    for _ in range(launches):
        getattr(stepping, sub_step)(fields_a, pml_a)
    run_curl(fields_b, pml_b, sub_step, launches=launches)
    return sum(differing(getattr(fields_a, n), getattr(fields_b, n))
               for n in curl_names(sub_step))


def leg_whole_step_mutations(harness: Any, budget: int = 4) -> None:
    """The three classes that exist ONLY at complete-step granularity.

    Every kernel launched here is the SHIPPED one and every one of them is
    byte-perfect in isolation, which is exactly the point: all that moves is WHEN a
    host pass runs, or whether the host copy the prefix scan reads is the one the
    device wrote. A per-sub-step gate cannot fail on any of this by construction.

    The FIRST DIVERGENT STEP is recorded per case, because it is the evidence that
    the class is a whole-step class: a defect that first appears at step 1 or later
    is one a single-launch comparison was structurally unable to see.
    """
    def sweep(label: str, must_catch: Optional[bool], why: str,
              scope: Optional[str] = None, extra: Optional[Dict[str, Any]] = None,
              **walk_kwargs: Any) -> None:
        ran = caught = launches = 0
        skipped: List[str] = []
        first_steps: Dict[str, Optional[int]] = {}
        for case, _ in CASES:
            fields, pml = build(case)
            if scope is not None and scope not in tags(fields, pml):
                skipped.append(case)
                continue
            reference_fields, reference_pml = build(case)
            plan, residency, live, _synced = compose(fields, pml)
            residency.sync_in()
            first: Optional[int] = None
            for step in range(budget):
                reference_step(reference_fields, reference_pml, live)
                walk(fields, pml, plan, residency, live, **walk_kwargs)
                residency.sync_out()
                total = sum(differing(getattr(fields, n),
                                      getattr(reference_fields, n))
                            for n in STATE if getattr(fields, n, None) is not None)
                if total and first is None:
                    first = step
            launches += sum(product.launches for product in plan.plans.values())
            ran += 1
            first_steps[case] = first
            if first is not None:
                caught += 1
        record: Dict[str, Any] = {"kind": "whole_step", "scope": scope,
                                  "budget": budget, "skipped_cases": skipped,
                                  "first_divergent_step_per_case": first_steps}
        record.update(extra or {})
        harness.record(label, harness.verdict(False, ran, launches, caught),
                       launches, caught, ran, must_catch, why, extra=record)

    # 1. THE STALE MIRROR, and this family is where the class stops being abstract.
    original_refresh = cyl.CylindricalComplexCurlPlan.refresh_prefix

    def no_pull(self) -> None:
        prefix = cyl.cylindrical_prefix(self._xp, self.sub_step, self._source_map,
                                        scratch=self.scratch)
        self._prefix_host[...] = prefix
        self.residency.sync_in((self._prefix_name,))
        self.prefix_syncs += 1

    cyl.CylindricalComplexCurlPlan.refresh_prefix = no_pull
    try:
        # THE BLINDNESS IS MEASURED, NOT CLAIMED. Under the SAME defect, the
        # sub-step-only route is byte-perfect at one launch AND at six, because a
        # curl-only loop never writes its own prefix source: step_B writes B and
        # reads Ey. Only a COMPLETE step makes the host copy stale, and it does so
        # immediately on the D side — update_H writes Hy on the DEVICE and step_D's
        # prefix scans it in the same step, which is why the whole-step first
        # divergence is step 0 rather than step 1.
        blindness = {}
        for case, _ in CASES:
            blindness[case] = {
                f"{sub_step}_x{n}": curl_only_probe(case, sub_step, n)
                for sub_step in ("step_B", "step_D") for n in (1, 6)}
        sub_step_total = sum(sum(row.values()) for row in blindness.values())
        assert sub_step_total == 0, (
            "the sub-step route DID see the stale prefix; this defect is then not a "
            "whole-step-only class and the leg's claim about it is wrong", blindness)
        sweep("prefix_scanned_from_the_STALE_HOST_copy", True,
              "the radial prefix is a SEQUENTIAL HOST SCAN, so every launch must "
              "pull the prefix SOURCE off the device first — between launches the "
              "device mirror is authoritative and the host copy is stale. THE "
              "SUB-STEP GATE CANNOT SEE THIS AT ANY LAUNCH COUNT (measured: 0 "
              "differing words at one launch and at six, on both sub-steps, over "
              "every case) because a curl-only loop never writes its own prefix "
              "source. In a COMPLETE step it is wrong from step 0, because update_H "
              "writes Hy on the device and step_D scans it. This is the residency "
              "cost this family pays, shown to be load-bearing",
              extra={"sub_step_route_blind_differing": blindness,
                     "sub_step_route_total_differing": sub_step_total})
    finally:
        cyl.CylindricalComplexCurlPlan.refresh_prefix = original_refresh

    # 2. THE SEAM, three ways. Scoped to WALLED cases: on a periodic z there is no
    #    wall pass at all, so these are structural nulls there rather than misses.
    sweep("wall_pass_dropped", True,
          "zero_metal_B holds the B samples that lie ON a metallic wall at exactly "
          "zero (MEEP step_boundaries(B_stuff)). It runs on the HOST between the "
          "curl and the constitutive pass and no Metal product carries it",
          scope="walled", skip=("zero_metal_B",))
    sweep("wall_pass_run_WITHOUT_its_sync_bracket", True,
          "the engine holds NumPy and the device mirror is authoritative between "
          "launches, so an array-path pass must be bracketed with sync_out and "
          "sync_in. Without the bracket the pass writes a HOST array nothing reads "
          "and the device keeps the un-cleared wall — the stale-mirror class, at the "
          "seam",
          scope="walled", unsynced=("zero_metal_B",))

    walled_order = ("step_B", "update_H", "zero_metal_B", "step_D", "update_E",
                    "zero_metal_D")
    sweep("wall_pass_moved_AFTER_the_constitutive_pass", True,
          "MEEP's order is step_db -> step_source -> step_boundaries, so the wipe "
          "runs BEFORE update_H reads B (driver.py:3285-3287). Moving it one pass "
          "later feeds the constitutive read an un-cleared wall; every kernel here "
          "is the shipped one and only the ORDER moves",
          scope="walled", order=walled_order)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

LEGS: Tuple[Tuple[str, Callable[[Dict[str, Any], str], None]], ...] = (
    ("execution", leg_execution),
    ("curl", leg_curl),
    ("constitutive", leg_constitutive),
    ("axis", leg_axis),
    ("whole_step", leg_whole_step),
    ("precondition", leg_precondition),
    ("split", leg_split),
    ("containment", leg_containment),
    ("mutations", leg_mutations),
)


def main(argv: Sequence[str]) -> int:
    parser = kit.argument_parser(__doc__ or "")
    args = parser.parse_args(list(argv))
    started = time.time()
    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)

    payload: Dict[str, Any] = {
        "gate": "metal_cylindrical_complex",
        "family": cyl.FAMILY,
        "slots": ["step_B", "step_D", "update_H", "update_E"],
        "environment": kit.environment_stamp(),
        "subnormal_policy": subnormal.mps_policy_report(),
        "legs": {},
    }
    kit.save(payload, out)

    reasons: List[str] = []
    if not payload["environment"].get("mps_available"):
        reasons.append("no MPS device available")
    reasons.extend(subnormal.mps_policy_reasons())
    if EXPANSION is None:
        reasons.append(
            "no cylindrical-complex expansion probe artifact: this family needs "
            f"{cyl.IMR_ROW_PROBE_PATTERN!r} and {cyl.AXIS_SCALAR_PROBE_PATTERN!r}, "
            "which no earlier artifact carries. Cut one with "
            "probe_metal_cylindrical_complex.py")
    if reasons:
        return kit.cannot_certify(payload, out, reasons)

    ran = kit.run_legs(LEGS, payload, out, kit.wanted_legs(args.legs))

    compared = 0
    certified = True
    for key in ("curl", "constitutive"):
        for row in payload["legs"].get(key, ()):
            compared += int(row["compared"])
            certified = certified and row["differing"] == 0
    # A CASE REFUSED ON THE PRECONDITION CONTRIBUTES NO COMPARISONS AND NO VERDICT.
    # Folding it in either way would be wrong in both directions: as a pass it would
    # certify bytes produced under a condition the claim excludes, and as a failure it
    # would report a coverage boundary as a defect.
    refused = [row["case"] for row in payload["legs"].get("whole_step", ())
               if row.get("refused")]
    for row in payload["legs"].get("whole_step", ()):
        if row.get("refused"):
            continue
        compared += int(row["compared"])
        certified = certified and row["first_divergent"] is None

    mutations = payload["legs"].get("mutations", ())
    armed = [row for row in mutations if row.get("must_catch") is True]
    caught = [row for row in armed if row["caught"] == row["ran"] and row["ran"] > 0]

    return kit.summarize(
        payload, out,
        claim=("metal_kernels.cylindrical_complex reproduces stepping.py word for "
               "word on a complex64 Dcyl grid at every m — the curl on both "
               "sub-steps in its three bodies (m = 0, |m| = 1, |m| >= 2) and the "
               "certified complex constitutive pair under this family's restated "
               "predicate — per sub-step AND per complete step"),
        scope=("the case matrix in CASES: all three m classes (m = 0 since "
               "2026-09-04), both signs of m, both z terminations, both branches of "
               "accurate_fields_near_cylorigin (at |m| = 3 where it binds a "
               "different uniform and at m = 0 where it has no effect), three "
               "zero-row counts, power-of-two and non-power-of-two Courant numbers "
               "and planted signed zeros. Real float32 storage (the cylindrical "
               "real product's), conductivity, dispersion, chi2/chi3, BFAST, "
               "grid.beta, a Bloch phase, a mirror plane, an off-diagonal chi1inv "
               "row on update_E and a radial extent below two rows are OUT OF SCOPE "
               "and refused BY NAME by the predicate"),
        stated_weakness=(
            "BEHAVIOURAL ONLY: torch.mps.compile_shader exposes no AIR, no GPU ISA "
            "and no optimisation report, so this gate catches a wrong ANSWER and "
            "never a wrong INSTRUCTION, and it cannot establish that the contraction "
            "guard was obeyed. It makes NO throughput claim — the host prefix scan "
            "hands back part of what the residency layer buys — it does not re-cut "
            "the 759-slot coverage number, and the Metal-vs-Triton admission diff is "
            "REFUSED on this host rather than estimated (leg containment). All "
            "arithmetic claims ride a CHECKED subnormal-free precondition under the "
            "FLUSH policy, the only one this executor can honour."),
        started=started, legs_run=ran, compared=compared, certified=certified,
        extra={"cases_refused_on_precondition": refused,
               "mutations_armed": len(armed),
               "mutations_caught_of_armed": len(caught),
               "mutation_verdicts": {row["mutation"]: row["verdict"]
                                     for row in mutations},
               "subnormal_windows": {
                   row["case"]: [row["subnormal_window"]["first_step"],
                                 row["subnormal_window"]["last_step"]]
                   for row in payload["legs"].get("whole_step", ())}})


if __name__ == "__main__":
    from metal_gate_runner import run_current_measurement

    raise SystemExit(run_current_measurement(__file__, sys.argv[1:]))
