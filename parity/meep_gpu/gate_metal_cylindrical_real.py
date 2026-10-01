"""BYTE GATE — the REAL cylindrical family (Dcyl, m = 0) on Metal.

WHAT THIS CERTIFIES: that ``metal_kernels.cylindrical_real``'s four products
reproduce ``stepping.py`` WORD FOR WORD as uint32 on a real float32 Dcyl grid at
m = 0 — the two-launch curl (radial scan, then curl) on both sub-steps and the
CERTIFIED real constitutive pair re-admitted under this family's restated predicate
— per SUB-STEP **and per COMPLETE STEP**, over a swept case matrix, with the
transcription choices that decide those words carried as ARMED MUTATIONS rather than
as comments.

THE ORACLE IS IN-PROCESS NumPy AND THAT IS THIS FAMILY'S WHOLE LICENCE. The engine
holds NumPy (``coverage.MIRRORED_HOST_MODULE``), ``numpy.cumsum`` in float32 IS a
sequential accumulation, and the device scan is column-serial — so the prefix has a
reproducible oracle here where the Triton track's CuPy host had none. Leg ``prefix``
is that claim, measured against ``stepping.cylindrical_rderiv_prefix`` CALLED rather
than re-derived.

WHY THE COMPLETE STEP IS THE ARBITER AND NOT A BONUS. Six green per-sub-step
comparisons say nothing about the object the engine would run. The complete step here
is the driver's TEN passes (``driver.py:3279-3304``), taken from the composer's own
residency table and asserted equal to it::

    step_B -> fill_B -> zero_metal_B -> fill_folded_far_ghosts_B -> update_H
    step_D -> fill_D -> zero_metal_D -> fill_folded_far_ghosts_D -> update_E

with the WALL PASS sitting BETWEEN the two fill passes. Four failure classes live
ONLY there:

* **a STALE MIRROR** — the engine holds NumPy, so a sub-step left on the array path
  writes the HOST array while the device mirror goes on holding the old bytes. Every
  array-path pass is bracketed with an explicit ``sync_out`` / ``sync_in`` and
  ``Residency.verify()`` is asserted clean INSIDE that bracket, which is where the
  bracket is load-bearing rather than ceremonial;
* **a SEAM** — ``zero_metal_B``/``zero_metal_D`` clear stored cell 0 of the walled z
  axis BETWEEN the curl and the constitutive pass, on volumes the curl kernel just
  wrote, on the HOST, while the device mirror is authoritative;
* **an ACCUMULATING AUXILIARY** — ``fu_*`` and ``f_w_*`` are STATE. A kernel right for
  one launch and wrong forever after is identical in a single-launch gate;
* **THE TWO-LAUNCH ORDER** — this family's curl READS the scan's output, and the scan
  is a function of THIS sub-step's source volume. A scan hoisted out of the loop feeds
  the curl the PREVIOUS timestep's radial derivative: a smooth, plausible, entirely
  wrong field rather than an error. Per-slot launch counters assert (24, 12, 12) per
  twelve cycles — total, scan, curl — so no slot may pass by NOT EXECUTING and no
  plan may pass by running the scan once.

WHAT THIS FAMILY'S "m PHASE" AND "ZERO-ROW COUNT" MUTATIONS ARE, stated rather than
rounded up. At m = 0 the i*m/r coupling and the near-axis zero-row count DO NOT
EXIST: ``stepping`` adds the coupling only under ``grid.m != 0`` (:347-355, :430-437)
and ``_cylindrical_axis_rows`` is ``slice(0, |m|)``, empty at m = 0. A gate that
claimed to mutate them here would be claiming to mutate nothing. So the STRUCTURAL
ABSENCE is what is armed: leg ``mutations`` INJECTS an |m| = 1-shaped i*m/r term and
an |m| >= 2-shaped near-axis zeroing into the m = 0 body and requires both CAUGHT,
which turns "this family does not carry them" from an omission a reader must trust
into a measurement. The m = 0 versus |m| >= 1 SPLIT itself is leg ``split``, at the
predicate, at the plan and at the composer.

WHAT THIS GATE DOES NOT CERTIFY, each stated rather than papered over:

* **it is BEHAVIOURAL, not generated-code evidence.** ``torch.mps.compile_shader``
  exposes no AIR, no GPU ISA and no optimisation report (``device.py:46-62``), so
  every leg catches a wrong ANSWER, never a wrong INSTRUCTION, and it cannot
  establish that the contraction guard was obeyed. That is this backend's standing
  certification gap against the Triton and CUDA tracks and it stays stated;
* **it runs under the FLUSH policy and nothing else is offerable.** MPS flushes
  float32 subnormals natively and exposes no lever, so ``keep`` is NOT OFFERABLE and
  the honest third policy value is REFUSE. Every claim rides a CHECKED subnormal-free
  precondition, reported as a WINDOW — ``[first_step, last_step]`` — because band
  entry is a RUN-and-WINDOW fact, not a family fact. A case whose census fires is
  REFUSED BY NAME and its comparisons are withheld from the certified total: a
  COVERAGE REFUSAL, not a failure;
* **it makes NO throughput claim.** The scan's dispatch is sized from its OUTPUT
  volume, so ``nr - 1`` threads in every ``nr`` return immediately, and the curl pays
  a second launch per sub-step. This family's number must be measured separately and
  may NOT be inherited from the certified families' benchmark;
* **it does not re-cut the 759-slot coverage number**, and the Metal-vs-Triton
  admission diff is REFUSED ON THIS HOST rather than fudged — see leg ``containment``.

THE LEGS:

1.  ``execution``     provenance, every specialisation COMPILED in both contraction
                      modes, and the enumeration count pinned.
2.  ``prefix``        the device scan alone against ``cylindrical_rderiv_prefix``
                      CALLED — the family's decisive inversion of the Triton track's
                      refusal, re-measured here per case per sub-step.
3.  ``curl``          per sub-step, the two-launch product against ``stepping.step_B``
                      / ``step_D``. Words, never ``allclose``; a vacuity floor and an
                      asserted launch split on every case.
4.  ``constitutive``  per side, the CERTIFIED real body under this family's
                      admission, plus the assertion that this family emits NO
                      constitutive source of its own.
5.  ``axis``          the r = 0 seam: the far ghost asserted as the SOURCE IDENTITY
                      it is, the near ghost MEASURED on the array path with the
                      substitution count floored, a floor on the ROW-0 words the
                      reference moved, and the DEADNESS of ``b_x`` measured rather
                      than asserted.
6.  ``whole_step``    the driver's ten passes, twelve cycles, per-slot launch
                      counters, the residency bracket checked inside itself, and a
                      per-step subnormal window.
7.  ``precondition``  the same per-step census driven over a state scaled INTO the
                      band, REQUIRED to fire. A precondition never demonstrated to
                      fire is decoration.
8.  ``split``         the m = 0 versus |m| >= 1 boundary and the family's other
                      un-inverted clauses, as REFUSALS BY NAME.
9.  ``containment``   Metal's admitted set beside Triton's, every slot named.
10. ``mutations``     armed, launch-counted, caught-of-armed reported honestly,
                      including the classes only a complete step can see.

    python -u gate_metal_cylindrical_real.py \\
        --out results/metal_cylindrical_real_<date>/gate.json
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
    coverage as metal_coverage,
    cylindrical_real as cyl,
    launch,
    preconditions,
    shaders,
    subnormal,
)
from meep_gpu.metal_kernels.device import Residency, compile_source  # noqa: E402
from meep_gpu.metal_kernels.launch import ConstitutivePlan  # noqa: E402
from meep_gpu.triton_kernels.coverage import CONSTITUTIVE_SIDES  # noqa: E402

import metal_gate_kit as kit  # noqa: E402

log, save, differing, words = kit.log, kit.save, kit.differing, kit.words
needle = kit.needle

#: Every volume one complete step can touch. ``fu_*`` and ``f_w_*`` are STATE: a
#: kernel right for one launch and wrong forever after diverges only once they
#: accumulate, which is why the whole-step leg compares all of it after every pass.
STATE: Tuple[str, ...] = (
    "Bx", "By", "Bz", "Dx", "Dy", "Dz", "Ex", "Ey", "Ez", "Hx", "Hy", "Hz",
    "fu_Bx", "fu_By", "fu_Bz", "fu_Dx", "fu_Dy", "fu_Dz",
    "f_w_Hx", "f_w_Hy", "f_w_Hz", "f_w_Ex", "f_w_Ey", "f_w_Ez")

#: The driver's complete step, ``(residency name, array-path function, takes_pml)``.
#: TEN passes, in the order ``driver.py:3279-3304`` runs them, with the wall clear
#: sitting BETWEEN the two fill passes. The names are the COMPOSER's own vocabulary
#: and :func:`_assert_pass_list` pins the order against
#: ``metal_coverage.RESIDENCY_ORDER``, so the walk and the residency model cannot
#: disagree about what a step is.
#:
#: THE INERT PASSES ARE WALKED ANYWAY. On an unfolded Dcyl grid the two fill passes
#: and the two far-ghost passes return immediately, and dropping them would make this
#: leg's step a step of the gate's own devising rather than the driver's. They are
#: recorded as NOT LIVE and their bracket is checked like every other.
PASSES: Tuple[Tuple[str, Callable[..., Any], bool], ...] = (
    ("step_B", stepping.step_B, True),
    ("fill_B", stepping.fill_symmetry_bc_B, False),
    ("zero_metal_B", stepping.zero_metal_B, False),
    ("fill_folded_far_ghosts_B", stepping.fill_folded_far_ghosts_B, False),
    ("update_H", stepping.update_H, True),
    ("step_D", stepping.step_D, True),
    ("fill_D", stepping.fill_symmetry_bc_D, False),
    ("zero_metal_D", stepping.zero_metal_D, False),
    ("fill_folded_far_ghosts_D", stepping.fill_folded_far_ghosts_D, False),
    ("update_E", stepping.update_E, True),
)

#: The four slots this family fills. ``update_P`` is deliberately absent: no Metal
#: product carries it, this family registers no arm there, and no case here
#: registers a polarization.
ARITHMETIC_SLOTS: Tuple[str, ...] = ("step_B", "update_H", "step_D", "update_E")

#: The budget every whole-step case runs. Twelve rather than four for the reason the
#: certified whole-step gates use twelve: the classes this leg exists for COMPOUND,
#: and a defect that needs three steps to reach the low bits is exactly the kind a
#: short budget reports as green.
CYCLES = 12

#: THE CASE MATRIX. Every axis is here because some mutation is REACHABLE on one
#: value and a structural NULL on the other:
#:
#:   RADIAL EXTENT  the scan's trip count is a RUNTIME loop over ``nxi``. Four extents
#:                  including one shorter than the z extent, so a kernel that happened
#:                  to work at one aspect ratio is not what is certified;
#:   COURANT        ``curl_bz_flat_grouping`` was measured NOT CAUGHT at 0.5 and
#:                  CAUGHT at 0.314159: distributing ``dtdx`` over the prefix
#:                  difference is EXACT whenever ``dtdx`` is a power of two, so a
#:                  matrix without a non-power-of-two Courant cannot see grouping
#:                  errors in this kernel at all. That is a property of the CASES and
#:                  it is why the tag exists and the mutation is scoped by it;
#:   z KIND         METALLIC drives the other ghost/mask arm AND makes the
#:                  ``zero_metal_*`` SEAM live; PERIODIC has no wall pass at all, so
#:                  every seam mutation is a structural null there and is SCOPED
#:                  rather than counted as a miss;
#:   SIGNED ZERO    this family puts signed zeros LIVE on the axis row — ``Dy[0] = 0``
#:                  writes one and the ownership mask writes three more — and random
#:                  normal state contains no zero at all. A defect invisible on the
#:                  state a gate happens to build is reported as UNCAUGHT, which is
#:                  the most dangerous row a mutation leg can print;
#:   MATRIX ROW     the configuration the composition sweep itself drives
#:                  (``dcyl_m0_real``), so the gate and the sweep share one grid
#:                  exactly rather than two that merely resemble each other.
CASES: Tuple[Tuple[str, Dict[str, Any]], ...] = (
    ("wide_metallic", dict(shape=(20, 1, 40), courant=0.5, z_kind="metallic")),
    ("wide_metallic_odd_courant",
     dict(shape=(20, 1, 40), courant=0.3141592653589793, z_kind="metallic")),
    ("wide_periodic", dict(shape=(20, 1, 40), courant=0.5, z_kind="periodic")),
    ("short_periodic_odd_courant",
     dict(shape=(13, 1, 9), courant=0.4142135623730951, z_kind="periodic")),
    ("narrow_metallic_odd_courant",
     dict(shape=(9, 1, 12), courant=0.3141592653589793, z_kind="metallic")),
    ("tall_metallic_odd_courant",
     dict(shape=(32, 1, 16), courant=0.6, z_kind="metallic")),
    ("signed_zeros_metallic",
     dict(shape=(20, 1, 40), courant=0.3141592653589793, z_kind="metallic",
          plant=True)),
    ("composition_matrix_row", dict(matrix_row=True)),
)

#: Scattered ON TOP of the random state rather than replacing it, so the class is
#: reachable without flattening the row-to-row variation the index needles need.
SIGN_ZERO_PLANTS = np.array([0.0, -0.0, 1.5, -1.5, 0.0, -0.0], dtype=np.float32)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _assert_pass_list() -> None:
    """The walk's pass list IS the composer's residency table, minus ``update_P``.

    Asserted rather than commented: if the composer ever gains or reorders a pass and
    this list does not, the whole-step leg would keep certifying a step the engine no
    longer runs, and it would do so silently and greenly.
    """
    expected = tuple(name for name in metal_coverage.RESIDENCY_ORDER
                     if name != "update_P")
    got = tuple(name for name, _fn, _p in PASSES)
    assert got == expected, (
        f"the walked pass list {got} is not the composer's residency order "
        f"{expected}; this leg would certify a step the engine does not run")


def plant_signed_zeros(fields: Any) -> int:
    """Scatter both signed zeros through the stored state, ON THE AXIS ROW especially.

    The axis row and its first neighbour are planted explicitly: ``Dy[r=0] = 0``,
    ``Bx[r=0] = 0`` and three ownership-mask writes all put an exact zero on row 0, and
    the D-side post-add reads ``Hp`` there. A scatter that missed those rows would
    leave this family's own rows the only ones without the class.
    """
    planted = 0
    for offset, name in enumerate(STATE):
        volume = getattr(fields, name, None)
        if volume is None:
            continue
        nr, ny, nz = volume.shape
        for index, value in enumerate(SIGN_ZERO_PLANTS):
            for row in range(min(2, nr)):          # the axis row and its neighbour
                volume[row, 0, (index * 3 + offset) % nz] = value
                planted += 1
            volume[(index * 5 + offset) % nr, 0, (index * 7 + offset) % nz] = value
            planted += 1
    return planted


def build(label: str, seed: int = 99) -> Tuple[Any, Any]:
    """A real Dcyl ``(Fields, PML)`` pair at m = 0, seeded in the PHYSICAL BAND.

    THE SEED COVERS ``fu_*`` AND ``f_w_*`` AS WELL AS THE TWELVE FIELD VOLUMES, and
    that is not cosmetic. Zero init is a FIXED POINT of the constitutive sub-step and
    a near-fixed-point of the split-field recurrence — a no-op agreeing with a no-op
    is trivially identical — so the auxiliaries carry physical-band values from step
    zero and the vacuity floors below have something to measure.

    ``matrix_row`` builds through ``metal_composition_matrix.cylindrical`` instead, so
    one case in this gate is the configuration the composition sweep itself drives
    rather than a second grid that merely resembles it.
    """
    from meep_gpu.fields import Fields  # noqa: PLC0415
    from meep_gpu.grid import Grid  # noqa: PLC0415
    from meep_gpu.pml import PML  # noqa: PLC0415

    options = dict(CASES)[label]
    if options.get("matrix_row"):
        fields, pml = matrix.cylindrical(m=0, complex_storage=False)
        shape = tuple(int(n) for n in fields.grid.shape)
    else:
        shape = tuple(int(n) for n in options["shape"])
        grid = Grid(resolution=1.0,
                    cell_size=(float(shape[0]), 0.0, float(shape[2])),
                    cylindrical=True, m=0,
                    boundaries={"z": options["z_kind"]},
                    courant=float(options["courant"]), xp=np)
        assert tuple(grid.shape) == shape, (tuple(grid.shape), shape)
        fields = Fields(grid=grid, force_complex_fields=False)
        matrix._epsilon(fields)
        fields.enable_pml_storage()
        pml = PML(grid=grid, thickness={"x": (0, max(2, shape[0] // 4)),
                                        "z": max(2, shape[2] // 4)})

    rng = np.random.default_rng(seed)
    for name in STATE:
        array = getattr(fields, name, None)
        if array is not None:
            array[...] = rng.uniform(-1.0, 1.0, size=shape).astype(array.dtype)
    if options.get("plant"):
        plant_signed_zeros(fields)
    return fields, pml


def dtdx_of(fields: Any) -> float:
    return float(fields.grid.dt) / float(fields.grid.dx)


def is_power_of_two(value: float) -> bool:
    """Is this float32 an exact power of two — mantissa bits all zero?

    THE TAG THIS DECIDES IS LOAD-BEARING. Distributing ``dtdx`` over a difference is
    EXACT when ``dtdx`` is a power of two, so ``curl_bz_flat_grouping`` is a
    structural null on such a case; scoping it by this predicate is what stops a
    structural null being reported as a partial catch.
    """
    word = int(words(np.float32(value))[0])
    return (word & 0x007FFFFF) == 0 and value != 0.0


def tags(fields: Any, pml: Any) -> Tuple[str, ...]:
    """What a case IS, derived from the ENGINE rather than typed beside the row.

    Every mutation scopes itself with these. Typing them into the matrix would put
    this family's two structural splits — whether a wall pass exists, and whether the
    Courant scale is exact — in a second place, and a scoping that disagreed with the
    grid would silently turn a reachable defect into a NEEDLE-MISSED row.
    """
    grid = fields.grid
    found = ["walled" if bool(grid.has_metallic) else "unwalled",
             "power_of_two_courant" if is_power_of_two(dtdx_of(fields))
             else "odd_courant"]
    if int(grid.shape[0]) < int(grid.shape[2]):
        found.append("narrow")
    if carries_signed_zeros(fields):
        found.append("signed_zeros")
    return tuple(found)


def carries_signed_zeros(fields: Any) -> bool:
    """Whether :func:`plant_signed_zeros` was written into this state.

    Recovered from the state's OWN BITS — a negative zero somewhere on the axis row —
    rather than from a label the caller carried along beside it. A tag that disagreed
    with the state would turn a reachable defect into a NEEDLE-MISSED row.
    """
    for name in ("Bx", "Ey", "Hy"):
        volume = getattr(fields, name, None)
        if volume is None:
            continue
        if np.any(words(volume[0]) == np.uint32(0x80000000)):
            return True
    return False


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

def build_curl_plan(fields: Any, pml: Any, sub_step: str,
                    residency: Optional[Residency] = None) -> Any:
    residency = Residency() if residency is None else residency
    plan = cyl.plan_cylindrical_real_curl(fields, pml, sub_step, residency)
    if plan is None:
        raise AssertionError(cyl.cylindrical_real_curl_coverage(
            fields, pml, sub_step, residency).reasons)
    return plan


def run_curl(fields: Any, pml: Any, sub_step: str,
             prefix_mutant: Optional[Any] = None,
             curl_mutant: Optional[Any] = None,
             launches: int = 1) -> Any:
    """Launch this family's curl through the ENGINE route, or with a mutant bound.

    THE MUTANT SEAMS ARE SEPARATE, and dropping either would not be a slowdown but a
    silent DISARMING: every mutation leg would launch the shipped kernel and report
    its defect as uncaught. The plan is otherwise built exactly as the composer
    builds it, so what is measured is the object the engine would run.
    """
    residency = Residency()
    plan = build_curl_plan(fields, pml, sub_step, residency)
    if prefix_mutant is not None:
        plan._prefix_functions[shaders.CONTRACT_OFF] = prefix_mutant
    if curl_mutant is not None:
        plan._curl_functions[shaders.CONTRACT_OFF] = curl_mutant
    residency.sync_in()
    for _ in range(launches):
        plan.run()
    residency.sync_out()
    return plan


def run_constitutive(fields: Any, pml: Any, side: str) -> Any:
    residency = Residency()
    plan = cyl.plan_cylindrical_real_constitutive(fields, pml, side, residency)
    if plan is None:
        raise AssertionError(cyl.cylindrical_real_constitutive_coverage(
            fields, pml, side, residency).reasons)
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
    return tuple(live)


def compose(fields: Any, pml: Any) -> Tuple[Any, Any, Tuple[str, ...],
                                            Tuple[str, ...]]:
    """Plan the step TWICE, and the second time is not redundant.

    The residency verdict needs the SYNCED set — which array-path passes the caller
    brackets with an explicit sync out and back — and no caller knows that set until
    it knows which slots the composer filled. So the first call is a throwaway that
    answers "which slots are mine", and the second is the composition this gate runs.
    """
    live = live_passes(fields, pml)
    scout = launch.plan_step(fields, pml, residency=Residency(), sources=())
    on_device = set(scout.plans)
    synced = tuple(name for name in live if name not in on_device)

    residency = Residency()
    plan = launch.plan_step(fields, pml, residency=residency, sources=(),
                            synced=synced)
    return plan, residency, live, synced


def _call(function: Callable[..., Any], fields: Any, pml: Any,
          takes_pml: bool) -> Any:
    """The seam passes take fields alone; the four arithmetic sub-steps take the PML."""
    return function(fields, pml) if takes_pml else function(fields)


def walk(fields: Any, pml: Any, plan: Any, residency: Any,
         skip: Sequence[str] = (), unsynced: Sequence[str] = (),
         order: Optional[Sequence[Tuple[str, Callable[..., Any], bool]]] = None,
         drift: Optional[List[Dict[str, Any]]] = None,
         step: int = -1) -> None:
    """One complete driver step, Metal where the composer filled the slot.

    A slot the composer did NOT fill runs on the ARRAY PATH and is bracketed with an
    explicit ``sync_out`` / ``sync_in``. THE BRACKET IS CHECKED INSIDE ITSELF: after
    the ``sync_in`` and before anything else, ``Residency.verify()`` must be clean,
    because that is the one moment at which a host pass that wrote an array nobody
    pushed back is visible. ``unsynced`` removes the bracket for one named pass, which
    is how leg ``mutations`` measures that it is load-bearing rather than ceremonial.
    """
    for name, function, takes_pml in (PASSES if order is None else order):
        if name in skip:
            continue
        if name in plan.plans:
            plan.plans[name].run()
            continue
        if name in unsynced:
            _call(function, fields, pml, takes_pml)   # THE MUTATION: no bracket
            continue
        residency.sync_out()
        _call(function, fields, pml, takes_pml)
        residency.sync_in()
        if drift is not None:
            stale = residency.verify()
            if stale:
                drift.append({"step": step, "pass": name, "stale": stale})


def reference_step(fields: Any, pml: Any, skip: Sequence[str] = (),
                   order: Optional[Sequence[Tuple[str, Callable[..., Any],
                                                  bool]]] = None) -> None:
    for name, function, takes_pml in (PASSES if order is None else order):
        if name in skip:
            continue
        _call(function, fields, pml, takes_pml)


# ---------------------------------------------------------------------------
# LEG execution
# ---------------------------------------------------------------------------

def leg_execution(payload: Dict[str, Any], out: str) -> None:
    """What was hashed, and does every specialisation BUILD.

    A specialisation that fails to COMPILE is a crash at plan time on a configuration
    nobody swept, so the whole enumeration is built here — in BOTH contraction modes,
    because the contraction pragma is the one directive this backend's byte identity
    rests on and a mode that failed to compile would silently never be exercised.
    """
    _assert_pass_list()
    sources = cyl.enumerate_cylindrical_real_sources()
    kit.provenance(os.path.dirname(os.path.abspath(out)), {
        "cylindrical_real.py": os.path.join(
            API_ROOT, "meep_gpu/metal_kernels/cylindrical_real.py"),
        "templates.py": os.path.join(API_ROOT, "meep_gpu/metal_kernels/templates.py"),
        "shaders.py": os.path.join(API_ROOT, "meep_gpu/metal_kernels/shaders.py"),
        "launch.py": os.path.join(API_ROOT, "meep_gpu/metal_kernels/launch.py"),
        "device.py": os.path.join(API_ROOT, "meep_gpu/metal_kernels/device.py"),
        "stepping.py": os.path.join(API_ROOT, "meep_gpu/stepping.py"),
        "gate": os.path.abspath(__file__),
    }, kernel_sources=sources, name="provenance_gate.json")

    built = 0
    for mode in shaders.CONTRACT_MODES:
        for source in cyl.enumerate_cylindrical_real_sources(mode).values():
            compile_source(source)
            built += 1
    kit.assert_moved(built, "no specialisation compiled", floor=2 * len(sources))
    assert len(sources) == 10, (
        f"the enumeration is {len(sources)} sources, not 10 (two scans and eight "
        f"curls). A different count means an axis became a specialisation, or the "
        f"unreachable periodic-r half was enumerated as if it shipped")

    # THE r AXIS IS NOT ENUMERABLE AS PERIODIC AND THE EMITTER SAYS SO BY NAME. A
    # kernel whose far ghost wrapped the outer face onto the axis row is the one
    # wrong answer this family's whole geometry cannot survive, so the refusal is
    # exercised rather than trusted.
    refused = None
    try:
        cyl.cylindrical_curl_source("step_B", (cyl.PERIODIC, cyl.METALLIC,
                                               cyl.METALLIC))
    except ValueError as exc:
        refused = str(exc)
    assert refused and "METALLIC" in refused, refused

    payload["legs"]["execution"] = {
        "specialisations": len(sources),
        "specialisation_labels": sorted(sources),
        "compiled": built,
        "contract_modes": list(shaders.CONTRACT_MODES),
        "periodic_r_axis_refused": refused,
        "environment": ENVIRONMENT,
        "cases": [label for label, _ in CASES],
        "cycles": CYCLES,
        "pass_list": [name for name, _fn, _p in PASSES],
        "no_generated_code_audit": (
            "torch.mps.compile_shader exposes no AIR, no GPU ISA and no optimisation "
            "report, so this gate cannot refuse a compile whose emitted code violates "
            "the policy nor establish that the contraction guard was obeyed. Every "
            "leg here is BEHAVIOURAL: it catches a wrong answer, never a wrong "
            "instruction."),
        "no_throughput_claim": (
            "the scan's dispatch is sized from its OUTPUT volume, so nr - 1 threads "
            "in every nr return immediately, and the curl pays a second launch per "
            "sub-step. Nothing here is a throughput measurement and this family's "
            "number may not be inherited from the certified families' benchmark."),
    }
    save(payload, out)
    log(f"[execution] specialisations={len(sources)} compiled={built} "
        f"cases={len(CASES)}")


# ---------------------------------------------------------------------------
# LEG prefix — the decisive inversion, re-measured through the SHIPPED plan
# ---------------------------------------------------------------------------

def reference_prefix(fields: Any, sub_step: str) -> Any:
    """``stepping``'s own prefix for this sub-step — CALLED, never re-derived.

    The B side scans Ep EXTENDED BY ITS ZERO WALL ROW at ``ir0 = 0.0``
    (stepping.py:344-360); the D side scans Hp raw at ``ir0 = 0.5`` (:447-448). Both
    are transcribed from the array path's own lines and then handed to the array
    path's own function, so a divergence here cannot be an argument about what the
    reference is.
    """
    if sub_step == "step_B":
        source = fields.Ey
        rows = source.shape[0]
        extended = np.empty((rows + 1,) + source.shape[1:], dtype=source.dtype)
        extended[:rows] = source
        extended[rows] = 0
        return stepping.cylindrical_rderiv_prefix(np, extended, 0.0)
    source = fields.Hy
    return stepping.cylindrical_rderiv_prefix(np, source, 0.5)


def leg_prefix(payload: Dict[str, Any], out: str) -> None:
    """The DEVICE scan alone against the array path's own prefix.

    THIS IS THE LEG THAT PAYS FOR THE DESIGN. The Triton cylindrical module refuses a
    device scan because on CuPy the ORACLE ITSELF is unreproducible — ``cupy.cumsum``
    in float32 is not a sequential accumulation. That premise is INVERTED here rather
    than ignored: this backend's engine holds NumPy, ``numpy.cumsum`` IS sequential,
    and a column-serial device scan reproduces it by construction. Construction is an
    argument; this is the measurement, on the SHIPPED PLAN's own scan launch rather
    than on a bare kernel a harness assembled.

    The row vectors are ALSO checked here, against ``_cylindrical_rderiv_weights``
    called: they are built in float64 and rounded to float32 exactly ONCE
    (stepping.py:1339-1341), and recomputing them in float32 would round twice on the
    divisor and put a silent half-ulp error on the whole radial ladder.
    """
    rows: List[Dict[str, Any]] = []
    for label, _ in CASES:
        fields, pml = build(label)
        for sub_step in ("step_B", "step_D"):
            expected = reference_prefix(fields, sub_step)
            residency = Residency()
            plan = build_curl_plan(fields, pml, sub_step, residency)
            residency.sync_in()
            plan.run_prefix()
            residency.sync_out()
            got = plan.prefix_host

            assert tuple(got.shape) == tuple(expected.shape), (
                label, sub_step, got.shape, expected.shape)
            differ = differing(got, expected)
            moved = differing(got, np.zeros_like(got))
            kit.assert_moved(moved, f"{label}/{sub_step} scan output is all zero",
                             floor=64)
            assert plan.prefix_launches == 1 and plan.curl_launches == 0, (
                label, sub_step, plan.prefix_launches, plan.curl_launches)

            weights, divisor = cyl.prefix_row_vectors(sub_step,
                                                      int(fields.grid.shape[0]))
            band = int(subnormal.census(got)) + int(subnormal.census(expected))
            row = {"case": label, "sub_step": sub_step, "tags": tags(fields, pml),
                   "scan_shape": list(plan.scan_shape),
                   "wall_row": bool(cyl.PREFIX_WALL_ROW[sub_step]),
                   "ir0": cyl.PREFIX_IR0[sub_step],
                   "compared": int(words(expected).size),
                   "differing": differ, "nonzero_words": moved,
                   "weights_words": int(weights.size),
                   "divisor_words": int(divisor.size),
                   "subnormal_words": band,
                   "digest": kit.state_digest({"prefix": expected})}
            rows.append(row)
            payload["legs"]["prefix"] = rows
            save(payload, out)
            log(f"[prefix] {label:<28} {sub_step} shape={plan.scan_shape} "
                f"compared={row['compared']} differing={differ} band={band}")

    # WHERE THE float64 INTERMEDIATE ACTUALLY BITES, measured rather than asserted.
    # The module rounds the ladders ONCE from float64 on the stated ground that a
    # float32 recomputation would round the divisor twice. At ir0 in {0.0, 0.5} the
    # counts are integers and the divisor half-integers — EXACT in float32 — so the
    # two spellings agree bitwise until the integers stop being representable. The
    # crossing is recorded here so the mutation row that measures 0/8 can cite a
    # number instead of an argument.
    def float32_ladder(rows_count: int, ir0: float) -> Tuple[Any, Any]:
        counts = np.arange(rows_count, dtype=np.float32) + np.float32(ir0)
        return counts, counts[1:] - np.float32(0.5)

    reachability: List[Dict[str, Any]] = []
    for sub_step in ("step_B", "step_D"):
        ir0 = cyl.PREFIX_IR0[sub_step]
        for radial in sorted({int(dict(CASES)[label].get("shape", (16, 1, 20))[0])
                              for label, _ in CASES if not
                              dict(CASES)[label].get("matrix_row")} | {16}):
            count = cyl.scan_rows(sub_step, radial)
            once = cyl.prefix_row_vectors(sub_step, radial)
            twice = float32_ladder(count, ir0)
            reachability.append({
                "sub_step": sub_step, "radial_rows": radial, "scan_rows": count,
                "weights_differing": differing(once[0], twice[0]),
                "divisor_differing": differing(once[1], twice[1])})
    crossing = None
    for exponent in range(10, 26):
        count = 2 ** exponent
        edge = np.arange(count - 1, count + 1, dtype=np.float64) + 0.5
        edge32 = np.arange(count - 1, count + 1, dtype=np.float32) + np.float32(0.5)
        if differing((edge - 0.5).astype(np.float32),
                     edge32 - np.float32(0.5)):
            crossing = count
            break
    assert all(row["weights_differing"] == 0 and row["divisor_differing"] == 0
               for row in reachability), reachability
    payload["legs"]["prefix_row_vector_rounding"] = {
        "per_shape": reachability,
        "first_divergent_row_count": crossing,
        "finding": (
            "the float64-then-round-once ladder and a pure float32 recomputation are "
            "BIT-IDENTICAL at every radial extent in this matrix, and first diverge "
            f"around {crossing} rows. ir0 is 0.0 or 0.5, so the counts are integers "
            "and the divisor half-integers, both exact in float32 until the integers "
            "themselves stop being representable. The module's stated reason for the "
            "float64 intermediate — a silent half-ulp error on the radial ladder — is "
            "therefore NOT REACHABLE on any constructible grid. The spelling is still "
            "right, for the other reason it also gives: _cylindrical_rderiv_weights "
            "is the one definition and calling it is what stops the two drifting."),
    }
    payload["legs"]["prefix"] = rows
    save(payload, out)
    log(f"[prefix] row-vector rounding: identical at every shape here, first "
        f"divergence at ~{crossing} rows")


# ---------------------------------------------------------------------------
# LEG curl — the two-launch product, per sub-step
# ---------------------------------------------------------------------------

def leg_curl(payload: Dict[str, Any], out: str) -> None:
    """Per sub-step, the whole product against ``stepping.step_B`` / ``step_D``.

    THE LAUNCH SPLIT IS ASSERTED, not reported. ``launches`` counts KERNEL launches
    and one ``run`` owes exactly one scan and one curl; a plan that ran the curl twice
    on a stale prefix, or the scan and no curl, would otherwise be indistinguishable
    from a plan that ran once correctly.
    """
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
            assert (plan.launches, plan.prefix_launches, plan.curl_launches) == (
                2, 1, 1), (label, sub_step, plan.launches, plan.prefix_launches,
                           plan.curl_launches)
            per = divergence(fields, after)
            row = {"case": label, "sub_step": sub_step, "tags": tags(fields, pml),
                   "shape": list(plan.shape), "bc": list(plan.bc),
                   "dtdx": plan.dtdx, "axis_coef": plan.axis_coef,
                   "moved": moved, "compared": compared_words(after),
                   "differing": sum(per.values()), "per_target": per,
                   "launches": plan.launches,
                   "prefix_launches": plan.prefix_launches,
                   "curl_launches": plan.curl_launches,
                   "digest": kit.state_digest(after)}
            rows.append(row)
            payload["legs"]["curl"] = rows
            save(payload, out)
            log(f"[curl] {label:<28} {sub_step} bc={plan.bc} moved={moved} "
                f"compared={row['compared']} differing={sum(per.values())}")
            restore(fields, before)
    payload["legs"]["curl"] = rows
    save(payload, out)


# ---------------------------------------------------------------------------
# LEG constitutive — no new device code, so the claim is the ADMISSION's
# ---------------------------------------------------------------------------

def leg_constitutive(payload: Dict[str, Any], out: str) -> None:
    """The CERTIFIED real body under this family's restated predicate.

    That this family emits NO constitutive source is ASSERTED rather than described:
    the enumeration carries no constitutive label, so an edit that added a body here
    would show up in the count the execution leg records. The plan CLASS is asserted
    too — it must be the certified ``ConstitutivePlan``, because "same kernel" is only
    a fact if the plan came from the certified builder.
    """
    assert not any("constitutive" in label for label in
                   cyl.enumerate_cylindrical_real_sources()), (
        "this family emitted a constitutive source: it is supposed to re-admit the "
        "certified real constitutive kernel, not replace it")
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
            assert isinstance(plan, ConstitutivePlan), type(plan)
            per = divergence(fields, after)
            rows.append({"case": label, "side": side, "tags": tags(fields, pml),
                         "moved": moved, "compared": compared_words(after),
                         "differing": sum(per.values()), "per_target": per,
                         "plan_class": type(plan).__name__})
            payload["legs"]["constitutive"] = rows
            save(payload, out)
            log(f"[constitutive] {label:<28} {side} moved={moved} "
                f"compared={rows[-1]['compared']} "
                f"differing={sum(per.values())}")
            restore(fields, before)
    payload["legs"]["constitutive"] = rows
    save(payload, out)


# ---------------------------------------------------------------------------
# LEG axis — the r = 0 seam, established three different ways
# ---------------------------------------------------------------------------

def leg_axis(payload: Dict[str, Any], out: str) -> None:
    """The two halves of the kernel's ``BCX = METALLIC`` choice, plus a deadness fact.

    ``stepping._boundary_kinds`` returns ``CYL_AXIS`` on r and the kernel compiles
    METALLIC anyway. That is two claims, not one:

    * the FAR ghost is a SOURCE IDENTITY — ``_shift_up`` is ONE branch over
      ``(MIRROR, METALLIC, CYL_AXIS)`` writing the same hard zero for all three. There
      is nothing to measure; the branch TEXT is asserted, so a future split of it
      fails here rather than in a field;
    * the NEAR ghost is NOT. ``_shift_down``'s ``CYL_AXIS`` branch writes the
      ``r_to_minus_r`` image where METALLIC writes zero, and the kernel's claim is
      that the ownership mask zeroes the only row that can consume it. That is a claim
      about ``stepping.py``, so it is MEASURED on the ARRAY PATH — no Metal involved —
      by stepping two engines that differ only in that one rule, WITH THE SUBSTITUTION
      COUNT FLOORED. A count of zero would mean the leg measured nothing; a nonzero
      DIVERGENCE would mean this family must carry the near ghost, and the leg refuses
      rather than relaxing.

    THE THIRD ROW IS A DEADNESS MEASUREMENT AND IT CORRECTS AN EARLIER CLAIM. The
    build round's mutation table restored the near ghost on ``b_x`` and recorded the
    resulting null as evidence that the METALLIC substitution is equivalent. Reading
    the emitted source, ``b_x`` is not consumed by ANY curl on EITHER sub-step — the B
    side's Bz curl is the prefix difference and the D side's Dz curl takes ``pb_x`` —
    so that null was a DEAD-STORE fact, not a masking fact, and it certified nothing.
    The live shift-down-along-r operands are ``c_x`` (Hz, partner of Dy) and ``pb_x``
    (the prefixed Hp, partner of Dz); leg ``mutations`` re-arms the nulls THERE, and
    pairs each with a mask-dropped twin that must be CAUGHT. This row records the
    deadness itself as a number.

    THE FOURTH ROW IS A VACUITY FLOOR NOBODY ELSE CARRIES: how many words the
    reference moved ON ROW 0. Every cylindrical addition except the prefix acts on
    that row, and a matrix whose reference never touched it would certify the r = 0
    machinery by never exercising it.
    """
    source = inspect.getsource(stepping._shift_up)
    far_identity = "if boundary in (MIRROR, METALLIC, CYL_AXIS):" in source
    assert far_identity, (
        "stepping._shift_up no longer serves CYL_AXIS from the METALLIC branch; the "
        "cylindrical kernel's r-axis = METALLIC specialisation rests on that identity "
        "and this gate's whole r-axis argument is void")

    original = stepping._shift_down
    rows: List[Dict[str, Any]] = []
    for label, _ in CASES:
        for sub_step in ("step_B", "step_D"):
            fields_a, pml_a = build(label, seed=23)
            fields_b, pml_b = build(label, seed=23)
            names = curl_names(sub_step)
            before = snapshot(fields_a, names)
            getattr(stepping, sub_step)(fields_a, pml_a)

            fired = {"count": 0}

            def metallic_near_ghost(xp, field, axis, boundary, component,
                                    phase=None, mirror_phase=None, scratch=None,
                                    scratch_tag="shift_down", _original=original):
                if boundary == stepping.CYL_AXIS:
                    fired["count"] += 1
                    boundary = stepping.METALLIC
                    mirror_phase = None
                return _original(xp, field, axis, boundary, component, phase,
                                 mirror_phase, scratch, scratch_tag)

            stepping._shift_down = metallic_near_ghost
            try:
                getattr(stepping, sub_step)(fields_b, pml_b)
            finally:
                stepping._shift_down = original

            moved = sum(differing(before[n], getattr(fields_a, n)) for n in names)
            near_ghost = sum(differing(getattr(fields_a, n), getattr(fields_b, n))
                             for n in names)
            row_zero = sum(differing(before[n][0], getattr(fields_a, n)[0])
                           for n in names)
            kit.assert_moved(moved, f"{label}/{sub_step} axis leg reference")
            kit.assert_census_floor(
                row_zero, f"{label}/{sub_step} ROW-0 words the reference moved")
            row = {"case": label, "sub_step": sub_step,
                   "tags": tags(fields_a, pml_a),
                   "moved": moved, "row_zero_words_moved": row_zero,
                   "near_ghost_substitutions_fired": fired["count"],
                   "near_ghost_differing": near_ghost,
                   "far_ghost_source_identity": far_identity}
            # THE NEAR GHOST IS ONLY REACHABLE ON THE D SIDE, and this leg MEASURED
            # that rather than assuming it: `_shift_down` is called only from the
            # BACKWARD operand path, so step_B — a forward difference throughout —
            # fires the CYL_AXIS branch ZERO times and its null is structural. The
            # floor is therefore asserted where the class exists and the other side
            # is recorded as a predicted null WITH its reason, because a predicted
            # null quietly dropped is indistinguishable from a case that was
            # forgotten.
            if bool(cyl.SUB_STEPS[sub_step]["backward"]):
                kit.assert_census_floor(
                    fired["count"],
                    f"{label}/{sub_step} CYL_AXIS near-ghost substitutions fired")
            else:
                kit.predicted_null(
                    row, "step_B is a forward difference throughout, so _shift_down "
                         "is never called and the CYL_AXIS near-ghost branch cannot "
                         "fire on this sub-step at all")
                assert fired["count"] == 0, (label, sub_step, fired["count"])
            rows.append(row)
            payload["legs"]["axis"] = rows
            save(payload, out)
            log(f"[axis] {label:<28} {sub_step} row0_moved={row_zero} "
                f"ghost_fired={fired['count']} near_ghost_differing={near_ghost}")
            assert near_ghost == 0, (
                f"{label}/{sub_step}: the r = 0 NEAR ghost is OBSERVABLE "
                f"({near_ghost} words). The kernel compiles the r axis METALLIC and "
                f"this family would have to carry the r_to_minus_r image — a "
                f"legitimate outcome, and a refusal rather than a test to relax")

    # THE DEADNESS OF `b_x`, measured rather than asserted: a garbage initialiser on
    # both sub-steps must change NOTHING, which is what makes the previous round's
    # `b_x` near-ghost null a dead-store fact rather than a masking one.
    deadness: Dict[str, int] = {}
    for label, _ in (CASES[0], CASES[1]):
        fields, pml = build(label)
        for sub_step in ("step_B", "step_D"):
            names = curl_names(sub_step)
            _before, after, moved = oracle(
                fields, lambda: getattr(stepping, sub_step)(fields, pml), names)
            kit.assert_moved(moved, f"{label}/{sub_step} deadness reference",
                             floor=64)
            plan = build_curl_plan(fields, pml, sub_step)
            mutant = kit.Counter(compile_source(needle(
                cyl.cylindrical_curl_source(sub_step, plan.bc),
                "float b_x = vx ? g1[ox] : 0.0f;",
                "float b_x = 12345.0f;")).cyl_pml_curl_step)
            run_curl(fields, pml, sub_step, curl_mutant=mutant)
            deadness[f"{label}/{sub_step}"] = sum(divergence(fields, after).values())
            assert mutant.launches == 1, (label, sub_step, mutant.launches)
    assert not any(deadness.values()), (
        "b_x is LIVE somewhere: the emitted source was read as consuming it on "
        "neither sub-step, and the correction this leg records would be wrong",
        deadness)

    payload["legs"]["axis_operand_deadness"] = {
        "b_x_garbage_differing": deadness,
        "finding": (
            "b_x — g1's shift-down along r — is consumed by NO curl on EITHER "
            "sub-step: the B side's Bz curl is the prefix difference and the D side's "
            "Dz curl takes the PREFIXED pb_x. A near-ghost mutation planted on b_x is "
            "therefore a dead-store edit and its null certifies nothing. The live "
            "shift-down-along-r operands are c_x (Hz) and pb_x (prefixed Hp), and the "
            "mutation leg arms the nulls there, each paired with a mask-dropped twin "
            "that must be CAUGHT."),
    }
    payload["legs"]["axis"] = rows
    save(payload, out)


# ---------------------------------------------------------------------------
# LEG whole_step — the real arbiter
# ---------------------------------------------------------------------------

def leg_whole_step(payload: Dict[str, Any], out: str, budget: int = CYCLES) -> None:
    """Per COMPLETE STEP and per PASS, with a census WINDOW and asserted counters.

    Every slot's launch counter is asserted against what the plan OWES per cycle: a
    slot that passes by NOT EXECUTING is the hollow pass this discipline exists to
    prevent, and for this family the count is THREE numbers per curl slot, because
    "the plan ran once" and "the scan ran and the curl did not" must be
    distinguishable.

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
        before = snapshot(reference_fields)
        plan, residency, live, synced = compose(fields, pml)
        assert set(plan.plans) == set(ARITHMETIC_SLOTS), (label, sorted(plan.plans))
        assert plan.residency.covered, (label, plan.residency.reasons)
        mirrors = len(residency.names)
        kit.assert_census_floor(mirrors, f"{label} mirrors registered", floor=8)
        residency.sync_in()

        first_divergent: Optional[Dict[str, Any]] = None
        drift: List[Dict[str, Any]] = []
        per_step: List[int] = []
        census_per_step: List[int] = []
        fired: List[int] = []
        compared = 0
        for step in range(budget):
            reference_step(reference_fields, reference_pml)
            walk(fields, pml, plan, residency, drift=drift, step=step)
            residency.sync_out()
            stale = residency.verify()
            if stale:
                drift.append({"step": step, "pass": "end_of_step", "stale": stale})
            per = {name: differing(getattr(fields, name),
                                   getattr(reference_fields, name))
                   for name in STATE if getattr(fields, name, None) is not None}
            compared += sum(int(words(getattr(reference_fields, name)).size)
                            for name in per)
            total = sum(per.values())
            per_step.append(total)
            count = state_census(reference_fields)
            census_per_step.append(count)
            if count:
                fired.append(step)
            if total and first_divergent is None:
                first_divergent = {"step": step,
                                   "targets": {k: v for k, v in per.items() if v}}

        evolved = sum(differing(before[name], getattr(reference_fields, name))
                      for name in before)
        kit.assert_moved(evolved, f"{label} whole-step state never evolved",
                         floor=64)
        counters = {slot: [plan.plans[slot].launches,
                           getattr(plan.plans[slot], "prefix_launches", None),
                           getattr(plan.plans[slot], "curl_launches", None)]
                    for slot in ARITHMETIC_SLOTS}
        for slot in ("step_B", "step_D"):
            assert counters[slot] == [2 * budget, budget, budget], (
                label, slot, counters[slot],
                "each curl slot owes exactly one scan and one curl per cycle; a "
                "plan that scanned once is byte-perfect on step 0 and feeds every "
                "step after it the previous timestep's radial derivative")
        for slot in ("update_H", "update_E"):
            assert counters[slot][0] == budget, (label, slot, counters[slot])
        assert not drift, (label, drift[:3])

        refused = bool(fired)
        if not refused:
            assert first_divergent is None, (label, first_divergent, per_step)
        window = {"first_step": fired[0] if fired else None,
                  "last_step": fired[-1] if fired else None,
                  "steps_censused": budget,
                  "subnormal_words_per_step": census_per_step}
        rows.append({
            "case": label, "tags": tags(fields, pml), "budget": budget,
            "live_passes": list(live), "walked_passes": [n for n, _f, _p in PASSES],
            "inert_passes": [n for n, _f, _p in PASSES if n not in live],
            "synced_passes": list(synced),
            "device_slots": sorted(plan.plans),
            "selected": dict(plan.selected),
            "mirrors_registered": mirrors,
            "per_step_differing": per_step,
            "first_divergent": first_divergent,
            "evolved_words": evolved,
            "subnormal_window": window,
            "residency_drift": drift,
            "refused": refused,
            "refusal": (f"{label}: REFUSED (subnormal precondition) — the census "
                        f"fired on steps {fired[0] if fired else None}.."
                        f"{fired[-1] if fired else None} of {budget}; every "
                        f"arithmetic claim on this backend rides a CHECKED "
                        f"subnormal-free precondition and this case does not meet it"
                        ) if refused else None,
            "compared": 0 if refused else compared,
            "launches": counters,
            "residency_verify": residency.verify(),
        })
        payload["legs"]["whole_step"] = rows
        save(payload, out)
        log(f"[whole_step] {label:<28} {budget} steps x {len(PASSES)} passes, "
            f"compared={rows[-1]['compared']} first_divergent="
            f"{None if first_divergent is None else first_divergent['step']} "
            f"evolved={evolved} counters={counters['step_B']} "
            f"window=[{window['first_step']}, {window['last_step']}]"
            f"{' REFUSED' if refused else ''}")
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
#: THE ``band_edge`` ROW IS RECORDED RATHER THAN ASSERTED, and it is where a WINDOW
#: earns its place over a scalar: the state is scaled UNIFORMLY and the scale itself
#: is a normal float32, so anything the census finds was reached BY THE RUN — through
#: cancellation in the split-field recurrence and down the radial ladder the prefix
#: accumulates — and not by the seeding. A census taken once at step 0 would say
#: "clean" about a run that is not.
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
        reference_fields, reference_pml = build("wide_metallic_odd_courant")
        if scale != 1.0:
            for name in STATE:
                array = getattr(reference_fields, name, None)
                if array is not None:
                    array *= np.array(scale, dtype=np.float32)
        fired: List[int] = []
        per_step: List[int] = []
        for step in range(budget):
            reference_step(reference_fields, reference_pml)
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
    fields, _pml = build("wide_metallic", seed=41)
    clean = preconditions.SubnormalWindow(0, 1, per_array_words=64)
    for name in ("Ey", "Hy", "Ez"):
        clean.observe(name, getattr(fields, name), step=1)
    clean_report = preconditions.assert_clean_or_refuse(clean, "control/clean")

    banded = preconditions.SubnormalWindow(0, 1, per_array_words=64)
    banded.observe("Ey_scaled", (fields.Ey * np.float32(1e-40)).astype(np.float32),
                   step=1)
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
# LEG split — m = 0 versus |m| >= 1, and the other un-inverted clauses
# ---------------------------------------------------------------------------

def leg_split(payload: Dict[str, Any], out: str) -> None:
    """|m| >= 1 is a DIFFERENT KERNEL, and this family says so BY NAME.

    Clause 2 (real storage required) happens to separate the two cylindrical products
    on the engine, because ``stepping.py:731-736`` makes complex storage mandatory at
    |m| >= 1. "Happens to" is an ENGINE COUPLING, not an inverted clause, and a
    residual separated only by a coupling is planted and measured rather than assumed.
    So the m clause is spelled as clause 14 and this is what checks it — at the
    predicate, at the plan and at the composer.

    THE OTHER UN-INVERTED CLAUSES ARE CHECKED HERE TOO, each by the NAME it refuses
    under, because a coverage boundary that only exists as prose is a boundary the
    next edit walks through:

    * clause 6 — a CARTESIAN grid, which this family's radial substitution would
      silently mis-step;
    * clause 10 — an installed chi2/chi3, which belongs to the Pade family. THE
      MAGNITUDE CEILING IS NOT THIS FAMILY'S TO PIN: ``nonlinear_update_e`` carries
      the measured ``chi * |chi1inv|^n <= 1e29`` clause and its own gate pins it. Here
      a nonlinearity is refused OUTRIGHT, at any magnitude, which is the stronger
      statement and the one this predicate actually makes;
    * clause 3 — an inactive absorber, which routes to the no-PML family;
    * clause 5 — a mirror plane, which is invisible in the boundary triple on a Dcyl
      grid (``is_axis`` outranks ``is_mirrored``) and is therefore asked twice.
    """
    record: Dict[str, Any] = {}
    residency = Residency()

    # 1. |m| >= 1, in BOTH storages, refused by clause 14 on all four slots.
    m_refusals: Dict[str, Any] = {}
    for m in (1, -1, 2):
        m_fields, m_pml = matrix.cylindrical(m=m, complex_storage=True)
        per_slot: Dict[str, Any] = {}
        for sub_step in ("step_B", "step_D"):
            verdict = cyl.cylindrical_real_curl_coverage(m_fields, m_pml, sub_step,
                                                         residency)
            named = [r for r in verdict.reasons if f"grid.m = {m}" in r]
            assert not verdict.covered and named, (m, sub_step, verdict.reasons)
            per_slot[sub_step] = named
            assert cyl.plan_cylindrical_real_curl(m_fields, m_pml, sub_step,
                                                  residency) is None
        for side in ("H", "E"):
            verdict = cyl.cylindrical_real_constitutive_coverage(
                m_fields, m_pml, side, residency)
            named = [r for r in verdict.reasons if f"grid.m = {m}" in r]
            assert not verdict.covered and named, (m, side, verdict.reasons)
            per_slot[side] = named
        m_plan = launch.plan_step(m_fields, m_pml, residency=Residency(), sources=())
        assert cyl.FAMILY not in set(m_plan.selected.values()), m_plan.selected
        assert "cylindrical m=0" not in set(m_plan.selected.values()), (
            m_plan.selected)
        per_slot["composer_selection"] = dict(m_plan.selected)
        m_refusals[str(m)] = per_slot
    record["clause_14_refusals"] = m_refusals

    # 2. THE OTHER DIRECTION: complex storage at m = 0. Clause 2 refuses it by name
    #    and names the |m| >= 1 family's kernel rather than shrugging.
    complex_fields, complex_pml = matrix.cylindrical(m=0, complex_storage=True)
    verdict = cyl.cylindrical_real_curl_coverage(complex_fields, complex_pml,
                                                 "step_B", residency)
    named = [r for r in verdict.reasons if "force_complex_fields=True" in r]
    assert not verdict.covered and named, verdict.reasons
    record["clause_2_refusal"] = named

    # 3. Clause 6: a Cartesian grid. The r-axis substitution IS the cylindrical
    #    radial derivative and is a wrong answer on a Cartesian leading axis.
    cart_fields, cart_pml = matrix.cart()
    verdict = cyl.cylindrical_real_curl_coverage(cart_fields, cart_pml, "step_B",
                                                 residency)
    named = [r for r in verdict.reasons if "grid.cylindrical is False" in r]
    assert not verdict.covered and named, verdict.reasons
    record["clause_6_refusal"] = named

    # 4. Clause 10: chi2/chi3, refused OUTRIGHT rather than above a ceiling.
    nl_fields, nl_pml = matrix.nonlinear(
        (matrix.cylindrical(m=0, complex_storage=False)))
    verdict = cyl.cylindrical_real_curl_coverage(nl_fields, nl_pml, "step_B",
                                                 residency)
    named = [r for r in verdict.reasons if "chi2/chi3 is installed" in r]
    assert not verdict.covered and named, verdict.reasons
    record["clause_10_refusal"] = named
    record["clause_10_note"] = (
        "refused at ANY magnitude. The measured 1e29 magnitude ceiling belongs to "
        "nonlinear_update_e and is pinned by ITS gate; this family carries no Pade "
        "factor at all, so a ceiling here would be a weaker statement than the "
        "outright refusal the predicate actually makes.")

    # 5. Clause 3: an inactive absorber routes to the no-PML family.
    flat_fields, flat_pml = matrix.cylindrical(m=0, complex_storage=False)
    verdict = cyl.cylindrical_real_curl_coverage(flat_fields, None, "step_B",
                                                 residency)
    named = [r for r in verdict.reasons if "no active PML layer" in r]
    assert not verdict.covered and named, verdict.reasons
    record["clause_3_refusal"] = named

    # 6. update_P is owned by the independent, geometry-free ADE family, never by
    #    this cylindrical family. Checked at the registry so ownership cannot drift.
    from meep_gpu.metal_kernels import arms as metal_arms  # noqa: PLC0415

    metal_arms.ensure_registered()
    families_on_p = sorted({arm.family for arm in metal_arms.registered("update_P")})
    assert families_on_p == ["ade_update_p"], families_on_p
    record["update_P_arms"] = families_on_p
    slots = sorted({arm.slot for arm in metal_arms.registered()
                    if arm.family == cyl.FAMILY})
    assert slots == sorted(ARITHMETIC_SLOTS), slots
    record["family_slots"] = slots

    payload["legs"]["split"] = record
    save(payload, out)
    log(f"[split] clause 14 refuses |m| >= 1 on 4 slots at m in (1, -1, 2); "
        f"clauses 2/3/6/10 refuse by name; update_P arms={families_on_p}")


# ---------------------------------------------------------------------------
# LEG containment — Metal's admitted set beside Triton's, every slot named
# ---------------------------------------------------------------------------

#: The reasons Triton's predicates give on THIS host that have nothing to do with the
#: cylindrical family. A row whose ONLY refusals are these is NOT COMPARABLE here: the
#: Triton products require CuPy, and the engine on this machine holds NumPy.
BACKEND_ONLY_FRAGMENTS: Tuple[str, ...] = (
    "array module is",
    "expansion probe artifact",
)


def leg_containment(payload: Dict[str, Any], out: str) -> None:
    """Metal-only / Triton-only / both / neither, per slot — and an honest refusal.

    The rule the earlier rounds set: a Metal-only slot is not automatically a defect,
    but it is NEVER accepted on argument. So both predicates are asked, per case per
    slot, and the answer is recorded with the exact refusal strings.

    WHAT IS ACTUALLY MEASURABLE HERE IS LESS THAN THAT, and saying so is the point.
    The Triton cylindrical product's predicate carries a BACKEND clause — it requires
    the engine's array module to be CuPy, and this host's engine holds NumPy — so on
    this machine it refuses every row for a reason that has nothing to do with
    cylindrical geometry. A row whose only Triton refusals are the backend and probe
    clauses is recorded NOT COMPARABLE rather than counted as Metal-only. A
    like-for-like admission diff needs a CuPy host and this gate does not claim one.
    """
    from meep_gpu.triton_kernels import cylindrical_triton as triton_cyl

    # THE TRITON PREDICATE IS NOT SUB-STEP AWARE — ``cylindrical_curl_coverage``
    # takes ``(fields, pml)`` and answers for the curl as a whole — so the same
    # verdict is read against both curl slots and the asymmetry is recorded rather
    # than smoothed over. That difference is itself a containment fact: this family's
    # predicate can refuse ONE sub-step (the D side's stored-Hy clause) where the
    # Triton one cannot.
    curl_coverage = getattr(triton_cyl, "cylindrical_curl_coverage", None)
    constitutive_coverage = getattr(triton_cyl, "cylindrical_constitutive_coverage",
                                    None)
    rows: List[Dict[str, Any]] = []
    counts = {"both": 0, "metal_only": 0, "triton_only": 0, "neither": 0,
              "not_comparable": 0, "no_triton_predicate": 0}
    for label, _ in CASES:
        fields, pml = build(label)
        residency = Residency()
        for slot in ARITHMETIC_SLOTS:
            if slot in ("step_B", "step_D"):
                mine = cyl.cylindrical_real_curl_coverage(fields, pml, slot,
                                                          residency)
                theirs = (curl_coverage(fields, pml)
                          if curl_coverage is not None else None)
            else:
                side = "H" if slot == "update_H" else "E"
                mine = cyl.cylindrical_real_constitutive_coverage(fields, pml, side,
                                                                  residency)
                theirs = (constitutive_coverage(fields, pml, side)
                          if constitutive_coverage is not None else None)
            if theirs is None:
                verdict = "no_triton_predicate_for_this_slot"
                counts["no_triton_predicate"] += 1
                reasons: List[str] = []
                admitted = None
            else:
                reasons = list(theirs.reasons)
                admitted = bool(theirs.covered)
                backend_only = bool(reasons) and all(
                    any(fragment in reason for fragment in BACKEND_ONLY_FRAGMENTS)
                    for reason in reasons)
                if backend_only:
                    verdict = "not_comparable_on_this_host"
                    counts["not_comparable"] += 1
                elif mine.covered and admitted:
                    verdict = "both"
                    counts["both"] += 1
                elif mine.covered:
                    verdict = "metal_only"
                    counts["metal_only"] += 1
                elif admitted:
                    verdict = "triton_only"
                    counts["triton_only"] += 1
                else:
                    verdict = "neither"
                    counts["neither"] += 1
            rows.append({"case": label, "slot": slot, "verdict": verdict,
                         "metal_admitted": bool(mine.covered),
                         "triton_admitted": admitted,
                         "triton_reasons": reasons,
                         "metal_reasons": list(mine.reasons)})
    assert all(row["metal_admitted"] for row in rows), (
        "a case in this gate's own matrix is not admitted by the family under test",
        [r for r in rows if not r["metal_admitted"]][:2])

    # A METAL-ONLY SLOT IS NEVER ACCEPTED ON ARGUMENT. Each one is looked up in THIS
    # RUN's whole-step evidence and must carry a complete-step byte comparison that
    # ran, compared something and did not diverge. If the evidence is absent — because
    # the whole-step leg was not run, or the case was refused on the subnormal
    # precondition — the containment break is UNJUSTIFIED and this leg fails rather
    # than recording a break it cannot support.
    whole_step = {row["case"]: row
                  for row in payload["legs"].get("whole_step", ())}
    justification: Dict[str, Any] = {}
    for row in rows:
        if row["verdict"] != "metal_only":
            continue
        case = row["case"]
        evidence = whole_step.get(case)
        assert evidence is not None, (
            f"{case}/{row['slot']} is METAL-ONLY and this run carries no whole-step "
            f"evidence for it. A Metal-only slot is measured with a complete-step "
            f"byte comparison or refused; run the whole_step leg with this one")
        assert not evidence.get("refused"), (
            f"{case}/{row['slot']} is METAL-ONLY and its whole-step case was REFUSED "
            f"on the subnormal precondition, so the admission rests on no measurement")
        assert (evidence["first_divergent"] is None
                and int(evidence["compared"]) > 0), (case, evidence["compared"],
                                                     evidence["first_divergent"])
        justification[case] = {
            "whole_step_compared": int(evidence["compared"]),
            "whole_step_first_divergent": evidence["first_divergent"],
            "budget": evidence["budget"],
            "triton_reasons": row["triton_reasons"]}

    payload["legs"]["containment"] = {
        "per_slot": rows,
        "counts": counts,
        "slots_examined": len(rows),
        "triton_module": triton_cyl.__name__,
        "verdict": (
            "NOT MEASURABLE ON THIS HOST for every row whose Triton refusals are the "
            "backend clause alone: that product requires CuPy and the engine here "
            "holds NumPy, so no such row carries a comparable Triton verdict and this "
            "gate claims no containment relation in either direction on it. "
            "Triton's full corpus result is 759/759; the older 702 figure was a "
            "nine-certified-family subtotal cut on a CuPy host and is not "
            "re-derived here."),
        "metal_only_justification": justification,
        "metal_only_finding": (
            "MEASURED CONTAINMENT BREAK, and it is substantive rather than the "
            "backend clause. The Triton real cylindrical product pins the boundary "
            "triple to ('axis', 'periodic', 'metallic') as a COMPILED-IN constant "
            "(cylindrical_triton.CYLINDRICAL_BOUNDARY_KINDS) and refuses a PERIODIC z "
            "by name; this family compiles BCZ as a specialisation axis and admits "
            "both. Every Metal-only slot here is a z-periodic Dcyl m = 0 row, and "
            "each one's admission is justified by the whole-step byte comparison "
            "recorded above — not by this argument. On the Triton side that clause is "
            "over-broad by these slots and it is a Triton-track item, not a defect "
            "here."
            if counts["metal_only"] else
            "no Metal-only slot in this matrix"),
        "backend_only_fragments": list(BACKEND_ONLY_FRAGMENTS),
    }
    save(payload, out)
    log(f"[containment] {len(rows)} slot-cases: {counts}; "
        f"metal-only justified by whole-step evidence on "
        f"{sorted(justification) or 'no case'}")


# ---------------------------------------------------------------------------
# LEG mutations — a leg that cannot fail certifies nothing
# ---------------------------------------------------------------------------

#: ``(label, kernel, sub_steps, scope, edits, must_catch, why)`` where ``edits`` is a
#: tuple of ``(old, new)`` pairs applied in order — a tuple rather than one pair
#: because two of this family's defects are genuinely two statements (moving the axis
#: post-add INTO the curl deletes one line and adds another, and a ghost restored
#: WITHOUT its mask is the ghost edit plus the mask edit).
#:
#: ``scope`` is a tag from :func:`tags` or ``None`` for the whole matrix. A defect
#: STRUCTURALLY ABSENT on some cases is scoped rather than reported as a partial
#: catch: rounding a structural null into a weakness is as dishonest as rounding a
#: miss into a pass.
#:
#: ``must_catch=False`` is a NULL CONTROL — a spelling asserted to be genuinely
#: equivalent rather than merely untested.
SOURCE_MUTATIONS: Tuple[Tuple[str, str, Tuple[str, ...], Optional[str],
                              Tuple[Tuple[str, str], ...], Optional[bool],
                              str], ...] = (
    # ---- THE RADIAL SCAN ---------------------------------------------------
    ("prefix_wall_row_dropped", "prefix", ("step_B",), None,
     (("(i < nxi - 1 ? src[i * nyz + base] : 0.0f)",
       "(i < nxi - 1 ? src[i * nyz + base] : 1.0f)"),),
     True,
     "the B side's ZERO WALL ROW carries a value instead of zero. Bz's last-row "
     "forward difference then reads minus the whole accumulated sum "
     "(stepping.py:331-342), which is the symptom that found the wall row in the "
     "first place. "
     "RE-ARMED 2026-08-19, IN BOUNDS. The original spelling replaced the guarded "
     "read with a bare src[i * nyz + base], which on the last row reads PAST THE "
     "END of the prefix volume - so it was caught only when the memory beyond the "
     "buffer happened to differ from zero. Measured: CAUGHT 5/8, then 7/8 on a "
     "re-run of the same bytes, and 8/8 when the case was run alone. A "
     "NON-DETERMINISTIC needle is not a needle, and this file already states the "
     "rule for the mirror-image defect: a gate must not rest on an out-of-bounds "
     "read. The re-arm keeps the defect class (wall row not zero) and reads the "
     "LAST VALID ROW instead, which is in bounds. "
     "RE-ARMED AGAIN 2026-08-27: that row is in bounds but NOT deterministic, and "
     "the determinism was asserted rather than measured. It is the WALL ROW -- the "
     "one this kernel never writes -- living in PLAN-OWNED scratch reused across "
     "launches, so it holds whatever the previous tenant left. "
     "prefix_loop_stops_one_row_short, four entries down, says exactly that of the "
     "same row: 'whatever the plan-owned scratch held, which is a stale value'. "
     "Measured on the re-armed bytes: CAUGHT 8/8, 5/8, 8/8 across three consecutive "
     "runs, with the same run-alone-passes signature the original had -- because "
     "bounds were never the cause. The cause is reading memory this kernel does not "
     "define. The needle is now a LITERAL 1.0f: same defect class (the wall row "
     "carries a value instead of zero), no read of undefined memory, nothing left "
     "for a prior launch to perturb"),
    ("prefix_reciprocal_multiply", "prefix", ("step_B", "step_D"), None,
     (("float inc = (w - prev) / divisor[i - 1];",
       "float inc = (w - prev) * (1.0f / divisor[i - 1]);"),), True,
     "x/y replaced by x*(1/y): algebraically equal, a DIFFERENT float32 number. This "
     "is the mutation that makes the module's `/` rather than a reciprocal a "
     "measurement instead of a style note"),
    ("prefix_weight_off_by_one", "prefix", ("step_B", "step_D"), None,
     (("* weights[i];", "* weights[i - 1];"),), True,
     "shift the radial weight ladder by one row — a half-cell error in ir0, the "
     "silent kind: a smooth, plausible, slightly wrong field (stepping.py:1297-1299)"),
    ("prefix_divisor_off_by_one", "prefix", ("step_B", "step_D"), None,
     (("/ divisor[i - 1];", "/ divisor[i];"),), True,
     "the divisor is the INTEGER SITE THE DIFFERENCE LANDS ON, not the next one "
     "(stepping.py:1312 builds it as counts[1:] - 0.5)"),
    ("prefix_loop_stops_one_row_short", "prefix", ("step_B", "step_D"), None,
     (("for (int i = 1; i < nxi; ++i)", "for (int i = 1; i < nxi - 1; ++i)"),), True,
     "the scan's trip count is a RUNTIME loop over nxi and the last row is the one "
     "the B side's forward difference reads. A short loop leaves it holding whatever "
     "the plan-owned scratch held, which is a stale value rather than a crash"),
    ("prefix_column_decomposition_swapped", "prefix", ("step_B", "step_D"), None,
     (("int k = int(idx) % nzi;\n    int j = int(idx) / nzi;",
       "int j = int(idx) % nzi;\n    int k = int(idx) / nzi;"),), True,
     "swap the (phi, z) column decomposition. On a Dcyl grid ny is 1, so the swapped "
     "form makes every thread but one fail the `j >= nyi` guard and return: the scan "
     "writes one column and leaves the rest of the prefix stale"),
    ("prefix_commuted_add", "prefix", ("step_B", "step_D"), None,
     (("acc = acc + inc;", "acc = inc + acc;"),), False,
     "NULL CONTROL. float addition IS commutative, so this cannot differ in a bit. "
     "Carried as the arm that shows this leg does not flag any edit at all"),

    # ---- Bz's SUBSTITUTED CURL ---------------------------------------------
    ("curl_bz_flat_grouping", "curl", ("step_B",), "odd_courant",
     (("float curl2 = dtdx * (pfx_up - pfx_here);",
       "float curl2 = (dtdx * pfx_up) - (dtdx * pfx_here);"),), True,
     "distribute dtdx over the prefix difference instead of scaling it once "
     "(stepping.py:375). SCOPED TO A NON-POWER-OF-TWO COURANT, and that scoping is a "
     "MEASUREMENT rather than caution: the round that built this family measured the "
     "same edit NOT CAUGHT at Courant 0.5 and CAUGHT at 0.314159, because scaling by "
     "a power of two is exact. Sweeping it unscoped would report a property of the "
     "CASES as a weakness of the kernel"),
    ("curl_bz_substitution_deleted", "curl", ("step_B",), None,
     (("float curl2 = dtdx * (pfx_up - pfx_here);",
       "float curl2 = dtdx * ((b_x - b) + (a - a_y));"),), True,
     "the CARTESIAN four-operand Bz curl where cylindrical needs the prefix "
     "difference — the whole substitution deleted"),
    ("curl_bz_difference_reversed", "curl", ("step_B",), None,
     (("float curl2 = dtdx * (pfx_up - pfx_here);",
       "float curl2 = dtdx * (pfx_here - pfx_up);"),), True,
     "stepping.py:347 is prefix_ext[1:] - prefix_ext[:-1]; reversing it negates Bz's "  # stepping.py live lines for the frozen device-text citation(s) in this string: 347->375
     "whole curl"),
    ("curl_bz_prefix_row_stride_one", "curl", ("step_B",), None,
     (("float pfx_up   = pfx[ii + nyz];", "float pfx_up   = pfx[ii + 1];"),), True,
     "the extended prefix is (nr+1, ny, nz) and the forward difference steps one ROW "
     "— a stride of nyz cells. A stride of one is the z neighbour, stays in bounds "
     "and crashes nothing"),
    ("curl_prefix_leaks_into_dx", "curl", ("step_D",), None,
     (("float curl0 = dtdx * ((c_y - c) + (b - b_z));",
       "float curl0 = dtdx * ((c_y - c) + (pb - b_z));"),), True,
     "feed Dx the PREFIXED Hp where the array path keeps it RAW: stepping.py:418-426 "  # stepping.py live lines for the frozen device-text citation(s) in this string: 418-426->447-455
     "substitutes for the Dz term ONLY, which is why the plan binds both g1 and pfx"),

    # ---- THE r = 0 SEAM: the axis rules and the ownership mask --------------
    ("axis_bx_not_zeroed", "curl", ("step_B",), None,
     (("    v0 = at_x ? 0.0f : v0;", "    v0 = v0 + 0.0f;"),), True,
     "delete Br[r = 0] = 0. A radial vector at r = 0 points nowhere "
     "(stepping.py:661-662)"),
    ("axis_dz_postadd_dropped", "curl", ("step_D",), None,
     (("v2 = at_x ? (v2 + (axis_coef * g1[ii])) : v2;", "v2 = v2 + 0.0f;"),), True,
     "delete the m = 0 on-axis Dz increment, (4*Courant)*Hp[r=0] (stepping.py:586)"),
    ("axis_dz_postadd_folded_into_curl", "curl", ("step_D",), None,
     # Anchored on the r MASK LINE rather than on the curl statement, because `at_x`
     # is declared by the mask block: the fold has to sit after that declaration and
     # before the recurrence, which is exactly where a transcription that folded it
     # would put it. curl2 is already the masked +0.0 there, so the appended line IS
     # the increment routed through the recurrence.
     (("    curl2 = at_x ? 0.0f : curl2;",
       "    curl2 = at_x ? 0.0f : curl2;\n"
       "    curl2 = at_x ? (curl2 - (axis_coef * g1[ii])) : curl2;"),
      ("v2 = at_x ? (v2 + (axis_coef * g1[ii])) : v2;", "v2 = v2 + 0.0f;")), True,
     "route the axis increment THROUGH the split-field recurrence instead of "
     "post-adding it after. TWO EDITS, because that is what the defect is. This is "
     "the exact error stepping.py:575-580 records as MEASURED WRONG for m = 0 under "
     "PML (Er 2.7e-01 / Hp 4.5e-01 against 3.6e-07 for the post-add), because Dz's "
     "dsig is R, whose sigma is zero on the axis row"),
    ("axis_dp_not_zeroed", "curl", ("step_D",), None,
     (("    v1 = at_x ? 0.0f : v1;", "    v1 = v1 + 0.0f;"),), True,
     "delete Dp[r = 0] = 0 (stepping.py:587)"),
    ("r_ownership_mask_dropped_on_dy", "curl", ("step_D",), None,
     (("    curl1 = at_x ? 0.0f : curl1;",
       "    // MUTANT: the r ownership mask on Dy dropped"),), True,
     "MEEP's little_owned_corner0 deliberately excludes r = 0 (vec.hpp:1100-1104) — "
     "the axis row belongs to the per-m rules. On Dy the mask is LOAD-BEARING: its "
     "radial operand pair at r = 0 is (Hz[0], the metallic zero ghost), whose "
     "difference is Hz[0] and is not zero"),
    ("r_ownership_mask_dropped_on_bx", "curl", ("step_B",), None,
     (("    curl0 = at_x ? 0.0f : curl0;",
       "    // MUTANT: the r ownership mask on Bx dropped"),), True,
     "the SAME mask on the B side, and it is worth its own row because the FIELD "
     "cannot carry it: v0 is overwritten by the axis rule at exactly at_x, so the "
     "only witness is the AUXILIARY fu_Bx. A gate comparing fields alone would report "
     "this defect as uncaught"),
    ("r_ownership_mask_dropped_on_dz", "curl", ("step_D",), None,
     (("    curl2 = at_x ? 0.0f : curl2;",
       "    // MUTANT: the r ownership mask on Dz dropped"),), False,
     "MEASURED NULL, 0/8, AND IT CORRECTS THE FAMILY'S STATED MECHANISM. This row was "
     "armed CAUGHT on the module's own reasoning — that the ownership mask is what "
     "zeroes the r = 0 curl of the two shift-down-along-r targets — and it came back "
     "0/8. Measured directly: on Dz at m = 0 the curl at r = 0 is ALREADY exactly "
     "+0.0 without the mask, because BOTH its operand differences vanish exactly. The "
     "radial pair is (prefix[0], the metallic zero ghost) and prefix row 0 is an "
     "exact +0.0 by construction (stepping.py:1314 zeros_like, and the census here "
     "confirms every word of that row is 0x00000000); the phi pair is a "
     "self-difference on the one-cell invariant axis, which wraps onto itself. So the "
     "mask writes the word that is already there. The CONCLUSION the family draws — "
     "the CYL_AXIS near ghost is unobservable — is unaffected and separately measured "
     "in leg axis; the MECHANISM it names is right for Dy and wrong for Dz, and that "
     "is what this row now records"),
    ("near_ghost_restored_on_hz", "curl", ("step_D",), None,
     (("float c_x = vx ? g2[ox] : 0.0f;", "float c_x = vx ? g2[ox] : (-g2[ii]);"),),
     False,
     "put the r_to_minus_r NEAR GHOST back on Hz — Dy's partner, one of the two LIVE "
     "shift-down-along-r operands (stepping.py:1877-1889). MUST NOT be caught: it is "
     "the arm that proves the METALLIC substitution is equivalent rather than merely "
     "untested, and here the null IS a masking fact, which the twin below measures"),
    ("near_ghost_restored_on_hz_WITH_mask_dropped", "curl", ("step_D",), None,
     (("float c_x = vx ? g2[ox] : 0.0f;", "float c_x = vx ? g2[ox] : (-g2[ii]);"),
      ("    curl1 = at_x ? 0.0f : curl1;",
       "    // MUTANT: the r ownership mask on Dy dropped")), True,
     "THE PAIRED TWIN, and the row that makes the null above mean something. Restore "
     "the near ghost AND drop the mask that hides it: CAUGHT here means the operand "
     "is live and the null next door is a MASKING measurement rather than a "
     "dead-store artifact — which is exactly what the build round's b_x null turned "
     "out to be (leg axis_operand_deadness)"),
    ("near_ghost_restored_on_prefix", "curl", ("step_D",), None,
     (("float pb_x = vx ? pfx[ox] : 0.0f;",
       "float pb_x = vx ? pfx[ox] : (-pfx[ii]);"),), False,
     "the same near ghost on the PREFIXED operand — Dz's. MUST NOT be caught, and the "
     "REASON IS NOT THE MASK: the ghost value here is -pfx[0] = -0.0, and "
     "-0.0 - (+0.0) is -0.0, which the phi self-difference then turns back into +0.0 "
     "at the add. The prefix's own exact zero row absorbs the sign before the mask is "
     "ever consulted, which is what the twin below measures"),
    ("near_ghost_restored_on_prefix_WITH_mask_dropped", "curl", ("step_D",), None,
     (("float pb_x = vx ? pfx[ox] : 0.0f;",
       "float pb_x = vx ? pfx[ox] : (-pfx[ii]);"),
      ("    curl2 = at_x ? 0.0f : curl2;",
       "    // MUTANT: the r ownership mask on Dz dropped")), False,
     "MEASURED NULL, 0/8, and it is the twin that FAILED TO BITE — deliberately kept "
     "for that. Restoring the ghost AND dropping the mask still changes nothing on "
     "Dz, which is the direct evidence that the mask is not what hides this ghost. "
     "Contrast the Hz twin above, which is CAUGHT at 11-40 words on the same cases: "
     "the two rows side by side are what separate 'masked away' from 'already zero'"),
    ("prefix_row_zero_negated", "prefix", ("step_B", "step_D"), None,
     (("float acc = 0.0f;", "float acc = -0.0f;"),), None,
     "RECORDED ONLY, and it is the row that keeps the previous one honest. The module "
     "writes prefix row 0 as an exact +0.0 rather than assuming it, on the stated "
     "ground that -0.0 would be a different WORD and this family compares words. It "
     "would be — in the PREFIX — but the prefix is scratch that no comparison reads, "
     "and both consumers absorb the sign: the B side forms pfx[1] - (-0.0), the D "
     "side (+0.0) - (-0.0), and both are the same float32 word as before. So the "
     "spelling is right and its stated justification is not the reason. Carried at "
     "must_catch=None so the artifact reports the number rather than a verdict this "
     "gate has no business asserting either way"),

    # ---- THE m = 0 / |m| >= 1 SPLIT, AS STRUCTURAL ABSENCE ------------------
    ("m0_imr_coupling_injected", "curl", ("step_B",), None,
     (("    float curl0 = dtdx * ((c_y - c) + (b - b_z));",
       "    float curl0 = dtdx * ((c_y - c) + (b - b_z));\n"
       "    curl0 = curl0 - (dtdx * c) / (float(2 * i) + 1.0f);"),), True,
     "INJECT the |m| >= 1 i*m/r coupling into the m = 0 body at |m| = 1 magnitude "
     "(stepping.py:375-383 adds it only under grid.m != 0, over the 2*ir + shift "
     "denominator). At m = 0 the coefficient is EXACTLY ZERO and the term does not "
     "exist — so the only way to arm the absence is to put the term back and require "
     "it CAUGHT. That turns 'this family carries no m phase' from an omission a "
     "reader must trust into a measurement"),
    ("m0_near_axis_zeroing_injected", "curl", ("step_B",), None,
     (("    v0 = at_x ? 0.0f : v0;",
       "    v0 = at_x ? 0.0f : v0;\n"
       "    v1 = at_x ? 0.0f : v1;\n"
       "    n1 = at_x ? 0.0f : n1;"),), True,
     "INJECT the |m| >= 2 near-axis zeroing — six volumes held at zero over "
     "slice(0, |m|) — into the m = 0 body. _cylindrical_axis_rows is slice(0, |m|), "
     "which is EMPTY at m = 0, so the row count this family carries is ZERO rows and "
     "the B side's only axis rule is Bx. Same discipline as the row above: the "
     "absence is armed by putting the rule back"),
    ("m0_D_side_zeroes_the_auxiliary_too", "curl", ("step_D",), None,
     (("    v1 = at_x ? 0.0f : v1;",
       "    v1 = at_x ? 0.0f : v1;\n    n1 = at_x ? 0.0f : n1;"),), True,
     "stepping.py:587 zeroes the Dp FIELD on the axis row and NOT fu_Dp. The "
     "|m| >= 2 rule zeroes both, so the two arms differ in exactly this and a "
     "transcription that copied the wrong one is invisible unless the auxiliaries "
     "are compared"),
)

#: Host-side mutations: defects the KERNEL SOURCE cannot express because the value is
#: computed on the host and BOUND, and the kernel takes a pointer.
#: ``(label, kind, sub_step, scope, must_catch, why)``.
HOST_MUTATIONS: Tuple[Tuple[str, str, str, Optional[str], Optional[bool],
                            str], ...] = (
    ("prefix_ir0_swapped", "ir0", "step_B", None, True,
     "ir0 is 0.0 on the B side (Ep at the node) and 0.5 on the D side (Hp half a cell "
     "out) — the two sub-lattices, stepping.py:1297-1299. The scan runs either way "
     "and the radial profile is smoothly wrong"),
    ("prefix_wall_row_given_to_the_D_side", "wall", "step_D", None, False,
     "MEASURED NULL, 0/8, and it PINS THE ASYMMETRY rather than testing the kernel. "
     "Give the D side the B side's zero wall row and its scan really does grow — the "
     "harness ASSERTS the plan's scan_shape went from nr to nr + 1, so this is not a "
     "no-op agreeing with a no-op — yet not one compared word moves. The reason is "
     "the module's own: a BACKWARD difference reads rows i and i-1 and never looks "
     "past the top, so the extra row is computed and never read, and the shared head "
     "of the weight ladder is bit-identical because counts = arange(rows) + ir0 "
     "agrees on the overlap. The defect in the OTHER direction — the B side's wall "
     "row removed — is armed as prefix_wall_row_dropped and is CAUGHT. It is armed "
     "there as a SOURCE edit and not as this flag, deliberately: clearing the flag on "
     "the B side would shrink the prefix volume under a forward difference that "
     "still reads ii + nyz, and a gate must not rest on an out-of-bounds read"),
    ("prefix_rows_rounded_twice", "weights", "step_B", None, False,
     "MEASURED NULL, 0/8, AND IT CORRECTS A STATED JUSTIFICATION. The module builds "
     "the weight and divisor ladders in float64 and rounds to float32 exactly ONCE "
     "(stepping.py:1339-1341), on the ground that recomputing them in float32 would "
     "round twice on the divisor and put a silent half-ulp error on the whole radial "
     "ladder. It would not, at any grid this family can run: ir0 is 0.0 or 0.5, so "
     "the counts are integers and the divisor half-integers, both EXACT in float32. "
     "Leg prefix measures the crossing — the two ladders are bit-identical at every "
     "row count in this matrix and first diverge around 2^23 radial rows, four orders "
     "of magnitude past anything constructible. The float64 intermediate is still the "
     "right spelling and for a better reason: _cylindrical_rderiv_weights is the ONE "
     "definition, and calling it is what stops the two drifting"),
    ("scan_source_from_the_sibling_component", "source", "step_B", None, True,
     "the B side's scan reads Ep (Ey) specifically and the D side Hp (Hy). Repointed "
     "at a SIBLING SOURCE OF THE SAME SUB-STEP — Ex, which the plan already mirrors — "
     "so the defect is a silent wrong field rather than an unregistered-mirror crash, "
     "which would be measuring the harness"),
    ("axis_coef_rounded_after_the_multiply", "axis_coef", "step_D", None, False,
     "NULL CONTROL that PINS THE MODULE'S OWN CLAIM. The array path forms "
     "4.0 * (dt/dx) in float64 and NumPy rounds that ONE scalar to float32; the "
     "module says binding float32(4) * float32(dtdx) instead cannot differ because "
     "scaling by four is exact. Carried so that claim is measured rather than "
     "asserted — and if the coefficient ever stopped being a power of two this row "
     "flips to CAUGHT"),
)


def compile_prefix_mutant(source: str) -> Any:
    return kit.Counter(compile_source(source).cyl_rderiv_prefix)


def compile_curl_mutant(source: str) -> Any:
    return kit.Counter(compile_source(source).cyl_pml_curl_step)


def apply_edits(source: str, edits: Sequence[Tuple[str, str]]) -> str:
    for old, new in edits:
        source = needle(source, old, new)
    return source


def leg_mutations(payload: Dict[str, Any], out: str) -> None:
    harness = kit.MutationHarness(payload, out)

    # ---- source needles, per SUB-STEP ---------------------------------------
    for (label, kernel, sub_steps, scope, edits, must_catch, why) in \
            SOURCE_MUTATIONS:
        missed = False
        ran = caught = launches = 0
        skipped: List[str] = []
        for case, _ in CASES:
            fields, pml = build(case)
            if scope is not None and scope not in tags(fields, pml):
                skipped.append(case)
                continue
            for sub_step in sub_steps:
                names = curl_names(sub_step)
                before, after, moved = oracle(
                    fields, lambda: getattr(stepping, sub_step)(fields, pml), names)
                kit.assert_moved(moved, f"{label}/{case}/{sub_step} reference",
                                 floor=64)
                plan = build_curl_plan(fields, pml, sub_step)
                shipped = (cyl.cylindrical_prefix_source(sub_step)
                           if kernel == "prefix"
                           else cyl.cylindrical_curl_source(sub_step, plan.bc))
                try:
                    mutant_source = apply_edits(shipped, edits)
                except LookupError:
                    missed = True
                    restore(fields, before)
                    continue
                counter = (compile_prefix_mutant(mutant_source) if kernel == "prefix"
                           else compile_curl_mutant(mutant_source))
                run_curl(fields, pml, sub_step,
                         prefix_mutant=counter if kernel == "prefix" else None,
                         curl_mutant=counter if kernel == "curl" else None)
                launches += counter.launches
                ran += 1
                if sum(divergence(fields, after).values()):
                    caught += 1
                restore(fields, before)
        harness.record(label, harness.verdict(missed, ran, launches, caught),
                       launches, caught, ran, must_catch, why,
                       extra={"kind": "source", "kernel": kernel,
                              "sub_steps": list(sub_steps), "scope": scope,
                              "edits": len(edits), "skipped_cases": skipped})

    # ---- host mutations -----------------------------------------------------
    for (label, kind, sub_step, scope, must_catch, why) in HOST_MUTATIONS:
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
            kit.assert_moved(moved, f"{label}/{case} reference", floor=64)
            plan = apply_host_mutation(kind, fields, pml, sub_step)
            launches += plan.launches
            ran += 1
            if sum(divergence(fields, after).values()):
                caught += 1
            restore(fields, before)
        harness.record(label, harness.verdict(False, ran, launches, caught),
                       launches, caught, ran, must_catch, why,
                       extra={"kind": "host", "sub_step": sub_step, "scope": scope,
                              "skipped_cases": skipped})

    # ---- whole-step mutations: the classes a single launch CANNOT see -------
    leg_whole_step_mutations(harness)


def apply_host_mutation(kind: str, fields: Any, pml: Any, sub_step: str) -> Any:
    """Plant one HOST-side defect, launch through the shipped route, restore.

    Every patch is applied to the module attribute the plan builder reads by GLOBAL
    NAME and removed in a ``finally``, so a mutation that raised cannot leak into the
    next row. The plan is otherwise built exactly as the engine builds it.
    """
    other = "step_D" if sub_step == "step_B" else "step_B"
    saved: Dict[str, Any] = {}
    try:
        if kind == "ir0":
            saved["ir0"] = cyl.PREFIX_IR0[sub_step]
            cyl.PREFIX_IR0[sub_step] = cyl.PREFIX_IR0[other]
        elif kind == "wall":
            saved["wall"] = cyl.PREFIX_WALL_ROW[sub_step]
            cyl.PREFIX_WALL_ROW[sub_step] = cyl.PREFIX_WALL_ROW[other]
            # NON-VACUITY, ASSERTED RATHER THAN HOPED. This mutation's verdict is a
            # NULL, and a null is only worth recording if the patch actually changed
            # the object: the scan must really cover one more row than it shipped
            # with, or the row is measuring nothing.
            residency = Residency()
            plan = build_curl_plan(fields, pml, sub_step, residency)
            assert plan.scan_shape[0] == plan.shape[0] + 1, (
                "the wall-row flag was patched and the plan's scan did not grow; the "
                "null this mutation records would be vacuous",
                plan.scan_shape, plan.shape)
            residency.sync_in()
            plan.run()
            residency.sync_out()
            return plan
        elif kind == "weights":
            original = cyl.prefix_row_vectors
            saved["fn"] = original

            def twice_rounded(step: str, radial_rows: int, dtype: Any = None):
                # The float64 intermediate removed: counts, weights and the divisor
                # all formed and stored in float32, so the divisor rounds twice.
                rows = cyl.scan_rows(step, radial_rows)
                counts = (np.arange(rows, dtype=np.float32)
                          + np.float32(cyl.PREFIX_IR0[step]))
                return (np.ascontiguousarray(counts).reshape(-1),
                        np.ascontiguousarray(counts[1:] - np.float32(0.5)
                                             ).reshape(-1))

            cyl.prefix_row_vectors = twice_rounded
        elif kind == "source":
            spec = cyl.SUB_STEPS[sub_step]
            sibling = spec["sources"][0]
            residency = Residency()
            plan = build_curl_plan(fields, pml, sub_step, residency)
            swapped = residency.mirror(sibling, getattr(fields, sibling))
            plan._prefix_args = ((plan._prefix_args[0], swapped)
                                 + tuple(plan._prefix_args[2:]))
            residency.sync_in()
            plan.run()
            residency.sync_out()
            return plan
        elif kind == "axis_coef":
            residency = Residency()
            plan = build_curl_plan(fields, pml, sub_step, residency)
            rounded = float(np.float32(4.0) * np.float32(plan.dtdx))
            plan.axis_coef = rounded
            plan._curl_args = tuple(plan._curl_args[:-1]) + (rounded,)
            residency.sync_in()
            plan.run()
            residency.sync_out()
            return plan
        else:  # pragma: no cover - a typo in the table is a hard failure
            raise ValueError(f"unknown host mutation {kind!r}")
        return run_curl(fields, pml, sub_step)
    finally:
        if "ir0" in saved:
            cyl.PREFIX_IR0[sub_step] = saved["ir0"]
        if "wall" in saved:
            cyl.PREFIX_WALL_ROW[sub_step] = saved["wall"]
        if "fn" in saved:
            cyl.prefix_row_vectors = saved["fn"]


def curl_only_probe(case: str, sub_step: str, launches: int) -> int:
    """Differing words after N launches of ONE sub-step, array path against Metal.

    The instrument that turns "a single-launch gate cannot see this" into a number. A
    nonzero result would mean the defect is NOT a whole-step-only class and the leg's
    claim about it is wrong.
    """
    fields_a, pml_a = build(case)
    fields_b, pml_b = build(case)
    for _ in range(launches):
        getattr(stepping, sub_step)(fields_a, pml_a)
    run_curl(fields_b, pml_b, sub_step, launches=launches)
    return sum(differing(getattr(fields_a, n), getattr(fields_b, n))
               for n in curl_names(sub_step))


#: The whole-step mutation budget. Shorter than :data:`CYCLES` because these arms are
#: expected to diverge at step 0 or 1 and the budget only has to outlast the class:
#: the first divergent step is RECORDED per case, which is the evidence that the class
#: is a whole-step class rather than a claim about it.
MUTATION_CYCLES = 4


def leg_whole_step_mutations(harness: Any, budget: int = MUTATION_CYCLES) -> None:
    """The classes that exist ONLY at complete-step granularity.

    Every kernel launched here is the SHIPPED one and every one of them is
    byte-perfect in isolation, which is exactly the point: all that moves is WHEN a
    host pass runs, whether its bracket exists, or whether the two launches of one
    curl plan happen in the order the plan owes.
    """
    def sweep(label: str, must_catch: Optional[bool], why: str,
              scope: Optional[str] = None, extra: Optional[Dict[str, Any]] = None,
              patch: Optional[Callable[[Any], Callable[[], None]]] = None,
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
            plan, residency, _live, _synced = compose(fields, pml)
            undo = patch(plan) if patch is not None else None
            try:
                residency.sync_in()
                first: Optional[int] = None
                for step in range(budget):
                    reference_step(reference_fields, reference_pml)
                    walk(fields, pml, plan, residency, **walk_kwargs)
                    residency.sync_out()
                    total = sum(differing(getattr(fields, n),
                                          getattr(reference_fields, n))
                                for n in STATE if getattr(fields, n, None) is not None)
                    if total and first is None:
                        first = step
            finally:
                if undo is not None:
                    undo()
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

    # 1. THE STALE PREFIX, and this family is where the class stops being abstract.
    #    The curl reads the prefix the scan wrote; hoisting the scan out of the loop
    #    is invisible for one launch and wrong forever after.
    def hoist_the_scan(plan: Any) -> Callable[[], None]:
        """Run each curl slot's scan ONCE up front, then only its curl per cycle.

        THE PLAN IS PROXIED RATHER THAN PATCHED, because ``CylindricalRealCurlPlan``
        carries ``__slots__`` and its ``run`` cannot be rebound on the instance —
        which is a good property of the shipped class and not an obstacle worth
        working around destructively. The proxy forwards ``launches`` so the
        harness's DISARMED check still reads the real launch count.
        """
        class CurlOnly:
            __slots__ = ("product",)

            def __init__(self, product: Any) -> None:
                self.product = product

            def run(self, contract: Optional[str] = None) -> None:
                self.product.run_curl(contract)

            @property
            def launches(self) -> int:
                return self.product.launches

        saved = {}
        for slot in ("step_B", "step_D"):
            product = plan.plans[slot]
            saved[slot] = product
            product.run_prefix()                      # scan ONCE, up front
            plan.plans[slot] = CurlOnly(product)

        def undo() -> None:
            for slot, original in saved.items():
                plan.plans[slot] = original

        return undo

    blindness = {case: {f"{sub_step}_x{n}": curl_only_probe(case, sub_step, n)
                        for sub_step in ("step_B", "step_D") for n in (1, 6)}
                 for case, _ in CASES[:3]}
    sub_step_total = sum(sum(row.values()) for row in blindness.values())
    assert sub_step_total == 0, (
        "the sub-step route already differs from the array path; nothing below "
        "means anything", blindness)
    sweep("prefix_scanned_ONCE_outside_the_loop", True,
          "the curl READS the scan's output and the scan is a function of THIS "
          "sub-step's source volume, so the two launches are ORDERED and the scan owes "
          "one launch per curl. Hoisted out of the loop the curl is fed the PREVIOUS "
          "timestep's radial derivative: a smooth, plausible, entirely wrong field "
          "rather than an error. A SINGLE-LAUNCH GATE CANNOT SEE THIS AT ALL — with "
          "one launch the hoisted scan IS the right scan — which is why it is armed "
          "here and not above",
          patch=hoist_the_scan,
          extra={"sub_step_route_blind_differing": blindness,
                 "sub_step_route_total_differing": sub_step_total})

    # 2. THE SEAM, three ways. Scoped to WALLED cases: on a periodic z there is no
    #    wall pass at all, so these are structural nulls there rather than misses.
    sweep("wall_pass_dropped", True,
          "zero_metal_B holds the B samples that lie ON a metallic wall at exactly "
          "zero (MEEP step_boundaries(B_stuff)). It runs on the HOST between the curl "
          "and the constitutive pass and no Metal product carries it",
          scope="walled", skip=("zero_metal_B",))
    sweep("wall_pass_run_WITHOUT_its_sync_bracket", True,
          "the engine holds NumPy and the device mirror is authoritative between "
          "launches, so an array-path pass must be bracketed with sync_out and "
          "sync_in. Without the bracket the pass writes a HOST array nothing reads "
          "and the device keeps the un-cleared wall — the stale-mirror class, at the "
          "seam",
          scope="walled", unsynced=("zero_metal_B",))

    moved_order = tuple(
        [entry for entry in PASSES if entry[0] != "zero_metal_B"][:4]
        + [entry for entry in PASSES if entry[0] == "zero_metal_B"]
        + [entry for entry in PASSES if entry[0] != "zero_metal_B"][4:])
    sweep("wall_pass_moved_AFTER_the_constitutive_pass", True,
          "MEEP's order is step_db -> step_source -> step_boundaries, so the wipe runs "
          "BEFORE update_H reads B (driver.py:3285-3287). Moving it one pass later "
          "feeds the constitutive read an un-cleared wall; every kernel here is the "
          "shipped one and only the ORDER moves",
          scope="walled", order=moved_order)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

LEGS: Tuple[Tuple[str, Callable[[Dict[str, Any], str], None]], ...] = (
    ("execution", leg_execution),
    ("prefix", leg_prefix),
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
        "gate": "metal_cylindrical_real",
        "family": cyl.FAMILY,
        "slots": list(ARITHMETIC_SLOTS),
        "environment": kit.environment_stamp(),
        "subnormal_policy": subnormal.mps_policy_report(),
        "legs": {},
    }
    kit.save(payload, out)

    reasons: List[str] = []
    if not payload["environment"].get("mps_available"):
        reasons.append("no MPS device available")
    reasons.extend(subnormal.mps_policy_reasons())
    if reasons:
        return kit.cannot_certify(payload, out, reasons)

    ran = kit.run_legs(LEGS, payload, out, kit.wanted_legs(args.legs))

    compared = 0
    certified = True
    for key in ("prefix", "curl", "constitutive"):
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
    nulls = [row for row in mutations if row.get("must_catch") is False]

    return kit.summarize(
        payload, out,
        claim=("metal_kernels.cylindrical_real reproduces stepping.py word for word "
               "on a real float32 Dcyl grid at m = 0 — the radial scan and the curl "
               "on both sub-steps and the certified real constitutive pair under this "
               "family's restated predicate — per sub-step AND per complete "
               "ten-pass driver step, against the in-process NumPy oracle"),
        scope=("the case matrix in CASES: four radial extents, both z terminations, "
               "power-of-two and non-power-of-two Courant numbers, planted signed "
               "zeros and the composition sweep's own dcyl_m0_real grid. |m| >= 1, "
               "complex storage, a Cartesian grid, conductivity, dispersion, "
               "chi2/chi3, BFAST, grid.beta, a Bloch phase, a mirror plane and an "
               "inactive absorber are OUT OF SCOPE and refused BY NAME by the "
               "predicate (leg split)"),
        stated_weakness=(
            "BEHAVIOURAL ONLY: torch.mps.compile_shader exposes no AIR, no GPU ISA "
            "and no optimisation report, so this gate catches a wrong ANSWER and "
            "never a wrong INSTRUCTION, and it cannot establish that the contraction "
            "guard was obeyed. It makes NO throughput claim — the scan's dispatch is "
            "sized from its output volume and the curl pays a second launch — it does "
            "not re-cut the 759-slot coverage number, and the Metal-vs-Triton "
            "admission diff is REFUSED on this host rather than estimated (leg "
            "containment). At m = 0 the i*m/r phase and the near-axis zero-row count "
            "DO NOT EXIST, so they are armed as INJECTIONS into the m = 0 body rather "
            "than as edits to terms this kernel carries. All arithmetic claims ride a "
            "CHECKED subnormal-free precondition under the FLUSH policy, the only one "
            "this executor can honour."),
        started=started, legs_run=ran, compared=compared, certified=certified,
        extra={"cases_refused_on_precondition": refused,
               "mutations_armed": len(armed),
               "mutations_caught_of_armed": len(caught),
               "mutations_null_controls": len(nulls),
               "mutation_verdicts": {row["mutation"]: row["verdict"]
                                     for row in mutations},
               "subnormal_windows": {
                   row["case"]: [row["subnormal_window"]["first_step"],
                                 row["subnormal_window"]["last_step"]]
                   for row in payload["legs"].get("whole_step", ())}})


if __name__ == "__main__":
    from metal_gate_runner import run_current_measurement

    raise SystemExit(run_current_measurement(__file__, sys.argv[1:]))
