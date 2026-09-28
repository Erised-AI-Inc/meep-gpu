"""BYTE GATE — the mirror FOLD composed with the ``grid.beta`` term, on Metal.

WHAT THIS CERTIFIES: that ``metal_kernels.folded_beta``'s four products reproduce
``stepping.py`` WORD FOR WORD on a folded grid at nonzero ``grid.beta`` — the real
and complex curls on both sub-steps, and the two certified constitutive bodies
re-admitted over a folded stored extent — per SUB-STEP and per COMPLETE DRIVER STEP.

WHAT MAKES THIS FAMILY DIFFERENT FROM ITS TWO PARENTS, and it is the only reason it
exists: THE FOLD WIDENS THE MASK THE BETA TERM LANDS INSIDE. Unfolded, the beta
increment meets ONE ownership mask (cell 0 on a non-periodic axis); folded, it meets
TWO, because a folded PERIODIC axis also masks its LAST plane for every target whose
Yee shift is 1 there. The array path adds the term at stepping.py:384-391 / :467-474
and masks at :397 / :479, whose ``_mask_non_owned_cells`` carries the cell-0 arm at
:1898-1902 and the top-plane arm at :1887-1897. Leg ``position`` measures that the
term sits between the curl and BOTH masks, on the DEVICE, by moving it and requiring
a divergence — a claim about the code rather than about a comment.

THIS GATE WALKS THE DRIVER'S REAL TEN-PASS LIST, which is what separates it from the
family's sibling tests. One complete folded step is::

    step_B -> fill_symmetry_bc_B -> zero_metal_B -> fill_folded_far_ghosts_B ->
    update_H -> step_D -> fill_symmetry_bc_D -> zero_metal_D ->
    fill_folded_far_ghosts_D -> update_E                    (driver.py:3282-3302)

THE WALL PASS SITS BETWEEN THE TWO FILL PASSES and it is the pass NO Metal family
carries, so on a walled case it runs on the ARRAY PATH between two device launches.
Leg ``whole_step`` therefore brackets every array-path pass with an explicit
``sync_out`` / ``sync_in`` over exactly the volumes the package's own residency model
says that pass touches (``metal_kernels.coverage.SUB_STEP_VOLUMES``), and asserts
``Residency.verify()`` is empty after every step. THREE failure classes live only
here — a STALE MIRROR (a host pass writes the host array under a live device
mirror), a SEAM (the wall clear lands between the curl and the constitutive read),
and an ACCUMULATING AUXILIARY (``fu_*`` and ``f_w_*`` are STATE, so a kernel right
for one launch and wrong forever after is identical in a single-launch leg) — and
the fold adds a fourth, its ghost plane being written by one pass and read by the
next. Each has its own armed mutation.

WHAT THIS GATE MEASURED AND THE SIBLING TESTS COULD NOT SEE. The shared builder
``metal_composition_matrix._fill`` writes a REAL uniform array into every volume,
so a complex64 volume comes back with an IDENTICALLY ZERO IMAGINARY PART. Every
complex product in the kernel then has an exact-zero operand, and three separate
measurements collapse to vacuous nulls: NAIVE and FMA_V1 agree word for word, the
contraction guard's control cannot fire, and the family's own complex arm is
certified on data that is not complex. :func:`enrich_complex` seeds an independent
imaginary part and asserts a floor, and leg ``guard``'s control is what proves the
difference — measured on this host at ``complex_bloch_x``: with the zero imaginary
part the NAIVE control moves 0 words, with it seeded it moves 4-11 per sub-step.

A MEASURED NULL WITHOUT ITS MECHANISM IS WORTHLESS, so leg ``spellings`` supplies the
mechanism for the two refuted spellings this family cannot tell apart. HALF ONE
measures that the spellings genuinely differ on this backend (the real negation on
exactly the one discriminating lane; the complex cross-term fold on 80 words). HALF
TWO measures that the pattern is unreachable HERE: both differences reduce to a signed
zero in the beta increment, and ``curl - (+/-0.0)`` is one word unless ``curl`` is a
negative zero. A needle exports the pre-beta ``curl0/curl1/curl2`` off the device and
censuses them — 0 negative zeros on the two curls the beta term is added to, over
46,272 words on 28 case/sub-step pairs with signed zeros PLANTED into the sources.
``curl2``, which the term never touches and the structural argument does not cover,
DOES reach ``-0.0`` (60 words), and that is the census's own non-vacuity control.

THE SUBNORMAL PRECONDITION IS CHECKED, NOT ASSUMED, AND IS REPORTED AS A WINDOW.
On MPS the float32 subnormal flush is native and has no lever, so ``keep`` is NOT
OFFERABLE and the honest third policy value is REFUSE. Band entry is a RUN-and-
WINDOW fact rather than a family fact, so leg ``whole_step`` censuses EVERY step and
records ``[first_step, last_step]`` per case, REFUSING any case that enters the band
BY NAME — a coverage refusal, not a failure, whose comparisons are withheld from the
certified total. Leg ``precondition`` drives the same census over a state scaled into
the band and REQUIRES it to fire, so the empty windows next door are a measurement
rather than a silence.

THE 1e29 VOLUME CLAUSE IS NOT THIS FAMILY'S. That plan-time check belongs to
``nonlinear_chi3``; this family REFUSES a nonlinearity by name (leg ``refusals``
pins it) and carries no Pade tree, so there is nothing here for it to bound.

THE STATED WEAKNESS, and it is this certification's one gap against the Triton
twin's: ``torch.mps.compile_shader`` exposes NO disassembly, no AIR and no
optimisation report (``device.py:46-62``). This gate cannot refuse a compile whose
emitted code violates the contraction policy and cannot establish that the pragma
was obeyed. EVERY LEG HERE IS BEHAVIOURAL: it catches a wrong answer, never a wrong
instruction.

    python -u gate_metal_folded_beta.py --out results/.../gate.json
"""

from __future__ import annotations

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

# THE POLICY IS SET BEFORE ANY `meep_gpu` MODULE IS REACHED. On MPS the float32
# subnormal flush is native and has no lever, so the resolved default `keep` is NOT
# OFFERABLE and every Metal predicate refuses by name. Setting `flush` is what puts
# the claim under the CHECKED precondition rather than under a pretence.
os.environ.setdefault("MEEP_GPU_SUBNORMAL_POLICY", "flush")

import metal_composition_matrix as matrix  # noqa: E402

ENVIRONMENT = matrix.prepare_environment()

from meep_gpu import stepping  # noqa: E402
from meep_gpu.metal_kernels import (  # noqa: E402
    arms, complex_fields, coverage as metal_coverage, device, folded_beta,
    folded_complex, launch, shaders, special_kz, subnormal, symmetry, templates,
)
from meep_gpu.metal_kernels.device import compile_source  # noqa: E402
from meep_gpu.metal_kernels.launch import SUB_STEPS  # noqa: E402
from meep_gpu.triton_kernels.coverage import (  # noqa: E402
    CONSTITUTIVE_SIDES, _boundary_kinds, zero_metal_axes,
)
from meep_gpu.triton_kernels import folded_complex as triton_folded_complex  # noqa: E402

import metal_gate_kit as kit  # noqa: E402

log, save, differing, words = kit.log, kit.save, kit.differing, kit.words
needle = kit.needle

MP = symmetry.CODE_MIRROR_PERIODIC
MM = symmetry.CODE_MIRROR_METALLIC

#: Everything the engine stores that a folded beta step can touch. The whole-step
#: leg compares ALL of it after every complete step: ``fu_*`` and ``f_w_*`` are STATE
#: and a kernel right for one launch and wrong forever after diverges only once they
#: accumulate.
STORED: Tuple[str, ...] = (
    "Bx", "By", "Bz", "Dx", "Dy", "Dz", "Ex", "Ey", "Ez", "Hx", "Hy", "Hz",
    "fu_Bx", "fu_By", "fu_Bz", "fu_Dx", "fu_Dy", "fu_Dz",
    "f_w_Hx", "f_w_Hy", "f_w_Hz", "f_w_Ex", "f_w_Ey", "f_w_Ez")

#: The volumes a curl plan is built from — the from-arrays route's whole input.
CURL_ARRAYS: Tuple[str, ...] = STORED[:18]

#: THE DRIVER'S OWN TEN PASSES, in its order (driver.py:3282-3287 magnetic,
#: :3293-3302 electric). Spelled once, so the reference walk and the device walk
#: cannot drift apart, and CHECKED against the package's own residency model below —
#: if the two ever disagreed, this gate would be walking an order the composer does
#: not believe in.
DRIVER_ORDER: Tuple[str, ...] = (
    "step_B", "fill_B_near", "zero_metal_B", "fill_B_far", "update_H",
    "step_D", "fill_D_near", "zero_metal_D", "fill_D_far", "update_E")

#: Which entry of ``metal_coverage.SUB_STEP_VOLUMES`` each pass above IS. The two
#: fill passes are separate slots here and one entry there, because the residency
#: model names the near seam ``fill_B`` and the far one by its stepping function.
RESIDENCY_NAME: Dict[str, str] = {
    "step_B": "step_B", "fill_B_near": "fill_B", "zero_metal_B": "zero_metal_B",
    "fill_B_far": "fill_folded_far_ghosts_B", "update_H": "update_H",
    "step_D": "step_D", "fill_D_near": "fill_D", "zero_metal_D": "zero_metal_D",
    "fill_D_far": "fill_folded_far_ghosts_D", "update_E": "update_E"}

#: The array-path function each pass runs when no device plan owns it.
HOST_PASS: Dict[str, Callable[[Any], None]] = {
    "fill_B_near": stepping.fill_symmetry_bc_B,
    "zero_metal_B": stepping.zero_metal_B,
    "fill_B_far": stepping.fill_folded_far_ghosts_B,
    "fill_D_near": stepping.fill_symmetry_bc_D,
    "zero_metal_D": stepping.zero_metal_D,
    "fill_D_far": stepping.fill_folded_far_ghosts_D}

#: THE CASE MATRIX. Every axis is here because a mutation is reachable on one value
#: and a MEASURED NULL on the other:
#:
#:   TERMINATION    MIRROR_PERIODIC masks its top plane and runs the far ghost pass;
#:                  MIRROR_METALLIC does NEITHER — the top plane is OWNED and
#:                  STEPPED, and on this family that plane CARRIES the beta term, so
#:                  a sweep on one termination proves nothing about the other;
#:   FULL COUNT     ``reflect_row`` is ``stored - 2`` at an even full count and
#:                  ``stored - 3`` at an odd one, so at the default extent a wrong
#:                  ``n - 2`` formula HAPPENS TO BE RIGHT;
#:   PARITY         an odd mirror plane negates and an even one does not; the parity
#:                  is the parents' fill coefficient and reaches this family through
#:                  the ghost plane its curl then reads;
#:   FOLD AXIS      Y is what all four corpus rows are; X proves the axis selects the
#:                  plane decomposition, the mask and the launch width together;
#:   TWO AXES       a corner unowned on two planes at MIXED parity;
#:   BETA SIGN      the corpus carries both, and the two coefficients are ``+/-``
#:                  halves of one pair, so a sign swap is only visible when they
#:                  differ;
#:   A LIVE WALL    ``boundaries={"x": "metallic"}`` makes ``zero_metal_*`` do real
#:                  work — measured 12-13 words per pass — which is the ONLY way the
#:                  seam and stale-mirror classes exist at all. On such a grid BOTH
#:                  parents' fill predicates REFUSE by name (the far pass reads a
#:                  plane the wall clear touches), so the two fills fall to the array
#:                  path and the step becomes a genuine mixed host/device walk;
#:   STORAGE        the two arms invert on it;
#:   BLOCH          a phase on the UNFOLDED axis, which two of the four corpus rows
#:                  carry, at a GENERAL value and at the zone edge. The phase word is
#:                  ``exp(2i.pi.k.L)`` with L the cell extent, so k = 0.5 / CROSS is
#:                  the value that lands EXACTLY on (-1.0, +0.0) — measured, not
#:                  assumed, and the two rows are what separate "a phase is live" from
#:                  "a GENERAL complex product is live". Leg ``guard``'s control turns
#:                  on precisely that distinction.
CROSS = 4.8

CASES: Tuple[Tuple[str, Dict[str, Any], bool], ...] = (
    ("real_y_periodic_even", dict(beta=0.3), False),
    ("real_y_periodic_odd", dict(beta=0.3, extent=2.1), False),
    ("real_y_metallic", dict(beta=0.3, boundaries={"y": "metallic"}), False),
    ("real_y_odd_parity", dict(beta=-0.41, phase=-1), False),
    ("real_x_fold", dict(beta=0.3, axis="X"), False),
    ("real_walled", dict(beta=0.3, boundaries={"x": "metallic"}), False),
    ("real_two_axis_mixed", dict(beta=0.3, axis="XY", phase=(1, -1)), False),
    ("complex_y_periodic_even", dict(complex_storage=True, beta=0.3), True),
    ("complex_bloch_x", dict(complex_storage=True, beta=0.3,
                             k_point=(0.3, 0.0, 0.0)), True),
    ("complex_zone_edge_x", dict(complex_storage=True, beta=0.3,
                                 k_point=(0.5 / CROSS, 0.0, 0.0)), True),
    ("complex_y_metallic", dict(complex_storage=True, beta=-0.41,
                                boundaries={"y": "metallic"}), True),
    # A SECOND general-phase row, on the OTHER termination and the other beta sign.
    # One row would leave every general-phase-scoped needle resting on a single grid.
    ("complex_bloch_metallic_fold", dict(complex_storage=True, beta=-0.41,
                                         k_point=(0.3, 0.0, 0.0),
                                         boundaries={"y": "metallic"}), True),
    ("complex_y_periodic_odd", dict(complex_storage=True, beta=0.3, extent=2.1),
     True),
    ("complex_walled", dict(complex_storage=True, beta=0.3,
                            boundaries={"x": "metallic"}), True),
)

#: ``CROSS`` (declared above the matrix, which reads it) is the transverse extent
#: every case is built at. The default matrix shape is a 16 x 12 plane; a claim is
#: only as strong as the words it moved, so this widens the unfolded axis to 48 cells
#: and changes no verdict — every predicate here reads kinds, flags and dtypes, never
#: an extent.
CASE_KWARGS: Dict[str, Dict[str, Any]] = {label: kwargs for label, kwargs, _ in CASES}
CASE_COMPLEX: Dict[str, bool] = {label: is_complex for label, _, is_complex in CASES}

#: This family's arm is bound from ``special_kz``'s artifact DELIBERATELY: the
#: complex arithmetic here is that family's exactly (the same ``c_mul`` with the same
#: purely-imaginary coefficient on the LEFT), so the record that licenses that family
#: licenses this one. The fold contributes no complex multiply of its own — its only
#: addition assigns a literal zero.
PROBE = special_kz.load_expansion_probe()
EXPANSION = special_kz.beta_expansion_from_probe(PROBE) if PROBE else None

#: THE PARENTS' FILL PROBE IS A DIFFERENT ARTIFACT and the distinction is load
#: bearing: ``folded_complex``'s FILL multiplies by the mirror parity as a complex
#: coefficient on the LEFT, a pattern ``special_kz``'s artifact never classified.
#: This family registers no fill, but the whole-step legs run the parents' fills.
FOLD_PROBE = folded_complex.load_expansion_probe()
FOLD_EXPANSION = (folded_complex.expansion_from_probe(FOLD_PROBE)
                  if FOLD_PROBE else None)


# ---------------------------------------------------------------------------
# Building, seeding and describing a case
# ---------------------------------------------------------------------------

def enrich_complex(fields: Any, seed: int = 20260816) -> int:
    """Seed an INDEPENDENT imaginary part into every complex64 volume.

    NOT COSMETIC, AND THE MEASUREMENT IS IN THE MODULE DOCSTRING.
    ``matrix._fill`` assigns a REAL array into a complex64 volume, so the imaginary
    plane comes back identically zero. Every complex product in this family's kernel
    then carries an exact-zero operand, and the arm choice, the contraction guard and
    the complex curl itself all become unmeasurable: measured on ``complex_bloch_x``,
    the NAIVE arm — which is the WRONG arm on this host — reproduced the array path
    word for word until this seeding existed.

    Returns the number of cells given a nonzero imaginary part, which every caller
    asserts a floor on. A census of zero here would mean the enrichment silently did
    nothing, which is the state this function exists to leave behind.
    """
    rng = np.random.default_rng(seed)
    planted = 0
    for name in STORED:
        array = getattr(fields, name, None)
        if array is None or array.dtype != np.complex64:
            continue
        array.imag = rng.uniform(-0.35, 0.35, array.shape).astype(np.float32)
        planted += int(np.count_nonzero(array.imag))
    return planted


def build(label: str, **overrides: Any) -> Tuple[Any, Any]:
    """One case, seeded so that nothing it certifies is certified on zeros."""
    kwargs = dict(CASE_KWARGS[label])
    kwargs.update(overrides)
    fields, pml = matrix.folded(cross=CROSS, **kwargs)
    if CASE_COMPLEX[label]:
        planted = enrich_complex(fields)
        kit.assert_census_floor(
            planted, f"{label} imaginary seeding", floor=64)
    return fields, pml


def tags(fields: Any, pml: Any) -> Tuple[str, ...]:
    """What a case IS, derived from the ENGINE rather than typed beside the row.

    A mutation scopes itself with these. Typing them into the matrix would put the
    fold's single point of failure — the MIRROR_METALLIC / MIRROR_PERIODIC split —
    in a second place, and a scoping that disagreed with the grid would silently turn
    a reachable defect into a NEEDLE-MISSED row.
    """
    grid = fields.grid
    codes, _ = symmetry.folded_axis_kinds(grid, pml)
    codes = tuple(int(code) for code in (codes or ()))
    found: List[str] = []
    found.append("mirror_periodic" if MP in codes else "no_mirror_periodic")
    found.append("mirror_metallic" if MM in codes else "no_mirror_metallic")
    folded_axes = [i for i, c in enumerate(codes) if c in symmetry.MIRROR_CODES]
    found.append("two_axis" if len(folded_axes) >= 2 else "one_axis")
    if len({int(grid.mirror_phase(a)) for a in folded_axes}) > 1:
        found.append("mixed_phase")
    if any(int(grid.mirror_phase(a)) < 0 for a in folded_axes):
        found.append("odd_parity")
    found.append("complex" if getattr(fields, "force_complex_fields", False)
                 or bool(getattr(grid, "has_bloch", False)) else "real")
    if any(zero_metal_axes(grid)):
        found.append("walled")
    if getattr(grid, "has_bloch", False):
        found.append("bloch")
        kinds = _boundary_kinds(grid, pml)
        phased, phase_words = special_kz.bloch_phase_words(grid, kinds, False)
        # A GENERAL complex product exists only where a live phase has BOTH words
        # nonzero. At the zone edge the phase is exactly -1+0j and every cross term
        # is exact, which is why the arm and contraction needles are null there and
        # the matrix carries the edge deliberately.
        if any(flag and phase_words[2 * axis] != 0.0
               and phase_words[2 * axis + 1] != 0.0
               for axis, flag in enumerate(phased)):
            found.append("general_phase")
    if float(getattr(grid, "beta", 0.0)) < 0.0:
        found.append("negative_beta")
    return tuple(found)


def snapshot(fields: Any, names: Sequence[str] = STORED) -> Dict[str, Any]:
    return {name: np.array(getattr(fields, name), copy=True) for name in names
            if getattr(fields, name, None) is not None}


def restore(fields: Any, state: Dict[str, Any]) -> None:
    for name, array in state.items():
        getattr(fields, name)[...] = array


def state_census(fields: Any) -> int:
    """How many stored words sit in the subnormal band, over EVERY stored volume.

    Read off the BITS by :func:`subnormal.census` — a value comparison would be
    answered by the very flushing this counts, and a complex64 volume is two words
    per cell with BOTH counted.
    """
    return sum(subnormal.census(getattr(fields, name))
               for name in STORED if getattr(fields, name, None) is not None)


def curl_names(sub_step: str) -> Tuple[str, ...]:
    spec = SUB_STEPS[sub_step]
    return tuple(spec["targets"]) + tuple("fu_" + n for n in spec["targets"])


def constitutive_names(side: str) -> Tuple[str, ...]:
    spec = CONSTITUTIVE_SIDES[side]
    return tuple(spec["targets"]) + tuple(spec["aux"])


def compared_words(state: Dict[str, Any]) -> int:
    return sum(int(words(array).size) for array in state.values())


def divergence(fields: Any, after: Dict[str, Any]) -> Dict[str, int]:
    return {name: differing(getattr(fields, name), array)
            for name, array in after.items()}


def oracle(fields: Any, apply: Callable[[], None],
           names: Sequence[str]) -> Tuple[Dict[str, Any], Dict[str, Any], int]:
    """Run the ARRAY PATH, capture what it wrote, and put the state back.

    Returns (before, after, moved). ``moved`` is the vacuity floor every leg
    asserts: a no-op agreeing with a no-op is trivially identical.
    """
    before = snapshot(fields, names)
    apply()
    after = snapshot(fields, names)
    restore(fields, before)
    moved = sum(differing(before[n], after[n]) for n in names)
    return before, after, moved


# ---------------------------------------------------------------------------
# Device runners — ONE route each, so the bytes the gate certifies are the engine's
# ---------------------------------------------------------------------------

def pml_vectors(pml: Any, sub_step: str) -> Dict[str, Any]:
    """The coefficient vectors for ONE sub-step, at ITS OWN sub-lattice.

    ``step_B`` writes the HALF-INTEGER lattice and takes the ``_h`` suffix;
    ``step_D`` takes the integer one. Binding the wrong pair is a whole-volume
    divergence that looks like a broken kernel and is a broken HARNESS, so the
    suffix is derived from ``SUB_STEPS`` rather than spelled per call site.
    """
    suffix = SUB_STEPS[sub_step]["suffix"]
    return {f"{stem}_{axis}": getattr(pml, f"{stem}_{axis}{suffix}")
            for axis in "xyz" for stem in ("kms", "sinv")}


def engine_curl(fields: Any, pml: Any, sub_step: str, residency: Any) -> Any:
    """The ENGINE route: the plan the composer would build, or an assertion."""
    if getattr(fields, "force_complex_fields", False) or getattr(
            fields.grid, "has_bloch", False):
        plan = folded_beta.plan_folded_beta_bloch_pml_curl(
            fields, pml, sub_step, residency, probe=PROBE)
        if plan is None:
            raise AssertionError(
                folded_beta.folded_beta_composition_bloch_curl_coverage(
                    fields, pml, sub_step, residency, PROBE).reasons)
        return plan
    plan = folded_beta.plan_folded_beta_pml_curl(fields, pml, sub_step, residency)
    if plan is None:
        raise AssertionError(folded_beta.folded_beta_composition_curl_coverage(
            fields, pml, sub_step, residency).reasons)
    return plan


def engine_constitutive(fields: Any, pml: Any, side: str, residency: Any) -> Any:
    if getattr(fields, "force_complex_fields", False) or getattr(
            fields.grid, "has_bloch", False):
        plan = folded_beta.plan_folded_beta_complex_constitutive(
            fields, pml, side, residency, probe=PROBE)
        if plan is None:
            raise AssertionError(
                folded_beta.folded_beta_complex_constitutive_coverage(
                    fields, pml, side, residency, PROBE).reasons)
        return plan
    plan = folded_beta.plan_folded_beta_constitutive(fields, pml, side, residency)
    if plan is None:
        raise AssertionError(folded_beta.folded_beta_constitutive_coverage(
            fields, pml, side, residency).reasons)
    return plan


def engine_fill(fields: Any, family: str, residency: Any) -> Any:
    """The PARENTS' fill plan, or None where their predicates refuse.

    THIS FAMILY REGISTERS NO FILL ARM, deliberately: both parents' fill predicates
    carry no beta clause and already admit a folded beta run, and a third arm would
    make both seam slots AMBIGUOUS and drop them to the array path. On a WALLED grid
    they refuse by name and the pass belongs to the array path — which is the case
    that makes the whole-step walk a mixed host/device one.
    """
    if getattr(fields, "force_complex_fields", False) or getattr(
            fields.grid, "has_bloch", False):
        return folded_complex.plan_folded_complex_fill(
            fields, family, "fill_" + family, residency, probe=FOLD_PROBE)
    return symmetry.plan_mirror_ghost_fill(fields, family, "fill_" + family,
                                           residency)


def from_arrays_curl(fields: Any, pml: Any, sub_step: str, codes: Any,
                     residency: Any, *, functions: Optional[Dict[str, Any]] = None,
                     has_beta: bool = True,
                     beta_override: Any = None) -> Any:
    """THE MUTATION SEAM: bare arrays, explicit codes, an overridable function.

    Dropping ``functions`` is not a slowdown, it is a silent DISARMING — every
    mutation leg would then launch the shipped kernel and report its defect as
    uncaught. ``beta_override`` is the HOST-side seam, for the mutations whose defect
    is in the coefficient WORDS rather than in the source text.
    """
    grid = fields.grid
    arrays = {name: getattr(fields, name) for name in CURL_ARRAYS
              if getattr(fields, name, None) is not None}
    flat = pml_vectors(pml, sub_step)
    magnetic = sub_step == "step_B"
    complex_storage = bool(getattr(fields, "force_complex_fields", False)
                           or getattr(grid, "has_bloch", False))
    if complex_storage:
        kinds = _boundary_kinds(grid, pml)
        phased, phase_words = special_kz.bloch_phase_words(
            grid, kinds, bool(SUB_STEPS[sub_step]["backward"]))
        beta_words = (beta_override if beta_override is not None
                      else special_kz.beta_curl_coefficients(
                          grid.beta, grid.dt, magnetic=magnetic,
                          complex_storage=True))
        return folded_beta.plan_folded_beta_bloch_pml_curl_from_arrays(
            sub_step, arrays, flat, codes, phased, phase_words,
            grid.dt / grid.dx, beta_words, EXPANSION, residency,
            functions=functions, has_beta=has_beta)
    plus, minus = (beta_override if beta_override is not None
                   else special_kz.beta_curl_coefficients(
                       grid.beta, grid.dt, magnetic=magnetic,
                       complex_storage=False))
    return folded_beta.plan_folded_beta_pml_curl_from_arrays(
        sub_step, arrays, flat, codes, grid.dt / grid.dx, plus, minus, residency,
        functions=functions, has_beta=has_beta)


def shipped_source(fields: Any, pml: Any, sub_step: str, *, has_beta: bool = True,
                   expansion: Optional[str] = None,
                   contract: str = shaders.CONTRACT_OFF) -> str:
    """The exact source the shipped plan would compile for this case and sub-step."""
    grid = fields.grid
    codes, _ = symmetry.folded_axis_kinds(grid, pml)
    backward = bool(SUB_STEPS[sub_step]["backward"])
    if getattr(fields, "force_complex_fields", False) or getattr(
            grid, "has_bloch", False):
        phased, _ = special_kz.bloch_phase_words(
            grid, _boundary_kinds(grid, pml), backward)
        return folded_beta.folded_beta_bloch_curl_source(
            codes, backward, phased, expansion or EXPANSION, has_beta, contract)
    return folded_beta.folded_beta_curl_source(codes, backward, has_beta, contract)


def entry_point(fields: Any, source: str) -> Any:
    library = compile_source(source)
    if getattr(fields, "force_complex_fields", False) or getattr(
            fields.grid, "has_bloch", False):
        return library.beta_bloch_pml_curl_step
    return library.beta_pml_curl_step


def reference_step(fields: Any, pml: Any) -> None:
    """ONE COMPLETE DRIVER STEP on the array path — the real TEN passes."""
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


def composed_plans(fields: Any, pml: Any, residency: Any) -> Dict[str, Any]:
    """The four arithmetic slots this family owns, plus the parents' two fills."""
    plans = {
        "step_B": engine_curl(fields, pml, "step_B", residency),
        "step_D": engine_curl(fields, pml, "step_D", residency),
        "update_H": engine_constitutive(fields, pml, "H", residency),
        "update_E": engine_constitutive(fields, pml, "E", residency),
        "fill_B": engine_fill(fields, "B", residency),
        "fill_D": engine_fill(fields, "D", residency),
    }
    missing = [name for name in ("step_B", "step_D", "update_H", "update_E")
               if plans[name] is None]
    assert not missing, f"no plan for {missing}"
    return plans


def synced_names(residency: Any, pass_name: str) -> Tuple[str, ...]:
    """Exactly the mirrored volumes the PACKAGE'S OWN model says a pass touches.

    Read from ``metal_coverage.sub_step_volumes`` rather than typed here: if the
    residency model and the array path ever disagreed about which volumes a seam pass
    writes, this walk would hold a stale mirror and the whole-step comparison would
    fail. That is the point — the model is under test too.
    """
    live = set(residency.names)
    return tuple(name for name in
                 metal_coverage.sub_step_volumes(RESIDENCY_NAME[pass_name])
                 if name in live)


def walk_step(fields: Any, pml: Any, plans: Dict[str, Any], residency: Any,
              *, order: Sequence[str] = DRIVER_ORDER,
              resync: bool = True) -> Dict[str, int]:
    """ONE complete step through the TEN passes, device where a plan owns it.

    Every array-path pass is bracketed with an explicit ``sync_out`` / ``sync_in``
    over the volumes the residency model names, which is what a driver adapter would
    have to do and is the only honest way to run a pass no Metal family carries.

    ``resync=False`` is the STALE MIRROR mutation: the host pass runs and its write
    is never carried back to the device, which is the defect the residency layer's
    whole invariant exists to refuse.
    """
    host_passes = 0
    for name in order:
        if name in plans and plans[name] is not None:
            plans[name].run()
            continue
        if name in ("fill_B_near", "fill_D_near", "fill_B_far", "fill_D_far"):
            family = name[5]
            plan = plans.get("fill_" + family)
            if plan is not None:
                (plan.run_near if name.endswith("_near") else plan.run_far)()
                continue
        touched = synced_names(residency, name)
        residency.sync_out(touched)
        HOST_PASS[name](fields)
        if resync:
            residency.sync_in(touched)
        host_passes += 1
    residency.sync_out()
    return {"host_passes": host_passes}


# ---------------------------------------------------------------------------
# LEG execution — provenance, the bound arm, and the compile sweep
# ---------------------------------------------------------------------------

def leg_execution(payload: Dict[str, Any], out: str) -> None:
    """What was hashed, what arm was bound, and does every specialisation BUILD.

    A specialisation that fails to COMPILE is a crash at plan time on a configuration
    nobody swept, so the whole enumeration is built here — in BOTH contraction modes,
    because the fast build is a leg's substrate and a fast build that did not compile
    would report its guard as a null.
    """
    sources = folded_beta.enumerate_folded_beta_sources(EXPANSION)
    kit.provenance(os.path.dirname(out), {
        "folded_beta.py": os.path.join(
            API_ROOT, "meep_gpu/metal_kernels/folded_beta.py"),
        "special_kz.py": os.path.join(
            API_ROOT, "meep_gpu/metal_kernels/special_kz.py"),
        "symmetry.py": os.path.join(API_ROOT, "meep_gpu/metal_kernels/symmetry.py"),
        "folded_complex.py": os.path.join(
            API_ROOT, "meep_gpu/metal_kernels/folded_complex.py"),
        "complex_fields.py": os.path.join(
            API_ROOT, "meep_gpu/metal_kernels/complex_fields.py"),
        "templates.py": os.path.join(API_ROOT, "meep_gpu/metal_kernels/templates.py"),
        "shaders.py": os.path.join(API_ROOT, "meep_gpu/metal_kernels/shaders.py"),
        "launch.py": os.path.join(API_ROOT, "meep_gpu/metal_kernels/launch.py"),
        "coverage.py": os.path.join(API_ROOT, "meep_gpu/metal_kernels/coverage.py"),
        "stepping.py": os.path.join(API_ROOT, "meep_gpu/stepping.py"),
        "gate": os.path.abspath(__file__),
    }, kernel_sources=sources, name="provenance_gate.json")

    # THE WALK ORDER IS THE PACKAGE'S OWN, ASSERTED RATHER THAN COPIED. If the
    # residency model ever renamed or reordered a pass, this gate would be walking a
    # step the composer does not believe in and every whole-step row below would be
    # a statement about a different object.
    walked = tuple(RESIDENCY_NAME[name] for name in DRIVER_ORDER)
    model = tuple(n for n in metal_coverage.RESIDENCY_ORDER if n != "update_P")
    assert walked == model, (walked, model)

    # DISTINCT SOURCES ARE COUNTED BESIDE THE CALLS, and the gap is not a defect: the
    # emitter maps every mirror code to METALLIC for the ghost gather, so many code
    # quadruples produce IDENTICAL text and `compile_source` memoizes by the source
    # STRING. Recording only the call count would claim hundreds of compilations this
    # leg did not perform — measured on this host, 656 calls resolve to
    # `distinct_sources` compilations and the rest are cache hits.
    built = 0
    distinct: Dict[str, int] = {}
    started = time.time()
    for mode in shaders.CONTRACT_MODES:
        emitted = folded_beta.enumerate_folded_beta_sources(EXPANSION, mode)
        distinct[mode] = len(set(emitted.values()))
        for source in emitted.values():
            compile_source(source)
            built += 1
    kit.assert_moved(built, "no specialisation compiled", floor=2 * len(sources))
    kit.assert_census_floor(sum(distinct.values()),
                            "distinct kernel sources compiled", floor=2)

    payload["legs"]["execution"] = {
        "expansion": EXPANSION,
        "fold_expansion": FOLD_EXPANSION,
        "probe_patterns": (PROBE or {}).get("patterns"),
        "specialisations": len(sources),
        "compile_calls": built,
        "distinct_sources_per_contract_mode": distinct,
        "environment": ENVIRONMENT,
        "cases": [label for label, _, _ in CASES],
        "driver_order": list(DRIVER_ORDER),
        "residency_model_order": list(model),
        "no_generated_code_audit": (
            "torch.mps.compile_shader exposes no disassembly, no AIR and no "
            "optimisation report (device.py:46-62): this gate cannot refuse a "
            "compile whose emitted code violates the contraction policy, nor "
            "establish that the pragma was obeyed. Every leg is BEHAVIOURAL — it "
            "catches a wrong answer, never a wrong instruction. This is the one gap "
            "against the Triton twin's certification and it is stated, not buried"),
    }
    save(payload, out)
    log(f"[execution] arm={EXPANSION} specialisations={len(sources)} "
        f"compile_calls={built} distinct={distinct} "
        f"({time.time() - started:.1f}s)")


# ---------------------------------------------------------------------------
# LEG curl — per sub-step, which is where BOTH masks live
# ---------------------------------------------------------------------------

def leg_curl(payload: Dict[str, Any], out: str) -> None:
    """SUB-STEP granularity. The two masks are INVISIBLE at whole-step granularity.

    The driver's fill passes overwrite exactly the planes the ownership masks
    protect, so a whole-step check alone would certify a mask-less kernel as correct.
    THIS IS THE LEG THAT CAN FAIL ON A MASK — and on this family the beta term rides
    the masked plane, so it is also the leg where the composition itself is visible.
    """
    rows: List[Dict[str, Any]] = []
    started = time.time()
    for label, _, _ in CASES:
        fields, pml = build(label)
        for sub_step in ("step_B", "step_D"):
            names = curl_names(sub_step)
            before, after, moved = oracle(
                fields, lambda: getattr(stepping, sub_step)(fields, pml), names)
            kit.assert_moved(moved, f"{label}/{sub_step} reference barely moved",
                             floor=64)
            residency = device.Residency()
            plan = engine_curl(fields, pml, sub_step, residency)
            residency.sync_in()
            plan.run()
            residency.sync_out()
            assert plan.launches == 1, (label, sub_step, plan.launches)
            per = divergence(fields, after)
            rows.append({"case": label, "sub_step": sub_step,
                         "tags": list(tags(fields, pml)),
                         "bc": list(plan.bc), "has_beta": bool(plan.has_beta),
                         "moved": moved, "compared": compared_words(after),
                         "differing": sum(per.values()), "per_target": per,
                         "plan_class": type(plan).__name__})
            log(f"[curl] {label:<26} {sub_step} bc={plan.bc} moved={moved} "
                f"differing={sum(per.values())} ({time.time() - started:.1f}s)")
            restore(fields, before)
            payload["legs"]["curl"] = rows
            save(payload, out)


# ---------------------------------------------------------------------------
# LEG constitutive — no new device code, so the claim is the ADMISSION's
# ---------------------------------------------------------------------------

def leg_constitutive(payload: Dict[str, Any], out: str) -> None:
    """The CERTIFIED bodies over a folded stored extent at nonzero beta.

    This family emits NO constitutive source at all, and that is ASSERTED rather than
    described: ``enumerate_folded_beta_sources`` carries no constitutive label, so a
    future edit that added a body here would be visible in the count the execution
    leg records. The plan CLASS is asserted too — the certified one, reached through
    the certified builder — because "same kernel" has to be a fact rather than a
    claim if the only thing this family contributes on these slots is a predicate.
    """
    assert not any("constitutive" in label for label in
                   folded_beta.enumerate_folded_beta_sources(EXPANSION)), (
        "this family emitted a constitutive source: it is supposed to RE-ADMIT the "
        "certified bodies, not replace them")
    rows: List[Dict[str, Any]] = []
    started = time.time()
    for label, _, is_complex in CASES:
        fields, pml = build(label)
        for side, sub_step in (("H", "update_H"), ("E", "update_E")):
            names = constitutive_names(side)
            before, after, moved = oracle(
                fields, lambda: getattr(stepping, sub_step)(fields, pml), names)
            kit.assert_moved(moved, f"{label}/{side} reference barely moved",
                             floor=64)
            residency = device.Residency()
            plan = engine_constitutive(fields, pml, side, residency)
            residency.sync_in()
            plan.run()
            residency.sync_out()
            assert plan.launches == 1, (label, side, plan.launches)
            expected = (complex_fields.ComplexConstitutivePlan if is_complex
                        else launch.ConstitutivePlan)
            assert isinstance(plan, expected), (label, side, type(plan), expected)
            per = divergence(fields, after)
            rows.append({"case": label, "side": side, "moved": moved,
                         "compared": compared_words(after),
                         "differing": sum(per.values()), "per_target": per,
                         "plan_class": type(plan).__name__})
            log(f"[constitutive] {label:<26} {side} moved={moved} "
                f"differing={sum(per.values())} ({time.time() - started:.1f}s)")
            restore(fields, before)
            payload["legs"]["constitutive"] = rows
            save(payload, out)


# ---------------------------------------------------------------------------
# LEG identity — the ONE claim a source diff against special_kz cannot make
# ---------------------------------------------------------------------------

def leg_identity(payload: Dict[str, Any], out: str) -> None:
    """``has_beta=False`` must BE the certified FOLDED curl, on the device.

    Two directions, and only together do they pin the composition:

    * against ``stepping.py`` on a beta = 0 folded run, which the array path steps
      through the plain folded path;
    * against the CERTIFIED FOLDED PLAN itself — ``symmetry.plan_folded_pml_curl``
      (real) or ``folded_complex.plan_folded_complex_pml_curl`` (complex) — on the
      same seeds. That is the claim a source diff against ``special_kz`` cannot make,
      because that parent carries no fold at all.

    The two sources are NOT string-equal (the beta slot leaves a comment and the
    top-plane block is inserted), so this is a device measurement rather than a
    tautology, and the leg asserts the inequality so it stays one.
    """
    rows: List[Dict[str, Any]] = []
    started = time.time()
    for label, _, is_complex in CASES:
        if "walled" in label:
            # The wall changes no curl code; skipping it here keeps the leg's
            # ten rows one per DISTINCT fold shape rather than per grid.
            continue
        for sub_step in ("step_B", "step_D"):
            fields, pml = build(label, beta=0.0)
            reference, reference_pml = build(label, beta=0.0)
            certified, certified_pml = build(label, beta=0.0)
            codes, _ = symmetry.folded_axis_kinds(fields.grid, pml)

            residency = device.Residency()
            plan = from_arrays_curl(fields, pml, sub_step, codes, residency,
                                    has_beta=False)
            residency.sync_in()
            plan.run()
            residency.sync_out()

            getattr(stepping, sub_step)(reference, reference_pml)

            certified_residency = device.Residency()
            if is_complex:
                certified_plan = folded_complex.plan_folded_complex_pml_curl(
                    certified, certified_pml, sub_step, certified_residency,
                    probe=FOLD_PROBE)
            else:
                certified_plan = symmetry.plan_folded_pml_curl(
                    certified, certified_pml, sub_step, certified_residency)
            assert certified_plan is not None, (label, sub_step, "no certified plan")
            certified_residency.sync_in()
            certified_plan.run()
            certified_residency.sync_out()

            names = curl_names(sub_step)
            versus_reference = sum(
                differing(getattr(fields, n), getattr(reference, n)) for n in names)
            versus_certified = sum(
                differing(getattr(fields, n), getattr(certified, n)) for n in names)
            moved = sum(differing(np.zeros_like(getattr(reference, n)),
                                  getattr(reference, n)) for n in names)
            kit.assert_moved(moved, f"{label}/{sub_step} identity leg moved nothing",
                             floor=64)

            beta_free = shipped_source(fields, pml, sub_step, has_beta=False)
            if is_complex:
                phased, _ = special_kz.bloch_phase_words(
                    fields.grid, _boundary_kinds(fields.grid, pml),
                    bool(SUB_STEPS[sub_step]["backward"]))
                parent = folded_complex.folded_bloch_curl_source(
                    codes, bool(SUB_STEPS[sub_step]["backward"]), phased,
                    FOLD_EXPANSION)
            else:
                parent = symmetry.folded_curl_source(
                    codes, bool(SUB_STEPS[sub_step]["backward"]))
            assert beta_free != parent, (
                label, sub_step,
                "the beta-free source is string-identical to the certified folded "
                "curl, so this leg is a tautology rather than a measurement")

            rows.append({"case": label, "sub_step": sub_step,
                         "tags": list(tags(fields, pml)),
                         "moved": moved,
                         "compared": compared_words(snapshot(reference, names)) * 2,
                         "differing_vs_stepping": versus_reference,
                         "differing_vs_certified_folded_plan": versus_certified,
                         "sources_are_string_equal": False})
            log(f"[identity] {label:<26} {sub_step} vs_stepping={versus_reference} "
                f"vs_certified={versus_certified} moved={moved} "
                f"({time.time() - started:.1f}s)")
            payload["legs"]["identity"] = rows
            save(payload, out)


# ---------------------------------------------------------------------------
# LEG whole_step — the real arbiter, and it walks TEN passes
# ---------------------------------------------------------------------------

def leg_whole_step(payload: Dict[str, Any], out: str, budget: int = 12) -> None:
    """Per COMPLETE STEP, over the driver's real ten-pass list.

    Every slot's launch counter is asserted at the EXACT count its plan owes per
    cycle, because a slot that passes by NOT EXECUTING is the hollow pass this
    discipline exists to prevent. ``Residency.verify()`` is asserted empty after every
    step, which is the stale-mirror invariant measured rather than assumed.

    THE SUBNORMAL CENSUS IS TAKEN PER STEP AND REPORTED AS A WINDOW. Band entry is a
    RUN-and-WINDOW fact rather than a family fact, so a row recording a scalar count
    cannot be read for what it covers. A case whose census fires is REFUSED BY NAME
    and its comparisons are withheld from the certified total: that is a COVERAGE
    REFUSAL, not a failure, and :func:`leg_precondition` is what proves the detector
    can fire at all.
    """
    rows: List[Dict[str, Any]] = []
    started = time.time()
    for label, _, _ in CASES:
        fields, pml = build(label)
        reference_fields, reference_pml = build(label)
        residency = device.Residency()
        plans = composed_plans(fields, pml, residency)
        residency.sync_in()

        first_divergent = None
        per_step: List[int] = []
        census_per_step: List[int] = []
        fired: List[int] = []
        host_passes = 0
        for step in range(budget):
            reference_step(reference_fields, reference_pml)
            host_passes = walk_step(fields, pml, plans, residency)["host_passes"]
            stale = {k: v for k, v in residency.verify().items() if v}
            assert not stale, (label, step, "STALE MIRROR after a complete step",
                               stale)
            per = {name: differing(getattr(fields, name),
                                   getattr(reference_fields, name))
                   for name in STORED if getattr(fields, name, None) is not None}
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
        kit.assert_moved(evolved, f"{label} whole-step state never evolved",
                         floor=256)
        for slot in ("step_B", "step_D", "update_H", "update_E"):
            assert plans[slot].launches == budget, (
                label, slot, plans[slot].launches, budget)
        fills: Dict[str, Any] = {}
        for slot in ("fill_B", "fill_D"):
            plan = plans[slot]
            if plan is None:
                fills[slot] = None
                continue
            owed = budget * (len(plan.near) + len(plan.far))
            assert plan.launches == owed, (label, slot, plan.launches, owed)
            assert len(plan.near) > 0, (label, slot)
            fills[slot] = {"near": len(plan.near), "far": len(plan.far),
                           "launches": plan.launches}
        walled = "walled" in tags(fields, pml)
        assert bool(host_passes > 2) == walled, (
            label, host_passes,
            "a walled case must run the two fills on the array path and an unwalled "
            "one must not: the host-pass count and the wall tag disagree")

        refused = bool(fired)
        # A REFUSAL IS NOT A PASS BY ANOTHER NAME. The byte claim is only withheld
        # where the precondition FAILED; where it held, a divergence is still a
        # failure and still stops the leg here.
        if not refused:
            assert first_divergent is None, (label, first_divergent, per_step)
        window = {"first_step": fired[0] if fired else None,
                  "last_step": fired[-1] if fired else None,
                  "steps_censused": budget,
                  "subnormal_words_per_step": census_per_step}
        rows.append({"case": label, "tags": list(tags(fields, pml)), "budget": budget,
                     "per_step_differing": per_step,
                     "first_divergent": first_divergent,
                     "evolved_words": evolved,
                     "host_passes_per_step": host_passes,
                     "device_fills": fills,
                     "subnormal_window": window,
                     "refused": refused,
                     "refusal": (
                         f"{label}: REFUSED (subnormal precondition) — the census "
                         f"fired on steps {fired[0]}..{fired[-1]} of {budget}; every "
                         f"arithmetic claim on this backend rides a CHECKED "
                         f"subnormal-free precondition and this case does not meet "
                         f"it") if refused else None,
                     "compared": 0 if refused else
                                 compared_words(snapshot(reference_fields)) * budget,
                     "launches": {k: (v.launches if v is not None else None)
                                  for k, v in plans.items()},
                     "residency_syncs": {"in": residency.syncs_in,
                                         "out": residency.syncs_out}})
        log(f"[whole_step] {label:<26} {budget} steps host_passes={host_passes} "
            f"first_divergent="
            f"{None if first_divergent is None else first_divergent['step']} "
            f"evolved={evolved} window=[{window['first_step']}, "
            f"{window['last_step']}]{' REFUSED' if refused else ''} "
            f"({time.time() - started:.1f}s)")
        payload["legs"]["whole_step"] = rows
        save(payload, out)


# ---------------------------------------------------------------------------
# LEG position — the composition's own hazard, isolated on the device
# ---------------------------------------------------------------------------

#: The two beta lines, per arm, spelled tightly enough to exclude the KERNEL
#: SIGNATURE. ``beta_plus`` alone also matches ``constant float& beta_plus``, and a
#: filter that caught the declaration would find three lines and report every
#: position mutation as NEEDLE-MISSED.
BETA_LINE_MARKERS: Tuple[str, ...] = (
    "beta_plus * b", "beta_minus * a",
    "float2(bpr, bpi), b", "float2(bmr, bmi), a")


def is_complex(fields: Any) -> bool:
    return bool(getattr(fields, "force_complex_fields", False)
                or getattr(fields.grid, "has_bloch", False))


def zero_literal(fields: Any) -> str:
    return templates.COMPLEX_ZERO if is_complex(fields) else "0.0f"


def move_beta_after(source: str, marker: str) -> str:
    """Relocate the two beta lines to just after the LAST mask line of one block.

    THE ANCHOR IS THE MASK, NOT ITS FLAG DECLARATION, and the distinction is the
    difference between a must-catch mutation and a null: anchoring on ``bool at_x =``
    inserts the beta lines BEFORE the mask assignments that follow it, which changes
    nothing about the order.
    """
    lines = source.splitlines()
    beta = [line for line in lines
            if any(mark in line for mark in BETA_LINE_MARKERS)]
    if len(beta) != 2:
        raise LookupError(f"expected two beta lines, found {len(beta)}")
    kept = [line for line in lines if line not in beta]
    positions = [index for index, line in enumerate(kept)
                 if marker in line and "?" in line and "curl" in line]
    if not positions:
        raise LookupError(f"no mask line carries {marker!r}")
    at = positions[-1]
    return "\n".join(kept[:at + 1] + beta + kept[at + 1:])


def leg_position(payload: Dict[str, Any], out: str) -> None:
    """WHERE THE BETA TERM SITS, measured per case and per sub-step.

    The array path adds the term after the curl and BEFORE both masks. Moving it past
    either one must change the answer, and HOW MANY words it changes is recorded
    rather than only whether it did — the count is the size of the plane the mask
    owns, and a count that stopped matching the geometry would be a finding even
    while the leg still "passed".

    The top-plane arm is a MEASURED NULL on a MIRROR_METALLIC fold, where the emitter
    writes no mask at all, and the row says so with its reason rather than being
    dropped or counted as a catch.
    """
    rows: List[Dict[str, Any]] = []
    started = time.time()
    for label, _, _ in CASES:
        for sub_step in ("step_B", "step_D"):
            fields, pml = build(label)
            reference, reference_pml = build(label)
            codes, _ = symmetry.folded_axis_kinds(fields.grid, pml)
            case_tags = tags(fields, pml)
            names = curl_names(sub_step)
            getattr(stepping, sub_step)(reference, reference_pml)

            shipped = shipped_source(fields, pml, sub_step)
            variants: Dict[str, Optional[str]] = {
                "after_cell_zero_mask": move_beta_after(shipped, "at_"),
                "after_both_masks": (move_beta_after(shipped, "last_")
                                     if "mirror_periodic" in case_tags else None),
            }
            per_variant: Dict[str, Any] = {}
            for name, mutant in variants.items():
                if mutant is None:
                    per_variant[name] = kit.predicted_null(
                        {"differing": 0},
                        "this fold is MIRROR_METALLIC: the top plane is OWNED and "
                        "STEPPED, the emitter writes no top-plane mask at all, and "
                        "there is no second mask to move the term past")
                    continue
                state = snapshot(fields, CURL_ARRAYS)
                residency = device.Residency()
                counter = kit.Counter(entry_point(fields, mutant))
                plan = from_arrays_curl(
                    fields, pml, sub_step, codes, residency,
                    functions={shaders.CONTRACT_OFF: counter})
                residency.sync_in()
                plan.run()
                residency.sync_out()
                moved_words = sum(differing(getattr(fields, n),
                                            getattr(reference, n)) for n in names)
                per_variant[name] = {"differing": moved_words,
                                     "launches": counter.launches}
                assert counter.launches == 1, (label, sub_step, name)
                restore(fields, state)
            rows.append({"case": label, "sub_step": sub_step,
                         "tags": list(case_tags), "variants": per_variant})
            log(f"[position] {label:<26} {sub_step} "
                f"{ {k: v['differing'] for k, v in per_variant.items()} } "
                f"({time.time() - started:.1f}s)")
            payload["legs"]["position"] = rows
            save(payload, out)

    for row in rows:
        assert row["variants"]["after_cell_zero_mask"]["differing"] > 0, row
        top = row["variants"]["after_both_masks"]
        if not top.get("predicted_null"):
            assert top["differing"] > 0, row
    fired = sum(1 for row in rows
                if not row["variants"]["after_both_masks"].get("predicted_null"))
    kit.assert_census_floor(
        fired, "cases where the TOP-plane arm of the position leg is reachable")


# ---------------------------------------------------------------------------
# LEG spellings — two refuted spellings that are UNOBSERVABLE here, proved in
# both halves rather than asserted in one
# ---------------------------------------------------------------------------

MICRO_TEMPLATE = """
#include <metal_stdlib>
using namespace metal;

__CONTRACT__
__HELPERS__

kernel void tail(device __T__* out [[buffer(0)]],
                 device const __T__* a [[buffer(1)]],
                 device const __T__* b [[buffer(2)]],
                 constant float& cr [[buffer(3)]],
                 constant float& ci [[buffer(4)]],
                 constant uint& n [[buffer(5)]],
                 uint idx [[thread_position_in_grid]])
{
    if (idx >= n) { return; }
__BODY__
}
"""


def micro(body: str, complex_storage: bool) -> str:
    return templates.substitute(MICRO_TEMPLATE, {
        "__CONTRACT__": templates.contraction_pragma(shaders.CONTRACT_OFF),
        "__HELPERS__": (templates.complex_helpers(EXPANSION)
                        if complex_storage else ""),
        "__T__": "float2" if complex_storage else "float",
        "__BODY__": body})


def run_micro(body: str, left: np.ndarray, right: np.ndarray,
              coefficient: Tuple[float, float]) -> np.ndarray:
    """Launch one micro-kernel over a value table and return what it wrote.

    The output starts at a SENTINEL and the caller asserts every lane moved, so a
    kernel that wrote nothing cannot be read as agreement.
    """
    import torch  # noqa: PLC0415

    complex_storage = left.dtype == np.complex64
    library = compile_source(micro(body, complex_storage))
    count = int(left.size)
    sentinel = np.full(count, np.complex64(-7.5 - 3.25j) if complex_storage
                       else np.float32(-7.5), dtype=left.dtype)

    def to_device(array: np.ndarray) -> Any:
        flat = np.ascontiguousarray(array).reshape(-1)
        if complex_storage:
            return torch.from_numpy(flat.view(np.float32).reshape(-1, 2).copy()
                                    ).to("mps")
        return torch.from_numpy(flat.copy()).to("mps")

    out = to_device(sentinel)
    library.tail(out, to_device(left), to_device(right),
                 float(coefficient[0]), float(coefficient[1]), count)
    torch.mps.synchronize()
    raw = out.cpu().numpy().reshape(-1)
    got = (raw.view(np.complex64) if complex_storage
           else raw.astype(np.float32)).reshape(left.shape)
    assert differing(sentinel, got) > 0, "VACUOUS: the micro-kernel wrote nothing"
    return got


def leg_spellings(payload: Dict[str, Any], out: str) -> None:
    """TWO REFUTED SPELLINGS THAT THIS FAMILY CANNOT TELL APART, and why.

    ``negation_spelled_as_zero_minus`` and ``complex_zero_cross_terms_folded`` are
    both recorded in leg ``mutations`` as measured nulls, and a measured null is
    worthless without its mechanism: "the spellings are the same" (false) and "this
    family cannot tell them apart" (true) are different claims.

    HALF ONE — THE SPELLINGS GENUINELY DIFFER ON THIS BACKEND, measured here on the
    discriminating operand patterns rather than inherited from the probe. Both are
    required to move at least one word.

    HALF TWO — THE FAMILY CANNOT REACH THE PATTERN, and this is a MEASUREMENT, not an
    argument. Both differences reduce to the same question: is the value the beta term
    is subtracted from ever a NEGATIVE ZERO? A needle replaces the beta insert with an
    export of ``curl0/curl1/curl2`` and an early return, so the gate reads the exact
    operand off the device on every case and sub-step and censuses the ``-0.0`` words.
    ``grid.beta`` is legal only on an effective-2-D grid, whose invariant axis stores
    ONE cell, so the z-shifted operand IS the centre operand and ``b - b_z`` is
    ``+0.0`` for every finite ``b``; a float sum carrying a ``+0.0`` addend is never
    ``-0.0``. The census is what turns that paragraph into a number.
    """
    rows: List[Dict[str, Any]] = []
    started = time.time()

    # --- HALF ONE, the real negation ---------------------------------------
    curl = np.array([-0.0, +0.0, -0.0, +0.0], dtype=np.float32)
    term = np.array([+0.0, -0.0, -0.0, +0.0], dtype=np.float32)
    shipped = run_micro("    out[idx] = a[idx] - b[idx];", curl, term, (0.0, 0.0))
    refuted = run_micro("    out[idx] = a[idx] + (0.0f - b[idx]);", curl, term,
                        (0.0, 0.0))
    negation_delta = differing(shipped, refuted)
    rows.append({"spelling": "curl + (0.0f - x) vs curl - x",
                 "arm": "real", "lanes": int(curl.size),
                 "differing": negation_delta,
                 "discriminating_pattern": "curl = -0.0 and x = +0.0"})
    log(f"[spellings] real negation differs on {negation_delta}/{curl.size} lanes "
        f"({time.time() - started:.1f}s)")
    assert negation_delta == 1, (
        "the refuted negation must differ on EXACTLY the (-0.0, +0.0) lane; if it "
        "does not, this backend canonicalizes zeros and the whole discussion "
        "changes", shipped, refuted)

    # --- HALF ONE, the complex cross terms ----------------------------------
    # The product ITSELF, not the subtraction: the shipped tail then subtracts it,
    # and the subtraction is exactly what half two shows absorbs the difference.
    values = np.array([complex(0.0, 0.0), complex(-0.0, -0.0), complex(0.0, -0.0),
                       complex(-0.0, 0.0), complex(1.5, -0.0), complex(-0.0, 1.5),
                       complex(0.75, -0.5), complex(-1.25, 0.25)],
                      dtype=np.complex64)
    partner = np.tile(values, len(values))
    other = np.repeat(values, len(values))
    cross_delta = 0
    per_coefficient: List[Dict[str, Any]] = []
    for coefficient in ((0.0, 0.5), (-0.0, 0.5), (0.0, -0.5), (-0.0, -0.5)):
        product = run_micro("    out[idx] = c_mul(float2(cr, ci), b[idx]);",
                            other, partner, coefficient)
        folded = run_micro(
            "    out[idx] = float2(-(ci * b[idx].y), ci * b[idx].x);",
            other, partner, coefficient)
        delta = differing(product, folded)
        cross_delta += delta
        per_coefficient.append({"coefficient": list(coefficient),
                                "differing": delta, "lanes": int(partner.size)})
    rows.append({"spelling": "float2(-(ci*b.y), ci*b.x) vs c_mul((cr, ci), b)",
                 "arm": "complex", "lanes": int(partner.size) * 4,
                 "differing": cross_delta, "per_coefficient": per_coefficient,
                 "discriminating_pattern":
                     "the fma's zero addend and the product's zero disagree in sign"})
    log(f"[spellings] complex cross terms differ on {cross_delta} words "
        f"({time.time() - started:.1f}s)")
    kit.assert_census_floor(
        cross_delta,
        "the folded cross terms reproduced c_mul on EVERY signed-zero pattern, so "
        "the spelling this gate refutes is not refutable here and the mutation row "
        "would be a null about nothing")

    # --- HALF TWO, the curl census, on every case and sub-step --------------
    census: List[Dict[str, Any]] = []
    for label, _, _ in CASES:
        for sub_step in ("step_B", "step_D"):
            fields, pml = build(label)
            # Signed zeros are PLANTED first: if the operand class the difference
            # needs can be produced at all on these grids, this is the state that
            # produces it, and the census below is then a statement about a state
            # that TRIED to reach the pattern rather than about a bland one.
            plant_signed_zeros(fields, tuple(SUB_STEPS[sub_step]["sources"]))
            codes, _ = symmetry.folded_axis_kinds(fields.grid, pml)
            insert = (special_kz._COMPLEX_BETA_INSERT if is_complex(fields)
                      else special_kz._REAL_BETA_INSERT)
            export = needle(
                shipped_source(fields, pml, sub_step), insert,
                "    f0[ii] = curl0; f1[ii] = curl1; f2[ii] = curl2;\n"
                "    return;")
            residency = device.Residency()
            counter = kit.Counter(entry_point(fields, export))
            plan = from_arrays_curl(fields, pml, sub_step, codes, residency,
                                    functions={shaders.CONTRACT_OFF: counter})
            residency.sync_in()
            plan.run()
            residency.sync_out()
            assert counter.launches == 1, (label, sub_step)
            # THE SPLIT IS THE WHOLE POINT, AND IT IS THIS LEG'S OWN CORRECTION.
            # The beta term is added to `curl0` and `curl1` ONLY — never to `curl2`
            # (special_kz._REAL_BETA_INSERT / ._COMPLEX_BETA_INSERT) — and the
            # structural argument only covers those two:
            #   curl0 = dtdx * ((c_y - c) + (b - b_z))
            #   curl1 = dtdx * ((a_z - a) + (c - c_x))
            #   curl2 = dtdx * ((b_x - b) + (a - a_y))
            # curl0 and curl1 each carry ONE difference along the INVARIANT axis,
            # which stores a single cell, so that addend is a hard +0.0 and their sum
            # can never be -0.0. `curl2` carries NO invariant-axis difference and CAN
            # be -0.0 — measured here at 48 words on real_x_fold/step_B — which is
            # what makes this census a measurement instead of a blind spot. Censusing
            # all three together (which is how this leg was first written) reported
            # curl2's negative zeros as a refutation of a claim that was never about
            # curl2.
            targets = tuple(SUB_STEPS[sub_step]["targets"])
            touched = np.concatenate([words(getattr(fields, name))
                                      for name in targets[:2]])
            untouched = words(getattr(fields, targets[2]))
            negative_zero = int(np.count_nonzero(touched == np.uint32(0x80000000)))
            control = int(np.count_nonzero(untouched == np.uint32(0x80000000)))
            positive_zero = int(np.count_nonzero(touched == np.uint32(0)))
            census.append({"case": label, "sub_step": sub_step,
                           "beta_touched_curls": list(targets[:2]),
                           "untouched_curl": targets[2],
                           "curl_words_exported": int(touched.size),
                           "negative_zero_curl_words": negative_zero,
                           "positive_zero_curl_words": positive_zero,
                           "negative_zero_words_in_the_untouched_curl": control})
            log(f"[spellings] {label:<28} {sub_step} beta-touched curl words="
                f"{touched.size} -0.0={negative_zero} +0.0={positive_zero} "
                f"| untouched curl -0.0={control} ({time.time() - started:.1f}s)")
            payload["legs"]["spellings"] = {"halves_one": rows, "curl_census": census}
            save(payload, out)
            kit.assert_census_floor(int(touched.size),
                                    f"{label}/{sub_step} exported curl words",
                                    floor=64)
            assert negative_zero == 0, (
                label, sub_step,
                "a NEGATIVE ZERO reached a curl the beta term is subtracted from. "
                "The two refuted spellings are then OBSERVABLE on this family after "
                "all, and their mutation rows must be re-armed as must_catch",
                census[-1])

    kit.assert_census_floor(
        sum(row["negative_zero_words_in_the_untouched_curl"] for row in census),
        "the census never saw a negative zero in ANY curl, including the one the "
        "beta term does not touch and the structural argument does not cover — so it "
        "cannot be shown to detect one at all and the zeros above are a blind spot "
        "rather than a result")

    payload["legs"]["spellings"] = {
        "halves_one": rows,
        "curl_census": census,
        "conclusion": (
            "BOTH refuted spellings are genuinely different on this backend (half "
            "one) and BOTH are unobservable on this family's grids (half two). Every "
            "difference they can produce is a signed zero in the beta increment, and "
            "`curl - (+/-0.0)` is one word unless `curl` is a NEGATIVE ZERO — which "
            "the census measures never happens on the TWO curls the beta term is "
            "added to, over "
            f"{sum(r['curl_words_exported'] for r in census)} exported words on "
            f"{len(census)} case/sub-step pairs, with signed zeros PLANTED into the "
            "source volumes. The reason is structural and is narrower than 'the curl': "
            "curl0 and curl1 each carry one difference along the INVARIANT axis, which "
            "stores a single cell, so that addend is a hard +0.0 and a float sum with "
            "a +0.0 addend is never -0.0. curl2 carries no such difference, is never "
            "given the beta term, and DOES reach -0.0 — "
            f"{sum(r['negative_zero_words_in_the_untouched_curl'] for r in census)} "
            "words across the matrix — which is this census's own control"),
    }
    save(payload, out)


# ---------------------------------------------------------------------------
# LEG guard — the contraction directive, MEASURED, PER ARM
# ---------------------------------------------------------------------------

COMPLEX_GUARD_NULL_REASON = (
    "the shipped complex arm is the FMA_V1 spelling, whose every product is already "
    "inside an explicit fma() or carries an exact zero operand: there is no IMPLICIT "
    "multiply-add pair for contract(fast) to fuse, so the directive is byte-invisible "
    "here. That is a property of the SPELLING, not of the harness — the NAIVE control "
    "below plants a body that DOES contract and must diverge wherever a general "
    "complex product is live")


def run_variant(label: str, sub_step: str, *, expansion: Optional[str] = None,
                contract: str = shaders.CONTRACT_OFF,
                has_beta: bool = True) -> Tuple[Dict[str, Any], int, int]:
    """One compiled variant against the array path, on a freshly built case.

    Returns (device state, differing words vs stepping.py, moved words).
    """
    fields, pml = build(label)
    reference, reference_pml = build(label)
    codes, _ = symmetry.folded_axis_kinds(fields.grid, pml)
    source = shipped_source(fields, pml, sub_step, has_beta=has_beta,
                            expansion=expansion, contract=contract)
    residency = device.Residency()
    counter = kit.Counter(entry_point(fields, source))
    plan = from_arrays_curl(fields, pml, sub_step, codes, residency,
                            functions={shaders.CONTRACT_OFF: counter},
                            has_beta=has_beta)
    residency.sync_in()
    plan.run()
    residency.sync_out()
    assert counter.launches == 1, (label, sub_step, expansion, contract)
    getattr(stepping, sub_step)(reference, reference_pml)
    names = curl_names(sub_step)
    state = {n: np.array(getattr(fields, n), copy=True) for n in names}
    bad = sum(differing(state[n], getattr(reference, n)) for n in names)
    moved = sum(differing(np.zeros_like(getattr(reference, n)),
                          getattr(reference, n)) for n in names)
    return state, bad, moved


def leg_guard(payload: Dict[str, Any], out: str) -> None:
    """``contract(fast)`` must DIVERGE on the arm that can contract.

    On Metal this is a SOURCE variant rather than a launch keyword, so the two modes
    are two compiled kernels. The REAL arm's hazard is this family's own tail —
    ``curl - (c * b)`` — which is exactly the shape a compiler contracts into an fma,
    and it is why the pragma is mandatory rather than tidy.

    THE FLOOR IS PER ARM. A single sum over every row would be satisfied by the real
    arm alone while every complex row sat at zero, and the null would not even be
    recorded. The complex rows carry :data:`COMPLEX_GUARD_NULL_REASON` and are backed
    by a NAIVE-arm CONTROL that must diverge exactly where a general complex product
    is live — without it, "the complex arm does not contract" and "this leg cannot see
    a contraction" would be the same measurement.
    """
    rows: List[Dict[str, Any]] = []
    started = time.time()
    for label, _, is_complex in CASES:
        for sub_step in ("step_B", "step_D"):
            _, guarded_bad, moved = run_variant(label, sub_step)
            _, fast_bad, _ = run_variant(label, sub_step,
                                         contract=shaders.CONTRACT_FAST)
            kit.assert_moved(moved, f"{label}/{sub_step} guard leg moved nothing",
                             floor=64)
            row = {"case": label, "sub_step": sub_step,
                   "arm": "complex" if is_complex else "real",
                   "guarded_differing": guarded_bad, "fast_differing": fast_bad,
                   "moved": moved}
            if is_complex:
                kit.predicted_null(row, COMPLEX_GUARD_NULL_REASON)
            rows.append(row)
            log(f"[guard] {label:<26} {sub_step} guarded={guarded_bad} "
                f"fast={fast_bad} ({time.time() - started:.1f}s)")
            payload["legs"]["guard"] = rows
            save(payload, out)
            assert guarded_bad == 0, row
            if is_complex:
                assert fast_bad == 0, (
                    f"{row}: the complex arm DID diverge under contract(fast), which "
                    f"contradicts the recorded null — an implicit multiply-add pair "
                    f"has appeared in the complex body")

    real_fast = sum(row["fast_differing"] for row in rows if row["arm"] == "real")
    kit.assert_census_floor(
        real_fast,
        "REAL-arm contraction guard effect (the contract=fast build agreed with "
        "stepping.py on every real case, so the one compile option this "
        "certification rests on is not measurable and its pinning is decorative)")

    # THE CONTROL. Plant the OTHER expansion arm — four separately rounded products
    # and two rounded adds, an explicitly contractable shape — into the same complex
    # kernel and require contract(fast) to move bytes wherever a general complex
    # product is live.
    other = "NAIVE" if EXPANSION == "FMA_V1" else "FMA_V1"
    control: List[Dict[str, Any]] = []
    for label, _, is_complex in CASES:
        if not is_complex:
            continue
        fields, pml = build(label)
        general = "general_phase" in tags(fields, pml)
        for sub_step in ("step_B", "step_D"):
            off, off_bad, _ = run_variant(label, sub_step, expansion=other)
            fast, _, _ = run_variant(label, sub_step, expansion=other,
                                     contract=shaders.CONTRACT_FAST)
            delta = sum(differing(off[n], fast[n]) for n in off)
            control.append({"case": label, "sub_step": sub_step, "arm": other,
                            "off_vs_fast": delta,
                            "wrong_arm_vs_stepping": off_bad,
                            "a_general_complex_product_is_live": bool(general)})
            log(f"[guard control] {label:<26} {sub_step} arm={other} "
                f"off_vs_fast={delta} wrong_arm_vs_stepping={off_bad} "
                f"general={general} ({time.time() - started:.1f}s)")
            assert bool(delta) == bool(general), (
                f"{label}/{sub_step}: the NAIVE control diverged={bool(delta)} while "
                f"a general complex product was live={bool(general)}. The control's "
                f"own explanation of WHICH rows can contract no longer matches what "
                f"it measured")
    payload["legs"]["guard_control"] = {
        "arm": other,
        "why": ("the NAIVE body spells `(a*b) - (c*d)`, an IMPLICIT multiply-add "
                "pair, so contract(fast) has something to fuse. A general complex "
                "product only exists where a Bloch phase is live and is neither 1+0j "
                "nor -1+0j: the ZONE-EDGE row carries a LIVE phase whose word is "
                "exactly (-1.0, +0.0), so every cross term is exact and it is a zero "
                "BESIDE a general-phase row that is not — which is what makes this "
                "control a discriminator rather than a pass/fail. The BETA "
                "coefficient itself is purely imaginary, so it never makes one, and "
                "the unphased complex rows are zeros for that reason"),
        "rows": control,
        "diverging_rows": sum(1 for r in control if r["off_vs_fast"]),
        "wrong_arm_diverging_rows": sum(1 for r in control
                                        if r["wrong_arm_vs_stepping"]),
    }
    save(payload, out)
    kit.assert_census_floor(
        sum(r["off_vs_fast"] for r in control),
        f"complex contraction CONTROL ({other} arm): contract(fast) moved nothing "
        f"even on a body that spells an implicit multiply-add pair, so this leg "
        f"cannot see a contraction in the complex kernel at all and the shipped "
        f"arm's zero rows are an absence of measurement rather than a null")
    kit.assert_census_floor(
        sum(r["wrong_arm_vs_stepping"] for r in control),
        f"the WRONG expansion arm ({other}) reproduced stepping.py on every complex "
        f"case, so which arm this host's reference takes is unmeasurable here and "
        f"the probe-bound choice is decorative")


# ---------------------------------------------------------------------------
# LEG band — the subnormal reach, per value scale, as a WINDOW
# ---------------------------------------------------------------------------

def leg_band(payload: Dict[str, Any], out: str) -> None:
    """Where the flush starts to bite, measured on THIS family's own step.

    ``step_B`` is linear in the state at fixed coefficients, so scaling the state
    scales every intermediate — which is what makes a scale sweep a clean probe of
    the band rather than a probe of the physics. The window is what is recorded: a
    census that reported a scalar would licence nothing about a run whose leading
    edge sweeps the band.
    """
    label = "real_y_periodic_even"
    rows: List[Dict[str, Any]] = []
    first_fired = last_fired = None
    started = time.time()
    for index, scale in enumerate((1.0, 1e-30, 1e-34, 1e-38, 1e-40, 1e-44)):
        fields, pml = build(label)
        reference, reference_pml = build(label)
        for name in STORED:
            for target in (fields, reference):
                array = getattr(target, name, None)
                if array is not None:
                    array *= np.array(scale, dtype=np.float32)
        band = state_census(reference)
        names = curl_names("step_B")
        residency = device.Residency()
        plan = engine_curl(fields, pml, "step_B", residency)
        residency.sync_in()
        plan.run()
        residency.sync_out()
        stepping.step_B(reference, reference_pml)
        differ = sum(differing(getattr(fields, n), getattr(reference, n))
                     for n in names)
        moved = sum(differing(np.zeros_like(getattr(reference, n)),
                              getattr(reference, n)) for n in names)
        after_band = state_census(reference)
        if differ:
            first_fired = index if first_fired is None else first_fired
            last_fired = index
        rows.append({"scale": scale, "subnormal_words_before": band,
                     "subnormal_words_after": after_band,
                     "differing": differ, "moved": moved})
        log(f"[band] scale={scale:g} subnormal_before={band} "
            f"subnormal_after={after_band} differing={differ} moved={moved} "
            f"({time.time() - started:.1f}s)")
        payload["legs"]["band"] = {"per_scale": rows}
        save(payload, out)

    assert rows[0]["differing"] == 0 and rows[1]["differing"] == 0, rows
    kit.assert_moved(rows[0]["moved"], "the physical-band case moved nothing",
                     floor=64)
    assert any(row["differing"] for row in rows), (
        "no scale reached the band: this leg measured nothing, so the precondition "
        "the whole certification rides on is unfalsifiable here")
    payload["legs"]["band"] = {
        "per_scale": rows,
        "window": {"first_fired_index": first_fired,
                   "last_fired_index": last_fired,
                   "first_fired_scale": rows[first_fired]["scale"],
                   "last_fired_scale": rows[last_fired]["scale"]},
        "note": ("step_B is LINEAR in the state at fixed coefficients, so a scale "
                 "sweep moves the whole computation through the band together. The "
                 "identity holds through 1e-30 and fails once the band is reached, "
                 "which is why the claim rides a CHECKED precondition rather than a "
                 "tolerance: a tolerance would be either meaninglessly loose or "
                 "would falsely admit the divergent case"),
    }
    save(payload, out)


# ---------------------------------------------------------------------------
# LEG precondition — the control that makes the per-step census FIRE
# ---------------------------------------------------------------------------

#: Scale, and what this leg REQUIRES of the census at it. ``clean`` is what makes a
#: firing row mean something — a detector that fired on everything would refuse the
#: physical band too and certify nothing — and ``fire`` is what makes an empty window
#: a measurement rather than a silence.
#:
#: ``record`` IS NOT A HEDGE, IT IS A MEASUREMENT THAT REFUTED AN ASSUMPTION. This
#: leg was first written with 1e-30 as a second ``clean`` row, on the reasoning that
#: leg ``band`` finds the SINGLE-STEP identity intact there. It failed, and the
#: failure is the point: on the COMPLEX case the census fires at steps 6-7 of 8 at
#: 1e-30 while the real case stays clean for all 8. Band entry is a RUN-and-WINDOW
#: fact rather than a family fact, so 1e-30 is recorded with its window instead of
#: being asserted either way.
PRECONDITION_SCALES: Tuple[Tuple[str, float, str], ...] = (
    ("physical", 1.0, "clean"),
    ("small_normal", 1e-30, "record"),
    ("subnormal_band", 1e-38, "fire"),
    ("deep_subnormal", 1e-41, "fire"),
)


def leg_precondition(payload: Dict[str, Any], out: str, budget: int = 12) -> None:
    """A precondition never demonstrated to FIRE is decoration.

    :func:`leg_whole_step` censuses every case per step and reports a window; on the
    physical band that window is empty, and an empty window is exactly what a broken
    census also produces. This leg drives the SAME per-step census over a state scaled
    INTO the band and requires it to fire, on BOTH storages, so the empty windows next
    door are a measurement rather than a silence.
    """
    rows: List[Dict[str, Any]] = []
    started = time.time()
    for case in ("real_y_periodic_even", "complex_y_periodic_even"):
        for label, scale, requirement in PRECONDITION_SCALES:
            reference, reference_pml = build(case)
            if scale != 1.0:
                for name in STORED:
                    array = getattr(reference, name, None)
                    if array is not None:
                        array *= np.array(scale, dtype=np.float32)
            fired: List[int] = []
            per_step: List[int] = []
            for step in range(budget):
                reference_step(reference, reference_pml)
                count = state_census(reference)
                per_step.append(count)
                if count:
                    fired.append(step)
            row = {"case": case, "scale": label, "factor": scale,
                   "requirement": requirement,
                   "steps_censused": budget, "subnormal_words_per_step": per_step,
                   "first_step": fired[0] if fired else None,
                   "last_step": fired[-1] if fired else None,
                   "census_fired": bool(fired),
                   "verdict": ("REFUSED (subnormal precondition)" if fired
                               else "precondition holds")}
            rows.append(row)
            log(f"[precondition] {case:<26} {label:<16} scale={scale:g} "
                f"require={requirement} window=[{row['first_step']}, "
                f"{row['last_step']}] fired={bool(fired)} "
                f"({time.time() - started:.1f}s)")
            payload["legs"]["precondition"] = rows
            save(payload, out)
            if requirement == "clean":
                assert not fired, (
                    f"{case}/{label}: the census fired on a band it must not — a "
                    f"detector that refuses the physical band certifies nothing")
            elif requirement == "fire":
                assert fired, (
                    f"{case}/{label}: the census DID NOT FIRE on the scaled control. "
                    f"Every empty window this gate reports would then be a silence "
                    f"rather than a measurement")
                kit.assert_census_floor(max(per_step), f"{case}/{label} control")

    for case in ("real_y_periodic_even", "complex_y_periodic_even"):
        here = [row for row in rows if row["case"] == case]
        assert any(row["census_fired"] for row in here), (case, here)
        assert not next(row for row in here
                        if row["scale"] == "physical")["census_fired"], here
    payload["legs"]["precondition"] = rows
    payload["legs"]["precondition_finding"] = {
        "what": ("at 1e-30 the COMPLEX case enters the band at step 6 of 8 while the "
                 "REAL case stays clean for all 8, on the same grid shape and the "
                 "same number of passes"),
        "why_it_matters": ("this leg asserted 1e-30 CLEAN on both storages when it "
                           "was written, and the assertion failed. Band entry is a "
                           "RUN-and-WINDOW fact rather than a family fact, which is "
                           "exactly why every census in this gate reports "
                           "[first_step, last_step] and not a scalar"),
    }
    save(payload, out)


# ---------------------------------------------------------------------------
# LEG mutations — a leg that cannot fail certifies nothing
# ---------------------------------------------------------------------------

#: Both sub-steps, which is the default scope.
BOTH_SUB_STEPS: Tuple[str, ...] = ("step_B", "step_D")

#: The four signed-zero sign combinations, plus two mixed-sign finites so a planted
#: cell is not always a zero. Complex volumes take the pairs as written; real ones
#: take the real halves.
SIGNED_ZERO_PLANTS = np.array(
    [complex(0.0, 0.0), complex(-0.0, -0.0), complex(0.0, -0.0), complex(-0.0, 0.0),
     complex(1.5, -0.0), complex(-0.0, 1.5)], dtype=np.complex64)


def plant_signed_zeros(fields: Any, names: Sequence[str], offset: int = 0) -> int:
    """SPRINKLE the signed-zero class onto the SOURCE volumes a curl reads.

    THE CLASS HAS TO BE CONSTRUCTED OR IT CANNOT BE MEASURED, and that is not a
    formality here: two of the spellings this gate refutes differ from the shipped
    one ONLY on a zero operand. Measured while writing this leg, on the physical-band
    random state, ``complex_zero_cross_terms_folded`` reported CAUGHT 0/14 — the
    probe's 78/41472 for the same defect came from an EXHAUSTIVE signed-zero table,
    and random uniform data contains no exact zeros at all.

    The scatter is SPARSE and row-dependent so neighbouring rows still differ: a
    plant that made whole rows identical would turn other needles into measured
    nulls, which is a failure mode the folded-complex gate hit and recorded.
    """
    planted = 0
    for index, name in enumerate(names):
        volume = getattr(fields, name, None)
        if volume is None:
            continue
        nx, ny, nz = volume.shape
        for row in range(ny):
            for slot, value in enumerate(SIGNED_ZERO_PLANTS):
                volume[(row * 3 + slot * 5 + index + offset) % nx, row,
                       (row * 2 + slot * 7 + index) % nz] = (
                    value if volume.dtype == np.complex64 else value.real)
                planted += 1
    return planted


def source_mutations() -> Tuple[Tuple[str, Optional[bool], str, Optional[str],
                                      Tuple[str, ...], bool,
                                      Callable[[str, Any, Any, str], str]], ...]:
    """(label, must_catch, why, scope tag, sub-steps, plant-zeros, transform).

    Each transform receives (shipped source, fields, pml, sub_step) and returns the
    mutant. ``scope`` is a tag from :func:`tags` and ``sub-steps`` restricts the
    DIRECTION; anything outside either is not run and is recorded as skipped, because
    a structural null and a survived defect must never look the same in the artifact.

    THE SUB-STEP AXIS EXISTS BECAUSE ONE NEEDLE MEASURED IT INTO EXISTENCE. See
    ``fold_ghost_spelled_as_a_periodic_wrap``: written unscoped it reported CAUGHT
    3/28, and the 25 misses are a STRUCTURAL property of the composition rather than
    a defect surviving.
    """

    def after_cell_zero(source, fields, pml, sub_step):
        return move_beta_after(source, "at_")

    def after_both(source, fields, pml, sub_step):
        return move_beta_after(source, "last_")

    def drop_top_mask(source, fields, pml, sub_step):
        codes, _ = symmetry.folded_axis_kinds(fields.grid, pml)
        mask = symmetry.folded_top_plane_mask(
            codes, bool(SUB_STEPS[sub_step]["backward"]),
            zero=zero_literal(fields))
        return needle(source, mask, "    // MUTANT: top-plane mask dropped")

    def force_top_mask_on_metallic(source, fields, pml, sub_step):
        # The INVERSE arm. A MIRROR_METALLIC fold STORES up to big_corner, so its top
        # plane is OWNED and STEPPED — and on this family that plane CARRIES the beta
        # term. Emitting the periodic mask there deletes a real cell of real physics.
        codes, _ = symmetry.folded_axis_kinds(fields.grid, pml)
        periodic = tuple(MP if int(c) == MM else int(c) for c in codes)
        backward = bool(SUB_STEPS[sub_step]["backward"])
        zero = zero_literal(fields)
        return needle(source,
                      symmetry.folded_top_plane_mask(codes, backward, zero=zero),
                      symmetry.folded_top_plane_mask(periodic, backward, zero=zero))

    def drop_cell_zero_mask(source, fields, pml, sub_step):
        codes, _ = symmetry.folded_axis_kinds(fields.grid, pml)
        reduced = symmetry._reduced_codes(codes)
        mask = templates.ownership_mask(
            reduced, bool(SUB_STEPS[sub_step]["backward"]),
            zero=zero_literal(fields))
        return needle(source, mask, "    // MUTANT: cell-0 ownership mask dropped")

    def fold_ghost_wraps(source, fields, pml, sub_step):
        # FOLD DELTA (1), ISOLATED. Both mirror codes take the METALLIC ghost branch
        # because a mirror plane REFLECTS rather than repeating; spelling the folded
        # axis's ghost as a PERIODIC wrap reads the far end of the array instead.
        codes, _ = symmetry.folded_axis_kinds(fields.grid, pml)
        backward = bool(SUB_STEPS[sub_step]["backward"])
        axis = next(i for i, c in enumerate(codes)
                    if int(c) in symmetry.MIRROR_CODES)
        letter = "xyz"[axis]
        return needle(source,
                      templates.ghost(letter, symmetry.CODE_METALLIC, backward),
                      templates.ghost(letter, symmetry.CODE_PERIODIC, backward))

    def swap_beta_signs(source, fields, pml, sub_step):
        if "beta_plus" in source:
            return (source.replace("beta_plus * b", "__TMP__ * b")
                          .replace("beta_minus * a", "beta_plus * a")
                          .replace("__TMP__ * b", "beta_minus * b"))
        return (source.replace("float2(bpr, bpi), b", "float2(__T__), b")
                      .replace("float2(bmr, bmi), a", "float2(bpr, bpi), a")
                      .replace("float2(__T__), b", "float2(bmr, bmi), b"))

    def swap_multiply_operands(source, fields, pml, sub_step):
        return (source.replace("(beta_plus * b)", "(b * beta_plus)")
                      .replace("(beta_minus * a)", "(a * beta_minus)"))

    def negate_with_zero_minus(source, fields, pml, sub_step):
        return needle(source, "curl0 = curl0 - (beta_plus * b);",
                      "curl0 = curl0 + (0.0f - (beta_plus * b));")

    def swap_complex_orientation(source, fields, pml, sub_step):
        return (source.replace("c_mul(float2(bpr, bpi), b)",
                               "c_mul(b, float2(bpr, bpi))")
                      .replace("c_mul(float2(bmr, bmi), a)",
                               "c_mul(a, float2(bmr, bmi))"))

    def fold_zero_cross_terms(source, fields, pml, sub_step):
        return (source.replace("c_mul(float2(bpr, bpi), b)",
                               "float2(-(bpi * b.y), bpi * b.x)")
                      .replace("c_mul(float2(bmr, bmi), a)",
                               "float2(-(bmi * a.y), bmi * a.x)"))

    def rotate_the_beta_partner(source, fields, pml, sub_step):
        # The phase words are named per AXIS (`pxr`/`pxi`, ...), so the letter is read
        # off the emitted source rather than assumed to be x — a needle that hard-coded
        # the axis would silently miss on a grid phased anywhere else.
        letter = next((axis for axis in "xyz"
                       if f"float2(p{axis}r, p{axis}i)" in source), None)
        if letter is None:
            raise LookupError("no phase word pair appears in this source")
        return needle(
            source, "c_mul(float2(bpr, bpi), b)",
            f"c_mul(float2(bpr, bpi), c_mul(b, float2(p{letter}r, p{letter}i)))")

    return (
        ("beta_after_the_cell_zero_mask", True,
         "the array path adds the term at stepping.py:384-391 / :467-474 and masks "
         "at :397 / :479; moving it past the cell-0 arm writes a beta contribution "
         "into a cell the array path zeroes", None, BOTH_SUB_STEPS, False, after_cell_zero),
        ("beta_after_both_masks", True,
         "the SECOND mask is the fold's own contribution, and it is the whole reason "
         "this family exists rather than special_kz's curl being re-admitted",
         "mirror_periodic", BOTH_SUB_STEPS, False, after_both),
        ("top_plane_mask_dropped", True,
         "the folded PERIODIC top plane sits past MEEP's big_corner and the FILL "
         "writes it, not the curl; masking it is what makes the two agree, and on "
         "this family that plane carries the beta term",
         "mirror_periodic", BOTH_SUB_STEPS, False, drop_top_mask),
        ("top_plane_mask_forced_on_a_metallic_fold", True,
         "the INVERSE defect and the reason the matrix carries both terminations: a "
         "MIRROR_METALLIC fold STEPS its top plane, so masking it deletes a real "
         "cell — one that carries the beta term",
         "mirror_metallic", BOTH_SUB_STEPS, False,
         force_top_mask_on_metallic),
        ("cell_zero_ownership_mask_dropped", True,
         "the parent's mask, WIDENED by the fold from `== METALLIC` to "
         "`!= PERIODIC`: on a folded axis it exists only because both mirror codes "
         "reduce to METALLIC", None, BOTH_SUB_STEPS, False, drop_cell_zero_mask),
        ("fold_ghost_spelled_as_a_periodic_wrap", True,
         "FOLD DELTA (1) isolated. A mirror plane REFLECTS rather than repeating, so "
         "both mirror codes must take the METALLIC ghost branch; a periodic wrap "
         "reads the far end of the array into the mirror plane. THE SCOPE IS A "
         "MEASUREMENT, NOT A CONVENIENCE: written unscoped this row reported CAUGHT "
         "3/28, and the 25 misses are structural. On a folded PERIODIC axis BOTH ENDS "
         "ARE MASKED — the cell-0 arm at the low end and the top-plane arm at the "
         "high end — so a wrong ghost read is deleted before it can be seen, which is "
         "the two masks doing exactly what they are for. And on the BACKWARD sub-step "
         "the ghost sits at the LOW end, where the cell-0 mask reaches it on either "
         "termination. The one combination that leaves it visible is a MIRROR_METALLIC "
         "fold at step_B, and the three cases that carry it are the three that caught "
         "it", "mirror_metallic", ("step_B",), False, fold_ghost_wraps),
        ("beta_coefficient_signs_swapped", True,
         "the two coefficients are the +/- halves of one pair (stepping.py:798-799 / "
         ":784), so swapping them is the smallest wrong transcription that is not a "
         "magnitude error", None, BOTH_SUB_STEPS, False, swap_beta_signs),
        ("real_multiply_operands_swapped", False,
         "A DECLARED EQUIVALENCE, measured rather than believed: float multiplication "
         "is commutative, so the shipped order is transcription fidelity to "
         "stepping.py:811 and not a bit requirement", "real", BOTH_SUB_STEPS, False,
         swap_multiply_operands),
        ("negation_spelled_as_zero_minus", False,
         "UNOBSERVABLE ON THIS FAMILY'S GRIDS, and that is proved rather than "
         "assumed: `curl + (0.0f - x)` differs from `curl - x` in exactly one input "
         "pattern (curl a NEGATIVE zero, x a positive zero), and grid.beta is legal "
         "only on an effective-2-D grid whose invariant axis stores ONE cell — so "
         "`b - b_z` is +0.0 for every finite b and a float sum carrying a +0.0 addend "
         "is never -0.0. The spellings ARE different on this backend (measured "
         "0.0f - x misses 22/1728 on the probe's exhaustive table); this family "
         "cannot tell them apart, which is a different claim and the only true one",
         "real", BOTH_SUB_STEPS, True, negate_with_zero_minus),
        ("complex_coefficient_orientation_swapped", False,
         "A SCOPED equivalence. `c_mul(coefficient, partner)` vs "
         "`c_mul(partner, coefficient)` is a null ONLY while the coefficient's real "
         "word is an exact zero — with a GENERAL coefficient the probe measured "
         "1080/41472 at up to 8 ulps. The scope is CHECKED below rather than trusted",
         "complex", BOTH_SUB_STEPS, True,
         swap_complex_orientation),
        ("complex_zero_cross_terms_folded", False,
         "UNOBSERVABLE ON THIS FAMILY'S GRIDS, and the classification is a "
         "MEASUREMENT that corrected a prediction. The shortcut the orientation null "
         "invites — a zero real word does not make the cross terms FREE, it makes "
         "them EXACT — was armed as must_catch on the strength of the probe's "
         "78/41472, and it reported CAUGHT 0/14 both on random data and with 2,988 "
         "signed zeros planted into the source volumes. The reason is the SAME one "
         "the negation row carries: dropping the fma's zero addend can only change "
         "the SIGN OF A ZERO in the beta increment, and `curl - (+/-0.0)` is one word "
         "unless `curl` is a negative zero — which leg `spellings` MEASURES never "
         "happens on an effective-2-D grid. Half one (the spellings really do differ) "
         "and half two (the pattern is unreachable) are both measured there",
         "complex", BOTH_SUB_STEPS, True,
         fold_zero_cross_terms),
        ("beta_partner_bloch_rotated", True,
         "the beta partner is the UNSHIFTED SAME-CELL snapshot (stepping.py:321 / "
         ":408) and the Bloch rotation applies to SHIFTED operands only; rotating it "
         "on the wrapped lane is the defect reusing the centre registers prevents BY "
         "CONSTRUCTION", "general_phase", BOTH_SUB_STEPS, False,
         rotate_the_beta_partner),
    )


def leg_mutations(payload: Dict[str, Any], out: str) -> None:
    """Armed, launch-counted, three-valued. Caught-of-armed is reported per row."""
    harness = kit.MutationHarness(payload, out)

    for (label, must_catch, why, scope, sub_steps, plant,
         transform) in source_mutations():
        planted = 0
        missed = False
        ran = caught = launches = 0
        skipped: List[str] = []
        for case, _, _ in CASES:
            fields, pml = build(case)
            case_tags = tags(fields, pml)
            if scope is not None and scope not in case_tags:
                skipped.append(case)
                continue
            codes, _ = symmetry.folded_axis_kinds(fields.grid, pml)
            for sub_step in sub_steps:
                sources = tuple(SUB_STEPS[sub_step]["sources"])
                if plant:
                    planted += plant_signed_zeros(fields, sources)
                reference, reference_pml = build(case)
                if plant:
                    plant_signed_zeros(reference, sources)
                getattr(stepping, sub_step)(reference, reference_pml)
                shipped = shipped_source(fields, pml, sub_step)
                try:
                    mutant = transform(shipped, fields, pml, sub_step)
                except LookupError:
                    missed = True
                    continue
                if mutant == shipped:
                    missed = True
                    continue
                state = snapshot(fields, CURL_ARRAYS)
                residency = device.Residency()
                counter = kit.Counter(entry_point(fields, mutant))
                plan = from_arrays_curl(
                    fields, pml, sub_step, codes, residency,
                    functions={shaders.CONTRACT_OFF: counter})
                residency.sync_in()
                plan.run()
                residency.sync_out()
                launches += counter.launches
                ran += 1
                names = curl_names(sub_step)
                if sum(differing(getattr(fields, n), getattr(reference, n))
                       for n in names):
                    caught += 1
                restore(fields, state)
        harness.record(label, harness.verdict(missed, ran, launches, caught),
                       launches, caught, ran, must_catch, why,
                       extra={"scope": scope, "sub_steps": list(sub_steps),
                              "signed_zeros_planted": planted,
                              "skipped_cases": skipped})
        if plant:
            kit.assert_census_floor(
                planted, f"{label}: signed-zero plants (the spelling this row "
                         f"refutes differs ONLY on a zero operand, so an unplanted "
                         f"state would report it as a null)", floor=64)

    # THE ORIENTATION NULL'S SCOPE, CHECKED RATHER THAN TRUSTED. The equivalence
    # above holds only while the coefficient's REAL word is an exact zero. Reading
    # that word here is what stops the null being inherited by a future edit that
    # changes the coefficient.
    coefficient_words: List[Dict[str, Any]] = []
    for case, _, is_complex in CASES:
        if not is_complex:
            continue
        fields, pml = build(case)
        for sub_step in ("step_B", "step_D"):
            (plus_re, plus_im), (minus_re, minus_im) = \
                special_kz.beta_curl_coefficients(
                    fields.grid.beta, fields.grid.dt,
                    magnetic=(sub_step == "step_B"), complex_storage=True)
            entry = {"case": case, "sub_step": sub_step,
                     "beta_plus_real_word": float(plus_re),
                     "beta_minus_real_word": float(minus_re),
                     "purely_imaginary": bool(float(plus_re) == 0.0
                                              and float(minus_re) == 0.0)}
            coefficient_words.append(entry)
            assert entry["purely_imaginary"], (
                entry, "the beta coefficient's REAL word is not an exact zero, so "
                "`complex_coefficient_orientation_swapped` is no longer a null and "
                "the row above records a scope that no longer holds")
            assert float(plus_im) != 0.0 and float(minus_im) != 0.0, entry
    payload["legs"]["orientation_null_scope"] = {
        "why": ("the orientation swap is byte-invisible only while the coefficient's "
                "real word is an exact zero; with a general coefficient the probe "
                "measured 1080/41472 words at up to 8 ulps. This is the scope, read "
                "off the shipped coefficients rather than assumed"),
        "rows": coefficient_words,
    }
    save(payload, out)

    # HOST-SIDE NEEDLE 1: THE SUB-STEP CONJUGATION. The two coefficients are built
    # with `magnetic=(sub_step == "step_B")`, which is where the B/D asymmetry of
    # stepping.py:798-799 lives. Flipping it is a defect no SOURCE edit can express.
    missed = False
    ran = caught = launches = 0
    skipped_real: List[str] = []
    for case, _, complex_storage in CASES:
        # SCOPED TO THE COMPLEX ARM, AND THE SCOPE IS A MEASUREMENT. `magnetic`
        # reaches the value ONLY through the `* (1j if magnetic else -1j)` factor
        # (special_kz.beta_curl_coefficients, transcribing stepping.py:798-799), so on
        # REAL storage the two sub-steps take the SAME coefficient words and flipping
        # the flag is the identity. Measured unscoped this row reported CAUGHT 14/28
        # and the 14 misses were exactly the seven real cases: the B/D asymmetry of
        # the beta coefficient lives entirely in the complex arm.
        if not complex_storage:
            skipped_real.append(case)
            continue
        fields, pml = build(case)
        codes, _ = symmetry.folded_axis_kinds(fields.grid, pml)
        for sub_step in ("step_B", "step_D"):
            reference, reference_pml = build(case)
            getattr(stepping, sub_step)(reference, reference_pml)
            wrong = special_kz.beta_curl_coefficients(
                fields.grid.beta, fields.grid.dt,
                magnetic=(sub_step != "step_B"), complex_storage=complex_storage)
            state = snapshot(fields, CURL_ARRAYS)
            residency = device.Residency()
            counter = kit.Counter(entry_point(
                fields, shipped_source(fields, pml, sub_step)))
            plan = from_arrays_curl(fields, pml, sub_step, codes, residency,
                                    functions={shaders.CONTRACT_OFF: counter},
                                    beta_override=wrong)
            residency.sync_in()
            plan.run()
            residency.sync_out()
            launches += counter.launches
            ran += 1
            names = curl_names(sub_step)
            if sum(differing(getattr(fields, n), getattr(reference, n))
                   for n in names):
                caught += 1
            restore(fields, state)
    harness.record("beta_coefficients_from_the_other_sub_step",
                   harness.verdict(missed, ran, launches, caught),
                   launches, caught, ran, True,
                   "the beta pair is built with `magnetic=(sub_step == 'step_B')` "
                   "(stepping.py:798-799); the two sub-steps carry CONJUGATE "
                   "coefficients and no source edit can express getting that wrong. "
                   "SCOPED to the complex arm because the conjugation is the whole "
                   "of the flag's effect: under real storage both sub-steps take "
                   "identical words and the flip is the identity",
                   extra={"scope": "complex", "skipped_cases": skipped_real})

    # HOST-SIDE NEEDLE 2: THE FAMILY'S OWN VACUITY FLOOR. `has_beta=False` compiles
    # the term out. If that agreed with the array path on a beta RUN, the beta term
    # would be doing no work on these grids and every row above would be a statement
    # about the certified fold rather than about this composition.
    missed = False
    ran = caught = launches = 0
    for case, _, _ in CASES:
        fields, pml = build(case)
        codes, _ = symmetry.folded_axis_kinds(fields.grid, pml)
        for sub_step in ("step_B", "step_D"):
            reference, reference_pml = build(case)
            getattr(stepping, sub_step)(reference, reference_pml)
            state = snapshot(fields, CURL_ARRAYS)
            residency = device.Residency()
            counter = kit.Counter(entry_point(
                fields, shipped_source(fields, pml, sub_step, has_beta=False)))
            plan = from_arrays_curl(fields, pml, sub_step, codes, residency,
                                    functions={shaders.CONTRACT_OFF: counter},
                                    has_beta=False)
            residency.sync_in()
            plan.run()
            residency.sync_out()
            launches += counter.launches
            ran += 1
            names = curl_names(sub_step)
            if sum(differing(getattr(fields, n), getattr(reference, n))
                   for n in names):
                caught += 1
            restore(fields, state)
    harness.record("beta_term_compiled_out_on_a_beta_run",
                   harness.verdict(missed, ran, launches, caught),
                   launches, caught, ran, True,
                   "THIS FAMILY'S OWN VACUITY FLOOR. If the beta-free build agreed "
                   "with the array path on a beta RUN, the term would be doing no "
                   "work on these grids and every other row here would be a claim "
                   "about the certified fold rather than about this composition")

    # WHOLE-STEP NEEDLE 1: THE STALE MIRROR. On a walled case the two fills and the
    # wall clear run on the ARRAY PATH between device launches. Dropping the sync
    # back leaves the device holding bytes the host has moved on from — a smooth,
    # plausible, WRONG field rather than an error.
    missed = False
    ran = caught = launches = 0
    skipped = []
    first_steps: Dict[str, Optional[int]] = {}
    for case, _, _ in CASES:
        fields, pml = build(case)
        if "walled" not in tags(fields, pml):
            skipped.append(case)
            continue
        reference, reference_pml = build(case)
        residency = device.Residency()
        plans = composed_plans(fields, pml, residency)
        residency.sync_in()
        first: Optional[int] = None
        for step in range(6):
            reference_step(reference, reference_pml)
            walk_step(fields, pml, plans, residency, resync=False)
            total = sum(differing(getattr(fields, n), getattr(reference, n))
                        for n in STORED if getattr(fields, n, None) is not None)
            if total and first is None:
                first = step
        launches += sum(plan.launches for plan in plans.values()
                        if plan is not None)
        ran += 1
        first_steps[case] = first
        if first is not None:
            caught += 1
    harness.record("stale_mirror_host_pass_not_resynced",
                   harness.verdict(missed, ran, launches, caught),
                   launches, caught, ran, True,
                   "the residency invariant: a mirror is valid only while nothing "
                   "else writes the host array. On a walled grid BOTH parents' fill "
                   "predicates refuse by name, so three passes per half run on the "
                   "array path and the sync back is the only thing keeping the "
                   "device copy live",
                   extra={"scope": "walled", "skipped_cases": skipped,
                          "first_divergent_step_per_case": first_steps,
                          "budget": 6})

    # WHOLE-STEP NEEDLE 2: THE SEAM ORDER. `zero_metal_*` runs BETWEEN the two fill
    # passes and the far pass reads a plane the wall clear touches. This is the
    # ordering the parents' fill predicates refuse to assume, stated as a needle: it
    # is what makes their refusal a measured fact rather than a cautious guess.
    SEAM_ORDER = ("step_B", "fill_B_near", "fill_B_far", "zero_metal_B", "update_H",
                  "step_D", "fill_D_near", "fill_D_far", "zero_metal_D", "update_E")
    assert sorted(SEAM_ORDER) == sorted(DRIVER_ORDER), (
        "the seam order must be a PERMUTATION of the driver's own: a needle that "
        "also dropped or duplicated a pass would be measuring something else")
    missed = False
    ran = caught = launches = 0
    skipped = []
    first_steps = {}
    for case, _, _ in CASES:
        fields, pml = build(case)
        if "walled" not in tags(fields, pml):
            skipped.append(case)
            continue
        reference, reference_pml = build(case)
        residency = device.Residency()
        plans = composed_plans(fields, pml, residency)
        residency.sync_in()
        first = None
        for step in range(6):
            reference_step(reference, reference_pml)
            walk_step(fields, pml, plans, residency, order=SEAM_ORDER)
            total = sum(differing(getattr(fields, n), getattr(reference, n))
                        for n in STORED if getattr(fields, n, None) is not None)
            if total and first is None:
                first = step
        launches += sum(plan.launches for plan in plans.values()
                        if plan is not None)
        ran += 1
        first_steps[case] = first
        if first is not None:
            caught += 1
    harness.record("wall_clear_moved_after_the_far_fill",
                   harness.verdict(missed, ran, launches, caught),
                   launches, caught, ran, True,
                   "driver.py:3286 / :3301 put the wall clear BETWEEN the two fill "
                   "passes; the far pass images a stored row that includes cells the "
                   "clear touches, so running the two back to back is a different "
                   "answer. Every kernel here is the SHIPPED one and only the ORDER "
                   "moves",
                   extra={"scope": "walled", "skipped_cases": skipped,
                          "first_divergent_step_per_case": first_steps,
                          "budget": 6})

    # EMISSION REFUSAL. A mirror code carrying a Bloch flag must be refused at
    # EMISSION and not merely by the predicate: a mis-baked flag is a plane of wrong
    # values, not a crash, and a gate hands codes and flags straight in.
    refusals = 0
    for codes in ((symmetry.CODE_PERIODIC, MP, symmetry.CODE_PERIODIC),
                  (MM, symmetry.CODE_PERIODIC, symmetry.CODE_PERIODIC)):
        axis = 1 if codes[1] in symmetry.MIRROR_CODES else 0
        flags = tuple(1 if i == axis else 0 for i in range(3))
        try:
            folded_beta.folded_beta_bloch_curl_source(codes, False, flags, EXPANSION)
        except ValueError:
            refusals += 1
    metallic_refusals = 0
    try:
        folded_beta.folded_beta_bloch_curl_source(
            (symmetry.CODE_METALLIC,) * 3, False, (1, 0, 0), EXPANSION)
    except ValueError:
        metallic_refusals += 1
    assert refusals == 2 and metallic_refusals == 1, (refusals, metallic_refusals)
    payload["legs"]["emission_refusals"] = {
        "phased_folded_axis_refused": refusals,
        "phased_metallic_axis_refused": metallic_refusals,
        "why": ("clause 9 held at the LAST place it can be seen. "
                "driver._require_bloch_is_representable (driver.py:1047-1075) "
                "refuses the configuration outright, so a kernel cannot lift what the "
                "array path will not run"),
    }
    save(payload, out)


# ---------------------------------------------------------------------------
# LEG refusals — every named refusal, INCLUDING the two the engine itself raises
# ---------------------------------------------------------------------------

def leg_refusals(payload: Dict[str, Any], out: str) -> None:
    """A refusal by name beats a quiet pass — and a refusal nobody drove is prose.

    A configuration this leg cannot CONSTRUCT is a hard failure, never a note: the
    whole point is that each refusal is exercised on a grid the engine really builds.
    Both directions are checked wherever another family owns what this one declines,
    because a clause that refuses without an owner is a silent coverage LOSS.
    """
    rows: List[Dict[str, Any]] = []
    residency = device.Residency()
    started = time.time()

    def record(name: str, covered: bool, reasons: Sequence[str], expect: str,
               owner: Optional[bool] = None, owner_note: str = "") -> None:
        text = " | ".join(reasons)
        row = {"refusal": name, "covered": bool(covered), "expected": expect,
               "reasons": list(reasons), "owner_admits": owner,
               "owner_note": owner_note}
        rows.append(row)
        log(f"[refusals] {name:<46} covered={covered} owner_admits={owner} "
            f"({time.time() - started:.1f}s)")
        payload["legs"]["refusals"] = rows
        save(payload, out)
        assert not covered, row
        assert expect in text, (row, f"the refusal must NAME {expect!r}")
        if owner is not None:
            assert owner, (row, owner_note)

    # CLAUSE 5, both directions.
    flat_fields, flat_pml = matrix.flat(beta=0.33)
    record("unfolded_beta_run",
           *_verdict(folded_beta.folded_beta_composition_curl_coverage(
               flat_fields, flat_pml, "step_B", residency)),
           "no mirror plane is active",
           owner=special_kz.beta_pml_curl_coverage(
               flat_fields, flat_pml, "step_B", residency).covered,
           owner_note="special_kz's certified curl must own the unfolded beta run")

    # CLAUSE 12, both directions.
    zero_fields, zero_pml = matrix.folded(cross=CROSS)
    record("folded_run_at_beta_zero",
           *_verdict(folded_beta.folded_beta_composition_curl_coverage(
               zero_fields, zero_pml, "step_B", residency)),
           "grid.beta is zero",
           owner=symmetry.folded_composition_curl_coverage(
               zero_fields, zero_pml, "step_B", residency).covered,
           owner_note="symmetry's folded curl must own the beta = 0 folded run")

    # CLAUSE 2, both directions.
    real_fields, real_pml = build("real_y_periodic_even")
    cplx_fields, cplx_pml = build("complex_y_periodic_even")
    record("complex_arm_on_a_real_storage_run",
           *_verdict(folded_beta.folded_beta_composition_bloch_curl_coverage(
               real_fields, real_pml, "step_B", residency, probe=PROBE)),
           "storage is real float32",
           owner=folded_beta.folded_beta_composition_curl_coverage(
               real_fields, real_pml, "step_B", residency).covered,
           owner_note="the real arm must own it")
    record("real_arm_on_a_complex_storage_run",
           *_verdict(folded_beta.folded_beta_composition_curl_coverage(
               cplx_fields, cplx_pml, "step_B", residency)),
           "force_complex_fields=True",
           owner=folded_beta.folded_beta_composition_bloch_curl_coverage(
               cplx_fields, cplx_pml, "step_B", residency, probe=PROBE).covered,
           owner_note="the complex arm must own it")

    # The real arm's off-diagonal refusal, and the ENGINE'S OWN RAISE beside it.
    offdiag_fields, offdiag_pml = matrix.folded(cross=CROSS, beta=0.3,
                                                rows={"Ex": ("Ey",)})
    record("real_storage_offdiagonal_epsilon_with_beta",
           *_verdict(folded_beta.folded_beta_composition_curl_coverage(
               offdiag_fields, offdiag_pml, "step_B", residency)),
           "implicit-i trick no longer cancels")
    engine_raised = None
    try:
        stepping.step_B(offdiag_fields, offdiag_pml)
    except Exception as exc:  # noqa: BLE001 - the raise IS the measurement
        engine_raised = repr(exc)
    assert engine_raised is not None, (
        "stepping.step_B did NOT raise on real storage + off-diagonal + beta, so the "
        "predicate's stated reason (stepping.py:800-810 raises; MEEP "
        "fields.cpp:548-549 aborts) is no longer a fact about this tree")
    payload["legs"]["engine_raises"] = {"real_offdiag_beta": engine_raised}

    # Beta off an effective-2-D grid: the GRID ITSELF refuses at construction, and
    # the predicate restates the clause rather than inferring it from that guard —
    # which is what lets it refuse a hand-built stand-in the engine could not produce.
    grid_raised = None
    try:
        matrix.cart(beta=0.3)
    except Exception as exc:  # noqa: BLE001
        grid_raised = repr(exc)
    assert grid_raised is not None, (
        "a 3-D grid accepted beta, so the dimension clause is no longer restating a "
        "fact (MEEP fields.cpp:546-547)")
    payload["legs"]["engine_raises"]["beta_off_two_dimensions"] = grid_raised
    # THE CLAUSE IS DRIVEN ON A STAND-IN BECAUSE THE ENGINE CANNOT BUILD ITS SUBJECT,
    # and that is exactly why the clause exists: `Grid._resolve_beta` refuses beta off
    # an effective-2-D grid at CONSTRUCTION (the raise recorded just above), so a
    # predicate that trusted the constructor would admit a hand-built `fields`
    # stand-in the engine could never produce — which is what this gate's own
    # from-arrays route is. The proxy changes ONE attribute and forwards the rest.
    class _Dimensions:
        def __init__(self, grid: Any, dimensions: int) -> None:
            self._grid, self.dimensions = grid, dimensions

        def __getattr__(self, name: str) -> Any:
            return getattr(self._grid, name)

    stand_in_reasons = folded_beta._dimension_reasons(
        _Dimensions(real_fields.grid, 3))
    rows.append({"refusal": "beta_on_a_three_dimensional_grid",
                 "covered": not stand_in_reasons,
                 "expected": "effective-2-D grid beta requires",
                 "reasons": list(stand_in_reasons), "owner_admits": None,
                 "owner_note": "driven on a one-attribute proxy: the engine REFUSES "
                               "to construct the subject (see engine_raises), which "
                               "is why the clause is restated rather than inferred"})
    assert stand_in_reasons and "effective-2-D grid beta requires" in \
        " | ".join(stand_in_reasons), stand_in_reasons
    assert not folded_beta._dimension_reasons(real_fields.grid), (
        "the dimension clause fired on the family's own 2-D grid, so it refuses "
        "everything and measures nothing")

    # Conductivity, nonlinearity, BFAST, dispersion, and the missing probe.
    # THE CONDUCTIVITY CLAUSE IS PER SUB-STEP AND IS DRIVEN THAT WAY. An ELECTRIC
    # conductivity lands on Dx/Dy/Dz and a MAGNETIC one on Bx/By/Bz, so asking only
    # `step_B` about an electric one reports the clause as ADMITTING — which is what
    # this leg did when it was written, and is a false green rather than a finding.
    electric = matrix.conductive(build("real_y_periodic_even"), magnetic=False)
    record("an_electric_conductivity_on_step_D",
           *_verdict(folded_beta.folded_beta_composition_curl_coverage(
               electric[0], electric[1], "step_D", residency)),
           "a conductivity is installed",
           owner=folded_beta.folded_beta_composition_curl_coverage(
               electric[0], electric[1], "step_B", residency).covered,
           owner_note="an electric conductivity does not touch step_B's targets, so "
                      "refusing that sub-step too would be a coverage loss")
    magnetic = matrix.conductive(build("real_y_periodic_even"), magnetic=True)
    record("a_magnetic_conductivity_on_step_B",
           *_verdict(folded_beta.folded_beta_composition_curl_coverage(
               magnetic[0], magnetic[1], "step_B", residency)),
           "a conductivity is installed",
           owner=folded_beta.folded_beta_composition_curl_coverage(
               magnetic[0], magnetic[1], "step_D", residency).covered,
           owner_note="and the mirror image: a magnetic conductivity leaves step_D "
                      "admitted")
    nonlinear = matrix.nonlinear(build("real_y_periodic_even"))
    record("an_instantaneous_nonlinearity",
           *_verdict(folded_beta.folded_beta_composition_curl_coverage(
               nonlinear[0], nonlinear[1], "step_B", residency)),
           "chi2/chi3 is installed")
    dispersive = matrix.dispersive(build("real_y_periodic_even"))
    record("a_susceptibility_on_the_E_side_constitutive",
           *_verdict(folded_beta.folded_beta_constitutive_coverage(
               dispersive[0], dispersive[1], "E", residency)),
           "D - sum P",
           owner=folded_beta.folded_beta_constitutive_coverage(
               dispersive[0], dispersive[1], "H", residency).covered,
           owner_note="dispersion must be scoped to the E side; refusing update_H "
                      "too would be a coverage loss wearing a refusal's clothes")
    record("the_complex_arm_without_a_measured_expansion_artifact",
           *_verdict(folded_beta.folded_beta_composition_bloch_curl_coverage(
               cplx_fields, cplx_pml, "step_B", residency,
               probe={"backend": "not-numpy"})),
           "expansion probe")

    # Cylindrical, refused TWICE — the grid flag alone is not the inversion, because
    # `_mirror_phases` puts `(-1)**grid.m` into the SAME SLOT the mirror phase
    # occupies (stepping.py:2346-2366).
    cyl_fields, cyl_pml = matrix.cylindrical()
    cylindrical = folded_beta.folded_beta_composition_bloch_curl_coverage(
        cyl_fields, cyl_pml, "step_B", residency, probe=PROBE)
    text = " | ".join(cylindrical.reasons)
    rows.append({"refusal": "cylindrical_coordinates", "covered": cylindrical.covered,
                 "expected": "cylindrical (Dcyl)", "reasons": list(cylindrical.reasons),
                 "owner_admits": None,
                 "owner_note": "refused on BOTH the grid flag and the r = 0 axis"})
    assert not cylindrical.covered
    assert "cylindrical (Dcyl)" in text and "cylindrical r = 0 axis" in text, text

    # THE GAP THIS FAMILY MADE VISIBLE IS NOW CLOSED, and the seam is the fold.
    #
    # This block used to assert that the UNFOLDED complex beta constitutive was
    # still nobody's, by pinning special_kz's refusal message. special_kz restated
    # that predicate one level out on 2026-08-19 and now ADMITS the unfolded case,
    # so the old assertion became false the moment the gap closed — a gate pinning
    # the ABSENCE of a product fails exactly when the product arrives, which is
    # the wrong time to go red for the wrong reason.
    #
    # What is worth pinning instead is the separation: this family requires a
    # mirror plane and special_kz's requires there be none, so the two must never
    # BOTH admit. That single clause is the whole disjointness, and it is the same
    # pin test_metal_folded_beta.py uses.
    unfolded_complex, unfolded_complex_pml = matrix.flat(beta=0.33,
                                                         complex_storage=True)
    for side in ("H", "E"):
        theirs = special_kz.beta_run_complex_constitutive_coverage(
            unfolded_complex, unfolded_complex_pml, side, residency, PROBE)
        mine = folded_beta.folded_beta_complex_constitutive_coverage(
            unfolded_complex, unfolded_complex_pml, side, residency, PROBE)
        rows.append({"refusal": f"unfolded_complex_beta_constitutive_{side}",
                     "covered": mine.covered,
                     "expected": "DISJOINT: special_kz admits, this family refuses",
                     "reasons": list(mine.reasons),
                     "owner_admits": bool(theirs.covered),
                     "owner_note": "special_kz.beta_run_complex_constitutive owns the "
                                   "UNFOLDED case; this family owns the folded one"})
        assert not (mine.covered and theirs.covered), (
            f"both products admit the unfolded complex beta constitutive at {side}: "
            f"plan_step would pick by ordering, which is not a fail-closed composer")
        assert not mine.covered, " | ".join(mine.reasons)

    payload["legs"]["refusals"] = rows
    save(payload, out)
    kit.assert_census_floor(len(rows), "named refusals driven", floor=14)


def _verdict(coverage: Any) -> Tuple[bool, Tuple[str, ...]]:
    return bool(coverage.covered), tuple(coverage.reasons)


# ---------------------------------------------------------------------------
# LEG composition — disjointness, stated as inversions and measured per slot
# ---------------------------------------------------------------------------

ARITHMETIC_SLOTS: Tuple[str, ...] = ("step_B", "step_D", "update_H", "update_E")


def leg_composition(payload: Dict[str, Any], out: str) -> None:
    """On every arithmetic slot of every case, EXACTLY ONE arm admits — this one's.

    Two admitters is not a tie the composer breaks, it is an UNSELECTED slot that
    falls to the array path, so a foreign arm admitting here would be a silent
    coverage LOSS as well as a disjointness defect. The FILL slots are checked the
    other way: this family must admit NOTHING there, and a parent must, except on a
    walled grid where both parents refuse by name and the array path owns them.
    """
    rows: List[Dict[str, Any]] = []
    started = time.time()
    for label, _, _ in CASES:
        fields, pml = build(label)
        residency = device.Residency()
        admitted: Dict[str, List[str]] = {}
        gated: Dict[str, List[str]] = {}
        for slot in ARITHMETIC_SLOTS + ("fill_B", "fill_D"):
            here: List[str] = []
            skipped: List[str] = []
            for spec in arms.registered(slot):
                context = arms.StepContext(
                    fields=fields, pml=pml, residency=residency,
                    contract_variants=(shaders.CONTRACT_OFF,),
                    extra={"beta_probe": PROBE, "folded_complex_probe": FOLD_PROBE,
                           "probe": PROBE})
                if spec.gate is not None and not spec.gate(context):
                    skipped.append(spec.family)
                    continue
                try:
                    verdict = spec.coverage(context, slot)
                except Exception as exc:  # noqa: BLE001 - a raise IS the finding
                    raise AssertionError(
                        f"{spec.family} raised on {slot}: a predicate must return a "
                        f"NAMED REFUSAL, never raise into the composer") from exc
                if verdict.covered and spec.wired:
                    here.append(spec.family)
            admitted[slot] = here
            gated[slot] = skipped
        walled = "walled" in tags(fields, pml)
        expected = (folded_beta.FAMILY_COMPLEX if CASE_COMPLEX[label]
                    else folded_beta.FAMILY_REAL)
        rows.append({"case": label, "tags": list(tags(fields, pml)),
                     "admitted_by_slot": admitted, "gated_out_by_slot": gated,
                     "walled": walled, "expected_family": expected,
                     "arms_consulted": {slot: len(arms.registered(slot))
                                        for slot in admitted}})
        log(f"[composition] {label:<26} {admitted} ({time.time() - started:.1f}s)")
        payload["legs"]["composition"] = rows
        save(payload, out)
        for slot in ARITHMETIC_SLOTS:
            assert admitted[slot] == [expected], (
                f"{label}/{slot}: exactly one arm must admit and it must be this "
                f"family's; {admitted[slot]} leaves the slot UNSELECTED and it falls "
                f"to the array path", rows[-1])
        for slot in ("fill_B", "fill_D"):
            assert not any(name.startswith("folded_beta")
                           for name in admitted[slot]), (
                f"{label}/{slot}: this family registered a fill arm, which makes both "
                f"seam slots AMBIGUOUS and drops them to the array path — a coverage "
                f"LOSS dressed as completeness", rows[-1])
            if walled:
                assert admitted[slot] == [], (
                    f"{label}/{slot}: a walled grid's fill must be refused by BOTH "
                    f"parents (the far pass reads a plane the wall clear touches)",
                    rows[-1])
            else:
                assert len(admitted[slot]) == 1, (label, slot, rows[-1])

    # update_P is independent of folded-beta arithmetic and belongs to ADE alone.
    assert {spec.family for spec in arms.registered("update_P")} == {
        "ade_update_p"}
    assert matrix.UNREGISTERED_SLOTS == (), matrix.UNREGISTERED_SLOTS
    # THE TWO ROWS MOVED OUT OF UNCARRIED WHEN THIS FAMILY LANDED.
    for row in ("fold_real_2d_beta", "fold_complex_2d_beta"):
        assert row not in matrix.UNCARRIED, row
        pinned = dict((name, expected)
                      for name, _, expected, _ in matrix.MATRIX)[row]
        assert set(matrix.ARITHMETIC_SLOTS) <= set(pinned), (row, pinned)
    for label in (folded_beta.LABEL_REAL, folded_beta.LABEL_COMPLEX):
        assert label in matrix.EXPECTED_WINNERS, label
    payload["legs"]["floors"] = {
        "update_P_carries_no_arm": True,
        "rows_moved_out_of_uncarried": ["fold_real_2d_beta", "fold_complex_2d_beta"],
        "labels_in_expected_winners": [folded_beta.LABEL_REAL,
                                       folded_beta.LABEL_COMPLEX],
        "why": ("moving a row out of UNCARRIED is a DELIBERATE act, legitimate only "
                "in the change that makes the family real and never to make a red "
                "floor go green. This gate pins the post-move state so the move "
                "cannot be quietly undone"),
    }
    save(payload, out)


# ---------------------------------------------------------------------------
# LEG containment — Metal-only is never accepted on argument
# ---------------------------------------------------------------------------

#: The two Triton clauses that are facts about THAT platform rather than about a
#: configuration, factored out and NAMED. This is the same method the Triton coverage
#: round used ("its number is modulo that one clause"); the count of stripped clauses
#: is recorded per row so nothing is factored out silently.
HOST_FACT_CLAUSES: Tuple[str, ...] = (
    "not cupy",
    "expansion probe artifact is available for this backend",
)


def _triton_verdict(coverage: Any) -> Tuple[bool, List[str], List[str]]:
    kept = [reason for reason in coverage.reasons
            if not any(clause in reason for clause in HOST_FACT_CLAUSES)]
    stripped = [reason for reason in coverage.reasons if reason not in kept]
    return (not kept), kept, stripped


def leg_containment(payload: Dict[str, Any], out: str) -> None:
    """Per slot: METAL only, TRITON only, BOTH, or NEITHER — every slot named.

    A new Metal-only slot is not automatically a defect, but it is NEVER accepted on
    argument: it is measured with a whole-step byte comparison or refused. Every case
    here has a ``whole_step`` row, so any Metal-only slot this leg finds is backed by
    that evidence and the row says which.

    THE TRITON SIDE IS EVALUATED MODULO TWO HOST-FACT CLAUSES, named in
    :data:`HOST_FACT_CLAUSES` and counted per row. That is a CLAUSE-SHAPE comparison,
    not a runnable one: Triton cannot execute on this host at all, and pretending
    otherwise would be a bigger dishonesty than naming the factor.
    """
    rows: List[Dict[str, Any]] = []
    tally = {"both": 0, "metal_only": 0, "triton_only": 0, "neither": 0}
    started = time.time()
    for label, _, is_complex in CASES:
        fields, pml = build(label)
        residency = device.Residency()
        for slot in ARITHMETIC_SLOTS:
            if slot in ("step_B", "step_D"):
                if is_complex:
                    metal = folded_beta.folded_beta_composition_bloch_curl_coverage(
                        fields, pml, slot, residency, probe=PROBE)
                    triton = triton_folded_complex.folded_beta_bloch_pml_curl_coverage(
                        fields, pml, slot)
                else:
                    metal = folded_beta.folded_beta_composition_curl_coverage(
                        fields, pml, slot, residency)
                    triton = triton_folded_complex.folded_beta_pml_curl_coverage(
                        fields, pml, slot)
            else:
                side = "H" if slot == "update_H" else "E"
                if is_complex:
                    metal = folded_beta.folded_beta_complex_constitutive_coverage(
                        fields, pml, side, residency, probe=PROBE)
                    # The Triton track carries NO complex folded-beta constitutive
                    # either; its nearest arm is the folded complex one, which refuses
                    # beta by name. Asking it is what turns "nobody carries this" into
                    # a measurement rather than an assumption.
                    triton = triton_folded_complex.folded_complex_constitutive_coverage(
                        fields, pml, side)
                else:
                    metal = folded_beta.folded_beta_constitutive_coverage(
                        fields, pml, side, residency)
                    triton = triton_folded_complex.folded_beta_run_constitutive_coverage(
                        fields, pml, side)
            triton_covered, kept, stripped = _triton_verdict(triton)
            if metal.covered and triton_covered:
                verdict = "both"
            elif metal.covered:
                verdict = "metal_only"
            elif triton_covered:
                verdict = "triton_only"
            else:
                verdict = "neither"
            tally[verdict] += 1
            rows.append({"case": label, "slot": slot, "verdict": verdict,
                         "metal_covered": bool(metal.covered),
                         "metal_reasons": list(metal.reasons),
                         "triton_covered_modulo_host_facts": triton_covered,
                         "triton_reasons_kept": kept,
                         "triton_reasons_stripped": stripped,
                         "stripped_clause_count": len(stripped),
                         "byte_evidence": ("leg whole_step walks this case's "
                                           "complete ten-pass step against "
                                           "stepping.py")})
        log(f"[containment] {label:<26} {tally} ({time.time() - started:.1f}s)")
        payload["legs"]["containment"] = {"rows": rows, "tally": tally}
        save(payload, out)

    metal_only = [f"{row['case']}/{row['slot']}" for row in rows
                  if row["verdict"] == "metal_only"]
    payload["legs"]["containment"] = {
        "rows": rows, "tally": tally, "metal_only": metal_only,
        "host_fact_clauses_factored_out": list(HOST_FACT_CLAUSES),
        "stripped_clauses_total": sum(row["stripped_clause_count"] for row in rows),
        "why": ("the Triton predicates cannot run on this host — they refuse "
                "`array module is 'numpy', not cupy` before reading anything about "
                "the configuration — so the two are compared MODULO the named "
                "host-fact clauses and the count of what was factored out is "
                "recorded. Every slot is named either way. A Metal-only slot is not "
                "automatically a defect, but it is never accepted on argument: each "
                "one here is covered by this gate's own whole-step byte comparison"),
    }
    save(payload, out)
    kit.assert_census_floor(len(rows), "containment slots compared", floor=52)
    assert tally["neither"] == 0, (
        "a slot this gate is byte-certifying is admitted by NEITHER track's "
        "predicate, which means the composer would leave it on the array path",
        rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

LEGS: Tuple[Tuple[str, Callable[[Dict[str, Any], str], None]], ...] = (
    ("execution", leg_execution),
    ("curl", leg_curl),
    ("constitutive", leg_constitutive),
    ("identity", leg_identity),
    ("whole_step", leg_whole_step),
    ("position", leg_position),
    ("spellings", leg_spellings),
    ("guard", leg_guard),
    ("band", leg_band),
    ("precondition", leg_precondition),
    ("mutations", leg_mutations),
    ("refusals", leg_refusals),
    ("composition", leg_composition),
    ("containment", leg_containment),
)


def main() -> int:
    parser = kit.argument_parser(__doc__)
    arguments = parser.parse_args()
    started = time.time()
    out = arguments.out
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    payload: Dict[str, Any] = {"legs": {}, "environment": kit.environment_stamp()}

    reasons: List[str] = []
    try:
        import torch

        if not torch.backends.mps.is_available():
            reasons.append("no MPS device on this host")
    except Exception as exc:  # noqa: BLE001
        reasons.append(f"torch unavailable: {exc!r}")
    if EXPANSION is None:
        reasons.append(
            "no special_kz expansion probe artifact: this family's complex arm binds "
            "from that record deliberately, because its arithmetic IS that family's. "
            "Cut one with gate_metal_special_kz.py")
    if FOLD_EXPANSION is None:
        reasons.append(
            "no folded-complex expansion probe artifact: the complex whole-step legs "
            "run the PARENT'S fill, whose parity is a complex coefficient on the "
            "LEFT — a pattern special_kz's artifact never classified")
    if reasons:
        return kit.cannot_certify(payload, out, reasons)

    ran = kit.run_legs(LEGS, payload, out, kit.wanted_legs(arguments.legs))

    compared = 0
    certified = True
    for key in ("curl", "constitutive"):
        for row in payload["legs"].get(key, ()):
            compared += int(row["compared"])
            certified = certified and row["differing"] == 0
    for row in payload["legs"].get("identity", ()):
        compared += int(row["compared"])
        certified = (certified and row["differing_vs_stepping"] == 0
                     and row["differing_vs_certified_folded_plan"] == 0)
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
    armed = [row for row in mutations if row["must_catch"] is True]
    caught = [row for row in armed if row["caught"] == row["ran"] and row["ran"] > 0]

    return kit.summarize(
        payload, out,
        claim=("metal_kernels.folded_beta reproduces stepping.py word for word on a "
               "folded grid at nonzero grid.beta — both curl arms, both re-admitted "
               "constitutive bodies — per sub-step and per COMPLETE DRIVER STEP over "
               "the real ten-pass list, under a CHECKED subnormal-free precondition"),
        scope=("the case matrix in CASES: both fold terminations, both full-count "
               "parities, both declared parities, one and two folded axes at mixed "
               "phase, an X fold, both beta signs, a live zero_metal wall, both "
               "storages, and a Bloch phase on the unfolded axis at a general value "
               "and at the zone edge. UNFOLDED beta runs, folded runs at beta = 0, "
               "dispersion on update_E, conductivity, nonlinearity, BFAST, "
               "off-diagonal chi1inv, cylindrical coordinates and update_P are OUT "
               "OF SCOPE and refused BY NAME (leg refusals drives every one)"),
        stated_weakness=(
            "NO GENERATED-CODE AUDIT EXISTS ON THIS BACKEND: torch.mps.compile_shader "
            "exposes no disassembly, so this gate cannot refuse a compile whose "
            "emitted code violates the contraction policy nor establish that the "
            "pragma was obeyed. Every leg is BEHAVIOURAL. The arithmetic claims also "
            "ride a CHECKED subnormal-free precondition (legs band and precondition), "
            "and the containment comparison against the Triton track is a "
            "CLAUSE-SHAPE comparison modulo two named host-fact clauses, because "
            "those predicates cannot run on this host at all"),
        started=started, legs_run=ran, compared=compared, certified=certified,
        extra={"cases_refused_on_precondition": refused,
               "subnormal_windows": {
                   row["case"]: [row["subnormal_window"]["first_step"],
                                 row["subnormal_window"]["last_step"]]
                   for row in payload["legs"].get("whole_step", ())},
               "mutations_armed": len(armed),
               "mutations_caught_of_armed": f"{len(caught)}/{len(armed)}",
               "mutations_declared_null": sum(1 for row in mutations
                                              if row["must_catch"] is False),
               "containment": payload["legs"].get("containment", {}).get("tally")})


if __name__ == "__main__":
    from metal_gate_runner import run_current_measurement

    raise SystemExit(run_current_measurement(__file__, sys.argv[1:]))
