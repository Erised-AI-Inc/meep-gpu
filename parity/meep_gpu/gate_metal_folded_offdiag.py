"""BYTE GATE — the FOLDED off-diagonal electric constitutive sub-step, on Metal.

WHAT THIS CERTIFIES: that ``metal_kernels.folded_offdiag_update_e`` reproduces
``stepping.update_E`` (stepping.py:954, the ``elif offdiagonal:`` branch at :1001-1008)
WORD FOR WORD on a mirror-folded grid carrying off-diagonal chi1inv rows — per
SUB-STEP and per COMPLETE STEP — subject to a CHECKED subnormal-free precondition
that is reported as a WINDOW rather than as a count.

WHY THIS FAMILY NEEDS ITS OWN GATE RATHER THAN INHERITING TWO GREEN ONES. Both
halves are already certified alone (:mod:`.offdiag_update_e` and :mod:`.symmetry`),
and that is exactly the trap: :func:`symmetry.folded_constitutive_coverage`
re-admits the CERTIFIED constitutive body onto a folded grid on the single argument
that the sub-step READS NO NEIGHBOUR, so the fold's ghost rules cannot reach it. The
off-diagonal row product DOES read neighbours (stepping.py:1243-1249), so that
argument is unavailable and the composition is a question, not a corollary. The
fold reaches this sub-step in ONE role and not the other, and this gate measures
both halves:

* the PARTNER-axis down shift is LIVE on a fold — ``_shift_down`` is called WITH a
  component and the plane's parity (stepping.py:1243-1245), so a MIRROR partner axis
  takes stepping.py:1870-1873, ``parity * field[MIRROR_SOURCE_INDEX]``: a live
  parity-weighted interior plane, neither the periodic wrap nor the metallic zero.
  And ``_mask_metallic_wall_coupling`` ABSTAINS on it (stepping.py:1282), so it is
  not masked away afterwards;
* the OWN-axis up shift is NOT parity weighted on either termination —
  ``_offdiagonal_terms`` calls ``_shift_up`` with four arguments (stepping.py:
  1248-1249), so the folded-PERIODIC reflect branch cannot fire and both mirror
  terminations fall to an exact ``0.0``. That is a TRANSCRIPTION claim, and leg
  ``mutations`` arms it on two row sets so the catch is discriminating rather than
  coincidental.

WHAT THE MUTATION LEG COVERS, because a gate that cannot fail certifies nothing.
The five defects a mirror is prone to each have their own arm here, in this
family's own shape — there is no fill in this family, so each is expressed against
the in-kernel ghost REDIRECT rather than against a written plane:

  mirror the wrong half        the image is taken from the FAR half of the array
                               instead of from stored row MIRROR_SOURCE_INDEX;
  drop the sign flip           the negated ghost lane is emitted unnegated (and,
                               at the other plane parity, spuriously negated);
  shift the mirror plane       the image is taken from stored row 1 and from row 3;
  read the ghost before it is  ``update_E`` is walked BEFORE the mirror fill that
  written                      writes the plane it reads — a defect that exists at
                               WHOLE-STEP granularity and nowhere else;
  fold the wrong axis          the mirror code and the sign are emitted on an
                               UNFOLDED axis while the grid is folded elsewhere.

THE THREE MEASURED NULLS ARE NAMED RATHER THAN ROUNDED UP, and one of them turns a
claim into a check. The own-axis up wrap is a null with the ``Ey`` row dead (that
pair is what makes the catch discriminating); ``0.0f - x`` is a null on ordinary
normal data because the term's own sum renormalises, and the VALUE-CLASS leg is what
gives it reach; and the far-half mutant is a null where
``folded_offdiag_update_e.mirror_arm_is_reachable`` says the fold cannot change a
byte at all — fold X with the single live slot ``Ey <- Ez``, the measured
counterexample a row-level reachability test gets wrong. Without that row the SCOPE
on the corresponding catch would be indistinguishable from a convenient exclusion.

THE STATED WEAKNESS, and it is this certification's one gap against the Triton
twin's: ``torch.mps.compile_shader`` exposes NO DISASSEMBLY. This gate cannot refuse
a compile whose emitted code violates the policy and cannot establish that the
contraction guard was obeyed. Every leg here is BEHAVIOURAL — it catches a wrong
answer, not a wrong instruction. It is PARTLY offset and only partly: the "this arm
reduces to the certified body" claim, which the Triton twin can settle only with a
PTX read, is settled here by STRING EQUALITY (leg ``reduction``).

    python -u gate_metal_folded_offdiag.py --out results/.../gate.json
"""

from __future__ import annotations

import difflib
import os
import re
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
from meep_gpu.fields import Fields  # noqa: E402
from meep_gpu.grid import Grid, Mirror  # noqa: E402
from meep_gpu.pml import PML  # noqa: E402
from meep_gpu.metal_kernels import (  # noqa: E402
    device, preconditions, shaders, subnormal, symmetry,
)
from meep_gpu.metal_kernels import folded_offdiag_update_e as folded  # noqa: E402
from meep_gpu.metal_kernels import offdiag_update_e as offdiag  # noqa: E402
from meep_gpu.metal_kernels.device import compile_source  # noqa: E402

import metal_gate_kit as kit  # noqa: E402

log, save, differing, words = kit.log, kit.save, kit.differing, kit.words
needle = kit.needle

P = folded.CODE_PERIODIC
M = folded.CODE_METALLIC
MM = folded.CODE_MIRROR_METALLIC
MP = folded.CODE_MIRROR_PERIODIC

#: The six volumes this sub-step WRITES — the byte claim's compared set.
#: ``f_w_E*`` is the previous-source auxiliary and is STATE: a kernel right for one
#: launch and wrong forever after is identical in a single-launch leg.
COMPARED: Tuple[str, ...] = ("Ex", "Ey", "Ez", "f_w_Ex", "f_w_Ey", "f_w_Ez")

#: Everything the sub-step touches, saved and restored around the oracle.
TOUCHED: Tuple[str, ...] = COMPARED + ("Dx", "Dy", "Dz", "Hx", "Hy", "Hz")

#: Every volume a COMPLETE step touches. ``fu_*`` and ``f_w_*`` accumulate.
STORED: Tuple[str, ...] = (
    "Bx", "By", "Bz", "Dx", "Dy", "Dz", "Ex", "Ey", "Ez", "Hx", "Hy", "Hz",
    "fu_Bx", "fu_By", "fu_Bz", "fu_Dx", "fu_Dy", "fu_Dz",
    "f_w_Hx", "f_w_Hy", "f_w_Hz", "f_w_Ex", "f_w_Ey", "f_w_Ez")

#: The driver's own order, five passes per half (driver.py:3282-3306), with no
#: source and no pole: the two injections and ``update_P`` are absent.
DRIVER_ORDER: Tuple[str, ...] = (
    "step_B", "fill_B_near", "fill_B_far", "update_H",
    "step_D", "fill_D_near", "fill_D_far", "update_E")

#: The row set every case uses unless it names its own. Ex takes BOTH partners and
#: Ez takes one, so the matrix carries a component with two live slots, one with a
#: single slot, and one with none.
DEFAULT_ROWS: Dict[str, Tuple[str, ...]] = {"Ex": ("Ey", "Ez"), "Ez": ("Ex",)}

#: THE CASE MATRIX. Every axis is here because some mutation is reachable on one
#: value of it and a MEASURED NULL on the other:
#:
#:   TERMINATION    MIRROR_METALLIC is what all 19 corpus rows carry; MIRROR_PERIODIC
#:                  is the unmeasured twin where the far face is live and the
#:                  array-path finding in the family docstring bites;
#:   PHASE          the weight is ``-phase``, so an EVEN plane takes the NEGATED
#:                  source arm and an ODD plane reduces to the certified text. The
#:                  polarity inverts the reading a reader arrives with, so both are
#:                  carried and the parity mutation is armed at both;
#:   FOLDED AXES    one and three, because the sign is a per-axis source
#:                  specialisation and a three-axis fold at mixed phase is the only
#:                  shape where negated and un-negated lanes coexist in one kernel;
#:   FULL COUNT     ``_far_reflect_rows`` is ``stored - 2`` at an even full count and
#:                  ``stored - 3`` at an odd one. It cannot reach THIS kernel (the
#:                  own-axis up shift takes no reflect row) and the case is carried
#:                  precisely so that predicted null is measured rather than assumed;
#:   WALL           a DECLARED metallic wall on an UNFOLDED axis, beside a folded
#:                  one: the wall declaration and the boundary code come apart
#:                  exactly on a fold, and the plan refuses the pairing by name;
#:   ROW SET        a live slot that MISSES the fold (the measured counterexample:
#:                  fold X with ``Ey <- Ez`` is byte-identical to METALLIC), one that
#:                  CROSSES it, and all six slots live at once;
#:   DIMENSIONALITY a claim is only as strong as the words it moved.
CASES: Tuple[Tuple[str, Dict[str, Any]], ...] = (
    ("2d_fold_Y_even_periodic", dict(axes="Y", phases=(1,))),
    ("2d_fold_Y_odd_periodic", dict(axes="Y", phases=(-1,))),
    ("2d_fold_Y_even_metallic", dict(axes="Y", phases=(1,),
                                     boundaries={"y": "metallic"})),
    ("2d_fold_Y_odd_metallic", dict(axes="Y", phases=(-1,),
                                    boundaries={"y": "metallic"})),
    ("2d_fold_X_even", dict(axes="X", phases=(1,))),
    ("2d_fold_XY_mixed", dict(axes="XY", phases=(1, -1))),
    ("2d_fold_XY_both_even", dict(axes="XY", phases=(1, 1))),
    ("2d_fold_Y_odd_full_count", dict(axes="Y", phases=(1,), extent=2.1)),
    ("2d_fold_Y_with_metallic_wall", dict(axes="Y", phases=(1,),
                                          boundaries={"x": "metallic",
                                                      "y": "metallic"})),
    ("3d_fold_Z_even", dict(axes="Z", phases=(1,), dims=3, extent=1.4)),
    ("3d_fold_XYZ_mixed", dict(axes="XYZ", phases=(1, -1, 1), dims=3, extent=1.4)),
    ("2d_fold_X_slot_misses_the_fold", dict(axes="X", phases=(1,),
                                            rows={"Ey": ("Ez",)})),
    ("2d_fold_X_slot_crosses_the_fold", dict(axes="X", phases=(1,),
                                             rows={"Ey": ("Ex",)})),
    ("2d_fold_Y_all_six_slots", dict(axes="Y", phases=(1,),
                                     rows={"Ex": ("Ey", "Ez"),
                                           "Ey": ("Ez", "Ex"),
                                           "Ez": ("Ex", "Ey")})),
)

#: The whole-step subset. Every case here must also compose with the FOLDED CURL
#: and the MIRROR FILL, which is what makes the ghost-ordering defect reachable.
WHOLE_STEP_CASES: Tuple[Tuple[str, Dict[str, Any]], ...] = (
    ("2d_fold_Y_even_periodic", dict(axes="Y", phases=(1,))),
    ("2d_fold_Y_odd_periodic", dict(axes="Y", phases=(-1,))),
    ("2d_fold_Y_even_metallic", dict(axes="Y", phases=(1,),
                                     boundaries={"y": "metallic"})),
    ("2d_fold_XY_mixed", dict(axes="XY", phases=(1, -1))),
    ("2d_fold_Y_odd_full_count", dict(axes="Y", phases=(1,), extent=2.1)),
    ("3d_fold_Z_even", dict(axes="Z", phases=(1,), dims=3, extent=1.4)),
)


# ---------------------------------------------------------------------------
# The builder — this gate's own, and the reason is measured
# ---------------------------------------------------------------------------

def build(axes: str = "Y", phases: Sequence[int] = (1,), boundaries: Any = None,
          dims: int = 2, extent: float = 2.0,
          rows: Optional[Dict[str, Sequence[str]]] = None,
          seed: int = 4) -> Tuple[Any, Any]:
    """A folded grid with off-diagonal rows installed through the PUBLIC installer.

    NOT ``matrix.folded``, and the difference is load bearing rather than stylistic:
    that builder installs a UNIFORM inverse epsilon, and a uniform coefficient makes
    a whole class of defect invisible — anything that hoists the row multiply out
    from BETWEEN the two shifts reads the same number either way. Here both the
    diagonal inverse epsilon and the off-diagonal rows vary in space.

    DETERMINISTIC IN ``seed``, so the whole-step leg can build the reference a
    SECOND TIME rather than deep-copying: ``Grid`` holds the array module itself and
    a deepcopy raises.

    EVERY VOLUME IS FILLED WITH PHYSICAL-BAND VALUES, and that is the first thing
    this gate establishes rather than an afterthought: zero-init is a FIXED POINT of
    this sub-step (``src = g*u + 0``, ``f += kps*src - kms*prev`` all vanish), so a
    no-op agreeing with a no-op would be trivially identical and would certify
    nothing.
    """
    size = [1.6, 1.6, 0.0] if dims == 2 else [1.3, 1.2, 1.1]
    for name in axes:
        size["XYZ".index(name)] = extent
    grid = Grid(resolution=10.0, cell_size=tuple(size), dimensions=dims,
                courant=0.35,
                symmetry=tuple(Mirror(name, int(phase))
                               for name, phase in zip(axes, phases)),
                boundaries=boundaries, xp=np)
    fields = Fields(grid=grid, force_complex_fields=False)
    shape = grid.shape
    rng = np.random.default_rng(seed)
    epsilon = (1.45 + 0.3 * rng.random(shape)).astype(np.float32)
    inverse = (np.float32(1.0) / epsilon).astype(np.float32)
    built = {row: {partner: (0.03 * rng.standard_normal(shape)).astype(np.float32)
                   for partner in partners}
             for row, partners in (rows or DEFAULT_ROWS).items()}
    fields.set_epsilon_volumes({c: epsilon for c in ("Ex", "Ey", "Ez")},
                               {c: inverse for c in ("Ex", "Ey", "Ez")}, built)
    fields.enable_pml_storage()
    folded_indices = {"XYZ".index(name) for name in axes}
    thickness = []
    for index in range(3):
        if grid.shape[index] < 6:
            thickness.append((0, 0))
        elif index in folded_indices:
            thickness.append((0, 2))
        else:
            thickness.append((2, 2))
    pml = PML(grid=grid, thickness=tuple(thickness))
    for name in TOUCHED:
        getattr(fields, name)[...] = (rng.standard_normal(shape) * 0.37
                                      ).astype(np.float32)
    return fields, pml


def tags(fields: Any, pml: Any) -> Tuple[str, ...]:
    """What a case IS, derived from the ENGINE rather than typed beside the row.

    A mutation scopes itself with these. Typing them into the matrix would put the
    fold's single point of failure in a second place, and a scoping that disagreed
    with the grid would silently turn a reachable defect into a NEEDLE-MISSED row.
    """
    codes, _ = symmetry.folded_axis_kinds(fields.grid, pml)
    codes = tuple(int(code) for code in codes)
    negate = folded.negated_axes(codes, folded.mirror_ghost_weights(fields.grid))
    found: List[str] = []
    found.append("mirror_periodic" if MP in codes else "mirror_metallic_only")
    folded_axes = [a for a, code in enumerate(codes) if code in folded.MIRROR_CODES]
    found.append("multi_axis" if len(folded_axes) > 1 else "one_axis")
    found.append("negated" if any(negate) else "un_negated")
    if any(negate) and not all(negate[a] for a in folded_axes):
        found.append("mixed_sign")
    found.append("three_d" if int(fields.grid.shape[2]) > 1 else "two_d")
    if any(offdiag.wall_mask_axes(fields.grid)):
        found.append("declared_wall")
    # THE REACHABILITY QUESTION IS ASKED AT SLOT LEVEL, and the family's own helper
    # is what answers it: the mirror rule enters this sub-step ONLY through the DOWN
    # shift of the PARTNER axis (stepping.py:1243-1245), so a fold is byte-visible
    # iff some SURVIVING row slot takes that axis's field AS ITS PARTNER. A row-level
    # test ("some live row is not Ex") gets this wrong on a measured counterexample.
    reachable, _notes = folded.mirror_arm_is_reachable(fields.grid, fields)
    found.append("ghost_reachable" if reachable else "ghost_unreachable")
    rows = offdiag.row_volumes_for(fields)
    if rows[0] is not None or rows[1] is not None:
        found.append("component0_live")
    return tuple(found)


def specialisation(fields: Any, pml: Any) -> Tuple[Tuple[int, ...], ...]:
    """The (row_mask, codes, walls, negate) quadruple the SHIPPED plan compiles."""
    codes, _ = symmetry.folded_axis_kinds(fields.grid, pml)
    weights = folded.mirror_ghost_weights(fields.grid)
    walls = offdiag.wall_mask_axes(fields.grid)
    negate = folded.negated_axes(codes, weights)
    rows = offdiag.row_volumes_for(fields)
    row_mask = tuple(int(value is not None) for value in rows)
    return (row_mask, tuple(int(c) for c in codes), tuple(int(w) for w in walls),
            tuple(int(n) for n in negate))


# ---------------------------------------------------------------------------
# Snapshots, the oracle and the device route
# ---------------------------------------------------------------------------

def snapshot(fields: Any, names: Sequence[str] = TOUCHED) -> Dict[str, Any]:
    return {name: np.array(getattr(fields, name), copy=True) for name in names
            if getattr(fields, name, None) is not None}


def restore(fields: Any, state: Dict[str, Any]) -> None:
    for name, array in state.items():
        getattr(fields, name)[...] = array


def oracle(fields: Any, pml: Any) -> Tuple[Dict[str, Any], Dict[str, Any], int]:
    """Run the ARRAY PATH's ``update_E``, capture what it wrote, put the state back.

    Returns ``(before, after, moved)``. ``moved`` is the vacuity floor every leg
    asserts before it reads a single divergence count.
    """
    before = snapshot(fields)
    stepping.update_E(fields, pml)
    after = snapshot(fields, COMPARED)
    restore(fields, before)
    moved = sum(differing(before[name], after[name]) for name in COMPARED)
    return before, after, moved


def compared_words(state: Dict[str, Any]) -> int:
    return sum(int(words(array).size) for array in state.values())


def divergence(fields: Any, after: Dict[str, Any]) -> Dict[str, int]:
    return {name: differing(getattr(fields, name), array)
            for name, array in after.items()}


def run_shipped(fields: Any, pml: Any) -> Any:
    """Launch the family through the ENGINE ROUTE — the plan the composer builds."""
    residency = device.Residency()
    plan = folded.plan_folded_offdiag_constitutive(fields, pml, residency)
    if plan is None:
        raise AssertionError(folded.folded_offdiag_constitutive_coverage(
            fields, pml, residency).reasons)
    plan.run()
    residency.sync_out()
    assert not residency.verify()
    return plan


def run_mutant(fields: Any, pml: Any, source: str) -> Any:
    """The SHIPPED plan with ONE compiled body swapped — same bindings, same launch.

    Everything except the emitted text is the engine's own: the mirrors, the
    argument tuple, the launch path and the launch counter. A mutation leg that
    rebuilt the plan around the mutant would be measuring a second object.
    """
    residency = device.Residency()
    plan = folded.plan_folded_offdiag_constitutive(fields, pml, residency)
    if plan is None:
        raise AssertionError(folded.folded_offdiag_constitutive_coverage(
            fields, pml, residency).reasons)
    plan._functions = {shaders.CONTRACT_OFF:
                       compile_source(source).offdiag_constitutive_step}
    plan.run()
    residency.sync_out()
    return plan


# ---------------------------------------------------------------------------
# The precondition, censused as a WINDOW
# ---------------------------------------------------------------------------

def census_window(fields: Any, results: Optional[Dict[str, Any]],
                  codes: Sequence[int], weights: Sequence[float],
                  step: int) -> Any:
    """One step's census over the operands, the RESULTS and the fold's intermediate.

    THE INTERMEDIATE IS NOT OPTIONAL HERE. The fold puts a deep-PML plane on the far
    face — exactly where tiny magnitudes live — and the ghost lane forms
    ``g[face 0] + w * g[row MIRROR_SOURCE_INDEX]`` on the mirror plane. A
    whole-volume aggregate cannot be moved by a 1e-40 boundary plane, so the
    intermediate is censused on THOSE PLANES specifically, reconstructed on the host
    in the kernel's own operand order. That is weaker than reading the device's
    registers and this gate says so; it is strictly stronger than censusing only
    what was stored.
    """
    window = preconditions.SubnormalWindow(
        first_step=step, last_step=step,
        per_array_words=1, per_intermediate_words=1)
    for name in ("Dx", "Dy", "Dz", "Ex", "Ey", "Ez",
                 "f_w_Ex", "f_w_Ey", "f_w_Ez"):
        window.observe(name, getattr(fields, name), step=step)
    for name in ("Ex", "Ey", "Ez"):
        window.observe("inv_eps_" + name, fields.inverse_epsilon_for(name),
                       step=step)
    for (row, partner), value in zip(folded.ROW_SLOTS,
                                     offdiag.row_volumes_for(fields)):
        if value is not None:
            window.observe(f"chi1inv_offdiag:{row}:{partner}", value, step=step)
    for name, array in (results or {}).items():
        window.observe("result:" + name, array, step=step)
    for axis, code in enumerate(codes):
        if int(code) not in folded.MIRROR_CODES:
            continue
        weight = np.float32(weights[axis])
        for name in ("Dx", "Dy", "Dz"):
            volume = getattr(fields, name)
            face: List[Any] = [slice(None)] * 3
            face[axis] = 0
            source: List[Any] = [slice(None)] * 3
            source[axis] = folded.MIRROR_SOURCE_INDEX
            pair = (volume[tuple(face)]
                    + weight * volume[tuple(source)]).astype(np.float32)
            window.observe_intermediate(f"ghost_pair[{axis}]{name}", pair, step=step)
    return window


def window_row(window: Any, fired: bool) -> Dict[str, Any]:
    report = window.report()
    return {"window": report["window"], "subnormal_words": report["subnormal_words"],
            "observed_words": report["observed_words"], "clean": report["clean"],
            "vacuous": report["vacuous"], "vacuity_reasons": report["vacuity_reasons"],
            "fired": fired}


# ---------------------------------------------------------------------------
# LEG execution — provenance, the bound arm, and the compile sweep
# ---------------------------------------------------------------------------

#: The four BC triples the compile sweep drives: the corpus's two (both folded
#: METALLIC, the shape all 19 rows carry), the unmeasured PERIODIC twin, and an
#: all-mirror triple. THE ENUMERATION IS WIDER THAN THE REACHABLE SET and this
#: artifact must not be read as claiming otherwise: the R01..R22 row-mask pattern the
#: 19 corpus rows drive is NOT measured anywhere — the predicate-coverage battery
#: records ``has_offdiagonal_epsilon`` and not the surviving slot set — so the sweep
#: compiles EVERY row mask rather than the corpus's presumed few.
SWEEP_TRIPLES: Tuple[Tuple[int, int, int], ...] = (
    (M, MM, P), (MM, MM, P), (P, MP, P), (MM, MM, MM))


def leg_execution(payload: Dict[str, Any], out: str) -> None:
    """What was hashed, what arm is bound, and does every specialisation BUILD.

    A specialisation that fails to COMPILE is a crash at plan time on a
    configuration nobody swept, so the sweep is over every row mask on four code
    triples in BOTH contraction modes — 63 x 4 x 2 = 504 shaders.
    """
    sources = {f"{label}": folded.folded_offdiag_source(*specialisation(*build(**kw)))
               for label, kw in CASES}
    kit.provenance(os.path.dirname(out), {
        "folded_offdiag_update_e.py": os.path.join(
            API_ROOT, "meep_gpu/metal_kernels/folded_offdiag_update_e.py"),
        "offdiag_update_e.py": os.path.join(
            API_ROOT, "meep_gpu/metal_kernels/offdiag_update_e.py"),
        "symmetry.py": os.path.join(API_ROOT, "meep_gpu/metal_kernels/symmetry.py"),
        "templates.py": os.path.join(API_ROOT, "meep_gpu/metal_kernels/templates.py"),
        "shaders.py": os.path.join(API_ROOT, "meep_gpu/metal_kernels/shaders.py"),
        "preconditions.py": os.path.join(
            API_ROOT, "meep_gpu/metal_kernels/preconditions.py"),
        "stepping.py": os.path.join(API_ROOT, "meep_gpu/stepping.py"),
        "gate": os.path.abspath(__file__),
    }, kernel_sources=sources, name="provenance_gate.json")

    built = 0
    for mode in shaders.CONTRACT_MODES:
        for mask in range(1, 1 << len(folded.ROW_SLOTS)):
            row_mask = tuple((mask >> bit) & 1
                             for bit in range(len(folded.ROW_SLOTS)))
            for codes in SWEEP_TRIPLES:
                walls = tuple(int(code == M) for code in codes)
                negate = tuple(int(code in folded.MIRROR_CODES) for code in codes)
                compile_source(folded.folded_offdiag_source(
                    row_mask, codes, walls, negate, mode))
                built += 1
    kit.assert_moved(built, "no specialisation compiled",
                     floor=2 * 63 * len(SWEEP_TRIPLES))

    arm = folded.ARM
    payload["legs"]["execution"] = {
        "environment": ENVIRONMENT,
        "family": folded.FAMILY,
        "slot": folded.SLOT,
        "arm_label": folded.LABEL,
        "arm_wired": bool(getattr(arm, "wired", False)),
        "bindings": folded.BINDING_COUNT,
        "corpus_digest": folded.corpus_digest(),
        "compiled": built,
        "launched_specialisations": {
            label: list(specialisation(*build(**kw))) for label, kw in CASES},
        "cases": [label for label, _ in CASES],
        "no_generated_code_audit": (
            "torch.mps.compile_shader exposes no disassembly: this gate cannot "
            "refuse a compile whose emitted code violates the policy, nor establish "
            "that the contraction guard was obeyed. Every leg is behavioural."),
    }
    save(payload, out)
    log(f"[execution] family={folded.FAMILY} slot={folded.SLOT} "
        f"bindings={folded.BINDING_COUNT} compiled={built}")


# ---------------------------------------------------------------------------
# LEG reduction — the claim the Triton twin needs a PTX read for
# ---------------------------------------------------------------------------

def leg_reduction(payload: Dict[str, Any], out: str) -> None:
    """An UNFOLDED triple must emit the CERTIFIED string, character for character.

    This is what makes "the fold adds one ghost line and one negated lane, and
    nothing else" a property of the code rather than of a comment. It is not a byte
    comparison of an output and contributes nothing to the certified total; it is
    recorded because it is the one place this port is AHEAD of the Triton twin,
    which can only settle the analogous claim with a PTX-verified-different binary.
    """
    equal = 0
    for mask in range(1, 1 << len(folded.ROW_SLOTS)):
        row_mask = tuple((mask >> bit) & 1 for bit in range(len(folded.ROW_SLOTS)))
        for codes in ((P, P, P), (M, P, P), (P, M, P), (P, P, M),
                      (M, M, P), (M, P, M), (P, M, M), (M, M, M)):
            walls = tuple(int(code == M) for code in codes)
            ours = folded.folded_offdiag_source(row_mask, codes, walls, (0, 0, 0))
            theirs = offdiag.offdiag_source(row_mask, codes, walls)
            assert ours == theirs, (row_mask, codes, "the unfolded emission DIFFERS "
                                    "from the certified source")
            equal += 1

    # THE FOLDED DIFF, stated as a number rather than as a description: the folded
    # source against the CERTIFIED source for the same grid with the mirror axis
    # declared METALLIC. Every differing line must be the ghost redirect, the
    # up-validity line beside it, or a lane whose sign moved.
    diffs: List[Dict[str, Any]] = []
    for label, kwargs in CASES:
        fields, pml = build(**kwargs)
        row_mask, codes, walls, negate = specialisation(fields, pml)
        unfolded = tuple(M if code in folded.MIRROR_CODES else code
                         for code in codes)
        ours = folded.folded_offdiag_source(row_mask, codes, walls, negate)
        base = offdiag.offdiag_source(row_mask, unfolded, walls)
        changed = [line for line in difflib.unified_diff(
            base.splitlines(), ours.splitlines(), lineterm="", n=0)
            if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))]
        # EVERY CHANGED LINE MUST BE ONE OF EXACTLY TWO THINGS, and the check is
        # spelled from the emitter's own name tables rather than from a pattern a
        # reader guessed: a line of the GHOST BLOCK of a folded axis (its down index,
        # its down-validity flag or its up-validity flag), or a line carrying a
        # `dn_`/`cn_` temporary — the hoist that lets the sign apply to the LOADED
        # value, and the lanes that read it. Anything else means the fold reached
        # arithmetic it has no business touching.
        ghost_names = tuple(
            name for axis in folded_axes_of(codes)
            for name in offdiag._TWO_WAY_NAMES["xyz"[axis]][:4])
        unexpected = [line for line in changed
                      if "dn_" not in line and "cn_" not in line
                      and not any(re.search(rf"\b{name}\b", line)
                                  for name in ghost_names)]
        assert not unexpected, (label, "the fold changed a line that is neither the "
                                "ghost block nor a ghosted lane", unexpected)
        diffs.append({"case": label, "changed_lines": len(changed),
                      "negate": list(negate), "codes": list(codes)})
        log(f"[reduction] {label:<32} folded diff = {len(changed)} lines "
            f"(negate={negate})")
    payload["legs"]["reduction"] = {
        "unfolded_sources_equal_to_certified": equal,
        "unfolded_sources_differing": 0,
        "folded_diff": diffs,
        "why": ("an unfolded code triple must emit offdiag_update_e.offdiag_source's "
                "string EXACTLY; the folded diff against the same grid declared "
                "METALLIC must be the ghost redirect plus the sign lanes and nothing "
                "else"),
    }
    save(payload, out)
    log(f"[reduction] {equal} unfolded (row_mask, codes) pairs emit the certified "
        f"source character for character, 0 differing")


# ---------------------------------------------------------------------------
# LEG byte — the family's central claim, per SUB-STEP
# ---------------------------------------------------------------------------

def leg_byte(payload: Dict[str, Any], out: str) -> None:
    """``update_E`` on the device against ``stepping.update_E``, uint32 word equality.

    Every case establishes NON-VACUITY FIRST: the array path must have moved words,
    because zero-init is a fixed point of this sub-step and a no-op agreeing with a
    no-op is trivially identical. The subnormal precondition is censused per case
    and reported as a WINDOW; a case that enters the band is REFUSED BY NAME and its
    comparisons are withheld from the certified total.
    """
    rows: List[Dict[str, Any]] = []
    for label, kwargs in CASES:
        fields, pml = build(**kwargs)
        before, after, moved = oracle(fields, pml)
        kit.assert_moved(moved, f"{label} the array path barely moved", floor=64)

        row_mask, codes, walls, negate = specialisation(fields, pml)
        weights = folded.mirror_ghost_weights(fields.grid)
        window = census_window(fields, after, codes, weights, step=0)
        report = window.report()
        assert not report["vacuous"], (label, report["vacuity_reasons"])
        refused = not report["clean"]

        plan = run_shipped(fields, pml)
        assert plan.launches == 1, (label, plan.launches)
        per = divergence(fields, after)
        total = sum(per.values())
        rows.append({
            "case": label, "tags": tags(fields, pml),
            "shape": list(int(n) for n in fields.grid.shape),
            "row_mask": list(row_mask), "codes": list(codes),
            "walls": list(walls), "negate": list(negate),
            "ghost_weights": list(weights),
            "moved": moved, "compared": 0 if refused else compared_words(after),
            "differing": total, "per_target": per,
            "subnormal": window_row(window, refused),
            "refused": refused,
            "refusal": (f"{label}: REFUSED (subnormal precondition) — "
                        f"{report['subnormal_words']} of {report['observed_words']} "
                        f"censused words are in the float32 subnormal band; on MPS "
                        f"the flush is native and has no lever, so byte-identity is "
                        f"not claimable here") if refused else None})
        log(f"[byte] {label:<32} moved={moved:<6} compared={compared_words(after):<7} "
            f"differing={total} negate={negate} "
            f"subnormal={report['subnormal_words']}/{report['observed_words']}"
            f"{' REFUSED' if refused else ''}")
        restore(fields, before)
        payload["legs"]["byte"] = rows
        save(payload, out)


# ---------------------------------------------------------------------------
# LEG value_class — the classes ordinary random data does not construct
# ---------------------------------------------------------------------------

def plant_signed_zero_lattice(fields: Any, axis: int) -> int:
    """Rows 0 and 1 negative zero, row MIRROR_SOURCE_INDEX positive zero.

    ENGINEERED SO THE SIGN SURVIVES TO THE OUTPUT, which is the whole difficulty.
    ``-x`` and ``0.0f - x`` part company on ``+0.0`` — but ``g[i] + ghost`` washes
    the sign straight back out unless ``g[i]`` is itself ``-0.0``, so a zeroed ghost
    plane ALONE is a measured NULL. With row 0 negative and row 2 positive the pair
    sum, the coefficient multiply and the row sum into ``f_w`` all carry it.
    """
    planted = 0
    for name in ("Dx", "Dy", "Dz"):
        volume = getattr(fields, name)
        for row, value in ((0, np.float32(-0.0)), (1, np.float32(-0.0)),
                           (folded.MIRROR_SOURCE_INDEX, np.float32(0.0))):
            index: List[Any] = [slice(None)] * 3
            index[axis] = row
            volume[tuple(index)] = value
            planted += int(np.asarray(volume[tuple(index)]).size)
    return planted


def plant_plane(fields: Any, axis: int, row: int, value: float) -> int:
    planted = 0
    for name in ("Dx", "Dy", "Dz"):
        volume = getattr(fields, name)
        index: List[Any] = [slice(None)] * 3
        index[axis] = row
        volume[tuple(index)] = np.float32(value)
        planted += int(np.asarray(volume[tuple(index)]).size)
    return planted


#: ``(label, plant, expect_clean)``. The subnormal plane is EXPECTED to fire and is
#: carried for exactly that reason: it is the coverage boundary stated as a case
#: rather than as a sentence, and it is what makes the clean rows a measurement.
VALUE_CLASSES: Tuple[Tuple[str, Optional[Callable[[Any, int], int]], bool], ...] = (
    ("signed_zero_lattice", plant_signed_zero_lattice, True),
    ("zeroed_ghost_plane",
     lambda f, a: plant_plane(f, a, folded.MIRROR_SOURCE_INDEX, 0.0), True),
    ("negative_zero_ghost_plane",
     lambda f, a: plant_plane(f, a, folded.MIRROR_SOURCE_INDEX, -0.0), True),
    ("subnormal_ghost_plane",
     lambda f, a: plant_plane(f, a, folded.MIRROR_SOURCE_INDEX, 1e-40), False),
)


def leg_value_class(payload: Dict[str, Any], out: str) -> None:
    """The shipped kernel on value classes random data does not reach.

    Signed zeros are NOT subnormal — the census counts a zero exponent with a
    NONZERO mantissa — so the signed-zero rows sit INSIDE the precondition and their
    comparisons are certified. The subnormal plane sits outside it and is REFUSED BY
    NAME, its comparisons withheld: that is a COVERAGE REFUSAL, not a failure, and
    it is the boundary this backend's native flush imposes.
    """
    rows: List[Dict[str, Any]] = []
    for label, plant, expect_clean in VALUE_CLASSES:
        fields, pml = build(axes="Y", phases=(1,))
        codes, _ = symmetry.folded_axis_kinds(fields.grid, pml)
        axis = next(a for a, code in enumerate(codes)
                    if int(code) in folded.MIRROR_CODES)
        planted = plant(fields, axis)
        kit.assert_census_floor(planted, f"{label} planted nothing", floor=8)

        before, after, moved = oracle(fields, pml)
        kit.assert_moved(moved, f"{label} the array path barely moved", floor=64)
        weights = folded.mirror_ghost_weights(fields.grid)
        window = census_window(fields, after, codes, weights, step=0)
        report = window.report()
        assert not report["vacuous"], (label, report["vacuity_reasons"])
        refused = not report["clean"]
        assert refused is not expect_clean, (
            label, "the value class did not land where it was declared to",
            report["subnormal_words"])

        zero_census = [subnormal.signed_zero_census(getattr(fields, n))
                       for n in ("Dx", "Dy", "Dz")]
        signed_zeros = {
            "negative_zero": sum(row["negative_zero"] for row in zero_census),
            "positive_zero": sum(row["positive_zero"] for row in zero_census)}
        plan = run_shipped(fields, pml)
        assert plan.launches == 1, (label, plan.launches)
        per = divergence(fields, after)
        rows.append({"class": label, "planted_words": planted,
                     "signed_zero_words": signed_zeros,
                     "moved": moved,
                     "compared": 0 if refused else compared_words(after),
                     "differing": sum(per.values()), "per_target": per,
                     "subnormal": window_row(window, refused),
                     "refused": refused,
                     "refusal": (f"{label}: REFUSED (subnormal precondition) — the "
                                 f"planted plane is IN the band, which is the "
                                 f"coverage boundary this backend's native flush "
                                 f"imposes") if refused else None})
        log(f"[value_class] {label:<28} planted={planted} signed_zeros={signed_zeros} "
            f"moved={moved} differing={sum(per.values())}"
            f"{' REFUSED' if refused else ''}")
        restore(fields, before)
        payload["legs"]["value_class"] = rows
        save(payload, out)


# ---------------------------------------------------------------------------
# LEG whole_step — the real arbiter
# ---------------------------------------------------------------------------

def composed_plans(fields: Any, pml: Any, residency: Any) -> Dict[str, Any]:
    plans = {
        "step_B": symmetry.plan_folded_pml_curl(fields, pml, "step_B", residency),
        "fill_B": symmetry.plan_mirror_ghost_fill(fields, "B", "fill_B", residency),
        "update_H": symmetry.plan_folded_constitutive(fields, pml, "H", residency),
        "step_D": symmetry.plan_folded_pml_curl(fields, pml, "step_D", residency),
        "fill_D": symmetry.plan_mirror_ghost_fill(fields, "D", "fill_D", residency),
        "update_E": folded.plan_folded_offdiag_constitutive(fields, pml, residency),
    }
    missing = [name for name, plan in plans.items() if plan is None]
    assert not missing, f"no plan for {missing}"
    return plans


def reference_step(fields: Any, pml: Any) -> None:
    """ONE COMPLETE DRIVER STEP on the array path — five passes per half."""
    stepping.step_B(fields, pml)
    stepping.fill_symmetry_bc_B(fields)
    stepping.fill_folded_far_ghosts_B(fields)
    stepping.update_H(fields, pml)
    stepping.step_D(fields, pml)
    stepping.fill_symmetry_bc_D(fields)
    stepping.fill_folded_far_ghosts_D(fields)
    stepping.update_E(fields, pml)


def walk(plans: Dict[str, Any], order: Sequence[str] = DRIVER_ORDER) -> None:
    for name in order:
        if name.endswith("_near"):
            plans[name[:-5]].run_near()
        elif name.endswith("_far"):
            plans[name[:-4]].run_far()
        else:
            plans[name].run()


def leg_whole_step(payload: Dict[str, Any], out: str, budget: int = 6) -> None:
    """Per COMPLETE STEP, reporting the FIRST DIVERGENT STEP.

    SIX GREEN SUB-STEP COMPARISONS SAY NOTHING ABOUT THE OBJECT THE ENGINE RUNS.
    Three failure classes live only here — a STALE MIRROR (the engine holds NumPy,
    so a sub-step left on the array path writes the HOST array and a device mirror
    held across it is a smooth, plausible, WRONG field), a SEAM (``zero_metal_B`` /
    ``zero_metal_D`` clear stored cell 0 between the curl and the constitutive
    sub-step on a walled run, stepping.py:2211/:2238), and an ACCUMULATING AUXILIARY
    (``fu_*`` and ``f_w_*`` are STATE) — and the FOLD adds a fourth: the mirror plane
    is WRITTEN by one pass and READ by the next, so a fold bug can be byte-perfect
    per sub-step and wrong per step.

    Per-slot launch counters are asserted, because a slot that passes by NOT
    EXECUTING is the hollow pass this discipline exists to prevent. The subnormal
    census runs EVERY STEP and is reported as a WINDOW ``[first, last]``: band entry
    is a RUN-and-WINDOW fact, not a family fact — measured on the chi3 round, a Q~20
    narrow-band Gaussian turn-on drags the leading edge through the whole band from
    step 55 to step 3,726 and then runs clean for 16,274 more steps, so a row
    recording a scalar count cannot be read for what it covers.
    """
    rows: List[Dict[str, Any]] = []
    for label, kwargs in WHOLE_STEP_CASES:
        fields, pml = build(**kwargs)
        reference_fields, reference_pml = build(**kwargs)
        for name in STORED:
            if getattr(fields, name, None) is not None:
                assert differing(getattr(fields, name),
                                 getattr(reference_fields, name)) == 0, (label, name)

        codes, _ = symmetry.folded_axis_kinds(fields.grid, pml)
        weights = folded.mirror_ghost_weights(fields.grid)
        residency = device.Residency()
        plans = composed_plans(fields, pml, residency)
        residency.sync_in()

        first_divergent: Optional[Dict[str, Any]] = None
        per_step: List[int] = []
        census_per_step: List[int] = []
        fired: List[int] = []
        for step in range(budget):
            reference_step(reference_fields, reference_pml)
            walk(plans)
            residency.sync_out()
            per = {name: differing(getattr(fields, name),
                                   getattr(reference_fields, name))
                   for name in STORED if getattr(fields, name, None) is not None}
            total = sum(per.values())
            per_step.append(total)
            window = census_window(reference_fields, None, codes, weights, step=step)
            report = window.report()
            assert not report["vacuous"], (label, step, report["vacuity_reasons"])
            census_per_step.append(report["subnormal_words"])
            if not report["clean"]:
                fired.append(step)
            if total and first_divergent is None:
                first_divergent = {"step": step,
                                   "targets": {k: v for k, v in per.items() if v}}

        evolved = sum(differing(np.zeros_like(getattr(reference_fields, name)),
                                getattr(reference_fields, name))
                      for name in ("Bx", "By", "Bz", "Ex", "Ey", "Ez"))
        kit.assert_moved(evolved, f"{label} whole-step state never evolved", floor=100)
        for slot in ("step_B", "update_H", "step_D", "update_E"):
            assert plans[slot].launches == budget, (label, slot,
                                                    plans[slot].launches)
        for slot in ("fill_B", "fill_D"):
            plan = plans[slot]
            assert len(plan.near) > 0, (label, slot)
            assert plan.launches == budget * (len(plan.near) + len(plan.far)), (
                label, slot, plan.launches)
        refused = bool(fired)
        # A REFUSAL IS NOT A PASS BY ANOTHER NAME. Where the precondition held, a
        # divergence is still a failure and still stops the leg here.
        if not refused:
            assert first_divergent is None, (label, first_divergent, per_step)
        window_bounds = {"first_step": fired[0] if fired else None,
                         "last_step": fired[-1] if fired else None,
                         "steps_censused": budget,
                         "subnormal_words_per_step": census_per_step}
        rows.append({
            "case": label, "tags": tags(fields, pml), "budget": budget,
            "per_step_differing": per_step, "first_divergent": first_divergent,
            "evolved_words": evolved, "subnormal_window": window_bounds,
            "refused": refused,
            "refusal": (f"{label}: REFUSED (subnormal precondition) — the census "
                        f"fired on steps {fired[0]}..{fired[-1]} of {budget}"
                        ) if refused else None,
            "compared": 0 if refused else compared_words(
                snapshot(reference_fields, STORED)) * budget,
            "launches": {name: plan.launches for name, plan in plans.items()}})
        log(f"[whole_step] {label:<32} {budget} steps, first_divergent="
            f"{None if first_divergent is None else first_divergent['step']}, "
            f"evolved={evolved} subnormal_window="
            f"[{window_bounds['first_step']}, {window_bounds['last_step']}]"
            f"{' REFUSED' if refused else ''}")
        payload["legs"]["whole_step"] = rows
        save(payload, out)


# ---------------------------------------------------------------------------
# LEG precondition — the control that makes the census FIRE
# ---------------------------------------------------------------------------

#: Scale, and whether the per-step census MUST stay clean at it. The clean rows are
#: what make the firing row mean something: a detector that fired on everything
#: would refuse the physical band too and certify nothing.
PRECONDITION_SCALES: Tuple[Tuple[str, float, bool], ...] = (
    ("physical", 1.0, True),
    ("small_normal", 1e-30, True),
    ("subnormal_band", 1e-38, False),
    ("deep_subnormal", 1e-40, False),
)


def leg_precondition(payload: Dict[str, Any], out: str, budget: int = 6) -> None:
    """A precondition never demonstrated to FIRE is decoration.

    :func:`leg_byte` and :func:`leg_whole_step` census every case and report empty
    windows on the physical band — and an empty window is exactly what a BROKEN
    census also produces. This leg drives the SAME census over a state scaled INTO
    the band and requires it to fire, through
    :func:`preconditions.demonstrate_firing`, so the empty windows next door are a
    measurement rather than a silence. The clean rows go through
    :func:`preconditions.assert_clean_or_refuse` — the helper a gate is supposed to
    call — so it is exercised rather than merely available.

    THE 1e-30 ROW IS THE CLIFF, NOT A THRESHOLD: eight decades below physical the
    census is still empty, and one decade further it is total.
    """
    rows: List[Dict[str, Any]] = []
    for label, scale, expect_clean in PRECONDITION_SCALES:
        reference_fields, reference_pml = build(axes="Y", phases=(1,))
        if scale != 1.0:
            for name in TOUCHED:
                array = getattr(reference_fields, name, None)
                if array is not None:
                    array[...] = (array * np.float32(scale)).astype(np.float32)
        codes, _ = symmetry.folded_axis_kinds(reference_fields.grid, reference_pml)
        weights = folded.mirror_ghost_weights(reference_fields.grid)

        fired: List[int] = []
        per_step: List[int] = []
        observed = 0
        merged = preconditions.SubnormalWindow(
            first_step=0, last_step=budget - 1, per_array_words=1,
            per_intermediate_words=1)
        for step in range(budget):
            reference_step(reference_fields, reference_pml)
            window = census_window(reference_fields, None, codes, weights, step=step)
            report = window.report()
            assert not report["vacuous"], (label, step, report["vacuity_reasons"])
            per_step.append(report["subnormal_words"])
            observed += report["observed_words"]
            if not report["clean"]:
                fired.append(step)
            for name in ("Dx", "Dy", "Dz", "Ex", "Ey", "Ez"):
                merged.observe(name, getattr(reference_fields, name), step=step)
            for axis, code in enumerate(codes):
                if int(code) not in folded.MIRROR_CODES:
                    continue
                volume = reference_fields.Dy
                face: List[Any] = [slice(None)] * 3
                face[axis] = 0
                source: List[Any] = [slice(None)] * 3
                source[axis] = folded.MIRROR_SOURCE_INDEX
                merged.observe_intermediate(
                    f"ghost_pair[{axis}]Dy",
                    (volume[tuple(face)] + np.float32(weights[axis])
                     * volume[tuple(source)]).astype(np.float32), step=step)

        row = {"scale": label, "factor": scale, "steps_censused": budget,
               "subnormal_words_per_step": per_step, "observed_words": observed,
               "first_step": fired[0] if fired else None,
               "last_step": fired[-1] if fired else None,
               "census_fired": bool(fired),
               "verdict": ("REFUSED (subnormal precondition)" if fired
                           else "precondition holds")}
        rows.append(row)
        log(f"[precondition] {label:<16} scale={scale:g} window="
            f"[{row['first_step']}, {row['last_step']}] fired={bool(fired)} "
            f"words_per_step={per_step}")
        payload["legs"]["precondition"] = rows
        save(payload, out)
        kit.assert_census_floor(observed, f"{label} censused nothing", floor=1000)
        if expect_clean:
            preconditions.assert_clean_or_refuse(
                merged, f"folded off-diagonal update_E, {label} band")
            assert not fired, (label, "the census fired on a band it must not")
        else:
            preconditions.demonstrate_firing(
                merged, f"folded off-diagonal update_E, {label} control")
            assert fired, (label, "the census DID NOT FIRE on the scaled control")


# ---------------------------------------------------------------------------
# LEG mutations — a gate that cannot fail certifies nothing
# ---------------------------------------------------------------------------

def ghost_line(axis: int) -> str:
    """The emitted mirror redirect for one axis, spelled the way the emitter spells
    it — derived from ``_TWO_WAY_NAMES`` rather than transcribed, so a rename in the
    certified emitter turns these needles into NEEDLE-MISSED rather than into a
    silent pass."""
    down, _up, _dvalid, _uvalid, _extent = offdiag._TWO_WAY_NAMES["xyz"[axis]]
    home = offdiag._HOME[axis]
    return (f"    {down} = ({home} == 0) ? {folded.MIRROR_SOURCE_INDEX} "
            f": {down};")


def up_valid_line(axis: int) -> str:
    _down, up, _dvalid, uvalid, extent = offdiag._TWO_WAY_NAMES["xyz"[axis]]
    return f"    {uvalid} = ({up} < {extent});"


def folded_axes_of(codes: Sequence[int]) -> Tuple[int, ...]:
    return tuple(a for a, code in enumerate(codes)
                 if int(code) in folded.MIRROR_CODES)


class Declined(Exception):
    """This grid cannot carry this defect at all — NOT a needle miss.

    "There is no spare axis with room for the image row" is a structural fact about
    the grid; "the transform matched nothing in the shipped source" is a broken
    needle that would report its defect as UNCAUGHT. Conflating them would let a
    real miss hide behind a plausible skip, so they are recorded separately: a
    declination lands in ``skipped_cases`` and a miss fails the row.
    """


def leg_mutations(payload: Dict[str, Any], out: str) -> None:
    harness = kit.MutationHarness(payload, out)

    def source_mutation(label: str, mutate: Callable[..., str],
                        must_catch: Optional[bool], why: str,
                        cases: Sequence[Tuple[str, Dict[str, Any]]] = CASES,
                        scope: Sequence[str] = (),
                        plant: Optional[Callable[[Any, int], int]] = None,
                        ) -> Dict[str, Any]:
        """One defect over a case set, launched through the SHIPPED plan.

        ``mutate(source, spec, codes)`` may raise :class:`LookupError` — that is a
        NEEDLE MISS and FAILS, because a transform that silently matched nothing
        would report its defect as UNCAUGHT, which is the most dangerous false pass
        a mutation leg can produce.

        ``scope`` IS A CONJUNCTION OF TAGS DERIVED FROM THE ENGINE, not a hand-typed
        case list, and it is the difference between an honest count and a rounded-up
        one. A defect that does not EXIST on a case — no negated lane to drop, no
        live slot that reads the ghost — reports as uncaught if it is run there, and
        a reader cannot tell that from a real weakness. Every scoped row carries the
        cases it skipped, and the structural nulls are armed separately with
        ``must_catch=False`` so the scoping is itself measured.
        """
        missed = False
        ran = caught = launches = 0
        per_case: Dict[str, int] = {}
        skipped: List[str] = []
        for case, kwargs in cases:
            fields, pml = build(**kwargs)
            present = tags(fields, pml)
            if any(tag not in present for tag in scope):
                skipped.append(case)
                continue
            row_mask, codes, walls, negate = specialisation(fields, pml)
            if plant is not None:
                plant(fields, folded_axes_of(codes)[0])
            _before, after, moved = oracle(fields, pml)
            kit.assert_moved(moved, f"{label}/{case} the array path barely moved",
                             floor=64)
            source = folded.folded_offdiag_source(row_mask, codes, walls, negate)
            try:
                mutant = mutate(source, (row_mask, codes, walls, negate), codes,
                                fields)
            except Declined:
                skipped.append(case)
                continue
            except LookupError:
                missed = True
                continue
            if mutant == source:
                missed = True
                continue
            plan = run_mutant(fields, pml, mutant)
            launches += plan.launches
            ran += 1
            found = sum(divergence(fields, after).values())
            per_case[case] = found
            if found:
                caught += 1
        return harness.record(label, harness.verdict(missed, ran, launches, caught),
                              launches, caught, ran, must_catch, why,
                              extra={"scope": list(scope), "skipped_cases": skipped,
                                     "differing_per_case": per_case})

    # ---------------------------------------------------------------- the fold's
    # 1. MIRROR THE WRONG HALF. The mirror plane sits at the NEAR face and images
    #    stored row 2; taking the image from the FAR half reconstructs the half the
    #    fold discarded. In-bounds by construction, so this measures a wrong ANSWER
    #    rather than a wrong ADDRESS.
    def wrong_half(source: str, spec: Any, codes: Sequence[int],
                   fields: Any) -> str:
        text = source
        for axis in folded_axes_of(codes):
            down, _up, _dv, _uv, extent = offdiag._TWO_WAY_NAMES["xyz"[axis]]
            home = offdiag._HOME[axis]
            text = needle(text, ghost_line(axis),
                          f"    {down} = ({home} == 0) ? ({extent} - 3) : {down};")
        return text

    source_mutation(
        "mirror_images_the_far_half", wrong_half, True,
        "the mirror plane is at the NEAR face and images stored row "
        f"{folded.MIRROR_SOURCE_INDEX} (stepping.py:1870-1873, MEEP's halved origin "
        "io = -2); imaging the far end of the array reconstructs the discarded half",
        scope=("ghost_reachable",))

    # 1b. THE SAME MUTANT WHERE THE GHOST IS UNREACHABLE — a MEASURED NULL, and the
    #     row that turns `mirror_arm_is_reachable` from a claim into a check. The
    #     mirror rule enters this sub-step only through the DOWN shift of the PARTNER
    #     axis, so a fold on an axis no live row slot takes as its partner cannot
    #     change a byte however wrongly it is spelled. Without this row the scoping
    #     above would be indistinguishable from a convenient exclusion.
    source_mutation(
        "mirror_images_the_far_half_where_the_ghost_is_unreachable", wrong_half,
        False,
        "THE NAMED NULL behind the scope of the row above: fold X with the single "
        "live slot Ey <- Ez takes no live partner on the folded axis, so the mirror "
        "arm is unreachable and the grid is byte-identical to the certified "
        "METALLIC code however the redirect is spelled",
        scope=("ghost_unreachable",))

    # 2. SHIFT THE MIRROR PLANE BY ONE CELL, both ways. `_mirror_source`
    #    (stepping.py:1582-1588) RAISES below three stored cells precisely because
    #    the row is exact; rows 1 and 3 are each a whole cell of the wrong field.
    for offset in (-1, 1):
        def shift_plane(source: str, spec: Any, codes: Sequence[int],
                        fields: Any, offset: int = offset) -> str:
            text = source
            for axis in folded_axes_of(codes):
                down, _up, _dv, _uv, _extent = offdiag._TWO_WAY_NAMES["xyz"[axis]]
                home = offdiag._HOME[axis]
                text = needle(
                    text, ghost_line(axis),
                    f"    {down} = ({home} == 0) ? "
                    f"{folded.MIRROR_SOURCE_INDEX + offset} : {down};")
            return text

        source_mutation(
            f"mirror_plane_off_by_{'minus_' if offset < 0 else ''}one", shift_plane,
            True,
            f"the ghost images stored row {folded.MIRROR_SOURCE_INDEX} exactly; row "
            f"{folded.MIRROR_SOURCE_INDEX + offset} is a whole cell of the wrong "
            f"field and is a COMPILED CONSTANT, so it is reachable wherever the "
            f"ghost lane is read at all",
            scope=("ghost_reachable",))

    # 3. DROP THE SIGN FLIP on a mirrored component. The sign is a SOURCE
    #    specialisation on this backend (the measured requirement: a RUNTIME weight
    #    flushes every subnormal, a compile-time one is exact), so the mutant is a
    #    different compiled body rather than a different passed word. SCOPED to the
    #    cases that HAVE a negated lane: on an ODD plane the weight is `-phase == +1`
    #    and the certified text is already exact, so dropping the negation there is
    #    the IDENTITY and reporting it as an uncaught defect would be dishonest.
    def drop_sign(source: str, spec: Any, codes: Sequence[int],
                   fields: Any) -> str:
        row_mask, codes_, walls, negate = spec
        if not any(negate):
            raise LookupError("no negated lane on this grid")
        return folded.folded_offdiag_source(row_mask, codes_, walls, (0, 0, 0))

    source_mutation(
        "drop_the_sign_flip_on_the_mirror_lane", drop_sign, True,
        "the ghost weight is mirror_parity('D'+axis, axis, phase) == -phase "
        "(fields.py:117-182, stepping.py:1870-1873), so an EVEN plane NEGATES; "
        "emitting the un-negated lane there drops the parity entirely",
        scope=("negated", "ghost_reachable"))

    # 4. THE SPURIOUS NEGATION, the other half of the same defect: an ODD plane
    #    reduces to the certified text, and negating it invents a parity.
    def add_sign(source: str, spec: Any, codes: Sequence[int],
                   fields: Any) -> str:
        row_mask, codes_, walls, negate = spec
        if any(negate):
            raise LookupError("this grid already negates")
        invented = tuple(int(code in folded.MIRROR_CODES) for code in codes_)
        return folded.folded_offdiag_source(row_mask, codes_, walls, invented)

    source_mutation(
        "spurious_negation_on_an_odd_plane", add_sign, True,
        "an ODD plane's weight is +1 and the certified term text is already exact; "
        "negating it invents a parity the array path does not apply",
        scope=("un_negated", "ghost_reachable"))

    # 5. FOLD THE WRONG AXIS. The axis selects the ghost redirect AND the sign lane
    #    together, so this is the mutation that proves those two agree with each
    #    other rather than merely compiling. The grid stays folded where it is: only
    #    the EMISSION moves, which is the defect a reader would actually write.
    def wrong_axis(source: str, spec: Any, codes: Sequence[int],
                   fields: Any) -> str:
        row_mask, codes_, walls, negate = spec
        folded_here = folded_axes_of(codes_)
        shape = tuple(int(n) for n in fields.grid.shape)
        # THE TARGET MUST BE A REAL AXIS WITH ROOM FOR THE IMAGE ROW, or the mutant
        # reads past the buffer and the leg is measuring undefined behaviour rather
        # than a wrong answer. A grid with no such axis DECLINES — that is a
        # structural fact about the grid, not a needle that matched nothing, and the
        # two are recorded separately.
        spare = [a for a in range(3)
                 if a not in folded_here and not walls[a]
                 and int(codes_[a]) in (P, M)
                 and shape[a] > folded.MIRROR_SOURCE_INDEX]
        if not spare:
            raise Declined("no spare unfolded axis with room for the image row")
        target = spare[0]
        moved_codes = list(codes_)
        moved_negate = [0, 0, 0]
        for axis in folded_here:
            moved_codes[axis] = M
        moved_codes[target] = int(codes_[folded_here[0]])
        moved_negate[target] = int(negate[folded_here[0]])
        return folded.folded_offdiag_source(row_mask, tuple(moved_codes), walls,
                                            tuple(moved_negate))

    source_mutation(
        "fold_applied_to_the_wrong_axis", wrong_axis, True,
        "the axis selects the ghost redirect and the sign lane together; moving the "
        "mirror to an UNFOLDED axis serves a metallic zero where the array path "
        "serves a live parity-weighted plane and a live plane where it serves zero",
        scope=("ghost_reachable",))

    # ------------------------------------------------------- the ghost rule itself
    # 6/7. THE MIRROR GHOST IS A REDIRECT TO A LIVE INTERIOR PLANE. It is neither the
    #      metallic mask nor the periodic wrap, and both confusions are one line.
    def as_metallic(source: str, spec: Any, codes: Sequence[int],
                   fields: Any) -> str:
        text = source
        for axis in folded_axes_of(codes):
            _down, _up, dvalid, _uv, _extent = offdiag._TWO_WAY_NAMES["xyz"[axis]]
            down = offdiag._TWO_WAY_NAMES["xyz"[axis]][0]
            text = needle(text, ghost_line(axis), f"    {dvalid} = ({down} >= 0);")
        return text

    source_mutation(
        "mirror_ghost_becomes_metallic", as_metallic, True,
        "the metallic arm masks the face-0 lane to an exact 0; the mirror arm serves "
        "parity * field[2]. `_mask_metallic_wall_coupling` ABSTAINS on a mirrored "
        "axis (stepping.py:1282), so the mask is not restored downstream",
        scope=("ghost_reachable",))

    def as_periodic(source: str, spec: Any, codes: Sequence[int],
                   fields: Any) -> str:
        text = source
        for axis in folded_axes_of(codes):
            down, _up, _dv, _uv, extent = offdiag._TWO_WAY_NAMES["xyz"[axis]]
            text = needle(text, ghost_line(axis),
                          f"    {down} = ({down} < 0) ? ({extent} - 1) : {down};")
        return text

    source_mutation(
        "mirror_ghost_becomes_periodic", as_periodic, True,
        "the periodic arm wraps to the far end; the mirror arm images stored row 2. "
        "Both are live reads, so this is a plane of plausible wrong values rather "
        "than a crash",
        scope=("ghost_reachable",))

    # 8. THE SIGN MUST APPLY ON THE GHOST LANE ONLY. Applied to the whole volume it
    #    negates every interior cell of the partner term.
    def negate_everything(source: str, spec: Any, codes: Sequence[int],
                   fields: Any) -> str:
        text = source
        for axis in folded_axes_of(codes):
            at = ("at_x", "at_y", "at_z")[axis]
            for stem in ("dn_", "cn_"):
                if f"({at} ? -{stem}" in text:
                    text = text.replace(f"({at} ? -{stem}", f"(true ? -{stem}")
        if text == source:
            raise LookupError("no negated lane predicate on this grid")
        return text

    source_mutation(
        "negation_applied_to_the_whole_volume", negate_everything, True,
        "the mirror ghost exists on the face-0 lane and nowhere else; negating every "
        "lane negates interior neighbours the array path leaves alone",
        scope=("negated", "ghost_reachable"))

    # 9. THE WALL MASK MUST ABSTAIN ON A FOLD PLANE. Zeroing it the metallic way is
    #    the defect stepping.py:1266-1277 measures at 2.0e-02 even / 3.5e-03 odd.
    def mask_the_fold_plane(source: str, spec: Any, codes: Sequence[int],
                   fields: Any) -> str:
        axis = folded_axes_of(codes)[0]
        at = ("at_x", "at_y", "at_z")[axis]
        return needle(source, "    float src0 = (gs0 * us0) + total0;",
                      f"    total0 = {at} ? 0.0f : total0;\n"
                      f"    float src0 = (gs0 * us0) + total0;")

    source_mutation(
        "fold_plane_masked_the_metallic_way", mask_the_fold_plane, True,
        "`_mask_metallic_wall_coupling` asks `is_metallic and NOT is_mirrored` "
        "(stepping.py:1282) and therefore ABSTAINS on the fold plane; masking it "
        "costs a measured 2.0e-02 (stepping.py:1266-1277)",
        scope=("component0_live",))

    # 10/11. THE OWN-AXIS UP SHIFT IS AN EXACT ZERO ON BOTH TERMINATIONS, because
    #        `_offdiagonal_terms` calls `_shift_up` with FOUR arguments (stepping.py:
    #        1248-1249) so the folded-PERIODIC reflect branch cannot fire. THE PAIR
    #        IS WHAT MAKES THE CATCH DISCRIMINATING: the same mutant on a row set
    #        where the folded axis is no live row's OWN axis is a MEASURED NULL, and
    #        reporting only the catch would leave a reader unable to tell a
    #        measurement from a coincidence.
    def wrap_own_axis(source: str, spec: Any, codes: Sequence[int],
                   fields: Any) -> str:
        text = source
        for axis in folded_axes_of(codes):
            _down, up, _dv, _uv, extent = offdiag._TWO_WAY_NAMES["xyz"[axis]]
            text = needle(text, up_valid_line(axis),
                          f"    {up} = ({up} == {extent}) ? 0 : {up};")
        return text

    source_mutation(
        "own_axis_up_wraps_with_a_live_Ey_row", wrap_own_axis, True,
        "the folded-PERIODIC reflect branch (stepping.py:1772-1780) cannot fire from "  # stepping.py live lines for the frozen device-text citation(s) in this string: 1772-1780->1819-1827
        "`_offdiagonal_terms`, so both terminations take the exact 0.0 at "
        "stepping.py:1828-1830; a wrap is a live read of the far end",
        cases=(("2d_fold_Y_live_Ey_row",
                dict(axes="Y", phases=(1,), rows={"Ey": ("Ez", "Ex")})),))

    source_mutation(
        "own_axis_up_wraps_with_the_Ey_row_dead", wrap_own_axis, False,
        "THE NAMED NULL that makes the row above discriminating: with no live row "
        "whose OWN axis is the folded one, the wrapped up index is never read",
        cases=(("2d_fold_Y_default_rows", dict(axes="Y", phases=(1,))),))

    # 12/13. THE REFUTED SPELLING. Measured in this family's own term shape over 30
    #        words: `0.0f - x` misses 12/30 on the GHOST VALUE (10/10 subnormals AND
    #        2/4 signed zeros) while `-x` is exact — but added to a NORMAL partner
    #        every spelling agrees, because the addition renormalises. So on ordinary
    #        data it is a NULL, and the value class is what gives it reach.
    def zero_minus_x(source: str, spec: Any, codes: Sequence[int],
                   fields: Any) -> str:
        text = source
        for axis in folded_axes_of(codes):
            at = ("at_x", "at_y", "at_z")[axis]
            for stem in ("dn_", "cn_"):
                text = text.replace(f"({at} ? -{stem}", f"({at} ? 0.0f - {stem}")
        if text == source:
            raise LookupError("no negated lane on this grid")
        return text

    source_mutation(
        "refuted_spelling_zero_minus_x_on_normals", zero_minus_x, False,
        "A NAMED NULL, and the reason it is a mutation rather than a comment lives "
        "in the row below: on ordinary normal data the term's own sum renormalises "
        "and every spelling agrees",
        cases=(("2d_fold_Y_even_periodic", dict(axes="Y", phases=(1,))),))

    source_mutation(
        "refuted_spelling_zero_minus_x_on_a_signed_zero_lattice", zero_minus_x, True,
        "the same mutant on the engineered signed-zero lattice, UNDER the "
        "subnormal-free precondition: `-x` and `0.0f - x` part company on +0.0, and "
        "with row 0 negative zero the sign survives the pair sum, the coefficient "
        "multiply and the row sum into f_w",
        cases=(("2d_fold_Y_even_periodic", dict(axes="Y", phases=(1,))),),
        plant=plant_signed_zero_lattice)

    # ------------------------------------------------- the whole-step-only defect
    # 14. READ THE GHOST BEFORE IT IS WRITTEN. Every kernel here is the SHIPPED one
    #     and each is byte-perfect per sub-step; ONLY THE ORDER MOVES. `update_E`
    #     reads the D plane the mirror fill writes (stored cell 0 on the folded axis
    #     for the shift-0 components, and the far ghost row on MIRROR_PERIODIC), so
    #     walking it before the fill is a stale read. This is the fold's fourth risk
    #     stated as a needle, and it is what makes leg_whole_step's
    #     `first_divergent=None` a RESULT rather than a property of a leg that
    #     cannot fail.
    STALE_ORDER = ("step_B", "fill_B_near", "fill_B_far", "update_H",
                   "step_D", "update_E", "fill_D_near", "fill_D_far")
    assert sorted(STALE_ORDER) == sorted(DRIVER_ORDER), (
        "the stale order must be a PERMUTATION of the driver's own: a needle that "
        "also dropped or duplicated a pass would be measuring something else")
    ran = caught = launches = 0
    first_steps: Dict[str, Optional[int]] = {}
    budget = 6
    for case, kwargs in WHOLE_STEP_CASES:
        fields, pml = build(**kwargs)
        reference_fields, reference_pml = build(**kwargs)
        residency = device.Residency()
        plans = composed_plans(fields, pml, residency)
        residency.sync_in()
        first: Optional[int] = None
        for step in range(budget):
            reference_step(reference_fields, reference_pml)
            walk(plans, STALE_ORDER)
            residency.sync_out()
            total = sum(differing(getattr(fields, name),
                                  getattr(reference_fields, name))
                        for name in STORED
                        if getattr(fields, name, None) is not None)
            if total and first is None:
                first = step
        launches += sum(plan.launches for plan in plans.values())
        ran += 1
        first_steps[case] = first
        if first is not None:
            caught += 1
    harness.record("read_the_ghost_before_it_is_written",
                   harness.verdict(False, ran, launches, caught),
                   launches, caught, ran, True,
                   "update_E reads the mirror plane the fill writes. Every kernel "
                   "here is the SHIPPED one and each is byte-perfect per sub-step; "
                   "only the ORDER moves, so this defect exists at whole-step "
                   "granularity and nowhere else",
                   extra={"first_divergent_step_per_case": first_steps,
                          "budget": budget})

    # 15. THE EMITTER'S OWN REFUSALS, at the last place they can be seen: a gate
    #     hands the triples straight in, so a sign on an unfolded axis and a wall on
    #     a folded one must be refused by the emitter and not merely by the plan.
    refusals = 0
    for row_mask, codes, walls, negate in (
            ((1, 0, 0, 0, 0, 0), (P, P, P), (0, 0, 0), (0, 1, 0)),
            ((1, 0, 0, 0, 0, 0), (P, MM, P), (0, 1, 0), (0, 1, 0)),
            ((0, 0, 0, 0, 0, 0), (P, MM, P), (0, 0, 0), (0, 1, 0))):
        try:
            folded.folded_offdiag_source(row_mask, codes, walls, negate)
        except ValueError:
            refusals += 1
    assert refusals == 3, refusals
    payload["legs"]["emission_refusals"] = {
        "refused": refusals,
        "why": ("a sign on an unfolded axis would be applied to an ordinary "
                "neighbour; a wall on a folded axis would zero a plane MEEP steps; "
                "an all-dead row mask is symmetry's folded constitutive product and "
                "emitting here would overlap the two families"),
    }
    save(payload, out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

LEGS: Tuple[Tuple[str, Callable[[Dict[str, Any], str], None]], ...] = (
    ("execution", leg_execution),
    ("reduction", leg_reduction),
    ("byte", leg_byte),
    ("value_class", leg_value_class),
    ("whole_step", leg_whole_step),
    ("precondition", leg_precondition),
    ("mutations", leg_mutations),
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
    # THE POLICY IS A PRECONDITION OF THE WHOLE ARTIFACT, not of one leg. On MPS the
    # flush is native and has NO lever, so `keep` is not offerable and the honest
    # third value is REFUSE; a gate that ran anyway would be certifying bytes under
    # a policy the device was never in.
    policy = subnormal.mps_policy_report()
    payload["subnormal_policy"] = policy
    reasons.extend(policy["reasons"])
    if reasons:
        return kit.cannot_certify(payload, out, reasons)

    ran = kit.run_legs(LEGS, payload, out, kit.wanted_legs(arguments.legs))

    compared = 0
    certified = True
    refused: List[str] = []
    for key, name_key in (("byte", "case"), ("value_class", "class")):
        for row in payload["legs"].get(key, ()):
            if row.get("refused"):
                refused.append(f"{key}:{row[name_key]}")
                continue
            compared += int(row["compared"])
            certified = certified and row["differing"] == 0
    # A CASE REFUSED ON THE PRECONDITION CONTRIBUTES NO COMPARISONS AND NO VERDICT.
    # Folding it into `certified` either way would be wrong in both directions: as a
    # pass it would certify bytes produced under a condition the claim EXCLUDES, and
    # as a failure it would report a coverage boundary as a defect.
    for row in payload["legs"].get("whole_step", ()):
        if row.get("refused"):
            refused.append(f"whole_step:{row['case']}")
            continue
        compared += int(row["compared"])
        certified = certified and row["first_divergent"] is None

    mutations = payload["legs"].get("mutations", ())
    caught = sum(1 for row in mutations if row["must_catch"] is True
                 and row["caught"] == row["ran"] and row["ran"] > 0)
    armed = sum(1 for row in mutations if row["must_catch"] is True)
    nulls = [row["mutation"] for row in mutations if row["must_catch"] is False]

    return kit.summarize(
        payload, out,
        claim=("metal_kernels.folded_offdiag_update_e reproduces stepping.update_E "
               "word for word on a mirror-folded grid carrying off-diagonal chi1inv "
               "rows, per sub-step and per complete step, subject to a CHECKED "
               "subnormal-free precondition reported as a window"),
        scope=("the case matrix in CASES: both fold terminations, both plane "
               "parities, one and three folded axes, both full-count parities, a "
               "declared metallic wall beside a fold, row sets whose live slot "
               "misses / crosses the fold and all six slots live, 2-D and 3-D. "
               "complex64 storage, a nonzero beta, BFAST, a registered "
               "susceptibility, chi2/chi3, a nonzero k_point and cylindrical "
               "coordinates are OUT OF SCOPE and refused by name"),
        stated_weakness=(
            "no generated-code audit exists on this backend: torch.mps.compile_shader "
            "exposes no disassembly, so this gate cannot refuse a compile whose "
            "emitted code violates the policy nor establish that the contraction "
            "guard was obeyed. Every leg is BEHAVIOURAL. Partly offset: the 'this arm "
            "reduces to the certified body' claim, which the Triton twin needs a PTX "
            "read for, is settled here by string equality (leg reduction)"),
        started=started, legs_run=ran, compared=compared, certified=certified,
        extra={"cases_refused_on_precondition": refused,
               "mutations_caught_of_armed": f"{caught}/{armed}",
               "measured_nulls": nulls,
               "subnormal_windows": {
                   row["case"]: [row["subnormal_window"]["first_step"],
                                 row["subnormal_window"]["last_step"]]
                   for row in payload["legs"].get("whole_step", ())}})


if __name__ == "__main__":
    from metal_gate_runner import run_current_measurement

    raise SystemExit(run_current_measurement(__file__, sys.argv[1:]))
