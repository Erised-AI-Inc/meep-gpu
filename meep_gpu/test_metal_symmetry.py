"""Merge-bar tests for the Metal folded (mirror-symmetry) family.

FOUR KINDS OF THING ARE PINNED HERE, and the first two carry the most weight
because no gate can carry them.

**1. THE STRUCTURAL REDUCTION.** The claim this family rests on is that the ONLY
device-code delta over the certified curl is the top-plane mask: the ghost gather
and the cell-0 mask are the CERTIFIED emitters' own output, reached by mapping
every mirror code to ``METALLIC`` on the way in. That is checkable by character and
is checked by character, in both directions — the folded source must CONTAIN the
certified fragments, and a mirror code must emit text identical to a metallic one.
On the Triton track the analogous claim is settled with a PTX read;
``torch.mps.compile_shader`` exposes no disassembly at all, so source text is the
only place it can be held here.

**2. THE SIGN RULE PER COMPONENT.** The fill's parity is derived from
``fields.mirror_parity`` ITSELF rather than from a second reading of the Yee table,
over every (family, axis, phase, pass, component) combination, and the emitted
Metal text is required to be the MEASURED spelling for that sign — ``-x`` for -1
and a plain copy for +1. A test that re-derived the expectation by hand would pin
the same reading twice; this one would fail if ``mirror_parity`` and the emitter
disagreed about a single component.

**3. THE FOLD'S OWN HAZARDS**, which are not the plain curl's:

* THE MIRROR PLANE IS WRITTEN BY ONE PASS AND READ BY THE NEXT. A fold bug can be
  byte-perfect per sub-step and wrong per step, so the whole-step leg compares per
  COMPLETE STEP over a stated budget and reports the FIRST DIVERGENT STEP;
* THE LIVE HALF VS THE DISCARDED HALF. The two ownership masks are INVISIBLE at
  whole-step granularity — the driver's fill passes overwrite exactly the planes
  they protect — so a mask-less kernel passes a whole-step check. The mask leg is
  therefore a SUB-STEP leg and is compared before any fill runs;
* THE TWO TERMINATIONS ARE DIFFERENT KERNELS. MIRROR_PERIODIC carries the
  top-plane mask and the far ghost pass; MIRROR_METALLIC carries neither. Every
  behavioural leg runs both, because a leg carrying one proves nothing about the
  other.

**4. THE DISJOINTNESS**, stated as an inversion rather than inherited: every other
Metal family must refuse a fold BY NAME, and this one must require it.

Device-touching tests are skipped without MPS. The full case matrix and the
mutation harness belong in ``parity/meep_gpu/gate_metal_symmetry.py``; what is here
is the smallest non-vacuous version, so a red merge bar names the defect without a
GPU sweep. EVERY behavioural leg asserts a VACUITY FLOOR — words actually moved —
because zero-init is a fixed point of half of this family's work and a no-op
agreeing with a no-op is trivially identical.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_PARITY = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir,
                       "parity", "meep_gpu")
_PARITY = os.path.abspath(_PARITY)
if _PARITY not in sys.path:
    sys.path.insert(0, _PARITY)

import metal_composition_matrix as matrix  # noqa: E402

from meep_gpu import fields as fields_module  # noqa: E402
from meep_gpu import stepping  # noqa: E402
from meep_gpu.metal_kernels import device, launch, shaders, symmetry  # noqa: E402
from meep_gpu.metal_kernels import coverage as metal_coverage  # noqa: E402
from meep_gpu.triton_kernels import symmetry as triton_symmetry  # noqa: E402

MODULE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "metal_kernels", "symmetry.py")

ENVIRONMENT = matrix.prepare_environment()

#: Both terminations of a fold on axis Y over a periodic x, which is the shape six
#: of the corpus's 86 folded rows carry. `(0, 3, 0)` is MIRROR_PERIODIC — the top
#: plane is NOT owned, so the mask and the far fill both fire; `(0, 2, 0)` is
#: MIRROR_METALLIC, where the top plane is owned and stepped and neither does.
PERIODIC_FOLD = (symmetry.CODE_PERIODIC, symmetry.CODE_MIRROR_PERIODIC,
                 symmetry.CODE_PERIODIC)
METALLIC_FOLD = (symmetry.CODE_PERIODIC, symmetry.CODE_MIRROR_METALLIC,
                 symmetry.CODE_PERIODIC)

ALL_CODES = (symmetry.CODE_PERIODIC, symmetry.CODE_METALLIC,
             symmetry.CODE_MIRROR_METALLIC, symmetry.CODE_MIRROR_PERIODIC)


def _mps_available() -> bool:
    try:
        import torch
    except Exception:  # noqa: BLE001
        return False
    return bool(getattr(getattr(torch.backends, "mps", None), "is_available",
                        lambda: False)())


requires_mps = pytest.mark.skipif(not _mps_available(),
                                  reason="no MPS device on this host")


def _source(codes=PERIODIC_FOLD, backward: bool = False) -> str:
    return symmetry.folded_curl_source(codes, backward)


def _differing(a, b) -> int:
    """uint32 WORD equality, never allclose: this is a bit-identity family."""
    left = np.ascontiguousarray(a, dtype=np.float32).reshape(-1).view(np.uint32)
    right = np.ascontiguousarray(b, dtype=np.float32).reshape(-1).view(np.uint32)
    return int(np.count_nonzero(left != right))


def _moved(before, after) -> int:
    return _differing(before, after)


# ---------------------------------------------------------------------------
# 1. The structural reduction — the ONLY new text is the top-plane mask
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("backward", (False, True))
@pytest.mark.parametrize("codes", [(cx, cy, cz) for cx in ALL_CODES
                                   for cy in ALL_CODES for cz in ALL_CODES])
def test_the_ghost_and_cell_zero_mask_are_the_certified_emitters_output(codes,
                                                                        backward):
    """THE WHOLE REDUCTION CLAIM, over the full 128-source product.

    The folded curl's ghost gather and cell-0 mask must be CHARACTER-IDENTICAL to
    what ``shaders.ghost`` and ``shaders.ownership_mask`` emit for the same codes
    with every mirror code read as METALLIC. If this holds, "both mirror codes take
    the metallic ghost branch" and "the cell-0 mask widens from ``== METALLIC`` to
    ``!= PERIODIC``" are not two claims about a copied emitter — they are one
    mapping applied to the certified one.
    """
    source = symmetry.folded_curl_source(codes, backward)
    reduced = symmetry._reduced_codes(codes)
    for axis, code in zip("xyz", reduced):
        assert shaders.ghost(axis, code, backward) in source, (codes, axis)
    assert shaders.ownership_mask(reduced, backward) in source, codes


@pytest.mark.parametrize("backward", (False, True))
@pytest.mark.parametrize("mirror", (symmetry.CODE_MIRROR_METALLIC,
                                    symmetry.CODE_MIRROR_PERIODIC))
def test_a_mirror_axis_gathers_exactly_as_a_metallic_one(mirror, backward):
    """Both mirror codes take the METALLIC ghost branch — the first fold delta.

    The array path's ghost VALUE on a fold is ``parity * field[2]`` (near) or
    ``parity * field[reflect_row]`` (far, folded PERIODIC), not zero. Serving zero
    is correct ONLY because the sole consumer of either ghost is a cell one of the
    two masks zeroes; the equivalence was measured on the Triton track at sub-step
    granularity with the far face driven live. What is held here is that the two
    codes really do emit the same gather, which is the premise of that argument.
    """
    for axis in "xyz":
        assert (symmetry.folded_ghost(axis, mirror, backward)
                == shaders.ghost(axis, shaders.METALLIC, backward))


@pytest.mark.parametrize("backward", (False, True))
def test_the_top_plane_mask_is_the_exact_complement_of_the_cell_zero_mask(backward):
    """Yee shift 0 and Yee shift 1 partition the nine (target, axis) pairs.

    ``_mask_non_owned_cells`` has two arms — ``if iyee[axis] == 0`` at cell 0 and
    ``if iyee[axis] != 0`` at the top plane — so the two emitters must cover every
    pair exactly once between them. A pair in NEITHER is a plane the curl steps
    that MEEP does not own; a pair in BOTH is a target masked twice.
    """
    top = {(target, axis) for target, axis, _ in symmetry._top_plane_pairs(backward)}
    # ``shaders.ownership_mask``'s own pair table, read back out of its emitted
    # text over an all-metallic grid so this test cannot drift from the emitter.
    all_metallic = (shaders.METALLIC,) * 3
    emitted = shaders.ownership_mask(all_metallic, backward)
    flags = {"at_x": 0, "at_y": 1, "at_z": 2}
    zero = set()
    for line in emitted.splitlines():
        target = int(line.split("curl")[1][0])
        flag = next(name for name in flags if name in line)
        zero.add((target, flags[flag]))
    assert zero & top == set(), sorted(zero & top)
    assert zero | top == {(t, a) for t in range(3) for a in range(3)}


@pytest.mark.parametrize("backward", (False, True))
@pytest.mark.parametrize("codes", (PERIODIC_FOLD, METALLIC_FOLD,
                                   (0, 0, 0), (1, 1, 1)))
def test_the_top_plane_mask_fires_only_on_a_folded_periodic_axis(codes, backward):
    """MIRROR_PERIODIC and nothing else. Getting this backwards is a plane of
    wrong values, not a crash: on MIRROR_METALLIC the top plane IS owned and
    stepped, so masking it there deletes a real cell, and on MIRROR_PERIODIC not
    masking it steps a slot the fill pass owns.
    """
    emitted = symmetry.folded_top_plane_mask(codes, backward)
    expected = any(code == symmetry.CODE_MIRROR_PERIODIC for code in codes)
    assert bool("curl" in emitted) is expected, (codes, emitted)


def test_the_folded_curl_binds_exactly_what_the_certified_curl_binds():
    """The fold adds NO kernel argument: the stored extent is what carries it.

    Also the 31-binding ceiling, asserted rather than left for a later edit to
    discover at compile time.
    """
    folded = _source().count("[[buffer(")
    certified = shaders.curl_source((0, 0, 0), False).count("[[buffer(")
    assert folded == certified == 20
    assert folded <= device.MAX_BUFFER_BINDINGS


def test_the_fill_binds_the_reflect_row_at_runtime_and_bakes_no_row_literal():
    """``reflect_row`` is ``n_full - stored + 2`` — ``stored - 2`` at an even full
    count and ``stored - 3`` at an odd one. Baking ``n - 2`` reflects about the
    window top instead of about the second mirror, which is a whole cell wrong on
    every odd-count run, and a single 3-D grid can carry BOTH terminations with
    DIFFERENT rows. So it must be a runtime scalar, on every emitted fill source.
    """
    for label, source in symmetry.enumerate_folded_sources().items():
        if not label.startswith("mirror_ghost_fill"):
            continue
        assert "constant int&       reflect_row" in source, label
        assert "last   = " in source, label
        if "/far" in label:
            assert "reflect_row * stride" in source, label


# ---------------------------------------------------------------------------
# 2. The sign rule per component — derived from the engine's own function
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pass_name", symmetry.FILL_PASSES)
@pytest.mark.parametrize("phase", (1, -1))
@pytest.mark.parametrize("axis", (0, 1, 2))
@pytest.mark.parametrize("family", ("B", "D"))
def test_the_fill_spells_the_sign_fields_mirror_parity_says(family, axis, phase,
                                                            pass_name):
    """THE SIGN RULE, PER COMPONENT, against ``fields.mirror_parity`` ITSELF.

    ``mirror_parity(c, axis, phase) == phase * (1 - 2 * iyee[c][axis])``, which
    collapses to ``+phase`` on the shift-0 components the NEAR pass writes and
    ``-phase`` on the shift-1 components the FAR pass writes. This asserts the
    emitted Metal text carries exactly that sign for exactly those components —
    from the ENGINE's function, not from a second reading of the Yee table, so a
    drift between the two fails here rather than becoming a plane wrong by twice
    the field.

    AND THE SPELLING IS THE MEASURED ONE. On this backend +1 is a PLAIN COPY and -1
    is ``-x``: a runtime weight flushes every subnormal at BOTH signs (12/30 on the
    fold's own value set) and ``0.0f - x`` misses 13/30, so the parity is a source
    specialisation here rather than the constexpr multiply the Triton twin uses.
    """
    targets = tuple(triton_symmetry.GHOST_FILL_FAMILIES[family]["targets"])
    shifts = tuple(triton_symmetry.TARGET_IYEE[name][axis] for name in targets)
    source = symmetry.mirror_ghost_fill_source(axis, phase, pass_name, shifts)
    wanted_shift = 0 if pass_name == "near" else 1
    written = 0
    for slot, name in enumerate(targets):
        parity = fields_module.mirror_parity(name, axis, phase)
        shift = triton_symmetry.TARGET_IYEE[name][axis]
        if shift != wanted_shift:
            # A component this pass does not own must not be written at all.
            assert f"f{slot}[base]" not in source or shift == 0, (name, pass_name)
            continue
        written += 1
        operand = (f"f{slot}[base + {triton_symmetry.MIRROR_SOURCE_INDEX} * stride]"
                   if pass_name == "near"
                   else f"f{slot}[base + reflect_row * stride]")
        target = ("f%d[base]" % slot if pass_name == "near"
                  else "f%d[base + last * stride]" % slot)
        spelling = operand if parity == 1 else f"-{operand}"
        assert f"    {target} = {spelling};" in source, (name, parity, source)
    assert written, (family, axis, pass_name,
                     "this pass wrote no component: a no-op fill launch is a "
                     "vacuous pass, and the emitter is supposed to refuse it")


def test_no_emitted_fill_uses_a_refuted_parity_spelling():
    """The three MEASURED-WRONG spellings, refused by source text.

    ``0.0f - x`` misses 13/30 on the fold's own value set (twelve subnormals plus
    one -0.0) and canonicalizes NaN; a RUNTIME weight ``w * x`` misses 12/30 at
    BOTH signs, because a runtime multiply by 1.0 is not the identity on this
    backend; and ``(at ? w : 1.0f) * x`` misses 12/30 by flushing the whole volume
    rather than the ghost lane. None may appear.
    """
    for label, source in symmetry.enumerate_folded_sources().items():
        if not label.startswith("mirror_ghost_fill"):
            continue
        assert "0.0f -" not in source, label
        assert "constant float&" not in source, (
            label, "the fill binds no float scalar at all: a runtime parity weight "
                   "flushes subnormals on this backend")


def test_a_parity_outside_plus_or_minus_one_is_refused():
    with pytest.raises(ValueError):
        symmetry._parity_spelling(0, "x")
    with pytest.raises(ValueError):
        symmetry.mirror_ghost_fill_source(1, 2, "near", (0, 1, 1))


def test_an_empty_fill_body_is_refused_rather_than_emitted():
    """A launch that writes nothing is a no-op a before/after comparison passes."""
    with pytest.raises(ValueError):
        symmetry.mirror_ghost_fill_source(0, 1, "near", (1, 1, 1))
    with pytest.raises(ValueError):
        symmetry.mirror_ghost_fill_source(0, 1, "far", (0, 0, 0))


# ---------------------------------------------------------------------------
# 3. The imports are the Triton objects, not copies
# ---------------------------------------------------------------------------

def test_the_fold_tables_are_the_triton_objects():
    """One definition. An equality assertion would be ``x == x``; identity is the
    honest pin, and it is what makes "no second transcription of the Yee table or
    of the MIRROR_METALLIC / MIRROR_PERIODIC split" a fact rather than a comment.
    """
    assert symmetry.TARGET_IYEE is triton_symmetry.TARGET_IYEE
    assert symmetry.GHOST_FILL_FAMILIES is triton_symmetry.GHOST_FILL_FAMILIES
    assert symmetry.folded_axis_kinds is triton_symmetry.folded_axis_kinds
    assert symmetry.MIRROR_SOURCE_INDEX == stepping.MIRROR_SOURCE_INDEX


def test_the_boundary_codes_match_the_shipped_kernels_two():
    """A plan built here and one built in ``shaders`` index the same table."""
    assert symmetry.CODE_PERIODIC == shaders.PERIODIC
    assert symmetry.CODE_METALLIC == shaders.METALLIC
    assert symmetry.MIRROR_CODES == (symmetry.CODE_MIRROR_METALLIC,
                                     symmetry.CODE_MIRROR_PERIODIC)


def test_no_module_level_torch():
    """The predicate must be readable on a host with no GPU and no torch."""
    import ast

    with open(MODULE, "r", encoding="utf-8") as handle:
        tree = ast.parse(handle.read())
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [alias.name for alias in node.names]
            assert "torch" not in names, names
            assert getattr(node, "module", "") != "torch"


# ---------------------------------------------------------------------------
# 4. The residency model — the two slots that were missing
# ---------------------------------------------------------------------------

def test_the_far_ghost_passes_are_in_the_residency_model():
    """The driver's FIFTH pass per half WRITES B and D (driver.py:3287, :3302).

    Without an entry, a mirror held across a step is stale in exactly the plane the
    fold owns and the predicate does not say so. That was the one blocking defect
    before any folded arm could be registered.
    """
    for name in ("fill_folded_far_ghosts_B", "fill_folded_far_ghosts_D"):
        assert name in metal_coverage.RESIDENCY_ORDER
        assert name in metal_coverage.SUB_STEP_VOLUMES
    assert metal_coverage.SUB_STEP_VOLUMES["fill_folded_far_ghosts_B"]["writes"] \
        == ("Bx", "By", "Bz")
    # The far ghost READS the array it writes, which is why it cannot be fused
    # across the wall clear that sits before it.
    assert metal_coverage.SUB_STEP_VOLUMES["fill_folded_far_ghosts_D"]["reads"] \
        == ("Dx", "Dy", "Dz")
    order = metal_coverage.RESIDENCY_ORDER
    assert order.index("zero_metal_B") < order.index("fill_folded_far_ghosts_B")
    assert order.index("fill_folded_far_ghosts_B") < order.index("update_H")


def test_the_fold_makes_its_own_seam_passes_live_without_any_source():
    """``fill_symmetry_bc_*`` runs on EVERY folded run, source list or not.

    Deriving the fill seam from the source list alone was measurably wrong: a
    folded run with no magnetic source reported ``fill_B`` NOT LIVE while the
    driver wrote B there twice.
    """
    fields, pml = matrix.folded()
    live = launch.live_sub_steps(fields, pml, ())
    assert "fill_B" in live and "fill_D" in live, live
    assert "fill_folded_far_ghosts_B" in live, live
    assert "fill_folded_far_ghosts_D" in live, live

    # A folded METALLIC axis has no far ghost pass at all.
    fields, pml = matrix.folded(boundaries={"y": "metallic"})
    live = launch.live_sub_steps(fields, pml, ())
    assert "fill_B" in live, live
    assert "fill_folded_far_ghosts_B" not in live, live

    # And an unfolded run is unchanged: no source, no fill seam.
    fields, pml = matrix.cart()
    live = launch.live_sub_steps(fields, pml, ())
    assert "fill_B" not in live and "fill_folded_far_ghosts_B" not in live, live


def test_folded_periodic_axes_agrees_with_the_array_paths_own_selector():
    """Same discipline ``zero_metal_axes`` follows: ask the engine's function."""
    for boundaries, expected in (({}, (False, True, False)),
                                 ({"y": "metallic"}, (False, False, False))):
        fields, _ = matrix.folded(boundaries=boundaries or None)
        grid = fields.grid
        assert metal_coverage.folded_periodic_axes(grid) == expected, boundaries
        assert tuple(stepping._stored_past_owned(grid, axis)
                     for axis in range(3)) == expected


def test_a_mirror_fill_plan_declares_the_far_pass_it_performs():
    """``replaces_sub_steps`` is DECLARED, so the composer need not read a label."""
    fields, pml = matrix.folded()
    residency = device.Residency()
    plan = symmetry.plan_mirror_ghost_fill(fields, "B", "fill_B", residency)
    assert plan is not None
    assert plan.replaces_sub_steps == ("fill_B", "fill_folded_far_ghosts_B")

    fields, pml = matrix.folded(boundaries={"y": "metallic"})
    plan = symmetry.plan_mirror_ghost_fill(fields, "D", "fill_D",
                                           device.Residency())
    assert plan is not None
    assert plan.replaces_sub_steps == ("fill_D",), (
        "a folded METALLIC axis runs no far ghost pass, so claiming it would be a "
        "residency waiver for work the driver never does")


def test_the_composed_folded_step_holds_its_mirrors_and_the_walled_one_refuses():
    """THE RESIDENCY VERDICT, on the configuration that reaches the stale-mirror
    class — measured in both directions, because a predicate that only ever says
    yes is not a predicate.

    On an unwalled folded run all six slots compose and the two far ghost passes
    are PLANNED (declared through ``replaces_sub_steps``, not inferred from a
    label), so the mirror set is safe to hold across one complete step. On a WALLED
    folded run the fill arm refuses the seam, the fills fall to the array path, and
    the verdict must name every pass that writes a mirrored volume — INCLUDING the
    two far ghost passes, which is exactly what it could not do before they were in
    the model.
    """
    fields, pml = matrix.folded()
    plan = launch.plan_step(fields, pml, device.Residency(), sources=())
    assert set(plan.selected) == {"step_B", "step_D", "update_H", "update_E",
                                  "fill_B", "fill_D"}, plan.selected
    assert "fill_folded_far_ghosts_B" in plan.live
    assert plan.residency.covered, plan.residency.reasons

    # A folded METALLIC axis has no far ghost pass, so it is not live at all.
    fields, pml = matrix.folded(boundaries={"y": "metallic"})
    plan = launch.plan_step(fields, pml, device.Residency(), sources=())
    assert "fill_folded_far_ghosts_B" not in plan.live
    assert plan.residency.covered, plan.residency.reasons

    fields, pml = matrix.folded(boundaries={"x": "metallic"})
    plan = launch.plan_step(fields, pml, device.Residency(), sources=())
    assert "fill_B" not in plan.selected and "fill_D" not in plan.selected
    assert not plan.residency.covered
    named = " ".join(plan.residency.reasons)
    for pass_name in ("fill_B", "zero_metal_B", "fill_folded_far_ghosts_B",
                      "fill_D", "zero_metal_D", "fill_folded_far_ghosts_D"):
        assert pass_name in named, (pass_name, plan.residency.reasons)
    # And each volume is named ONCE: the far pass reads the array it writes, and a
    # refusal that listed every name twice would read as six stale volumes.
    assert "('Bx', 'By', 'Bz', 'Bx'" not in named


# ---------------------------------------------------------------------------
# 5. Coverage — the inverted clause, and every refusal by name
# ---------------------------------------------------------------------------

def test_an_unfolded_grid_is_admitted_by_the_wide_verdict_and_refused_by_routing():
    """The wide verdict exists so the gate can prove the reduction; registering it
    would make every unfolded row ambiguous.
    """
    fields, pml = matrix.cart()
    residency = device.Residency()
    wide = symmetry.folded_pml_curl_coverage(fields, pml, "step_B", residency)
    narrow = symmetry.folded_composition_curl_coverage(fields, pml, "step_B",
                                                       residency)
    assert wide.covered, wide.reasons
    assert not narrow.covered
    assert any("no mirror plane is active" in reason for reason in narrow.reasons)
    # And the registered arm is the NARROW one.
    from meep_gpu.metal_kernels import arms

    folded_arms = [spec for spec in arms.registered("step_B")
                   if spec.family == symmetry.FAMILY]
    assert len(folded_arms) == 1
    assert folded_arms[0].coverage is symmetry._curl_arm_coverage


@pytest.mark.parametrize("slot,side", (("update_H", "H"), ("update_E", "E")))
def test_the_folded_constitutive_requires_a_fold_and_an_active_absorber(slot, side):
    """Two independent inversions on the same slot.

    Clause 5 separates this from ``constitutive``; the ABSORBER clause separates it
    from ``no_pml_constitutive``, which requires an inactive one. Either alone
    would leave a configuration both admit.
    """
    residency = device.Residency()
    fields, pml = matrix.cart()
    verdict = symmetry.folded_constitutive_coverage(fields, pml, side, residency)
    assert not verdict.covered
    assert any("no mirror plane is active" in reason for reason in verdict.reasons)

    fields, pml = matrix.folded_no_pml()
    verdict = symmetry.folded_constitutive_coverage(fields, pml, side,
                                                    device.Residency())
    assert not verdict.covered
    assert any("no active PML layer" in reason for reason in verdict.reasons)


def test_the_curl_refuses_conductivity_and_the_constitutive_does_not():
    """The two contracts are separate FUNCTIONS, deliberately.

    A conductivity routes the CURL to the three-history conductive recurrence and
    changes ``update_H``/``update_E`` not at all. Inheriting the curl-only refusal
    would silently narrow an independent sub-step — 77 update_H slots on the
    measured corpus.
    """
    fields, pml = matrix.conductive(matrix.folded())
    residency = device.Residency()
    curl = symmetry.folded_composition_curl_coverage(fields, pml, "step_D",
                                                     residency)
    assert not curl.covered
    assert any("conductivity is installed" in reason for reason in curl.reasons)
    for side in ("H", "E"):
        verdict = symmetry.folded_constitutive_coverage(fields, pml, side,
                                                        residency)
        assert verdict.covered, (side, verdict.reasons)


@pytest.mark.parametrize("build,needle", (
    (lambda: matrix.folded(complex_storage=True), "force_complex_fields=True"),
    (lambda: matrix.nonlinear(matrix.folded()), "chi2/chi3 is installed"),
    (lambda: matrix.dispersive(matrix.folded()), None),
))
def test_the_named_refusals(build, needle):
    """Each configuration another family owns is refused BY NAME, not by omission."""
    fields, pml = build()
    residency = device.Residency()
    verdict = symmetry.folded_composition_curl_coverage(fields, pml, "step_B",
                                                        residency)
    if needle is None:
        # Dispersion is ADMITTED for the curl: it changes the VALUES update_E
        # writes into the array the curl differences, not an operation the curl
        # performs. The E-side constitutive is what refuses it.
        assert verdict.covered, verdict.reasons
        e_side = symmetry.folded_constitutive_coverage(fields, pml, "E", residency)
        assert not e_side.covered
        assert any("susceptibility is registered" in reason
                   for reason in e_side.reasons)
        return
    assert not verdict.covered
    assert any(needle in reason for reason in verdict.reasons), verdict.reasons


def test_the_offdiagonal_row_is_refused_on_the_e_side_only():
    """The row product READS NEIGHBOURS; this sub-step is element-wise."""
    fields, pml = matrix.folded(rows={"Ex": ("Ey",)})
    assert fields.has_offdiagonal_epsilon
    residency = device.Residency()
    assert not symmetry.folded_constitutive_coverage(fields, pml, "E",
                                                     residency).covered
    assert symmetry.folded_constitutive_coverage(fields, pml, "H",
                                                 residency).covered


def test_the_wall_seam_pairing_is_refused_by_name():
    """``zero_metal_*`` runs BETWEEN the two fill passes, and the far pass reads a
    plane the wall clear touches. Refused rather than fused — and the refusal is
    SCOPED to the two seam slots: the curl and constitutive slots still compose.
    """
    fields, pml = matrix.folded(boundaries={"x": "metallic"})
    residency = device.Residency()
    verdict = symmetry.mirror_ghost_fill_coverage(fields, "B", residency)
    assert not verdict.covered
    assert any("live zero_metal walls" in reason for reason in verdict.reasons), \
        verdict.reasons
    assert symmetry.folded_composition_curl_coverage(fields, pml, "step_B",
                                                     residency).covered
    # And the un-walled twin admits, so the clause is not refusing everything.
    fields, pml = matrix.folded()
    assert symmetry.mirror_ghost_fill_coverage(fields, "B",
                                               device.Residency()).covered


def test_a_plan_built_with_no_residency_is_refused():
    """Two sub-steps mirroring one volume separately would each hold a private copy
    and the second launch would read the first one's stale bytes.
    """
    fields, pml = matrix.folded()
    for verdict in (
            symmetry.folded_composition_curl_coverage(fields, pml, "step_B", None),
            symmetry.folded_constitutive_coverage(fields, pml, "H", None),
            symmetry.mirror_ghost_fill_coverage(fields, "B", None)):
        assert not verdict.covered
        assert any("residency was not declared" in reason
                   for reason in verdict.reasons)


def test_every_other_metal_family_refuses_a_fold_by_name():
    """DISJOINTNESS AS AN INVERSION, measured over the registered table.

    On a folded configuration exactly one arm may admit each of the six slots, and
    every arm that refuses must give a reason — a slot filled because the others
    were silent is a composer whose ambiguity detection is off.
    """
    from meep_gpu.metal_kernels import arms

    fields, pml = matrix.folded()
    context = arms.StepContext(fields, pml, device.Residency(),
                               (shaders.CONTRACT_OFF,), sources=())
    for slot in ("step_B", "step_D", "update_H", "update_E", "fill_B", "fill_D"):
        admitted = []
        for spec in arms.registered(slot):
            if not spec.wired:
                continue
            if spec.gate is not None and not spec.gate(context):
                continue
            try:
                verdict = spec.coverage(context, slot)
            except Exception:  # noqa: BLE001 - a raising predicate is a refusal
                continue
            if verdict.covered:
                admitted.append(spec.family)
            else:
                assert verdict.reasons, (slot, spec.family)
        assert admitted == [symmetry.FAMILY], (slot, admitted)


# ---------------------------------------------------------------------------
# 6. Behavioural — the fold's own hazards, on the device
# ---------------------------------------------------------------------------

SNAPSHOT = ("Bx", "By", "Bz", "Dx", "Dy", "Dz", "Ex", "Ey", "Ez",
            "Hx", "Hy", "Hz", "fu_Bx", "fu_By", "fu_Bz",
            "fu_Dx", "fu_Dy", "fu_Dz", "f_w_Hx", "f_w_Hy", "f_w_Hz",
            "f_w_Ex", "f_w_Ey", "f_w_Ez")


def _snapshot(fields):
    return {name: np.array(getattr(fields, name), copy=True)
            for name in SNAPSHOT if getattr(fields, name, None) is not None}


def _restore(fields, state):
    for name, array in state.items():
        getattr(fields, name)[...] = array


#: THE AXES A BEHAVIOURAL LEG MUST CARRY, each because a mutation is reachable on
#: one value and a measured null on the other:
#:
#: * TERMINATION — MIRROR_PERIODIC carries the top-plane mask and the far ghost
#:   pass, MIRROR_METALLIC carries neither (measured below: dropping the top-plane
#:   mask is caught on the first and a null on the second);
#: * PHASE — the parity is a SOURCE SPECIALISATION on this backend, so an odd plane
#:   is a DIFFERENT COMPILED FILL rather than a different scalar;
#: * FULL-COUNT PARITY — ``reflect_row`` is ``stored - 2`` at an even full count
#:   and ``stored - 3`` at an odd one, so at the default extent the wrong ``n - 2``
#:   formula HAPPENS TO BE RIGHT and a matrix carrying only it measures nothing
#:   about the reflect row (measured: 0 words at extent 2.0, 32 at extent 2.1);
#: * NUMBER OF FOLDED AXES and MIXED PHASE — a doubly-unowned corner must carry the
#:   product of both parities, which one folded axis cannot exercise at all.
FOLD_CASES = (
    ("periodic_even", dict()),
    ("periodic_odd", dict(phase=-1)),
    ("metallic_even", dict(boundaries={"y": "metallic"})),
    ("metallic_odd", dict(phase=-1, boundaries={"y": "metallic"})),
    ("x_fold", dict(axis="X")),
    ("odd_full_count", dict(extent=2.1)),
    ("two_axis_mixed_phase", dict(axis="XY", phase=(1, -1))),
    ("two_axis_mixed_phase_odd", dict(axis="XY", phase=(-1, 1), extent=2.1)),
)


@requires_mps
def test_every_shipped_specialisation_compiles_in_both_contraction_modes():
    """152 sources x 2 modes, measured rather than assumed.

    The curl's static product is 2 directions x 4^3 boundary quadruples = 128 and
    the fill's is 2 families x 3 axes x 2 phases x 2 passes = 24. The CORPUS drives
    12 of the first and at most 20 of the second, but a specialisation that fails to
    COMPILE is a crash at plan time on a configuration nobody swept, so the whole
    product is compiled here. Measured 2026-08-16: 304/304, ~1.1s warm.
    """
    from meep_gpu.metal_kernels.device import compile_source

    for mode in shaders.CONTRACT_MODES:
        sources = symmetry.enumerate_folded_sources(mode)
        assert len(sources) == 152, len(sources)
        for label, source in sources.items():
            compile_source(source)


@requires_mps
@pytest.mark.parametrize("sub_step", ("step_B", "step_D"))
@pytest.mark.parametrize("label,kwargs", FOLD_CASES, ids=[c[0] for c in FOLD_CASES])
def test_the_folded_curl_reproduces_stepping_bit_for_bit(label, kwargs, sub_step):
    """SUB-STEP granularity, which is where the two ownership masks live.

    THE MASKS ARE INVISIBLE AT WHOLE-STEP GRANULARITY — the driver's fill passes
    overwrite exactly the planes they protect — so a whole-step check certifies a
    mask-less kernel as correct. This leg is the one that can fail on a mask, and
    it runs BOTH terminations because ``drop_top_plane_mask`` is only reachable on
    MIRROR_PERIODIC.
    """
    fields, pml = matrix.folded(**kwargs)
    spec = launch.SUB_STEPS[sub_step]
    before = _snapshot(fields)
    getattr(stepping, sub_step)(fields, pml)
    after = _snapshot(fields)

    _restore(fields, before)
    residency = device.Residency()
    plan = symmetry.plan_folded_pml_curl(fields, pml, sub_step, residency)
    assert plan is not None, symmetry.folded_composition_curl_coverage(
        fields, pml, sub_step, residency).reasons
    residency.sync_in()
    plan.run()
    residency.sync_out()
    assert plan.launches == 1

    compared = tuple(spec["targets"]) + tuple("fu_" + n for n in spec["targets"])
    moved = sum(_moved(before[n], after[n]) for n in compared)
    assert moved > 100, (label, sub_step, moved,
                         "VACUOUS: the reference barely moved, so identity here "
                         "would be a no-op agreeing with a no-op")
    for name in compared:
        assert _differing(getattr(fields, name), after[name]) == 0, (label, name)


@requires_mps
@pytest.mark.parametrize("label,kwargs", FOLD_CASES, ids=[c[0] for c in FOLD_CASES])
def test_the_mirror_fill_reproduces_both_array_path_passes(label, kwargs):
    """THE MIRROR PLANE ITSELF — near and far, in the driver's own order.

    The array path applies the axes in X, Y, Z order so a corner unowned on two
    planes carries the PRODUCT of both parities; the plan holds a launch per axis
    and walks that order. The two passes are compared TOGETHER because that is what
    the driver does on an unwalled run, and separately below.
    """
    fields, pml = matrix.folded(**kwargs)
    for family, near_fill, far_fill in (
            ("B", stepping.fill_symmetry_bc_B, stepping.fill_folded_far_ghosts_B),
            ("D", stepping.fill_symmetry_bc_D, stepping.fill_folded_far_ghosts_D)):
        names = tuple(triton_symmetry.GHOST_FILL_FAMILIES[family]["targets"])
        before = {n: np.array(getattr(fields, n), copy=True) for n in names}
        near_fill(fields)
        far_fill(fields)
        after = {n: np.array(getattr(fields, n), copy=True) for n in names}

        for name in names:
            getattr(fields, name)[...] = before[name]
        residency = device.Residency()
        plan = symmetry.plan_mirror_ghost_fill(fields, family, "fill_" + family,
                                               residency)
        assert plan is not None, symmetry.mirror_ghost_fill_coverage(
            fields, family, residency).reasons
        residency.sync_in()
        plan.run_near()
        plan.run_far()
        residency.sync_out()
        assert plan.launches == len(plan.near) + len(plan.far) > 0

        moved = sum(_moved(before[n], after[n]) for n in names)
        assert moved > 0, (label, family, "VACUOUS: the fill wrote nothing")
        for name in names:
            assert _differing(getattr(fields, name), after[name]) == 0, \
                (label, family, name)


@requires_mps
@pytest.mark.parametrize("label,kwargs", FOLD_CASES, ids=[c[0] for c in FOLD_CASES])
def test_a_complete_step_agrees_and_the_first_divergent_step_is_reported(label,
                                                                        kwargs):
    """THE WHOLE-STEP LEG — the only one that can see the fold's fourth risk.

    Three failure classes live ONLY in a complete step: a STALE MIRROR (a sub-step
    left on the array path writes the HOST array and a device mirror held across it
    is a smooth, plausible, WRONG field); a SEAM (``zero_metal_*`` clears stored
    cell 0 between the curl and the constitutive sub-step); and an ACCUMULATING
    AUXILIARY (``fu_*`` and ``f_w_*`` are STATE, and a kernel right for one launch
    and wrong forever after is identical in a single-launch gate). THE FOLD ADDS A
    FOURTH: the folded axis's ghost plane is written by one pass and READ by the
    next, so a fold bug can be byte-perfect per sub-step and wrong per step.

    Every slot's launch counter is asserted, because a slot that passes by NOT
    EXECUTING is the hollow pass this whole discipline exists to make impossible.
    """
    budget = 6
    fields, pml = matrix.folded(**kwargs)
    reference_fields, reference_pml = matrix.folded(**kwargs)

    residency = device.Residency()
    plans = {
        "step_B": symmetry.plan_folded_pml_curl(fields, pml, "step_B", residency),
        "fill_B": symmetry.plan_mirror_ghost_fill(fields, "B", "fill_B", residency),
        "update_H": symmetry.plan_folded_constitutive(fields, pml, "H", residency),
        "step_D": symmetry.plan_folded_pml_curl(fields, pml, "step_D", residency),
        "fill_D": symmetry.plan_mirror_ghost_fill(fields, "D", "fill_D", residency),
        "update_E": symmetry.plan_folded_constitutive(fields, pml, "E", residency),
    }
    assert all(plan is not None for plan in plans.values()), {
        name: plan for name, plan in plans.items() if plan is None}
    residency.sync_in()

    first_divergent = None
    per_step = []
    for step in range(budget):
        # The reference: the driver's own order, five passes per half.
        stepping.step_B(reference_fields, reference_pml)
        stepping.fill_symmetry_bc_B(reference_fields)
        stepping.fill_folded_far_ghosts_B(reference_fields)
        stepping.update_H(reference_fields, reference_pml)
        stepping.step_D(reference_fields, reference_pml)
        stepping.fill_symmetry_bc_D(reference_fields)
        stepping.fill_folded_far_ghosts_D(reference_fields)
        stepping.update_E(reference_fields, reference_pml)

        plans["step_B"].run()
        plans["fill_B"].run_near()
        plans["fill_B"].run_far()
        plans["update_H"].run()
        plans["step_D"].run()
        plans["fill_D"].run_near()
        plans["fill_D"].run_far()
        plans["update_E"].run()
        residency.sync_out()

        differing = {name: _differing(getattr(fields, name),
                                      getattr(reference_fields, name))
                     for name in SNAPSHOT
                     if getattr(fields, name, None) is not None}
        total = sum(differing.values())
        per_step.append(total)
        if total and first_divergent is None:
            first_divergent = (step, {k: v for k, v in differing.items() if v})

    assert first_divergent is None, (label, "FIRST DIVERGENT STEP", first_divergent,
                                     "per-step differing words", per_step)
    # Vacuity: the state must actually have evolved over the budget.
    moved = sum(_moved(np.zeros_like(getattr(reference_fields, name)),
                       getattr(reference_fields, name))
                for name in ("Bx", "By", "Bz", "Ex", "Ey", "Ez"))
    assert moved > 100, (label, moved, "VACUOUS whole-step leg")
    # Every slot ran, every step.
    assert plans["step_B"].launches == budget
    assert plans["update_E"].launches == budget
    assert plans["fill_B"].launches == budget * (len(plans["fill_B"].near)
                                                 + len(plans["fill_B"].far))


@requires_mps
def test_the_top_plane_mask_is_measurably_load_bearing():
    """A GATE THAT CANNOT FAIL CERTIFIES NOTHING.

    Drop the top-plane mask and the sub-step must DIFFER on a MIRROR_PERIODIC run
    — otherwise the mask leg above is decorative. It must NOT differ on a
    MIRROR_METALLIC run, where the emitter produces no mask lines at all, and that
    half is asserted too so the leg is a discriminating test rather than a
    one-sided one.
    """
    from meep_gpu.metal_kernels.device import compile_source

    for kwargs, expect_caught in ((dict(), True),
                                  (dict(boundaries={"y": "metallic"}), False)):
        fields, pml = matrix.folded(**kwargs)
        codes, _ = symmetry.folded_axis_kinds(fields.grid, pml)
        before = _snapshot(fields)
        stepping.step_D(fields, pml)
        after = _snapshot(fields)
        _restore(fields, before)

        source = symmetry.folded_curl_source(codes, backward=True)
        mask = symmetry.folded_top_plane_mask(codes, backward=True)
        assert bool("curl" in mask) is expect_caught, (kwargs, mask)
        mutant = source.replace(mask, "    // MUTANT: top-plane mask dropped")
        assert (mutant != source) is expect_caught or not expect_caught

        residency = device.Residency()
        plan = symmetry.plan_folded_pml_curl(fields, pml, "step_D", residency)
        assert plan is not None
        plan._functions = {
            shaders.CONTRACT_OFF: compile_source(mutant).folded_pml_curl_step}
        residency.sync_in()
        plan.run()
        residency.sync_out()

        compared = ("Dx", "Dy", "Dz")
        caught = any(_differing(getattr(fields, n), after[n]) for n in compared)
        assert caught is expect_caught, (kwargs, "the top-plane mask mutation was "
                                         "expected to be %s" %
                                         ("caught" if expect_caught else "a null"))


@requires_mps
@pytest.mark.parametrize("extent,caught", ((2.0, False), (2.1, True)))
def test_the_reflect_row_mutation_is_caught_at_an_odd_full_count_only(extent,
                                                                     caught):
    """``_far_reflect_rows`` is ``n_full - stored + 2``, NOT ``n - 2``.

    At an EVEN full count the two agree and the wrong formula is a MEASURED NULL;
    at an ODD one it reflects about the window top instead of about the second
    mirror, which is a whole cell wrong. Measured on this host: 0 differing words
    at extent 2.0 (shape 12 on the folded axis, reflect 10 == n - 2) and 32 at
    extent 2.1 (shape 13, reflect 10 vs n - 2 = 11).

    THE POINT OF THE PARAMETRISATION IS THE FIRST ROW. A leg that ran only the
    even count would report the mutation uncaught and be indistinguishable from a
    kernel that ignores ``reflect_row`` entirely.
    """
    fields, pml = matrix.folded(extent=extent)
    names = ("Bx", "By", "Bz")
    before = {n: np.array(getattr(fields, n), copy=True) for n in names}
    stepping.fill_symmetry_bc_B(fields)
    stepping.fill_folded_far_ghosts_B(fields)
    after = {n: np.array(getattr(fields, n), copy=True) for n in names}
    for name in names:
        getattr(fields, name)[...] = before[name]

    near = list(symmetry.ghost_fill_axis_entries(fields.grid, "B", "near"))
    far = [dict(entry, reflect_row=int(fields.grid.shape[entry["axis"]]) - 2)
           for entry in symmetry.ghost_fill_axis_entries(fields.grid, "B", "far")]
    assert far, "no far pass: this leg would be vacuous"
    residency = device.Residency()
    plan = symmetry.plan_mirror_ghost_fill_from_arrays(
        "B", "fill_B", {n: getattr(fields, n) for n in names}, near, far,
        residency)
    residency.sync_in()
    plan.run_near()
    plan.run_far()
    residency.sync_out()

    moved = sum(_moved(before[n], after[n]) for n in names)
    assert moved > 0, "VACUOUS: the reference fill wrote nothing"
    differing = sum(_differing(getattr(fields, n), after[n]) for n in names)
    assert bool(differing) is caught, (extent, differing)


@requires_mps
def test_the_fill_axis_order_is_a_measured_null_on_the_real_fold():
    """A MEASURED NULL, and the reachability is asserted so it is not a vacuous one.

    The array path applies the axes in X, Y, Z order so a doubly-unowned corner
    carries the product of both parities. Under REAL storage every fill is a
    multiply by exactly +/-1 — on this backend, ``-x`` (a sign-bit operation) or a
    plain copy — so two axes' fills COMMUTE BITWISE and the order cannot change a
    word. That is measured here rather than assumed, over a TWO-AXIS MIXED-PHASE
    fold where the corner really is written twice: ``Bz`` has Yee shift 1 on both
    x and y, so ``Bz[-1, -1, k]`` is written by the far pass on BOTH axes.

    UNDER COMPLEX STORAGE THE SAME QUESTION IS NOT A NULL — the Triton track
    measured 8 of 128 diverging words at mixed phase, because a complex multiply by
    (+/-1, +0) is not associative. That is the folded complex family's problem and
    is why this null may not be generalised.
    """
    fields, pml = matrix.folded(axis="XY", phase=(1, -1))
    names = ("Bx", "By", "Bz")
    before = {n: np.array(getattr(fields, n), copy=True) for n in names}
    stepping.fill_symmetry_bc_B(fields)
    stepping.fill_folded_far_ghosts_B(fields)
    after = {n: np.array(getattr(fields, n), copy=True) for n in names}
    for name in names:
        getattr(fields, name)[...] = before[name]

    near = list(reversed(symmetry.ghost_fill_axis_entries(fields.grid, "B", "near")))
    far = list(reversed(symmetry.ghost_fill_axis_entries(fields.grid, "B", "far")))
    assert len(far) == 2, "REACHABILITY: two folded PERIODIC axes are required"
    assert [entry["phase"] for entry in far] == [-1, 1], (
        "REACHABILITY: the two parities must DIFFER, or the product is the same "
        "number either way for a trivial reason")
    residency = device.Residency()
    plan = symmetry.plan_mirror_ghost_fill_from_arrays(
        "B", "fill_B", {n: getattr(fields, n) for n in names}, near, far,
        residency)
    residency.sync_in()
    plan.run_near()
    plan.run_far()
    residency.sync_out()

    moved = sum(_moved(before[n], after[n]) for n in names)
    assert moved > 0, "VACUOUS: the reference fill wrote nothing"
    differing = sum(_differing(getattr(fields, n), after[n]) for n in names)
    assert differing == 0, (differing, "the axis order was expected to be a null "
                            "under real storage; if this fires, the parity is no "
                            "longer an exact sign flip on this backend")


@requires_mps
def test_a_flipped_parity_is_caught_on_every_fold_case():
    """The other half of the null above: the fill's SIGN is load-bearing.

    A leg whose only mutation is a measured null certifies nothing. Flipping the
    plane's declared phase must move words on every case, including the metallic
    termination where only the near pass runs.
    """
    for label, kwargs in FOLD_CASES:
        fields, pml = matrix.folded(**kwargs)
        names = ("Bx", "By", "Bz")
        before = {n: np.array(getattr(fields, n), copy=True) for n in names}
        stepping.fill_symmetry_bc_B(fields)
        stepping.fill_folded_far_ghosts_B(fields)
        after = {n: np.array(getattr(fields, n), copy=True) for n in names}
        for name in names:
            getattr(fields, name)[...] = before[name]

        flip = lambda entries: [dict(e, phase=-e["phase"]) for e in entries]  # noqa: E731
        residency = device.Residency()
        plan = symmetry.plan_mirror_ghost_fill_from_arrays(
            "B", "fill_B", {n: getattr(fields, n) for n in names},
            flip(symmetry.ghost_fill_axis_entries(fields.grid, "B", "near")),
            flip(symmetry.ghost_fill_axis_entries(fields.grid, "B", "far")),
            residency)
        residency.sync_in()
        plan.run_near()
        plan.run_far()
        residency.sync_out()
        differing = sum(_differing(getattr(fields, n), after[n]) for n in names)
        assert differing > 0, (label, "a flipped parity was NOT caught")


@requires_mps
def test_the_folded_curl_reduces_to_the_certified_curl_on_an_unfolded_grid():
    """The reduction, MEASURED rather than argued from the source text alone."""
    fields, pml = matrix.cart()
    codes = [1 if kind == "metallic" else 0
             for kind in stepping._boundary_kinds(fields.grid, pml)]
    before = _snapshot(fields)

    residency = device.Residency()
    certified = launch.plan_pml_curl(fields, pml, "step_B", residency)
    assert certified is not None
    residency.sync_in()
    certified.run()
    residency.sync_out()
    plain = _snapshot(fields)

    _restore(fields, before)
    residency = device.Residency()
    folded = symmetry.plan_folded_pml_curl_from_arrays(
        "step_B",
        {name: getattr(fields, name) for name in SNAPSHOT
         if getattr(fields, name, None) is not None},
        {f"{stem}_{axis}": getattr(pml, f"{stem}_{axis}_h")
         for axis in "xyz" for stem in ("kms", "sinv")},
        codes, float(fields.grid.dt / fields.grid.dx), residency)
    residency.sync_in()
    folded.run()
    residency.sync_out()

    moved = sum(_moved(before[n], plain[n]) for n in ("Bx", "By", "Bz"))
    assert moved > 100, (moved, "VACUOUS reduction leg")
    for name in ("Bx", "By", "Bz", "fu_Bx", "fu_By", "fu_Bz"):
        assert _differing(getattr(fields, name), plain[name]) == 0, name


@requires_mps
@pytest.mark.parametrize("phase", (1, -1))
@pytest.mark.parametrize("label,scale", (("physical", 1.0),
                                         ("subnormal_band", 1e-38),
                                         ("deep", 1e-40)))
def test_the_fill_is_band_safe_and_the_curl_is_not(label, scale, phase):
    """WHAT THE SUBNORMAL PRECONDITION ACTUALLY BOUNDS, measured rather than
    assumed for the whole family at once.

    On MPS the float32 subnormal flush is native and has no lever, so every
    ARITHMETIC claim rides on a checked subnormal-free precondition. THE FILL
    PERFORMS NO ARITHMETIC: its parity is a sign-bit operation or a plain copy, and
    a subnormal CAN live in a Metal buffer — it is the arithmetic that flushes, not
    the storage. So the fill is expected to be identical INSIDE the band, and it is:
    measured 2026-08-16 with 576 subnormal operand words in play, 0 differing at
    1e-38 and at 1e-40, at BOTH parities.

    THE CURL IS THE OPPOSITE and the contrast is the point — same grid, same scale,
    1,152 subnormal operand words, 1,104 differing. A family-wide "byte-identical
    subject to a subnormal-free precondition" would understate the fill and
    overstate nothing; stating it per sub-step is what makes the precondition a
    measurement rather than a blanket.
    """
    from meep_gpu.metal_kernels import subnormal as subnormal_module

    fields, pml = matrix.folded(phase=phase)
    names = ("Bx", "By", "Bz")
    for name in names:
        getattr(fields, name)[...] = (
            getattr(fields, name) * np.float32(scale)).astype(np.float32)
    census = sum(subnormal_module.census(getattr(fields, n)) for n in names)
    assert (census > 0) is (scale != 1.0), (label, census,
                                            "the band control did not reach the band")

    before = {n: np.array(getattr(fields, n), copy=True) for n in names}
    stepping.fill_symmetry_bc_B(fields)
    stepping.fill_folded_far_ghosts_B(fields)
    after = {n: np.array(getattr(fields, n), copy=True) for n in names}
    for name in names:
        getattr(fields, name)[...] = before[name]

    residency = device.Residency()
    plan = symmetry.plan_mirror_ghost_fill(fields, "B", "fill_B", residency)
    assert plan is not None
    residency.sync_in()
    plan.run_near()
    plan.run_far()
    residency.sync_out()

    moved = sum(_moved(before[n], after[n]) for n in names)
    assert moved > 0, (label, "VACUOUS: the reference fill wrote nothing")
    differing = sum(_differing(getattr(fields, n), after[n]) for n in names)
    assert differing == 0, (label, phase, census, differing)


@requires_mps
def test_the_curl_does_need_the_precondition():
    """The other half of the leg above: the CURL diverges in the band.

    A leg that only showed the fill clean would read as "this family is band-safe",
    which is false of the arithmetic half of it. Measured on the same grid at the
    same scale: 1,152 subnormal operand words in, 1,104 differing words out.
    """
    from meep_gpu.metal_kernels import subnormal as subnormal_module

    fields, pml = matrix.folded()
    for name in ("Ex", "Ey", "Ez", "Bx", "By", "Bz",
                 "fu_Bx", "fu_By", "fu_Bz"):
        getattr(fields, name)[...] = (
            getattr(fields, name) * np.float32(1e-38)).astype(np.float32)
    census = sum(subnormal_module.census(getattr(fields, n))
                 for n in ("Ex", "Ey", "Ez", "Bx", "By", "Bz"))
    assert census > 0, "the control did not reach the band"

    names = ("Bx", "By", "Bz", "fu_Bx", "fu_By", "fu_Bz")
    before = {n: np.array(getattr(fields, n), copy=True) for n in names}
    stepping.step_B(fields, pml)
    after = {n: np.array(getattr(fields, n), copy=True) for n in names}
    for name in names:
        getattr(fields, name)[...] = before[name]

    residency = device.Residency()
    plan = symmetry.plan_folded_pml_curl(fields, pml, "step_B", residency)
    assert plan is not None
    residency.sync_in()
    plan.run()
    residency.sync_out()
    differing = sum(_differing(getattr(fields, n), after[n]) for n in names)
    assert differing > 0, (
        census, "the curl was expected to DIVERGE in the subnormal band; if it "
        "does not, the cliff the whole precondition rests on was not reproduced "
        "and the precondition is decorative")


@requires_mps
def test_the_fold_planes_carry_no_subnormal_over_a_stated_window():
    """THE PRECONDITION IS A WINDOW, NOT A SCALAR, and it is taken ON THE PLANES.

    A folded run puts a deep-PML plane on the far face, which is exactly where tiny
    magnitudes live, and the fill reads and writes exactly those planes (stored row
    0, row 2, ``reflect_row``, and the last stored slot). A whole-volume aggregate
    that a 1e-40 boundary plane cannot move would report clean and prove nothing,
    so the census is per plane and it reports the FIRST and LAST step it fired —
    the chi3 measurement is the shape to expect (first at step 55, LAST at 3,726,
    then clean for 16,274 more).
    """
    from meep_gpu.metal_kernels.preconditions import SubnormalWindow

    budget = 8
    fields, pml = matrix.folded()
    window = SubnormalWindow(first_step=0, last_step=budget - 1,
                             per_array_words=64)
    hits = []
    for step in range(budget):
        stepping.step_B(fields, pml)
        stepping.fill_symmetry_bc_B(fields)
        stepping.fill_folded_far_ghosts_B(fields)
        stepping.update_H(fields, pml)
        stepping.step_D(fields, pml)
        stepping.fill_symmetry_bc_D(fields)
        stepping.fill_folded_far_ghosts_D(fields)
        stepping.update_E(fields, pml)
        for name in ("Bx", "By", "Bz", "Dx", "Dy", "Dz"):
            volume = getattr(fields, name)
            for label, plane in (("near_write", volume[:, 0, :]),
                                 ("near_read", volume[:, 2, :]),
                                 ("far_write", volume[:, -1, :])):
                if window.observe(f"{name}/{label}", plane, step=step):
                    hits.append(step)
    report = window.report()
    assert not report["vacuous"], report["vacuity_reasons"]
    assert report["clean"], {
        "window": [min(hits), max(hits)] if hits else None,
        "subnormal_words": report["subnormal_words"],
        "note": "REFUSED by the subnormal precondition: on MPS the flush is native "
                "and has no lever, so byte-identity is not claimable for this case",
    }
