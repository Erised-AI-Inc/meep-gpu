"""What ``dispersive_kernels.py`` must be true of on a laptop, before a device slot is spent.

THREE THINGS THIS FILE PINS, and none of them is "the kernel is right" -- only a
device gate can say that (``parity/meep_gpu/gate_cuda_dispersive.py``):

1. **THE EMITTED TEXT.** The pole chain is sequential and left-to-right, the
   accumulation is two separate statements, ``prev`` is loaded before the store,
   component *c* reads axis *c*'s coefficient vector, and the arity-(0,0,0) body
   reduces CHARACTER FOR CHARACTER to the certified non-dispersive kernel's three
   source lines. An emitter is a machine that writes kernels; what a reader can
   check by eye on one hand-written body has to be checked mechanically here.

2. **THE PREDICATES' ADMITTED SET, AND ITS DISJOINTNESS.** Every clause is
   exercised against an object that trips it, by name, and the whole E-side
   family -- the certified diagonal one, the off-diagonal one, the no-PML null
   one and this file's two -- is checked to admit AT MOST ONE arm per
   configuration. Two arms admitting one slot is a widening past the evidence,
   not extra coverage, and the union census reports it as a FINDING.

3. **THE TRANSCRIPTION, MEASURED AGAINST ``stepping`` ITSELF.** A NumPy
   transcription of the emitted body is run against ``stepping.update_E`` on real
   ``Grid``/``Fields``/``PML``/``PolarizationState`` objects, bit for bit as
   uint32, at every swept arity on both arms -- and the two pole-order defects
   are planted to show the comparison can FAIL. That last part is the point: the
   pair is INVISIBLE AT ONE POLE and lethal at two, which is measured here rather
   than quoted, so a gate leg that scored them at a single pole would be caught
   as a harness defect on a laptop instead of shipping as a kernel pass.

WHAT THIS FILE COMPILES: nothing. It imports no CuPy, launches no kernel and
certifies nothing. ``dispersive_kernels`` imports CuPy lazily for exactly this
reason -- and because the predicate census runs on a laptop too, where a family
that cannot be imported is counted as a coverage gap it does not have.
"""

from __future__ import annotations

import ast
import json
import pathlib
import re

import numpy as np
import pytest

from meep_gpu import stepping
from meep_gpu.cuda_kernels import coverage, dispersive_kernels as dispersive
from meep_gpu.cuda_kernels import no_pml_constitutive
from meep_gpu.dispersion import PolarizationState, Susceptibility
from meep_gpu.fields import Fields
from meep_gpu.grid import Grid, Mirror
from meep_gpu.pml import PML

HERE = pathlib.Path(__file__).resolve().parent


class NumpyWearingCupysName:
    """NumPy behind CuPy's ``__name__``.

    The predicates' first question is whether the backend is CuPy at all, and
    that is the one thing about the device library a laptop cannot supply.
    Everything else they read -- the fold, the stored extent, the coefficient
    vector lengths, the dtype and the contiguity -- is a real object either way.
    Same shim the CUDA gates use, so the laptop legs stay commensurable.
    """

    __name__ = "cupy"

    def __getattr__(self, item):
        return getattr(np, item)


# ---------------------------------------------------------------------------
# The fixture -- real engine objects, never a duck type
# ---------------------------------------------------------------------------

def build(poles=2, pml_on=True, boundaries=("periodic", "periodic", "periodic"),
          cell=(8.0, 10.0, 12.0), courant=0.35, seed=0, mirrors=(),
          kinds=None, drives=None):
    """A frozen ``(fields, layer, grid)`` triple carrying ``poles`` susceptibilities.

    ``drives`` optionally restricts which components each state drives, through a
    per-component sigma mapping -- ``PolarizationState`` filters on a trivial
    sigma exactly as MEEP's ``needs_P`` does (dispersion.py:645-647), so a
    per-component zero is the engine's own way of spelling "this term does not
    drive that component" and never a fixture-only hack.
    """
    xp = NumpyWearingCupysName()
    kwargs = {}
    if mirrors:
        kwargs["symmetry"] = [Mirror("XYZ"[axis]) for axis in mirrors]
    grid = Grid(resolution=1.0, cell_size=tuple(cell), boundaries=tuple(boundaries),
                xp=xp, courant=courant, **kwargs)
    # THE LOW FACE IS DROPPED ON A MIRRORED AXIS, and not as a fixture nicety:
    # cell 0 of a folded axis lies ON the mirror plane, which is a boundary
    # condition rather than an outer wall, and ``PML`` refuses a per-side request
    # that names it (pml.py:399-412). A folded case graded from both walls would
    # be absorbing across the mirror plane.
    thickness = {"xyz"[axis]: ((0, 0) if grid.shape[axis] < 6
                               else (0, 2) if grid.is_mirrored(axis) else (2, 2))
                 for axis in range(3)}
    layer = PML(grid=grid, thickness=thickness) if pml_on else None
    fields = Fields(grid=grid)
    fields.enable_field_storage()
    if pml_on:
        fields.enable_pml_storage()

    rng = np.random.default_rng(seed)
    epsilon, inverse = {}, {}
    for component in ("Ex", "Ey", "Ez"):
        # THREE INDEPENDENT VOLUMES, never one array bound thrice: binding one is
        # defect 2 of the complex template, and a fixture that collapsed them
        # could not see it.  Drawn AWAY from 1.0, or dropping the multiply
        # entirely would be bit-identical.
        values = rng.uniform(1.2, 3.4, size=grid.shape).astype(np.float32)
        epsilon[component] = values
        inverse[component] = (np.float32(1.0) / values).astype(np.float32)
    fields.set_epsilon_volumes(epsilon, inverse)

    kinds = kinds or ["lorentzian", "drude"]
    for index in range(poles):
        term = Susceptibility(frequency=0.20 + 0.05 * index, gamma=0.01,
                              kind=kinds[index % len(kinds)])
        if drives is None:
            sigma = 0.4 + 0.05 * index
        else:
            sigma = {name: (0.4 + 0.05 * index if name in drives[index] else 0.0)
                     for name in ("Ex", "Ey", "Ez")}
        fields.polarizations.append(
            PolarizationState(term, sigma, grid, fields._field_dtype()))

    names = ["Ex", "Ey", "Ez", "Dx", "Dy", "Dz"]
    if pml_on:
        names += ["f_w_Ex", "f_w_Ey", "f_w_Ez"]
    for name in names:
        # THE AUXILIARIES START NONZERO. A zero f_w makes ``kms * prev`` exactly
        # zero on the first launch whatever kms holds, so a mis-indexed
        # coefficient would only show from step two.
        getattr(fields, name)[...] = rng.uniform(-1.0, 1.0,
                                                 size=grid.shape).astype(np.float32)
    for state in fields.polarizations:
        for component in state.driven():
            state.P[component][...] = rng.uniform(
                -1.0, 1.0, size=grid.shape).astype(np.float32)
            state.P_prev[component][...] = rng.uniform(
                -1.0, 1.0, size=grid.shape).astype(np.float32)
    return fields, layer, grid


def outputs(arm):
    return (("Ex", "Ey", "Ez", "f_w_Ex", "f_w_Ey", "f_w_Ez") if arm == "pml"
            else ("Ex", "Ey", "Ez"))


def snapshot(fields, arm):
    return {name: getattr(fields, name).copy() for name in outputs(arm)}


def restore(fields, frozen):
    for name, values in frozen.items():
        getattr(fields, name)[...] = values


def moved_words(before, after):
    """Output WORDS the step changed, as raw uint32 -- never ``allclose``.

    ``-0.0 == 0.0`` and ``NaN != NaN`` both lie, and this family's whole hazard
    is a step that leaves every word at ``+0.0`` and reports IDENTICAL against
    any reference at all.
    """
    return sum(int(np.count_nonzero(
        np.ascontiguousarray(before[name]).ravel().view(np.uint32)
        != np.ascontiguousarray(after[name]).ravel().view(np.uint32)))
        for name in before)


def tables_for(layer):
    return {f"{stem}_{axis}": getattr(layer, f"{stem}_{axis}_h").reshape(-1)
            for axis in "xyz" for stem in ("kps", "kms")}


def transcribed_kernel(arm, fields, tables, plan, defect=None):
    """The EMITTED device body, transcribed to NumPy float32, line for line.

    Four statements on arm 1 -- the four ``constitutive_apply`` spells and the
    pole chain that feeds it -- and one store on arm 2. This compiles nothing and
    certifies nothing; what it settles is whether the transcription reproduces
    ``stepping`` at all, and whether this harness can fail, both before a device
    slot is spent.
    """
    for axis_index, (target, source, axis) in enumerate(dispersive.ELECTRIC_TERMS):
        chain = list(plan[target])
        if defect == "reverse_pole_order":
            chain = list(reversed(chain))
        if defect == "drop_one_pole" and chain:
            chain = chain[:-1]
        if defect == "sum_then_subtract" and chain:
            total = np.asarray(chain[0]).astype(np.float32)
            for array in chain[1:]:
                total = (total + array).astype(np.float32)
            value = (getattr(fields, source) - total).astype(np.float32)
        else:
            value = getattr(fields, source).astype(np.float32)
            for array in chain:
                value = (value - array).astype(np.float32)
        if defect == "drop_inverse_epsilon_dispersive":
            product = value
        elif defect == "inv_eps_left":
            product = (fields.inverse_epsilon_for(target) * value).astype(np.float32)
        else:
            product = (value * fields.inverse_epsilon_for(target)).astype(np.float32)
        if arm == "no_pml":
            getattr(fields, target)[...] = product
            continue
        shape = [1, 1, 1]
        shape[axis_index] = -1
        kps = np.asarray(tables[f"kps_{axis}"]).reshape(shape)
        kms = np.asarray(tables[f"kms_{axis}"]).reshape(shape)
        auxiliary = getattr(fields, "f_w_" + target)
        field = getattr(fields, target)
        previous = auxiliary.copy()
        auxiliary[...] = product
        accumulated = (field + kps * product).astype(np.float32)
        field[...] = (accumulated - kms * previous).astype(np.float32)


# ---------------------------------------------------------------------------
# 1. THE EMITTED TEXT
# ---------------------------------------------------------------------------

def test_the_zero_arity_body_reduces_to_the_certified_kernels_own_source_lines():
    """Arity (0,0,0) IS the certified non-dispersive body, as a string.

    ``displacement_minus_polarization`` returns the D array ITSELF when nothing
    drives a component (fields.py:1096-1098) -- no scratch, no subtraction, no
    intermediate. The header claims the emitted body reduces to certified bytes
    "by construction rather than by rounding"; this is the construction, checked
    rather than asserted, and it is what makes the degenerate configuration a
    disjointness decision instead of a second engine.
    """
    certified = (HERE / "constitutive_kernels.py").read_text(encoding="utf-8")
    start = certified.index("    float src_x = Dx[idx]")
    block = certified[start:certified.index("\n\n", start)]
    assert dispersive._source_lines((0, 0, 0)) == block


@pytest.mark.parametrize("counts", dispersive.POLE_COUNTS_SWEPT)
@pytest.mark.parametrize("arm", dispersive.DISPERSIVE_ARMS)
def test_the_pole_chain_is_sequential_left_to_right_and_never_a_running_sum(arm, counts):
    """``((D - P0) - P1) - P2``, one statement per contributor, in order.

    The pre-accumulated ``D - (P0 + P1 + ...)`` is the form a reader of the
    phrase "D minus the sum of the polarizations" writes by accident, and float32
    addition is not associative. Pinned as TEXT because the arithmetic is the
    text: there is no other artifact on a laptop to check it against.
    """
    source = dispersive.dispersive_source(arm, counts)
    for (target, sourcename, axis), count in zip(dispersive.ELECTRIC_TERMS, counts):
        subtractions = re.findall(
            rf"^    s_{axis} = s_{axis} - (P_{target}_\d+)\[idx\];$",
            source, flags=re.M)
        assert subtractions == [f"P_{target}_{index}" for index in range(count)], (
            f"{arm} {counts}: {target}'s chain is {subtractions}, not the "
            f"in-order sequential chain fields.py:1102-1103 walks")
        if count:
            assert f"    float s_{axis} = {sourcename}[idx];" in source
    assert " + P_" not in source, "a pole is being ADDED; the chain subtracts"


@pytest.mark.parametrize("counts", dispersive.POLE_COUNTS_SWEPT)
def test_the_pml_accumulation_stays_two_statements_with_prev_read_before_the_store(counts):
    """``float a = f + kps*src; f = a - kms*prev;`` and ``prev`` loaded first.

    The flattened ``f + (kps*src - kms*prev)`` is a different float32 number and
    is what both complex constitutive kernels are written as; reading ``prev``
    after the store loses the history term and reads as a slightly weaker
    absorber. Both are gate mutations that MUST be caught, and both are one
    editing accident away in an emitter that grew a pole chain above them.
    """
    source = dispersive.dispersive_source("pml", counts)
    assert ("    float prev = fw[idx];\n"
            "    fw[idx] = src;\n"
            "    float a = f[idx] + kps * src;\n"
            "    f[idx] = a - kms * prev;\n") in source


@pytest.mark.parametrize("counts", dispersive.POLE_COUNTS_SWEPT)
def test_component_c_reads_axis_cs_coefficient_vector(counts):
    """``kps_x[i]``, ``kps_y[j]``, ``kps_z[k]`` -- E_CONSTITUTIVE_TERMS, stepping.py:228.

    MEEP's ``dsigw`` is the absorption a component accumulates along the
    direction IT POINTS IN. Every component reading the x table instead is the
    sibling module's "highest-consequence confusion in the file": a smooth,
    converged, entirely wrong absorber, and the gate arms it as
    ``own_axis_to_x_for_all_three``.
    """
    source = dispersive.dispersive_source("pml", counts)
    assert "constitutive_apply(Ex, f_w_Ex, idx, src_x, kps_x[i], kms_x[i]);" in source
    assert "constitutive_apply(Ey, f_w_Ey, idx, src_y, kps_y[j], kms_y[j]);" in source
    assert "constitutive_apply(Ez, f_w_Ez, idx, src_z, kps_z[k], kms_z[k]);" in source
    assert ("    int k = idx % nz;\n"
            "    int j = (idx / nz) % ny;\n"
            "    int i = idx / (ny * nz);\n") in source, (
        "the arrays are C-contiguous -- the predicate refuses anything else -- so "
        "the last axis is the fastest")


@pytest.mark.parametrize("counts", dispersive.POLE_COUNTS_SWEPT)
def test_the_no_pml_arm_carries_no_auxiliary_no_coefficient_and_no_accumulation(counts):
    """Arm 2 is a STORE (stepping.py:1022), which is why it is a second string.

    A kernel serving both branches behind a flag would be two kernels wearing one
    name, and the flag would be the only thing standing between a layerless run
    and an absorber profile it has no coefficient vectors for.
    """
    source = dispersive.dispersive_source("no_pml", counts)
    for absent in ("f_w_", "kps_", "kms_", "constitutive_apply", "int nx"):
        assert absent not in source, f"{absent!r} leaked into the no-PML arm"
    assert "    Ex[idx] = src_x;\n    Ey[idx] = src_y;\n    Ez[idx] = src_z;\n" in source


@pytest.mark.parametrize("counts", dispersive.POLE_COUNTS_SWEPT)
@pytest.mark.parametrize("arm", dispersive.DISPERSIVE_ARMS)
def test_the_inverse_epsilon_pointers_are_not_restrict_and_the_pole_pointers_are(arm, counts):
    """An isotropic run hands ONE inv_eps pointer three times; two poles never share one.

    ``restrict`` on mutually aliasing arguments is a promise the caller cannot
    keep and the compiler is licensed to miscompile. ``set_isotropic_epsilon_volume``
    (fields.py:1321-1326) makes the epsilon case real on ordinary runs;
    ``PolarizationState.__init__`` (dispersion.py:645-647) allocates a P buffer
    per state, which is what pays for the pole pointers' promise -- and the
    predicate and the launcher both check it rather than trusting the sentence.
    """
    source = dispersive.dispersive_source(arm, counts)
    for target, _source, _axis in dispersive.ELECTRIC_TERMS:
        assert f"    const float* inv_eps_{target}" in source or \
               f"const float* inv_eps_{target}," in source
        assert f"__restrict__ inv_eps_{target}" not in source
    for name in dispersive.pole_parameter_names(counts):
        assert f"const float* __restrict__ {name}," in source


def test_an_arity_past_the_measured_cap_is_refused_by_name_not_clamped():
    """``POLE_COUNT_CAP`` is what the gate swept, and emitting past it is an extrapolation."""
    assert dispersive.POLE_COUNT_CAP == max(
        max(counts) for counts in dispersive.POLE_COUNTS_SWEPT), (
        "the cap and the swept ceiling have drifted apart; the cap is only "
        "meaningful as the largest arity a device actually ran")
    with pytest.raises(ValueError, match="POLE_COUNT_CAP"):
        dispersive.dispersive_source("pml", (dispersive.POLE_COUNT_CAP + 1, 0, 0))
    with pytest.raises(ValueError, match="negative"):
        dispersive.dispersive_source("pml", (-1, 0, 0))
    with pytest.raises(ValueError, match="one count per E component"):
        dispersive.dispersive_source("pml", (1, 1))


def test_the_corpus_arities_this_family_serves_are_all_swept():
    """1, 2, 5 and 6 are the arities the eleven corpus rows actually carry.

    Measured from ``parity/meep_gpu/results/cuda_predicate_coverage_2026-08-20_closeout/``:
    ``absorbed_power_density.py`` 1; ``material-dispersion.py`` 2;
    ``TestLoadDump.*_2d``, ``absorber-1d.py`` and ``TestAbsorber.test_absorber``
    5; ``stochastic_emitter*.py`` 6. A sweep that missed one would leave a corpus
    row's own body unrun while the predicate admitted it.
    """
    swept = {count for counts in dispersive.POLE_COUNTS_SWEPT for count in counts}
    assert {1, 2, 5, 6} <= swept


def _module_level_tuple(source: str, name: str):
    """Read a module-level literal off the SYNTAX TREE, without importing.

    ``ade_kernels.py`` and ``constitutive_kernels.py`` import CuPy at module
    scope, so on the merge bar they cannot be imported at all. The option tuple
    is still a fact about their text, and this is how a laptop reads it.
    """
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == name
                for target in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} is not a module-level assignment")


def test_the_contraction_guard_is_spelled_here_and_matches_every_sibling():
    """``--fmad=false`` is CORRECTNESS on this sub-step, and is not imported.

    A bit-identity probe loads these modules BY PATH, outside the package, where
    an import could pick up a different tuple than the one a gate compiled. Every
    certified sibling spells its own for that reason; this pins that the
    spellings AGREE, so "the same guard" is a fact rather than a habit -- and
    that this module did not quietly drop it while gaining three more
    contraction candidates per component than the sibling has.
    """
    assert dispersive._COMPILE_OPTIONS == ("--fmad=false",)
    for sibling in ("constitutive_kernels.py", "ade_kernels.py"):
        text = (HERE / sibling).read_text(encoding="utf-8")
        assert _module_level_tuple(text, "_COMPILE_OPTIONS") == \
            dispersive._COMPILE_OPTIONS, sibling
    assert "from .constitutive_kernels import" not in \
        (HERE / "dispersive_kernels.py").read_text(encoding="utf-8"), (
        "the option tuple must be SPELLED here, not imported")


def test_every_emitted_kernel_name_is_in_exactly_one_half_of_the_partition():
    """A kernel in neither set ships unmeasured; a kernel in both is a record bug.

    ``UNCERTIFIED_KERNELS`` is read off the SYNTAX TREE and must stay
    UNANNOTATED, for the reason ``complex_pml_kernels.py`` spells its own that
    way: an annotated assignment is an ``ast.AnnAssign``, which this reader does
    not match, and annotating it made the name invisible and the partition
    unenforced.
    """
    text = (HERE / "dispersive_kernels.py").read_text(encoding="utf-8")
    dead = _module_level_tuple(text, "UNCERTIFIED_KERNELS")
    live = set(dispersive.CERTIFIED_KERNELS)
    emitted = {name for arm in dispersive.DISPERSIVE_ARMS
               for name in re.findall(
                   r'extern "C" __global__ void (\w+)\(',
                   dispersive.dispersive_source(arm, (1, 1, 1)))}
    assert emitted == set(dispersive.ARM_KERNEL_NAMES.values())
    assert live & set(dead) == set(), "a kernel is in BOTH halves of the partition"
    assert emitted == live | set(dead), (
        f"emitted={sorted(emitted)} but the partition covers "
        f"{sorted(live | set(dead))}; a kernel in neither set has no gate verdict "
        f"and no statement that it lacks one")


def test_an_uncertified_kernel_has_no_certification_record_block():
    """Moving a name into ``CERTIFIED_KERNELS`` without a record block must fail here.

    The failure this guards is the cheap one: a gate runs green on a device,
    somebody promotes the name, and the record that says WHICH artifact licensed
    it is never written -- so six weeks later the verdict is a memory.
    """
    blocks = json.loads((HERE / "certification.json").read_text(encoding="utf-8"))
    text = (HERE / "dispersive_kernels.py").read_text(encoding="utf-8")
    for name in _module_level_tuple(text, "UNCERTIFIED_KERNELS"):
        assert name not in blocks, (
            f"{name} is declared UNCERTIFIED and yet certification.json carries a "
            f"block for it; one of the two is describing a run that did not happen")
    # BLOCKS ARE FAMILY-DATED AND CARRY A certified_kernels LIST; they are NOT keyed by
    # kernel name. constitutive_2026-08-15 has done it that way since the record existed,
    # so looking the kernel up as a top-level key asked the record a question it never
    # answered -- it only passed while this family had no block at all. The INTENT is
    # unchanged and is what is checked: a name promoted to CERTIFIED must be named by
    # some block, or the artifact that licensed it is a memory.
    certified_by_record = set()
    for block in blocks.values():
        if isinstance(block, dict):
            certified_by_record |= set(block.get("certified_kernels") or ())
    for name in dispersive.CERTIFIED_KERNELS:
        assert name in certified_by_record, (
            f"{name} is declared CERTIFIED with no certification.json block naming "
            f"the gate artifact that licensed it")


# ---------------------------------------------------------------------------
# 2. THE PREDICATES
# ---------------------------------------------------------------------------

def test_the_pml_arm_admits_the_configuration_its_eight_corpus_rows_carry():
    fields, layer, grid = build(poles=2, pml_on=True)
    assert dispersive.covers_real_pml_dispersive_constitutive(
        fields, layer, grid) == (True, "covered")


def test_the_no_pml_arm_admits_the_configuration_its_three_corpus_rows_carry():
    fields, layer, grid = build(poles=5, pml_on=False)
    assert dispersive.covers_no_pml_dispersive_constitutive(
        fields, layer, grid) == (True, "covered")


def test_a_mirror_fold_is_admitted_on_both_arms_and_three_planes_are_not():
    """Five of arm 1's eight corpus rows are folded; the budget is the sibling's TWO.

    The fold reaches this sub-step only through the STORED EXTENT, and the extent
    is what bounds both the field index and the coefficient vector -- which is why
    ``CONSTITUTIVE_FOLD_ADMISSION`` measured two simultaneous planes and why a
    third would be an extrapolation from that measurement rather than part of it.
    """
    fields, layer, grid = build(poles=2, pml_on=True, mirrors=(1,))
    assert dispersive.covers_real_pml_dispersive_constitutive(
        fields, layer, grid)[0] is True
    fields, layer, grid = build(poles=2, pml_on=True, mirrors=(0, 1))
    assert dispersive.covers_real_pml_dispersive_constitutive(
        fields, layer, grid)[0] is True
    fields, layer, grid = build(poles=2, pml_on=True, mirrors=(0, 1, 2))
    covered, reason = dispersive.covers_real_pml_dispersive_constitutive(
        fields, layer, grid)
    assert covered is False and "3 mirror planes at once" in reason


@pytest.mark.parametrize("mutate,needle", [
    (lambda f, l, g: setattr(f, "force_complex_fields", True), "complex64 storage"),
    (lambda f, l, g: f.polarizations.clear(), "no susceptibility is registered"),
    (lambda f, l, g: setattr(f, "_stored_E", False), "recomputed from D"),
    (lambda f, l, g: setattr(g, "beta", 0.25), "special_kz"),
    (lambda f, l, g: setattr(g, "bfast_scaled_k", (0.3, 0.0, 0.0)), "BFAST"),
    (lambda f, l, g: setattr(f, "_chi3_components", {"Ez": 1.0}),
     "instantaneous chi2/chi3"),
])
def test_every_pml_arm_refusal_is_reachable_and_names_itself(mutate, needle):
    """A clause nothing can trip is decoration, and a fail-closed arm made of
    decoration is an arm that admits everything.

    ``_chi3_components`` rather than ``has_nonlinearity``: that property reads
    ``_chi2_components`` ALONE (fields.py:966-967), so a chi3-only run answers
    False through it -- which is exactly the configuration this parametrisation
    plants.
    """
    fields, layer, grid = build(poles=2, pml_on=True)
    mutate(fields, layer, grid)
    covered, reason = dispersive.covers_real_pml_dispersive_constitutive(
        fields, layer, grid)
    assert covered is False and needle in reason, reason


def test_an_off_diagonal_row_is_refused_by_name_and_that_costs_exactly_one_corpus_slot():
    """``absorbed_power_density.py`` is off-diagonal AND dispersive AND folded.

    The refusal is the honest one and the header names its price. What the row
    needs is ``offdiag_emitter.py``'s row product -- which reads the OTHER
    components' ``D - sum P`` volumes at NEIGHBOURING cells (stepping.py:1001-1008,
    ``_offdiagonal_terms`` :1196) -- crossed with this file's pole chain, and the
    shipped CUDA off-diagonal family additionally refuses a fold.
    """
    fields, layer, grid = build(poles=1, pml_on=True, mirrors=(1,))
    fields._chi1inv_offdiagonal = {"Ex": {"Ey": fields.Dx}}
    covered, reason = dispersive.covers_real_pml_dispersive_constitutive(
        fields, layer, grid)
    assert covered is False and "off-diagonal chi1inv" in reason
    # ...and the shipped off-diagonal arm refuses it too, on the fold. Both
    # refusals together are what leaves the slot unserved rather than either one.
    covered, reason = coverage.covers_real_pml_offdiag_constitutive(
        fields, layer, grid)
    assert covered is False


def test_a_susceptibility_kind_outside_the_lorentz_drude_pair_is_refused():
    """The pole this arm SUBTRACTS is advanced by a difference equation this
    package transcribed; another kind is a number nothing here produced."""
    fields, layer, grid = build(poles=1, pml_on=True)
    object.__setattr__(fields.polarizations[0].susceptibility, "kind", "noisy")
    covered, reason = dispersive.covers_real_pml_dispersive_constitutive(
        fields, layer, grid)
    assert covered is False and "outside" in reason and "noisy" in reason


def test_two_susceptibilities_sharing_one_P_buffer_are_refused():
    """The pole pointers carry ``__restrict__``; an alias there is undefined behaviour.

    Unreachable from a real ``PolarizationState`` -- each allocates its own
    (dispersion.py:645-647) -- and checked anyway, because the promise in the
    device signature is the caller's to keep and the launcher is not the only
    thing that can bind these.
    """
    fields, layer, grid = build(poles=2, pml_on=True)
    fields.polarizations[1].P["Ex"] = fields.polarizations[0].P["Ex"]
    covered, reason = dispersive.covers_real_pml_dispersive_constitutive(
        fields, layer, grid)
    assert covered is False and "aliases" in reason


def test_the_no_pml_arm_refuses_pml_storage_behind_an_inert_layer():
    """``drive_field`` would hand ``f_w`` to update_P while this arm wrote E.

    CONSERVATIVE and labelled so in the predicate: the two disagree and nothing
    in the record would say which was meant.
    """
    fields, layer, grid = build(poles=2, pml_on=False)
    fields.enable_pml_storage()
    covered, reason = dispersive.covers_no_pml_dispersive_constitutive(
        fields, None, grid)
    assert covered is False and "PML storage enabled while the layer is inert" in reason


def test_the_two_halves_of_stepping_line_954_are_complementary_and_never_both():
    """``stores_E`` False is the NULL arm's slot; True is this one's.

    stepping.py:983 is one ``if`` and the two families are its two sides. A
    configuration admitted by both would be a slot the census reports as a
    disjointness FINDING; a configuration admitted by neither is a real gap.
    """
    fields, layer, grid = build(poles=2, pml_on=False)
    assert dispersive.covers_no_pml_dispersive_constitutive(
        fields, None, grid)[0] is True
    assert no_pml_constitutive.covers_no_pml_null_constitutive(
        fields, None, grid, "E")[0] is False

    bare, _layer, bare_grid = build(poles=0, pml_on=False)
    bare._stored_E = False
    assert no_pml_constitutive.covers_no_pml_null_constitutive(
        bare, None, bare_grid, "E")[0] is True
    covered, reason = dispersive.covers_no_pml_dispersive_constitutive(
        bare, None, bare_grid)
    assert covered is False and "stores_E is False" in reason


E_SIDE_ARMS = {
    "cuda_constitutive":
        lambda f, l, g: coverage.covers_real_pml_constitutive(f, l, g, "E"),
    "cuda_offdiag":
        lambda f, l, g: coverage.covers_real_pml_offdiag_constitutive(f, l, g),
    "cuda_no_pml":
        lambda f, l, g: no_pml_constitutive.covers_no_pml_null_constitutive(
            f, l, g, "E"),
    "cuda_dispersive":
        lambda f, l, g: dispersive.covers_real_pml_dispersive_constitutive(f, l, g),
    "cuda_no_pml_dispersive":
        lambda f, l, g: dispersive.covers_no_pml_dispersive_constitutive(f, l, g),
}


@pytest.mark.parametrize("poles,pml_on,mirrors", [
    (0, True, ()), (1, True, ()), (2, True, (1,)), (6, True, ()),
    (0, False, ()), (2, False, ()), (5, False, (1,)),
])
def test_at_most_one_e_side_arm_admits_any_configuration(poles, pml_on, mirrors):
    """DISJOINTNESS, measured on objects rather than argued from clause order.

    The arms are meant to PARTITION on the absorber, the storage and the
    features. Two admitting one slot is a widening past the evidence: the union
    census would attribute it to whichever family it asked first and the
    over-admitting one would never be looked at again.
    """
    fields, layer, grid = build(poles=poles, pml_on=pml_on, mirrors=mirrors)
    admitting = [name for name, call in E_SIDE_ARMS.items()
                 if call(fields, layer, grid)[0]]
    assert len(admitting) <= 1, (
        f"poles={poles} pml={pml_on} mirrors={mirrors}: admitted by {admitting}")


def test_arm_three_is_the_shipped_ade_predicate_with_one_clause_inverted():
    """No absorber: admitted here, refused there. With one: the reverse.

    That exhaustive swap over the SAME object is what "a predicate widening, not
    a new kernel" means, and it is the claim the gate's arm3 leg then measures on
    a device against ``stepping.update_P``.
    """
    fields, layer, grid = build(poles=2, pml_on=False)
    assert dispersive.covers_no_pml_ade_update_p(fields, None, grid) == (True, "covered")
    assert coverage.covers_real_pml_ade_update_p(fields, None, grid)[0] is False

    fields, layer, grid = build(poles=2, pml_on=True)
    assert coverage.covers_real_pml_ade_update_p(fields, layer, grid) == (True, "covered")
    covered, reason = dispersive.covers_no_pml_ade_update_p(fields, layer, grid)
    assert covered is False and "an active PML layer is installed" in reason


def test_arm_three_refuses_a_fields_whose_storage_mode_disagrees_with_its_layer():
    """THE SINGLE MOST LIKELY SILENT WRONG ANSWER IN DISPERSION, from the other side.

    ``drive_field`` returns ``f_w`` in PML storage mode and the stored E without
    it; the two agree EXACTLY outside the absorber, so binding the wrong one
    passes every no-PML case. Arm 3 binds the stored E, so the state that breaks
    it is PML storage behind an inert layer -- and the check is an IDENTITY check
    on the array ``drive_field`` actually hands back, not an existence check on
    some array.
    """
    fields, layer, grid = build(poles=1, pml_on=False)
    fields.enable_pml_storage()
    covered, reason = dispersive.covers_no_pml_ade_update_p(fields, None, grid)
    assert covered is False and "PML storage mode" in reason
    assert fields.drive_field("Ex") is fields.f_w_Ex, (
        "the fixture did not actually reach the state the clause is about")


def test_arm_three_refuses_a_run_that_would_launch_no_kernel_at_all():
    """A family claiming a slot it never touches is coverage the census would count.

    A susceptibility whose sigma is identically zero drives nothing -- MEEP's
    ``needs_P`` is false for the same reason (dispersion.py:645-647) -- so
    ``update_P`` iterates and does nothing. That is the NULL-arm question, which
    has no ADE counterpart today, and answering it "covered" here would be a
    vacuous pass.
    """
    fields, layer, grid = build(poles=1, pml_on=False,
                                drives=[()])
    assert fields.polarizations[0].driven() == ()
    covered, reason = dispersive.covers_no_pml_ade_update_p(fields, None, grid)
    assert covered is False and "drives nothing" in reason


def test_the_pole_plan_is_the_array_paths_own_contributor_order():
    """``resolve_pole_plan`` walks what ``displacement_minus_polarization`` walks.

    ORDER, not membership: the chain is left-associated float32 subtraction, so a
    plan that returned the same arrays in a different order is a different number
    -- measured below at 1253 differing words on a 8x10x12 grid at two poles.
    """
    fields, _layer, _grid = build(poles=3, pml_on=True,
                                  drives=[("Ex", "Ez"), ("Ey",), ("Ex", "Ey", "Ez")])
    plan = dispersive.resolve_pole_plan(fields)
    assert dispersive.pole_counts_of(plan) == (2, 2, 2)
    for component in ("Ex", "Ey", "Ez"):
        contributors = [state for state in fields.polarizations
                        if state.drives(component)]
        assert [id(array) for array in plan[component]] == \
            [id(state.P[component]) for state in contributors]


def test_a_cached_pole_plan_goes_stale_the_moment_update_P_runs():
    """The buffers ROTATE (dispersion.py:689-691), so a plan is good for one launch.

    This is why :func:`resolve_pole_plan` is called inside the launcher and never
    hoisted. Stale here is not a crash: it reads the previous step's polarization
    forever, which is smooth, finite and wrong.
    """
    fields, layer, grid = build(poles=2, pml_on=True)
    before = dispersive.resolve_pole_plan(fields)
    stepping.update_P(fields, layer)
    after = dispersive.resolve_pole_plan(fields)
    assert all(a is not b for a, b in zip(before["Ex"], after["Ex"])), (
        "the rotation did not move a single buffer; this test cannot see the "
        "defect it exists for")


# ---------------------------------------------------------------------------
# 3. THE TRANSCRIPTION, against stepping itself
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("poles", [0, 1, 2, 5, 6])
@pytest.mark.parametrize("arm,pml_on", [("pml", True), ("no_pml", False)])
def test_the_emitted_body_transcribes_to_stepping_update_E_bit_for_bit(arm, pml_on, poles):
    """ONE frozen state, run twice: the array path, then the transcription.

    The oracle is ``stepping`` ITSELF on real objects -- there is no second
    transcription on the oracle leg to drift. THE NON-VACUITY FLOOR IS PART OF
    THE TEST: a step that moved no output word is a fixed point, and against a
    fixed point a reference with deliberately swapped coefficients still reports
    IDENTICAL. The Triton cylindrical_complex module records exactly that
    failure; half its first cut could not fail.
    """
    fields, layer, grid = build(poles=poles, pml_on=pml_on, seed=10 + poles)
    frozen = snapshot(fields, arm)
    inputs = {name: getattr(fields, name).copy()
              for name in outputs(arm) + ("Dx", "Dy", "Dz")}

    stepping.update_E(fields, layer)
    reference = snapshot(fields, arm)
    moved = moved_words(frozen, reference)
    assert moved > 0, (
        f"{arm} poles={poles}: the array path changed no output word; this case "
        f"certifies nothing")

    restore(fields, inputs)
    transcribed_kernel(arm, fields, tables_for(layer) if pml_on else None,
                       dispersive.resolve_pole_plan(fields))
    for name in outputs(arm):
        assert np.array_equal(
            np.ascontiguousarray(reference[name]).ravel().view(np.uint32),
            np.ascontiguousarray(getattr(fields, name)).ravel().view(np.uint32)), (
            f"{arm} poles={poles}: {name} diverged from stepping.update_E")


@pytest.mark.parametrize("defect", [
    "sum_then_subtract", "reverse_pole_order", "drop_one_pole",
    "drop_inverse_epsilon_dispersive",
])
def test_the_comparison_can_fail_and_the_pole_defects_are_invisible_at_one_pole(defect):
    """A gate whose battery cannot fail scores identically whether it works or not.

    AND the half of this that is a real measurement rather than a smoke test:
    ``sum_then_subtract`` and ``reverse_pole_order`` are EXACTLY bit-identical at
    ONE pole and lethal at two. A gate leg that scored either on a single-pole
    row would report a harness defect as a kernel pass -- which is why
    :data:`~.dispersive_kernels.POLE_COUNTS_SWEPT` carries multi-pole arities and
    why the gate's mutation plan is pinned to them.
    """
    expected_at_one_pole = defect in ("sum_then_subtract", "reverse_pole_order")
    for poles, must_diverge in ((1, not expected_at_one_pole), (2, True)):
        fields, layer, grid = build(poles=poles, pml_on=True, seed=99)
        inputs = {name: getattr(fields, name).copy()
                  for name in outputs("pml") + ("Dx", "Dy", "Dz")}
        stepping.update_E(fields, layer)
        reference = snapshot(fields, "pml")
        restore(fields, inputs)
        transcribed_kernel("pml", fields, tables_for(layer),
                           dispersive.resolve_pole_plan(fields), defect=defect)
        differing = sum(int(np.count_nonzero(
            np.ascontiguousarray(reference[name]).ravel().view(np.uint32)
            != np.ascontiguousarray(getattr(fields, name)).ravel().view(np.uint32)))
            for name in outputs("pml"))
        if must_diverge:
            assert differing > 0, (
                f"{defect} at {poles} pole(s) was bit-identical; this comparison "
                f"cannot see the defect it is scored on")
        else:
            assert differing == 0, (
                f"{defect} DIVERGED at one pole; it is documented as provably "
                f"inert there ((D - P0) has one association), so either the "
                f"fixture or the transcription is wrong, not the kernel")


def test_the_null_operand_order_control_is_inert_at_every_arity():
    """``inv_eps * s`` versus ``s * inv_eps``: IEEE multiply commutes.

    MUST BE UNCAUGHT, paired with ``drop_inverse_epsilon_dispersive`` above,
    which edits the same expression and must be caught. Without the pair, "the
    operand order is inert" would be a claim about a leg nobody showed could
    fail -- and the array path's order (stepping.py:1011, D on the left) is kept
    for transcription discipline, not because it changes a bit.
    """
    for poles in (1, 5):
        fields, layer, grid = build(poles=poles, pml_on=True, seed=7)
        inputs = {name: getattr(fields, name).copy()
                  for name in outputs("pml") + ("Dx", "Dy", "Dz")}
        stepping.update_E(fields, layer)
        reference = snapshot(fields, "pml")
        restore(fields, inputs)
        transcribed_kernel("pml", fields, tables_for(layer),
                           dispersive.resolve_pole_plan(fields),
                           defect="inv_eps_left")
        for name in outputs("pml"):
            assert np.array_equal(
                np.ascontiguousarray(reference[name]).ravel().view(np.uint32),
                np.ascontiguousarray(getattr(fields, name)).ravel().view(np.uint32)), (
                f"poles={poles}: inv_eps_left changed a bit; it must not")


def test_the_recorded_gate_verdict_pins_the_emitted_bodies_that_ship_today():
    """An emitter edit must fail HERE, on a laptop, not on a device run hours later.

    ``GATE_VERDICT["emitted_corpus_digest"]`` is deliberately NOT a file hash: a
    file hash stops matching when a docstring gains a comma, which makes it
    useless as a pin on the thing that decides the verdict. What NVRTC compiled is
    the emitted device STRINGS, and this digest is taken over every arm at every
    swept arity.

    THE DISTINCTION IS LOAD-BEARING AND WAS EXERCISED ON 2026-08-20: the fold cap
    in this module moved from a shared constant to :data:`ADE_FOLD_PLANES_SWEPT`
    AFTER the gate ran, changing the file's sha256 and not one emitted byte. A
    file-hash pin would have read that as a dead verdict; this reads it as what it
    is, a predicate clause that compiles nothing.
    """
    assert dispersive.GATE_VERDICT["emitted_corpus_digest"] == \
        dispersive.corpus_digest(), (
        "the emitted device bodies changed since gate_cuda_dispersive.py released. "
        "Re-run it on a CUDA host under BOTH float32 subnormal policies and "
        "re-record GATE_VERDICT; the old numbers describe different bytes.")


def test_the_gate_verdict_names_an_artifact_and_both_policy_legs():
    """A verdict with no artifact behind it is a memory.

    ``results/`` is untracked, so the artifact is a declared skip on most hosts --
    the digest above still pins the bodies. Where it IS present, both legs are
    required to exist, to carry the policy the record files them under, and to
    report the case counts and the release this constant transcribes. A re-record
    that ran one leg and copied the other's numbers fails here.
    """
    record = dispersive.GATE_VERDICT
    assert set(record["legs"]) == {"keep", "flush"}
    artifact = HERE.parents[1] / record["artifact"]
    if not artifact.is_dir():
        pytest.skip(f"the 2026-08-20 gate artifact is not on this host ({artifact})")
    for leg, policy in record["legs"].items():
        payload = artifact / leg / "gate.json"
        assert payload.exists(), (
            f"{leg}/ is missing: the record claims a cut under BOTH float32 "
            f"policies and only part of that run is on this host")
        document = json.loads(payload.read_text(encoding="utf-8"))
        assert document["summary"]["released"] is record["released"], leg
        assert document["summary"]["scored_cases"] == \
            record["scored_cases_per_leg"], leg
        assert document["summary"]["single_launch_identical"] == \
            record["single_launch_identical"], leg
        assert document["summary"]["multi_step_identical"] == \
            record["multi_step_identical"], leg
        assert document["emitted_corpus_digest"] == \
            record["emitted_corpus_digest"], leg
        assert document["subnormal_policy_stamp"]["policy"] == policy, leg
        assert document["verdict_flips_against_planted_defect"]["all_flipped"] is \
            record["verdict_flips_against_planted_defect"], leg


def test_this_familys_fold_cap_is_its_own_gates_number_and_not_a_siblings():
    """A CAP IS A FACT ABOUT A GATE, and a gate answers for ONE family.

    Measured 2026-08-20: the sibling constitutive record's ``folded_planes_swept``
    went 2 -> 3 when THAT family re-ran with a third plane. This module's clause
    read it directly, so the raise would have silently widened this family to
    three planes on a sweep it never ran -- and because the corpus drives no
    three-plane dispersive row, the widening would have bought ZERO slots and been
    invisible to the census. An over-claim with no measurement behind it and
    nothing to catch it.

    So the cap is local, and this test is what stops it being re-pointed at a
    number some other gate owns.
    """
    assert dispersive.ADE_FOLD_PLANES_SWEPT == 2, (
        "raising this means adding a three-plane spec to gate_cuda_dispersive.py, "
        "running it on a device under both float32 policies, and recording what "
        "came back -- not editing the number")
    # READ OFF THE SYNTAX TREE, not off the text: the module's PROSE names the
    # sibling record deliberately (that is the whole explanation for why the cap
    # is local), and a substring check would forbid explaining the decision. What
    # must not come back is a REFERENCE -- an import or a load.
    tree = ast.parse((HERE / "dispersive_kernels.py").read_text(encoding="utf-8"))
    referenced = [node for node in ast.walk(tree)
                  if (isinstance(node, ast.Name)
                      and node.id == "CONSTITUTIVE_FOLD_ADMISSION")
                  or (isinstance(node, ast.alias)
                      and node.name == "CONSTITUTIVE_FOLD_ADMISSION")]
    assert not referenced, (
        "the fold clause is reading a sibling family's record again; a shared cap "
        "is a way to widen a family nobody measured")
    fields, layer, grid = build(poles=2, pml_on=True, mirrors=(0, 1))
    assert dispersive.covers_real_pml_dispersive_constitutive(
        fields, layer, grid)[0] is True, "two planes is what the gate scored"
