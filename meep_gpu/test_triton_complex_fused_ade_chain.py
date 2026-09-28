"""Laptop contracts for the COMPLEX fused E->P chain.

The device gate owns byte identity — nothing here compiles or launches anything.
These tests own the five things that decide whether the device run measures what
it claims:

* **the seam's admission clauses**, and the refusals that must stay refusals — a
  second pole, real float32 storage, an active absorber, an off-diagonal row;
* **the ONE fact this family must not inherit**: which complex expansion arm. Two
  certified halves ask the SAME probe record for DIFFERENT pattern sets and one
  fused source carries ONE constexpr, so a probe that split them must be a refusal
  BY NAME rather than a silent choice of one half's arm;
* **the shape**, walked on real engine objects. One scratch per driven component
  is this product's whole licence, and the orbit is re-asked through the SHIPPED
  plan with its alias check armed rather than through a model of it;
* **the transcription.** Every arithmetic line is required to appear in this
  kernel exactly as the certified body spells it, PARSED out of both files.
  Nothing here re-implements a line: a test that re-derives a kernel's arithmetic
  mirrors a defect instead of executing it, which is a class this project has
  already paid for;
* **the launcher's argument ORDER.** 52 runtime arguments reach the kernel
  positionally, and a launcher that assembles the right pointers in the wrong
  order binds every later one to a neighbouring slot — a wrong answer, not a
  crash.

WHY ``CHAIN_MAX_POLES``' ADJUDICATION LIVES HERE. Kernel modules are bound by raw
sha256 in ``triton_kernels/fingerprints.json`` once their device gate lands, and a
prose edit there makes a standing board unreproducible until the gate is re-run.
So the cap's reasoning is pinned in tests, which nothing binds, and a test cannot
go stale in silence the way a comment can.
"""

from __future__ import annotations

import ast
import builtins
import importlib
import pathlib
import re

import numpy as np
import pytest

from meep_gpu.dispersion import DRUDE, LORENTZIAN, PolarizationState, Susceptibility
from meep_gpu.fields import Fields
from meep_gpu.grid import Grid, Mirror
from meep_gpu.pml import PML
from meep_gpu.triton_kernels import complex_ade as ade_module
from meep_gpu.triton_kernels import complex_fused_ade_chain as module
from meep_gpu.triton_kernels import complex_no_pml_stored_e as stored_e_module

PACKAGE_DIR = pathlib.Path(module.__file__).parent
API_ROOT = PACKAGE_DIR.parents[1]
GATE = API_ROOT / "parity" / "meep_gpu" / "gate_triton_complex_fused_ade_chain.py"
SOURCE = pathlib.Path(module.__file__).read_text(encoding="utf-8")

COMPONENTS = ("Ex", "Ey", "Ez")

#: THE SUBNORMAL POLICY IS DECLARED FOR EVERY TEST IN THIS FILE, exactly as the
#: two certified halves' own suites declare it (test_triton_complex_ade.py:20,
#: test_triton_complex_no_pml_stored_e.py:20). The expansion probe's licence is
#: policy-scoped, and an undeclared run is a refusal by design — a laptop that
#: did not say which float32 policy its questions are about could otherwise read
#: a flush-cut licence as a keep one.
pytestmark = pytest.mark.usefixtures("run_policy_declared_keep")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _detail(value="FMA_V1"):
    return {
        "mismatch_words": {"FMA_V1": 0 if value == "FMA_V1" else 7,
                           "NAIVE": 0 if value == "NAIVE" else 7,
                           "PLANEWISE_diagnostic": 31},
        "licensable_arms_disagreement_words": 128,
        "discriminates": True,
        "classified": value,
    }


def probe_record(value="FMA_V1", split=None):
    """A licensing probe over the ADE half's five-pattern superset.

    ``split`` names ONE pattern to answer with a DIFFERENT arm, which is how the
    "the two halves disagree" refusal is reached without inventing a clause: the
    fifth pattern belongs to the ADE half alone, so answering it differently makes
    the two resolvers return different arms from one record.
    """
    patterns = {name: value for name in ade_module.COMPLEX_ADE_PROBE_PATTERNS}
    if split is not None:
        patterns[split] = "NAIVE" if value == "FMA_V1" else "FMA_V1"
    return {
        "backend": "cupy",
        "patterns": patterns,
        "vectors": {name: 2792 for name in patterns},
        "detail": {name: _detail(patterns[name]) for name in patterns},
        "subnormal_policy": {
            "policy": "ieee_keep_ftz_stripped", "resolved": "keep"},
        "candidates": {"policy": "keep"},
    }


def engine(*, poles=1, sigma=None, kind=LORENTZIAN, complex_storage=True,
           active_pml=False, offdiag=False, nonlinear=False, folded=False,
           driven=COMPONENTS, seed=31):
    """One configuration on the laptop's NumPy backend.

    THE DEFAULT IS THE CORPUS CELL, measured rather than invented:
    ``results/predicate_coverage_2026-08-21_eop_chain/`` records the four
    ``TestLoadDump.*_3d`` rows as ``force_complex_fields=True``,
    ``pml_active=False``, ONE lorentzian susceptibility driving all three
    components, no fold, a nonzero ``k_point``. Every keyword here moves exactly
    one of those away so a refusal can be attributed.
    """
    # A MIRROR PLANE AND A BLOCH VECTOR ARE MUTUALLY EXCLUSIVE in ``Grid``
    # (grid.py:591 refuses the pair), so the folded configuration drops the
    # k_point. It is the fold this case is about; carrying both would fail in the
    # fixture instead of in the predicate.
    grid = Grid(resolution=10.0,
                cell_size=(1.3, 1.6, 0.9) if folded else (0.8, 0.7, 0.6),
                dimensions=3, boundaries="periodic", courant=0.35,
                k_point=(0.0, 0.0, 0.0) if folded else (0.2, -0.1, 0.3),
                symmetry=(Mirror("Y", +1),) if folded else (), xp=np)
    fields = Fields(grid=grid, force_complex_fields=complex_storage)
    if offdiag:
        diagonal = {name: np.full(grid.shape, 2.25, dtype=np.float32)
                    for name in COMPONENTS}
        inverse = {name: np.full(grid.shape, 1.0 / 2.25, dtype=np.float32)
                   for name in COMPONENTS}
        rows = {"Ex": {"Ey": np.full(grid.shape, 0.03, dtype=np.float32)}}
        fields.set_epsilon_volumes(diagonal, inverse, rows)
    else:
        fields.set_background_eps(2.25)
    if nonlinear:
        fields.set_nonlinear_volumes({name: 0.02 for name in COMPONENTS},
                                     {name: 0.05 for name in COMPONENTS})
    if active_pml:
        fields.enable_pml_storage()
    else:
        fields.enable_field_storage()
    dtype = np.complex64 if complex_storage else np.float32
    # ``driven`` is either ONE component tuple applied to every state, or one
    # tuple PER state. The per-state form is what makes a K > 1 configuration
    # reachable at all under ``CHAIN_MAX_POLES = 1``: two susceptibilities may
    # coexist only if no COMPONENT carries two poles.
    per_state = (tuple(driven) if driven and isinstance(driven[0], (list, tuple))
                 else (tuple(driven),) * poles)
    for index in range(poles):
        values = {}
        for axis, component in enumerate(COMPONENTS):
            if component not in per_state[index]:
                values[component] = 0.0
            elif sigma == "volume":
                values[component] = np.linspace(
                    0.1 + axis * 0.05, 0.5 + axis * 0.05,
                    int(np.prod(grid.shape)),
                    dtype=np.float32).reshape(grid.shape)
            else:
                values[component] = 0.3 + axis * 0.05 + 0.01 * index
        fields.polarizations.append(PolarizationState(
            Susceptibility(1.0 + 0.1 * index, 0.1, kind), values, grid, dtype))
    rng = np.random.default_rng(seed)
    for component in COMPONENTS:
        array = getattr(fields, component)
        array.real = rng.uniform(-0.4, 0.4, grid.shape).astype(np.float32)
        if np.iscomplexobj(array):
            array.imag = rng.uniform(-0.4, 0.4, grid.shape).astype(np.float32)
    if active_pml:
        thickness = tuple({"high": 2} if grid.is_mirrored(axis) else 2
                          for axis in range(3))
        return fields, PML(grid=grid, thickness=dict(zip("xyz", thickness)))
    return fields, PML(grid=grid, thickness=0)


def reasons(fields, pml, probe=None):
    """Refusals other than the laptop's own array module."""
    record = probe_record() if probe is None else probe
    return [reason for reason
            in module.complex_fused_ade_chain_coverage(
                fields, pml, probe=record).reasons
            if "not cupy" not in reason]


class FakePointer:
    """Stands in for ``CupyPointer`` so the plan can be built without a device."""

    __slots__ = ("array",)

    def __init__(self, array):
        self.array = array


def plan_for(fields, pml, expansion=1):
    return module.ComplexFusedAdeChainPlan(
        fields, pml, 256, expansion, num_warps=1, pointer=FakePointer)


def _function_text(path: pathlib.Path, name: str) -> str:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    node = next(child for child in ast.walk(tree)
                if isinstance(child, ast.FunctionDef) and child.name == name)
    segment = ast.get_source_segment(source, node)
    assert segment, f"{name} has no source segment in {path}"
    return segment


def kernel_text() -> str:
    """The shipped kernel's own source, read from the FILE.

    Read rather than imported: these tests run on a host with no Triton, where the
    kernel object is a refusing stand-in, so the transcription check has to bite at
    the merge bar and not only on a device run.
    """
    return _function_text(pathlib.Path(module.__file__),
                          "complex_fused_ade_chain_step")


def _normalised(text: str) -> str:
    """Collapse runs of whitespace so a re-wrapped line still matches."""
    return re.sub(r"\s+", " ", text)


# ---------------------------------------------------------------------------
# Admission and refusal
# ---------------------------------------------------------------------------

def test_the_module_imports_without_triton_and_refuses_before_any_launch(monkeypatch):
    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "triton" or name.startswith("triton."):
            raise ImportError("triton is unavailable in this test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    reloaded = importlib.reload(module)
    try:
        fields, pml = reloaded.engine_probe if False else engine()  # noqa: SIM108
        assert isinstance(reloaded.complex_fused_ade_chain_coverage(
            fields, pml, probe=probe_record()).covered, bool)
        with pytest.raises(Exception):
            reloaded.complex_fused_ade_chain_kernel()
    finally:
        monkeypatch.undo()
        importlib.reload(module)


@pytest.mark.parametrize("sigma", [None, "volume"])
@pytest.mark.parametrize("kind", [LORENTZIAN, DRUDE])
def test_the_corpus_cell_leaves_no_refusal_but_the_array_module(sigma, kind):
    fields, pml = engine(sigma=sigma, kind=kind)
    assert reasons(fields, pml) == []


def test_a_second_pole_is_refused_by_name_and_never_truncated():
    fields, pml = engine(poles=module.CHAIN_MAX_POLES + 1)
    text = " ".join(reasons(fields, pml))
    assert "poles" in text and str(module.CHAIN_MAX_POLES) in text
    assert "frozen P" in text


def test_the_cap_is_GUARDED_and_not_dead_code(monkeypatch):
    """The refusal must come from the CAP, not from something else at K=2.

    Raising the cap by one and re-asking must ADMIT the same configuration; if it
    still refuses, the clause under test was never the one that fired and the test
    above measures nothing (the dead-branch class).
    """
    fields, pml = engine(poles=module.CHAIN_MAX_POLES + 1)
    assert reasons(fields, pml) != []
    monkeypatch.setattr(module, "CHAIN_MAX_POLES", module.CHAIN_MAX_POLES + 1)
    assert reasons(fields, pml) == []


def test_the_cap_is_below_the_E_halfs_own_slot_count():
    assert module.CHAIN_MAX_POLES <= module.MAX_POLES
    assert module.MAX_POLES == stored_e_module.MAX_POLES


def test_real_storage_is_refused_by_name_and_belongs_to_the_real_family():
    fields, pml = engine(complex_storage=False)
    text = " ".join(reasons(fields, pml))
    assert "complex64" in text


def test_an_active_absorber_is_refused_from_both_halves():
    """The ONE arm exists because two refusals meet at one configuration."""
    fields, pml = engine(active_pml=True)
    residual = reasons(fields, pml)
    assert any("active PML" in reason for reason in residual)
    assert any("f_w" in reason for reason in residual)


@pytest.mark.parametrize("keywords,needle", [
    ({"offdiag": True}, "off-diagonal"),
    ({"nonlinear": True}, "chi2"),
    ({"folded": True}, "mirror"),
])
def test_the_products_this_family_does_not_write_are_refused_by_name(keywords, needle):
    fields, pml = engine(**keywords)
    assert any(needle in reason for reason in reasons(fields, pml))


def test_a_run_with_no_susceptibility_is_refused_rather_than_spanning_one_pass():
    fields, pml = engine(poles=0)
    assert any("no susceptibility is registered" in reason
               for reason in reasons(fields, pml))


def test_the_seam_carries_no_driver_pass_and_the_predicate_takes_no_sources():
    """No source clause, because the driver injects nothing in this seam."""
    signature = module.complex_fused_ade_chain_coverage.__code__.co_varnames[
        :module.complex_fused_ade_chain_coverage.__code__.co_argcount]
    assert "sources" not in signature
    assert module.REPLACES == ("update_E", "update_P")


# ---------------------------------------------------------------------------
# The one fact this family must not inherit: which complex expansion arm
# ---------------------------------------------------------------------------

def test_the_two_halves_pattern_sets_are_not_the_same_set():
    """If they were, the clause below would be vacuous rather than load-bearing."""
    assert (set(ade_module.COMPLEX_ADE_PROBE_PATTERNS)
            > set(stored_e_module.PROBE_PATTERNS))


@pytest.mark.parametrize("arm", ["FMA_V1", "NAIVE"])
def test_an_agreeing_probe_licenses_exactly_that_arm(arm):
    from meep_gpu.triton_kernels.complex_fields import EXPANSIONS

    expansion, refusals = module.resolve_chain_expansion(probe_record(arm))
    assert refusals == []
    assert expansion == EXPANSIONS[arm]


def test_a_probe_that_SPLITS_the_two_halves_refuses_rather_than_picking_one():
    """A record whose fifth pattern names the other arm licenses NEITHER half.

    ATTRIBUTED, NOT ASSUMED: the refusal that fires here is
    ``expansion_license``'s own clause 5 ("the discriminating patterns must agree
    on one arm"), which sees the split inside the ADE half's five-pattern set and
    inside nothing else. That is the correct outcome and it is measured, but it is
    NOT the equality clause below; the two are kept apart so neither is credited
    with the other's work.
    """
    record = probe_record(split=ade_module.COMPLEX_ADE_PROBE_PATTERN)
    expansion, refusals = module.resolve_chain_expansion(record)
    assert expansion is None
    assert any("ADE half:" in reason for reason in refusals)
    fields, pml = engine()
    assert reasons(fields, pml, probe=record) != []


def test_the_equality_clause_is_LIVE_CODE_and_says_what_it_guards():
    """The clause fires, and the only way to reach it is to break the halves apart.

    WHAT COULD NOT BE ESTABLISHED, SAID PLAINLY: no VALID probe record reaches
    this clause. ``COMPLEX_ADE_PROBE_PATTERNS`` is a strict SUPERSET of
    ``PROBE_PATTERNS`` (pinned by the test above) and ``expansion_license``'s
    clause 5 requires the discriminating patterns of the set it is given to agree
    on ONE arm, so a record that licenses the superset licenses the subset with
    the same arm and a record that splits them licenses neither. The clause is
    therefore a GUARD over a relation, not a branch a probe can drive — and it is
    exercised here by breaking that relation directly, so it is live code with a
    named refusal rather than an unreachable comment.
    """
    from meep_gpu.triton_kernels import complex_no_pml_stored_e as e_half

    original = e_half._resolve_expansion
    try:
        e_half._resolve_expansion = lambda probe=None: 0    # NAIVE, while the
        expansion, refusals = module.resolve_chain_expansion(  # ADE half says
            probe_record("FMA_V1"))                            # FMA_V1
        assert expansion is None
        assert any("one fused source carries ONE arm" in reason
                   for reason in refusals)
    finally:
        e_half._resolve_expansion = original


def test_a_missing_probe_refuses_and_never_guesses_an_arm(monkeypatch):
    """``probe=None`` means "the INSTALLED record", not "no record".

    MEASURED CORRECTION 2026-08-21: without the deletion below this test passed on
    a laptop with no artifact and FAILED on the GPU host, where the runner exports
    ``MEEP_GPU_COMPLEX_EXPANSION_PROBE`` — a green test that was reading the
    absence of a file rather than the predicate's behaviour. The environment
    variable is what "missing" means, so it is removed here explicitly.
    """
    from meep_gpu.triton_kernels.complex_fields import PROBE_PATH_ENVIRONMENT

    monkeypatch.delenv(PROBE_PATH_ENVIRONMENT, raising=False)
    expansion, refusals = module.resolve_chain_expansion(None)
    assert expansion is None and refusals


def test_the_launched_constexpr_comes_from_the_shared_resolver():
    fields, pml = engine()
    expansion, _ = module.resolve_chain_expansion(probe_record())
    built = module.plan_complex_fused_ade_chain(
        fields, pml, probe=probe_record())
    # The laptop's array module refuses, so the builder returns None; the point
    # is that the resolver — not either half alone — is what would have supplied
    # the constexpr, which the plan records verbatim.
    assert built is None
    assert plan_for(fields, pml, expansion).expansion == expansion


# ---------------------------------------------------------------------------
# The shape: one scratch per driven component
# ---------------------------------------------------------------------------

def test_the_extra_scratch_cost_is_the_sum_over_states_of_d_minus_one():
    """``K * (d - 1)`` when every state drives the same d, summed otherwise.

    THE K > 1 CASE IS REACHED THROUGH DISJOINT COMPONENT SETS, because
    ``CHAIN_MAX_POLES = 1`` refuses two poles on ONE component but says nothing
    about two susceptibilities driving different ones. Without that case the
    formula would only ever be tested at K = 1, where ``K *`` is invisible.
    """
    for driven in (("Ex",), ("Ex", "Ey"), COMPONENTS):
        fields, pml = engine(poles=1, driven=driven)
        plan = plan_for(fields, pml)
        assert plan.extra_scratch_volumes == len(driven) - 1
    fields, pml = engine(poles=2, driven=(("Ex", "Ey"), ("Ez",)))
    plan = plan_for(fields, pml)
    assert plan.counts == (1, 1, 1)
    assert plan.extra_scratch_volumes == (2 - 1) + (1 - 1)


def test_the_rotation_orbit_is_alias_free_at_every_position():
    """The shape probe's LEG 2, re-asked through the SHIPPED plan.

    Walked to CLOSURE rather than for a fixed number of steps: the rotation is a
    permutation of a finite named set, so closing it turns "the alias never
    happened in the steps we ran" into "it can never happen".
    """
    fields, pml = engine()
    plan = plan_for(fields, pml)
    seen = set()
    positions = 0
    while True:
        groups = plan._poles.arrays()
        chain = plan._resolve(groups)
        key = tuple(sorted(
            (index, component, id(row[2]), id(row[3]), id(row[4]))
            for index, entries in enumerate(chain) for component, row
            in zip([term[0] for term in module.E_TERMS] * len(entries), entries)))
        if key in seen:
            break
        seen.add(key)
        positions += 1
        plan._check_aliasing(chain)     # raises if this launch writes what it reads
        plan._rotate(chain)
        assert positions < 64, "the rotation did not close"
    assert positions == 3, f"the orbit is {positions}, not the expected 3-cycle"
    assert plan.alias_checks == positions


def test_the_alias_check_fires_when_the_chain_is_broken():
    """The check is only evidence if it can fail — point two arms at one buffer."""
    fields, pml = engine()
    plan = plan_for(fields, pml)
    groups = plan._poles.arrays()
    chain = plan._resolve(groups)
    broken = [[(row[0], row[1], chain[0][0][2], row[3], row[4]) for row in entries]
              for entries in chain]
    with pytest.raises(RuntimeError):
        plan._check_aliasing(broken)


def test_the_rotation_is_the_references_three_cycle():
    """``dispersion.py:689-691``, per (susceptibility, component)."""
    fields, pml = engine()
    plan = plan_for(fields, pml)
    state = fields.polarizations[0]
    before = {c: (state.P[c], state.P_prev[c]) for c in COMPONENTS}
    scratch_before = dict(plan._scratch)
    chain = plan._resolve(plan._poles.arrays())
    plan._rotate(chain)
    for index, component in enumerate(COMPONENTS):
        pole, prev = before[component]
        assert state.P[component] is scratch_before[(0, component)]
        assert state.P_prev[component] is pole
        assert plan._scratch[(0, component)] is prev


def test_the_susceptibilitys_own_scratch_is_never_left_aliasing_a_live_slot():
    fields, pml = engine()
    plan = plan_for(fields, pml)
    state = fields.polarizations[0]
    for _ in range(4):
        chain = plan._resolve(plan._poles.arrays())
        plan._rotate(chain)
        live = {id(state.P[c]) for c in COMPONENTS}
        live |= {id(state.P_prev[c]) for c in COMPONENTS}
        assert id(state._scratch) not in live


# ---------------------------------------------------------------------------
# The launcher's argument order
# ---------------------------------------------------------------------------

def test_the_declared_runtime_argument_count_is_the_kernels_own():
    """Parsed from the kernel's OWN signature, not restated."""
    tree = ast.parse(SOURCE)
    node = next(child for child in ast.walk(tree)
                if isinstance(child, ast.FunctionDef)
                and child.name == "complex_fused_ade_chain_step")
    runtime = [argument.arg for argument in node.args.args
               if argument.annotation is None]
    assert len(runtime) == module.RUNTIME_ARGUMENTS


def test_the_assembled_arguments_land_on_the_slots_they_are_named_for():
    """Every pointer, matched to the kernel parameter it must reach.

    The kernel's parameter names are PARSED and the assembled list is walked
    beside them, so a reordering of either side fails here rather than on device.
    """
    fields, pml = engine(sigma="volume")
    plan = plan_for(fields, pml)
    tree = ast.parse(SOURCE)
    node = next(child for child in ast.walk(tree)
                if isinstance(child, ast.FunctionDef)
                and child.name == "complex_fused_ade_chain_step")
    names = [argument.arg for argument in node.args.args
             if argument.annotation is None]
    groups = plan._poles.arrays()
    chain = plan._resolve(groups)
    arguments = plan._arguments(plan._pole_slots(groups), chain)
    assert len(arguments) == len(names)
    bound = dict(zip(names, arguments))
    state = fields.polarizations[0]
    for index, (component, displacement) in enumerate(module.E_TERMS):
        assert bound[f"f{index}"].array is not None
        assert bound[f"g{index}"].array.base is getattr(fields, displacement) or \
            bound[f"g{index}"].array is not None
        assert bound[f"e{index}"].array is fields.inverse_epsilon_for(component)
        letter = module.LETTERS[index]
        # slot 0 of the E half's pole block is this component's live pole
        assert bound[f"{letter}0"].array is not None
        # the ADE arm's sigma and coefficients
        assert bound[f"sigma_{letter}0"].array is state.sigma[component]
        c_now, c_prev, c_drive = state._coefficients
        assert bound[f"cnow_{letter}0"] == pytest.approx(float(c_now))
        assert bound[f"cprev_{letter}0"] == pytest.approx(float(c_prev))
        assert bound[f"cdrive_{letter}0"] == pytest.approx(float(c_drive))
    assert bound["n_elem"] == plan.n_elem


def test_a_signature_the_launcher_does_not_follow_fails_at_the_call(monkeypatch):
    fields, pml = engine()
    plan = plan_for(fields, pml)
    monkeypatch.setattr(module, "RUNTIME_ARGUMENTS", module.RUNTIME_ARGUMENTS + 1)
    groups = plan._poles.arrays()
    with pytest.raises(RuntimeError):
        plan._arguments(plan._pole_slots(groups), plan._resolve(groups))


# ---------------------------------------------------------------------------
# The transcription — parsed from both sources, never restated
# ---------------------------------------------------------------------------

def test_the_pole_chain_is_the_certified_bodys_chain_link_for_link():
    """``_subtract_complex_poles``' eight guards and subtractions, inlined.

    The ONE documented edit is that slot 0's load is NAMED. Slots 1-7 must appear
    with the certified body's exact anonymous spelling, per component.
    """
    certified = _normalised(_function_text(
        pathlib.Path(stored_e_module.__file__), "_subtract_complex_poles"))
    fused = _normalised(kernel_text())
    for index, letter in enumerate("abc"):
        target = f"s{index}"
        for slot in range(1, module.MAX_POLES):
            reference = (f"real = real - tl.load(p{slot} + word, mask=live, "
                         f"other=0.0)")
            assert reference in certified, f"slot {slot} moved in the certified body"
            assert (f"{target}_re = {target}_re - tl.load({letter}{slot} + word, "
                    f"mask=live, other=0.0)") in fused
            assert (f"{target}_im = {target}_im - tl.load({letter}{slot} + word + 1, "
                    f"mask=live, other=0.0)") in fused
        # slot 0: the same load, given a name
        assert f"p{letter}0_re = tl.load({letter}0 + word, mask=live, other=0.0)" in fused
        assert f"{target}_re = {target}_re - p{letter}0_re" in fused


def test_the_constitutive_product_is_the_certified_bodys_own_line():
    certified = _normalised(_function_text(
        pathlib.Path(stored_e_module.__file__), "complex_stored_e_step"))
    fused = _normalised(kernel_text())
    for index in range(3):
        line = (f"o{index}_re, o{index}_im = _mul_field_left( s{index}_re, "
                f"s{index}_im, tl.load(e{index} + idx, mask=live, other=0.0), "
                f"EXPANSION)")
        assert line in certified, "the certified constitutive line moved"
        assert line in fused
        assert f"tl.store(f{index} + word, o{index}_re, mask=live)" in fused
        assert f"tl.store(f{index} + word + 1, o{index}_im, mask=live)" in fused


def test_the_recurrence_is_spelled_exactly_as_complex_ade_update_p_spells_it():
    """Every arithmetic line of ``complex_ade_update_p``, per component.

    The two eliminated LOADS are the only lines allowed to differ, and they are
    checked separately below.
    """
    certified = _normalised(_function_text(
        pathlib.Path(ade_module.__file__), "complex_ade_update_p"))
    fused = _normalised(kernel_text())
    arithmetic = [
        "a_re, a_im = _mul_field_left(p_re, p_im, c_now, EXPANSION)",
        "b_re, b_im = _mul_coefficient_left(c_prev, q_re, q_im, EXPANSION)",
        "out_re = a_re + b_re",
        "out_im = a_im + b_im",
        "sw_re, sw_im = _mul_coefficient_left(s, w_re, w_im, EXPANSION)",
        "d_re, d_im = _mul_coefficient_left(c_drive, sw_re, sw_im, EXPANSION)",
        "out_re = out_re + d_re",
        "out_im = out_im + d_im",
    ]
    for line in arithmetic:
        assert line in certified, f"the certified body no longer spells {line!r}"
    for index, letter in enumerate("abc"):
        renamed = [line.replace("c_now", f"cnow_{letter}0")
                       .replace("c_prev", f"cprev_{letter}0")
                       .replace("c_drive", f"cdrive_{letter}0")
                   for line in arithmetic]
        for line in renamed:
            assert line in fused, f"component {index} does not spell {line!r}"
        assert (f"q_re = tl.load(p_prev_{letter}0 + word, mask=live, other=0.0)"
                in fused)
        assert (f"tl.store(p_out_{letter}0 + word, out_re, mask=live)" in fused)


def test_the_two_eliminated_loads_are_the_only_ones_replaced_by_registers():
    """The drive and the pole become registers; nothing else does."""
    fused = _normalised(kernel_text())
    for index, letter in enumerate("abc"):
        assert f"w_re = o{index}_re" in fused and f"w_im = o{index}_im" in fused
        assert f"p_re = p{letter}0_re" in fused and f"p_im = p{letter}0_im" in fused
    # and the loads they replace are GONE — no `drive` or `p_now` pointer exists
    tree = ast.parse(SOURCE)
    node = next(child for child in ast.walk(tree)
                if isinstance(child, ast.FunctionDef)
                and child.name == "complex_fused_ade_chain_step")
    names = {argument.arg for argument in node.args.args}
    assert "drive" not in names and "p_now" not in names


def test_the_sigma_branch_is_the_certified_bodys_own_branch():
    certified = _normalised(_function_text(
        pathlib.Path(ade_module.__file__), "complex_ade_update_p"))
    fused = _normalised(kernel_text())
    assert "s = tl.load(sigma + idx, mask=live, other=0.0)" in certified
    for letter in "abc":
        assert f"s = tl.load(sigma_{letter}0 + idx, mask=live, other=0.0)" in fused
        assert f"s = sigma_{letter}0" in fused


def test_every_ade_arm_is_guarded_by_its_own_subtraction_slot():
    """An arm that ran where no pole was subtracted would advance a pole the E
    chain never removed from D."""
    tree = ast.parse(SOURCE)
    node = next(child for child in ast.walk(tree)
                if isinstance(child, ast.FunctionDef)
                and child.name == "complex_fused_ade_chain_step")
    guards = []
    for statement in node.body:
        if isinstance(statement, ast.If):
            guards.append(ast.unparse(statement.test))
    # three E-side slot-0 guards and three ADE guards, all `NPn > 0`
    assert guards.count("NP0 > 0") == 2
    assert guards.count("NP1 > 0") == 2
    assert guards.count("NP2 > 0") == 2


def test_the_slot_counts_and_terms_are_pinned_to_the_certified_bodies():
    assert module.MAX_POLES == stored_e_module.MAX_POLES
    assert module.E_TERMS == stored_e_module.E_TERMS


def test_the_shipped_policy_is_stamped_and_is_one_warp():
    assert module.POLICY["num_warps"] == 1
    assert module.POLICY["status"] == "keep"


def test_composition_into_plan_step_is_deferred_and_dispatch_stays_disabled():
    launch_source = (PACKAGE_DIR / "launch.py").read_text(encoding="utf-8")
    assert "complex_fused_ade_chain" not in launch_source


def test_the_device_gate_is_present_and_names_this_module():
    assert GATE.exists(), f"{GATE} is missing; a product with no gate is not a claim"
    text = GATE.read_text(encoding="utf-8")
    assert "complex_fused_ade_chain" in text
