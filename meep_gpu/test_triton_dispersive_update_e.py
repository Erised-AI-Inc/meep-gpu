"""Tests for the Triton DISPERSIVE ``update_E`` kernel — coverage, tables, plan.

Everything here runs on a laptop: no GPU, no CuPy, no Triton. What needs hardware —
byte identity against the array path, the mutation legs, throughput — lives in the
shared gate (``parity/meep_gpu/probe_fused_kernel_bit_identity.py --track triton
--experiments dispersive``), because a byte comparison against the array path is
only meaningful on the device that runs it.

What is pinned HERE is everything that decides whether the kernel is ever ALLOWED
to run, plus the two things about it that are wrong SILENTLY rather than loudly:

* **coverage**, with mutations proving each added clause is load-bearing;
* **disjointness** from :func:`coverage.constitutive_coverage`. Two predicates
  admitting one configuration means ``plan_step`` picks a kernel by ordering, and
  the wrong pick is a smooth, plausible, wrong field;
* **the pole order**, derived from the same list ``Fields`` itself walks;
* **the per-launch pole resolution**, because ``PolarizationState.update`` rotates
  the buffers this kernel reads and a cached view is stale after one component of
  one step.

The transcription itself is pinned by
``parity/meep_gpu/validate_pml_reference_vs_stepping.py``, also on the laptop:
140/140 dispersive rows bit-identical to ``stepping.update_E`` with 224/224 pole
controls discriminating.
"""

from __future__ import annotations

import ast
import builtins
import hashlib
import importlib
import json
import pathlib
import re
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from meep_gpu import stepping
from meep_gpu.dispersion import PolarizationState, Susceptibility
from meep_gpu.fields import Fields
from meep_gpu.grid import Grid
from meep_gpu.pml import PML
from meep_gpu.triton_kernels import coverage as coverage_module
from meep_gpu.triton_kernels import dispersive_update_e as module
from meep_gpu.triton_kernels import launch as launch_module

PACKAGE_DIR = pathlib.Path(module.__file__).parent


def build(cell_size=(0.8, 0.8, 0.8), **grid_kwargs):
    grid = Grid(resolution=10.0, cell_size=cell_size, **grid_kwargs)
    fields = Fields(grid=grid, force_complex_fields=False)
    fields.enable_pml_storage()
    return fields, PML(grid=grid, thickness=2)


def with_poles(counts=(1, 1, 1), kind="lorentzian", **grid_kwargs):
    """A real dispersive ``Fields`` whose component ``c`` is driven by ``counts[c]`` poles.

    Built the way the ENGINE builds one: pole ``n`` gets a sigma of exactly zero on
    any component it must not drive, and ``sigma_is_trivial`` (dispersion.py:600)
    drops it there. So a ``(1, 0, 2)`` configuration is an ordinary anisotropic
    material, not a harness contrivance.
    """
    fields, pml = build(**grid_kwargs)
    fields.enable_field_storage()
    components = ("Ex", "Ey", "Ez")
    states = []
    for index in range(max(counts) if max(counts) else 0):
        term = Susceptibility(frequency=1.0, gamma=0.1, kind=kind)
        sigmas = {name: (0.3 + 0.05 * index) if counts[axis] > index else 0.0
                  for axis, name in enumerate(components)}
        state = PolarizationState(term, sigmas, fields.grid, np.float32)
        fields.polarizations.append(state)
        states.append(state)
    return fields, pml, states


def reasons(fields, pml):
    """The refusal reasons, minus the CuPy-backend one every laptop run carries."""
    return [r for r in module.dispersive_constitutive_coverage(fields, pml).reasons
            if "not cupy" not in r]


# ---------------------------------------------------------------------------
# The optional dependency must stay optional
# ---------------------------------------------------------------------------

@pytest.fixture
def restored_module():
    """Put the module's ORIGINAL objects back after a reload with Triton blocked.

    ``importlib.reload`` mutates the module object every other module holds, and
    rebinds its names to FRESH objects. Siblings that did ``from ... import E_TERMS``
    keep the object bound at their own import, so after this test the source module
    and its consumers hold two equal-but-distinct tuples. MEASURED 2026-08-17: that
    is exactly how ``test_triton_folded_offdiag_dispersive_update_e`` failed its
    ``module.E_TERMS is dispersive_update_e.E_TERMS`` restatement check — but only in
    a full-suite run, and it passed alone, so the failure read as record drift in a
    file this one does not touch.

    Reloading a second time (the fixture pattern in ``test_triton_cylindrical.py``)
    restores a WORKING module but not the original objects — it mints a third tuple,
    and an identity check still fails. So snapshot ``__dict__`` and put it back.

    Requested FIRST in the signature so it is set up first and therefore torn down
    LAST, after ``monkeypatch`` has restored ``__import__``.
    """
    live = importlib.import_module("meep_gpu.triton_kernels.dispersive_update_e")
    pristine = dict(live.__dict__)
    yield
    live.__dict__.clear()
    live.__dict__.update(pristine)


def test_the_module_imports_and_answers_coverage_with_triton_absent(restored_module,
                                                                   monkeypatch):
    """A missing optional dependency must not break the engine — or this module.

    The kernel is built behind :func:`constitutive_step_dispersive_kernel` rather
    than decorated at module scope for exactly this reason: ``@triton.jit`` runs at
    import time, and the predicate has to be importable and answerable on the machine
    that is the merge bar.
    """
    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "triton" or name.startswith("triton."):
            raise ImportError("triton is blocked for this test")
        return real_import(name, *args, **kwargs)

    for name in [n for n in sys.modules if n.startswith("triton")]:
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setattr(builtins, "__import__", blocked)
    reloaded = importlib.reload(
        importlib.import_module("meep_gpu.triton_kernels.dispersive_update_e"))
    fields, pml, _ = with_poles()
    verdict = reloaded.dispersive_constitutive_coverage(fields, pml)
    assert verdict.covered is False           # NumPy backend, and it says so
    assert any("not cupy" in reason for reason in verdict.reasons)
    with pytest.raises(ImportError, match="optional"):
        reloaded.constitutive_step_dispersive_kernel()


def test_the_module_names_no_other_track_and_builds_no_options_dict():
    """File ownership and the banned launch route, enforced rather than agreed.

    ``options={...}`` silently DROPS an unrecognised name, so a guard passed that way
    compiles contracted with no error; the JIT launch keyword raises. The rule is
    package-wide and this file is inside the package, so the same check applies.
    """
    tree = ast.parse(pathlib.Path(module.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)) and ast.get_docstring(node):
            node.body = node.body[1:]
    code = ast.unparse(tree)
    assert "options=" not in code
    assert code.count("enable_fp_fusion=") == 1
    assert ("enable_fp_fusion=ENABLE_FP_FUSION if guard is None else bool(guard)"
            in code), "the launch site must spell the shared guard constant"


# ---------------------------------------------------------------------------
# The tables the kernel hard-codes, against the array path's own
# ---------------------------------------------------------------------------

def test_the_term_table_is_steppings_own():
    """``E_TERMS`` must be ``stepping.E_CONSTITUTIVE_TERMS`` with the axis resolved.

    A swapped component-to-axis mapping is a wrong coefficient in every cell and no
    error anywhere, so it is read off the engine rather than trusted here.
    """
    axis_of = {"x": 0, "y": 1, "z": 2}
    expected = tuple((target, source, axis_of[axis])
                     for target, source, axis in stepping.E_CONSTITUTIVE_TERMS)
    assert module.E_TERMS == expected


def test_the_half_integer_pairing_is_the_one_stepping_uses():
    """``update_E`` reads the HALF-INTEGER tables (stepping.py:1015).

    Checked by fetching both sets off a real layer and asserting the plan's choice is
    the one ``_constitutive_coefficients`` actually returns. Swapped, it is a
    half-cell error in the absorber profile — converged, smooth and wrong.
    """
    fields, pml, _ = with_poles()
    assert module.HALF_INTEGER is True
    for axis in ("x", "y", "z"):
        kps, kms = stepping._constitutive_coefficients(pml, axis, half_integer=True)
        assert kps is getattr(pml, f"kps_{axis}_h")
        assert kms is getattr(pml, f"kms_{axis}_h")
        assert kps is not getattr(pml, f"kps_{axis}")


def test_max_poles_matches_the_slots_the_kernel_actually_compiles():
    """The predicate's refusal threshold and the kernel's slot count are one number.

    They are in two places — a ``tl.constexpr`` chain in the kernel and an integer in
    the predicate — and if they drift the predicate admits a run whose last poles are
    never subtracted. That is a smooth wrong field, so it is checked against the
    kernel's own SOURCE rather than against a second constant.
    """
    source = pathlib.Path(module.__file__).read_text(encoding="utf-8")
    body = source.split("def constitutive_step_dispersive(", 1)[1]
    for letter in ("a", "b", "c"):
        slots = sorted(set(int(n) for n in re.findall(
            rf"tl\.load\({letter}(\d) \+ idx", body)))
        assert slots == list(range(module.MAX_POLES)), (letter, slots)
    guards = sorted(set(int(n) for n in re.findall(r"if NP\d > (\d):", body)))
    assert guards == list(range(module.MAX_POLES))


def test_the_shared_clause_builders_this_predicate_composes_from_still_exist():
    """This predicate REUSES ``coverage``'s clauses rather than restating them.

    That is the right call — a second transcription of the grid rules could drift
    from the one gating the other three kernels — but the names it reaches for are
    module-private. So a rename in the shared file must fail HERE, at the merge bar,
    rather than silently dropping a clause from this predicate.
    """
    for name in module.SHARED_CLAUSES:
        assert callable(getattr(coverage_module, name, None)), name


# ---------------------------------------------------------------------------
# Coverage — what is admitted
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("counts", [(1, 1, 1), (2, 2, 2), (3, 2, 1), (1, 0, 2),
                                    (6, 6, 6), (8, 8, 8)])
def test_a_real_dispersive_pml_configuration_is_refused_only_for_the_backend(counts):
    """Every pole set the gate sweeps, plus the MAX_POLES boundary, is admitted."""
    fields, pml, _ = with_poles(counts)
    assert reasons(fields, pml) == [], (counts, reasons(fields, pml))


def test_a_drude_pole_is_admitted_too():
    fields, pml, _ = with_poles((1, 1, 1), kind="drude")
    assert reasons(fields, pml) == [], reasons(fields, pml)


def test_the_pole_order_is_the_list_fields_itself_walks():
    """``poles_per_component`` must reproduce ``displacement_minus_polarization``'s list.

    ``Fields`` builds ``[state for state in self.polarizations if state.drives(c)]``
    (fields.py:1097) and subtracts in THAT order; the order is bit-load-bearing
    (reversing it was caught 20/20 on the real engine at two poles). Deriving it
    twice is how the two drift, so this asserts the module's single derivation equals
    the engine's.
    """
    fields, _, states = with_poles((3, 2, 1))
    resolved = module.poles_per_component(fields)
    for component in ("Ex", "Ey", "Ez"):
        expected = [s for s in fields.polarizations if s.drives(component)]
        assert list(resolved[component]) == expected
    assert [len(resolved[c]) for c in ("Ex", "Ey", "Ez")] == [3, 2, 1]
    assert list(resolved["Ex"]) == states[:3]


# ---------------------------------------------------------------------------
# Disjointness — the two predicates must never both admit
# ---------------------------------------------------------------------------

def _constitutive_reasons(fields, pml):
    return [r for r in coverage_module.constitutive_coverage(fields, pml, "E").reasons
            if "not cupy" not in r]


def test_the_two_update_E_predicates_are_disjoint():
    """The whole point of clause (g), measured on both sides of the split.

    With a pole registered the plain predicate refuses and this one admits; with none
    registered the plain one admits and this one refuses. If BOTH ever admitted,
    ``plan_step`` would have two legal kernels for one sub-step and would pick by
    ordering — which is how a wrong answer gets chosen at random.
    """
    dispersive_fields, pml, _ = with_poles((1, 1, 1))
    assert reasons(dispersive_fields, pml) == []
    assert _constitutive_reasons(dispersive_fields, pml) != []

    plain, plain_pml = build()
    plain.enable_field_storage()
    assert _constitutive_reasons(plain, plain_pml) == []
    plain_reasons = reasons(plain, plain_pml)
    assert plain_reasons != []
    assert any("no susceptibility is registered" in r for r in plain_reasons)


def test_an_inert_zero_sigma_pole_lands_on_exactly_one_side_of_the_split():
    """The asymmetry between the two predicates, made explicit rather than accidental.

    A pole whose sigma is identically zero drives nothing: ``sigma_is_trivial``
    allocates no P and the run is byte-identical to the non-dispersive engine. The
    plain predicate's clause fires on ``polarizations`` being non-empty, so it refuses
    such a run; THIS predicate also refuses it, because clause (g) counts states and
    this one drives nothing worth a kernel. Both refusing is safe — the array path
    serves it — and it is asserted so the gap is a recorded decision rather than a
    surprise at integration.
    """
    fields, pml, states = with_poles((1, 1, 1))
    for state in states:
        state._driven = ()
    assert module.poles_per_component(fields) == {"Ex": (), "Ey": (), "Ez": ()}
    verdict = reasons(fields, pml)
    assert verdict == [], verdict  # states exist, so (g) is satisfied
    assert _constitutive_reasons(fields, pml) != []


# ---------------------------------------------------------------------------
# Coverage — what is refused, by name
# ---------------------------------------------------------------------------

def test_no_active_layer_is_refused_with_its_own_reason():
    """Without PML ``update_E`` writes ``field[...] = constitutive`` — a DIFFERENT sub-step."""
    fields, _, _ = with_poles()
    verdict = reasons(fields, None)
    assert any("no active PML" in r for r in verdict), verdict
    assert any("stepping.py:993" in r for r in verdict), verdict  # stepping.py live lines for the frozen device-text citation(s) in this string: 993->1022


def test_off_diagonal_epsilon_is_refused():
    """Admitted by the CURL predicate, refused here — the per-sub-step split, again."""
    fields, pml, _ = with_poles()
    fields._chi1inv_offdiagonal = {"Ex": {"Ey": 0.1}}
    assert fields.has_offdiagonal_epsilon
    verdict = reasons(fields, pml)
    assert any("off-diagonal" in r for r in verdict), verdict
    assert [r for r in coverage_module.pml_curl_coverage(fields, pml).reasons
            if "not cupy" not in r] == []


def test_a_mirror_plane_is_refused():
    fields, pml, _ = with_poles(symmetry=("X",))
    assert any("mirror" in r for r in reasons(fields, pml))


def test_a_nonzero_k_point_is_refused():
    fields, pml, _ = with_poles(k_point=(0.3, 0.0, 0.0))
    assert any("k_point" in r for r in reasons(fields, pml))


def test_complex_storage_is_refused():
    grid = Grid(resolution=10.0, cell_size=(0.8, 0.8, 0.8))
    fields = Fields(grid=grid, force_complex_fields=True)
    fields.enable_pml_storage()
    assert any("complex" in r for r in
               module.dispersive_constitutive_coverage(fields, PML(grid=grid,
                                                                   thickness=2)).reasons)


def test_a_kind_outside_lorentzian_drude_is_refused():
    fields, pml, states = with_poles()
    states[0].susceptibility = type("S", (), {"kind": "gyrotropic-lorentzian"})()
    assert any("gyrotropic" in r for r in reasons(fields, pml))


def test_more_poles_than_the_kernel_has_slots_is_refused_by_name():
    """Refused, never truncated. A dropped pole is a smooth wrong field."""
    fields, pml, _ = with_poles((module.MAX_POLES + 1,) * 3)
    verdict = reasons(fields, pml)
    assert any(f"MAX_POLES={module.MAX_POLES}" in r for r in verdict), verdict
    assert sum("driven by" in r for r in verdict) == 3


def test_a_missing_P_volume_is_refused():
    fields, pml, states = with_poles((1, 1, 1))
    del states[0].P["Ez"]
    assert any("P[Ez] is not allocated" in r for r in reasons(fields, pml))


def test_a_missing_P_prev_volume_is_refused_even_though_this_kernel_never_reads_it():
    """``update_P`` ROTATES the three buffers as a set (dispersion.py:687-691).

    A malformed ``P_prev`` is this kernel's ``P`` one step later, so the layout clause
    covers it now rather than one sub-step too late.
    """
    fields, pml, states = with_poles((1, 1, 1))
    del states[0].P_prev["Ez"]
    assert any("P_prev[Ez] is not allocated" in r for r in reasons(fields, pml))


def test_a_non_contiguous_pole_volume_is_refused():
    fields, pml, states = with_poles((1, 1, 1))
    states[0].P["Ez"] = np.zeros(
        tuple(n * 2 for n in fields.grid.shape), dtype=np.float32)[::2, ::2, ::2]
    verdict = reasons(fields, pml)
    assert any("contiguous" in r for r in verdict), verdict


def test_a_float64_pole_volume_is_refused():
    fields, pml, states = with_poles((1, 1, 1))
    states[0].P["Ez"] = np.zeros(fields.grid.shape, dtype=np.float64)
    assert any("float32" in r for r in reasons(fields, pml))


def test_a_scalar_inverse_epsilon_is_refused():
    fields, pml, _ = with_poles()
    fields.inverse_epsilon_for = lambda component: 1.0
    assert any("scalar" in r for r in reasons(fields, pml))


# ---------------------------------------------------------------------------
# Mutations — each added clause is load-bearing, measured not asserted
# ---------------------------------------------------------------------------

def _mutate(function, needle, replacement):
    """Recompile one function with one line changed — the predicate's own gate."""
    import inspect
    import textwrap

    source = textwrap.dedent(inspect.getsource(function))
    assert needle in source, needle
    namespace = dict(function.__globals__)
    exec(compile(source.replace(needle, replacement), "<mutated>", "exec"),  # noqa: S102
         namespace)
    return namespace[function.__name__]


def test_mutation_dropping_the_empty_pole_set_clause_breaks_disjointness(monkeypatch):
    """Clause (g) removed: BOTH update_E predicates then admit a non-dispersive run.

    This is the failure the clause exists to prevent, and it is silent — ``plan_step``
    would install whichever kernel its branch order reached first.
    """
    mutated = _mutate(module.dispersive_constitutive_coverage,
                      "if not states:", "if False:")
    monkeypatch.setattr(module, "dispersive_constitutive_coverage", mutated)
    plain, pml = build()
    plain.enable_field_storage()
    assert _constitutive_reasons(plain, pml) == []
    assert reasons(plain, pml) == [], "the mutation should have made both admit"


def test_mutation_dropping_the_pole_count_clause_admits_an_uncarryable_run(monkeypatch):
    mutated = _mutate(module.dispersive_constitutive_coverage,
                      "if count > MAX_POLES:", "if False:")
    monkeypatch.setattr(module, "dispersive_constitutive_coverage", mutated)
    fields, pml, _ = with_poles((module.MAX_POLES + 3,) * 3)
    assert reasons(fields, pml) == [], "the mutation should have admitted it"


def test_mutation_dropping_the_pml_clause_admits_the_wrong_sub_step(monkeypatch):
    mutated = _mutate(
        module.dispersive_constitutive_coverage,
        'if pml is None or not getattr(pml, "is_active", False):', "if False:")
    monkeypatch.setattr(module, "dispersive_constitutive_coverage", mutated)
    fields, _, _ = with_poles()
    # _grid_reasons still refuses it, which is the point: the two clauses are
    # independent and the added one is the sub-step-specific REASON, not the guard.
    verdict = reasons(fields, None)
    assert verdict != []
    assert not any("stepping.py:993" in r for r in verdict)  # stepping.py live lines for the frozen device-text citation(s) in this string: 993->1022


# ---------------------------------------------------------------------------
# The plan — pointers resolved per launch, order re-checked per launch
# ---------------------------------------------------------------------------

def test_the_live_binding_re_resolves_P_after_the_ADE_rotation():
    """``PolarizationState.update`` rotates P/P_prev/_scratch EVERY component EVERY step.

    A plan caching device views is stale after the first component of the first step,
    and stale in a way that still computes. This asserts the binding hands back the
    CURRENT array after a rotation, not the one it was built with.
    """
    fields, _, states = with_poles((1, 1, 1))
    binding = module.LivePoleBinding(fields, module.poles_per_component(fields))
    first = binding.arrays()
    assert [len(group) for group in first] == [1, 1, 1]
    states[0].update(lambda component: getattr(fields, component), fields.grid.dt)
    second = binding.arrays()
    for component_index, component in enumerate(("Ex", "Ey", "Ez")):
        assert second[component_index][0] is states[0].P[component]
        assert second[component_index][0] is not first[component_index][0]


def test_the_live_binding_refuses_a_polarization_set_that_changed():
    """A susceptibility appended mid-run is a wrong ANSWER, not a crash — so it raises.

    The driver already refuses it; this assertion costs three comparisons against a
    tuple of at most eight objects and the failure mode it catches is silent.
    """
    fields, _, _ = with_poles((1, 1, 1))
    binding = module.LivePoleBinding(fields, module.poles_per_component(fields))
    extra = PolarizationState(Susceptibility(frequency=1.0, gamma=0.1),
                              {"Ex": 0.2, "Ey": 0.2, "Ez": 0.2},
                              fields.grid, np.float32)
    fields.polarizations.append(extra)
    with pytest.raises(RuntimeError, match="bit-load-bearing"):
        binding.arrays()


def test_the_static_binding_reports_its_own_counts():
    binding = module.StaticPoleBinding([[1, 2], [], [3]])
    assert binding.counts == (2, 0, 1)
    assert binding.arrays() == ((1, 2), (), (3,))
    with pytest.raises(ValueError, match="one group per E component"):
        module.StaticPoleBinding([[1], [2]])


def test_planning_a_refused_configuration_returns_None_without_needing_triton():
    """None means refused, and the refusal must not depend on the optional dependency.

    The Triton import stays BELOW the predicate in every builder in this package: a
    NumPy host must be able to plan (to ``None``) on a machine that has never heard
    of Triton.
    """
    fields, pml, _ = with_poles()
    assert module.plan_dispersive_constitutive(fields, pml) is None


# ---------------------------------------------------------------------------
# Composition — the specialized update_E must be selected fail-closed
# ---------------------------------------------------------------------------

def _composition_fields():
    """Smallest object carrying the state inventory ``plan_step`` consults."""
    return SimpleNamespace(polarizations=())


def _install_composition_stubs(monkeypatch, *, ordinary_e, dispersive_e):
    """Isolate update_E routing from every other already-tested plan branch."""
    refused = coverage_module.Coverage(False, ("outside this routing test",))
    monkeypatch.setattr(
        launch_module, "pml_curl_coverage",
        lambda fields, pml, sub_step: refused)
    monkeypatch.setattr(
        launch_module, "conductive_pml_curl_coverage",
        lambda fields, pml, sub_step: refused,
        raising=False,
    )
    monkeypatch.setattr(
        launch_module,
        "constitutive_coverage",
        lambda fields, pml, side: ordinary_e if side == "E" else refused,
    )
    monkeypatch.setattr(
        launch_module,
        "dispersive_constitutive_coverage",
        lambda fields, pml: dispersive_e,
        raising=False,
    )


def test_plan_step_installs_the_dispersive_update_E_when_the_ordinary_one_refuses(
        monkeypatch):
    """One admitted specialized predicate closes update_E; branch order does not.

    This is the composition that turns the already-gated kernel product into a
    five-sub-step dispersive plan.  The builder is replaced with a sentinel so the
    routing test remains runnable without importing Triton on the laptop.
    """
    ordinary = coverage_module.Coverage(False, ("ordinary E refuses poles",))
    specialized = coverage_module.Coverage(True, ())
    _install_composition_stubs(
        monkeypatch, ordinary_e=ordinary, dispersive_e=specialized)
    sentinel = object()
    monkeypatch.setattr(
        launch_module,
        "plan_dispersive_constitutive",
        lambda fields, pml, block=None: sentinel,
        raising=False,
    )

    plan = launch_module.plan_step(_composition_fields(), object())

    assert plan.plans["update_E"] is sentinel
    assert "update_E" in plan.replaces
    assert "update_E" not in plan.reasons


def test_plan_step_refuses_an_overlap_between_ordinary_and_dispersive_update_E(
        monkeypatch):
    """Two predicates admitting update_E is a defect, never an ordering choice."""
    admitted = coverage_module.Coverage(True, ())
    _install_composition_stubs(
        monkeypatch, ordinary_e=admitted, dispersive_e=admitted)
    monkeypatch.setattr(
        launch_module, "plan_constitutive", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        launch_module,
        "plan_dispersive_constitutive",
        lambda *args, **kwargs: object(),
        raising=False,
    )

    plan = launch_module.plan_step(_composition_fields(), object())

    assert "update_E" not in plan.plans
    assert any("both" in reason and "update_E" in reason
               for reason in plan.reasons["update_E"])


def test_plan_step_reports_every_update_E_refusal_product(monkeypatch):
    """A refusal explains every eligible product instead of hiding one branch.

    The tuple grew when stored-E and then null constitutive were wired. Both
    entries are correct rather than incidental: ``pml=object()`` cannot answer
    ``is_active``, and an unreadable layer consults both inert-layer arms through
    ``launch.absorber_inactive`` rather than being assumed active. They refuse by
    name on the gridless double, in the same table order the composer uses.
    """
    ordinary = coverage_module.Coverage(False, ("ordinary reason",))
    specialized = coverage_module.Coverage(False, ("dispersive reason",))
    _install_composition_stubs(
        monkeypatch, ordinary_e=ordinary, dispersive_e=specialized)

    plan = launch_module.plan_step(_composition_fields(), object())

    assert plan.reasons["update_E"] == (
        "ordinary: ordinary reason",
        "dispersive: dispersive reason",
        "no-PML stored E: fields carries no grid",
        "no-PML null: fields carries no grid",
    )


def test_the_dispersive_update_E_composition_is_public_but_dispatch_is_not():
    """MEEP-side users may build the experimental plan without activating it."""
    import meep_gpu.triton_kernels as public

    for name in (
        "dispersive_constitutive_coverage",
        "plan_dispersive_constitutive",
    ):
        assert name in public.__all__
        assert callable(getattr(public, name))


def test_the_dispersive_composition_fingerprint_names_the_exact_gated_bytes():
    """The public route, specialized kernel, and A6000 verdict are one record."""
    record = json.loads((PACKAGE_DIR / "fingerprints.json").read_text(encoding="utf-8"))
    specialized = record["specialized_kernel_sources"]["dispersive_update_e.py"]
    assert specialized["kernel"] == "constitutive_step_dispersive"
    assert specialized["sha256"] == hashlib.sha256(
        pathlib.Path(module.__file__).read_bytes()).hexdigest()

    gate = record["dispersive_composition_gate"]
    # ``slurm_job_id`` NAMES THE JOB THAT FIRST CERTIFIED THIS WELD and does not move
    # when the record is re-cut; ``records`` names the artifact the digests are bound
    # to. Both siblings re-cut on 2026-08-30 carry the same split (2319 and 2317
    # beside a 2026-08-30 records line), and ``recut_composition_records.MUTABLE``
    # deliberately excludes the job id for that reason.
    assert gate["slurm_job_id"] == 2298
    # RE-PINNED 2026-08-31 from a fresh GPU-host run of probe_triton_engine_route,
    # bound by recut_composition_records.py with the move declared in CLAIM_RECUTS.
    # Both counters moved, for two different reasons, and neither is a regression:
    #
    #   4/4 -> 8/8   the WHOLE-STEP LEG gained six cases, not the gate. It runs on
    #                every case that composes a non-empty plan, and the four cases
    #                that composed nothing at all on 2026-08-11 now compose through
    #                wired arms — so six complete driver steps are compared as
    #                uint32 on eight cases instead of four, and all eight are
    #                identical. Strictly more measured.
    #   4/4 -> 3/4   the plain PML curl no longer declines BOTH curls of
    #                `2d_cond_pml`. That case carries a D-side conductivity only, so
    #                `pml_curl_coverage` refuses it in AGGREGATE (all six
    #                CURL_TARGETS) while its `step_B` verdict names no conductivity
    #                reason at all — measured on the device: `step_B` covered, zero
    #                reasons; `step_D` refused, three named. The composition then
    #                serves all four of its sub-steps byte-identically.
    #
    # Pinned EXACTLY rather than by prefix, and the trailing prose with it: a re-cut
    # that replaced a counter and dropped the record's more specific sentence would
    # pass a `startswith` check.
    assert gate["real_engine_route"]["covered_curl_identical"] == "4/4"
    assert gate["real_engine_route"]["covered_whole_step_identical"] == (
        "8/8 over six complete steps")
    assert gate["real_engine_route"]["refusals_returned_none"] == "3/4"
    # THE COUNTER THAT REPLACES WHAT 4/4 USED TO MEAN, and the reason dropping to
    # 3/4 above costs this weld nothing. The old counter collapsed two builder calls
    # into one bit with `all(... is None)`; this one is an EQUIVALENCE per curl
    # sub-step — `plan_pml_curl` returns None exactly where its own named-sub-step
    # predicate refuses — asked on ALL EIGHT cases rather than the four the probe
    # lists as refusals, so it also catches the direction the old one never looked
    # at: a builder returning None where its predicate ADMITS.
    assert gate["real_engine_route"]["builder_matches_its_predicate"] == "8/8"
    assert gate["real_engine_route"]["dispersive_replaces"] == [
        "step_B", "update_H", "step_D", "update_E", "update_P",
    ]
