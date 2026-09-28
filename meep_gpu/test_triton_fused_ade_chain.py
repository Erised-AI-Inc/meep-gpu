"""Laptop contracts for the fused E->P chain: ``update_E`` welded to ``update_P``.

The CUDA gate owns byte identity — nothing here compiles or launches anything.
These tests own the four things that decide whether the device run measures what
it claims:

* **the seam's admission clauses**, and the refusals that must stay refusals — a
  seventh pole, complex storage, an off-diagonal row, and the arm/absorber pairing
  that decides which field the fused register IS;
* **the shape**, walked on real engine objects. One scratch per driven component
  is this product's whole licence, and
  :func:`test_the_rotation_orbit_is_alias_free_at_every_position` re-asks the shape
  probe's LEG 2 enumeration through the SHIPPED plan rather than through a model of
  it, with the alias check armed;
* **the transcription.** Every arithmetic line is required to appear in this
  kernel exactly as the certified body spells it, PARSED out of both files. Nothing
  here re-implements a line: a test that re-derives a kernel's arithmetic mirrors a
  defect instead of executing it, which is a class this project has already paid
  for;
* **the launcher's argument ORDER.** 154 runtime arguments reach the kernel
  positionally, and a launcher that assembles the right pointers in the wrong order
  binds every later one to a neighbouring slot — a wrong answer, not a crash.

WHY ``CHAIN_MAX_POLES``' ADJUDICATION LIVES HERE. Kernel modules are bound by raw
sha256 in ``triton_kernels/fingerprints.json`` once their device gate lands, and a
prose edit there makes a standing board unreproducible until the gate is re-run.
So the cap's reasoning is pinned in tests, which nothing binds, and a test cannot
go stale in silence the way a comment can.
"""

from __future__ import annotations

import ast
import inspect
import pathlib
import re

import numpy as np
import pytest

from meep_gpu.dispersion import DRUDE, LORENTZIAN, PolarizationState, Susceptibility
from meep_gpu.fields import Fields
from meep_gpu.grid import Grid, Mirror
from meep_gpu.pml import PML
from meep_gpu.triton_kernels import dispersive_update_e as dispersive_module
from meep_gpu.triton_kernels import fused_ade_chain as module
from meep_gpu.triton_kernels import no_pml_stored_e as stored_e_module

PACKAGE_DIR = pathlib.Path(module.__file__).parent
API_ROOT = PACKAGE_DIR.parents[1]
GATE = API_ROOT / "parity" / "meep_gpu" / "gate_triton_fused_ade_chain.py"
SOURCE = pathlib.Path(module.__file__).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def engine(arm: str, poles: int = 2, sigma=None, kind=LORENTZIAN,
           complex_storage: bool = False, offdiag: bool = False,
           nonlinear: bool = False):
    """One configuration on the laptop's NumPy backend, for the arm named.

    THE ARM DECIDES THE CONFIGURATION, not just the predicate: ``no_pml`` is an
    inert layer with stored E, ``dispersive`` an absorbing one, and ``folded`` an
    absorbing one with a live mirror plane. The three are what the three arms
    exist for and each must be refused by the other two.
    """
    folded = arm == "folded"
    grid = Grid(resolution=10.0,
                cell_size=(1.3, 1.6, 0.9) if folded else (0.8, 0.8, 0.8),
                symmetry=(Mirror("Y", +1),) if folded else (), xp=np)
    fields = Fields(grid=grid, force_complex_fields=complex_storage)
    names = ("Ex", "Ey", "Ez")
    if offdiag:
        diagonal = {name: np.full(grid.shape, 2.25, dtype=np.float32)
                    for name in names}
        inverse = {name: np.full(grid.shape, 1.0 / 2.25, dtype=np.float32)
                   for name in names}
        rows = {"Ex": {"Ey": np.full(grid.shape, 0.03, dtype=np.float32)}}
        fields.set_epsilon_volumes(diagonal, inverse, rows)
    else:
        fields.set_background_eps(2.25)
    if nonlinear:
        fields.set_nonlinear_volumes({name: 0.02 for name in names},
                                     {name: 0.05 for name in names})
    if arm == "no_pml":
        fields.enable_field_storage()
    else:
        fields.enable_pml_storage()
    for index in range(poles):
        fields.polarizations.append(PolarizationState(
            Susceptibility(0.8 + 0.1 * index, 0.1, kind),
            0.4 if sigma is None else sigma, grid,
            np.complex64 if complex_storage else np.float32))
    if arm == "no_pml":
        return fields, PML(grid=grid, thickness=0)
    # A MIRRORED axis takes the HIGH FACE ONLY: cell 0 lies on the mirror plane,
    # which is a boundary condition and not an absorber (pml.py:408 refuses the
    # low face by name).
    thickness = tuple({"high": 2} if grid.is_mirrored(axis) else 2
                      for axis in range(3))
    return fields, PML(grid=grid, thickness=dict(zip("xyz", thickness)))


def reasons(fields, pml, arm):
    """Refusals other than the laptop's own array module."""
    return [reason for reason
            in module.fused_ade_chain_coverage(fields, pml, arm).reasons
            if "not cupy" not in reason]


class FakePointer:
    """Stands in for ``CupyPointer`` so the plan can be built without a device."""

    __slots__ = ("array",)

    def __init__(self, array):
        self.array = array


def plan_for(fields, pml, arm):
    return module.FusedAdeChainPlan(fields, pml, arm, 256, num_warps=1,
                                    pointer=FakePointer)


def kernel_text() -> str:
    """The shipped kernel's own source, read from the FILE.

    Read rather than imported: these tests run on a host with no Triton, where the
    kernel object is ``None``, so the transcription check has to bite at the merge
    bar and not only on a device run.
    """
    return _function_text(pathlib.Path(module.__file__), "fused_ade_chain_step")


def _function_text(path: pathlib.Path, name: str) -> str:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    node = next(child for child in ast.walk(tree)
                if isinstance(child, ast.FunctionDef) and child.name == name)
    segment = ast.get_source_segment(source, node)
    assert segment, f"{name} has no source segment in {path}"
    return segment


def _normalised(text: str) -> str:
    """Collapse runs of whitespace so a re-wrapped line still matches."""
    return re.sub(r"\s+", " ", text)


# ---------------------------------------------------------------------------
# Admission and refusal
# ---------------------------------------------------------------------------

def test_the_module_imports_without_triton_and_refuses_before_any_launch():
    fields, pml = engine("dispersive")
    verdict = module.fused_ade_chain_coverage(fields, pml, "dispersive")
    assert not verdict.covered                       # the laptop has no CuPy
    assert any("not cupy" in reason for reason in verdict.reasons)
    assert reasons(fields, pml, "dispersive") == []
    assert module.plan_fused_ade_chain(fields, pml, "dispersive") is None
    if module.fused_ade_chain_step is None:
        with pytest.raises(ImportError, match="triton"):
            module.fused_ade_chain_kernel()


@pytest.mark.parametrize("arm", module.ARMS)
@pytest.mark.parametrize("poles", (1, 2, 5, 6))
@pytest.mark.parametrize("sigma", (
    0.4,
    {"Ex": 0.4, "Ey": 0.0, "Ez": 0.25},
    {"Ex": 0.0, "Ey": 0.0, "Ez": 0.25},
))
def test_the_covered_shapes_leave_no_refusal_but_the_array_module(arm, poles, sigma):
    fields, pml = engine(arm, poles=poles, sigma=sigma)
    assert reasons(fields, pml, arm) == []


def test_an_unknown_arm_is_refused_by_name():
    fields, pml = engine("dispersive")
    verdict = module.fused_ade_chain_coverage(fields, pml, "complex")
    assert not verdict.covered
    assert any("arm must be one of" in reason for reason in verdict.reasons)


@pytest.mark.parametrize("arm", module.ARMS)
def test_a_seventh_pole_is_refused_by_name_and_never_truncated(arm):
    """A pole the E chain subtracts but no ADE arm advances is a frozen P."""
    fields, pml = engine(arm, poles=module.CHAIN_MAX_POLES + 1)
    verdict = reasons(fields, pml, arm)
    assert verdict, "seven poles must not be admitted"
    assert all(f"is driven by {module.CHAIN_MAX_POLES + 1} poles" in reason
               for reason in verdict)
    assert any("frozen P" in reason for reason in verdict)
    assert module.plan_fused_ade_chain(fields, pml, arm) is None
    # ...and the PLAN refuses it too, so a caller bypassing the predicate cannot
    # build a chain whose seventh pole is silently unadvanced.
    with pytest.raises(ValueError, match="ADE arms per component"):
        plan_for(fields, pml, arm)


def test_the_cap_is_GUARDED_and_not_dead_code(monkeypatch):
    """Six poles admit; the SAME configuration is refused with the cap at five."""
    fields, pml = engine("dispersive", poles=6)
    assert reasons(fields, pml, "dispersive") == []
    monkeypatch.setattr(module, "CHAIN_MAX_POLES", 5)
    assert any("is driven by 6 poles" in reason
               for reason in reasons(fields, pml, "dispersive"))


def test_the_cap_is_below_the_E_halfs_own_slot_count():
    """The ADE half is the binding bound, and the E chain has room to spare."""
    assert module.CHAIN_MAX_POLES < module.MAX_POLES


@pytest.mark.parametrize("arm", module.ARMS)
def test_exactly_one_arm_admits_each_configuration(arm):
    """The three arms are DISJOINT, and each pairing is decided together.

    The register the recurrence reads is whichever field THAT arm's E half wrote,
    so asking the no-PML pair of an absorbing run (or the reverse) would admit a
    configuration whose drive is the other field — the two agree exactly outside
    the absorber, so every no-PML cell would look right. The fold splits the two
    absorbing arms the same way: ``folded_dispersive_constitutive_coverage``
    REQUIRES a real fold and ``dispersive_constitutive_coverage`` refuses one.
    """
    fields, pml = engine(arm)
    admitting = [name for name in module.ARMS if not reasons(fields, pml, name)]
    assert admitting == [arm], admitting


def test_the_folded_arm_is_the_dispersive_body_under_a_different_admission():
    """No third kernel: the fold changes the predicate and nothing else."""
    assert set(module.PML_ARMS) == {"dispersive", "folded"}
    fields, pml = engine("folded")
    plan = plan_for(fields, pml, "folded")
    assert plan._pml_flag == 1
    unfolded, unfolded_pml = engine("dispersive")
    twin = plan_for(unfolded, unfolded_pml, "dispersive")
    assert plan._pml_flag == twin._pml_flag
    assert plan._sv.keys() == twin._sv.keys()
    # ...and the pole ORDER comes from the unfolded module, which is where
    # folded_dispersive_update_e's own builder takes it from too.
    assert module._E_HALF["folded"][1] is dispersive_module


@pytest.mark.parametrize("arm", ("dispersive", "folded"))
def test_the_pml_arms_name_the_drive_field_identity_themselves(arm):
    """Not inherited: the ADE predicate only asks that ``f_w`` EXISTS.

    The hazard this clause exists for is ``drive_field`` handing back the STORED E
    under an absorber — "the single most likely silent wrong answer in dispersion"
    (fields.py:1152-1155), which agrees everywhere except inside the PML. The
    register this kernel passes the recurrence is what it stored to ``f_w``, so a
    ``Fields`` whose reader disagrees is refused BY NAME rather than admitted.
    """
    fields, pml = engine(arm)
    assert reasons(fields, pml, arm) == []
    fields.drive_field = lambda component: getattr(fields, component)
    verdict = reasons(fields, pml, arm)
    assert any("is not f_w_Ey" in reason for reason in verdict), verdict
    assert len([r for r in verdict if "is not f_w_" in r]) == 3


@pytest.mark.parametrize("arm", module.ARMS)
def test_a_run_with_no_susceptibility_is_refused_rather_than_spanning_one_pass(arm):
    fields, pml = engine(arm, poles=0)
    verdict = reasons(fields, pml, arm)
    assert any("no susceptibility is registered" in reason for reason in verdict)


@pytest.mark.parametrize(
    ("keywords", "needle"),
    (
        ({"complex_storage": True}, "complex64"),
        ({"offdiag": True}, "off-diagonal"),
        ({"nonlinear": True}, "chi2/chi3"),
    ),
)
def test_the_products_this_family_does_not_write_are_refused_by_name(keywords, needle):
    fields, pml = engine("no_pml", **keywords)
    verdict = reasons(fields, pml, "no_pml")
    assert any(needle in reason for reason in verdict), verdict


# ---------------------------------------------------------------------------
# The shape: one scratch per driven component
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("arm", "poles", "sigma", "driven"),
    (
        ("dispersive", 6, None, 3),                                  # the worst row
        ("no_pml", 5, None, 3),
        ("no_pml", 2, None, 3),
        ("dispersive", 2, {"Ex": 0.4, "Ey": 0.0, "Ez": 0.25}, 2),
        ("no_pml", 1, {"Ex": 0.0, "Ey": 0.0, "Ez": 0.25}, 1),
    ),
)
def test_the_extra_scratch_cost_is_K_times_d_minus_one(arm, poles, sigma, driven):
    """LEG 3 of the shape probe, re-measured through the plan that pays it."""
    fields, pml = engine(arm, poles=poles, sigma=sigma)
    plan = plan_for(fields, pml, arm)
    assert len(fields.polarizations[0].driven()) == driven
    assert plan.extra_scratch_volumes == poles * (driven - 1)
    # The FIRST driven component reuses the susceptibility's own scratch, which is
    # what makes the extra K*(d-1) rather than K*d.
    for index, state in enumerate(fields.polarizations):
        first = plan._first_driven[index]
        assert plan._scratch[(index, first)] is state._scratch


@pytest.mark.parametrize("arm", module.ARMS)
@pytest.mark.parametrize("poles", (1, 2, 6))
def test_the_rotation_orbit_is_alias_free_at_every_position(arm, poles):
    """The shape probe's LEG 2, walked THROUGH THE SHIPPED PLAN to closure.

    The rotation is a permutation of a finite buffer set, so its orbit closes; nine
    positions is three full turns of the three-cycle each (susceptibility,
    component) walks. At every one of them the launch's write set must be disjoint
    from its read set — that is this shape's entire licence over
    ``fused_ade_state``'s shared scratch, which conflicts at EVERY position for
    d > 1.
    """
    fields, pml = engine(arm, poles=poles)
    plan = plan_for(fields, pml, arm)
    starts = []
    for _position in range(9):
        groups = plan._poles.arrays()
        chain = plan._resolve(groups)
        plan._check_aliasing(chain)                       # raises if it aliases
        outputs = [row[2] for entries in chain for row in entries]
        inputs = [row[3] for entries in chain for row in entries]
        inputs += [row[4] for entries in chain for row in entries]
        assert len({id(value) for value in outputs}) == len(outputs)
        assert not ({id(value) for value in outputs} & {id(value) for value in inputs})
        starts.append(tuple(id(state.P[name]) for state in fields.polarizations
                            for name in state.driven()))
        plan._rotate(chain)
    assert plan.alias_checks == 9
    # ORBIT 3, not 2d + 1: the permutation decomposes into d independent 3-cycles.
    assert starts[0] == starts[3] == starts[6]
    assert starts[0] != starts[1] != starts[2]


def test_the_alias_check_fires_when_the_chain_is_broken():
    """A check never demonstrated to fire is decoration."""
    fields, pml = engine("dispersive", poles=2)
    plan = plan_for(fields, pml, "dispersive")
    groups = plan._poles.arrays()
    chain = plan._resolve(groups)
    plan._check_aliasing(chain)

    # Two arms on one scratch: one recurrence would overwrite the other's result.
    shared = list(chain[0][0])
    collided = [list(entries) for entries in chain]
    collided[0][1] = tuple(list(chain[0][1])[:2] + [shared[2]]
                           + list(chain[0][1])[3:])
    with pytest.raises(RuntimeError, match="SAME output buffer"):
        plan._check_aliasing(collided)

    # An output that is also an input: the alias fused_ade_state cannot avoid.
    aliased = [list(entries) for entries in chain]
    row = list(chain[0][0])
    row[2] = row[3]
    aliased[0][0] = tuple(row)
    with pytest.raises(RuntimeError, match="also read by this launch"):
        plan._check_aliasing(aliased)


def test_the_rotation_is_the_references_three_cycle():
    """``dispersion.py:687-691`` per (susceptibility, component), not per state."""
    fields, pml = engine("no_pml", poles=1)
    state = fields.polarizations[0]
    plan = plan_for(fields, pml, "no_pml")
    before = {name: (state.P[name], state.P_prev[name],
                     plan._scratch[(0, name)]) for name in state.driven()}
    chain = plan._resolve(plan._poles.arrays())
    plan._rotate(chain)
    for name in state.driven():
        pole, previous, scratch = before[name]
        assert state.P[name] is scratch          # this step's result
        assert state.P_prev[name] is pole        # the history shifts down
        assert plan._scratch[(0, name)] is previous   # the retired buffer


def test_the_susceptibilitys_own_scratch_is_never_left_aliasing_a_live_slot():
    fields, pml = engine("dispersive", poles=2)
    plan = plan_for(fields, pml, "dispersive")
    for _position in range(4):
        chain = plan._resolve(plan._poles.arrays())
        plan._rotate(chain)
        for state in fields.polarizations:
            live = {id(state.P[name]) for name in state.driven()}
            live |= {id(state.P_prev[name]) for name in state.driven()}
            assert id(state._scratch) not in live


# ---------------------------------------------------------------------------
# The launcher's argument order
# ---------------------------------------------------------------------------

def test_the_declared_runtime_argument_count_is_the_kernels_own():
    tree = ast.parse(SOURCE)
    node = next(child for child in ast.walk(tree)
                if isinstance(child, ast.FunctionDef)
                and child.name == "fused_ade_chain_step")
    runtime = [a.arg for a in node.args.args if a.annotation is None]
    constexpr = [a.arg for a in node.args.args if a.annotation is not None]
    assert len(runtime) == module.RUNTIME_ARGUMENTS
    assert len(set(a.arg for a in node.args.args)) == len(node.args.args)
    # NP0..2, PML, one SIGMA flag per live slot per component, BLOCK.
    assert len(constexpr) == 4 + 3 * module.CHAIN_MAX_POLES + 1


def test_the_assembled_arguments_land_on_the_slots_they_are_named_for():
    """Positional binding, checked position by position against the signature."""
    fields, pml = engine("dispersive", poles=2,
                         sigma={"Ex": np.full(
                             Grid(resolution=10.0, cell_size=(0.8, 0.8, 0.8),
                                  xp=np).shape, 0.3, dtype=np.float32),
                             "Ey": 0.2, "Ez": 0.25})
    plan = plan_for(fields, pml, "dispersive")
    groups = plan._poles.arrays()
    chain = plan._resolve(groups)
    arguments = plan._arguments(plan._pole_slots(groups), chain)
    assert len(arguments) == module.RUNTIME_ARGUMENTS

    tree = ast.parse(SOURCE)
    node = next(child for child in ast.walk(tree)
                if isinstance(child, ast.FunctionDef)
                and child.name == "fused_ade_chain_step")
    names = [a.arg for a in node.args.args if a.annotation is None]
    slot = dict(zip(names, arguments))

    def held(name):
        value = slot[name]
        return getattr(value, "array", value)

    for index, (component, displacement, _axis) in enumerate(module.E_TERMS):
        assert held(f"f{index}") is getattr(fields, component)
        assert held(f"w{index}") is getattr(fields, "f_w_" + component)
        assert held(f"g{index}") is getattr(fields, displacement)
        assert held(f"e{index}") is fields.inverse_epsilon_for(component)
    order = dispersive_module.poles_per_component(fields)
    for letter, (component, displacement, _axis) in zip(("a", "b", "c"),
                                                        module.E_TERMS):
        states = order[component]
        for position in range(module.MAX_POLES):
            expected = (states[position].P[component] if position < len(states)
                        else getattr(fields, displacement))
            assert held(f"{letter}{position}") is expected
        for position in range(module.CHAIN_MAX_POLES):
            if position >= len(states):
                assert held(f"p_out_{letter}{position}") is getattr(
                    fields, displacement)
                assert slot[f"cnow_{letter}{position}"] == 0.0
                continue
            state = states[position]
            assert held(f"p_out_{letter}{position}") is plan._scratch[
                (position, component)]
            assert held(f"p_prev_{letter}{position}") is state.P_prev[component]
            c_now, c_prev, c_drive = state._coefficients
            assert slot[f"cnow_{letter}{position}"] == float(c_now)
            assert slot[f"cprev_{letter}{position}"] == float(c_prev)
            assert slot[f"cdrive_{letter}{position}"] == float(c_drive)
            sigma = state.sigma[component]
            if getattr(sigma, "shape", ()):
                assert held(f"sigma_{letter}{position}") is sigma
                assert plan._sv[f"SV_{letter}{position}"] == 1
            else:
                assert slot[f"sigma_{letter}{position}"] == float(sigma)
                assert plan._sv[f"SV_{letter}{position}"] == 0
    assert (arguments[-4], arguments[-3], arguments[-2]) == tuple(plan.shape)
    assert arguments[-1] == plan.n_elem


def test_a_signature_the_launcher_does_not_follow_fails_at_the_call(monkeypatch):
    fields, pml = engine("no_pml", poles=1)
    plan = plan_for(fields, pml, "no_pml")
    groups = plan._poles.arrays()
    chain = plan._resolve(groups)
    monkeypatch.setattr(module, "RUNTIME_ARGUMENTS", module.RUNTIME_ARGUMENTS + 1)
    with pytest.raises(RuntimeError, match="runtime arguments"):
        plan._arguments(plan._pole_slots(groups), chain)


# ---------------------------------------------------------------------------
# The transcription — parsed out of both files, never re-implemented
# ---------------------------------------------------------------------------

def test_the_pole_chain_is_the_certified_bodys_chain_link_for_link():
    fused = kernel_text()
    for name, path, letters in (
            ("stored_e_constitutive_step",
             pathlib.Path(stored_e_module.__file__), ("a", "b", "c")),
            ("constitutive_step_dispersive",
             pathlib.Path(dispersive_module.__file__), ("a", "b", "c"))):
        certified = _normalised(_function_text(path, name))
        for index, letter in enumerate(letters):
            for slot in range(module.MAX_POLES):
                guard = f"if NP{index} > {slot}:"
                load = (f"{letter}{slot} + idx, mask=live, other=0.0)")
                assert guard in _normalised(certified), (name, guard)
                assert _normalised(
                    f"s{index} = s{index} - tl.load({load}") in certified or \
                    _normalised(f"tl.load({load}") in certified, (name, load)
                assert guard in _normalised(fused), guard
                assert _normalised(
                    f"p{letter}{slot} = tl.load({load}") in _normalised(fused)
                assert _normalised(
                    f"s{index} = s{index} - p{letter}{slot}") in _normalised(fused)


def test_the_constitutive_product_is_the_dispersive_bodys_own_line():
    """``src{c} = s{c} * tl.load(e{c} ...)`` is verbatim; the no-PML store is split."""
    certified = _normalised(_function_text(
        pathlib.Path(dispersive_module.__file__), "constitutive_step_dispersive"))
    stored = _normalised(_function_text(
        pathlib.Path(stored_e_module.__file__), "stored_e_constitutive_step"))
    fused = _normalised(kernel_text())
    for index in range(3):
        line = (f"src{index} = s{index} * tl.load(e{index} + idx, mask=live, "
                f"other=0.0)")
        assert line in certified, line
        assert line in fused, line
        # The certified no-PML body stores the product UNNAMED; this is the one
        # line the fusion rewrites, and the rewrite is stated rather than hidden.
        unnamed = (f"tl.store(f{index} + idx, s{index} * tl.load(e{index} + idx, "
                   f"mask=live, other=0.0), mask=live)")
        assert unnamed in stored, unnamed
        assert unnamed not in fused
        assert f"tl.store(f{index} + idx, src{index}, mask=live)" in fused


def test_the_pml_tail_is_the_dispersive_bodys_own_lines():
    certified = _normalised(_function_text(
        pathlib.Path(dispersive_module.__file__), "constitutive_step_dispersive"))
    fused = _normalised(kernel_text())
    for index in range(3):
        for line in (
                f"prev{index} = tl.load(w{index} + idx, mask=live, other=0.0)",
                f"tl.store(w{index} + idx, src{index}, mask=live)",
                f"v{index} = tl.load(f{index} + idx, mask=live, other=0.0)",
                f"v{index} = v{index} + kp_{index} * src{index}",
                f"v{index} = v{index} - km_{index} * prev{index}",
                f"tl.store(f{index} + idx, v{index}, mask=live)"):
            assert line in certified, line
            assert line in fused, line
    for line in ("kp_0 = tl.load(kp0 + i, mask=live, other=0.0)",
                 "km_2 = tl.load(km2 + k, mask=live, other=0.0)",
                 "k = idx % nz", "plane = idx // nz", "j = plane % ny",
                 "i = plane // ny"):
        assert line in certified, line
        assert line in fused, line


def test_the_recurrence_is_spelled_exactly_as_ade_update_p_spells_it():
    """The ONE rename is the destination; the arithmetic is character for character."""
    certified = _normalised(_function_text(
        PACKAGE_DIR / "kernels.py", "ade_update_p"))
    fused = _normalised(kernel_text())
    expression = "((p * c_now) + (c_prev * q)) + (c_drive * (s * w))"
    assert f"tl.store(p_out + idx, {expression}, mask=live)" in certified
    assert "q = tl.load(p_prev + idx, mask=live, other=0.0)" in certified
    assert "s = tl.load(sigma + idx, mask=live, other=0.0)" in certified
    assert "p = tl.load(p_now + idx, mask=live, other=0.0)" in certified
    assert "w = tl.load(drive + idx, mask=live, other=0.0)" in certified
    arms = 0
    for index, letter in enumerate(("a", "b", "c")):
        for slot in range(module.CHAIN_MAX_POLES):
            assert (f"tl.store(p_out_{letter}{slot} + idx, {expression}, "
                    f"mask=live)") in fused
            assert (f"q = tl.load(p_prev_{letter}{slot} + idx, mask=live, "
                    f"other=0.0)") in fused
            assert (f"s = tl.load(sigma_{letter}{slot} + idx, mask=live, "
                    f"other=0.0)") in fused
            assert f"s = sigma_{letter}{slot}" in fused
            # THE TWO FUSIONS, and they are the only substitutions:
            assert f"p = p{letter}{slot}" in fused     # p_now[idx] -> the register
            assert f"w = src{index}" in fused          # drive[idx] -> the register
            assert f"c_now = cnow_{letter}{slot}" in fused
            assert f"c_prev = cprev_{letter}{slot}" in fused
            assert f"c_drive = cdrive_{letter}{slot}" in fused
            arms += 1
    assert arms == 3 * module.CHAIN_MAX_POLES
    # ...and the drive is NEVER re-loaded from a buffer.
    assert "tl.load(drive" not in fused
    assert "w = tl.load(" not in fused


def test_every_ade_arm_is_guarded_by_its_own_subtraction_slot():
    """A live arm whose pole the E chain did not subtract reuses a stale register."""
    tree = ast.parse(kernel_text())
    guards: dict = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test = ast.unparse(node.test)
        for statement in node.body:
            for child in ast.walk(statement):
                if not isinstance(child, ast.Call):
                    continue
                if getattr(child.func, "attr", None) != "store":
                    continue
                target = ast.unparse(child.args[0])
                guards.setdefault(target, set()).add(test)
    for index, letter in enumerate(("a", "b", "c")):
        for slot in range(module.CHAIN_MAX_POLES):
            target = f"p_out_{letter}{slot} + idx"
            assert guards.get(target) == {f"NP{index} > {slot}"}, target


def test_the_slot_counts_and_terms_are_pinned_to_the_certified_bodies():
    assert module.MAX_POLES == stored_e_module.MAX_POLES
    assert module.MAX_POLES == dispersive_module.MAX_POLES
    assert module.E_TERMS == dispersive_module.E_TERMS
    assert tuple(term[:2] for term in module.E_TERMS) == stored_e_module.E_TERMS
    fused = kernel_text()
    for index in range(3):
        links = len(re.findall(rf"if NP{index} > \d+:", fused))
        # MAX_POLES subtraction links plus CHAIN_MAX_POLES ADE arms.
        assert links == module.MAX_POLES + module.CHAIN_MAX_POLES


# ---------------------------------------------------------------------------
# The seam, the policy and the wiring
# ---------------------------------------------------------------------------

def test_the_seam_carries_no_driver_pass_and_the_predicate_takes_no_sources():
    """Measured from ``driver.py``, not asserted: nothing sits between the two."""
    driver = (API_ROOT / "meep_gpu" / "driver.py").read_text(encoding="utf-8")
    match = re.search(
        r'if fast is None or not fast\.dispatch\("update_E", self\.fields\):\n'
        r"\s+update_E\(self\.fields, self\.pml\)\n"
        r'\s+if fast is None or not fast\.dispatch\("update_P", self\.fields\):\n'
        r"\s+update_P\(self\.fields, self\.pml\)\n", driver)
    assert match, "the E->P seam is no longer two adjacent dispatch blocks"
    signature = inspect.signature(module.fused_ade_chain_coverage)
    assert "sources" not in signature.parameters
    assert list(module.REPLACES) == ["update_E", "update_P"]


def test_the_shipped_policy_is_stamped_and_is_one_warp():
    assert module.POLICY["num_warps"] == 1
    assert module.POLICY["status"] == "keep"
    signature = inspect.signature(module.plan_fused_ade_chain)
    assert signature.parameters["num_warps"].default == 1
    assert module.FusedAdeChainPlan.launches_per_run == 1
    assert module.FusedAdeChainPlan.replaces_sub_steps == module.REPLACES


def test_composition_into_plan_step_is_deferred_and_dispatch_stays_disabled():
    launch = (PACKAGE_DIR / "launch.py").read_text(encoding="utf-8")
    assert "fused_ade_chain" not in launch
    init = (PACKAGE_DIR / "__init__.py").read_text(encoding="utf-8")
    assert "fused_ade_chain" not in init


def test_the_device_gate_is_present_and_names_this_module():
    assert GATE.exists(), GATE
    text = GATE.read_text(encoding="utf-8")
    assert "fused_ade_chain" in text
    assert "gate_provenance" in text
