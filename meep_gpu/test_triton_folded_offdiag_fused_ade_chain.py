"""Laptop contracts for the folded off-diagonal dispersive E->P chain:
the fold + tensor-epsilon + dispersion ``update_E`` welded to ``update_P``.

The device gate owns byte identity — nothing here compiles or launches anything.
These tests own the five things that decide whether the device run measures what
it claims:

* **the seam's admission clauses**, and the refusals that must stay refusals — a
  third pole, complex storage, no fold, no row, no pole, an inert absorber, the
  no-PML storage mode, chi3, and the drive/driven/extent breaks the seam names;
* **disjointness both ways.** ``fused_ade_chain`` refuses this cell on all three
  of its arms and ``complex_fused_ade_chain`` refuses it by storage, so the board's
  at-most-one-admitter assertion stays at zero clashes;
* **the shape**, walked on real engine objects. One scratch per driven component
  is this product's whole licence, and the orbit is re-asked through the SHIPPED
  plan with its alias check armed — the row volumes counted among the reads;
* **the transcription.** The certified E body must be an AST-identical PREFIX of
  this kernel's body, both halves of the certified signature prefixes of the
  fused lists, and each ADE arm ``ade_update_p``'s own lines modulo the six
  declared edits — PARSED out of the files, never re-implemented;
* **the launcher's argument ORDER.** 91 runtime arguments reach the kernel
  positionally, and a launcher that assembles the right pointers in the wrong
  order binds every later one to a neighbouring slot — a wrong answer, not a
  crash.

WHY ``CHAIN_MAX_POLES``' ADJUDICATION LIVES HERE. Kernel modules are bound by the
gate's curated digests once their device gate lands, and a prose edit there makes
a standing board unreproducible until the gate is re-run. So the cap's reasoning
is pinned in tests, which nothing binds, and a test cannot go stale in silence
the way a comment can.
"""

from __future__ import annotations

import ast
import inspect
import pathlib
import re

import numpy as np
import pytest

from meep_gpu.dispersion import PolarizationState, Susceptibility
from meep_gpu.fields import Fields
from meep_gpu.grid import Grid, Mirror
from meep_gpu.pml import PML
from meep_gpu.triton_kernels import complex_fused_ade_chain as complex_chain
from meep_gpu.triton_kernels import dispersive_update_e as dispersive_module
from meep_gpu.triton_kernels import folded_offdiag_dispersive_update_e as fod_module
from meep_gpu.triton_kernels import folded_offdiag_fused_ade_chain as module
from meep_gpu.triton_kernels import folded_offdiag_update_e as folded_module
from meep_gpu.triton_kernels import fused_ade_chain as chain
from meep_gpu.triton_kernels import offdiag_update_e as offdiag_module

PACKAGE_DIR = pathlib.Path(module.__file__).parent
API_ROOT = PACKAGE_DIR.parents[1]
PARITY = API_ROOT / "parity" / "meep_gpu"
GATE = PARITY / "gate_triton_folded_offdiag_fused_ade_chain.py"
BOARD = PARITY / "build_triton_fusion_matrix.py"
REACHABILITY = PARITY / "dispatch_reachability.py"
PLANNER_TEST = API_ROOT / "meep_gpu" / "test_triton_planner_composition.py"
SOURCE = pathlib.Path(module.__file__).read_text(encoding="utf-8")

COMPONENTS = ("Ex", "Ey", "Ez")
AUXILIARIES = tuple("f_w_" + name for name in COMPONENTS)
DISPLACEMENTS = ("Dx", "Dy", "Dz")

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def engine(fold: str = "Y", phase: int = +1, poles: int = 1, sigma=None,
           rows: str = "all", pml: bool = True, complex_storage: bool = False,
           nonlinear: bool = False, offdiag: bool = True, boundaries=None,
           storage: str = "pml", seed: int = 41):
    """One configuration on the laptop's NumPy backend.

    THE DEFAULT IS THE CELL: a mirror plane, distinct per-component inverse
    epsilon volumes with every off-diagonal row surviving, one Lorentz pole
    driving all three components, PML storage and an active absorber that takes
    the HIGH face only on the folded axis (cell 0 lies on the mirror plane, which
    is a boundary condition and not an absorber; pml.py refuses the low face by
    name). ``sigma`` may be a scalar, a per-component dict, ``"volume"`` (a full
    grid volume on every component) or ``"mixed"`` (a volume on Ex beside scalars).
    """
    grid = Grid(resolution=10.0, cell_size=(1.3, 1.6, 0.9), dimensions=3,
                courant=0.35,
                symmetry=(Mirror(fold, phase),) if fold else (),
                boundaries=boundaries, xp=np)
    fields = Fields(grid=grid, force_complex_fields=complex_storage)
    if storage == "pml":
        fields.enable_pml_storage()
    else:
        fields.enable_field_storage()
    shape = grid.shape
    rng = np.random.default_rng(seed)
    diagonal = {name: np.full(shape, value, np.float32)
                for name, value in zip(COMPONENTS, (2.0, 2.5, 3.0))}
    inverse = {name: np.full(shape, 1.0 / value, np.float32)
               for name, value in zip(COMPONENTS, (2.0, 2.5, 3.0))}
    installed = None
    if offdiag:
        if rows == "all":
            installed = {
                row: {partner: rng.uniform(-0.2, 0.2, shape).astype(np.float32)
                      for partner in COMPONENTS if partner != row}
                for row in COMPONENTS}
        else:
            installed = {"Ex": {"Ey": rng.uniform(-0.2, 0.2, shape)
                                .astype(np.float32)}}
    fields.set_epsilon_volumes(diagonal, inverse, chi1inv_offdiagonal=installed)

    if isinstance(sigma, str) and sigma == "volume":
        sigma = np.full(shape, 0.3, np.float32)
    elif isinstance(sigma, str) and sigma == "mixed":
        sigma = {"Ex": np.full(shape, 0.3, np.float32), "Ey": 0.2, "Ez": 0.25}
    dtype = np.complex64 if complex_storage else np.float32
    for index in range(poles):
        fields.polarizations.append(PolarizationState(
            Susceptibility(0.8 + 0.1 * index, 0.1, "lorentzian"),
            0.4 if sigma is None else sigma, grid, dtype))

    for name in COMPONENTS + AUXILIARIES + DISPLACEMENTS:
        array = getattr(fields, name)
        if array is None:
            continue  # the no-PML storage mode allocates no f_w
        array[...] = rng.uniform(-0.4, 0.4, shape).astype(dtype)
    for state in fields.polarizations:
        for slot in ("P", "P_prev"):
            for array in getattr(state, slot).values():
                array[...] = rng.uniform(-0.15, 0.15, shape).astype(array.dtype)
    if nonlinear:
        fields.set_nonlinear_volumes({"Ez": 0.04}, {"Ez": 0.02})

    if not pml:
        return fields, PML(grid=grid, thickness=0)
    thickness = tuple({"high": 2} if grid.is_mirrored(axis) else 2
                      for axis in range(3))
    return fields, PML(grid=grid, thickness=dict(zip("xyz", thickness)))


def reasons(fields, pml):
    """Refusals other than the laptop's own array module."""
    return [reason for reason
            in module.folded_offdiag_fused_ade_chain_coverage(fields, pml).reasons
            if "not cupy" not in reason]


class FakePointer:
    """Stands in for ``CupyPointer`` on this plan's own ADE bindings."""

    __slots__ = ("array",)

    def __init__(self, array):
        self.array = array


def plan_for(fields, pml, block: int = 256):
    return module.FoldedOffdiagFusedAdeChainPlan(
        fields, pml, block, num_warps=1, pointer=FakePointer)


def _function_node(path: pathlib.Path, name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return next(child for child in ast.walk(tree)
                if isinstance(child, ast.FunctionDef) and child.name == name)


def _function_text(path: pathlib.Path, name: str) -> str:
    source = path.read_text(encoding="utf-8")
    node = _function_node(path, name)
    segment = ast.get_source_segment(source, node)
    assert segment, f"{name} has no source segment in {path}"
    return segment


def _normalised(text: str) -> str:
    """Collapse runs of whitespace so a re-wrapped line still matches."""
    return re.sub(r"\s+", " ", text)


def kernel_text() -> str:
    """The shipped kernel's own EXECUTABLE source, read from the FILE.

    Read rather than imported: these tests run on a host with no Triton, where the
    kernel object is ``None``, so the transcription check has to bite at the merge
    bar and not only on a device run. The signature and the body are what is
    checked; the docstring is dropped, because it NAMES the loads this kernel
    eliminates and a needle test must not be satisfied by prose.
    """
    path = pathlib.Path(module.__file__)
    source = path.read_text(encoding="utf-8")
    node = _function_node(path, module.KERNEL_NAME)
    lines = source.splitlines(keepends=True)
    head = "".join(lines[node.lineno - 1:node.body[0].lineno - 1])
    body = "".join(lines[node.body[1].lineno - 1:node.end_lineno])
    assert ast.get_docstring(node), "the kernel carries a docstring"
    return head + body


def kernel_node() -> ast.FunctionDef:
    return _function_node(pathlib.Path(module.__file__), module.KERNEL_NAME)


def certified_node() -> ast.FunctionDef:
    return _function_node(pathlib.Path(fod_module.__file__),
                          "folded_offdiag_dispersive_constitutive_step")


def _statements(node: ast.FunctionDef) -> list:
    """A function's body with its docstring skipped."""
    return node.body[1:] if ast.get_docstring(node) else list(node.body)


def _runtime_names(node: ast.FunctionDef) -> list:
    return [a.arg for a in node.args.args if a.annotation is None]


def _constexpr_names(node: ast.FunctionDef) -> list:
    return [a.arg for a in node.args.args if a.annotation is not None]


def _assigned_literal(path: pathlib.Path, name: str):
    """A module-scope literal assignment, read off the file without importing it."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        target = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
        elif isinstance(node, ast.AnnAssign):
            target = node.target
        if isinstance(target, ast.Name) and target.id == name:
            return ast.literal_eval(node.value)
    raise AssertionError(f"{path.name} assigns no literal {name}")


# ---------------------------------------------------------------------------
# Admission and refusal
# ---------------------------------------------------------------------------

def test_the_module_imports_without_triton_and_refuses_before_any_launch():
    fields, pml = engine()
    verdict = module.folded_offdiag_fused_ade_chain_coverage(fields, pml)
    assert not verdict.covered                       # the laptop has no CuPy
    assert any("not cupy" in reason for reason in verdict.reasons)
    assert reasons(fields, pml) == []
    assert module.plan_folded_offdiag_fused_ade_chain(fields, pml) is None
    assert isinstance(module.explain_folded_offdiag_fused_ade_chain(
        fields, pml).covered, bool)
    if module.folded_offdiag_fused_ade_chain_step is None:
        with pytest.raises(ImportError, match="triton"):
            module.folded_offdiag_fused_ade_chain_kernel()


@pytest.mark.parametrize("rows", ("one", "all"))
@pytest.mark.parametrize("phase", (+1, -1))
@pytest.mark.parametrize("fold", ("X", "Y"))
@pytest.mark.parametrize("sigma", (
    0.4,
    {"Ex": 0.4, "Ey": 0.0, "Ez": 0.25},
    "volume",
))
@pytest.mark.parametrize("poles", (1, 2))
def test_the_covered_shapes_leave_no_refusal_but_the_array_module(
        poles, sigma, fold, phase, rows):
    fields, pml = engine(fold=fold, phase=phase, poles=poles, sigma=sigma,
                         rows=rows)
    assert reasons(fields, pml) == []


def test_a_third_pole_is_refused_by_name_and_never_truncated():
    """A pole the E chain subtracts but no ADE arm advances is a frozen P."""
    fields, pml = engine(poles=module.CHAIN_MAX_POLES + 1)
    verdict = reasons(fields, pml)
    assert verdict, "three poles must not be admitted"
    assert all(f"is driven by {module.CHAIN_MAX_POLES + 1} poles" in reason
               for reason in verdict)
    assert any("frozen P" in reason for reason in verdict)
    assert module.plan_folded_offdiag_fused_ade_chain(fields, pml) is None
    # ...and the PLAN refuses it too, so a caller bypassing the predicate cannot
    # build a chain whose third pole is silently unadvanced.
    with pytest.raises(ValueError, match="ADE arms per component"):
        plan_for(fields, pml)


def test_the_cap_is_GUARDED_and_not_dead_code(monkeypatch):
    """Two poles admit; the SAME configuration is refused with the cap at one."""
    fields, pml = engine(poles=2)
    assert reasons(fields, pml) == []
    monkeypatch.setattr(module, "CHAIN_MAX_POLES", 1)
    assert any("is driven by 2 poles" in reason for reason in reasons(fields, pml))
    with pytest.raises(ValueError, match="ADE arms per component"):
        plan_for(fields, pml)


def test_the_cap_is_below_the_E_halfs_own_slot_count():
    """The ADE half is the binding bound, and the E chain has room to spare."""
    assert module.CHAIN_MAX_POLES < module.MAX_POLES
    assert module.CHAIN_MAX_POLES == 2


def test_fused_ade_chain_refuses_the_cell_on_all_three_arms():
    """The disjointness the board asserts, measured on the cell itself.

    The folded arm refuses by the ROW, the dispersive arm by the FOLD, the no-PML
    arm by the ABSORBER — each a refusal by name from its own E-half predicate, so
    ``assert_at_most_one_e_to_p_product_admits`` keeps zero clashes.
    """
    fields, pml = engine()
    assert reasons(fields, pml) == []
    for arm in chain.ARMS:
        verdict = [reason for reason
                   in chain.fused_ade_chain_coverage(fields, pml, arm).reasons
                   if "not cupy" not in reason]
        assert verdict, arm
        assert any("off-diagonal chi1inv row" in reason for reason in verdict), arm
    dispersive = chain.fused_ade_chain_coverage(fields, pml, "dispersive").reasons
    assert any("mirror plane" in reason for reason in dispersive)
    no_pml = chain.fused_ade_chain_coverage(fields, pml, "no_pml").reasons
    assert any("active PML layer" in reason for reason in no_pml)
    assert any("mirror plane" in reason for reason in no_pml)


def test_the_complex_chain_refuses_the_cell_by_storage():
    fields, pml = engine()
    verdict = complex_chain.complex_fused_ade_chain_coverage(fields, pml).reasons
    assert any("force_complex_fields=False" in reason for reason in verdict), verdict


@pytest.mark.parametrize(
    ("keywords", "needle"),
    (
        ({"fold": ""}, "no mirror plane"),
        ({"offdiag": False}, "no off-diagonal"),
        ({"poles": 0}, "no susceptibility is registered"),
        ({"pml": False}, "no active PML"),
        ({"storage": "field"}, "f_w_Ex is not allocated"),
        ({"complex_storage": True}, "force_complex_fields=True"),
        ({"nonlinear": True}, "chi2/chi3"),
    ),
)
def test_the_E_halfs_refusals_are_inherited_whole_and_prefixed(keywords, needle):
    """Nothing the E half refuses is admitted here, and the reader can tell who said it."""
    fields, pml = engine(**keywords)
    verdict = reasons(fields, pml)
    assert any(reason.startswith("E half: ") and needle in reason
               for reason in verdict), verdict
    assert module.plan_folded_offdiag_fused_ade_chain(fields, pml) is None


def test_a_run_with_nothing_driven_is_refused_rather_than_spanning_one_pass():
    fields, pml = engine(poles=0)
    verdict = reasons(fields, pml)
    assert any("no polarization drives any electric component" in reason
               and "update_E alone" in reason for reason in verdict), verdict


def test_the_no_pml_storage_mode_is_refused_by_the_seam_itself():
    """Not only inherited: the seam names the drive it would have handed over."""
    fields, pml = engine(storage="field")
    verdict = reasons(fields, pml)
    assert any("Fields is not in PML storage mode" in reason
               and "split-field drive" in reason for reason in verdict), verdict


def test_the_ade_halfs_refusals_are_inherited_per_state_and_component():
    fields, pml = engine()
    fields.polarizations[0]._scratch = None
    verdict = reasons(fields, pml)
    assert any(reason.startswith("ADE half 0.Ey: ") and "_scratch" in reason
               for reason in verdict), verdict


def test_the_seam_names_the_drive_field_identity_itself():
    """Not inherited: the ADE predicate only asks that ``f_w`` EXISTS.

    The hazard this clause exists for is ``drive_field`` handing back the STORED E
    under an absorber — the two agree everywhere except inside the PML. The
    register this kernel passes the recurrence is what it stored to ``f_w``, so a
    ``Fields`` whose reader disagrees is refused BY NAME rather than admitted.
    """
    fields, pml = engine()
    assert reasons(fields, pml) == []
    fields.drive_field = lambda component: getattr(fields, component)
    verdict = reasons(fields, pml)
    assert any("is not f_w_Ey" in reason for reason in verdict), verdict
    assert len([r for r in verdict if "is not f_w_" in r]) == 3


def test_a_state_whose_driven_disagrees_with_the_subtraction_is_refused():
    """A pole the E chain subtracts and the ADE half never advances is a frozen P."""
    fields, pml = engine()
    fields.polarizations[0].driven = lambda: ("Ex",)
    verdict = reasons(fields, pml)
    assert any("driven()=('Ex',) disagrees with" in reason
               and "('Ex', 'Ey', 'Ez')" in reason for reason in verdict), verdict


def test_the_dropped_ade_guard_extent_is_named():
    """This kernel emits ONE bounds guard, the E half's; every P must match it."""
    fields, pml = engine()
    state = fields.polarizations[0]
    state.P["Ex"] = state.P["Ex"][:-1]
    verdict = reasons(fields, pml)
    assert any("P[Ex] holds" in reason and "the launch walks" in reason
               and "drops the certified update_P guard" in reason
               for reason in verdict), verdict


# ---------------------------------------------------------------------------
# The shape: one scratch per driven component
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("poles", "sigma", "driven"),
    (
        (1, None, 3),                                   # the corpus row: K=1, d=3
        (1, {"Ex": 0.4, "Ey": 0.0, "Ez": 0.25}, 2),
        (2, None, 3),
        (2, "mixed", 3),
    ),
)
def test_the_extra_scratch_cost_is_K_times_d_minus_one(poles, sigma, driven):
    """LEG 3 of the shape probe, re-measured through the plan that pays it."""
    fields, pml = engine(poles=poles, sigma=sigma)
    plan = plan_for(fields, pml)
    assert len(fields.polarizations[0].driven()) == driven
    assert plan.extra_scratch_volumes == poles * (driven - 1)
    # The FIRST driven component reuses the susceptibility's own scratch, which is
    # what makes the extra K*(d-1) rather than K*d.
    for index, state in enumerate(fields.polarizations):
        first = plan._first_driven[index]
        assert plan._scratch[(index, first)] is state._scratch


@pytest.mark.parametrize("poles", (1, 2))
def test_the_rotation_orbit_is_alias_free_at_every_position(poles):
    """The shape probe's LEG 2, walked THROUGH THE SHIPPED PLAN to closure.

    The rotation is a permutation of a finite buffer set, so its orbit closes; nine
    positions is three full turns of the three-cycle each (susceptibility,
    component) walks. At every one of them the launch's write set must be disjoint
    from its read set — with the row volumes and the neighbour-gathered P counted
    among the reads — which is this shape's entire licence.
    """
    fields, pml = engine(poles=poles)
    plan = plan_for(fields, pml)
    starts = []
    for _position in range(9):
        groups = plan._e._poles.arrays()
        chain_ = plan._resolve(groups)
        plan._check_aliasing(chain_)                     # raises if it aliases
        outputs = [row[2] for entries in chain_ for row in entries]
        inputs = [row[3] for entries in chain_ for row in entries]
        inputs += [row[4] for entries in chain_ for row in entries]
        inputs += [value for value in offdiag_module.row_volumes_for(fields)
                   if value is not None]
        assert len({id(value) for value in outputs}) == len(outputs)
        assert not ({id(value) for value in outputs} & {id(value) for value in inputs})
        starts.append(tuple(id(state.P[name]) for state in fields.polarizations
                            for name in state.driven()))
        plan._rotate(chain_)
    assert plan.alias_checks == 9
    # ORBIT 3, not 2d + 1: the permutation decomposes into d independent 3-cycles.
    assert starts[0] == starts[3] == starts[6]
    assert starts[0] != starts[1] != starts[2]


def test_the_alias_check_fires_when_the_chain_is_broken():
    """A check never demonstrated to fire is decoration."""
    fields, pml = engine(poles=2)
    plan = plan_for(fields, pml)
    groups = plan._e._poles.arrays()
    chain_ = plan._resolve(groups)
    plan._check_aliasing(chain_)

    # Two arms on one scratch: one recurrence would overwrite the other's result.
    shared = list(chain_[0][0])
    collided = [list(entries) for entries in chain_]
    collided[0][1] = tuple(list(chain_[0][1])[:2] + [shared[2]]
                           + list(chain_[0][1])[3:])
    with pytest.raises(RuntimeError, match="SAME output buffer"):
        plan._check_aliasing(collided)

    # An output that is also an input: the alias fused_ade_state cannot avoid.
    aliased = [list(entries) for entries in chain_]
    row = list(chain_[0][0])
    row[2] = row[3]
    aliased[0][0] = tuple(row)
    with pytest.raises(RuntimeError, match="also read by this launch"):
        plan._check_aliasing(aliased)

    # A scratch that IS a row volume: the E half gathers it at neighbour offsets
    # while the outputs are written — the addition over the chain's inventory.
    rowed = [list(entries) for entries in chain_]
    row = list(chain_[0][0])
    row[2] = fields.chi1inv_offdiagonal_for("Ex")["Ey"]
    rowed[0][0] = tuple(row)
    with pytest.raises(RuntimeError, match="also read by this launch"):
        plan._check_aliasing(rowed)


def test_the_rotation_is_the_references_three_cycle():
    """``dispersion.py:687-691`` per (susceptibility, component), not per state."""
    fields, pml = engine(poles=1)
    state = fields.polarizations[0]
    plan = plan_for(fields, pml)
    before = {name: (state.P[name], state.P_prev[name],
                     plan._scratch[(0, name)]) for name in state.driven()}
    chain_ = plan._resolve(plan._e._poles.arrays())
    plan._rotate(chain_)
    for name in state.driven():
        pole, previous, scratch = before[name]
        assert state.P[name] is scratch          # this step's result
        assert state.P_prev[name] is pole        # the history shifts down
        assert plan._scratch[(0, name)] is previous   # the retired buffer


def test_the_susceptibilitys_own_scratch_is_never_left_aliasing_a_live_slot():
    fields, pml = engine(poles=2)
    plan = plan_for(fields, pml)
    for _position in range(4):
        chain_ = plan._resolve(plan._e._poles.arrays())
        plan._rotate(chain_)
        for state in fields.polarizations:
            live = {id(state.P[name]) for name in state.driven()}
            live |= {id(state.P_prev[name]) for name in state.driven()}
            assert id(state._scratch) not in live


def test_the_plan_refuses_a_chain_whose_slots_disagree_with_the_E_half():
    """The pointer the ADE arm re-reads must be the subtraction slot's array."""
    fields, pml = engine(poles=2)
    plan = plan_for(fields, pml)
    groups = plan._e._poles.arrays()
    reversed_groups = tuple(tuple(reversed(group)) for group in groups)
    with pytest.raises(RuntimeError, match="not the array the subtraction slot"):
        plan._resolve(reversed_groups)
    # ...and the live binding refuses an order that changed under the plan.
    fields.polarizations.reverse()
    with pytest.raises(RuntimeError, match="no longer valid"):
        plan._e._poles.arrays()


def test_the_plan_composes_the_certified_E_plan_rather_than_subclassing_it():
    fields, pml = engine()
    plan = plan_for(fields, pml)
    assert isinstance(plan._e, fod_module.FoldedOffdiagDispersiveConstitutivePlan)
    assert not issubclass(module.FoldedOffdiagFusedAdeChainPlan,
                          fod_module.FoldedOffdiagDispersiveConstitutivePlan)
    assert plan.counts == plan._e.counts
    assert plan._e.row_mask == (1,) * 6
    assert plan._e._base.block == plan.block


# ---------------------------------------------------------------------------
# The launcher's argument order
# ---------------------------------------------------------------------------

def test_the_declared_runtime_argument_count_is_the_kernels_own():
    node = kernel_node()
    runtime = _runtime_names(node)
    constexpr = _constexpr_names(node)
    assert len(runtime) == module.RUNTIME_ARGUMENTS == 91
    assert len(set(a.arg for a in node.args.args)) == len(node.args.args)
    # The certified nineteen (NP0..2, six row flags, three boundary codes, three
    # ghost flags, three wall flags, BLOCK) plus one SIGMA flag per live slot.
    assert len(constexpr) == 19 + 3 * module.CHAIN_MAX_POLES == 25


def test_the_assembled_arguments_land_on_the_slots_they_are_named_for():
    """Positional binding, checked position by position against the signature."""
    fields, pml = engine(poles=2, sigma="mixed", rows="one")
    plan = plan_for(fields, pml)
    groups = plan._e._poles.arrays()
    chain_ = plan._resolve(groups)
    arguments = plan._arguments(chain_)
    assert len(arguments) == module.RUNTIME_ARGUMENTS

    names = _runtime_names(kernel_node())
    slot = dict(zip(names, arguments))

    def held(name):
        value = slot[name]
        return getattr(value, "array", value)

    for index, (component, displacement, _axis) in enumerate(module.E_TERMS):
        assert held(f"f{index}") is getattr(fields, component)
        assert held(f"w{index}") is getattr(fields, "f_w_" + component)
        assert held(f"g{index}") is getattr(fields, displacement)
        assert held(f"e{index}") is fields.inverse_epsilon_for(component)
    # Distinct per-component inverse volumes, so a swap would be visible.
    assert held("e0") is not held("e1") and held("e1") is not held("e2")

    order = dispersive_module.poles_per_component(fields)
    for letter, (component, displacement, _axis) in zip(module.LETTERS,
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
                assert held(f"p_prev_{letter}{position}") is getattr(
                    fields, displacement)
                assert slot[f"sigma_{letter}{position}"] == 0.0
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

    # The six row slots in ROW_SLOTS order, a dead slot padded with the row
    # component's OWN D pointer — the certified base plan's rule.
    rows = offdiag_module.row_volumes_for(fields)
    for name, (row, _partner), value in zip(("u01", "u02", "u11", "u12",
                                             "u21", "u22"),
                                            offdiag_module.ROW_SLOTS, rows):
        if value is None:
            assert held(name) is getattr(fields, "D" + row[1])
        else:
            assert held(name) is value
    assert rows[0] is not None and rows[1] is None      # rows="one": Ex/Ey only

    # The six coefficient columns, xyz by kps/kms, flattened to 1-D views.
    for name, (axis, stem) in zip(("kp0", "km0", "kp1", "km1", "kp2", "km2"),
                                  [(axis, stem) for axis in "xyz"
                                   for stem in ("kps", "kms")]):
        column = getattr(pml, f"{stem}_{axis}_h")
        assert held(name).ndim == 1
        assert np.shares_memory(held(name), column)
    assert (slot["gwx"], slot["gwy"], slot["gwz"]) == \
        folded_module.mirror_ghost_weights(fields.grid)
    assert (slot["nx"], slot["ny"], slot["nz"]) == tuple(plan.shape)
    assert slot["n_elem"] == plan.n_elem
    # ...and the geometry tail sits exactly where the ADE block begins.
    assert names[names.index("n_elem") + 1] == "p_out_a0"


def test_the_pole_slots_are_the_E_halfs_live_binding_position_by_position():
    fields, pml = engine(poles=2)
    plan = plan_for(fields, pml)
    chain_ = plan._resolve(plan._e._poles.arrays())
    arguments = plan._arguments(chain_)
    names = _runtime_names(kernel_node())
    slot = dict(zip(names, arguments))
    groups = plan._e._poles.arrays()
    for letter, group, (component, displacement, _axis) in zip(
            module.LETTERS, groups, module.E_TERMS):
        for position in range(module.MAX_POLES):
            bound = slot[f"{letter}{position}"].array
            if position < len(group):
                assert bound is group[position]
            else:
                assert bound is getattr(fields, displacement)


def test_a_signature_the_launcher_does_not_follow_fails_at_the_call(monkeypatch):
    fields, pml = engine(poles=1)
    plan = plan_for(fields, pml)
    chain_ = plan._resolve(plan._e._poles.arrays())
    monkeypatch.setattr(module, "RUNTIME_ARGUMENTS", module.RUNTIME_ARGUMENTS + 1)
    with pytest.raises(RuntimeError, match="runtime arguments"):
        plan._arguments(chain_)


# ---------------------------------------------------------------------------
# The transcription — parsed out of both files, never re-implemented
# ---------------------------------------------------------------------------

def test_the_certified_body_is_an_ast_identical_prefix_of_the_fused_body():
    """The whole E half, contiguous and verbatim, before any ADE arm."""
    certified = [ast.dump(statement) for statement in _statements(certified_node())]
    fused = [ast.dump(statement) for statement in _statements(kernel_node())]
    assert len(certified) >= 40, "the certified body is not the body this test knows"
    assert fused[:len(certified)] == certified
    # What follows is the ADE half and nothing else: every trailing statement is
    # an `if NP{c} > k:` arm.
    trailing = _statements(kernel_node())[len(certified):]
    assert len(trailing) == 3 * module.CHAIN_MAX_POLES
    for statement in trailing:
        assert isinstance(statement, ast.If)
        assert re.fullmatch(r"NP[012] > \d", ast.unparse(statement.test))


def test_both_halves_of_the_certified_signature_are_prefixes_of_the_fused_one():
    certified = certified_node()
    fused = kernel_node()
    certified_runtime = _runtime_names(certified)
    certified_constexpr = _constexpr_names(certified)
    fused_runtime = _runtime_names(fused)
    fused_constexpr = _constexpr_names(fused)
    assert fused_runtime[:len(certified_runtime)] == certified_runtime
    assert fused_constexpr[:len(certified_constexpr)] == certified_constexpr
    assert len(certified_runtime) == 55 and len(certified_constexpr) == 19
    # The ADE runtime block sits between n_elem and NP0; the SV_* flags after BLOCK.
    everything = [a.arg for a in fused.args.args]
    assert everything[everything.index("n_elem") + 1] == "p_out_a0"
    assert everything[everything.index("NP0") - 1] == \
        f"cdrive_c{module.CHAIN_MAX_POLES - 1}"
    assert certified_constexpr[-1] == "BLOCK"
    assert fused_constexpr[len(certified_constexpr):] == [
        f"SV_{letter}{slot}" for letter in module.LETTERS
        for slot in range(module.CHAIN_MAX_POLES)]
    # ...and the ADE block is laid out as the chain lays it, per component.
    expected = []
    for letter in module.LETTERS:
        for stem in ("p_out", "p_prev", "sigma", "cnow", "cprev", "cdrive"):
            expected.extend(f"{stem}_{letter}{slot}"
                            for slot in range(module.CHAIN_MAX_POLES))
    assert fused_runtime[len(certified_runtime):] == expected


def test_the_recurrence_is_spelled_exactly_as_ade_update_p_spells_it():
    """The renames are the destination and the pointer; the arithmetic is verbatim."""
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
    for index, letter in enumerate(module.LETTERS):
        for slot in range(module.CHAIN_MAX_POLES):
            assert (f"tl.store(p_out_{letter}{slot} + idx, {expression}, "
                    f"mask=live)") in fused
            assert (f"q = tl.load(p_prev_{letter}{slot} + idx, mask=live, "
                    f"other=0.0)") in fused
            assert (f"s = tl.load(sigma_{letter}{slot} + idx, mask=live, "
                    f"other=0.0)") in fused
            assert f"s = sigma_{letter}{slot}" in fused
            # THE TWO SUBSTITUTIONS, and they are the only ones: the pole is
            # RE-READ through the once-bound pointer, the drive is the register.
            assert (f"p = tl.load({letter}{slot} + idx, mask=live, "
                    f"other=0.0)") in fused
            assert f"w = src{index}" in fused
            assert f"c_now = cnow_{letter}{slot}" in fused
            assert f"c_prev = cprev_{letter}{slot}" in fused
            assert f"c_drive = cdrive_{letter}{slot}" in fused
            arms += 1
    assert arms == 3 * module.CHAIN_MAX_POLES
    # ...and the drive is NEVER re-loaded from a buffer, nor the pole from a
    # second binding.
    assert "tl.load(drive" not in fused
    assert "p_now" not in fused
    assert "w = tl.load(" not in fused


def test_each_pole_pointer_is_bound_exactly_once():
    names = [a.arg for a in kernel_node().args.args]
    for letter in module.LETTERS:
        for slot in range(module.MAX_POLES):
            assert names.count(f"{letter}{slot}") == 1
    assert "p_now" not in names and "drive" not in names


def test_every_ade_arm_is_guarded_by_its_own_subtraction_slot():
    """A live arm whose pole the E chain did not subtract advances a dead slot."""
    guards: dict = {}
    for node in ast.walk(kernel_node()):
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
    for index, letter in enumerate(module.LETTERS):
        for slot in range(module.CHAIN_MAX_POLES):
            target = f"p_out_{letter}{slot} + idx"
            assert guards.get(target) == {f"NP{index} > {slot}"}, target


def test_the_callers_NP_guards_are_the_ade_arms_alone():
    """The E chain's ``if NP > k`` links live inside the certified helper."""
    fused = kernel_text()
    for index in range(3):
        links = len(re.findall(rf"if NP{index} > \d+:", fused))
        assert links == module.CHAIN_MAX_POLES
    certified = _function_text(pathlib.Path(fod_module.__file__),
                               "folded_offdiag_dispersive_constitutive_step")
    assert not re.search(r"if NP[012] > \d+:", certified)
    helper = _function_text(pathlib.Path(fod_module.__file__), "_load_d_minus_p")
    assert len(re.findall(r"if NP > \d+:", helper)) == module.MAX_POLES


def test_the_helpers_and_constexprs_are_bound_to_the_certified_modules_objects():
    """A called JIT helper resolves through THIS module's globals, so each is
    bound here, by name, to the certified module's own object."""
    for line in ("_masked_row_sum = _folded._masked_row_sum",
                 "_load_d_minus_p = _fod._load_d_minus_p",
                 "_folded_dispersive_term = _fod._folded_dispersive_term",
                 "PERIODIC = tl.constexpr(_fod.CODE_PERIODIC)",
                 "METALLIC = tl.constexpr(_fod.CODE_METALLIC)",
                 "MIRROR_ROW = tl.constexpr(_fod.MIRROR_SOURCE_INDEX)"):
        assert line in SOURCE, line
    fused = kernel_text()
    for name in ("_load_d_minus_p", "_folded_dispersive_term", "_masked_row_sum",
                 "PERIODIC", "METALLIC", "MIRROR_ROW"):
        assert name in fused, name
    if module.triton is not None:  # pragma: no cover - the device host
        assert module._masked_row_sum is folded_module._masked_row_sum
        assert module._load_d_minus_p is fod_module._load_d_minus_p
        assert module._folded_dispersive_term is fod_module._folded_dispersive_term
        assert module.PERIODIC.value == fod_module.CODE_PERIODIC
        assert module.METALLIC.value == fod_module.CODE_METALLIC
        assert module.MIRROR_ROW.value == fod_module.MIRROR_SOURCE_INDEX
    else:
        assert module._masked_row_sum is None
        assert module.folded_offdiag_fused_ade_chain_step is None


def test_the_slot_counts_and_terms_are_pinned_to_the_certified_bodies():
    assert module.MAX_POLES == dispersive_module.MAX_POLES
    assert module.MAX_POLES == fod_module.MAX_POLES
    assert module.E_TERMS is dispersive_module.E_TERMS
    assert module.E_TERMS is fod_module.E_TERMS
    assert module.LETTERS == ("a", "b", "c")


def test_the_lift_edits_are_six_and_every_needle_is_present():
    """The declared edits are the ONLY text this module adds to what it lifts."""
    assert len(module.LIFT_EDITS) == 6
    certified = _normalised(_function_text(
        pathlib.Path(fod_module.__file__),
        "folded_offdiag_dispersive_constitutive_step"))
    ade = _normalised(_function_text(PACKAGE_DIR / "kernels.py", "ade_update_p"))
    fused = _normalised(kernel_text())
    for edit in module.LIFT_EDITS:
        assert set(edit) == {"line", "became", "why"}
        line = _normalised(edit["line"])
        assert line in certified or line in ade, edit["line"]
        became = _normalised(edit["became"])
        if became.startswith("("):
            # An elision: the certified copy of the line survives at most once.
            assert fused.count(line) <= 1, edit["line"]
        else:
            assert became in fused, edit["became"]
            assert became not in certified or "n_elem" in became, edit["became"]
    assert module.LIFT_EDITS[0]["became"] == f"def {module.KERNEL_NAME}("


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
    signature = inspect.signature(module.folded_offdiag_fused_ade_chain_coverage)
    assert "sources" not in signature.parameters
    assert "arm" not in signature.parameters
    assert list(signature.parameters) == ["fields", "pml"]
    assert list(module.REPLACES) == ["update_E", "update_P"]
    assert module.CELL == ("folded off-diagonal dispersive", "ADE update_P")
    assert module.CARRIES_DEPOSIT_REPAIR is False


def test_the_cell_labels_are_the_planners_own():
    """The E-half label is an arm the planner writes; the ADE label its update_P."""
    launch = (PACKAGE_DIR / "launch.py").read_text(encoding="utf-8")
    assert f'"{module.CELL[0]}"' in launch
    assert f'"{module.CELL[1]}"' in launch


def test_the_shipped_policy_is_stamped_and_is_one_warp():
    assert module.POLICY["num_warps"] == 1
    assert module.POLICY["status"] == "keep"
    signature = inspect.signature(module.plan_folded_offdiag_fused_ade_chain)
    assert signature.parameters["num_warps"].default == 1
    assert "kernel" in signature.parameters
    assert module.FoldedOffdiagFusedAdeChainPlan.launches_per_run == 1
    assert module.FoldedOffdiagFusedAdeChainPlan.replaces_sub_steps == module.REPLACES
    assert module.FoldedOffdiagFusedAdeChainPlan.performs_device_work is True
    assert module.FAMILY == "folded_offdiag_fused_ade_chain"
    assert module.KERNEL_NAME == "folded_offdiag_fused_ade_chain_step"


def test_composition_into_plan_step_is_deferred_and_dispatch_stays_disabled():
    launch = (PACKAGE_DIR / "launch.py").read_text(encoding="utf-8")
    assert "folded_offdiag_fused_ade_chain" not in launch
    init = (PACKAGE_DIR / "__init__.py").read_text(encoding="utf-8")
    assert "folded_offdiag_fused_ade_chain" not in init
    assert module.INSTALLABLE is False
    reason = module.INSTALLABLE_REASON
    assert "update_E -> update_P" in reason
    assert "LIST of per-susceptibility plans" in reason
    assert "label" in reason and "outside this package" in reason
    assert "plan_step" in reason


def test_the_module_names_neither_the_dispatcher_nor_the_hand_cuda_track():
    for needle in ("fastpath", "cuda_kernels"):
        assert needle not in SOURCE, needle


def test_the_device_gate_is_present_and_names_this_module():
    assert GATE.exists(), GATE
    text = GATE.read_text(encoding="utf-8")
    assert "folded_offdiag_fused_ade_chain" in text
    assert "gate_provenance" in text


@pytest.mark.parametrize(
    "pin",
    (
        "products",
        "e_to_p_products",
        "e_to_p_verdict_keys",
        # THE PIN THAT OUTLIVED THE PLUMBING ROUND. GATE_BOUND names an ARTIFACT
        # and the board refuses a PRODUCTS row whose artifact is absent, so this
        # row could only be typed once the device campaign had run. It ran on
        # 2026-09-11 (the GPU host GPU 7, cc 8.6, released with no reasons), the row
        # was typed, and the strict xfail that guarded the gap came off with it —
        # which is the whole reason it was strict.
        "gate_bound",
        "certified_but_not_installed",
        "not_an_arm",
    ),
)
def test_the_board_pins_read_by_ast(pin):
    """The board, the reachability join and the planner test each name the product."""
    name = module.FAMILY
    if pin == "products":
        entry = _assigned_literal(BOARD, "PRODUCTS")[name]
        assert entry["seam"] == "E->P"
        assert tuple(entry["cell"]) == module.CELL
        assert entry["wired"] is False
        assert entry["module"].startswith("folded_offdiag_fused_ade_chain.py:")
        assert entry["module"].endswith("folded_offdiag_fused_ade_chain_coverage")
    elif pin == "e_to_p_products":
        assert name in _assigned_literal(BOARD, "E_TO_P_PRODUCTS")
    elif pin == "e_to_p_verdict_keys":
        assert tuple(_assigned_literal(BOARD, "E_TO_P_VERDICT_KEYS")[name]) == (name,)
    elif pin == "gate_bound":
        # RE-READ 2026-09-15, off the artifact rather than off its DIRECTORY NAME.
        # This pin used to spell the dated campaign directory, and it went red the
        # first time the board was repointed at a fresher run of the same gate --
        # a rename, not a change in what is certified. What the row has to be true
        # about is the ARTIFACT: it names this family's gate.json, that file is in
        # the tree, and the run it records RELEASED. A repoint to another released
        # run of the same gate passes; a row pointed at an absent or unreleased
        # artifact still fails, which is what the board itself refuses on.
        bound = _assigned_literal(BOARD, "GATE_BOUND")[name]
        assert bound.startswith("results/")
        assert bound.endswith(f"/{name}/gate.json")
        artifact = BOARD.parent / bound
        assert artifact.is_file(), artifact
        import json as _json  # noqa: PLC0415
        record = _json.loads(artifact.read_text(encoding="utf-8"))
        release = record.get("canonical_verdict") or record.get("release") or {}
        assert release.get("released") is True, (bound, release.get("reasons"))
    elif pin == "certified_but_not_installed":
        entry = _assigned_literal(REACHABILITY, "CERTIFIED_BUT_NOT_INSTALLED")[name]
        assert len(entry) > 40
    elif pin == "not_an_arm":
        assert name in _assigned_literal(PLANNER_TEST, "NOT_AN_ARM")
    else:  # pragma: no cover
        raise AssertionError(pin)
