"""Laptop contracts for the fused dispersive D/E/ADE chain.

The CUDA gate owns byte identity — nothing here compiles or launches anything.
These tests own the four things that decide whether the device run is measuring
what it claims: the seam's own admission clauses, the refusals that must stay
refusals (multi-pole, electric source, no PML), the ALIASING DISCIPLINE the kernel
source has to carry, and the arithmetic being spelled exactly as the two
separately certified products spell it.

The last two are structural checks on source text, which is unusual and is the
point: the register reuse this product exists for is invisible to any host-side
assertion, and a store hoisted above a history load would still produce a plausible
field on the one lattice a numerical test happens to run.

WHY ``CHAIN_MAX_POLES``' REASON LIVES HERE AND NOT BESIDE THE CONSTANT.
``fused_dispersive_chain.py`` is bound BY RAW SHA256 in
``triton_kernels/fingerprints.json`` under ``triton_fused_dispersive_chain_device_
gate``, and the 2026-08-20 closed fusion matrix RAISES on any drift — a prose edit
to that module would make a standing board unreproducible until its device gate is
re-run. So the cap's adjudication is pinned in these tests, which nothing binds,
and recorded in ``parity/meep_gpu/results/covering_refusals_2026-08-20/``. A test
is the better home anyway: a comment can go stale in silence, and
``test_the_pole_cap_is_GUARDED_and_not_dead_code`` cannot.
"""

from __future__ import annotations

import ast
import inspect
import pathlib
import re
from types import SimpleNamespace

import numpy as np
import pytest

from meep_gpu.dispersion import DRUDE, LORENTZIAN, PolarizationState, Susceptibility
from meep_gpu.fields import Fields
from meep_gpu.grid import Grid
from meep_gpu.pml import PML
from meep_gpu.triton_kernels import dispersive_fused_pair as pair_module
from meep_gpu.triton_kernels import fused_ade_state as ade_module
from meep_gpu.triton_kernels import fused_dispersive_chain as module
from meep_gpu.triton_kernels import launch as launch_module
from meep_gpu.triton_kernels.dispersive_update_e import MAX_POLES

PACKAGE_DIR = pathlib.Path(module.__file__).parent
PROBE = (PACKAGE_DIR.parents[1] / "parity" / "meep_gpu"
         / "probe_triton_fused_dispersive_chain.py")
RUNNER = (PACKAGE_DIR.parents[1] / "parity" / "meep_gpu"
          / "run_triton_fused_dispersive_chain_direct.sh")
SOURCE = pathlib.Path(module.__file__).read_text(encoding="utf-8")


def one_state(sigma=None, kind=LORENTZIAN, states=1):
    """A dispersive PML configuration on the laptop's NumPy backend."""
    grid = Grid(resolution=10.0, cell_size=(0.8, 0.8, 0.8))
    fields = Fields(grid=grid, force_complex_fields=False)
    fields.enable_pml_storage()
    for index in range(states):
        fields.polarizations.append(PolarizationState(
            Susceptibility(frequency=0.8 + 0.1 * index, gamma=0.1, kind=kind),
            0.4 if sigma is None else sigma, grid, np.float32))
    return fields, PML(grid=grid, thickness=2)


def reasons(fields, pml, sources=()):
    """Refusals other than the laptop's own array module."""
    return [reason for reason in
            module.fused_dispersive_chain_coverage(fields, pml, sources).reasons
            if "not cupy" not in reason]


# ---------------------------------------------------------------------------
# Admission and refusal
# ---------------------------------------------------------------------------

def test_the_module_imports_without_triton_and_refuses_before_any_launch():
    fields, pml = one_state()
    verdict = module.fused_dispersive_chain_coverage(fields, pml, ())
    assert not verdict.covered  # the laptop has no CuPy
    assert reasons(fields, pml, ()) == []
    assert module.plan_fused_dispersive_chain(fields, pml, ()) is None
    if module.fused_curl_dispersive_E_ade is None:
        with pytest.raises(ImportError, match="triton"):
            module.fused_curl_dispersive_E_ade_kernel()


@pytest.mark.parametrize("kind", (LORENTZIAN, DRUDE))
@pytest.mark.parametrize("sigma", (
    0.4,
    {"Ex": 0.4, "Ey": 0.0, "Ez": 0.25},
    {"Ex": 0.0, "Ey": 0.0, "Ez": 0.25},
))
def test_the_covered_shapes_leave_no_refusal_but_the_array_module(kind, sigma):
    fields, pml = one_state(sigma=sigma, kind=kind)
    assert reasons(fields, pml, ()) == []


def test_a_second_susceptibility_is_refused_by_name_not_truncated():
    fields, pml = one_state(states=2)
    verdict = reasons(fields, pml, ())
    assert verdict, "two susceptibilities must not be admitted"
    assert any("2 susceptibilities are registered" in reason for reason in verdict)
    assert any("bracketed" in reason for reason in verdict), verdict
    assert module.plan_fused_dispersive_chain(fields, pml, ()) is None


@pytest.mark.parametrize(
    ("sources", "refused"),
    (
        ((), False),
        ((SimpleNamespace(field_type="B"),), False),
        ((SimpleNamespace(field_type="D"),), True),
        (None, True),
    ),
)
def test_the_electric_source_seam_is_inherited_from_the_pair(sources, refused):
    fields, pml = one_state()
    verdict = reasons(fields, pml, sources)
    found = any("source" in reason and ("electric" in reason or "not declared" in reason)
                for reason in verdict)
    assert found is refused, verdict


def test_an_inactive_pml_is_refused_for_the_seams_own_reason():
    """Without PML ``drive_field`` returns E, so the register would be the wrong field."""
    grid = Grid(resolution=10.0, cell_size=(0.8, 0.8, 0.8))
    fields = Fields(grid=grid, force_complex_fields=False)
    fields.polarizations.append(PolarizationState(
        Susceptibility(frequency=0.8, gamma=0.1), 0.4, grid, np.float32))
    verdict = reasons(fields, None, ())
    assert any("drive_field" in reason and "stored E" in reason
               for reason in verdict), verdict


def test_a_zero_sigma_state_drives_nothing_and_is_refused_rather_than_launched():
    fields, pml = one_state(sigma=0.0)
    verdict = reasons(fields, pml, ())
    assert any("no driven components" in reason for reason in verdict), verdict


#: The cap clause's own words, so a paraphrase drifting in the module cannot leave
#: these tests quietly passing on a string nothing emits any more.
CAP_MARK = "live slot per component"
STATE_MARK = "susceptibilities are registered"


def _cap_fires(states: int) -> bool:
    fields, pml = one_state(states=states)
    return any(CAP_MARK in reason for reason in reasons(fields, pml, ()))


def test_the_pole_cap_is_INHERITED_behind_the_single_susceptibility_clause():
    """THE NAME OF THIS TEST USED TO BE A CLAIM ITS BODY DID NOT MEASURE.

    It asserted that SOME reason mentioning "susceptibilit" appeared at three
    states — which is the ``_single_state`` clause (:380-386) answering, not the
    cap at :420-423 — so the cap could have been deleted outright and the test
    would still have passed.

    What is true, and measured here: ``poles_per_component(fields)[c]`` is a SUBSET
    of ``fields.polarizations`` (dispersive_update_e.py:454-468), and
    ``_single_state`` refuses every state count but one BEFORE the loop runs. So
    the cap comparison cannot fire while that clause stands, at any N.
    """
    assert module.CHAIN_MAX_POLES == 1
    fired = {n: _cap_fires(n) for n in range(1, MAX_POLES + 1)}
    assert not any(fired.values()), fired

    # NON-VACUITY. "It never fires" means nothing on a sweep whose verdict never
    # moves, so the sweep must ADMIT at one end and REFUSE at the other.
    assert reasons(*one_state(states=1), ()) == []
    two = reasons(*one_state(states=2), ())
    assert two and any(STATE_MARK in reason for reason in two), two


def test_the_pole_cap_is_GUARDED_and_not_dead_code(monkeypatch):
    """The inverse floor: a clause that can never fire under ANY widening is dead.

    Reported inherited above; shown reachable here. ``_single_state`` alone is
    widened to accept any state count — every other clause untouched — and the cap
    must then refuse by name, naming the count it saw. Without this the first test
    is indistinguishable from one written against code that had been deleted.
    """
    def accept_any(fields):
        return tuple(fields.polarizations)[0], []

    monkeypatch.setattr(module, "_single_state", accept_any)
    for n in range(2, MAX_POLES + 1):
        verdict = reasons(*one_state(states=n), ())
        assert any(f"driven by {n} poles" in reason and CAP_MARK in reason
                   for reason in verdict), (n, verdict)
    # And it is not the ONLY thing written for one pole: the identity clause must
    # refuse too, or a future widening would raise the cap and walk into a body
    # whose ADE arm still reuses a single subtraction register.
    verdict = reasons(*one_state(states=2), ())
    assert any("the ADE arm and the subtraction slot must be the same pole" in reason
               for reason in verdict), verdict


def test_raising_the_pole_cap_ALONE_changes_no_verdict(monkeypatch):
    """The widening this family was asked to price, priced.

    Raising the cap to the sub-step's own bound must move nothing, because
    ``_single_state`` still refuses everything the cap would have. The measured
    corpus consequence is in the module's own comment: on all 15 E->P rows
    poles-per-component equals the susceptibility count, so no row has one state
    and more than one pole, and the seam delta of this widening is zero.
    """
    baseline = {n: reasons(*one_state(states=n), ()) for n in range(1, MAX_POLES + 1)}
    monkeypatch.setattr(module, "CHAIN_MAX_POLES", MAX_POLES)
    after = {n: reasons(*one_state(states=n), ()) for n in range(1, MAX_POLES + 1)}
    assert after == baseline


# ---------------------------------------------------------------------------
# The plan's own invariants
# ---------------------------------------------------------------------------

class _FakePointer:
    """Stands in for CupyPointer so the plan can be built without a device."""

    __slots__ = ("array",)

    def __init__(self, array):
        self.array = array


def _plan_arguments(fields, pml, state, components, counts):
    from meep_gpu.triton_kernels.dispersive_update_e import StaticPoleBinding

    volumes = [np.zeros(fields.grid.shape, dtype=np.float32) for _ in range(3)]
    poles = StaticPoleBinding([
        (state.P[name],) if count else () for name, count in
        zip(("Ex", "Ey", "Ez"), counts)])
    return dict(
        shape=fields.grid.shape, dtdx=0.5, bc=(0, 0, 0),
        zero_metal=(False, False, False), block=256,
        targets=[getattr(fields, n) for n in ("Dx", "Dy", "Dz")],
        auxiliaries=[getattr(fields, n) for n in ("fu_Dx", "fu_Dy", "fu_Dz")],
        sources=[getattr(fields, n) for n in ("Hx", "Hy", "Hz")],
        curl_coefficients=[getattr(pml, f"{s}_{a}")
                           for a in "xyz" for s in ("kms", "sinv")],
        e_targets=[getattr(fields, n) for n in ("Ex", "Ey", "Ez")],
        e_aux=[getattr(fields, n) for n in ("f_w_Ex", "f_w_Ey", "f_w_Ez")],
        inverse_epsilon=volumes,
        e_coefficients=[getattr(pml, f"{s}_{a}_h")
                        for a in "xyz" for s in ("kps", "kms")],
        poles=poles, state=state, components=components,
        pointer=_FakePointer,
    )


def test_an_ade_arm_without_its_subtraction_slot_is_a_construction_error():
    """The arm reuses the subtraction's register; a live arm with no pole is a wrong answer."""
    fields, pml = one_state()
    state = fields.polarizations[0]
    good = _plan_arguments(fields, pml, state, ("Ex", "Ey", "Ez"), (1, 1, 1))
    module.FusedDispersiveChainPlan(**good)  # constructs

    bad = _plan_arguments(fields, pml, state, ("Ex", "Ey", "Ez"), (1, 1, 0))
    with pytest.raises(ValueError, match="ADE arm live"):
        module.FusedDispersiveChainPlan(**bad)


def test_the_destination_chain_is_the_array_paths_rotation():
    """``_entries`` must reproduce dispersion.py:686-691's shared-scratch rotation."""
    fields, pml = one_state()
    state = fields.polarizations[0]
    plan = module.FusedDispersiveChainPlan(
        **_plan_arguments(fields, pml, state, ("Ex", "Ey", "Ez"), (1, 1, 1)))
    entries = plan._entries()
    assert entries["Ex"][0] is state._scratch
    assert entries["Ey"][0] is state.P_prev["Ex"]
    assert entries["Ez"][0] is state.P_prev["Ey"]
    assert entries["__tail__"][0] is state.P_prev["Ez"]
    # Every destination is distinct, and no destination is a P the kernel reads.
    outputs = [entries[name][0] for name in ("Ex", "Ey", "Ez")]
    assert len({id(array) for array in outputs}) == 3
    for name in ("Ex", "Ey", "Ez"):
        assert all(array is not state.P[name] for array in outputs)


def test_the_shipped_policy_is_stamped_and_is_one_warp():
    assert module.POLICY["num_warps"] == 1
    assert module.POLICY["status"] == "keep"
    signature = inspect.signature(module.plan_fused_dispersive_chain)
    assert signature.parameters["num_warps"].default == 1


# ---------------------------------------------------------------------------
# Source-level contracts: the aliasing discipline and the arithmetic
# ---------------------------------------------------------------------------

def _kernel_function() -> ast.FunctionDef:
    tree = ast.parse(SOURCE)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "fused_curl_dispersive_E_ade":
            return node
    raise AssertionError("the chain kernel is not in the module source")


def _calls(node, attribute):
    for child in ast.walk(node):
        if (isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute)
                and child.func.attr == attribute):
            yield child


def test_every_ade_history_load_precedes_every_ade_store():
    """The rotation makes o1 the volume q0p names; order must not be the compiler's.

    Triton pointer arguments carry no ``noalias``, so a conservative compiler would
    preserve a store-after-load order — this test refuses to depend on that.
    """
    kernel = _kernel_function()
    history_names = {"q0p", "q1p", "q2p"}
    destination_names = {"o0", "o1", "o2"}

    def base(call):
        argument = call.args[0]
        if isinstance(argument, ast.BinOp) and isinstance(argument.left, ast.Name):
            return argument.left.id
        return None

    load_lines = [call.lineno for call in _calls(kernel, "load")
                  if base(call) in history_names]
    store_lines = [call.lineno for call in _calls(kernel, "store")
                   if base(call) in destination_names]
    assert len(load_lines) == 3, load_lines
    assert len(store_lines) == 3, store_lines
    assert max(load_lines) < min(store_lines), (load_lines, store_lines)


def test_the_pole_register_is_reused_rather_than_reloaded():
    """The ADE arm must read the subtraction's register, not a second load of P."""
    kernel = _kernel_function()
    names = {"a0": "pa0", "b0": "pa1", "c0": "pa2"}
    for pointer, register in names.items():
        loads = [call for call in _calls(kernel, "load") if
                 isinstance(call.args[0], ast.BinOp)
                 and isinstance(call.args[0].left, ast.Name)
                 and call.args[0].left.id == pointer]
        assert len(loads) == 1, f"{pointer} is loaded {len(loads)} times"
    stores = [call for call in _calls(kernel, "store")
              if isinstance(call.args[0], ast.BinOp)
              and isinstance(call.args[0].left, ast.Name)
              and call.args[0].left.id in {"o0", "o1", "o2"}]
    spelled = {ast.dump(call.args[1]) for call in stores}
    assert len(spelled) == 3
    for register in ("pa0", "pa1", "pa2"):
        assert any(register in ast.dump(call.args[1]) for call in stores)


def test_the_recurrence_is_spelled_exactly_as_the_separate_ade_product_spells_it():
    """Byte identity is a measurement, but a DIFFERENT association guarantees failure."""
    ade_source = pathlib.Path(ade_module.__file__).read_text(encoding="utf-8")

    def normalized(text):
        text = re.sub(r"\s+", "", text)
        return re.sub(r"[a-z]+[0-9]", "X", text)

    separate = re.findall(
        r"\(\(p[0-9] \* c_now\) \+ \(c_prev \* q[0-9]\)\) \+\s*\(c_drive \* \(s[0-9] \* d[0-9]\)\)",
        ade_source)
    assert len(separate) == 3, separate
    fused = re.findall(
        r"\(\(pa[0-9] \* c_now\) \+ \(c_prev \* h[0-9]\)\) \+\s*\(c_drive \* \(sv[0-9] \* src[0-9]\)\)",
        SOURCE)
    assert len(fused) == 3, fused
    assert {normalized(text) for text in separate} == {normalized(text) for text in fused}


def test_the_constitutive_half_is_the_pairs_arithmetic_unchanged():
    """Only the pole-0 load is renamed; every other electric statement is verbatim."""
    pair_source = pathlib.Path(pair_module.__file__).read_text(encoding="utf-8")

    def body(text, name):
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == name:
                return [ast.dump(statement) for statement in node.body]
        raise AssertionError(name)

    # Docstrings differ by design; everything after them must not.
    pair_statements = body(pair_source, "fused_curl_dispersive_E")[1:]
    chain_statements = set(body(SOURCE, "fused_curl_dispersive_E_ade")[1:])
    carried = [statement for statement in pair_statements
               if statement in chain_statements]
    dropped = [statement for statement in pair_statements
               if statement not in chain_statements]
    assert len(carried) >= 30, (len(carried), len(pair_statements))
    # The ONLY statements the chain does not carry verbatim are the three pole-0
    # subtractions, which are split into a named load plus the same subtraction.
    assert len(dropped) == 3, [statement[:120] for statement in dropped]
    for statement in dropped:
        assert "'NP" in statement or "NP0" in statement or "NP1" in statement \
            or "NP2" in statement, statement[:200]


def test_composition_into_plan_step_is_deferred_and_dispatch_stays_disabled():
    launch_source = pathlib.Path(launch_module.__file__).read_text(encoding="utf-8")
    assert "fused_dispersive_chain" not in launch_source
    fastpath = pathlib.Path(launch_module.__file__).parents[1] / "fastpath.py"
    assert "fused_dispersive_chain" not in fastpath.read_text(encoding="utf-8")


def test_the_device_gate_is_present_and_named_by_the_module():
    assert PROBE.exists(), PROBE
    probe_source = PROBE.read_text(encoding="utf-8")
    assert "fused_dispersive_chain" in probe_source
    assert "plan_fused_dispersive_chain" in probe_source


def test_every_artifact_write_is_provenance_stamped_and_policy_re_read():
    """The gate's verdict must be bindable to the bytes that produced it.

    WHAT THIS CATCHES, and it is not hypothetical: the 2026-08-14 device run of
    this gate PASSED — 3/3 cases, 36/36 complete steps, 7 mutations caught — and
    its artifact recorded ZERO source digests. ``source_sha256`` was absent from
    the payload entirely, so a green verdict could not be attached to any tree,
    and the product could not be welded into ``fingerprints.json`` however many
    times it was re-run.

    The stamp is asserted INSIDE ``save``, not at its call sites. ``main`` calls
    ``save`` eleven times as the legs land, and a stamp on the last call alone
    would leave every partial artifact unattributable — including the one an
    aborted run leaves behind, which is the artifact a failure is read from.

    The policy stamp is asserted to be RE-READ rather than carried, for the same
    reason the signed-zero census carries a floor. Measured on the first attempt
    of the 2026-08-19 re-run: the install-time stamp recorded ``installed: true``
    beside ``ftz_removed: 0``, ``nvrtc_calls: 0`` and all ten Triton counters at
    zero, in an artifact whose kernels demonstrably compiled and launched 36
    times. A record like that cannot tell an installed policy from an inert one.
    """
    probe_source = PROBE.read_text(encoding="utf-8")
    tree = ast.parse(probe_source)
    save = next(node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == "save")

    # READ OFF THE AST, NOT OFF THE TEXT. The first spelling of this test matched
    # substrings of ``ast.unparse(save)``, which INCLUDES THE DOCSTRING — and the
    # docstring names both ``gate_provenance`` and ``stamp``. Measured against a
    # mutant with the import and the call deleted: the text check still read
    # True/True. A contract a comment can satisfy measures the comment.
    imported = {alias.asname or alias.name
                for node in ast.walk(save) if isinstance(node, ast.ImportFrom)
                and node.module == "gate_provenance"
                for alias in node.names}
    assert "stamp" in {name.rsplit(".", 1)[-1] for name in imported} or imported, (
        "save() does not import from gate_provenance: the artifact would record "
        "nothing about the bytes that produced it")
    called = {ast.unparse(node.func) for node in ast.walk(save)
              if isinstance(node, ast.Call)}
    assert called & imported, (
        f"save() imports {sorted(imported)} from gate_provenance but calls none "
        f"of them; it calls {sorted(called)}")
    assert any(name.endswith("policy_stamp") for name in called), (
        f"save() does not RE-READ the subnormal policy stamp (calls: "
        f"{sorted(called)}); an install-time stamp records zero for every "
        f"counter the mechanism increments")
    assigned = {ast.unparse(target) for node in ast.walk(save)
                if isinstance(node, ast.Assign) for target in node.targets}
    assert any("subnormal_policy" in target for target in assigned), (
        f"save() calls policy_stamp() but assigns it nowhere: {sorted(assigned)}")

    # ...and the gate installs a policy at all, explicitly rather than by
    # inheriting whatever the host's default resolves to.
    assert "install_subnormal_policy" in probe_source
    assert "--subnormal-policy" in probe_source


def test_the_keep_policy_cache_token_is_accepted_under_keep_and_refused_under_flush():
    """CuPy's cache key is computed ABOVE the seam the -ftz strip installs at.

    So the directory NAME is the only separator there is: a cache shared with a
    flush run either gets poisoned with stripped binaries or silently serves
    flushed ones under this record's name. ``subnormal_policy.cupy_cache_reasons``
    owns the rule: a directory carrying the token is accepted under keep and
    refused under flush, so a run must not hand the same directory to both.
    """
    from meep_gpu import subnormal_policy

    assert subnormal_policy.cupy_cache_reasons(
        subnormal_policy.KEEP, "/x/cupy_cache_ftz_stripped_1") == []
    assert subnormal_policy.cupy_cache_reasons(
        subnormal_policy.FLUSH, "/x/cupy_cache_ftz_stripped_1") != []
