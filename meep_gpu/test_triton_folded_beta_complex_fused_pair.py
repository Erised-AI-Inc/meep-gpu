"""Laptop contracts for the FOLDED COMPLEX BETA fused ELECTRIC D/E pair on Triton.

Everything here runs on the NumPy merge-bar machine. The load-bearing ones are two:

* the TRANSCRIPTION leg — this product claims to be
  ``folded_complex_fused_pair.folded_complex_fused_curl_constitutive_D`` plus
  exactly the seven statements ``folded_complex.folded_beta_bloch_pml_curl_step``
  (K3b) adds to K1, and that claim is checked here as EXACT statement-list
  equalities against BOTH shipped sources, with the insert's POSITION checked
  separately (an order-preserving deletion equality alone cannot see where the
  insert sits);
* the DEPOSIT FLAG — all THREE corpus rows of this cell (the two binary gratings
  and ``test_eigsrc_kz_0_complex``) declare ELECTRIC sources, so
  :data:`~.folded_beta_complex_fused_pair.CARRIES_DEPOSIT_REPAIR` at False takes
  the product from three seam-instances to zero. That is asserted DIRECTLY, by
  flipping the flag and re-asking the predicate.

The device bytes are the gate's
(``parity/meep_gpu/probe_triton_folded_beta_complex_fused_pair.py``); nothing
here launches.
"""

from __future__ import annotations

import ast
import importlib
import json
import pathlib
import sys
from typing import List

import numpy
import pytest

from meep_gpu.fields import Fields
from meep_gpu.grid import Grid, Mirror
from meep_gpu.pml import PML
from meep_gpu.test_triton_kernels import PACKAGE_DIR

MODULE_NAME = "meep_gpu.triton_kernels.folded_beta_complex_fused_pair"
MODULE_PATH = PACKAGE_DIR / "folded_beta_complex_fused_pair.py"
API_ROOT = pathlib.Path(__file__).resolve().parents[1]
DRIVER_PATH = API_ROOT / "meep_gpu" / "driver.py"
PARITY_DIR = API_ROOT / "parity" / "meep_gpu"
GATE = PARITY_DIR / "probe_triton_folded_beta_complex_fused_pair.py"

#: test_eigsrc_kz_0_complex's own beta — the cell's smallest; any nonzero value
#: exercises the same insert (the binary gratings carry -0.685 and -0.912).
BETA_CORPUS = 0.2



@pytest.fixture(scope="module")
def product():
    return importlib.import_module(MODULE_NAME)


class _Magnetic:
    field_type = "B"


class _Electric:
    """An electric source that publishes the index the injection writes."""

    field_type = "D"

    def __init__(self, index=(3, 3, 0)) -> None:
        self._point_ix, self._point_iy, self._point_iz = (
            numpy.array([index[0]]), numpy.array([index[1]]),
            numpy.array([index[2]]))


class _ElectricWithoutIndex:
    field_type = "D"


def _statements(text: str) -> List[str]:
    out: List[str] = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if line.strip():
            out.append(line.strip())
    return out


def _shipped_body(path: pathlib.Path, name: str) -> List[str]:
    """One shipped kernel's BODY statements, docstring and signature removed.

    Read from the FILE rather than imported (the merge bar has no Triton), and
    EXACT — never ``ast.unparse``d, because the comparison includes the
    PARENTHESISATION whose float32 grouping the product is about.
    """
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    node = next((found for found in ast.walk(tree)
                 if isinstance(found, ast.FunctionDef) and found.name == name), None)
    assert node is not None, f"{name} is not defined in {path}"
    body = node.body
    if (isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)):
        body = body[1:]
    segments = [ast.get_source_segment(text, statement) for statement in body]
    assert all(segment is not None for segment in segments), name
    return _statements("\n".join(segments))


@pytest.fixture(scope="module")
def bodies():
    return {
        "fused": _shipped_body(MODULE_PATH,
                               "folded_beta_complex_fused_curl_constitutive_D"),
        "base": _shipped_body(PACKAGE_DIR / "folded_complex_fused_pair.py",
                              "folded_complex_fused_curl_constitutive_D"),
        "beta_curl": _shipped_body(PACKAGE_DIR / "folded_complex.py",
                                   "folded_beta_bloch_pml_curl_step"),
        "plain_curl": _shipped_body(PACKAGE_DIR / "folded_complex.py",
                                    "folded_bloch_pml_curl_step"),
        "magnetic_twin": _shipped_body(
            PACKAGE_DIR / "folded_beta_complex_fused_magnetic_pair.py",
            "folded_beta_complex_fused_curl_constitutive_B"),
    }




def _build(cell_size=(1.2, 1.0, 0.0), beta=BETA_CORPUS, complex_storage=True,
           mirrors=(("Y", 1),), k_point=(0.0, 0.0, 0.0), seed=17,
           offdiag=False, **grid_kwargs):
    """A folded complex-beta Grid/Fields/PML triple on NumPy, PML storage on.

    Default: the corpus family — effective-2-D, complex64 storage, a Y fold, an
    active PML on every non-mirror face, beta != 0. Every refusal test perturbs
    exactly one clause off this.
    """
    grid = Grid(resolution=10.0, cell_size=cell_size, dimensions=2,
                boundaries=None, courant=0.35, k_point=k_point, beta=beta,
                symmetry=tuple(Mirror(axis, phase) for axis, phase in mirrors),
                **grid_kwargs)
    fields = Fields(grid=grid, force_complex_fields=complex_storage)
    if offdiag:
        diagonal = {name: numpy.full(grid.shape, 2.25, dtype=numpy.float32)
                    for name in ("Ex", "Ey", "Ez")}
        inverse = {name: numpy.full(grid.shape, 1.0 / 2.25, dtype=numpy.float32)
                   for name in ("Ex", "Ey", "Ez")}
        rows = {"Ex": {"Ey": numpy.full(grid.shape, 0.05, dtype=numpy.float32)}}
        fields.set_epsilon_volumes(diagonal, inverse, rows)
    else:
        fields.set_background_eps(2.25)
    fields.enable_pml_storage()
    folded = {"xyz".index(axis.lower()) for axis, _ in mirrors}
    thickness = tuple(
        (((0, 2) if axis in folded else (2, 2)) if grid.shape[axis] >= 6
         else (0, 0)) for axis in range(3))
    return fields, PML(grid=grid, thickness=thickness)


def _reasons(product, fields, pml, sources=()):

    verdict = product.folded_beta_complex_fused_pair_coverage(
        fields, pml, sources)
    return [r for r in verdict.reasons if "array module" not in r]


# ---------------------------------------------------------------------------
# THE TRANSCRIPTION — the claim this whole product rests on
# ---------------------------------------------------------------------------

def test_removing_the_beta_insert_reproduces_the_shipped_folded_pair(bodies):
    """``this kernel`` minus its ``if HAS_BETA:`` pass IS ``folded_complex_fused_curl_constitutive_D``.

    An EXACT statement-list equality, in order, against the shipped body read from the
    file (``test_triton_special_kz.kernel_body_and_insert``). If any statement of the
    root moved, this fails and names the first difference.
    """
    from meep_gpu.test_triton_special_kz import kernel_body_and_insert

    body, insert = kernel_body_and_insert(PACKAGE_DIR / "folded_beta_complex_fused_pair.py", "folded_beta_complex_fused_curl_constitutive_D")
    root, root_insert = kernel_body_and_insert(PACKAGE_DIR / "folded_complex_fused_pair.py",
                                               "folded_complex_fused_curl_constitutive_D")
    assert insert and not root_insert, "the beta pass is missing, or the root has one"
    assert body == root, (
        "first difference at index "
        + str(next((i for i, (a, b) in enumerate(zip(body, root)) if a != b),
                   "length")))


def test_the_beta_insert_is_exactly_the_shipped_folded_curls_delta(bodies):
    """The pass is ``folded_beta_bloch_pml_curl_step``'s own, statement for statement, and that curl is
    ``folded_bloch_pml_curl_step`` plus exactly that pass: the lifter adds nothing of its own
    to the beta pass (the increment, its ownership mask and the ladder)."""
    from meep_gpu.test_triton_special_kz import kernel_body_and_insert

    _body, insert = kernel_body_and_insert(PACKAGE_DIR / "folded_beta_complex_fused_pair.py", "folded_beta_complex_fused_curl_constitutive_D")
    curl_body, curl_insert = kernel_body_and_insert(PACKAGE_DIR / "folded_complex.py",
                                                    "folded_beta_bloch_pml_curl_step")
    root_curl, _ = kernel_body_and_insert(PACKAGE_DIR / "folded_complex.py",
                                          "folded_bloch_pml_curl_step")
    assert curl_insert and insert == curl_insert, (insert[:3], curl_insert[:3])
    assert curl_body == root_curl


def test_the_insert_sits_after_the_curl_and_before_both_masks(bodies):
    """The POSITION, which the deletion equality above cannot see.

    After the last curl form and before the wall clear, the stores and the ghost carries -- the array path's order (the
    curl forms, then the beta pass after every component, then the seam passes), and
    K3b's own placement, asserted in the fused body AND in the shipped beta curl.
    The pass's increment carries BOTH folded ownership masks: the cell-0 block on a
    non-periodic axis and the top-plane block on a folded PERIODIC one.
    """
    last_form = "curl2_im, km_x, si_x, km_y, si_y, du2)"
    for label, after in (("fused", "if ZM_X:"), ("beta_curl", "tl.store(u0 + 2 * idx, n0_re, mask=live & du0)")):
        body = bodies[label]
        at = body.index("if HAS_BETA:")
        assert body.index(last_form) < at < body.index(after), label
        insert = body[at:body.index(after)]
        assert any("last_" in line and "MIRROR_PERIODIC" not in line for line in insert)
        assert any(line.endswith("== MIRROR_PERIODIC:") for line in insert)
        assert any(line.endswith("!= PERIODIC:") for line in insert)


def test_every_statement_comes_from_a_shipped_body(bodies):
    """Nothing in this kernel was written here."""
    shipped = set(bodies["beta_curl"]) | set(bodies["base"])
    invented = [line for line in bodies["fused"] if line not in shipped]
    assert invented == [], invented


def test_the_electric_weld_is_not_the_magnetic_one(bodies):
    """The two twins DIFFER, and where they differ is the D seam."""
    assert bodies["fused"] != bodies["magnetic_twin"]
    ours = "\n".join(bodies["fused"])
    theirs = "\n".join(bodies["magnetic_twin"])
    # The E-side constitutive binds inverse epsilon; update_H never does.
    assert "tl.load(ie0 + idx, mask=own0, other=0.0)" in ours
    assert "ie0" not in theirs


# ---------------------------------------------------------------------------
# Contracts
# ---------------------------------------------------------------------------

def test_the_package_does_not_import_the_module_eagerly():
    for name in list(sys.modules):
        if name == MODULE_NAME:
            del sys.modules[name]
    importlib.import_module("meep_gpu.triton_kernels")
    eagerly_imported = MODULE_NAME in sys.modules
    importlib.import_module(MODULE_NAME)  # restore for the other tests
    assert not eagerly_imported


def test_backward_matches_the_sub_step_table(product):
    from meep_gpu.triton_kernels import launch  # noqa: PLC0415

    assert product.BACKWARD == int(
        launch.SUB_STEPS[product.CURL_SUB_STEP]["backward"])
    assert product.BACKWARD == 1
    assert product.CURL_SUB_STEP == "step_D"
    assert product.CONSTITUTIVE_SIDE == "E"
    assert product.FILL_FAMILY == "D"


def test_replaces_names_the_five_seam_passes(product):
    assert product.REPLACES == ("step_D", "fill_D", "zero_metal_D",
                                "fill_folded_far_ghosts_D", "update_E")


def test_the_licensed_multiply_helpers_are_the_connection_alone(product):
    """The only complex helper any kernel of this module calls is the Bloch
    connection; the beta pass's quarter turn and the parity carries have real
    operands and are formed part by part."""
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    called = {getattr(node.func, "id", "") for node in ast.walk(tree)
              if isinstance(node, ast.Call)}
    multiplies = {name for name in called
                  if name.startswith(("_mul", "_rotate", "_connect"))}
    assert multiplies <= set(product.MULTIPLY_HELPERS), multiplies
    assert set(product.MULTIPLY_HELPERS) == {"_connect"}


def test_the_single_launch_site_passes_the_packages_shared_guard_constant():
    text = MODULE_PATH.read_text(encoding="utf-8")
    assert "enable_fp_fusion=ENABLE_FP_FUSION if guard is None else bool(guard)" \
        in text
    assert text.count("kernel[self._grid](") == 1


def test_the_beta_words_come_from_the_shipped_host_rounding():
    """The builder computes the two quarter-turn words through
    ``special_kz.beta_curl_coefficients`` (magnetic=False on this side, complex
    storage), never a second rounding."""
    text = MODULE_PATH.read_text(encoding="utf-8")
    assert ("beta_curl_coefficients(fields.grid.beta, fields.grid.dt,\n"
            "                                   magnetic=False, "
            "complex_storage=True)") in text
    assert text.count("beta_curl_coefficients(") == 1


def test_the_plan_validates_the_beta_words_shape(product):
    """The two-word check and the has_beta binding exist in the plan, and the words
    reach the launch after the runtime group; a full instantiation needs device
    pointers, so what is pinned here is the source."""
    text = MODULE_PATH.read_text(encoding="utf-8")
    assert "beta_words must be the two real beta words" in text
    assert "self.has_beta = int(has_beta)" in text
    assert "HAS_BETA=self.has_beta," in text
    assert "*self.beta_words," in text


# ---------------------------------------------------------------------------
# Admission and refusals
# ---------------------------------------------------------------------------

def test_the_corpus_family_is_admitted_WITH_its_electric_deposit(product):
    """A Y fold, complex storage, nonzero beta, active PML, ELECTRIC source —
    the corpus rows' own shape, deposit included."""
    fields, pml = _build()
    assert _reasons(product, fields, pml, (_Electric(),)) == []


def test_a_magnetic_source_never_reaches_this_seam(product):
    fields, pml = _build()
    assert _reasons(product, fields, pml, (_Magnetic(),)) == []


@pytest.mark.parametrize("sources,needle", [
    (None, "was not declared"),
    ((_ElectricWithoutIndex(),), "does not publish the index"),
])
def test_the_source_seam_refuses_by_name(product, sources, needle):
    fields, pml = _build()
    reasons = _reasons(product, fields, pml, sources)
    assert any(needle in reason for reason in reasons), reasons


def test_the_flag_at_False_would_cost_the_whole_cell(product, monkeypatch):
    """All three corpus rows declare electric sources; with
    ``CARRIES_DEPOSIT_REPAIR`` False the seam clause refuses them outright."""
    fields, pml = _build()
    assert product.CARRIES_DEPOSIT_REPAIR is True
    monkeypatch.setattr(product, "CARRIES_DEPOSIT_REPAIR", False)
    reasons = _reasons(product, fields, pml, (_Electric(),))
    assert any("is electric" in reason for reason in reasons), reasons


def test_a_zero_beta_run_is_refused_by_name(product):
    """The inverted clause: a beta = 0 folded complex run is the beta-less
    twin's."""
    fields, pml = _build(beta=0.0)
    reasons = _reasons(product, fields, pml, (_Electric(),))
    assert any("beta" in reason for reason in reasons), reasons


def test_an_unfolded_run_is_refused(product):
    """An unfolded complex beta grid belongs to
    ``complex_beta_fused_electric_pair``."""
    fields, pml = _build(mirrors=(), k_point=(0.3, 0.0, 0.0))
    reasons = _reasons(product, fields, pml, (_Electric(),))
    assert reasons, "an unfolded complex beta run must be refused here"


def test_real_storage_is_refused(product):
    """A real-storage folded beta run belongs to
    ``folded_beta_fused_electric_pair``."""
    fields, pml = _build(complex_storage=False)
    reasons = _reasons(product, fields, pml, (_Electric(),))
    assert any("real float32" in reason for reason in reasons), reasons


def test_an_offdiagonal_row_is_refused_as_the_stencil(product):
    fields, pml = _build(offdiag=True)
    reasons = _reasons(product, fields, pml, (_Electric(),))
    assert any("STENCIL" in reason for reason in reasons), reasons


# ---------------------------------------------------------------------------
# Disjointness from the neighbours, in BOTH directions
# ---------------------------------------------------------------------------

def test_this_arm_and_the_beta_less_folded_pair_never_co_admit(product):
    from meep_gpu.triton_kernels import folded_complex_fused_pair as beta_less

    for beta in (BETA_CORPUS, 0.0):
        fields, pml = _build(beta=beta)
        sources = (_Electric(),)
        mine = [r for r in product.folded_beta_complex_fused_pair_coverage(
            fields, pml, sources).reasons
            if "array module" not in r]
        theirs = [r for r in beta_less.folded_complex_fused_pair_coverage(
            fields, pml, sources).reasons
            if "array module" not in r]
        assert mine or theirs, (
            f"both predicates admitted a beta={beta!r} folded complex run")
        if beta == 0.0:
            assert mine, "a beta = 0 folded complex run must be refused here"
        else:
            assert mine == [] and theirs, \
                "a folded complex beta run is this arm's"


# ---------------------------------------------------------------------------
# The weld is never wider than either half
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kwargs,sources", [
    ({}, ()),
    ({}, (_Electric(),)),
    ({}, (_Magnetic(),)),
    ({}, None),
    ({"beta": 0.0}, (_Electric(),)),
    ({"complex_storage": False}, (_Electric(),)),
    ({"mirrors": (), "k_point": (0.3, 0.0, 0.0)}, (_Electric(),)),
    ({"offdiag": True}, (_Electric(),)),
])
def test_the_weld_is_never_wider_than_the_halves(product, kwargs, sources):
    from meep_gpu.triton_kernels import folded_complex

    fields, pml = _build(**kwargs)
    weld = product.folded_beta_complex_fused_pair_coverage(
        fields, pml, sources)
    curl = folded_complex.folded_beta_bloch_pml_curl_coverage(
        fields, pml, "step_D")
    electric = folded_complex.folded_complex_constitutive_coverage(
        fields, pml, "E")
    assert (not weld.covered) or (curl.covered and electric.covered), (
        weld.reasons, curl.reasons, electric.reasons)


def test_the_plan_refuses_rather_than_raises_on_a_numpy_host(product):
    fields, pml = _build()
    assert product.plan_folded_beta_complex_fused_pair(
        fields, pml, (_Electric(),)) is None
    assert product.plan_folded_beta_complex_fused_pair(
        fields, pml, None) is None


# ---------------------------------------------------------------------------
# The deferral, the wiring records and the gate
# ---------------------------------------------------------------------------

def test_the_composer_routes_this_product_and_the_driver_seam_released_it():
    """ROUTED 2026-09-02 by the installer wave, RELEASED 2026-09-13 — both halves.

    This test replaces the deferral it used to make. Until 2026-09-02 it asserted
    that ``launch.py`` and ``fastpath.py`` named this module NOWHERE, which was the
    seam that kept a certified-but-unrouted product deferred. The wave routed it:
    ``launch.CERTIFIED_FUSED_PRODUCTS`` holds its row and
    ``_install_certified_fused_products`` builds it through the shared
    ``_install_fused_pair``, taking both slots only when ``_pair_may_absorb`` finds
    the arm table has already given them to the arms this kernel implements.

    WIRING WAS NOT RELEASING, and this is the file where that sentence ended. It
    was named ``..._and_dispatch_still_refuses_it`` for as long as the label sat
    outside ``fastpath.RELEASED_FUSED_ARMS``; on 2026-09-13 the Phase B batch
    ``dispatch_fused_route_2026-09-13_phaseB`` drove it through the driver's own
    consults on ``folded_complex_beta_2d`` — the corpus case this product's D seam
    was selected for — so the release names that case and the pending rung no longer
    holds it.

    THE PARTITION IS STILL EXACTLY ONE OF TWO, in the other direction: the label
    is in ``ARM_CERTIFICATION`` (its ledger entry was cut by ``seed_triton_welds.py``
    from the 2026-09-12 identity fleet) and out of ``PENDING_DEVICE_GATE_ARMS``.
    Both are asserted rather than one, because a label released while still pending
    would be a plan claiming a certification the rung says it lacks.
    """
    import pathlib as _pathlib  # noqa: PLC0415

    from meep_gpu import fastpath as _fastpath  # noqa: PLC0415
    from meep_gpu.triton_kernels import launch as _launch  # noqa: PLC0415

    source = _pathlib.Path(_launch.__file__).read_text(encoding="utf-8")
    assert "plan_folded_beta_complex_fused_pair" in source
    row = _launch.CERTIFIED_FUSED_PRODUCTS["folded_beta_complex_fused_pair"]
    assert row["module"] == "folded_beta_complex_fused_pair"
    assert row["builder"] == "plan_folded_beta_complex_fused_pair"
    label = row["label"]
    assert label == 'fused pair D (folded complex beta)'
    assert _fastpath.arm_is_fused(label)
    assert label in _fastpath.RELEASED_FUSED_ARMS
    # THE CASE LIST GREW ON 2026-09-14, and this assertion is re-read off the route
    # table rather than re-typed: the target round added `folded_complex_beta_bloch_2d`,
    # the bloch-True corner of the same arm, and a hand-typed tuple went stale the day
    # it landed. Every case named here must be a DRIVE row that asks this arm for a
    # DISPATCH, which is the property the release actually rests on.
    released_cases = _fastpath.RELEASED_FUSED_ARMS[label]
    assert set(released_cases) == {"folded_complex_beta_2d",
                                   "folded_complex_beta_bloch_2d"}
    import sys as _sys  # noqa: PLC0415
    _route_dir = str(_pathlib.Path(_fastpath.__file__).parent.parent
                     / "parity" / "meep_gpu")
    if _route_dir not in _sys.path:
        _sys.path.insert(0, _route_dir)
    import gate_dispatch_fused_route as _route  # noqa: PLC0415
    for _case in released_cases:
        assert _case in _route.DRIVE, _case
        assert _route.DRIVE[_case]["expect"] == "dispatch", _case
        assert label in _route.DRIVE[_case]["arms"], _case
    assert label in _fastpath.ARM_CERTIFICATION
    assert label not in _fastpath.PENDING_DEVICE_GATE_ARMS
    assert label in _fastpath.FUSED_ARM_CONSTITUENTS
def test_the_module_is_named_in_the_deposit_repair_wiring_records():
    text = (PACKAGE_DIR.parent
            / "test_fused_pair_deposit_wiring.py").read_text(encoding="utf-8")
    assert f"triton_kernels/{MODULE_PATH.name}" in text


def test_the_module_is_declared_ROUTED_rather_than_NOT_AN_ARM():
    """It LEFT ``test_triton_planner_composition.NOT_AN_ARM`` on 2026-09-02.

    Both directions, because both failures are wrong in opposite ways: a routed
    module still named in that tuple would tell a reader the planner cannot reach
    it, and a module missing from ``SUPPORT_MODULES`` would break the lazy-import
    seam this package rests on.
    """
    from meep_gpu.test_triton_planner_composition import NOT_AN_ARM  # noqa: PLC0415
    from meep_gpu.triton_kernels import launch as _launch  # noqa: PLC0415

    assert "folded_beta_complex_fused_pair" not in NOT_AN_ARM
    assert "folded_beta_complex_fused_pair" in _launch.SUPPORT_MODULES
def test_the_module_claims_identity_ONLY_through_the_run_that_measured_it(product):
    """RELEASED 2026-09-13: the product carries the weld its gate run measured.

    Until the gate released, this asserted the STRICT ABSENCE of a fingerprint while
    ``DEVICE_STATUS`` said the module had never run. The Phase B batch ran the gate
    on the fleet (``results/triton_fleet_2026-09-13_batch_C``, canonical verdict
    released, source digest bound to this checkout) and ``seed_triton_welds.py`` cut
    the ledger entry from that artifact, so the strict absence inverts to a STRICT
    PRESENCE: this module's own device-gate weld is checked in. The fleet seed does
    not rewrite the source ``DEVICE_STATUS`` string, so its ``gate.json`` clause is
    kept but guarded — it holds the moment an in-place re-run updates that constant.
    """
    fingerprints = json.loads(
        (PACKAGE_DIR / "fingerprints.json").read_text(encoding="utf-8"))
    assert "triton_folded_beta_complex_fused_pair_device_gate" in fingerprints, \
        "the released product must carry the weld its gate run measured"
    if "NOT YET RUN" not in product.DEVICE_STATUS:
        assert "gate.json" in product.DEVICE_STATUS


def test_the_gate_exists_and_names_this_product():
    assert GATE.exists(), GATE
    text = GATE.read_text(encoding="utf-8")
    assert "folded_beta_complex_fused_pair" in text
    assert "folded_beta_complex_fused_curl_constitutive_D" in text


def test_the_board_and_the_battery_all_know_this_product():
    board = (PARITY_DIR
             / "build_triton_fusion_matrix.py").read_text(encoding="utf-8")
    battery = (PARITY_DIR / "triton_predicate_battery.py").read_text(
        encoding="utf-8")
    assert '"folded_beta_complex_fused_pair"' in board
    assert "folded_beta_complex_fused_pair_coverage" in battery
    assert 'seam["folded_beta_complex_fused_pair_D"]' in battery
    assert "folded_beta_complex_fused_pair_D@unified_expansion" not in battery, "the retired census key"


def load_gate():
    import importlib.util  # noqa: PLC0415

    spec = importlib.util.spec_from_file_location("_fbcfp_gate", GATE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_gates_transcription_and_pairing_legs_pass_here():
    gate = load_gate()
    for name in ("transcription_leg", "seam_binding_leg", "mutation_pairing_leg"):
        leg = getattr(gate, name)()
        assert leg.get("passed") is True, (name, leg)


def test_every_mutation_the_gate_declares_is_ARMED_against_the_shipped_source():
    gate = load_gate()
    source = gate.shipped_source()
    for name, _kind, _reason, rewrite in gate.mutation_table():
        mutated, count = rewrite(source)
        assert count >= 1, name
        assert mutated != source, name


def test_the_gate_carries_the_identity_leg_and_the_beta_mutations():
    """The HAS_BETA=0 identity leg and the beta pass's shared mutations -- the pass
    dropped, its partners swapped, its increment unmasked, the
    quarter turn reversed and the top-plane mask dropped -- each in the table the probe
    scores, on a case that carries everything its ``needs`` names."""
    text = GATE.read_text(encoding="utf-8")
    assert "def identity_has_beta_off_leg(" in text
    assert "fold_passes.beta_mutations(True, True, CURL_ANCHOR)" in text
    gate = load_gate()
    names = {row[0] for row in gate.mutation_table()}
    for required in ("b_pass_dropped", "b_partners_swapped", "b_increment_unmasked",
                     "b_quarter_turn_reversed", "b_top_plane_mask_dropped"):
        assert required in names, required
        case = gate.MUTATION_CASE.get(required, gate.DEFAULT_MUTATION_CASE)
        assert set(gate.MUTATION_NEEDS[required]) <= set(gate.CASE_CARRIES[case]), (
            required, case)


def test_the_gate_binds_the_seam_passes_the_driver_actually_calls():
    gate = load_gate()
    text = DRIVER_PATH.read_text(encoding="utf-8")
    positions = []
    for name in gate.SEAM_PASSES:
        index = text.find(f"{name}(self.fields")
        assert index > 0, f"driver.py never calls {name}(self.fields...)"
        positions.append(index)
    assert positions == sorted(positions), gate.SEAM_PASSES


def test_the_gate_pins_the_files_this_product_actually_rests_on():
    gate = load_gate()
    names = set(gate.source_hashes())
    for required in (
            "meep_gpu/triton_kernels/folded_beta_complex_fused_pair.py",
            "meep_gpu/triton_kernels/folded_complex_fused_pair.py",
            "meep_gpu/triton_kernels/folded_complex.py",
            "meep_gpu/triton_kernels/special_kz.py",
            "meep_gpu/deposit_repair.py",
            "meep_gpu/stepping.py",
            "meep_gpu/driver.py",
            "meep_gpu/test_triton_folded_beta_complex_fused_pair.py"):
        assert required in names, required


def test_the_gate_declares_the_cell_the_board_scored():
    gate = load_gate()
    assert gate.CELL == ("D->E", "folded complex beta PML", "folded complex")
    assert sorted(gate.CELL_ROWS) == [
        "tests:TestEigCoeffs.test_binary_grating_special_kz_0_13_2",
        "tests:TestEigCoeffs.test_binary_grating_special_kz_1_17_7",
        "tests:TestSpecialKz.test_eigsrc_kz_0_complex",
    ]
