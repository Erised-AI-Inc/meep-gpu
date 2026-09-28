"""Laptop contracts for the FOLDED real-beta fused ELECTRIC D/E pair on Triton.

Everything here runs on the NumPy merge-bar machine. The load-bearing ones are two:

* the TRANSCRIPTION leg — this product claims to be
  ``folded_fused_pair.folded_fused_curl_constitutive_D`` plus exactly the three
  lines ``folded_complex.folded_beta_pml_curl_step`` adds to
  ``symmetry.pml_curl_step_folded``, and that claim is checked here as EXACT
  statement-list equalities against the shipped sources, with the insert's
  POSITION checked separately (an order-preserving deletion equality alone
  cannot see where the insert sits);
* the DEPOSIT FLAG — the single corpus row of this cell
  (``tests:TestSpecialKz.test_eigsrc_kz_1_real_imag``) declares TWO ELECTRIC
  sources, so :data:`~.folded_beta_fused_electric_pair.CARRIES_DEPOSIT_REPAIR`
  at False takes the product from one seam-instance to zero. That is asserted
  DIRECTLY, by flipping the flag and re-asking the predicate. Unlike the
  unfolded beta pair there is no asymmetric twin here: the SAME row injects
  magnetically into the B seam too, so the magnetic twin carries the flag as
  well, and that symmetry is pinned.

The device bytes are the gate's
(``parity/meep_gpu/probe_triton_folded_beta_fused_electric_pair.py``); nothing
here launches.
"""

from __future__ import annotations

import ast
import importlib
import pathlib
import sys
from typing import List

import numpy
import pytest

from meep_gpu.fields import Fields
from meep_gpu.grid import Grid, Mirror
from meep_gpu.pml import PML
from meep_gpu.test_triton_kernels import PACKAGE_DIR

MODULE_NAME = "meep_gpu.triton_kernels.folded_beta_fused_electric_pair"
MODULE_PATH = PACKAGE_DIR / "folded_beta_fused_electric_pair.py"
API_ROOT = pathlib.Path(__file__).resolve().parents[1]
DRIVER_PATH = API_ROOT / "meep_gpu" / "driver.py"
PARITY_DIR = API_ROOT / "parity" / "meep_gpu"
GATE = PARITY_DIR / "probe_triton_folded_beta_fused_electric_pair.py"

#: TestSpecialKz.test_eigsrc_kz_1_real_imag's own beta — the single row this
#: cell has.
BETA_CORPUS = 0.2

#: The three lines the beta term adds, and the ONLY thing that may separate this
#: kernel from ``folded_fused_pair.folded_fused_curl_constitutive_D``.
BETA_INSERT = (
    "if HAS_BETA:",
    "curl0 = curl0 - (beta_plus * b)",
    "curl1 = curl1 - (beta_minus * a)",
)


@pytest.fixture(scope="module")
def product():
    return importlib.import_module(MODULE_NAME)


class _Magnetic:
    field_type = "B"


class _Electric:
    """An electric source that publishes the index the injection writes."""

    field_type = "D"

    def __init__(self, index=(2, 2, 0)) -> None:
        self._point_ix, self._point_iy, self._point_iz = index


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
        "fused": _shipped_body(MODULE_PATH, "folded_beta_fused_curl_constitutive_D"),
        "base": _shipped_body(PACKAGE_DIR / "folded_fused_pair.py",
                              "folded_fused_curl_constitutive_D"),
        "beta_curl": _shipped_body(PACKAGE_DIR / "folded_complex.py",
                                   "folded_beta_pml_curl_step"),
        "plain_curl": _shipped_body(PACKAGE_DIR / "symmetry.py",
                                    "pml_curl_step_folded"),
        "magnetic_twin": _shipped_body(
            PACKAGE_DIR / "folded_beta_fused_magnetic_pair.py",
            "folded_beta_fused_curl_constitutive_B"),
    }


def _build(cell_size=(1.2, 1.0, 0.0), boundaries=None, pml_thickness=2,
           complex_storage=False, beta=BETA_CORPUS, k_point=(0.0, 0.0, 0.0),
           mirrors=(("Y", 1),), seed=17, thickness=None, **grid_kwargs):
    """A folded real-beta Grid/Fields/PML triple on NumPy, PML storage enabled.

    Default: the corpus family — 2-D Cartesian (the ONLY place MEEP allows
    beta), real storage, a Y fold over the PERIODIC declaration, an active PML
    on every non-mirror face, k = 0. Every refusal test perturbs exactly one
    clause off this.
    """
    grid = Grid(resolution=10.0, cell_size=cell_size, dimensions=2,
                boundaries=boundaries, courant=0.35, k_point=k_point, beta=beta,
                symmetry=tuple(Mirror(axis, phase) for axis, phase in mirrors),
                **grid_kwargs)
    fields = Fields(grid=grid, force_complex_fields=complex_storage)
    fields.set_background_eps(2.25)
    fields.enable_pml_storage()
    rng = numpy.random.default_rng(seed)
    for name in ("Bx", "By", "Bz", "Dx", "Dy", "Dz", "Ex", "Ey", "Ez",
                 "f_w_Ex", "f_w_Ey", "f_w_Ez"):
        getattr(fields, name)[...] = rng.uniform(
            -0.4, 0.4, size=grid.shape).astype(numpy.float32)
    if thickness is None:
        folded = {"xyz".index(axis.lower()) for axis, _ in mirrors}
        thickness = tuple(
            (((0, pml_thickness) if axis in folded
              else (pml_thickness, pml_thickness))
             if grid.shape[axis] >= 6 else (0, 0))
            for axis in range(3))
    return fields, PML(grid=grid, thickness=thickness)


def _reasons(product, fields, pml, sources=()):
    verdict = product.folded_beta_fused_electric_pair_coverage(fields, pml, sources)
    return [r for r in verdict.reasons if "array module" not in r]


# ---------------------------------------------------------------------------
# THE TRANSCRIPTION — the claim this whole product rests on
# ---------------------------------------------------------------------------

def test_removing_the_beta_insert_reproduces_the_shipped_folded_pair(bodies):
    """``this kernel`` minus the three beta lines IS
    ``folded_fused_curl_constitutive_D`` — an EXACT statement-list equality."""
    without_beta = [line for line in bodies["fused"] if line not in BETA_INSERT]
    assert without_beta == bodies["base"], (
        "the folded beta fused electric pair is not "
        "folded_fused_curl_constitutive_D plus the beta insert; first difference "
        "at index " + str(next((i for i, (a, b) in enumerate(
            zip(without_beta, bodies["base"])) if a != b), "length")))


def test_the_beta_insert_is_exactly_the_shipped_folded_curls_delta(bodies):
    """The three lines are ``folded_beta_pml_curl_step``'s own delta over
    ``pml_curl_step_folded``."""
    delta = [line for line in bodies["beta_curl"]
             if line not in bodies["plain_curl"]]
    assert delta == list(BETA_INSERT), delta


def test_the_insert_sits_after_the_curl_and_before_both_masks(bodies):
    """The POSITION, which the deletion equality above cannot see.

    After the third ``dtdx`` curl line, before the cell-0 ownership mask — the
    array path's order (S:438-445 after :429, before :450) and K3a's own
    placement, asserted in the fused body AND in the shipped beta curl.
    """
    for label in ("fused", "beta_curl"):
        body = bodies[label]
        at = body.index("if HAS_BETA:")
        assert body[at - 1] == "curl2 = dtdx * ((b_x - b) + (a - a_y))", (
            label, body[at - 1])
        assert body[at + 3] == "at_x, at_y, at_z = i == 0, j == 0, k == 0", (
            label, body[at + 3])


def test_every_statement_comes_from_a_shipped_body(bodies):
    """Nothing in this kernel was written here."""
    shipped = set(bodies["beta_curl"]) | set(bodies["base"])
    invented = [line for line in bodies["fused"] if line not in shipped]
    assert invented == [], invented


def test_the_electric_weld_is_not_the_magnetic_one(bodies):
    """The two twins DIFFER, and where they differ is the D seam."""
    assert bodies["fused"] != bodies["magnetic_twin"]
    ours = set(bodies["fused"])
    theirs = set(bodies["magnetic_twin"])
    for component in range(3):
        line = (f"src{component} = v{component} * tl.load("
                f"ie{component} + idx, mask=own{component}, other=0.0)")
        assert line in ours, line
        assert line not in theirs, line
    # zero_metal_D clears TWO tangential components per axis; zero_metal_B one.
    assert "v2 = tl.where(at_x, 0.0, v2)" in ours
    assert "v2 = tl.where(at_x, 0.0, v2)" not in theirs


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


def test_the_module_answers_coverage_but_the_kernel_fails_clearly_without_triton(
        product):
    if product.folded_beta_fused_curl_constitutive_D is not None:
        pytest.skip("triton is installed here; the absence arm is exercised on "
                    "the merge-bar box without it")
    fields, pml = _build()
    assert product.folded_beta_fused_electric_pair_coverage(
        fields, pml, ()).reasons
    with pytest.raises(ImportError, match="triton"):
        product.folded_beta_fused_curl_constitutive_D_kernel()


def test_backward_matches_the_sub_step_table(product):
    from meep_gpu.triton_kernels import launch  # noqa: PLC0415

    assert product.BACKWARD == int(
        launch.SUB_STEPS[product.CURL_SUB_STEP]["backward"])
    assert product.BACKWARD == 1
    assert product.CURL_SUB_STEP == "step_D"
    assert product.CONSTITUTIVE_SIDE == "E"


def test_replaces_names_the_five_seam_passes(product):
    assert product.REPLACES == ("step_D", "fill_D", "zero_metal_D",
                                "fill_folded_far_ghosts_D", "update_E")


def test_the_single_launch_site_passes_the_packages_shared_guard_constant():
    text = MODULE_PATH.read_text(encoding="utf-8")
    assert "enable_fp_fusion=ENABLE_FP_FUSION if guard is None else bool(guard)" \
        in text
    assert text.count("kernel[self._grid](") == 1


def test_the_carry_device_function_is_imported_not_copied():
    """``_carry_ghost_E`` comes from ``folded_fused_pair``, so the two modules
    cannot drift apart on the destination arithmetic."""
    text = MODULE_PATH.read_text(encoding="utf-8")
    assert "from .folded_fused_pair import _carry_ghost_E" in text
    tree = ast.parse(text)
    defined = {node.name for node in ast.walk(tree)
               if isinstance(node, ast.FunctionDef)}
    assert "_carry_ghost_E" not in defined


# ---------------------------------------------------------------------------
# Admission and refusals
# ---------------------------------------------------------------------------

def test_the_corpus_family_is_admitted_WITH_its_electric_deposit(product):
    """A Y fold over PERIODIC, nonzero beta, active PML, ELECTRIC source —
    the corpus row's own shape, deposit included."""
    fields, pml = _build()
    assert _reasons(product, fields, pml, (_Electric(),)) == []


def test_a_metallic_fold_is_admitted(product):
    fields, pml = _build(boundaries={"x": "metallic", "y": "metallic",
                                     "z": "periodic"})
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
    """The measurement the module's docstring makes, executed rather than
    restated: the single corpus row declares electric sources, and with
    ``CARRIES_DEPOSIT_REPAIR`` False the seam clause refuses it outright."""
    fields, pml = _build()
    assert product.CARRIES_DEPOSIT_REPAIR is True
    monkeypatch.setattr(product, "CARRIES_DEPOSIT_REPAIR", False)
    reasons = _reasons(product, fields, pml, (_Electric(),))
    assert any("is electric" in reason for reason in reasons), reasons


def test_the_magnetic_twin_carries_the_flag_too_because_the_row_sources_both_seams(
        product):
    """UNLIKE the unfolded beta pair, the twins here are SYMMETRIC: the one
    corpus row declares sources ['D', 'D', 'B', 'B'], so each twin repairs its
    own seam's pair and both flags are True."""
    twin = importlib.import_module(
        "meep_gpu.triton_kernels.folded_beta_fused_magnetic_pair")
    assert twin.CARRIES_DEPOSIT_REPAIR is True
    assert product.CARRIES_DEPOSIT_REPAIR is True


def test_the_flag_is_passed_to_the_clause_that_reads_it_and_not_merely_declared():
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and getattr(node.func, "attr", "") == "seam_source_reasons"]
    assert len(calls) == 1, calls
    keywords = {keyword.arg: ast.unparse(keyword.value)
                for keyword in calls[0].keywords}
    assert keywords["carries_repair"] == "CARRIES_DEPOSIT_REPAIR", keywords
    assert ast.unparse(calls[0].args[2]) == "'D'", ast.unparse(calls[0].args[2])


def test_a_zero_beta_run_is_refused_by_name(product):
    """The inverted clause: a beta = 0 folded run is the plain folded pair's."""
    fields, pml = _build(beta=0.0)
    reasons = _reasons(product, fields, pml, (_Electric(),))
    assert any("beta" in reason for reason in reasons), reasons


def test_an_unfolded_run_is_refused(product):
    """An unfolded beta grid belongs to ``beta_fused_electric_pair``."""
    fields, pml = _build(mirrors=())
    reasons = _reasons(product, fields, pml, (_Electric(),))
    assert reasons, "an unfolded beta run must be refused here"


def test_complex_storage_is_refused(product):
    fields, pml = _build(complex_storage=True)
    reasons = _reasons(product, fields, pml, (_Electric(),))
    assert reasons, "complex storage must be refused"


def test_a_fold_beside_a_wall_on_the_same_axis_never_admits(product):
    """``stepping._zero_metal`` skips a folded axis, so a configuration that
    reports both a fold and a wall on one axis is a drift the predicate must
    refuse rather than compile.

    MEASURED: the refusal fires one rung EARLIER than the weld's own wall/fold
    clause — the curl half's ``folded_axis_kinds`` sees ``_stored_past_owned``
    and ``is_metallic`` disagreeing about the termination and refuses the drift
    by name. Either spelling is a refusal; admitting is the failure.
    """
    fields, pml = _build()
    walls = [False, True, False]
    fields.grid.has_metallic = True
    original = fields.grid.is_metallic
    fields.grid.is_metallic = lambda axis: bool(walls[axis])
    try:
        reasons = _reasons(product, fields, pml, (_Electric(),))
        assert any(("wall" in reason) or ("disagree" in reason)
                   for reason in reasons), reasons
    finally:
        fields.grid.is_metallic = original


# ---------------------------------------------------------------------------
# Disjointness from the plain folded pair, in BOTH directions
# ---------------------------------------------------------------------------

def test_this_arm_and_the_plain_folded_pair_never_co_admit(product):
    from meep_gpu.triton_kernels import folded_fused_pair  # noqa: PLC0415

    for beta in (BETA_CORPUS, 0.0):
        fields, pml = _build(beta=beta)
        sources = (_Electric(),)
        mine = [r for r in product.folded_beta_fused_electric_pair_coverage(
            fields, pml, sources).reasons if "array module" not in r]
        theirs = [r for r in folded_fused_pair.folded_fused_pair_coverage(
            fields, pml, sources).reasons if "array module" not in r]
        assert mine or theirs, (
            f"both predicates admitted a beta={beta!r} folded run")
        if beta == 0.0:
            assert mine, "a beta = 0 folded run must be refused here"
        else:
            assert mine == [] and theirs, "a folded beta run is this arm's"


# ---------------------------------------------------------------------------
# The weld is never wider than either half
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kwargs,sources", [
    ({}, ()),
    ({}, (_Electric(),)),
    ({}, (_Magnetic(),)),
    ({}, None),
    ({"beta": 0.0}, (_Electric(),)),
    ({"complex_storage": True}, (_Electric(),)),
    ({"mirrors": ()}, (_Electric(),)),
    ({"boundaries": {"x": "metallic", "y": "metallic", "z": "periodic"}},
     (_Electric(),)),
    ({"k_point": (0.2, 0.0, 0.0)}, (_Electric(),)),
])
def test_the_weld_is_never_wider_than_the_halves(product, kwargs, sources):
    from meep_gpu.triton_kernels import folded_complex  # noqa: PLC0415

    fields, pml = _build(**kwargs)
    weld = product.folded_beta_fused_electric_pair_coverage(fields, pml, sources)
    curl = folded_complex.folded_beta_pml_curl_coverage(fields, pml, "step_D")
    electric = folded_complex.folded_beta_run_constitutive_coverage(
        fields, pml, "E")
    assert (not weld.covered) or (curl.covered and electric.covered), (
        weld.reasons, curl.reasons, electric.reasons)


def test_the_plan_refuses_rather_than_raises_on_a_numpy_host(product):
    fields, pml = _build()
    assert product.plan_folded_beta_fused_electric_pair(
        fields, pml, (_Electric(),)) is None
    assert product.plan_folded_beta_fused_electric_pair(fields, pml, None) is None


# ---------------------------------------------------------------------------
# The deferral, the wiring records and the gate
# ---------------------------------------------------------------------------

def test_the_composer_routes_this_product_and_dispatch_still_refuses_it():
    """ROUTED 2026-09-02 by the installer wave, and RELEASED at dispatch 2026-09-13.

    This test replaces the deferral it used to make. Until 2026-09-02 it asserted
    that ``launch.py`` and ``fastpath.py`` named this module NOWHERE, which was the
    seam that kept a certified-but-unrouted product deferred. The wave routed it:
    ``launch.CERTIFIED_FUSED_PRODUCTS`` holds its row and
    ``_install_certified_fused_products`` builds it through the shared
    ``_install_fused_pair``, taking both slots only when ``_pair_may_absorb`` finds
    the arm table has already given them to the arms this kernel implements.

    WIRING WAS NOT RELEASING, and until 2026-09-13 the second half asserted that
    sentence from the PRODUCT's side: the label this product wrote was not in
    ``fastpath.RELEASED_FUSED_ARMS``, so clause (8) refused the whole plan. Phase B
    of the 2026-09-13 batch (driver record
    ``dispatch_fused_route_2026-09-13_phaseB``) released this arm — it seeded a
    ``fingerprints.json`` weld and took an ``ARM_CERTIFICATION`` row — so the label
    is now IN ``RELEASED_FUSED_ARMS`` and clause (8) ADMITS it. It still sits in
    exactly one of ``ARM_CERTIFICATION`` (its gate has a tracked ledger entry) and
    ``PENDING_DEVICE_GATE_ARMS`` (it does not).
    """
    import pathlib as _pathlib  # noqa: PLC0415

    from meep_gpu import fastpath as _fastpath  # noqa: PLC0415
    from meep_gpu.triton_kernels import launch as _launch  # noqa: PLC0415

    source = _pathlib.Path(_launch.__file__).read_text(encoding="utf-8")
    assert "plan_folded_beta_fused_electric_pair" in source
    row = _launch.CERTIFIED_FUSED_PRODUCTS["folded_beta_fused_electric_pair"]
    assert row["module"] == "folded_beta_fused_electric_pair"
    assert row["builder"] == "plan_folded_beta_fused_electric_pair"
    label = row["label"]
    assert label == 'fused pair D (folded real beta)'
    assert _fastpath.arm_is_fused(label)
    assert label in _fastpath.RELEASED_FUSED_ARMS
    assert (label in _fastpath.ARM_CERTIFICATION) != (
        label in _fastpath.PENDING_DEVICE_GATE_ARMS)
    assert label in _fastpath.FUSED_ARM_CONSTITUENTS
def test_the_module_is_named_in_the_deposit_repair_wiring_records():
    text = (PACKAGE_DIR.parent
            / "test_fused_pair_deposit_wiring.py").read_text(encoding="utf-8")
    assert f"triton_kernels/{MODULE_PATH.name}" in text


def test_the_module_claims_identity_ONLY_through_the_run_that_measured_it(product):
    import json  # noqa: PLC0415

    fingerprints = json.loads(
        (PACKAGE_DIR / "fingerprints.json").read_text(encoding="utf-8"))
    if "NOT RELEASED" in product.DEVICE_STATUS:
        assert MODULE_PATH.stem not in json.dumps(fingerprints), \
            "an unreleased product may not carry a checked-in fingerprint"
    else:
        assert "gate.json" in product.DEVICE_STATUS


def test_the_gate_exists_and_names_this_product():
    assert GATE.exists(), GATE
    text = GATE.read_text(encoding="utf-8")
    assert "folded_beta_fused_electric_pair" in text
    assert "folded_beta_fused_curl_constitutive_D" in text


def test_the_weld_driver_the_board_and_the_battery_all_know_this_product():
    driver = (PARITY_DIR / "drive_triton_weld_gates.py").read_text(encoding="utf-8")
    board = (PARITY_DIR
             / "build_triton_fusion_matrix.py").read_text(encoding="utf-8")
    battery = (PARITY_DIR / "triton_predicate_battery.py").read_text(
        encoding="utf-8")
    assert "probe_triton_folded_beta_fused_electric_pair.py" in driver
    assert '"folded_beta_fused_electric_pair"' in board
    assert "folded_beta_fused_electric_pair_coverage" in battery


def test_the_module_is_declared_ROUTED_rather_than_NOT_AN_ARM():
    """It LEFT ``test_triton_planner_composition.NOT_AN_ARM`` on 2026-09-02.

    Both directions, because both failures are wrong in opposite ways: a routed
    module still named in that tuple would tell a reader the planner cannot reach
    it, and a module missing from ``SUPPORT_MODULES`` would break the lazy-import
    seam this package rests on.
    """
    from meep_gpu.test_triton_planner_composition import NOT_AN_ARM  # noqa: PLC0415
    from meep_gpu.triton_kernels import launch as _launch  # noqa: PLC0415

    assert "folded_beta_fused_electric_pair" not in NOT_AN_ARM
    assert "folded_beta_fused_electric_pair" in _launch.SUPPORT_MODULES
def load_gate():
    import importlib.util  # noqa: PLC0415

    spec = importlib.util.spec_from_file_location("_fbfep_gate", GATE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_gates_no_device_legs_all_pass_here():
    gate = load_gate()
    for name in ("transcription_leg", "equivalence_leg", "corpus_admission_leg",
                 "mutation_arming_leg"):
        leg = getattr(gate, name)()
        assert leg.get("passed") is True, (name, leg)


def test_every_mutation_the_gate_declares_is_ARMED_against_the_shipped_source():
    gate = load_gate()
    source = gate.shipped_source()
    for name, _kind, _reason, rewrite in gate.mutation_table():
        mutated, count = rewrite(source)
        assert count >= 1, name
        assert mutated != source, name


def test_every_mutation_is_scored_on_a_case_that_enters_the_branch_it_rewrites():
    gate = load_gate()
    for name, _kind, _reason, _rewrite in gate.mutation_table():
        gate.mutation_case_for(name)   # raises if the pairing is dead


def test_the_gate_binds_the_seam_passes_the_driver_actually_calls():
    gate = load_gate()
    text = DRIVER_PATH.read_text(encoding="utf-8")
    positions = []
    for name in gate.SEAM_PASSES:
        index = text.find(f"{name}(self.fields")
        assert index > 0, f"driver.py never calls {name}(self.fields...)"
        positions.append(index)
    assert positions == sorted(positions), gate.SEAM_PASSES


def test_the_gate_carries_a_carry_family_and_a_null_control():
    gate = load_gate()
    assert gate.CARRY_CASES, "the deposit carry is this cell's whole value"
    text = GATE.read_text(encoding="utf-8")
    assert "bracket" in text and "null" in text.lower()


def test_the_gate_carries_the_certified_kernel_identity_arm():
    """``HAS_BETA = 0`` must reproduce ``folded_fused_curl_constitutive_D``."""
    text = GATE.read_text(encoding="utf-8")
    assert "has_beta" in text
    assert "folded_fused_curl_constitutive_D" in text


def test_the_gate_pins_the_files_this_product_actually_rests_on():
    gate = load_gate()
    names = set(gate.source_hashes())
    for required in (
            "meep_gpu/triton_kernels/folded_beta_fused_electric_pair.py",
            "meep_gpu/triton_kernels/folded_fused_pair.py",
            "meep_gpu/triton_kernels/folded_complex.py",
            "meep_gpu/triton_kernels/symmetry.py",
            "meep_gpu/deposit_repair.py",
            "meep_gpu/stepping.py",
            "meep_gpu/driver.py",
            "meep_gpu/test_triton_folded_beta_fused_electric_pair.py"):
        assert required in names, required


def test_the_gate_declares_the_cell_the_board_scored():
    gate = load_gate()
    assert gate.CELL == ("D->E", "folded real beta PML", "folded beta run")
