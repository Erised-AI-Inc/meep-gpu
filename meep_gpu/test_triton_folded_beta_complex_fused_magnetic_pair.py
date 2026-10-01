"""Laptop contracts for the FOLDED COMPLEX BETA fused MAGNETIC B/H pair on Triton.

Everything here runs on the NumPy merge-bar machine. The load-bearing ones are two:

* the TRANSCRIPTION leg — this product claims to be
  ``folded_complex_fused_magnetic_pair.folded_complex_fused_curl_constitutive_B``
  plus
  exactly the seven statements ``folded_complex.folded_beta_bloch_pml_curl_step``
  (K3b) adds to K1, and that claim is checked here as EXACT statement-list
  equalities against BOTH shipped sources, with the insert's POSITION checked
  separately (an order-preserving deletion equality alone cannot see where the
  insert sits);
* the DEPOSIT FLAG — and it is worth LESS here than on the electric twin, which
  is a fact about the corpus rather than about the bracket. Of this cell's three
  rows only ``test_eigsrc_kz_0_complex`` declares in-seam ``B`` sources; the two
  binary gratings hold none, so
  :data:`~.folded_beta_complex_fused_magnetic_pair.CARRIES_DEPOSIT_REPAIR` at False
  costs ONE seam-instance of three, not all three. Both halves are asserted
  directly, by flipping the flag and re-asking the predicate with each source
  kind.

The device bytes are the gate's
(``parity/meep_gpu/probe_triton_folded_beta_complex_fused_magnetic_pair.py``); nothing
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

MODULE_NAME = "meep_gpu.triton_kernels.folded_beta_complex_fused_magnetic_pair"
MODULE_PATH = PACKAGE_DIR / "folded_beta_complex_fused_magnetic_pair.py"
API_ROOT = pathlib.Path(__file__).resolve().parents[1]
DRIVER_PATH = API_ROOT / "meep_gpu" / "driver.py"

#: The architecture whose run record this module's weld is read on. Each weld keeps
#: one record per compute capability, so "the run that measured it" is a question with
#: an architecture in it.
CERTIFIED_CAPABILITY = "8.6"
PARITY_DIR = API_ROOT / "parity" / "meep_gpu"
GATE = PARITY_DIR / "probe_triton_folded_beta_complex_fused_magnetic_pair.py"

#: test_eigsrc_kz_0_complex's own beta — the cell's smallest; any nonzero value
#: exercises the same insert (the binary gratings carry -0.685 and -0.912).
BETA_CORPUS = 0.2

#: The seven statements the K3b insert adds, and the ONLY thing that may
#: separate this kernel from ``folded_complex_fused_curl_constitutive_B``.
BETA_INSERT = (
    "if HAS_BETA:",
    "t_re, t_im = _mul_imag_coefficient_left(bp_re, bp_im, b_re, b_im, EXPANSION)",
    "curl0_re = curl0_re - t_re",
    "curl0_im = curl0_im - t_im",
    "t_re, t_im = _mul_imag_coefficient_left(bm_re, bm_im, a_re, a_im, EXPANSION)",
    "curl1_re = curl1_re - t_re",
    "curl1_im = curl1_im - t_im",
)


@pytest.fixture(scope="module")
def product():
    return importlib.import_module(MODULE_NAME)


class _Magnetic:
    """A magnetic source that publishes the index the injection writes.

    THE IN-SEAM SOURCE ON THIS SIDE. The driver injects a magnetic current
    between ``step_B`` and ``update_H`` (driver.py:3283), which is work inside the
    seam this kernel closes, so it is the deposit repair's bracket that carries it
    — and a source that cannot name the cells it writes cannot be repaired.
    """

    field_type = "B"

    def __init__(self, index=(3, 3, 0)) -> None:
        self._point_ix, self._point_iy, self._point_iz = (
            numpy.array([index[0]]), numpy.array([index[1]]),
            numpy.array([index[2]]))


class _MagneticWithoutIndex:
    field_type = "B"


class _Electric:
    """An ELECTRIC source, which never reaches this seam.

    It publishes an index for symmetry with the magnetic fixture; the point of
    the class is that the seam clause is quiet about it either way, because the
    driver injects it in the OTHER seam entirely.
    """

    field_type = "D"

    def __init__(self, index=(3, 3, 0)) -> None:
        self._point_ix, self._point_iy, self._point_iz = (
            numpy.array([index[0]]), numpy.array([index[1]]),
            numpy.array([index[2]]))


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
                               "folded_beta_complex_fused_curl_constitutive_B"),
        "base": _shipped_body(PACKAGE_DIR / "folded_complex_fused_magnetic_pair.py",
                              "folded_complex_fused_curl_constitutive_B"),
        "beta_curl": _shipped_body(PACKAGE_DIR / "folded_complex.py",
                                   "folded_beta_bloch_pml_curl_step"),
        "plain_curl": _shipped_body(PACKAGE_DIR / "folded_complex.py",
                                    "folded_bloch_pml_curl_step"),
        "electric_twin": _shipped_body(
            PACKAGE_DIR / "folded_beta_complex_fused_pair.py",
            "folded_beta_complex_fused_curl_constitutive_D"),
    }


def _probe_record(patterns):
    from meep_gpu.test_triton_complex_fields import stamp_probe_record

    return stamp_probe_record(
        {"backend": "cupy", "patterns": {name: "FMA_V1" for name in patterns}})


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


def _reasons(product, fields, pml, sources=(), probe=None):
    if probe is None:
        probe = _probe_record(product.PRODUCT_PROBE_PATTERNS)
    from meep_gpu.expansion_refusal import declaring_run_policy

    with declaring_run_policy("keep"):
        verdict = product.folded_beta_complex_fused_magnetic_pair_coverage(
            fields, pml, sources, probe=probe)
    return [r for r in verdict.reasons if "array module" not in r]


# ---------------------------------------------------------------------------
# THE TRANSCRIPTION — the claim this whole product rests on
# ---------------------------------------------------------------------------

def test_removing_the_beta_insert_reproduces_the_shipped_folded_pair(bodies):
    """``this kernel`` minus the seven beta statements IS
    ``folded_complex_fused_curl_constitutive_B`` — an EXACT statement-list
    equality."""
    without_beta = [line for line in bodies["fused"] if line not in BETA_INSERT]
    assert without_beta == bodies["base"], (
        "the folded beta complex fused electric pair is not "
        "folded_complex_fused_curl_constitutive_B plus the beta insert; first "
        "difference at index " + str(next((i for i, (a, b) in enumerate(
            zip(without_beta, bodies["base"])) if a != b), "length")))


def test_the_beta_insert_is_exactly_the_shipped_folded_curls_delta(bodies):
    """The seven statements are K3b's own delta over K1 — the same set."""
    delta = [line for line in bodies["beta_curl"]
             if line not in bodies["plain_curl"]]
    assert delta == list(BETA_INSERT), delta


def test_the_insert_sits_after_the_curl_and_before_both_masks(bodies):
    """The POSITION, which the deletion equality above cannot see.

    After the third ``dtdx`` curl line, before the cell-0 ownership mask — the
    array path's order (S:356-363 against S:369) and K3b's own placement,
    asserted in the fused body AND in the shipped beta curl.
    """
    curl_line = "curl2_re, curl2_im = _mul_coefficient_left(dtdx, t2_re, t2_im, EXPANSION)"
    mask_line = "at_x, at_y, at_z = i == 0, j == 0, k == 0"
    for label in ("fused", "beta_curl"):
        body = bodies[label]
        at = body.index("if HAS_BETA:")
        assert body[at - 1] == curl_line, (label, body[at - 1])
        assert body[at + len(BETA_INSERT)] == mask_line, (
            label, body[at + len(BETA_INSERT)])


def test_every_statement_comes_from_a_shipped_body(bodies):
    """Nothing in this kernel was written here."""
    shipped = set(bodies["beta_curl"]) | set(bodies["base"])
    invented = [line for line in bodies["fused"] if line not in shipped]
    assert invented == [], invented


def test_the_magnetic_weld_is_not_the_electric_one(bodies):
    """The two twins DIFFER, and where they differ is the constitutive read.

    ``update_H`` binds no inverse permeability — B and H differ by no volume in
    any configuration this family admits — so the magnetic body carries NO ``ie``
    pointer at all, where its electric twin loads one per component. A magnetic
    weld that carried an ``ie`` argument would be transcribing the wrong half,
    and the assertion is written in BOTH directions so a copy-paste that lost the
    load would fail here rather than on a device.
    """
    assert bodies["fused"] != bodies["electric_twin"]
    ours = "\n".join(bodies["fused"])
    theirs = "\n".join(bodies["electric_twin"])
    assert "tl.load(ie0 + idx, mask=own0, other=0.0)" in theirs
    assert "ie0" not in ours


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
    assert product.BACKWARD == 0
    assert product.CURL_SUB_STEP == "step_B"
    assert product.CONSTITUTIVE_SIDE == "H"
    assert product.FILL_FAMILY == "B"


def test_replaces_names_the_five_seam_passes(product):
    assert product.REPLACES == ("step_B", "fill_B", "zero_metal_B",
                                "fill_folded_far_ghosts_B", "update_H")


def test_the_probe_pattern_set_is_the_six_pattern_union(product):
    from meep_gpu.triton_kernels.complex_fields import PROBE_PATTERNS
    from meep_gpu.triton_kernels.folded_complex import PARITY_PROBE_PATTERN
    from meep_gpu.triton_kernels.special_kz import BETA_PROBE_PATTERN

    assert set(product.PRODUCT_PROBE_PATTERNS) == (
        set(PROBE_PATTERNS) | {PARITY_PROBE_PATTERN, BETA_PROBE_PATTERN})
    assert len(product.PRODUCT_PROBE_PATTERNS) == 6


def test_the_licensed_multiply_helpers_are_the_four_named(product):
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    called = {getattr(node.func, "id", "") for node in ast.walk(tree)
              if isinstance(node, ast.Call)}
    multiplies = {name for name in called
                  if name.startswith("_mul") or name.startswith("_rotate")}
    assert multiplies == set(product.LICENSED_MULTIPLY_HELPERS)


def test_the_single_launch_site_passes_the_packages_shared_guard_constant():
    text = MODULE_PATH.read_text(encoding="utf-8")
    assert "enable_fp_fusion=ENABLE_FP_FUSION if guard is None else bool(guard)" \
        in text
    assert text.count("kernel[self._grid](") == 1


def test_the_beta_words_come_from_the_shipped_host_rounding():
    """The builder computes the coefficient words through
    ``special_kz.beta_curl_coefficients`` (magnetic=True on this side — the
    ``+1j`` magnetic sign rides IN the four host-rounded words, so the kernel body
    is character-identical to the electric twin's insert and only the words
    differ), never a second rounding."""
    text = MODULE_PATH.read_text(encoding="utf-8")
    assert "beta_curl_coefficients(grid.beta, grid.dt, magnetic=True," in text
    assert text.count("beta_curl_coefficients(") == 1


def test_the_plan_validates_the_beta_words_shape(product):
    """The two-pair check and the has_beta binding exist in the plan; a full
    instantiation needs device pointers, so what is pinned here is the source."""
    text = MODULE_PATH.read_text(encoding="utf-8")
    assert "beta_words is two (re, im) pairs" in text
    assert "self.has_beta = int(has_beta)" in text
    assert "HAS_BETA=self.has_beta," in text
    assert "bp_re, bp_im, bm_re, bm_im," in text


# ---------------------------------------------------------------------------
# Admission and refusals
# ---------------------------------------------------------------------------

def test_the_corpus_family_is_admitted_WITH_its_magnetic_deposit(product):
    """A Y fold, complex storage, nonzero beta, active PML, MAGNETIC source —
    ``test_eigsrc_kz_0_complex``'s own shape, deposit included. That row is the
    one of this cell's three that declares in-seam ``B`` sources; the two
    binary gratings are the quiet family below."""
    fields, pml = _build()
    assert _reasons(product, fields, pml, (_Magnetic(),)) == []


def test_an_electric_source_never_reaches_this_seam(product):
    """The two binary-grating rows: electric sources only, injected in the OTHER
    seam, so this seam's clause is quiet about them."""
    fields, pml = _build()
    assert _reasons(product, fields, pml, (_Electric(),)) == []


@pytest.mark.parametrize("sources,needle", [
    (None, "was not declared"),
    ((_MagneticWithoutIndex(),), "does not publish the index"),
])
def test_the_source_seam_refuses_by_name(product, sources, needle):
    fields, pml = _build()
    reasons = _reasons(product, fields, pml, sources)
    assert any(needle in reason for reason in reasons), reasons


def test_the_flag_at_False_would_cost_ONE_of_this_cells_three_rows(product,
                                                                   monkeypatch):
    """THE FLAG IS WORTH LESS HERE THAN ON THE ELECTRIC TWIN, and the difference
    is the corpus rather than the bracket.

    All three of the electric twin's rows inject electrically, so its flag at
    False costs it the whole cell. On THIS seam only
    ``tests:TestSpecialKz.test_eigsrc_kz_0_complex`` declares magnetic sources;
    the two binary-grating rows hold none, and the clause is quiet on them at
    either setting. Both halves are asserted so the flag's real price is a
    measurement rather than an inherited sentence."""
    fields, pml = _build()
    assert product.CARRIES_DEPOSIT_REPAIR is True
    monkeypatch.setattr(product, "CARRIES_DEPOSIT_REPAIR", False)
    reasons = _reasons(product, fields, pml, (_Magnetic(),))
    assert any("is magnetic" in reason for reason in reasons), reasons
    # ...and the two rows with no magnetic source are untouched by the flag.
    assert _reasons(product, fields, pml, (_Electric(),)) == []


def test_a_zero_beta_run_is_refused_by_name(product):
    """The inverted clause: a beta = 0 folded complex run is the beta-less
    twin's."""
    fields, pml = _build(beta=0.0)
    reasons = _reasons(product, fields, pml, (_Magnetic(),))
    assert any("beta" in reason for reason in reasons), reasons


def test_an_unfolded_run_is_refused(product):
    """An unfolded complex beta grid belongs to
    ``complex_beta_fused_electric_pair``."""
    fields, pml = _build(mirrors=(), k_point=(0.3, 0.0, 0.0))
    reasons = _reasons(product, fields, pml, (_Magnetic(),))
    assert reasons, "an unfolded complex beta run must be refused here"


def test_real_storage_is_refused(product):
    """A real-storage folded beta run belongs to
    ``folded_beta_fused_electric_pair``."""
    fields, pml = _build(complex_storage=False)
    reasons = _reasons(product, fields, pml, (_Magnetic(),))
    assert any("real float32" in reason for reason in reasons), reasons


def test_an_offdiagonal_row_is_refused_as_a_NEIGHBOUR_READ(product):
    """No beta variant of the off-diagonal arms is shipped, so an off-diagonal
    beta row is refused BY NAME rather than claimed. The refusal arrives on the
    CURL half here (the electric twin takes it on the constitutive half, where
    the row product is the stencil); either way the corpus carries no such row."""
    fields, pml = _build(offdiag=True)
    reasons = _reasons(product, fields, pml, (_Magnetic(),))
    assert any("off-diagonal chi1inv row is installed" in reason
               for reason in reasons), reasons
    assert any("reads neighbours" in reason for reason in reasons), reasons


def test_a_probe_without_the_beta_pattern_is_refused(product):
    """The census probe's shape — five patterns, no parity — and a four-pattern
    record must BOTH refuse; the licence is the six-pattern union."""
    from meep_gpu.triton_kernels.complex_fields import PROBE_PATTERNS

    fields, pml = _build()
    four = _probe_record(PROBE_PATTERNS)
    reasons = _reasons(product, fields, pml, (_Magnetic(),), probe=four)
    assert any("EXTENDED" in reason or "not classified" in reason
               for reason in reasons), reasons


# ---------------------------------------------------------------------------
# Disjointness from the neighbours, in BOTH directions
# ---------------------------------------------------------------------------

def test_this_arm_and_the_beta_less_folded_pair_never_co_admit(product):
    from meep_gpu.expansion_refusal import declaring_run_policy
    from meep_gpu.triton_kernels import folded_complex_fused_magnetic_pair as beta_less

    probe = _probe_record(product.PRODUCT_PROBE_PATTERNS)
    with declaring_run_policy("keep"):
        for beta in (BETA_CORPUS, 0.0):
            fields, pml = _build(beta=beta)
            sources = (_Electric(),)
            mine = [r for r in product.folded_beta_complex_fused_magnetic_pair_coverage(
                fields, pml, sources, probe=probe).reasons
                if "array module" not in r]
            theirs = [r for r in beta_less.folded_complex_fused_magnetic_pair_coverage(
                fields, pml, sources, probe=probe).reasons
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
    from meep_gpu.expansion_refusal import declaring_run_policy
    from meep_gpu.triton_kernels import folded_complex

    probe = _probe_record(product.PRODUCT_PROBE_PATTERNS)
    fields, pml = _build(**kwargs)
    with declaring_run_policy("keep"):
        weld = product.folded_beta_complex_fused_magnetic_pair_coverage(
            fields, pml, sources, probe=probe)
        curl = folded_complex.folded_beta_bloch_pml_curl_coverage(
            fields, pml, "step_B", probe=probe)
        electric = folded_complex.folded_complex_constitutive_coverage(
            fields, pml, "E", probe=probe)
    assert (not weld.covered) or (curl.covered and electric.covered), (
        weld.reasons, curl.reasons, electric.reasons)


def test_the_plan_refuses_rather_than_raises_on_a_numpy_host(product):
    fields, pml = _build()
    probe = _probe_record(product.PRODUCT_PROBE_PATTERNS)
    assert product.plan_folded_beta_complex_fused_magnetic_pair(
        fields, pml, (_Electric(),), probe=probe) is None
    assert product.plan_folded_beta_complex_fused_magnetic_pair(
        fields, pml, None, probe=probe) is None


# ---------------------------------------------------------------------------
# The deferral, the wiring records and the gate
# ---------------------------------------------------------------------------

def test_the_composer_routes_this_product_and_dispatch_admits_it():
    """ROUTED 2026-09-02 by the installer wave, and RELEASED at dispatch on 2026-09-13.

    This test replaces the deferral it used to make. Until 2026-09-02 it asserted
    that ``launch.py`` and ``fastpath.py`` named this module NOWHERE, which was the
    seam that kept a certified-but-unrouted product deferred. The wave routed it:
    ``launch.CERTIFIED_FUSED_PRODUCTS`` holds its row and
    ``_install_certified_fused_products`` builds it through the shared
    ``_install_fused_pair``, taking both slots only when ``_pair_may_absorb`` finds
    the arm table has already given them to the arms this kernel implements.

    Phase B (batch 2026-09-13) then RELEASED the arm, checkable from the PRODUCT's
    side rather than from the composer's: the label this product writes is now in
    ``fastpath.RELEASED_FUSED_ARMS``, so clause (8) admits the whole plan; and it
    sits in exactly one of ``ARM_CERTIFICATION`` (its gate has a tracked ledger
    entry, and its weld was seeded from the identity fleet) and
    ``PENDING_DEVICE_GATE_ARMS`` (it does not, having left the pending rung).
    """
    import pathlib as _pathlib  # noqa: PLC0415

    from meep_gpu import fastpath as _fastpath  # noqa: PLC0415
    from meep_gpu.triton_kernels import launch as _launch  # noqa: PLC0415

    source = _pathlib.Path(_launch.__file__).read_text(encoding="utf-8")
    assert "plan_folded_beta_complex_fused_magnetic_pair" in source
    row = _launch.CERTIFIED_FUSED_PRODUCTS["folded_beta_complex_fused_magnetic_pair"]
    assert row["module"] == "folded_beta_complex_fused_magnetic_pair"
    assert row["builder"] == "plan_folded_beta_complex_fused_magnetic_pair"
    label = row["label"]
    assert label == 'fused pair B (folded complex beta)'
    assert _fastpath.arm_is_fused(label)
    assert label in _fastpath.RELEASED_FUSED_ARMS
    assert (label in _fastpath.ARM_CERTIFICATION) != (
        label in _fastpath.PENDING_DEVICE_GATE_ARMS)
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

    assert "folded_beta_complex_fused_magnetic_pair" not in NOT_AN_ARM
    assert "folded_beta_complex_fused_magnetic_pair" in _launch.SUPPORT_MODULES
def test_the_module_claims_identity_ONLY_through_the_run_that_measured_it():
    """This asserted the module had NO fingerprints entry, because it had none.

    Phase B (batch 2026-09-13) RELEASED this arm and seeded its weld from the
    identity fleet, so the absence it pinned is gone and the property worth pinning
    is the weld's SOUNDNESS: it must record a PASS, name this module, and name it at
    the bytes on disk. An entry that drifts from the file it certifies is worse than
    no entry, because it reads as evidence.

    AND THE RUN IS NAMED, which is what the title has always said and what the ledger
    shape now makes checkable: the artifact digest, the timestamp and the host belong
    to ONE run of the gate, kept under ``runs[<compute capability>]``, and they are
    read out of the record for the architecture this module's arm dispatches on. Read
    off the entry instead, they would be the last architecture rebound while reading
    as though they described every card.
    """
    import hashlib  # noqa: PLC0415

    from meep_gpu import fastpath  # noqa: PLC0415

    fingerprints = json.loads(
        (PACKAGE_DIR / "fingerprints.json").read_text(encoding="utf-8"))
    weld = fingerprints[
        "triton_folded_beta_complex_fused_magnetic_pair_device_gate"]
    assert weld["status"] == "PASS"
    assert CERTIFIED_CAPABILITY in fastpath.live_capabilities(weld), (
        f"the weld records no live run on compute capability "
        f"{CERTIFIED_CAPABILITY}: {fastpath.live_capabilities(weld)}")
    run = weld[fastpath.RUNS][CERTIFIED_CAPABILITY]
    assert run["artifact_sha256"] and run["recorded_utc"] and run["host"]
    module_key = "meep_gpu/triton_kernels/folded_beta_complex_fused_magnetic_pair.py"
    assert module_key in weld["source_sha256"], \
        "the weld does not name the module it certifies"
    live = hashlib.sha256((API_ROOT / module_key).read_bytes()).hexdigest()
    assert live == weld["source_sha256"][module_key], \
        f"{module_key} drifted from the bytes the gate executed"


def test_the_gate_exists_and_names_this_product():
    assert GATE.exists(), GATE
    text = GATE.read_text(encoding="utf-8")
    assert "folded_beta_complex_fused_magnetic_pair" in text
    assert "folded_beta_complex_fused_curl_constitutive_B" in text


def test_the_board_and_the_battery_all_know_this_product():
    board = (PARITY_DIR
             / "build_triton_fusion_matrix.py").read_text(encoding="utf-8")
    battery = (PARITY_DIR / "triton_predicate_battery.py").read_text(
        encoding="utf-8")
    assert '"folded_beta_complex_fused_magnetic_pair"' in board
    assert "folded_beta_complex_fused_magnetic_pair_coverage" in battery
    assert "folded_beta_complex_fused_magnetic_pair_B@unified_expansion" in battery


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
    text = GATE.read_text(encoding="utf-8")
    assert "def identity_has_beta_off_leg(" in text
    assert "m_beta_insert_dropped" in text
    assert "m_beta_partners_swapped" in text
    assert "m_beta_after_the_masks" in text
    assert "beta_licence" in text


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
            "meep_gpu/triton_kernels/folded_beta_complex_fused_magnetic_pair.py",
            "meep_gpu/triton_kernels/folded_complex_fused_magnetic_pair.py",
            "meep_gpu/triton_kernels/folded_complex.py",
            "meep_gpu/triton_kernels/special_kz.py",
            "meep_gpu/deposit_repair.py",
            "meep_gpu/stepping.py",
            "meep_gpu/driver.py",
            "meep_gpu/test_triton_folded_beta_complex_fused_magnetic_pair.py"):
        assert required in names, required


def test_the_gate_declares_the_cell_the_board_scored():
    gate = load_gate()
    assert gate.CELL == ("B->H", "folded complex beta PML", "folded complex")
    assert sorted(gate.CELL_ROWS) == [
        "tests:TestEigCoeffs.test_binary_grating_special_kz_0_13_2",
        "tests:TestEigCoeffs.test_binary_grating_special_kz_1_17_7",
        "tests:TestSpecialKz.test_eigsrc_kz_0_complex",
    ]
