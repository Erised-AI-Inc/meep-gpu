"""Laptop contracts for the tensor / off-diagonal epsilon update_E Triton tranche.

Everything here runs on the NumPy merge-bar machine: the optional-import
contract, the predicate's admission AND per-clause refusals on real
Grid/Fields/PML objects, the plan builders' shapes, and an in-test
transcription of the off-diagonal sub-step pinned against ``stepping.update_E``
itself — including the two transcription subtleties this family adds (the
coefficient multiply BETWEEN the shifts, and the metallic wall-coupling mask).
The kernel's device bytes are the gate's
(``parity/meep_gpu/gate_triton_offdiag.py``); nothing here launches.
"""

from __future__ import annotations

import importlib
import pathlib
import sys
from types import SimpleNamespace

import numpy
import pytest

from meep_gpu import stepping
from meep_gpu.fields import Fields, IYEE_SHIFTS
from meep_gpu.grid import Grid
from meep_gpu.pml import PML
from meep_gpu.test_triton_kernels import GUARD_SPELLING, PACKAGE_DIR, code_of

MODULE_PATH = PACKAGE_DIR / "offdiag_update_e.py"
MODULE_NAME = "meep_gpu.triton_kernels.offdiag_update_e"
PARITY_DIR = pathlib.Path(__file__).resolve().parents[1] / "parity" / "meep_gpu"
GATE_PATH = PARITY_DIR / "gate_triton_offdiag.py"
COMPOSITION_PATH = PARITY_DIR / "probe_triton_offdiag_composition.py"
SLURM_PATH = PARITY_DIR / "run_triton_offdiag.slurm"

E_NAMES = ("Ex", "Ey", "Ez")

#: The oracle tensor test_tensor_epsilon._EPS_TENSOR carries (symmetric,
#: positive definite, every off-diagonal entry nonzero).
EPS_TENSOR = numpy.array([
    [2.0, 0.35, 0.20],
    [0.35, 2.5, 0.15],
    [0.20, 0.15, 3.0],
], dtype=numpy.float64)


@pytest.fixture(scope="module")
def od():
    """The shipped module itself must be importable without a test-only stub."""
    return importlib.import_module(MODULE_NAME)


def _tensor_rows(shape, scale=1.0, varying_seed=None):
    """(diagonal epsilon volumes, inverse volumes, offdiag rows) from EPS_TENSOR.

    ``varying_seed`` multiplies every row entry by a seeded smooth field so the
    coefficient-between-shifts association is byte-visible (a uniform
    coefficient forgives the hoist)."""
    inverse = numpy.linalg.inv(EPS_TENSOR)
    epsilon = {name: numpy.full(shape, 1.0 / inverse[i, i], dtype=numpy.float32)
               for i, name in enumerate(E_NAMES)}
    inv_eps = {name: numpy.full(shape, inverse[i, i], dtype=numpy.float32)
               for i, name in enumerate(E_NAMES)}
    modulation = 1.0
    if varying_seed is not None:
        rng = numpy.random.default_rng(varying_seed)
        modulation = rng.uniform(0.5, 1.5, size=shape).astype(numpy.float32)
    rows = {
        row: {partner: (numpy.full(shape, inverse[i, j], dtype=numpy.float32)
                        * numpy.float32(scale) * modulation
                        ).astype(numpy.float32)
              for j, partner in enumerate(E_NAMES) if j != i}
        for i, row in enumerate(E_NAMES)
    }
    return epsilon, inv_eps, rows


def _build(cell_size=(0.8, 0.8, 0.8), dimensions=3, boundaries=None,
           pml_thickness=2, complex_storage=False, storage=True,
           k_point=(0.0, 0.0, 0.0), beta=0.0, courant=0.35,
           rows="full", seed=11, thickness=None, varying_seed=None,
           **grid_kwargs):
    """A real off-diagonal Grid/Fields/PML triple on NumPy.

    Default: the family's target class — real storage, an active PML, the full
    uniform inverse tensor installed, k = 0, no fold. Every refusal test
    perturbs exactly one clause off this. ``rows`` selects the install:
    'full' (all six entries), 'ez_ex' / 'ey_ex' / 'ez_ey' (one row, one
    partner — the last two are the second-partner-only slots R12/R22, the
    kernel's two else-arm specializations no other single form compiles),
    'zeros' (explicit zero rows — the install drops them), None (diagonal
    only).
    """
    grid = Grid(resolution=10.0, cell_size=cell_size, dimensions=dimensions,
                boundaries=boundaries, courant=courant, k_point=k_point,
                beta=beta, **grid_kwargs)
    fields = Fields(grid=grid, force_complex_fields=complex_storage)
    shape = tuple(grid.shape)
    epsilon, inv_eps, full_rows = _tensor_rows(shape, varying_seed=varying_seed)
    if rows == "full":
        installed = full_rows
    elif rows == "ez_ex":
        installed = {"Ez": {"Ex": full_rows["Ez"]["Ex"]}}
    elif rows == "ey_ex":
        installed = {"Ey": {"Ex": full_rows["Ey"]["Ex"]}}
    elif rows == "ez_ey":
        installed = {"Ez": {"Ey": full_rows["Ez"]["Ey"]}}
    elif rows == "zeros":
        installed = {row: {partner: numpy.zeros(shape, dtype=numpy.float32)
                           for partner in partners}
                     for row, partners in full_rows.items()}
    elif rows is None:
        installed = None
    else:
        raise ValueError(rows)
    fields.set_epsilon_volumes(epsilon, inv_eps,
                               chi1inv_offdiagonal=installed)
    if storage:
        fields.enable_pml_storage()
        rng = numpy.random.default_rng(seed)
        for name in ("Dx", "Dy", "Dz", "Ex", "Ey", "Ez",
                     "f_w_Ex", "f_w_Ey", "f_w_Ez"):
            getattr(fields, name)[...] = rng.uniform(
                -0.4, 0.4, size=shape).astype(numpy.float32)
    if thickness is None:
        thickness = tuple(
            (pml_thickness, pml_thickness) if shape[axis] >= 6 else (0, 0)
            for axis in range(3))
    return fields, PML(grid=grid, thickness=thickness)


def _reasons(od, fields, pml):
    verdict = od.offdiag_constitutive_coverage(fields, pml)
    return [r for r in verdict.reasons if "array module" not in r]


# ---------------------------------------------------------------------------
# The optional dependency must stay optional; the restatements must not drift
# ---------------------------------------------------------------------------

def test_the_package_does_not_import_the_module_eagerly():
    for name in list(sys.modules):
        if name == MODULE_NAME:
            del sys.modules[name]
    importlib.import_module("meep_gpu.triton_kernels")
    eagerly_imported = MODULE_NAME in sys.modules
    importlib.import_module(MODULE_NAME)  # restore for the other tests
    assert not eagerly_imported, (
        "the package __init__ must not pull this unwired module in")


def test_the_module_answers_coverage_but_the_kernel_fails_clearly_without_triton(od):
    if od.offdiag_constitutive_step is not None:
        pytest.skip("triton is installed here; the absence arm is exercised on "
                    "the merge-bar box without it")
    fields, pml = _build()
    assert od.offdiag_constitutive_coverage(fields, pml).reasons
    with pytest.raises(ImportError, match="triton"):
        od.offdiag_constitutive_step_kernel()


def test_the_restated_constants_pin_their_originators(od):
    def value(x):
        return getattr(x, "value", x)

    assert value(od.PERIODIC) == 0 and value(od.METALLIC) == 1
    assert od.DEFAULT_BLOCK == 256
    kernels_source = (PACKAGE_DIR / "kernels.py").read_text(encoding="utf-8")
    assert "\nPERIODIC = tl.constexpr(0)\n" in kernels_source
    assert "\nMETALLIC = tl.constexpr(1)\n" in kernels_source
    assert "\nDEFAULT_BLOCK = 256\n" in kernels_source
    assert od.BOUNDARY_CODES == {"periodic": 0, "metallic": 1}


def test_the_term_table_is_steppings_own(od):
    """E_TERMS restates stepping.E_CONSTITUTIVE_TERMS with the axis as an index."""
    axis_of = {"x": 0, "y": 1, "z": 2}
    assert od.E_TERMS == tuple(
        (target, source, axis_of[axis_name])
        for target, source, axis_name in stepping.E_CONSTITUTIVE_TERMS)
    assert od.HALF_INTEGER is True  # stepping.py:1015, half_integer=True


def test_the_partner_table_and_row_slots_are_the_cycle_direction_rule(od):
    """cycle_direction X -> Y -> Z: own + 1 first, own + 2 second
    (stepping.py:1235-1237) — Ez takes the Ex partner then the Ey partner.
    Unlike the nonlinear family's commutative Dsqr, here the slot order BINDS
    coefficients to partner volumes, so both tables are pinned."""
    assert od.TRANSVERSE_PARTNERS == tuple(
        ((own + 1) % 3, (own + 2) % 3) for own in range(3))
    axes = "xyz"
    expected = []
    for own, row in enumerate(E_NAMES):
        for offset in (1, 2):
            expected.append((row, "E" + axes[(own + offset) % 3]))
    assert od.ROW_SLOTS == tuple(expected)


def test_the_wall_mask_axis_table_is_iyee_shifts(od):
    """_mask_metallic_wall_coupling zeroes face 0 of every axis where the
    component's Yee shift is 0 (stepping.py:1279-1283); the table restates
    IYEE_SHIFTS and must track it."""
    assert od.WALL_MASK_AXES == tuple(
        tuple(axis for axis in range(3) if IYEE_SHIFTS[name][axis] == 0)
        for name in E_NAMES)


def test_the_shared_clause_builders_this_predicate_composes_from_still_exist(od):
    from meep_gpu.triton_kernels import coverage

    for name in od.SHARED_CLAUSES:
        assert callable(getattr(coverage, name)), (
            f"coverage.{name} was renamed or removed; the predicate composes "
            f"from it and would silently drop a clause")


def test_the_module_docstring_states_dispatch_is_untouched(od):
    assert "NOT WIRED" in od.__doc__
    assert "fast-path hook" in od.__doc__ or "plan_fast_path" in od.__doc__


def test_the_module_does_not_reach_into_another_tracks_file():
    code = code_of(MODULE_PATH)
    assert "fastpath" not in code
    assert "cuda_kernels" not in code


def test_the_guard_is_the_packages_one_spelling_and_appears_once():
    code = MODULE_PATH.read_text(encoding="utf-8")
    assert code.count("enable_fp_fusion=") == 1  # the one plan run() method
    assert code.count(GUARD_SPELLING) == 1


def test_no_fingerprint_entry_is_claimed():
    """No GATE entry for this tranche in the table's ledger — the boundary.

    It used to forbid the STRING anywhere in the file, which is a stricter
    spelling than the sibling tranches use (``test_triton_bfast`` and
    ``test_triton_special_kz`` both scan the top-level KEYS), and stricter than the
    boundary itself: what must not happen is this tranche claiming a gate record it
    did not cut. The shared ``family_recert_2026-08-14`` transcription is the one
    place the module may be CLAIMED, and it exists because the alternative was
    measured to be worse — the dispatcher pointed at this family's provenance by a
    path into a gitignored results directory, so every run artifact that dispatched
    it named a family, a dead path and nothing else. The digest under that key is
    additionally what makes a drift in this module VISIBLE: no ``host_sha256``
    entry covers it. Anywhere else the module may be NAMED only as the import record
    of a run an entry describes, never by a weld (see the third narrowing below).
    """
    import json

    from meep_gpu.test_triton_weld_contract import _welds

    recorded = json.loads((PACKAGE_DIR / "fingerprints.json")
                          .read_text(encoding="utf-8"))
    assert not any("offdiag_update_e" in name for name in recorded), (
        "this tranche's provenance lives in the gate's results directory, "
        "not in a gate entry of the table's fingerprints.json")
    # THE EXACT PATH, not the bare substring. Measured 2026-08-19: the substring
    # spelling also matched a SIBLING tranche's file — folded_offdiag_update_e.py
    # contains "offdiag_update_e" — so welding the folded off-diagonal family's own
    # gate result tripped this guard, which is not the boundary it defends. The
    # docstring above already records one narrowing for exactly this reason (the
    # string spelling being "stricter than the boundary itself"); this is the same
    # correction one step further. What is still forbidden is unchanged: THIS
    # tranche's module claimed by an entry that did not cut it.
    #
    # AND NAMED IS NOT CLAIMED, the third narrowing (2026-10-04). Since 2026-09-30
    # ``bit_identity_gate`` binds every ``meep_gpu/`` file its identity run IMPORTED
    # (``rebind_triton_welds.bind_bit_identity``), and that run imports the whole
    # Triton package, this module included -- so the bytes are named because the run
    # executed them, which is the opposite of a claim the entry did not cut. What a
    # claim IS in this ledger is a WELD: an entry with ``status: PASS`` over a
    # ``source_sha256`` map, the set ``test_triton_weld_contract._welds`` enumerates and
    # the one a "which weld pins this module?" lookup walks. So outside the shared
    # family-recert transcription: no weld may name this module at all, and any other
    # entry may name it ONLY as a key of its own ``source_sha256``. That such a map is
    # exactly the import record of a run the entry describes is read from the run's
    # artifact, by the next test.
    naming = _entries_naming_this_module(recorded)
    claimed = sorted(key for key in naming if key in _welds(recorded))
    assert not claimed, (
        f"{claimed} are welds (status PASS over a source_sha256 map) that name this "
        "tranche's module: a weld claims the bytes it pins, and this tranche's "
        "provenance lives in its gate's results directory")
    for key, found in sorted(naming.items()):
        assert found == [(("source_sha256", GUARDED_MODULE), "key")], (
            f"{key} names this tranche's module at {found}; outside the family-recert "
            "transcription it may appear only as a key of an entry's own "
            "source_sha256, as the import record of a run that entry describes")


#: The path ``test_no_fingerprint_entry_is_claimed`` guards, spelled as the ledger
#: keys it (repository-relative).
GUARDED_MODULE = "meep_gpu/triton_kernels/offdiag_update_e.py"


def _entries_naming_this_module(recorded):
    """``{entry key: [(trail, "key" | "value"), ...]}`` for every entry outside the shared
    family-recert transcription in which :data:`GUARDED_MODULE` appears, as a mapping key
    or inside a string value, at any depth."""
    def occurrences(node, trail=()):
        if isinstance(node, dict):
            for key, value in node.items():
                if GUARDED_MODULE in str(key):
                    yield trail + (key,), "key"
                yield from occurrences(value, trail + (key,))
        elif isinstance(node, list):
            for index, value in enumerate(node):
                yield from occurrences(value, trail + (index,))
        elif isinstance(node, str) and GUARDED_MODULE in node:
            yield trail, "value"

    naming = {key: sorted(occurrences(value), key=repr)
              for key, value in recorded.items()
              if key != "family_recert_2026-08-14" and isinstance(value, (dict, list, str))}
    return {key: found for key, found in naming.items() if found}


def test_an_entry_naming_this_module_pins_the_import_record_of_its_own_run():
    """Named because it ran, verified against what ran.

    The third narrowing in ``test_no_fingerprint_entry_is_claimed`` lets an entry that
    is not a weld name this module as a key of its own ``source_sha256``. What makes
    that the record of an execution rather than a claim is checked here, from the
    artifact: the map is EXACTLY the ``meep_gpu/`` part of the ``imported_source_sha256``
    of a run the entry records, in the artifact that run's ``records`` names and its
    ``artifact_sha256`` pins -- which is the rule ``rebind_triton_welds.bind_bit_identity``
    writes ``bit_identity_gate``'s map by. A ledger in which nothing outside the
    family-recert transcription names the module has nothing to check here.

    Declared under ``[evidence_archive]`` in ``tools/ci/declared_resources.txt``: it
    reads the run's artifact, which a checkout holds only with the archive restored.
    """
    import hashlib
    import json

    from conftest import requires_resource_skip

    from meep_gpu import fastpath

    recorded = json.loads((PACKAGE_DIR / "fingerprints.json")
                          .read_text(encoding="utf-8"))
    naming = _entries_naming_this_module(recorded)
    root = PACKAGE_DIR.parent.parent
    if naming and not (root / "parity" / "meep_gpu" / "results").is_dir():
        requires_resource_skip(
            "parity_meep_gpu_results",
            f"{sorted(naming)} name this module in their source_sha256, and whether "
            "that map is the import record of their own run is read from the run's "
            "artifact; parity/meep_gpu/results is gitignored and absent here")
    for key in sorted(naming):
        entry = recorded[key]
        pins = entry.get("source_sha256") or {}
        matched = []
        for capability, run in sorted((entry.get(fastpath.RUNS) or {}).items()):
            words = str(run.get("records") or "").split()
            if not words or not run.get("artifact_sha256"):
                continue
            spelled = words[0].rstrip("/")
            spelled = spelled[len("apps/api/"):] if spelled.startswith("apps/api/") else spelled
            artifact = root / spelled
            if artifact.is_dir():
                artifact = artifact / "gate.json"
            assert artifact.is_file(), (
                f"{key} runs[{capability!r}] names {words[0]}, which is not here while "
                "the results tree is")
            assert hashlib.sha256(artifact.read_bytes()).hexdigest() == \
                run["artifact_sha256"], f"{key} runs[{capability!r}]: artifact moved"
            imported = json.loads(artifact.read_text(encoding="utf-8")).get(
                "imported_source_sha256") or {}
            if {name: digest for name, digest in imported.items()
                    if name.startswith("meep_gpu/")} == pins:
                matched.append(capability)
        assert matched, (
            f"{key} pins this tranche's module, and no run it records imported exactly "
            "its meep_gpu/ pin set: the pin is not the import record of a run the "
            "entry describes")


def test_the_gate_and_its_composition_probe_exist_beside_the_other_tranches():
    assert GATE_PATH.exists()
    assert COMPOSITION_PATH.exists()


# ---------------------------------------------------------------------------
# The positive verdict — refused only for the backend on this laptop
# ---------------------------------------------------------------------------

def test_the_oracle_configuration_is_refused_only_for_the_backend(od):
    """The uniform full tensor with PML — test_tensor_epsilon's 'pml' family."""
    fields, pml = _build()
    assert _reasons(od, fields, pml) == []


def test_the_metallic_and_mixed_boundary_classes_are_admitted(od):
    fields, pml = _build(boundaries="metallic")
    assert _reasons(od, fields, pml) == []
    fields, pml = _build(boundaries=("periodic", "metallic", "periodic"))
    assert _reasons(od, fields, pml) == []


def test_a_single_row_single_partner_install_is_admitted(od):
    fields, pml = _build(rows="ez_ex")
    assert fields.chi1inv_offdiagonal_for("Ez").keys() == {"Ex"}
    assert _reasons(od, fields, pml) == []


def test_an_isotropic_aliased_inverse_epsilon_is_admitted(od):
    """Three aliases of one array are the isotropic install; the kernel binds
    three pointers either way (the certified predicate's own stance)."""
    fields, pml = _build()
    shared = numpy.full(fields.grid.shape, 0.5, dtype=numpy.float32)
    epsilon = numpy.full(fields.grid.shape, 2.0, dtype=numpy.float32)
    _, _, rows = _tensor_rows(tuple(fields.grid.shape))
    fields.set_epsilon_volumes({c: epsilon for c in E_NAMES},
                               {c: shared for c in E_NAMES},
                               chi1inv_offdiagonal=rows)
    assert _reasons(od, fields, pml) == []


# ---------------------------------------------------------------------------
# The refusals, one per silent-wrong-answer surface
# ---------------------------------------------------------------------------

def test_zero_rows_are_refused_and_the_two_predicates_are_disjoint(od):
    """The INVERTED clause: explicit zero rows are dropped at install
    (fields.py:1302-1303), the run is the diagonal engine's bit for bit, and
    it belongs to constitutive_coverage(side='E') — never to both predicates
    at once."""
    from meep_gpu.triton_kernels import coverage

    fields, pml = _build(rows="zeros")
    assert not fields.has_offdiagonal_epsilon
    reasons = _reasons(od, fields, pml)
    assert any("no off-diagonal chi1inv row" in r for r in reasons), reasons
    plain = [r for r in coverage.constitutive_coverage(fields, pml, "E").reasons
             if "array module" not in r]
    assert plain == [], plain  # the certified kernel's side of the split

    live_fields, live_pml = _build()
    assert _reasons(od, live_fields, live_pml) == []
    plain_on_offdiag = coverage.constitutive_coverage(live_fields, live_pml, "E")
    assert any("off-diagonal" in r for r in plain_on_offdiag.reasons), (
        "the shipped predicate stopped refusing off-diagonal rows — the two "
        "update_E predicates now overlap and plan_step would pick by ordering")


def test_the_certified_curl_and_h_predicates_admit_the_offdiag_run(od):
    """The refusal is deliberately per-sub-step: the curls and update_H admit
    an offdiag run (coverage.py:26-31, :343-346), so this kernel is the LAST
    uncovered sub-step of such a run — the composition probe's premise."""
    from meep_gpu.triton_kernels import coverage

    fields, pml = _build()
    for sub_step in ("step_B", "step_D"):
        reasons = [r for r in coverage.pml_curl_coverage(
            fields, pml, sub_step).reasons if "array module" not in r]
        assert reasons == [], (sub_step, reasons)
    reasons = [r for r in coverage.constitutive_coverage(
        fields, pml, "H").reasons if "array module" not in r]
    assert reasons == [], reasons


def test_chi2_chi3_is_refused_toward_the_nonlinear_family(od):
    """MEEP's most-general case scales the whole row product by the Pade
    factor (stepping.py:1095-1116) — a later fused leg, refused here by the
    shared clause, which this test proves is load-bearing: nothing else
    refuses the run."""
    fields, pml = _build()
    fields.set_nonlinear_volumes({}, {c: 0.08 for c in E_NAMES})
    reasons = _reasons(od, fields, pml)
    assert any("chi2/chi3" in r for r in reasons), reasons
    assert [r for r in reasons if "chi2/chi3" not in r] == [], (
        "the chi clause is not the only refusal — this test no longer "
        "demonstrates it is load-bearing")


def test_no_active_layer_is_refused(od):
    """Without PML the offdiag update_E is a REAL stored write
    (field[...] = constitutive, stepping.py:1019-1022 — stores_E is forced by
    the install, fields.py:1254-1255), a different sub-step Phase A refuses."""
    fields, pml = _build(pml_thickness=0)
    assert fields.stores_E  # the no-PML sub-step is reachable, not a no-op
    reasons = _reasons(od, fields, pml)
    assert any("no active PML" in r for r in reasons), reasons


def test_complex_storage_is_refused_as_phase_b(od):
    fields, pml = _build(complex_storage=True)
    reasons = _reasons(od, fields, pml)
    assert any("force_complex_fields" in r for r in reasons), reasons


def test_a_nonzero_k_point_is_refused(od):
    fields, pml = _build(k_point=(0.3, 0.0, 0.0))
    reasons = _reasons(od, fields, pml)
    assert any("k_point" in r for r in reasons), reasons


def test_beta_is_refused_by_the_predicate_and_by_the_engine_itself(od):
    """TWO distinct facts, asserted separately: the predicate refuses beta
    (the shared clause), AND stepping's own curl pass raises for
    real-storage + offdiag + beta (stepping.py:800-810, MEEP
    fields.cpp:548-549) — the engine-level refusal the refusals leg must
    distinguish from the predicate's."""
    fields, pml = _build(cell_size=(0.8, 0.8, 0.0), dimensions=2, beta=0.25)
    reasons = _reasons(od, fields, pml)
    assert any("beta" in r for r in reasons), reasons
    with pytest.raises(ValueError, match="off-diagonal epsilon needs complex"):
        stepping.step_B(fields, pml)


def test_bfast_is_refused(od):
    fields, pml = _build(bfast_scaled_k=(0.5, 0.0, 0.0))
    reasons = _reasons(od, fields, pml)
    assert any("BFAST" in r for r in reasons), reasons


def test_a_fold_is_refused_by_the_predicate_while_the_engine_steps_it(od):
    """THE STALE-DOCSTRING GUARD. stepping.py:1219-1220 and
    fields.py:1237-1241 claim the INSTALL refuses folded+rows; the code does
    not (fields.py:1262-1310 installs unchanged; fold-equivalence measured
    8.3e-13..4.7e-12, pinned by test_tensor_epsilon). The refusal is THIS
    kernel family's, and the run must remain steppable by the array path."""
    fields, pml = _build(symmetry=("X",),
                         thickness=((0, 2), (2, 2), (2, 2)))
    assert fields.has_offdiagonal_epsilon, (
        "the install refused folded rows — the stale docstrings came true and "
        "this family's fold story (and test_tensor_fold_equivalence) changed "
        "underfoot")
    reasons = _reasons(od, fields, pml)
    assert any("mirror" in r or "folded" in r for r in reasons), reasons
    stepping.update_E(fields, pml)  # the array path steps it without complaint


def test_a_registered_polarization_is_refused_by_name(od):
    fields, pml = _build()
    fields.polarizations.append(SimpleNamespace(
        driven=lambda: ("Ez",), drives=lambda name: name == "Ez"))
    reasons = _reasons(od, fields, pml)
    assert any("susceptibility is registered" in r for r in reasons), reasons


def test_cylindrical_is_refused(od):
    fields, pml = _build()
    fields.grid.cylindrical = True
    reasons = _reasons(od, fields, pml)
    assert any("cylindrical" in r for r in reasons), reasons


def test_a_scalar_inverse_epsilon_is_refused_by_name(od):
    """This family carries NO scalar arm at all — the shared clause is the
    refusal (coverage.py:649-674)."""
    fields, pml = _build()
    fields._inv_eps_components["Ez"] = 0.5
    reasons = _reasons(od, fields, pml)
    assert any("scalar, not a volume" in r for r in reasons), reasons


def test_a_malformed_row_volume_is_refused_by_name(od):
    """A row planted past the installer (wrong dtype) must be refused, not
    stepped — this package's rule against inferring coverage from another
    module's guard."""
    fields, pml = _build()
    fields._chi1inv_offdiagonal["Ex"]["Ey"] = numpy.full(
        fields.grid.shape, 0.1, dtype=numpy.float64)
    reasons = _reasons(od, fields, pml)
    assert any("chi1inv_offdiag[Ex][Ey]" in r and "float32" in r
               for r in reasons), reasons


def test_missing_storage_is_refused(od):
    fields, pml = _build(storage=False, pml_thickness=0)
    # The install forces stores_E (fields.py:1254-1255), so un-store it the
    # way only a planted defect could, to prove the clause answers by name.
    fields._stored_E = False
    reasons = _reasons(od, fields, pml)
    assert any("is not allocated" in r for r in reasons), reasons
    assert any("recomputed from D" in r for r in reasons), reasons


def test_a_diagonal_key_row_planted_past_the_installer_is_refused_not_crashed(od):
    """THE PREDICATE-BUILDER CONTRACT (None means REFUSED, offdiag_update_e.py
    plan builder docstring): a row planted past the installer under a DIAGONAL
    key sets ``has_offdiagonal_epsilon`` while every one of the six ROW_SLOTS
    is dead — the flag-based clause (b) admitted this with ZERO reasons and
    the builder then raised 'no row slot survives'. The slot-counted clause
    must refuse it, and the engine builder must answer None, not raise."""
    fields, pml = _build(rows=None)
    fields._chi1inv_offdiagonal["Ex"] = {"Ex": numpy.full(
        fields.grid.shape, 0.1, dtype=numpy.float32)}
    assert fields.has_offdiagonal_epsilon  # the flag alone would admit
    reasons = _reasons(od, fields, pml)
    assert any("no off-diagonal chi1inv row" in r for r in reasons), reasons
    assert od.plan_offdiagonal_constitutive(fields, pml) is None


def test_a_row_installed_as_an_alias_of_an_output_is_refused_not_crashed(od):
    """The alias is REACHABLE THROUGH THE PUBLIC INSTALLER: set_epsilon_volumes
    keeps the caller's array without copying (asarray + astype(copy=False),
    fields.py:1296), so a row that IS fields.Ex arrives legally installed.
    The builder refuses the alias with a raise (schedule-dependent answer);
    the predicate must therefore refuse it FIRST, by name, and the engine
    builder must answer None."""
    fields, pml = _build()
    epsilon, inv_eps, _rows = _tensor_rows(tuple(fields.grid.shape))
    fields.set_epsilon_volumes(epsilon, inv_eps,
                               chi1inv_offdiagonal={"Ez": {"Ex": fields.Ex}})
    installed = fields.chi1inv_offdiagonal_for("Ez")["Ex"]
    assert installed is fields.Ex, (
        "the installer copied the row — the alias hazard this test pins is "
        "gone and both the test and the predicate clause should be revisited")
    reasons = _reasons(od, fields, pml)
    assert any("alias" in r and "chi1inv_offdiag[Ez][Ex]" in r
               for r in reasons), reasons
    assert od.plan_offdiagonal_constitutive(fields, pml) is None


# ---------------------------------------------------------------------------
# Predicate mutations: each added clause is load-bearing
# ---------------------------------------------------------------------------

def _mutated_predicate(od, needle: str, replacement: str):
    import inspect
    import textwrap

    source = textwrap.dedent(
        inspect.getsource(od.offdiag_constitutive_coverage))
    assert needle in source, f"needle not found: {needle!r}"
    namespace = dict(vars(od))
    exec(compile(source.replace(needle, replacement), "<mutated>", "exec"),
         namespace)
    return namespace["offdiag_constitutive_coverage"]


def test_mutation_dropping_the_polarization_clause_admits_the_ade_configuration(od):
    fields, pml = _build()
    fields.polarizations.append(SimpleNamespace(
        driven=lambda: ("Ez",), drives=lambda name: name == "Ez"))
    mutated = _mutated_predicate(
        od, "if states or getattr(fields, \"has_polarizations\", False):",
        "if False:")
    kept = [r for r in mutated(fields, pml).reasons if "array module" not in r]
    # _susceptibility_reasons still names the SimpleNamespace's missing kind;
    # the load-bearing question is whether the D-vs-(D - sum P) source clause
    # survives, so filter the kind/shape reasons the shared helper contributes.
    kept = [r for r in kept if "kind" not in r]
    assert kept == [], (
        "with the polarization clause dropped nothing else refuses this run — "
        "so the clause is load-bearing and this mutation would silently cover "
        "the dispersive+offdiag configuration")


def test_mutation_dropping_the_inverted_row_clause_breaks_disjointness(od):
    fields, pml = _build(rows="zeros")
    mutated = _mutated_predicate(
        od, "if not any(value is not None for value in row_volumes_for(fields)):",
        "if False:")
    kept = [r for r in mutated(fields, pml).reasons if "array module" not in r]
    assert kept == [], (
        "with the inverted row clause dropped the zero-row run is admitted — "
        "both update_E predicates would then cover it")


# ---------------------------------------------------------------------------
# Transcription: the in-test reference against stepping.update_E itself
# ---------------------------------------------------------------------------

def _face(axis, index):
    face = [slice(None)] * 3
    face[axis] = index
    return tuple(face)


def _shift_down(field, axis, boundary):
    """f[i-1]: stepping._shift_down's plain PERIODIC/METALLIC branches."""
    rolled = numpy.roll(field, 1, axis=axis)
    if boundary == "metallic":
        rolled[_face(axis, 0)] = 0
    return rolled


def _shift_up(field, axis, boundary):
    """f[i+1]: stepping._shift_up's plain PERIODIC/METALLIC branches."""
    rolled = numpy.roll(field, -1, axis=axis)
    if boundary == "metallic":
        rolled[_face(axis, -1)] = 0
    return rolled


def _reference_coupling(volumes, rows, own_axis, boundaries, wall_axes,
                        component):
    """stepping._offdiagonal_terms (:1206-1224) + the wall mask (:1227-1254):
    pair down the partner axis, coefficient multiply BETWEEN the shifts, the
    PRODUCT up the own axis, 0.25 applied last, offset 1 then offset 2, then
    face-0 zeroing on every wall-masked axis with iyee 0."""
    axes = "xyz"
    total = None
    for offset in (1, 2):
        partner_axis = (own_axis + offset) % 3
        partner = "E" + axes[partner_axis]
        coefficient = rows.get(partner)
        if coefficient is None:
            continue
        values = volumes[partner]
        pair = values + _shift_down(values, partner_axis,
                                    boundaries[partner_axis])
        product = pair * coefficient
        term = 0.25 * (product + _shift_up(product, own_axis,
                                           boundaries[own_axis]))
        total = term if total is None else total + term
    if total is not None:
        iyee = IYEE_SHIFTS[component]
        for axis in range(3):
            if iyee[axis] != 0:
                continue
            if wall_axes[axis]:
                total[_face(axis, 0)] = 0
    return total


def _reference_update_e(state, coefficients, rows_by_component, inv_eps,
                        boundaries, wall_axes):
    """The gate reference's twin: the whole offdiag sub-step on plain dicts
    (stepping.py:1001-1008 + the PML tail's no-scratch branch, :2130-2135)."""
    volumes = {"Ex": state["Dx"], "Ey": state["Dy"], "Ez": state["Dz"]}
    for own_axis, target in enumerate(E_NAMES):
        gs = volumes[target]
        us = inv_eps[target]
        constitutive = gs * us
        coupling = _reference_coupling(volumes, rows_by_component[target],
                                       own_axis, boundaries, wall_axes, target)
        if coupling is not None:
            constitutive = constitutive + coupling
        kps, kms = coefficients[target]
        fw = state["f_w_" + target]
        field = state[target]
        fw_previous = fw.copy()
        fw[...] = constitutive
        field += kps * fw
        field -= kms * fw_previous


def _state_of(fields):
    return {name: getattr(fields, name).copy()
            for name in ("Dx", "Dy", "Dz", "Ex", "Ey", "Ez",
                         "f_w_Ex", "f_w_Ey", "f_w_Ez")}


def _wall_axes_of(grid):
    return tuple(
        int(grid.is_metallic(axis) and not grid.is_mirrored(axis))
        for axis in range(3))


def _pin_against_stepping(od, fields, pml, steps=3):
    state = _state_of(fields)
    boundaries = tuple(stepping._boundary_kinds(fields.grid, pml))
    wall_axes = _wall_axes_of(fields.grid)
    coefficients = {
        target: (getattr(pml, f"kps_{axis}_h"), getattr(pml, f"kms_{axis}_h"))
        for target, _source, axis in stepping.E_CONSTITUTIVE_TERMS}
    rows_by_component = {name: fields.chi1inv_offdiagonal_for(name)
                         for name in E_NAMES}
    inv_eps = {name: fields.inverse_epsilon_for(name) for name in E_NAMES}
    for step in range(steps):
        stepping.update_E(fields, pml)
        _reference_update_e(state, coefficients, rows_by_component, inv_eps,
                            boundaries, wall_axes)
        for name in ("Ex", "Ey", "Ez", "f_w_Ex", "f_w_Ey", "f_w_Ez"):
            ours = state[name]
            theirs = getattr(fields, name)
            assert ours.dtype == theirs.dtype == numpy.float32
            assert ours.tobytes() == theirs.tobytes(), (
                f"{name} diverged from stepping.update_E at sub-step call "
                f"{step + 1}: max abs delta "
                f"{numpy.max(numpy.abs(ours - theirs)):.3e}")


def test_the_reference_matches_stepping_on_the_uniform_tensor_pml_class(od):
    fields, pml = _build()
    _pin_against_stepping(od, fields, pml)


def test_the_reference_matches_stepping_on_metallic_walls(od):
    """The wall-coupling mask live on every transverse axis."""
    fields, pml = _build(boundaries="metallic")
    _pin_against_stepping(od, fields, pml)


def test_the_reference_matches_stepping_on_mixed_boundaries_varying_rows(od):
    """Spatially varying coefficients — the between-shifts association's
    byte-visible class — on mixed boundaries."""
    fields, pml = _build(boundaries=("periodic", "metallic", "periodic"),
                         varying_seed=31)
    _pin_against_stepping(od, fields, pml)


def test_the_reference_matches_stepping_on_a_single_row_install(od):
    fields, pml = _build(rows="ez_ex", boundaries="metallic")
    _pin_against_stepping(od, fields, pml)


def test_the_reference_matches_stepping_on_the_second_partner_only_installs(od):
    """R12-only (Ey<-Ex) and R22-only (Ez<-Ey): the two single-slot forms
    whose kernel specializations execute the else-arm blocks NO other single
    form compiles (offdiag_update_e.py components 1 and 2). Pinned against
    stepping here so their transcription cannot drift dark on the laptop;
    the gate's sweep carries the matching device rows. Each runs with the
    metallic axis playing both its live roles for that row: the partner-axis
    down-shift ghost and the wall-coupling mask."""
    fields, pml = _build(rows="ey_ex",
                         boundaries=("metallic", "periodic", "periodic"))
    _pin_against_stepping(od, fields, pml)
    fields, pml = _build(rows="ez_ey",
                         boundaries=("periodic", "metallic", "periodic"))
    _pin_against_stepping(od, fields, pml)


def test_the_reference_matches_stepping_on_a_reduced_2d_grid(od):
    """(n, n, 1): the invariant axis (a real Grid collapses z — only z may be
    zero) wraps onto itself and the partner pair there is g + g (MEEP's
    stride(d)=0 double-read), NOT the curl's exact zero. The gate's raw-array
    sweep additionally carries the (1, 160, 160) leading-axis form the
    from_arrays route can express."""
    fields, pml = _build(cell_size=(1.6, 1.6, 0.0), dimensions=2)
    assert fields.grid.shape[2] == 1
    _pin_against_stepping(od, fields, pml)


def test_the_invariant_axis_pair_is_double_not_zero(od):
    fields, pml = _build(cell_size=(1.6, 1.6, 0.0), dimensions=2)
    g = numpy.asarray(getattr(fields, "Dx"))
    pair = g + _shift_down(g, 2, "periodic")
    assert pair.tobytes() == (g + g).tobytes()


def test_the_between_shifts_association_differs_from_the_hoist_on_varying_rows(od):
    """The one genuinely new arithmetic element: (pair*u) shifted vs the
    four-point-average-times-u[i] hoist. On a VARYING coefficient the two are
    different ALGEBRA (the half-cell registration error). On a UNIFORM
    coefficient they are algebraically equal but MEASURED bitwise-different
    all the same — distributivity (a*u + b*u vs (a+b)*u) is not a bitwise
    identity in f32 — refining the plan's predicted null for m1's uniform
    case into a recorded-outcome row: the transcription must still carry the
    between-shifts form, and the sweep must still carry varying volumes,
    because only the varying case pins the REGISTRATION rather than the
    rounding."""
    fields, pml = _build(varying_seed=47)
    volumes = {n: numpy.asarray(getattr(fields, "D" + n[1])) for n in E_NAMES}
    rows = fields.chi1inv_offdiagonal_for("Ez")
    boundaries = ("periodic",) * 3

    def hoist(rows_map):
        total = None
        for offset in (1, 2):
            partner_axis = (2 + offset) % 3
            partner = "E" + "xyz"[partner_axis]
            coefficient = rows_map[partner]
            g = volumes[partner]
            four = (g + _shift_down(g, partner_axis, boundaries[partner_axis]))
            four = four + _shift_up(four, 2, boundaries[2])
            term = 0.25 * (four * coefficient)
            total = term if total is None else total + term
        return total

    transcribed = _reference_coupling(volumes, rows, 2, boundaries,
                                      (0, 0, 0), "Ez")
    assert hoist(rows).tobytes() != transcribed.tobytes(), (
        "the hoist coincided with the transcription on varying rows — the "
        "central subtlety is not byte-visible on this seed and the gate's m1 "
        "needle would be vacuous")

    uniform = {partner: numpy.full_like(value, 0.11)
               for partner, value in rows.items()}
    transcribed_uniform = _reference_coupling(volumes, uniform, 2, boundaries,
                                              (0, 0, 0), "Ez")
    assert hoist(uniform).tobytes() != transcribed_uniform.tobytes(), (
        "the two forms coincided bitwise on a uniform coefficient on this "
        "seed — the measured distributivity-rounding separation vanished; "
        "re-measure before trusting the gate's m1 uniform-case expectation")


def test_the_wall_mask_is_load_bearing_on_metallic_walls(od):
    """Dropping the mask from the reference must break byte equality with
    stepping on an all-metallic run (the 2.6e-02-class defect, measured on
    the uniform-tensor metallic oracle)."""
    fields, pml = _build(boundaries="metallic")
    state = _state_of(fields)
    boundaries = tuple(stepping._boundary_kinds(fields.grid, pml))
    coefficients = {
        target: (getattr(pml, f"kps_{axis}_h"), getattr(pml, f"kms_{axis}_h"))
        for target, _source, axis in stepping.E_CONSTITUTIVE_TERMS}
    rows_by_component = {name: fields.chi1inv_offdiagonal_for(name)
                         for name in E_NAMES}
    inv_eps = {name: fields.inverse_epsilon_for(name) for name in E_NAMES}
    stepping.update_E(fields, pml)
    # wall_axes forced dark — the m6-class defect, in the reference.
    _reference_update_e(state, coefficients, rows_by_component, inv_eps,
                        boundaries, (0, 0, 0))
    assert any(
        state[name].tobytes() != getattr(fields, name).tobytes()
        for name in ("Ex", "Ey", "Ez")), (
        "the unmasked reference still matched stepping — the wall mask is not "
        "byte-visible here and the gate's m6 needle would be vacuous")


def test_the_zero_row_seam_is_bit_identical_to_the_diagonal_engine(od):
    """Explicit zero rows reduce to the diagonal engine byte for byte (the
    install-time drop) — and the control is non-vacuous: live rows differ."""
    trivial, pml = _build(rows="zeros", seed=77)
    control, control_pml = _build(rows=None, seed=77)
    live, live_pml = _build(seed=77)
    stepping.update_E(trivial, pml)
    stepping.update_E(control, control_pml)
    stepping.update_E(live, live_pml)
    for name in ("Ex", "Ey", "Ez", "f_w_Ex", "f_w_Ey", "f_w_Ez"):
        assert getattr(trivial, name).tobytes() == getattr(control, name).tobytes()
    assert any(
        getattr(live, name).tobytes() != getattr(control, name).tobytes()
        for name in ("Ex", "Ey", "Ez")), (
        "the live-rows control did not separate from the diagonal run — the "
        "seam check is vacuous on this seed")


# ---------------------------------------------------------------------------
# Plan shapes — buildable on this laptop, launchable only on the device
# ---------------------------------------------------------------------------

def _plan_arrays(shape=(4, 3, 5), seed=9):
    rng = numpy.random.default_rng(seed)
    arrays = {}
    for name in ("Ex", "Ey", "Ez", "f_w_Ex", "f_w_Ey", "f_w_Ez",
                 "Dx", "Dy", "Dz"):
        arrays[name] = rng.uniform(-1, 1, size=shape).astype(numpy.float32)
    for name in E_NAMES:
        arrays["inv_eps_" + name] = rng.uniform(
            0.2, 0.9, size=shape).astype(numpy.float32)
    flat = {}
    for axis, n in zip("xyz", shape):
        flat["kps_" + axis] = rng.uniform(0.5, 1.0, size=n).astype(numpy.float32)
        flat["kms_" + axis] = rng.uniform(0.5, 1.0, size=n).astype(numpy.float32)
    rows = {"Ez": {"Ex": rng.uniform(-0.2, 0.2, size=shape)
                   .astype(numpy.float32)}}
    return arrays, flat, rows


def test_the_from_arrays_plan_reports_its_shape(od):
    arrays, flat, rows = _plan_arrays()
    plan = od.plan_offdiagonal_constitutive_from_arrays(
        arrays, flat, rows, (0, 1, 0), (0, 1, 0))
    assert plan.row_mask == (0, 0, 0, 0, 1, 0)  # Ez/Ex is slot 4 (ROW_SLOTS)
    assert plan.boundary_codes == (0, 1, 0)
    assert plan.wall_axes == (0, 1, 0)


def test_an_all_dead_row_mask_is_refused_toward_the_certified_kernel(od):
    arrays, flat, _rows = _plan_arrays()
    with pytest.raises(ValueError, match="certified plain constitutive"):
        od.plan_offdiagonal_constitutive_from_arrays(
            arrays, flat, {}, (0, 0, 0), (0, 0, 0))


def test_a_scalar_inverse_epsilon_is_refused_by_the_plan(od):
    arrays, flat, rows = _plan_arrays()
    arrays["inv_eps_Ez"] = 0.5
    with pytest.raises(ValueError, match="no scalar arm"):
        od.plan_offdiagonal_constitutive_from_arrays(
            arrays, flat, rows, (0, 0, 0), (0, 0, 0))


def test_an_output_aliasing_an_input_is_refused(od):
    arrays, flat, rows = _plan_arrays()
    arrays["Ex"] = arrays["Dx"]  # the schedule-dependent wrong answer
    with pytest.raises(ValueError, match="alias"):
        od.plan_offdiagonal_constitutive_from_arrays(
            arrays, flat, rows, (0, 0, 0), (0, 0, 0))


def test_a_row_volume_aliasing_an_output_is_refused(od):
    arrays, flat, rows = _plan_arrays()
    rows = {"Ez": {"Ex": arrays["f_w_Ez"]}}
    with pytest.raises(ValueError, match="alias"):
        od.plan_offdiagonal_constitutive_from_arrays(
            arrays, flat, rows, (0, 0, 0), (0, 0, 0))


def test_a_coefficient_vector_aliasing_an_output_is_refused(od):
    arrays, flat, rows = _plan_arrays()
    shape = arrays["Ex"].shape
    flat["kps_x"] = arrays["Ex"].ravel()[:shape[0]]  # view, same base address
    with pytest.raises(ValueError, match="alias"):
        od.plan_offdiagonal_constitutive_from_arrays(
            arrays, flat, rows, (0, 0, 0), (0, 0, 0))


def test_aliased_isotropic_inverse_epsilon_inputs_are_accepted(od):
    """Inputs may alias EACH OTHER: three aliases of one inverse-epsilon
    array are the isotropic install."""
    arrays, flat, rows = _plan_arrays()
    shared = arrays["inv_eps_Ex"]
    arrays["inv_eps_Ey"] = shared
    arrays["inv_eps_Ez"] = shared
    plan = od.plan_offdiagonal_constitutive_from_arrays(
        arrays, flat, rows, (0, 0, 0), (0, 0, 0))
    assert plan.row_mask == (0, 0, 0, 0, 1, 0)


def test_bad_boundary_codes_and_wall_axes_are_refused(od):
    arrays, flat, rows = _plan_arrays()
    with pytest.raises(ValueError, match="boundary codes"):
        od.plan_offdiagonal_constitutive_from_arrays(
            arrays, flat, rows, (0, 2, 0), (0, 0, 0))
    with pytest.raises(ValueError, match="wall axes"):
        od.plan_offdiagonal_constitutive_from_arrays(
            arrays, flat, rows, (0, 0, 0), (0, 3, 0))


def test_wall_mask_axes_asks_the_masks_own_question(od):
    fields, pml = _build(boundaries=("periodic", "metallic", "periodic"))
    assert od.wall_mask_axes(fields.grid) == (0, 1, 0)
    fields, pml = _build(boundaries="metallic")
    assert od.wall_mask_axes(fields.grid) == (1, 1, 1)
    fields, pml = _build()
    assert od.wall_mask_axes(fields.grid) == (0, 0, 0)


def test_row_volumes_for_binds_slots_in_cycle_order(od):
    fields, _pml = _build(rows="ez_ex")
    slots = od.row_volumes_for(fields)
    assert [value is not None for value in slots] == [
        False, False, False, False, True, False]
    assert slots[4] is fields.chi1inv_offdiagonal_for("Ez")["Ex"]


def test_planning_a_refused_configuration_returns_none_without_triton(od):
    fields, pml = _build()
    assert od.plan_offdiagonal_constitutive(fields, pml) is None, (
        "this laptop's NumPy backend must refuse, and the builder must answer "
        "None without needing the optional dependency")
