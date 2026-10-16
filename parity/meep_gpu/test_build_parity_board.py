"""The bit-parity coverage board: its witnesses, its columns, and its refusals.

The emitted-text witnesses of columns (a) and (b) are tested against planted pairs in
``test_parity_board_emit.py``; this file tests the source-text fallback, the columns built
on them, the development-evidence column (d-dev) and its writer, and the whole board.

Every test reads the tree or builds a synthetic one under ``tmp_path``; none writes the
committed board. No test asks whether the COMMITTED board describes this tree: every lane
that edits a kernel module moves the board's subject, and ``build_parity_board.py check``
answers that question by name when the board is re-cut.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from typing import Any, Dict

import pytest

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import build_parity_board as board  # noqa: E402
import parity_board_text as witness  # noqa: E402

ROOT = board.REPO_ROOT
LIVE = {"fixture_set": "array_path_oracle/real-test", "fingerprint_sha256": "a" * 64}


# ---------------------------------------------------------------------------
# The text witnesses
# ---------------------------------------------------------------------------


def _forms(table: str, source: str) -> Dict[str, int]:
    return witness.form_counts(table, witness.own_text(source)).as_dict()


def test_the_curl_groupings_are_told_apart_in_c_text():
    old = 'K = "float curl = dtdx * ((sf - f1) + (f2 - ss));"\n'
    new = 'K = "float curl = dtdx * (((sf - f1) + f2) - ss);"\n'
    assert _forms("cuda", old)["old_curl"] == 1 and _forms("cuda", old)["new_curl"] == 0
    assert _forms("cuda", new)["new_curl"] == 1 and _forms("cuda", new)["old_curl"] == 0


def test_the_complex_helpers_read_as_their_operators():
    old = 'K = "cf curl = mul_coefficient_left(dtdx, cf_add(cf_sub(sf, f_1), cf_sub(f_2, ss)));"\n'
    new = 'K = "cf curl = mul_coefficient_left(dtdx, cf_sub(cf_add(cf_sub(sf, f_1), f_2), ss));"\n'
    assert _forms("cuda", old)["old_curl"] == 1
    assert _forms("cuda", new)["new_curl"] == 1 and _forms("cuda", new)["old_curl"] == 0


def test_the_constitutive_groupings_are_told_apart_in_one_and_two_statements():
    one = 'K = "a0 = f0[ii] + ((kp_0 * src0) - (km_0 * prev0));"\n'
    split = 'K = "float a = f[idx] + kps * src; f[idx] = a - kms * prev;"\n'
    fused_old = 'K = "f[idx] = (f[idx] + kps * src) - kms * prev;"\n'
    assert _forms("metal", one)["new_constitutive"] == 1
    assert _forms("metal", one)["old_constitutive"] == 0
    assert _forms("cuda", split)["old_constitutive"] == 1
    assert _forms("cuda", fused_old)["old_constitutive"] == 1
    assert _forms("cuda", fused_old)["new_constitutive"] == 0


def test_a_template_placeholder_keeps_the_coefficient_name():
    template = 'LINES = ("{prefix}acc = {prefix}acc + {kp} * {prefix}src;",)\n'
    fstring = 'def f(kp):\n    return f"{prefix}acc = {prefix}acc + {kp}[i] * {prefix}src;"\n'
    assert _forms("metal", template)["old_constitutive"] == 1
    assert _forms("metal", fstring)["old_constitutive"] == 1


def test_prose_comments_and_docstrings_spell_no_form():
    source = '''"""The package grouping was dtdx * ((sf - f1) + (f2 - ss)); see below."""
MESSAGE = ("kernels accumulate ((f + kps*src) - kms*prev) and must not be read "
           "as a form that any kernel compiles")
K = """
// float curl = dtdx * ((sf - f1) + (f2 - ss));
float x = a;
"""
'''
    counts = _forms("cuda", source)
    assert counts["old_curl"] == counts["old_constitutive"] == 0


def test_the_triton_forms_are_read_from_jit_bodies():
    source = '''import triton
import triton.language as tl

@triton.jit
def old_kernel(f, g1s, g1, g2, g2s, dtdx, kp, km, src, prev):
    curl = dtdx * ((g1s - g1) + (g2 - g2s))
    a = (f + kp * src) - km * prev
    return curl, a

@triton.jit
def new_kernel(f, g1s, g1, g2, g2s, dtdx, kp, km, src, prev, cb, i):
    dw = record_in(cb, 0, i)
    curl = dtdx * (((g1s - g1) + g2) - g2s)
    a = tl.where(dw, f + ((kp * src) - (km * prev)), src)
    return curl, a

def host_helper(a, b, c, d):
    return (a - b) + (c - d)
'''
    own = witness.own_text(source)
    counts = witness.form_counts("triton", own).as_dict()
    assert counts == {"old_curl": 1, "new_curl": 1, "old_constitutive": 1,
                      "new_constitutive": 1, "expressions": counts["expressions"]}
    assert witness.record_reads("triton", own)


def test_the_record_witness_per_table():
    cuda = 'K = "    const int cbh_y_a0 = cb[12];\\n    int in_h_y = (j >= cbh_y_a0);"\n'
    metal = 'from . import chunk_bounds as _chunk_bounds\nX = _chunk_bounds.prologue(())\n'
    none = 'K = "float v = f[idx] - curl * sinv;"\n'
    assert witness.record_reads("cuda", witness.own_text(cuda))
    assert witness.record_reads("metal", witness.own_text(metal))
    assert not witness.record_reads("cuda", witness.own_text(none))
    assert witness.pml_form_markers("cuda", witness.own_text(none))


@pytest.mark.parametrize("source, b_state, a_state", [
    ('K = "int x = cb[0]; f[idx] = dtdx * (((sf - f1) + f2) - ss);"\n', "landed", "landed"),
    ('K = "float v = sinv * f[idx]; c = dtdx * ((sf - f1) + (f2 - ss));"\n', "gap", "gap"),
    ('K = "int x = 1;"\n', "refused", "refused"),
    ('K = "c = dtdx * ((sf - f1) + (f2 - ss)); d = dtdx * (((sf - f1) + f2) - ss);"\n',
     "refused", "refused"),
])
def test_the_source_text_columns_name_their_state(source, b_state, a_state):
    cells = board.source_text_columns("cuda", {"kind": "kernel"}, source)
    assert cells["b_select"]["witness"] == cells["a_form"]["witness"] == board.SOURCE_TEXT
    assert cells["b_select"]["state"] == b_state
    assert cells["a_form"]["state"] == a_state
    if a_state == "refused":
        assert cells["a_form"]["why"]


def test_the_landed_roots_read_landed_on_their_source_text_too():
    """The three tables' root modules spell the select and MEEP's forms in their own text
    (lanes KC1, KT1, KM1), so the source-text fallback still sees them."""
    for table, path in (("cuda", "meep_gpu/cuda_kernels/step_curl_kernels.py"),
                        ("cuda", "meep_gpu/cuda_kernels/constitutive_kernels.py"),
                        ("triton", "meep_gpu/triton_kernels/kernels.py"),
                        ("metal", "meep_gpu/metal_kernels/shaders.py")):
        cells = board.source_text_columns(table, {"kind": "kernel"}, (ROOT / path).read_text())
        assert cells["b_select"]["state"] == "landed", (path, cells["b_select"])
        assert cells["a_form"]["state"] == "landed", (path, cells["a_form"])


# ---------------------------------------------------------------------------
# The rows
# ---------------------------------------------------------------------------


def test_the_rows_are_the_step_lists_and_carry_no_verdict():
    rows = board.load_rows(ROOT)
    counts = {table: len(rows["tables"][table]["modules"]) for table in board.TABLES}
    # KERNELS.md section 8: 75 CUDA (74 + fastpath_cuda), 72 Triton + the new forms, 90
    # Metal (89 + metal_dispatch) + the new chunk_bounds and softfp64.
    assert counts == {"cuda": 75, "triton": 73, "metal": 92}
    for table in board.TABLES:
        for row in rows["tables"][table]["modules"]:
            assert (ROOT / row["path"]).is_file(), row
            assert not set(row) & set(board.COLUMNS), row


def test_a_verdict_in_the_rows_file_is_refused(tmp_path):
    document = json.loads((ROOT / board.ROWS_FILE).read_text())
    document["tables"]["cuda"]["modules"][0]["certified"] = True
    target = tmp_path / board.ROWS_FILE
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps(document))
    with pytest.raises(board.BoardRefusal, match="never a verdict"):
        board.load_rows(tmp_path)


def test_a_test_file_belongs_to_the_longest_module_it_names():
    modules = ["no_pml", "no_pml_ade", "dispersive_kernels", "step_curl_kernels"]
    assert board.named_owner("no_pml_ade", modules) == "no_pml_ade"
    assert board.named_owner("no_pml_planner", modules) == "no_pml"
    assert board.named_owner("dispersive", modules) == "dispersive_kernels"
    assert board.named_owner("step_curl_pml_real", modules) == "step_curl_kernels"
    assert board.named_owner("unrelated", modules) is None


# ---------------------------------------------------------------------------
# (c): outcomes and the pending list
# ---------------------------------------------------------------------------


def _write_pending(root: Path, text: str) -> None:
    path = root / board.PENDING_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_pending_blocks_name_kernel_lanes_and_ignore_host_lanes(tmp_path):
    _write_pending(tmp_path, "# Removed by KC2, then KC7.\na::t1\n"
                             "# Drift against base + KC0 + KT0. Re-gated at certification.\n"
                             "a::t2\n")
    listed = board.pending_blocks(tmp_path)
    assert listed["a::t1"]["lanes"] == [("cuda", "C2"), ("cuda", "C7")]
    assert listed["a::t2"]["lanes"] == []


def _outcomes(files: Dict[str, Any]) -> Dict[str, Any]:
    return {"files": files}


def test_host_identity_is_read_from_outcomes_and_the_pending_list():
    files = ["meep_gpu/cuda_kernels/test_x.py"]
    row = {"module": "x", "step": "C2", "kind": "kernel"}
    passing = _outcomes({files[0]: {"passed": 3, "failed": [], "skipped": 1}})
    cell = board.host_column(row, "cuda", files, passing, None, {}, ["x"])
    assert cell["state"] == "landed" and cell["passed"] == 3
    listed = {"meep_gpu.cuda_kernels.test_x::t1": {"lanes": [("cuda", "C2")], "block": 0}}
    failing = _outcomes({files[0]: {"passed": 3, "failed": ["meep_gpu.cuda_kernels.test_x::t1"],
                                    "skipped": 0}})
    cell = board.host_column(row, "cuda", files, failing, None, listed, ["x"])
    assert cell["state"] == "gap" and cell["of_which_name_its_own_lane"] == 1
    certification = {"meep_gpu.cuda_kernels.test_x::t1": {"lanes": [], "block": 0}}
    cell = board.host_column(row, "cuda", files, failing, None, certification, ["x"])
    assert cell["state"] == "landed" and cell["listed_for_certification"] == 1
    cell = board.host_column(row, "cuda", files, failing, None, {}, ["x"])
    assert cell["state"] == "gap" and "undeclared" in cell["why"]


def test_host_identity_is_a_level_above_the_select_and_the_forms():
    passing = {"state": "landed", "passed": 5}
    cells = {"b_select": {"state": "landed"},
             "a_form": {"state": "gap", "why": "spells the 0.9.2 grouping and no MEEP grouping"}}
    gated = board.levels_gate(cells, dict(passing), "kernel")
    assert gated["state"] == "gap" and gated["tests_state"] == "landed"
    assert "(b) MEEP-order A form is a gap" in gated["why"]
    no_select = {"b_select": {"state": "gap", "why": "x"}, "a_form": {"state": "refused"}}
    assert board.levels_gate(no_select, dict(passing), "kernel")["state"] == "gap"
    assert board.levels_gate(no_select, dict(passing), "a_only")["state"] == "landed"
    clean = {"b_select": {"state": "landed"}, "a_form": {"state": "refused"}}
    assert board.levels_gate(clean, dict(passing), "kernel") == passing


def test_host_identity_credits_only_files_that_execute_the_arithmetic(tmp_path):
    """A test file of a lift, a predicate, a seam order or a census measures no arithmetic:
    (c) credits a CUDA file only where it runs the emitted text through the host
    interpreter, a Triton or Metal file only where it compares with ``meep_gpu.stepping``
    (a Metal file of predicate and compile claims measures none)."""
    cuda = tmp_path / "meep_gpu" / "cuda_kernels"
    triton = tmp_path / "meep_gpu"
    cuda.mkdir(parents=True)
    (cuda / "test_runs.py").write_text("from .test_root_chunk_forms import F32, Interpreter\n")
    (cuda / "test_derives.py").write_text("class ScatterInterpreter(Interpreter):\n    pass\n")
    (cuda / "test_lift.py").write_text('"""Mentions test_root_chunk_forms only."""\nimport ast\n')
    (triton / "test_triton_pins.py").write_text("from meep_gpu import stepping\n")
    (triton / "test_triton_relative.py").write_text("from . import stepping\n")
    (triton / "test_triton_contract.py").write_text("from meep_gpu.fields import Fields\n")
    (triton / "test_metal_steps.py").write_text("from meep_gpu import stepping\n")
    (triton / "test_metal_compiles.py").write_text(
        '"""Claims a laptop can check without a GPU."""\nimport torch\n')
    rel = lambda path: path.relative_to(tmp_path).as_posix()  # noqa: E731
    for name, want in (("test_runs.py", True), ("test_derives.py", True), ("test_lift.py", False)):
        assert board.executes_arithmetic(tmp_path, "cuda", rel(cuda / name)) is want, name
    for name, want in (("test_triton_pins.py", True), ("test_triton_relative.py", True),
                       ("test_triton_contract.py", False)):
        assert board.executes_arithmetic(tmp_path, "triton", rel(triton / name)) is want, name
    for name, want in (("test_metal_steps.py", True), ("test_metal_compiles.py", False)):
        assert board.executes_arithmetic(tmp_path, "metal", rel(triton / name)) is want, name
    assert board.executes_arithmetic(tmp_path, "metal", "meep_gpu/test_metal_absent.py") is False
    row = {"module": "x", "step": "C2", "kind": "kernel"}
    files = ["meep_gpu/cuda_kernels/test_x.py", "meep_gpu/cuda_kernels/test_x_lift.py"]
    passing = _outcomes({f: {"passed": 3, "failed": [], "skipped": 0} for f in files})
    none = {files[0]: False, files[1]: False}
    cell = board.host_column(row, "cuda", files, passing, None, {}, ["x"], none)
    assert cell["state"] == "refused", cell
    assert "none of its test files runs the emitted device text" in cell["why"]
    assert cell["not_credited"] == ["test_x.py", "test_x_lift.py"]
    one = {files[0]: True, files[1]: False}
    cell = board.host_column(row, "cuda", files, passing, None, {}, ["x"], one)
    assert cell["state"] == "landed" and cell["passed"] == 3, cell
    assert cell["not_credited"] == ["test_x_lift.py"]


def test_host_identity_refuses_what_it_cannot_measure():
    row = {"module": "x", "step": "C2", "kind": "kernel"}
    assert board.host_column(row, "cuda", [], None, "absent", {}, ["x"])["state"] == "refused"
    files = ["meep_gpu/cuda_kernels/test_x.py"]
    cell = board.host_column(row, "cuda", files, None, "no test outcomes supplied", {}, ["x"])
    assert cell == {"state": "refused", "why": "no test outcomes supplied",
                    "test_files": {"named": ["test_x.py"], "documented": []}}
    skipped = _outcomes({files[0]: {"passed": 0, "failed": [], "skipped": 4}})
    assert board.host_column(row, "cuda", files, skipped, None, {}, ["x"])["state"] == "refused"


# ---------------------------------------------------------------------------
# The subject
# ---------------------------------------------------------------------------


def _mini_tree(tmp_path: Path) -> Path:
    root = tmp_path / "tree"
    (root / "meep_gpu").mkdir(parents=True)
    (root / "parity" / "meep_gpu").mkdir(parents=True)
    (root / "meep_gpu" / "stepping.py").write_text("X = 1\n")
    (root / "parity" / "meep_gpu" / "gate_x.py").write_text("Y = 2\n")
    return root


def test_an_outcomes_run_on_a_moving_tree_is_refused(tmp_path):
    root = _mini_tree(tmp_path)
    before = board.outcome_manifest(root)["sha256"]
    (root / "meep_gpu" / "stepping.py").write_text("X = 2\n")
    with pytest.raises(board.BoardRefusal, match="the outcomes subject and the working tree disagree"):
        board.build_outcomes(root, [], before)


def test_an_outcomes_record_of_another_tree_is_refused(tmp_path):
    root = _mini_tree(tmp_path)
    record = {"schema": board.OUTCOMES_SCHEMA, "version": board.OUTCOMES_VERSION,
              "subject_manifest_sha256": board.outcome_manifest(root)["sha256"],
              "subject_files": 2, "files": {}}
    path = tmp_path / "outcomes.json"
    path.write_text(json.dumps(record))
    assert board.read_outcomes(root, path)[0] is not None
    (root / "parity" / "meep_gpu" / "gate_x.py").write_text("Y = 3\n")
    with pytest.raises(board.BoardRefusal, match="the outcomes subject and the working tree disagree"):
        board.read_outcomes(root, path)


def test_the_boards_own_files_are_not_its_outcomes_subject(tmp_path):
    root = _mini_tree(tmp_path)
    before = board.outcome_manifest(root)
    own = root / board.BOARD_DIR / "board.json"
    own.parent.mkdir(parents=True)
    own.write_text("{}")
    (root / "parity" / "meep_gpu" / "build_parity_board.py").write_text("# edited\n")
    assert board.outcome_manifest(root) == before
    (root / "parity" / "meep_gpu" / "results").mkdir()
    (root / "parity" / "meep_gpu" / "results" / "gate.json").write_text("{}")
    assert board.outcome_manifest(root) == before


def test_a_board_of_another_tree_is_refused(tmp_path):
    root = _mini_tree(tmp_path)
    path = tmp_path / "board.json"
    path.write_text(json.dumps({"subject": board.board_manifest(root)}))
    board.check_board(root, path)
    (root / "meep_gpu" / "stepping.py").write_text("X = 9\n")
    with pytest.raises(board.BoardRefusal, match="the board subject and the working tree disagree"):
        board.check_board(root, path)


# ---------------------------------------------------------------------------
# (d) and (e): the ledgers
# ---------------------------------------------------------------------------


def test_the_bound_digest_is_the_planners():
    from meep_gpu import fastpath  # noqa: PLC0415

    assert board.BOUND_FIELDS == fastpath.BOUND_FIELDS
    ledger = json.loads((ROOT / board.LEDGERS["cuda"]).read_text())
    for key, entry in ledger.items():
        if isinstance(entry, dict) and board.bound_digest(entry) is not None:
            assert board.bound_digest(entry) == fastpath.bound_digest(entry), key
            assert tuple(board.live_runs(entry)) == fastpath.live_capabilities(entry), key


@pytest.mark.parametrize("text, expected", [
    ("meep_x86_flush; ieee_keep_ftz_stripped", {"keep", "flush"}),
    ("keep (installed by the gate before its first device compile: requested=keep)", {"keep"}),
    ("flush - native and uncontrollable on MPS; keep is refused BY NAME", {"flush"}),
    ("flush (requested explicitly; MPS has no lever and keep is a plan-time refusal)", {"flush"}),
    ("NOT INSTALLED - see _mixed_policy", None),
    (None, None),
])
def test_a_runs_policy_names_its_baselines(text, expected):
    assert board.policy_baselines(text) == expected


def _ledger_tree(tmp_path: Path) -> Dict[str, Any]:
    root = tmp_path / "tree"
    module = root / "meep_gpu" / "cuda_kernels" / "k.py"
    module.parent.mkdir(parents=True)
    module.write_text("K = 1\n")
    entry = {"source_sha256": {"meep_gpu/cuda_kernels/k.py": board._sha256(module)},
             "status": "PASS", "array_path_oracle": dict(LIVE)}
    entry["runs"] = {"8.6": {"bound_sha256": board.bound_digest(entry), "host": "a named GPU",
                             "subnormal_policy": "meep_x86_flush; ieee_keep_ftz_stripped"}}
    return {"root": root, "ledger": {"w": entry},
            "row": {"module": "k", "path": "meep_gpu/cuda_kernels/k.py"}}


def test_a_live_released_weld_on_the_live_oracle_is_certified(tmp_path):
    case = _ledger_tree(tmp_path)
    cells = board.device_columns(case["root"], "cuda", case["row"], case["ledger"], LIVE)
    assert {name: cell["state"] for name, cell in cells.items()} == {
        "device_identical:keep": "landed", "device_identical:flush": "landed",
        "certified:keep": "landed", "certified:flush": "landed"}


@pytest.mark.parametrize("plant, column, needle", [
    ("oracle", "device_identical:keep", "records array path array_path_oracle/real-v1"),
    ("bytes", "device_identical:keep", "moved since its run"),
    ("status", "device_identical:keep", "status 'FAIL'"),
    ("policy", "device_identical:flush", "no live run under flush"),
    ("host", "certified:keep", "names its hardware"),
])
def test_each_condition_of_a_certification_is_load_bearing(tmp_path, plant, column, needle):
    case = _ledger_tree(tmp_path)
    entry = case["ledger"]["w"]
    if plant == "oracle":
        entry["array_path_oracle"] = {"fixture_set": "array_path_oracle/real-v1",
                                      "fingerprint_sha256": "b" * 64}
    elif plant == "bytes":
        (case["root"] / case["row"]["path"]).write_text("K = 2\n")
    elif plant == "status":
        entry["status"] = "FAIL"
    elif plant == "policy":
        entry["runs"]["8.6"]["subnormal_policy"] = "keep ONLY (ieee_keep_ftz_stripped)"
    elif plant == "host":
        entry["runs"]["8.6"]["host"] = ""
    cells = board.device_columns(case["root"], "cuda", case["row"], case["ledger"], LIVE)
    assert cells[column]["state"] == "gap", cells[column]
    assert needle in cells[column]["why"], cells[column]


def test_a_module_no_entry_pins_is_refused_and_the_route_record_is_no_weld(tmp_path):
    case = _ledger_tree(tmp_path)
    ledger = {"driver_dispatch": case["ledger"]["w"]}
    cells = board.device_columns(case["root"], "cuda", case["row"], ledger, LIVE)
    assert all(cell["state"] == "refused" for cell in cells.values())


def test_the_metal_table_is_judged_under_flush_only(tmp_path):
    case = _ledger_tree(tmp_path)
    cells = board.device_columns(case["root"], "metal", case["row"], case["ledger"], LIVE)
    assert set(cells) == {"device_identical:flush", "certified:flush"}


# ---------------------------------------------------------------------------
# The array path against MEEP
# ---------------------------------------------------------------------------


_CONSTRUCTION = {
    "constructor": {"symmetry": [{"object": "meep_gpu.grid:Mirror"}], "k_point": [0.0, 0.0, 0.0],
                    "force_complex_fields": False, "cylindrical": False, "beta": 0.0},
    "calls": [("set_epsilon_components", {"chi1inv_offdiagonal": None}), ("setup_pml", {}),
              ("add_source", {"source_data": {"size": [0.0, 0.0, 0.0], "is_integrated": False,
                                              "source_type": "gaussian"}})],
}


@pytest.mark.parametrize("family, expected", [
    ("base_pml", True), ("mirror_symmetry", True), ("complex_bloch", False), ("folds", False),
    ("cylindrical", False), ("conductive", False), ("offdiag", False), ("dispersive", False),
    ("sources", False), ("nonlinear", False),
])
def test_a_case_exercises_the_families_its_construction_names(family, expected):
    families = {f["family"]: f for f in board.load_rows(ROOT)["families"]}
    assert board.exercises(families[family]["exercised_if"], _CONSTRUCTION) is expected


def _case(differing: int = 0, caught: bool = True) -> Dict[str, Any]:
    return {"tier_c": {"stored": {"words": 100, "word_differing": differing, "value_differing": 0},
                       "reader": {"words": 100, "word_differing": 0, "value_differing": 0},
                       "planted_flip": {"caught": caught}, "steps": [0, 1]},
            "tier_a": {"digests": 24, "stored_equal": 24, "reader_equal": 24,
                       "planted_flip": {"caught": True}}}


def test_a_case_is_identical_only_with_every_word_equal_and_both_flips_caught():
    assert board.case_identity(_case())[0] is True
    assert board.case_identity(_case(differing=1))[0] is False
    assert board.case_identity(_case(caught=False))[0] is False


def test_the_meep_rows_refuse_a_missing_baseline_and_another_array_path(tmp_path):
    root = tmp_path / "tree"
    ledger = {"sets": {"keep": {"array_path_oracle": {"fixture_set": "array_path_oracle/real-v1",
                                                      "fingerprint_sha256": "c" * 64},
                                "cases": {}}}}
    path = root / board.MEEP_IDENTITY_FILE
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(ledger))
    families = board.load_rows(ROOT)["families"]
    rows = board.meep_rows(root, families, LIVE)
    for row in rows["families"]:
        assert row["flush"] == {"state": "refused", "why": "the record holds no flush set"}
        assert row["keep"]["state"] == "refused"
        assert "minted on array path array_path_oracle/real-v1" in row["keep"]["why"]


# ---------------------------------------------------------------------------
# The whole board, on this tree
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def built():
    return board.build_board(ROOT, outcomes_path=ROOT / "parity" / "meep_gpu" / "absent.json",
                             oracle=dict(LIVE))


def test_every_row_is_counted_once_per_column(built):
    rows = board.load_rows(ROOT)
    for table in board.TABLES:
        info = built["tables"][table]
        for name, tally in info["summary"].items():
            assert tally["d"] + tally["refused"] + tally["not_applicable"] == info["rows"], (table, name)
            assert tally["n"] <= tally["d"]
        kernels = sum(1 for row in rows["tables"][table]["modules"]
                      if row["kind"] in ("kernel", "a_only"))
        assert info["summary"]["host_identical"]["not_applicable"] == info["rows"] - kernels


def test_without_outcomes_column_c_is_refused_by_name(built):
    """No module reads host-identical without a recorded run: a module is refused by name,
    or is a gap only because (a) or (b) already is (the level below it)."""
    for table in board.TABLES:
        tally = built["tables"][table]["summary"]["host_identical"]
        assert tally["n"] == 0
        assert any("no test outcomes supplied" in why for why in tally["refused_by_reason"])
        for row in built["tables"][table]["modules"]:
            cell = row["cells"]["host_identical"]
            if cell["state"] == "gap":
                assert cell["why"].startswith(("(a) ", "(b) ")), (row["module"], cell)


def test_a_ledger_on_another_array_path_certifies_nothing_and_says_why(built):
    for table in board.TABLES:
        for name, tally in built["tables"][table]["summary"].items():
            if name.startswith(("device_identical", "certified")):
                assert tally["n"] == 0, (table, name)
        assert built["certified_zero_because"][table].startswith("0 of ")
    assert set(built["tables"]["metal"]["summary"]) >= {"certified:flush"}
    assert "certified:keep" not in built["tables"]["metal"]["summary"]


def test_the_markdown_states_every_count_as_a_fraction(built):
    text = board.render_markdown(built)
    assert "| CUDA | 75 |" in text and "| Triton | 73 |" in text and "| Metal | 92 |" in text
    assert "n/a (R13)" in text
    assert "(d-dev) keep, NOT certification" in text
    assert "DEVELOPMENT EVIDENCE, NOT CERTIFICATION" in text


def test_every_device_form_row_names_the_witness_it_was_read_with(built):
    for table in board.TABLES:
        for row in built["tables"][table]["modules"]:
            if row["kind"] not in ("kernel", "a_only"):
                continue
            for column in ("b_select", "a_form"):
                cell = row["cells"][column]
                if cell["state"] == "n/a":
                    continue
                assert cell["witness"] == board.EMITTED or \
                    cell["witness"].startswith(board.SOURCE_TEXT + " (no emit route: "), \
                    (table, row["module"], column, cell)
        witnesses = built["tables"][table]["witnesses"]
        assert witnesses.get("emitted", 0) > witnesses.get("source text", 0), (table, witnesses)


def test_the_development_evidence_column_never_feeds_the_certified_one(built):
    for table in board.TABLES:
        for row in built["tables"][table]["modules"]:
            for name, cell in row["cells"].items():
                if name.startswith("certified") and cell["state"] == "landed":
                    assert row["cells"][name.replace("certified", "device_identical")][
                        "state"] == "landed"


def test_the_generator_imports_nothing_from_a_device_library():
    for name in ("build_parity_board.py", "parity_board_text.py", "parity_board_emit.py",
                 "record_dev_evidence.py"):
        tree = ast.parse((_HERE / name).read_text())
        imported = {alias.name.split(".")[0] for node in ast.walk(tree)
                    if isinstance(node, ast.Import) for alias in node.names}
        imported |= {(node.module or "").split(".")[0] for node in ast.walk(tree)
                     if isinstance(node, ast.ImportFrom)}
        assert not imported & {"cupy", "torch", "triton", "meep"}, (name, imported)


# ---------------------------------------------------------------------------
# (d-dev): development evidence and its writer
# ---------------------------------------------------------------------------

import record_dev_evidence as evidence  # noqa: E402

_SUBJECT = {"module_sha256": "1" * 64, "emitted_sha256": "2" * 64}


def _record(**changes: Any) -> Dict[str, Any]:
    record = {"table": "cuda", "module": "k", "backend": "cuda", "subnormal_baseline": "keep",
              "words_compared": 1000, "words_differing": 0, "gate": "smoke",
              "subject": {**_SUBJECT, "array_path_oracle": dict(LIVE)},
              "date": "2026-10-09", "host_class": "a GPU model"}
    record.update(changes)
    return record


def test_development_evidence_counts_only_on_the_live_subject():
    row = {"module": "k", "kind": "kernel"}
    cells = board.dev_evidence_columns("cuda", row, _SUBJECT, [_record()], None, LIVE)
    assert cells["dev_evidence:keep"]["state"] == "landed"
    assert cells["dev_evidence:keep"]["not"] == "certification"
    assert cells["dev_evidence:flush"]["state"] == "refused"
    for moved, needle in (({"module_sha256": "9" * 64}, "the module's bytes"),
                          ({"emitted_sha256": "9" * 64}, "its emitted device text")):
        live = {**_SUBJECT, **moved}
        cell = board.dev_evidence_columns("cuda", row, live, [_record()], None, LIVE)[
            "dev_evidence:keep"]
        assert cell["state"] == "refused" and cell["stale_records"] == 1, cell
        assert needle in cell["why"] and "moved since" in cell["why"], cell
    other_oracle = {"fixture_set": "array_path_oracle/real-v1", "fingerprint_sha256": "b" * 64}
    cell = board.dev_evidence_columns("cuda", row, _SUBJECT, [_record()], None, other_oracle)[
        "dev_evidence:keep"]
    assert cell["state"] == "refused" and "the array path" in cell["why"]
    cell = board.dev_evidence_columns("cuda", row, _SUBJECT, [_record(words_differing=3)],
                                      None, LIVE)["dev_evidence:keep"]
    assert cell["state"] == "gap" and "3 of 1,000 words differ" in cell["why"]


def test_the_writer_refuses_a_stale_subject_by_name(tmp_path):
    before = {**_SUBJECT, "array_path_oracle": dict(LIVE), "table": "cuda", "module": "k"}
    arguments = {"table": "cuda", "module": "k", "backend": "cuda", "baseline": "keep",
                 "words_compared": 10, "words_differing": 0, "gate": "smoke",
                 "host_class": "a GPU model", "date": "2026-10-09"}
    record = evidence.make_record(tmp_path, arguments, before, dict(before))
    assert record["subject"] == {**_SUBJECT, "array_path_oracle": dict(LIVE)}
    moved = dict(before, emitted_sha256="3" * 64)
    with pytest.raises(evidence.EvidenceRefusal,
                       match="the evidence subject and the working tree disagree: "
                             "emitted_sha256 moved"):
        evidence.make_record(tmp_path, arguments, before, moved)


@pytest.mark.parametrize("change, needle", [
    ({"baseline": "keep", "table": "metal", "backend": "metal"}, "owner ruling R13"),
    ({"backend": "metal"}, "does not run the cuda table's kernels"),
    ({"words_compared": 0}, "compared nothing"),
    ({"words_differing": 11}, "11 words differing out of 10"),
    ({"host_class": ""}, "host class is required"),
    ({"date": "9 Oct"}, "is not YYYY-MM-DD"),
])
def test_the_writer_refuses_what_is_not_evidence(tmp_path, change, needle):
    arguments = {"table": "cuda", "module": "k", "backend": "cuda", "baseline": "keep",
                 "words_compared": 10, "words_differing": 0, "gate": "smoke",
                 "host_class": "a GPU model", "date": "2026-10-09"}
    arguments.update(change)
    before = {**_SUBJECT, "array_path_oracle": dict(LIVE), "table": arguments["table"],
              "module": "k"}
    with pytest.raises(evidence.EvidenceRefusal, match=needle):
        evidence.make_record(tmp_path, arguments, before, dict(before))


def test_the_writer_refuses_a_host_name(tmp_path):
    import socket  # noqa: PLC0415

    name = socket.gethostname().split(".")[0]
    if not name:
        pytest.skip("this machine reports no host name")
    arguments = {"table": "cuda", "module": "k", "backend": "cuda", "baseline": "keep",
                 "words_compared": 10, "words_differing": 0, "gate": "smoke",
                 "host_class": f"GPU on {name}", "date": "2026-10-09"}
    before = {**_SUBJECT, "array_path_oracle": dict(LIVE), "table": "cuda", "module": "k"}
    with pytest.raises(evidence.EvidenceRefusal, match="never a host name"):
        evidence.make_record(tmp_path, arguments, before, dict(before))


def test_a_rerun_of_the_same_leg_replaces_its_record(tmp_path):
    root = tmp_path / "tree"
    evidence.write_record(root, _record())
    evidence.write_record(root, _record(words_compared=2000))
    evidence.write_record(root, _record(gate="another smoke"))
    records, unavailable = board.read_dev_evidence(root)
    assert unavailable is None
    assert [(r["gate"], r["words_compared"]) for r in records] == [
        ("another smoke", 1000), ("smoke", 2000)]
    document = json.loads((root / board.DEV_EVIDENCE_FILE).read_text())
    assert "NOT CERTIFICATION" in document["about"]


def test_a_record_the_writer_did_not_make_is_refused(tmp_path):
    root = tmp_path / "tree"
    path = root / board.DEV_EVIDENCE_FILE
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"schema": board.DEV_EVIDENCE_SCHEMA,
                                "version": board.DEV_EVIDENCE_VERSION,
                                "records": [{"table": "cuda", "module": "k"}]}))
    with pytest.raises(board.BoardRefusal, match="not written by record_dev_evidence.py"):
        board.read_dev_evidence(root)


def test_the_evidence_record_is_the_boards_subject_and_not_the_outcomes(tmp_path):
    root = _mini_tree(tmp_path)
    outcome_before = board.outcome_manifest(root)
    board_before = board.board_manifest(root)
    evidence.write_record(root, _record())
    assert board.outcome_manifest(root) == outcome_before
    assert board.board_manifest(root) != board_before
