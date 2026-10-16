"""Host tests for the timing-row store and its cuts (``timing_rows.py``).

Synthetic release trees and row records under ``tmp_path``; nothing is timed and no
device is read. Groups, in the repository's order for a new engine:

* KNOWN VALUE -- a row's medians, spreads and gate status; the identical-case table
  read off the newest valid row of each key; re-time one of N rows and re-cut;
* DEGENERATE -- a row with no import record, an intruded row, an edited record, a row
  with no environment record, a dispatch check that was not cut on the release tree;
* SCALING (what a change re-times) -- a step-path change re-times every row that loaded
  it; an arm's change re-times only the keys whose launch counters show it ran and owes
  the others a dispatch check; an arm the ledgers cannot attribute resolves by its
  table's slice; ``bill`` says it before the change lands;
* SHAPE -- records are sealed and append-only; one dispatch-preference record per table
  per architecture, a second architecture added beside the first and never replacing
  it; a partial set names its missing keys; the Apple family's record name;
* REALISTIC -- records built from a ladder directory and a bench row as those drivers
  write them.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
for _path in (HERE, REPO_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import cut_timing_record as cut  # noqa: E402
import test_cut_timing_record as fixtures  # noqa: E402
import timing_rows as TR  # noqa: E402
from meep_gpu import dispatch_preference as dp  # noqa: E402

ARCH = "8.6"
OTHER_ARCH = "9.0"
ENVIRONMENT = {"reference": {"name": "build_record.json", "sha256": "ab" * 32,
                             "record": {"platform": "linux-64"}}, "meep_builds": {}}
#: The host gates a session records around every unit: the quiet window it opened on
#: (two clear readings), the watcher's block, and the session's variance verdict.
CLEAR = {"utc": "2026-10-20T09:00:00Z", "intrusions": [], "recorded": {}}
GATES = {"quiet": [CLEAR, CLEAR],
         "watcher": {"every_s": 15.0, "readings": 4, "first": CLEAR, "last": CLEAR,
                     "intrusions": []},
         "variance": {"ok": True, "verdict": "GO"},
         "power": {"gated": False, "recorded": None}}


def gates_with(**changes):
    return dict(GATES, **changes)


# ---------------------------------------------------------------------------
# A synthetic release tree
# ---------------------------------------------------------------------------

FILES = {
    "meep_gpu/__init__.py": "",
    "meep_gpu/driver.py": "STEP = 1\n",
    "meep_gpu/fastpath.py": "PLAN = 1\n",
    "meep_gpu/cuda_kernels/__init__.py": "",
    "meep_gpu/cuda_kernels/fused_electric_pair.py": "KERNEL = 'electric'\n",
    "meep_gpu/cuda_kernels/fused_magnetic_pair.py": "KERNEL = 'magnetic'\n",
    "meep_gpu/cuda_kernels/compile_cache.py": "CACHE = 1\n",
    "meep_gpu/triton_kernels/__init__.py": "",
    "meep_gpu/triton_kernels/pml_curl.py": "KERNEL = 'pml'\n",
    "parity/meep_gpu/gate_cuda_electric.py":
        "from meep_gpu.cuda_kernels import fused_electric_pair\n",
    "parity/meep_gpu/gate_cuda_magnetic.py":
        "from meep_gpu.cuda_kernels import fused_magnetic_pair\n",
    "parity/meep_gpu/bench_fused_products.py": "BENCH = 1\n",
    "meep_gpu/dispatch_preference.py":
        'SUBJECT_FILES = {"fused_pairs": "cuda_kernels/fused_pairs.py"}\n',
}

#: Ledger entries: the electric pair's pins its module and its gate; the magnetic
#: pair's its own. ``family_recert`` is a sweep name, which no ledger holds.
LEDGER = {
    "cuda_electric_gate": {"source_sha256": {
        "meep_gpu/cuda_kernels/fused_electric_pair.py": "0" * 64,
        "meep_gpu/cuda_kernels/compile_cache.py": "0" * 64,
        "parity/meep_gpu/gate_cuda_electric.py": "0" * 64}},
    "cuda_magnetic_gate": {"source_sha256": {
        "meep_gpu/cuda_kernels/fused_magnetic_pair.py": "0" * 64,
        "meep_gpu/cuda_kernels/compile_cache.py": "0" * 64,
        "parity/meep_gpu/gate_cuda_magnetic.py": "0" * 64}},
}


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def release_tree(tmp_path, overrides=None):
    root = tmp_path / "release"
    files = dict(FILES, **(overrides or {}))
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    (root / "meep_gpu/cuda_kernels/certification.json").write_text(json.dumps(LEDGER))
    (root / "meep_gpu/triton_kernels/fingerprints.json").write_text(json.dumps({}))
    return str(root)


def change(root, name, text="CHANGED = 1\n"):
    with open(os.path.join(root, name), "w") as handle:
        handle.write(text)


def imported(names):
    """The import record a process that loaded ``names`` from the ORIGINAL tree wrote."""
    return {name: sha(FILES[name]) for name in names}


ALL_LOADED = ("meep_gpu/__init__.py", "meep_gpu/driver.py", "meep_gpu/fastpath.py",
              "meep_gpu/cuda_kernels/__init__.py",
              "meep_gpu/cuda_kernels/fused_electric_pair.py",
              "meep_gpu/cuda_kernels/fused_magnetic_pair.py",
              "meep_gpu/cuda_kernels/compile_cache.py",
              "meep_gpu/triton_kernels/__init__.py", "meep_gpu/triton_kernels/pml_curl.py",
              "parity/meep_gpu/bench_fused_products.py")


def counters(*arms, process="gpu:cuda", leg="fused"):
    """Launch counters: each arm ``(slot, backend, arm label, gate)`` dispatched 6 times."""
    return [{"process": process, "leg": leg,
             "slots": {slot: {"arm": arm, "backend": backend, "dispatches": 6,
                              "family": "f", "gate": gate}
                       for slot, backend, arm, gate in arms}}]


ELECTRIC = ("step_D", "cuda", "cuda:fused electric pair", "cuda_electric_gate")
MAGNETIC = ("step_B", "cuda", "cuda:fused magnetic pair", "cuda_magnetic_gate")
TRITON_SINGLE = ("step_B", "triton", "PML", "family_recert_2026-08-14")


def gpu_leg(rates, struck=None):
    return TR.leg_median([{"pass": i + 1, "rate": rate, "spread": 0.01,
                           "struck": (struck or {}).get(i + 1)}
                          for i, rate in enumerate(rates)], "gpu")


def meep_leg(rates, ranks, struck=None, control=None):
    return dict(TR.leg_median([{"pass": i + 1, "rate": rate, "spread": 0.02,
                                "struck": (struck or {}).get(i + 1)}
                               for i, rate in enumerate(rates)], "meep"),
                ranks=ranks, control=control)


def ladder_row(cells, session, ended, *, arms=(ELECTRIC,), loaded=ALL_LOADED,
               route_rates=(2000.0, 2010.0, 1990.0), meep_rates=(500.0, 505.0, 495.0),
               architecture=ARCH, environment=ENVIRONMENT, gates=None):
    key = TR.ladder_key(architecture, "pml_3d", cells, "priority")
    record = TR.base_record("ladder", key, session={"id": session, "started_utc": ended,
                                                    "ended_utc": ended},
                            role="release", architecture=architecture,
                            unit={"name": "u01", "rc": 0}, environment=environment,
                            gates=GATES if gates is None else gates)
    record["processes"] = [{"role": "gpu:cuda:p1", "artifact": "rows.jsonl",
                            "ran_package": True, TR.IMPORTED_KEY: imported(loaded),
                            TR.DATA_KEY: {}}]
    record["launch_counters"] = counters(*arms)
    record["legs"] = {"gpu": {"cuda": gpu_leg(route_rates)}, "array": None,
                      "meep": {"meep@32": meep_leg(meep_rates, 32)}}
    return record


def store_of(tmp_path, records):
    store = str(tmp_path / "store")
    for record in records:
        TR.write_record(store, record)
    return store


# ---------------------------------------------------------------------------
# Known value: medians, gate status, re-time one of N and re-cut
# ---------------------------------------------------------------------------

def test_a_gpu_leg_carries_the_5_percent_gate_and_its_median():
    passed = gpu_leg([2000.0, 2010.0, 1990.0])
    assert passed["median"] == 2000.0 and passed["status"] == "passed"
    assert passed["gate"] == "passed the 5 % gate"
    wide = gpu_leg([2844.0, 2878.9, 2617.9])
    assert wide["status"] == "struck" and "between-pass spread" in wide["gate"]
    assert wide["median"] == 2844.0, "a struck leg keeps its median, with its status"
    struck = gpu_leg([700.0, 705.0, 702.0], struck={2: "TIMED-NOT-REPORTABLE: floors"})
    assert struck["status"] == "struck" and "pass 2" in struck["gate"]


def test_meep_pass_spread_is_recorded_and_never_gated():
    """Owner ruling: MEEP's own pass spread strikes nothing. A 12 % spread enters the
    median with its status; a pass struck for what is not spread is left out by name."""
    leg = meep_leg([420.0, 470.0, 440.0], 8)
    assert leg["status"] == "ungated" and leg["median"] == 440.0
    assert leg["between_pass_spread"] == pytest.approx(50.0 / 440.0, rel=1e-5)
    assert TR.MEEP_SPREAD_RULE in leg["gate"]
    partly = meep_leg([420.0, 470.0, 440.0], 8, struck={2: "geometry digest differs"})
    assert partly["median"] == 430.0 and "left out" in partly["gate"]
    assert partly["between_pass_spread"] == pytest.approx(20.0 / 430.0, rel=1e-5), \
        "a MEEP leg's spread is over the passes that enter its median"
    gone = meep_leg([420.0], 8, struck={1: "the MEEP build is not single precision"})
    assert gone["status"] == "struck" and gone["median"] is None


def test_retime_one_of_n_rows_and_recut(tmp_path):
    """Three keys timed in one session; one re-timed later in its own session. The cut
    takes the newest valid row of each key: the re-timed number for that key, the first
    session's for the other two, and nothing assembled by hand."""
    root = release_tree(tmp_path)
    first = [ladder_row(cells, "s1", "2026-10-20T10:00:00Z")
             for cells in (512000, 2097152, 7077888)]
    again = ladder_row(2097152, "s2", "2026-10-21T10:00:00Z",
                       route_rates=(2600.0, 2610.0, 2590.0))
    store = store_of(tmp_path, first + [again])
    selection = TR.select(TR.load_store(store), root)
    assert all(item["owed"] == "nothing" for item in selection.values())
    table = TR.cut_ladder_table(selection, ARCH)
    by_cells = {row["cells"]: row for row in table["rows"]}
    assert sorted(by_cells) == [512000, 2097152, 7077888]
    assert by_cells[2097152]["session"] == "s2"
    assert by_cells[2097152]["gpu"]["cuda"]["median"] == 2600.0
    assert {by_cells[c]["session"] for c in (512000, 7077888)} == {"s1"}
    ratio = by_cells[2097152]["ratios"]["cuda"]
    assert ratio["ratio"] == pytest.approx(2600.0 / 500.0) and ratio["enters"] is True
    assert "passed the 5 % gate" in ratio["why"] and TR.MEEP_SPREAD_RULE in ratio["why"]
    text = TR.render_table(table)
    assert "| pml_3d | 2,097,152 | priority | cuda | 2600.0 |" in text
    assert by_cells[2097152]["variance"] == "GO"


def test_a_struck_route_median_is_printed_with_its_status_and_does_not_enter(tmp_path):
    root = release_tree(tmp_path)
    row = ladder_row(2097152, "s1", "2026-10-20T10:00:00Z",
                     route_rates=(2844.0, 2878.9, 2617.9))
    selection = TR.select(TR.load_store(store_of(tmp_path, [row])), root)
    ratio = TR.cut_ladder_table(selection, ARCH)["rows"][0]["ratios"]["cuda"]
    assert ratio["enters"] is False and "struck by the 5 % gate" in ratio["why"]


# ---------------------------------------------------------------------------
# Degenerate
# ---------------------------------------------------------------------------

def test_a_row_without_an_import_record_is_refused_by_name(tmp_path):
    root = release_tree(tmp_path)
    row = ladder_row(512000, "s1", "2026-10-20T10:00:00Z")
    row["processes"][0].pop(TR.IMPORTED_KEY)
    selection = TR.select(TR.load_store(store_of(tmp_path, [row])), root)
    item = selection[row["key"]]
    assert item["selected"] is None and item["owed"] == "a re-time"
    assert any("no import record" in why for why in item["assessments"][0]["refused"])


def test_an_intruded_row_and_a_row_without_its_environment_are_refused(tmp_path):
    root = release_tree(tmp_path)
    intruded = ladder_row(512000, "s1", "2026-10-20T10:00:00Z",
                          gates=gates_with(watcher=dict(GATES["watcher"], intrusions=[
                              {"what": ["busy"]}])))
    bare = ladder_row(2097152, "s1", "2026-10-20T10:00:00Z",
                      environment={"reference": None})
    selection = TR.select(TR.load_store(store_of(tmp_path, [intruded, bare])), root)
    assert "intruded" in selection[intruded["key"]]["assessments"][0]["refused"][0]
    assert "R27" in selection[bare["key"]]["assessments"][0]["refused"][0]
    # a legacy import may admit the second, by name; never the first
    allowed = TR.select(TR.load_store(str(tmp_path / "store")), root,
                        allow_unrecorded_environment=True)
    assert allowed[bare["key"]]["selected"] is not None
    assert allowed[intruded["key"]]["selected"] is None


def test_a_refused_later_attempt_leaves_the_older_valid_row_in_force(tmp_path):
    root = release_tree(tmp_path)
    older = ladder_row(512000, "s1", "2026-10-20T10:00:00Z")
    later = ladder_row(512000, "s2", "2026-10-21T10:00:00Z", route_rates=(1.0, 1.0, 1.0),
                       gates=gates_with(watcher=dict(GATES["watcher"], intrusions=[
                           {"what": ["busy"]}])))
    item = TR.select(TR.load_store(store_of(tmp_path, [older, later])), root)[older["key"]]
    assert item["selected"]["session"]["id"] == "s1" and item["owed"] == "nothing"
    assert [a["status"] for a in item["assessments"]] == ["refused", "valid"]


def test_records_are_sealed_and_append_only(tmp_path):
    row = ladder_row(512000, "s1", "2026-10-20T10:00:00Z")
    store = store_of(tmp_path, [row])
    with pytest.raises(TR.RowRefused) as refused:
        TR.write_record(store, row)
    assert refused.value.code == "exists"
    path = TR.record_path(store, TR.seal(row))
    held = json.load(open(path))
    held["legs"]["gpu"]["cuda"]["median"] = 9999.0
    with open(path, "w") as handle:
        json.dump(held, handle)
    loaded = TR.load_store(store)
    assert "record_digest_mismatch" in loaded[0]["_unreadable"]
    root = release_tree(tmp_path)
    assert TR.select(loaded, root)[row["key"]]["assessments"][0]["status"] == "refused"


def test_a_key_in_another_spelling_is_refused():
    with pytest.raises(TR.RowRefused):
        TR.parse_key("ladder|8.6|pml_3d|512000")
    with pytest.raises(TR.RowRefused):
        TR.fused_key(ARCH, "cuda", "pml_2d", "512000", 1)


def test_a_row_without_its_host_gates_is_refused_and_a_hand_collected_row_too(tmp_path):
    """R18 as evidence: a row is cut only when its unit opened on a quiet window of two
    clear readings and was watched while it ran. A row collected by hand has neither
    and is kept, never cut; so is one whose window saw an intrusion."""
    root = release_tree(tmp_path)
    cases = {
        "no_window": gates_with(quiet=None),
        "one_reading": gates_with(quiet=[CLEAR]),
        "window_not_clear": gates_with(quiet=[CLEAR, dict(CLEAR, intrusions=["busy"])]),
        "not_watched": gates_with(watcher=None),
        "watched_never": gates_with(watcher=dict(GATES["watcher"], readings=0)),
        "by_hand": {"quiet": None, "variance": None, "watcher": None,
                    "power": {"gated": False, "recorded": None}, "collected_by_hand": True},
    }
    rows = {name: ladder_row(512000, f"s_{name}", "2026-10-20T10:00:00Z",
                             architecture=f"9.{index}", gates=gates)
            for index, (name, gates) in enumerate(sorted(cases.items()))}
    selection = TR.select(TR.load_store(store_of(tmp_path, list(rows.values()))), root)
    for name, row in rows.items():
        item = selection[row["key"]]
        assert item["selected"] is None and item["owed"] == "a re-time", name
        assert any("R18" in why for why in item["assessments"][0]["refused"]), name
    assert "collected by hand" in selection[rows["by_hand"]["key"]]["assessments"][0][
        "refused"][0]


def test_a_ladder_unit_that_exited_nonzero_is_refused(tmp_path):
    """A ladder that stopped early (its gate gave up, a signal) wrote some passes of its
    rows: they are recorded and never cut. A fused unit's status says only whether a
    case was reportable, and its row's verdict speaks for it."""
    root = release_tree(tmp_path)
    stopped = ladder_row(512000, "s1", "2026-10-20T10:00:00Z")
    stopped["unit"] = {"name": "u01", "rc": 4}
    unknown = ladder_row(2097152, "s1", "2026-10-20T10:00:00Z")
    unknown["unit"] = {"name": "u01"}
    selection = TR.select(TR.load_store(store_of(tmp_path, [stopped, unknown])), root)
    for row in (stopped, unknown):
        assert any("exited rc" in why for why in
                   selection[row["key"]]["assessments"][0]["refused"])


def test_a_gpu_process_without_ledgers_or_package_modules_is_refused(tmp_path):
    root = release_tree(tmp_path)
    no_ledgers = ladder_row(512000, "s1", "2026-10-20T10:00:00Z")
    no_ledgers["processes"][0].pop(TR.DATA_KEY)
    harness_only = ladder_row(2097152, "s1", "2026-10-20T10:00:00Z",
                              loaded=("parity/meep_gpu/bench_fused_products.py",))
    selection = TR.select(TR.load_store(store_of(tmp_path, [no_ledgers, harness_only])),
                          root)
    assert any("no ledger digests" in why for why in
               selection[no_ledgers["key"]]["assessments"][0]["refused"])
    assert any("no package module" in why for why in
               selection[harness_only["key"]]["assessments"][0]["refused"])


def test_two_attempts_of_one_session_are_both_kept(tmp_path):
    """The same command resumes a session and re-runs a unit that did not finish: that
    attempt's record lands beside the first, never over it, and the newer is cut."""
    root = release_tree(tmp_path)
    first = ladder_row(512000, "s1", "2026-10-20T10:00:00Z", route_rates=(1.0, 1.0, 1.0))
    again = ladder_row(512000, "s1", "2026-10-20T12:00:00Z")
    store = store_of(tmp_path, [first, again])
    assert len(TR.load_store(store)) == 2
    item = TR.select(TR.load_store(store), root)[first["key"]]
    assert item["selected"]["session"]["ended_utc"] == "2026-10-20T12:00:00Z"


def test_the_release_tree_must_state_its_subject_files(tmp_path):
    """Without ``SUBJECT_FILES`` the emitters would read as one arm's kernel text and be
    narrowed: the attribution refuses by name, and reads the RELEASE tree's list, not
    the harness's own package."""
    root = release_tree(tmp_path)
    assert TR.Attribution(root).subject_files == {"meep_gpu/cuda_kernels/fused_pairs.py"}
    change(root, "meep_gpu/dispatch_preference.py", "OTHER = 1\n")
    with pytest.raises(TR.RowRefused) as refused:
        TR.Attribution(root)
    assert refused.value.code == "no_subject_files"


# ---------------------------------------------------------------------------
# What a change re-times
# ---------------------------------------------------------------------------

def test_a_step_path_change_retimes_every_row_that_loaded_it(tmp_path):
    root = release_tree(tmp_path)
    rows = [ladder_row(512000, "s1", "2026-10-20T10:00:00Z", arms=(ELECTRIC,)),
            ladder_row(2097152, "s1", "2026-10-20T10:00:00Z", arms=(MAGNETIC,))]
    store = store_of(tmp_path, rows)
    change(root, "meep_gpu/driver.py")
    selection = TR.select(TR.load_store(store), root)
    for row in rows:
        item = selection[row["key"]]
        assert item["owed"] == "a re-time"
        assert any(why.startswith("meep_gpu/driver.py") for why in
                   item["assessments"][0]["stale"])


def test_an_arm_change_retimes_only_the_keys_whose_launch_counters_show_it_ran(tmp_path):
    """Both rows LOADED both pair modules (the table's predicates are read for every
    plan); only the first LAUNCHED the electric pair. A change to the electric pair's
    module re-times the first; the second is owed a dispatch check, and is valid again
    once a check cut on the release tree shows it still launches what it launched."""
    root = release_tree(tmp_path)
    electric = ladder_row(512000, "s1", "2026-10-20T10:00:00Z", arms=(ELECTRIC,))
    magnetic = ladder_row(2097152, "s1", "2026-10-20T10:00:00Z", arms=(MAGNETIC,))
    store = store_of(tmp_path, [electric, magnetic])
    records = TR.load_store(store)
    # before it lands: the timing half of import_closure.py bill
    billed = TR.bill(records, ["meep_gpu/cuda_kernels/fused_electric_pair.py"], root)
    assert billed == {"retime": [electric["key"]], "check": [magnetic["key"]],
                      "untouched": []}
    change(root, "meep_gpu/cuda_kernels/fused_electric_pair.py")
    selection = TR.select(records, root)
    first = selection[electric["key"]]
    assert first["owed"] == "a re-time"
    assert "kernel text of an arm this row launched" in first["assessments"][0]["stale"][0]
    assert first["assessments"][0]["arms"][0]["rule"].startswith("ledger:cuda:")
    second = selection[magnetic["key"]]
    assert second["owed"] == "a dispatch check" and second["selected"] is None
    assert list(second["assessments"][0]["exempt"]) == [
        "meep_gpu/cuda_kernels/fused_electric_pair.py"]
    # the dispatch check, cut on the release tree: same launches -> the row is valid
    check = dict(magnetic, kind="check", key=TR.check_key(magnetic["key"]),
                 session={"id": "c1", "started_utc": "2026-10-22T10:00:00Z",
                          "ended_utc": "2026-10-22T10:00:00Z"})
    check["processes"] = [dict(check["processes"][0], **{TR.IMPORTED_KEY: {
        name: TR.artifact_drift._sha256(__import__("pathlib").Path(root) / name)
        for name in ALL_LOADED}})]
    TR.write_record(store, check)
    selection = TR.select(TR.load_store(store), root)
    assert selection[magnetic["key"]]["selected"]["session"]["id"] == "s1"
    assert selection[magnetic["key"]]["owed"] == "nothing"
    assert selection[electric["key"]]["owed"] == "a re-time"


def test_a_check_that_launched_another_arm_leaves_the_row_owed(tmp_path):
    root = release_tree(tmp_path)
    magnetic = ladder_row(2097152, "s1", "2026-10-20T10:00:00Z", arms=(MAGNETIC,))
    store = store_of(tmp_path, [magnetic])
    change(root, "meep_gpu/cuda_kernels/fused_electric_pair.py")
    check = dict(magnetic, kind="check", key=TR.check_key(magnetic["key"]),
                 session={"id": "c1", "started_utc": "2026-10-22T10:00:00Z",
                          "ended_utc": "2026-10-22T10:00:00Z"},
                 launch_counters=counters(MAGNETIC, ELECTRIC))
    check["processes"] = [dict(check["processes"][0], **{TR.IMPORTED_KEY: {
        name: TR.artifact_drift._sha256(__import__("pathlib").Path(root) / name)
        for name in ALL_LOADED}})]
    stale_check = dict(check, session={"id": "c0", "started_utc": "2026-10-23T10:00:00Z",
                                       "ended_utc": "2026-10-23T10:00:00Z"},
                       launch_counters=counters(MAGNETIC))   # cut on the OLD bytes
    stale_check["processes"] = magnetic["processes"]
    TR.write_record(store, check)
    TR.write_record(store, stale_check)
    item = TR.select(TR.load_store(store), root)[magnetic["key"]]
    witness = item["assessments"][0]["dispatch_check"]
    assert witness["session"] == "c1", "a check not cut on the release tree is no witness"
    assert witness["ok"] is False and witness["differs"]
    assert item["owed"] == "a dispatch check"


def test_an_arm_no_ledger_attributes_resolves_by_its_tables_slice(tmp_path):
    """The 1.0 Triton single arms cite the sweep ``family_recert_2026-08-14``, which no
    ledger holds: every kernel file of their table re-times the row; a kernel file of
    another table owes it a dispatch check."""
    root = release_tree(tmp_path)
    row = ladder_row(512000, "s1", "2026-10-20T10:00:00Z", arms=(TRITON_SINGLE,))
    store = store_of(tmp_path, [row])
    change(root, "meep_gpu/triton_kernels/pml_curl.py")
    item = TR.select(TR.load_store(store), root)[row["key"]]
    assert item["owed"] == "a re-time"
    assert item["assessments"][0]["arms"][0]["rule"].startswith("backend-slice:triton")
    root2 = release_tree(tmp_path / "second")
    change(root2, "meep_gpu/cuda_kernels/fused_magnetic_pair.py")
    item = TR.select(TR.load_store(store), root2)[row["key"]]
    assert item["owed"] == "a dispatch check"


def test_a_row_with_no_launch_counters_is_never_narrowed(tmp_path):
    """An ERROR row launched nothing it can name: a moved kernel file re-times it, and
    no dispatch check can witness it."""
    root = release_tree(tmp_path)
    row = ladder_row(512000, "s1", "2026-10-20T10:00:00Z", arms=())
    row["launch_counters"] = []
    store = store_of(tmp_path, [row])
    change(root, "meep_gpu/cuda_kernels/fused_magnetic_pair.py")
    item = TR.select(TR.load_store(store), root)[row["key"]]
    assert item["owed"] == "a re-time"
    assert "no launched slot" in item["assessments"][0]["stale"][0]
    assert TR.bill(TR.load_store(store), ["meep_gpu/cuda_kernels/fused_magnetic_pair.py"],
                   root)["retime"] == [row["key"]]
    assert TR.dispatch_witness(row, [dict(row, kind="check",
                                          key=TR.check_key(row["key"]))], root)[
        "ok"] is False


def test_a_later_attempt_that_errored_does_not_shadow_an_earlier_timing(tmp_path):
    root = release_tree(tmp_path)
    store = str(tmp_path / "store")
    fused_row(tmp_path, store, case="pml_2d", architecture=ARCH, session="s1",
              ended="2026-10-21T10:00:00Z")
    fused_row(tmp_path, store, case="pml_2d", architecture=ARCH, session="s2",
              ended="2026-10-22T10:00:00Z", verdict="ERROR")
    item = next(iter(TR.select(TR.load_store(store), root).values()))
    assert item["selected"]["session"]["id"] == "s1" and item["owed"] == "nothing"
    # a key whose only valid row ERRORed is still selected, so the cut names it, and
    # it is owed a re-time
    lonely = str(tmp_path / "store2")
    fused_row(tmp_path / "y", lonely, case="pml_2d", architecture=ARCH, session="s2",
              ended="2026-10-22T10:00:00Z", verdict="ERROR")
    item = next(iter(TR.select(TR.load_store(lonely), root).values()))
    assert item["selected"]["legs"]["verdict"] == "ERROR" and item["owed"] == "a re-time"


def test_a_file_the_row_never_loaded_moves_nothing(tmp_path):
    root = release_tree(tmp_path)
    row = ladder_row(512000, "s1", "2026-10-20T10:00:00Z",
                     loaded=[n for n in ALL_LOADED if "magnetic" not in n])
    change(root, "meep_gpu/cuda_kernels/fused_magnetic_pair.py")
    item = TR.select(TR.load_store(store_of(tmp_path, [row])), root)[row["key"]]
    assert item["owed"] == "nothing" and item["assessments"][0]["moved"] == {}


def test_the_emitters_and_the_planners_stay_per_table():
    """What the code says every plan runs is never narrowed to an arm."""
    attribution = TR.Attribution(REPO_ROOT)
    for name in ("meep_gpu/cuda_kernels/fused_pairs.py",
                 "meep_gpu/cuda_kernels/fused_polarization_pair.py",
                 "meep_gpu/metal_kernels/launch.py", "meep_gpu/metal_kernels/device.py",
                 "meep_gpu/cuda_kernels/registry.py", "meep_gpu/triton_kernels/launch.py",
                 "meep_gpu/fastpath.py", "meep_gpu/fastpath_cuda.py",
                 "meep_gpu/deposit_repair.py", "meep_gpu/host_writes.py",
                 "parity/meep_gpu/bench_fused_products.py"):
        assert attribution.step_path(name) is not None, name
    # the release's own route pins join them where no arm's entry claims the file ...
    assert attribution.route_pins, "no driver_dispatch pins read from the ledgers"
    assert "meep_gpu/cuda_kernels/arms.py" in attribution.route_pins
    assert not attribution.pinned_by_an_arm("meep_gpu/cuda_kernels/arms.py")
    # ... and a fused pair's own module, pinned by the CUDA route record AND by its
    # weld, is that arm's kernel text: narrowed to the rows that launched it
    assert "meep_gpu/cuda_kernels/fused_electric_pair.py" in attribution.route_pins
    assert attribution.step_path("meep_gpu/cuda_kernels/fused_electric_pair.py") is None


def test_a_ledger_change_owes_a_dispatch_check_and_never_hides(tmp_path):
    """A ledger is what the dispatch path reads to admit arms: a change to one moves no
    kernel text and may change what dispatches. A row that recorded it is owed a
    dispatch check (a row with no launch counters a re-time), never left valid."""
    root = release_tree(tmp_path)
    ledger = "meep_gpu/cuda_kernels/certification.json"
    held = TR.artifact_drift._sha256(__import__("pathlib").Path(root) / ledger)
    row = ladder_row(512000, "s1", "2026-10-20T10:00:00Z", arms=(ELECTRIC,))
    row["processes"][0][TR.DATA_KEY] = {ledger: held}
    blind = ladder_row(2097152, "s1", "2026-10-20T10:00:00Z", arms=())
    blind["launch_counters"] = []
    blind["processes"][0][TR.DATA_KEY] = {ledger: held}
    # an arm that resolves by its table's whole kernel slice still does not own its
    # table's ledger: it is no kernel text
    triton = "meep_gpu/triton_kernels/fingerprints.json"
    sliced = ladder_row(7077888, "s1", "2026-10-20T10:00:00Z", arms=(TRITON_SINGLE,))
    sliced["processes"][0][TR.DATA_KEY] = {triton: TR.artifact_drift._sha256(
        __import__("pathlib").Path(root) / triton)}
    store = store_of(tmp_path, [row, blind, sliced])
    records = TR.load_store(store)
    assert TR.bill(records, [ledger], root) == {"retime": [blind["key"]],
                                                "check": [row["key"]],
                                                "untouched": [sliced["key"]]}
    assert TR.bill(records, [triton], root)["check"] == [sliced["key"]]
    with open(os.path.join(root, ledger), "w") as handle:
        json.dump(dict(LEDGER, new_run={"source_sha256": {}}), handle)
    with open(os.path.join(root, triton), "w") as handle:
        json.dump({"new_run": {"source_sha256": {}}}, handle)
    selection = TR.select(records, root)
    assert selection[row["key"]]["owed"] == "a dispatch check"
    assert ledger in selection[row["key"]]["assessments"][0]["exempt"]
    assert selection[blind["key"]]["owed"] == "a re-time"
    assert selection[sliced["key"]]["owed"] == "a dispatch check"
    assert selection[sliced["key"]]["assessments"][0]["stale"] == []


# ---------------------------------------------------------------------------
# Shape: one record per table per architecture, added never replaced; missing keys
# ---------------------------------------------------------------------------

def fused_row(tmp_path, store, *, case, architecture, session, ended, verdict="TIMED",
              cells=24000, fused_ms=1.4, base_ms=0.5, device="EXAMPLE GPU",
              pass_index=1, environment=ENVIRONMENT, libraries=None):
    row = fixtures.bench_row(case=case, table="triton", grid=(cells, 1, 1),
                             fused_ms=fused_ms, base_ms=base_ms, utc=ended)
    for plan in row["plans"].values():
        plan["environment"]["device"] = dict(plan["environment"]["device"],
                                             compute_capability=architecture, name=device)
    if verdict != "TIMED":
        row = {"case": case, "drive_table": "triton", "verdict": verdict,
               "error": "TypeError", "utc": ended, "prefer_gpu": True,
               "provenance": row["provenance"]}
    row[TR.IMPORTED_KEY] = imported(ALL_LOADED)
    row[TR.DATA_KEY] = {}
    if libraries is not None:
        row[TR.LIBRARIES_KEY] = libraries
    rows_path = tmp_path / "out" / f"{case}_{architecture}_{session}_{pass_index}" / "rows.jsonl"
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    rows_path.write_text(json.dumps(row) + "\n")
    key = TR.fused_key(architecture, "triton", case, f"res{cells}", pass_index)
    record, line = TR.fused_record(str(rows_path), key=key,
                                   session={"id": session, "started_utc": ended,
                                            "ended_utc": ended},
                                   environment=environment, gates=GATES,
                                   unit={"name": "u", "rc": 0})
    return TR.write_record(store, record, bench_line=line)


RELEASED = {"fused pair D": ["pml_2d", "absorber_1d"], "fused pair B": ["pml_2d"]}


def test_a_partial_set_names_its_missing_keys_instead_of_failing(tmp_path):
    """H1: a released case whose only rows are ERROR rows. The record is cut from the
    rest and names what it lacks, by product and case, with the rows' own reason."""
    root = release_tree(tmp_path)
    store = str(tmp_path / "store")
    for index in (1, 2):
        fused_row(tmp_path, store, case="pml_2d", architecture=ARCH, session=f"s{index}",
                  ended=f"2026-10-2{index}T10:00:00Z", pass_index=index)
    fused_row(tmp_path, store, case="absorber_1d", architecture=ARCH, session="s1",
              ended="2026-10-21T10:00:00Z", verdict="ERROR")
    selection = TR.select(TR.load_store(store), root)
    out = str(tmp_path / "package")
    results = TR.cut_records(selection, out, base=str(tmp_path),
                             released={"triton": RELEASED})
    assert [r.get("refused") for r in results] == [None]
    record = dp.load_record(dp.record_path("triton", ARCH, root=out), "triton")
    assert record["missing"] == [{"product": "fused pair D", "case": "absorber_1d",
                                  "reason": "no timing of this table: verdict ERROR"}]
    assert record["cut"]["account_for_release"] is True
    assert "MISSING FROM THE RELEASE: 1" in open(
        dp.record_path("triton", ARCH, root=out)[:-5] + ".report.txt").read()


def test_a_catalog_key_never_timed_is_named_and_never_dropped(tmp_path):
    """A cut from part of the round: the table names every catalog key with no valid
    measured row, never timed included; the records name every catalog fused key of
    their table and architecture without one, and a table the catalog names that no
    row reaches is named as not cut."""
    root = release_tree(tmp_path)
    timed = ladder_row(512000, "s1", "2026-10-20T10:00:00Z")
    store = store_of(tmp_path, [timed])
    fused_row(tmp_path, store, case="pml_2d", architecture=ARCH, session="s1",
              ended="2026-10-21T10:00:00Z", pass_index=1)
    fused_row(tmp_path, store, case="pml_2d", architecture=ARCH, session="s2",
              ended="2026-10-22T10:00:00Z", pass_index=2)
    catalog = {"schema": TR.CATALOG_SCHEMA, "architecture": ARCH,
               "ladder": {"cases": ["pml_3d"], "sizes": {"priority": [512000, 2097152]}},
               "fused": {"tables": {
                   "triton": {"drive": "release",
                              "sweep": {"cases": ["pml_2d"], "res": [24000, 300],
                                        "passes": 2}},
                   "cuda": {"drive": ["pml_2d"]}}}}
    expected = TR.catalog_keys(catalog, skip_release_drive=True)
    selection = TR.select(TR.load_store(store), root)
    table = TR.cut_ladder_table(selection, ARCH, expected)
    assert [row["key"] for row in table["rows"]] == [timed["key"]]
    assert [item["key"] for item in table["missing"]] == [
        TR.ladder_key(ARCH, "pml_3d", 2097152, "priority")]
    assert table["missing"][0]["why"] == ["no row of this key in the store"]
    assert "Keys without a valid row: 1" in TR.render_table(table)
    out = str(tmp_path / "package")
    results = {r["table"]: r for r in TR.cut_records(
        selection, out, base=str(tmp_path), expected=expected,
        released={"triton": {"fused pair D": ["pml_2d"], "fused pair B": ["pml_2d"]}})}
    assert results["triton"].get("refused") is None
    assert [item.split(":")[0] for item in results["triton"]["missing_keys"]] == [
        TR.fused_key(ARCH, "triton", "pml_2d", "res300", 1),
        TR.fused_key(ARCH, "triton", "pml_2d", "res300", 2)]
    report = open(dp.record_path("triton", ARCH, root=out)[:-5] + ".report.txt").read()
    assert "ROW KEYS WITHOUT A VALID MEASURED ROW: 2" in report and "res300|p2" in report
    assert results["cuda"]["refused"].startswith("not cut: no valid row of cuda")
    assert results["cuda"]["missing_keys"] == [
        TR.fused_key(ARCH, "cuda", "pml_2d", "drive", 1) + ": no row of this key in the store"]
    # the command line takes the catalog and refuses one of another architecture
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(catalog))
    md, js = str(tmp_path / "t.md"), str(tmp_path / "t.json")
    assert TR.main(["table", "--store", store, "--release-root", root, "--catalog",
                    str(path), "--architecture", ARCH, "--out-md", md,
                    "--out-json", js]) == 0
    assert len(json.load(open(js))["missing"]) == 1
    assert TR.main(["table", "--store", store, "--release-root", root, "--catalog",
                    str(path), "--architecture", OTHER_ARCH, "--out-md", md,
                    "--out-json", js]) == 2


def test_a_ratio_of_a_session_without_a_go_variance_does_not_enter(tmp_path):
    """R18: the variance precondition stays in force. A session timed with
    ``--no-variance`` prints its ratios and none of them enters."""
    root = release_tree(tmp_path)
    row = ladder_row(512000, "s1", "2026-10-20T10:00:00Z",
                     gates=gates_with(variance={"ok": None, "not_run": "a smoke"}))
    selection = TR.select(TR.load_store(store_of(tmp_path, [row])), root)
    line = TR.cut_ladder_table(selection, ARCH)["rows"][0]
    assert line["variance"] == "not run: a smoke"
    assert line["ratios"]["cuda"]["ratio"] == pytest.approx(2000.0 / 500.0)
    assert line["ratios"]["cuda"]["enters"] is False
    assert "variance precondition: not run" in line["ratios"]["cuda"]["why"]


def test_a_second_architecture_is_added_beside_the_first_and_never_replaces_it(tmp_path):
    root = release_tree(tmp_path)
    store = str(tmp_path / "store")
    for architecture in (ARCH, OTHER_ARCH):
        for index in (1, 2):
            fused_row(tmp_path, store, case="pml_2d", architecture=architecture,
                      session=f"{architecture}_s{index}",
                      ended=f"2026-10-2{index}T10:00:00Z", pass_index=index)
    out = str(tmp_path / "package")
    released = {"triton": {"fused pair D": ["pml_2d"], "fused pair B": ["pml_2d"]}}
    # the first architecture alone
    only = {k: v for k, v in TR.select(TR.load_store(store), root).items()
            if f"|{ARCH}|" in k}
    TR.cut_records(only, out, base=str(tmp_path), released=released)
    first = dp.record_path("triton", ARCH, root=out)
    before = open(first, "rb").read()
    # then both: the second lands beside it, under its own name
    results = TR.cut_records(TR.select(TR.load_store(store), root), out,
                             base=str(tmp_path), released=released)
    assert sorted(os.path.basename(r["path"]) for r in results) == [
        "timing_cc86.json", "timing_cc90.json"]
    assert open(first, "rb").read() == before, "a re-cut of the same rows is the same file"
    assert dp.record_capabilities(dp.load_record(
        dp.record_path("triton", OTHER_ARCH, root=out))) == (OTHER_ARCH,)
    # a cut whose rows ran on another card of the first architecture is refused
    other_card = str(tmp_path / "store2")
    for index in (1, 2):
        fused_row(tmp_path / "x", other_card, case="pml_2d", architecture=ARCH,
                  session=f"o{index}", ended=f"2026-10-2{index}T10:00:00Z",
                  device="ANOTHER GPU", pass_index=index)
    refused = TR.cut_records(TR.select(TR.load_store(other_card), root), out,
                             base=str(tmp_path), released=released)
    assert "existing_record_names_another_device" in refused[0]["refused"]
    assert open(first, "rb").read() == before


def test_the_command_line_cutter_has_no_override_for_another_device(tmp_path, capsys):
    fixtures.standard_tree(tmp_path)
    out = str(fixtures.out_path(tmp_path))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as handle:
        json.dump({"host": [{"device": "ANOTHER GPU"}]}, handle)
    code = cut.main(["--table", "triton", "--rows", str(tmp_path / "tree"), "--out", out,
                     "--base", str(tmp_path), "--no-account-for-release"])
    assert code == cut.EXIT_REFUSED
    assert "existing_record_names_another_device" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        cut.main(["--table", "triton", "--rows", str(tmp_path / "tree"), "--out", out,
                  "--supersede"])


def test_the_apple_family_names_the_metal_record_and_no_other_table():
    package = os.path.join(REPO_ROOT, "meep_gpu")
    assert dp.record_path("metal", "applegpu_g13s", root=package).endswith(
        os.path.join("metal_kernels", "timing_applegpu_g13s.json"))
    assert dp.record_path("cuda", "9.0", root=package).endswith(
        os.path.join("cuda_kernels", "timing_cc90.json"))
    for table, value in (("metal", "8.6"), ("triton", "applegpu_g13s"),
                         ("cuda", "applegpu_g13s"), ("metal", "AppleGPU")):
        with pytest.raises(dp.ConsultRefused) as refused:
            dp.record_path(table, value)
        assert refused.value.code == "unreadable_capability"
    from meep_gpu import metal_runs  # noqa: PLC0415
    assert dp.APPLE_FAMILY.pattern == metal_runs.ARCHITECTURE_KEY.pattern


# ---------------------------------------------------------------------------
# Realistic: records from what the ladder and the bench write
# ---------------------------------------------------------------------------

def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_ladder_records_read_the_ledger_and_the_row_files(tmp_path):
    ladder = tmp_path / "ladder_s1_u01"
    bench = fixtures.bench_row(case="pml_3d", table="cuda")
    bench["plans"]["fused"]["launch_counters"] = {
        "step_B": {"arm": "cuda:fused magnetic pair", "backend": "cuda", "dispatches": 6}}
    bench["plans"]["fused"]["families"] = {
        "cuda:fused magnetic pair": {"family": "fm", "gate": "cuda_magnetic_gate"}}
    bench[TR.IMPORTED_KEY] = imported(ALL_LOADED)
    bench[TR.DATA_KEY] = {"meep_gpu/cuda_kernels/certification.json": "2" * 64}
    meep_rows = ladder / "meep" / "rows.jsonl"
    write_jsonl(meep_rows, [{"row": "measurement", "row_id": f"r32_p{p}",
                             TR.IMPORTED_KEY: {"parity/meep_gpu/timing_cases.py": "1" * 64}}
                            for p in (1, 2)])
    ledger = []
    for p, (gpu, meep_rate) in enumerate(((2000.0, 500.0), (2020.0, 560.0)), 1):
        gpu_rows = ladder / "gpu_cuda" / f"pass{p}" / "pml_3d_res80" / "rows.jsonl"
        write_jsonl(gpu_rows, [bench])
        ledger += [
            {"ledger": "gpu", "index": 10 * p, "tier": "priority", "case": "pml_3d",
             "utc": f"2026-10-20T10:0{p}:00Z",
             "cells_planned": 512000, "pass": p, "route": "cuda",
             "rows_file": str(gpu_rows), "struck": None,
             "measured": {"legs": {"fused": {"mcell_steps_per_s": gpu, "spread": 0.01}}}},
            {"ledger": "meep", "index": 10 * p + 1, "tier": "priority", "case": "pml_3d",
             "utc": f"2026-10-20T10:0{p}:10Z",
             "cells_planned": 512000, "pass": p, "ranks": 32, "row_id": f"r32_p{p}",
             "rows_file": str(meep_rows), "struck": None,
             "configuration": {"build": "reference", "threads_per_rank": 1},
             "measured": {"mcell_steps_per_s": meep_rate, "spread": 0.02}},
            {"ledger": "array_control", "index": 10 * p + 2, "tier": "priority",
             "utc": f"2026-10-20T10:0{p}:20Z",
             "case": "pml_3d", "cells_planned": 512000, "pass": p,
             "mcell_steps_per_s_mean": 300.0, "sources": [{"spread": 0.03}]},
            {"ledger": "gpu", "index": 1, "tier": "preflight", "case": "pml_3d",
             "cells_planned": 110592, "pass": 0, "route": "cuda"},
        ]
    write_jsonl(ladder / "ladder_rows.jsonl", ledger)
    snapshot = ladder / "tree" / "parity" / "meep_gpu"
    snapshot.mkdir(parents=True)
    (snapshot / "timing_ladder.py").write_text("LADDER = 1\n")
    session = {"id": "s1", "started_utc": "2026-10-20T10:00:00Z",
               "ended_utc": "2026-10-20T11:00:00Z"}
    records = TR.ladder_records(str(ladder), session=session,
                                architecture=ARCH, environment=ENVIRONMENT,
                                gates=GATES, unit={"name": "u", "rc": 0})
    assert [r["key"] for r in records] == [TR.ladder_key(ARCH, "pml_3d", 512000,
                                                         "priority")]
    record = records[0]
    assert record["legs"]["gpu"]["cuda"]["median"] == 2010.0
    assert record["legs"]["meep"]["meep@32"]["median"] == 530.0
    assert record["legs"]["meep"]["meep@32"]["status"] == "ungated"
    assert record["legs"]["array"]["median"] == 300.0
    roles = sorted(p["role"] for p in record["processes"])
    assert roles == ["gpu:cuda:p1", "gpu:cuda:p2", "ladder", "meep:meep@32:p1",
                     "meep:meep@32:p2"]
    ladder_process = next(p for p in record["processes"] if p["role"] == "ladder")
    assert ladder_process[TR.IMPORTED_KEY] == {
        "parity/meep_gpu/timing_ladder.py": sha("LADDER = 1\n")}
    assert all(p[TR.IMPORTED_KEY] for p in record["processes"])
    assert not any(p["ran_package"] for p in record["processes"] if p["role"].startswith(
        "meep:"))
    assert TR.dispatched(record) == [("cuda", "cuda:fused magnetic pair",
                                      "cuda_magnetic_gate")]
    assert "session_mismatch" not in record["gates"] and TR.refusals(record) == []
    gpu = next(p for p in record["processes"] if p["role"] == "gpu:cuda:p1")
    assert gpu[TR.DATA_KEY] == bench[TR.DATA_KEY], "the ledgers ride beside the record"
    path = TR.write_record(str(tmp_path / "store"), record)
    assert TR.load_store(str(tmp_path / "store"))[0]["_path"] == path
    # R5: the same directory read as a LATER session (a ladder resumed across sessions,
    # or collected under bounds it did not run in) carries legs of another session
    later = dict(session, id="s2", started_utc="2026-10-20T10:01:30Z")
    resumed = TR.ladder_records(str(ladder), session=later, architecture=ARCH,
                                environment=ENVIRONMENT, gates=GATES,
                                unit={"name": "u", "rc": 0})[0]
    assert resumed["gates"]["session_mismatch"]
    assert any("another session" in why and "R5" in why for why in TR.refusals(resumed))
    # A STANDALONE ARRAY CONTROL (``--array-control row``) is a process of the row that
    # ran the package; ``gpu_array_control.py`` writes no import record, so the row is
    # refused by name rather than admitted on the GPU processes' records alone
    control_rows = ladder / "array_control" / "pass1" / "pml_3d_res80" / "rows.jsonl"
    write_jsonl(control_rows, [{"kind": "array_control", "verdict": "TIMED"}])
    standalone = [dict(entry, mode="standalone",
                       sources=[{"row_dir": str(control_rows.parent), "spread": 0.03}])
                  if entry.get("ledger") == "array_control" and entry["pass"] == 1
                  else entry for entry in ledger]
    write_jsonl(ladder / "ladder_rows.jsonl", standalone)
    alone = TR.ladder_records(str(ladder), session=session, architecture=ARCH,
                              environment=ENVIRONMENT, gates=GATES,
                              unit={"name": "u", "rc": 0})[0]
    assert "array_control:p1" in [p["role"] for p in alone["processes"]]
    assert any("no import record: process array_control:p1" in why
               for why in TR.refusals(alone))
    # with an import record of its own it is admitted, and its files are read
    write_jsonl(control_rows, [{"kind": "array_control", "verdict": "TIMED",
                                TR.IMPORTED_KEY: imported(ALL_LOADED),
                                TR.DATA_KEY: {}}])
    alone = TR.ladder_records(str(ladder), session=session, architecture=ARCH,
                              environment=ENVIRONMENT, gates=GATES,
                              unit={"name": "u", "rc": 0})[0]
    assert TR.refusals(alone) == []


def test_a_fused_record_keeps_the_bench_line_verbatim(tmp_path):
    store = str(tmp_path / "store")
    path = fused_row(tmp_path, store, case="pml_2d", architecture=ARCH, session="s1",
                     ended="2026-10-21T10:00:00Z")
    line = open(path[:-5] + ".rows.jsonl", "rb").read()
    original = open(tmp_path / "out" / f"pml_2d_{ARCH}_s1_1" / "rows.jsonl", "rb").read()
    assert line == original
    record = json.load(open(path))
    assert record["legs"]["verdict"] == "TIMED"
    assert record["processes"][0][TR.IMPORTED_KEY]
    # a row that stamps another architecture than its key is refused
    other = fused_row(tmp_path, store, case="pml_2d", architecture=OTHER_ARCH,
                      session="s9", ended="2026-10-21T10:00:00Z")
    held = json.load(open(other))
    moved = dict(held, key=held["key"].replace(f"|{OTHER_ARCH}|", f"|{ARCH}|"),
                 session=dict(held["session"], id="s10"))
    moved.pop(TR.DIGEST_FIELD)
    rows_path = tmp_path / "out" / f"pml_2d_{OTHER_ARCH}_s9_1" / "rows.jsonl"
    record, line = TR.fused_record(str(rows_path), key=moved["key"],
                                   session=moved["session"], environment=ENVIRONMENT,
                                   gates={}, unit={})
    assert any(why.startswith("architecture") for why in TR.refusals(record))


def test_the_fused_report_is_a_view_of_the_selected_rows(tmp_path):
    root = release_tree(tmp_path)
    store = str(tmp_path / "store")
    for index in (1, 2):
        fused_row(tmp_path, store, case="pml_2d", architecture=ARCH, session=f"s{index}",
                  ended=f"2026-10-2{index}T10:00:00Z", pass_index=index)
    drive = TR.fused_key(ARCH, "triton", "offdiag_2d", "drive", 1)
    rows_path = tmp_path / "drive" / "rows.jsonl"
    rows_path.parent.mkdir(parents=True)
    row = fixtures.bench_row(case="offdiag_2d", welded=fixtures.ONE_PAIR,
                             bracketed_slots=())
    row[TR.IMPORTED_KEY] = imported(ALL_LOADED)
    row[TR.DATA_KEY] = {}
    row["per_leg"]["array"] = dict(row["per_leg"]["unfused"], median_seconds_per_step=2e-3)
    row["ratios"]["array_over_fused"] = 2e-3 / row["per_leg"]["fused"][
        "median_seconds_per_step"]
    rows_path.write_text(json.dumps(row) + "\n")
    record, line = TR.fused_record(str(rows_path), key=drive,
                                   session={"id": "s1", "started_utc": "a",
                                            "ended_utc": "2026-10-21T10:00:00Z"},
                                   environment=ENVIRONMENT, gates=GATES,
                                   unit={"name": "u", "rc": 0})
    TR.write_record(store, record, bench_line=line)
    out = tmp_path / "report"
    assert TR.main(["report", "--store", store, "--release-root", root,
                    "--architecture", ARCH, "--out-dir", str(out)]) == 0
    staged = (out / "sweep" / "triton" / "pml_2d_res24000" / "rows.jsonl").read_text()
    assert len(staged.splitlines()) == 2, "both passes of the point, in one file"
    assert len((out / "drive_rows.jsonl").read_text().splitlines()) == 1
    assert "offdiag_2d" in (out / "fused_timing.md").read_text()
    assert (out / "missing_keys.txt").read_text() == ""


def test_the_bench_records_the_ledgers_libraries_and_switches_beside_its_import_record(
        monkeypatch):
    """What the 1.0 round's exit hook and its joiner kept beside the import record is
    the bench's own now: the ledgers of every kernel table whose modules the process
    holds, the libraries it loaded, and its switches, a path by its last component (a
    record names no machine)."""
    import bench_fused_products as bench  # noqa: PLC0415

    assert (bench.DATA_KEY, bench.LIBRARIES_KEY, bench.ENVIRONMENT_KEY) == (
        TR.DATA_KEY, TR.LIBRARIES_KEY, TR.ENVIRONMENT_KEY)
    held = bench.admission_data({"meep_gpu/driver.py": "0" * 64,
                                 "meep_gpu/cuda_kernels/arms.py": "0" * 64})
    assert sorted(held) == ["meep_gpu/cuda_kernels/certification.json",
                            "meep_gpu/cuda_kernels/fingerprints.json"]
    assert held["meep_gpu/cuda_kernels/certification.json"] == TR.artifact_drift._sha256(
        __import__("pathlib").Path(REPO_ROOT) / "meep_gpu/cuda_kernels/certification.json")
    assert sorted(TR.DATA_FILES) == sorted(
        f"meep_gpu/{directory}/{name}" for directory, names in bench.TABLE_LEDGERS.items()
        for name in names), "one list of ledgers on both sides"
    libraries = bench.loaded_libraries()
    assert libraries["python"] and libraries["numpy"]
    monkeypatch.setenv("MEEP_GPU_SUBNORMAL_POLICY", "keep")
    monkeypatch.setenv("CUPY_CACHE_DIR", os.path.join(os.sep, "a", "b", "cupy_keep"))
    monkeypatch.setenv("LD_LIBRARY_PATH", os.pathsep.join(
        [os.path.join(os.sep, "x", "stub"), os.path.join(os.sep, "y", "lib")]))
    environment = bench.process_environment()
    assert environment["MEEP_GPU_SUBNORMAL_POLICY"] == "keep"
    assert environment["CUPY_CACHE_DIR"] == "cupy_keep"
    assert environment["LD_LIBRARY_PATH"] == ["stub", "lib"]


# ---------------------------------------------------------------------------
# One environment per cut (R27)
# ---------------------------------------------------------------------------

def rebuilt_environment(tmp_path):
    """A rebuilt reference environment: its build record on disk, and the block a row
    that ran in it records (``cert8.environment_block``)."""
    path = tmp_path / "rebuilt_build_record.json"
    path.write_text(json.dumps({"platform": "linux-64", "rebuilt": True}))
    return str(path), {"reference": {"name": path.name,
                                     "sha256": TR.file_sha256(str(path)),
                                     "record": {"platform": "linux-64"}},
                       "meep_builds": {}}


def test_rows_of_two_environments_never_pool_and_a_rebuild_owes_every_row(
        tmp_path, capsys):
    """R27: an environment is no file of the import record. Rows timed before and after
    a rebuild of the reference environment ran two programs although every import
    record matches the tree: no table, record or report pools them, and naming the
    round's environment makes every row of the other one stale and owed a re-time."""
    root = release_tree(tmp_path)
    path, rebuilt = rebuilt_environment(tmp_path)
    before = ladder_row(512000, "s1", "2026-10-20T10:00:00Z")
    after = ladder_row(2097152, "s2", "2026-10-21T10:00:00Z", environment=rebuilt)
    store = store_of(tmp_path, [before, after])
    # no environment named: both rows valid, and the cut refuses to pool them
    selection = TR.select(TR.load_store(store), root)
    assert {k: v["owed"] for k, v in selection.items()} == {
        before["key"]: "nothing", after["key"]: "nothing"}
    with pytest.raises(TR.RowRefused) as refused:
        TR.cut_ladder_table(selection, ARCH)
    assert refused.value.code == "mixed_environment"
    assert "2 reference environments" in str(refused.value)
    md, js = str(tmp_path / "t.md"), str(tmp_path / "t.json")
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps({"schema": TR.CATALOG_SCHEMA, "architecture": ARCH,
                                   "ladder": {"cases": ["pml_3d"], "sizes": {
                                       "priority": [512000, 2097152]}}}))
    table_args = ["table", "--store", store, "--release-root", root, "--catalog",
                  str(catalog), "--architecture", ARCH, "--out-md", md, "--out-json", js]
    assert TR.main(table_args) == 2
    assert "REFUSED" in capsys.readouterr().out and not os.path.exists(js)
    # the round's environment named: the older row is stale, owed a re-time, and named
    selection = TR.select(TR.load_store(store), root, environment=TR.file_sha256(path))
    stale = selection[before["key"]]
    assert stale["owed"] == "a re-time" and stale["selected"] is None
    assert "reference environment moved (R27)" in stale["assessments"][0]["stale"][0]
    assert selection[after["key"]]["owed"] == "nothing"
    table = TR.cut_ladder_table(selection, ARCH)
    assert [row["key"] for row in table["rows"]] == [after["key"]]
    assert [item["key"] for item in table["missing"]] == [before["key"]]
    assert TR.main(table_args + ["--environment-record", path]) == 0
    assert [item["key"] for item in json.load(open(js))["missing"]] == [before["key"]]
    # a row that recorded none is stale against a named environment, not pooled
    bare = ladder_row(7077888, "s3", "2026-10-22T10:00:00Z",
                      environment={"reference": None})
    selection = TR.select(TR.load_store(store_of(tmp_path / "b", [bare])), root,
                          allow_unrecorded_environment=True,
                          environment=TR.file_sha256(path))
    assert selection[bare["key"]]["owed"] == "a re-time"


def test_fused_records_and_the_report_of_two_environments_are_refused(tmp_path, capsys):
    root = release_tree(tmp_path)
    path, rebuilt = rebuilt_environment(tmp_path)
    store = str(tmp_path / "store")
    fused_row(tmp_path, store, case="pml_2d", architecture=ARCH, session="s1",
              ended="2026-10-21T10:00:00Z", pass_index=1)
    fused_row(tmp_path, store, case="pml_2d", architecture=ARCH, session="s2",
              ended="2026-10-22T10:00:00Z", pass_index=2, environment=rebuilt)
    out = str(tmp_path / "package")
    released = {"triton": {"fused pair D": ["pml_2d"], "fused pair B": ["pml_2d"]}}
    results = TR.cut_records(TR.select(TR.load_store(store), root), out,
                             base=str(tmp_path), released=released)
    assert "2 reference environments" in results[0]["refused"]
    assert not os.path.exists(dp.record_path("triton", ARCH, root=out))
    capsys.readouterr()
    assert TR.main(["report", "--store", store, "--release-root", root,
                    "--architecture", ARCH, "--out-dir", str(tmp_path / "r")]) == 2
    assert "REFUSED mixed_environment" in capsys.readouterr().out
    assert not os.path.exists(tmp_path / "r" / "sweep")
    # in the rebuilt environment the first pass is owed, and the cut names it
    results = TR.cut_records(
        TR.select(TR.load_store(store), root, environment=TR.file_sha256(path)), out,
        base=str(tmp_path), released=released)
    assert results[0]["missing_keys"] == [
        TR.fused_key(ARCH, "triton", "pml_2d", "res24000", 1) + ": owed a re-time"]


def test_rows_whose_processes_loaded_two_versions_of_a_library_never_pool(tmp_path):
    root = release_tree(tmp_path)
    store = str(tmp_path / "store")
    for index, version in ((1, "1.26.4"), (2, "2.0.2")):
        fused_row(tmp_path, store, case="pml_2d", architecture=ARCH, session=f"s{index}",
                  ended=f"2026-10-2{index}T10:00:00Z", pass_index=index,
                  libraries={"python": "3.11.9", "numpy": version})
    released = {"triton": {"fused pair D": ["pml_2d"], "fused pair B": ["pml_2d"]}}
    results = TR.cut_records(TR.select(TR.load_store(store), root),
                             str(tmp_path / "package"), base=str(tmp_path),
                             released=released)
    assert "more than one version of numpy" in results[0]["refused"]
    assert "python" not in results[0]["refused"].split("version of ")[1]


def test_a_dispatch_check_of_another_environment_is_no_witness(tmp_path):
    root = release_tree(tmp_path)
    path, rebuilt = rebuilt_environment(tmp_path)
    magnetic = ladder_row(2097152, "s1", "2026-10-20T10:00:00Z", arms=(MAGNETIC,),
                          environment=rebuilt)
    store = store_of(tmp_path, [magnetic])
    change(root, "meep_gpu/cuda_kernels/fused_electric_pair.py")
    check = dict(magnetic, kind="check", key=TR.check_key(magnetic["key"]),
                 session={"id": "c1", "started_utc": "2026-10-22T10:00:00Z",
                          "ended_utc": "2026-10-22T10:00:00Z"},
                 environment=ENVIRONMENT)
    check["processes"] = [dict(check["processes"][0], **{TR.IMPORTED_KEY: {
        name: TR.artifact_drift._sha256(__import__("pathlib").Path(root) / name)
        for name in ALL_LOADED}})]
    TR.write_record(store, check)
    environment = TR.file_sha256(path)
    item = TR.select(TR.load_store(store), root, environment=environment)[magnetic["key"]]
    assert item["owed"] == "a dispatch check"
    assert item["assessments"][0]["dispatch_check"]["check"] is None
    # the same check in the round's environment is the witness
    TR.write_record(store, dict(check, environment=rebuilt,
                                session=dict(check["session"], id="c2")))
    item = TR.select(TR.load_store(store), root, environment=environment)[magnetic["key"]]
    assert item["owed"] == "nothing"


def test_a_release_drive_table_with_no_sweep_and_no_rows_is_named_not_cut(tmp_path):
    """A catalog table whose DRIVE cases are the release's and that has no sweep names
    no key of its own (the cut accounts for its cases through the record's release
    block): with no row in the store it is still named, as not cut, by the records and
    by the report."""
    root = release_tree(tmp_path)
    store = str(tmp_path / "store")
    for index in (1, 2):
        fused_row(tmp_path, store, case="pml_2d", architecture=ARCH, session=f"s{index}",
                  ended=f"2026-10-2{index}T10:00:00Z", pass_index=index)
    catalog = {"schema": TR.CATALOG_SCHEMA, "architecture": ARCH,
               "fused": {"tables": {
                   "triton": {"drive": "release",
                              "sweep": {"cases": ["pml_2d"], "res": [24000],
                                        "passes": 2}},
                   "cuda": {"drive": "release"}}}}
    expected = TR.catalog_keys(catalog, skip_release_drive=True)
    assert not [key for key in expected if "|cuda|" in key]
    assert TR.catalog_tables(catalog) == [("cuda", ARCH), ("triton", ARCH)]
    released = {"triton": {"fused pair D": ["pml_2d"], "fused pair B": ["pml_2d"]}}
    results = {r["table"]: r for r in TR.cut_records(
        TR.select(TR.load_store(store), root), str(tmp_path / "package"),
        base=str(tmp_path), released=released, expected=expected,
        named=TR.catalog_tables(catalog))}
    assert sorted(results) == ["cuda", "triton"]
    assert results["triton"].get("refused") is None
    assert results["cuda"]["refused"].startswith("not cut: no valid row of cuda")
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(catalog))
    out = tmp_path / "report"
    TR.main(["report", "--store", store, "--release-root", root, "--architecture", ARCH,
             "--catalog", str(path), "--out-dir", str(out)])
    named = (out / "missing_keys.txt").read_text()
    assert f"cuda on {ARCH}: no valid row of this table" in named
    assert f"triton on {ARCH}: its DRIVE cases are the release's" in named
