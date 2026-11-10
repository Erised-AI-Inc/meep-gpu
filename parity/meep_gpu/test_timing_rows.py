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
    (root / "meep_gpu/cuda_kernels/fingerprints.json").write_text(json.dumps(LEDGER))
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
    """MEEP's own pass spread strikes nothing. A 12 % spread enters the
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
    assert "no reference environment build record" in selection[bare["key"]][
        "assessments"][0]["refused"][0]
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
    """An idle box as evidence: a row is cut only when its unit opened on a quiet window
    of two clear readings and was watched while it ran. A row collected by hand has
    neither and is kept, never cut; so is one whose window saw an intrusion."""
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
        assert any("otherwise idle box" in why
                   for why in item["assessments"][0]["refused"]), name
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
    ledger = "meep_gpu/cuda_kernels/fingerprints.json"
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
# Shape: one record per table, architecture and card, never replacing another card's; missing keys
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
    """A released case whose only rows are ERROR rows (as a harness defect left eleven
    cases of the 2026-10-13 fused-product timing). The record is cut from the rest and
    names what it lacks, by product and case, with the rows' own reason."""
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
    """A cut from part of a timing run: the table names every catalog key with no valid
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
    """The variance precondition stays in force. A session timed with
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
    """A second architecture lands at its own name, and a second CARD of the first
    architecture beside the first card's record, under its own card name: the first card
    stays the architecture's representative, at the name the reader opens, byte for
    byte. A re-cut of the second card rewrites its own file."""
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
    assert {r["card"] for r in results} == {"EXAMPLE GPU"}
    assert open(first, "rb").read() == before, "a re-cut of the same rows is the same file"
    assert dp.record_capabilities(dp.load_record(
        dp.record_path("triton", OTHER_ARCH, root=out))) == (OTHER_ARCH,)
    # a cut whose rows ran on another card of the first architecture lands beside the
    # first card's record, under its own card name, and replaces nothing
    other_card = str(tmp_path / "store2")
    for index in (1, 2):
        fused_row(tmp_path / "x", other_card, case="pml_2d", architecture=ARCH,
                  session=f"o{index}", ended=f"2026-10-2{index}T10:00:00Z",
                  device="ANOTHER GPU", pass_index=index)
    beside, = TR.cut_records(TR.select(TR.load_store(other_card), root), out,
                             base=str(tmp_path), released=released)
    card_file = os.path.join(os.path.dirname(first), "timing_cc86-another_gpu.json")
    assert beside.get("refused") is None, beside
    assert beside["path"] == card_file and beside["card"] == "ANOTHER GPU"
    assert dp.record_path("triton", ARCH, root=out) == first
    assert open(first, "rb").read() == before
    assert open(card_file[:-5] + ".report.txt").read().splitlines()[4] == (
        "card: ANOTHER GPU (architecture 8.6)")
    held = open(card_file, "rb").read()
    # a re-cut of the second card, from a store with one more pass, rewrites its own file
    fused_row(tmp_path / "x", other_card, case="pml_2d", architecture=ARCH, session="o3",
              ended="2026-10-23T10:00:00Z", device="ANOTHER GPU", pass_index=3)
    again, = TR.cut_records(TR.select(TR.load_store(other_card), root), out,
                            base=str(tmp_path), released=released)
    assert again["path"] == card_file and again.get("refused") is None
    assert open(card_file, "rb").read() != held
    assert open(first, "rb").read() == before
    assert sorted(name for name in os.listdir(os.path.dirname(first))
                  if name.endswith(".json")) == ["timing_cc86-another_gpu.json",
                                                 "timing_cc86.json", "timing_cc90.json"]
    # a third card whose name has the second's slug is another card, and is refused
    collides = str(tmp_path / "store3")
    for index in (1, 2):
        fused_row(tmp_path / "y", collides, case="pml_2d", architecture=ARCH,
                  session=f"c{index}", ended=f"2026-10-2{index}T10:00:00Z",
                  device="Another-GPU", pass_index=index)
    held = open(card_file, "rb").read()
    refused, = TR.cut_records(TR.select(TR.load_store(collides), root), out,
                              base=str(tmp_path), released=released)
    assert refused["refused"].startswith("existing_record_names_another_device"), refused
    assert card_file in refused["refused"], "the refusal names the file it is about"
    assert open(card_file, "rb").read() == held and open(first, "rb").read() == before


def test_an_unreadable_file_at_the_architectures_name_refuses_the_cut(tmp_path):
    """A file at the representative's name that cannot say whose record it is could be
    this card's own, damaged: the cut is written neither over it nor beside it."""
    root = release_tree(tmp_path)
    store = str(tmp_path / "store")
    for index in (1, 2):
        fused_row(tmp_path, store, case="pml_2d", architecture=ARCH, session=f"s{index}",
                  ended=f"2026-10-2{index}T10:00:00Z", pass_index=index)
    out = str(tmp_path / "package")
    bare = dp.record_path("triton", ARCH, root=out)
    os.makedirs(os.path.dirname(bare))
    with open(bare, "w") as handle:
        handle.write("not a record")
    refused, = TR.cut_records(TR.select(TR.load_store(store), root), out,
                              base=str(tmp_path), released={"triton": RELEASED})
    assert refused["refused"].startswith("existing_record_unreadable"), refused
    assert refused["path"] == bare
    assert os.listdir(os.path.dirname(bare)) == ["timing_cc86.json"]


def test_a_store_whose_rows_name_two_cards_is_refused_and_writes_nothing(tmp_path):
    """One store, one card: the rows of one table and architecture selected from a store
    that holds two cards' rows are not one card's evidence, and the cut says so."""
    root = release_tree(tmp_path)
    store = str(tmp_path / "store")
    for case, device in (("pml_2d", "EXAMPLE GPU"), ("absorber_1d", "ANOTHER GPU")):
        for index in (1, 2):
            fused_row(tmp_path, store, case=case, architecture=ARCH,
                      session=f"{case}_s{index}", ended=f"2026-10-2{index}T10:00:00Z",
                      device=device, pass_index=index)
    out = tmp_path / "package"
    refused, = TR.cut_records(TR.select(TR.load_store(store), root), str(out),
                              base=str(tmp_path), released={"triton": RELEASED})
    assert refused["refused"].startswith("mixed_device"), refused
    assert "card" not in refused
    assert not out.exists()


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
    bench[TR.DATA_KEY] = {"meep_gpu/cuda_kernels/fingerprints.json": "2" * 64}
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
    # One session: the same directory read as a LATER session (a ladder resumed across
    # sessions, or collected under bounds it did not run in) carries legs of another session
    later = dict(session, id="s2", started_utc="2026-10-20T10:01:30Z")
    resumed = TR.ladder_records(str(ladder), session=later, architecture=ARCH,
                                environment=ENVIRONMENT, gates=GATES,
                                unit={"name": "u", "rc": 0})[0]
    assert resumed["gates"]["session_mismatch"]
    assert any("another session" in why and "timed in one session" in why
               for why in TR.refusals(resumed))
    # A standalone array control (``--array-control row``) is a process of the row that
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
    """What the 1.0 timing run's exit hook and its joiner kept beside the import record is
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
# One environment per cut
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
    """An environment is no file of the import record. Rows timed before and after
    a rebuild of the reference environment ran two programs although every import
    record matches the tree: no table, record or report pools them, and naming the
    timing run's reference environment makes every row of the other one stale and owed a re-time."""
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
    # the timing run's environment named: the older row is stale, owed a re-time, and named
    selection = TR.select(TR.load_store(store), root, environment=TR.file_sha256(path))
    stale = selection[before["key"]]
    assert stale["owed"] == "a re-time" and stale["selected"] is None
    assert "reference environment moved" in stale["assessments"][0]["stale"][0]
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
    # the same check in the timing run's reference environment is the witness
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


# ---------------------------------------------------------------------------
# The gate of record: the 5 % gate is the between-pass spread; the window spread is recorded
# ---------------------------------------------------------------------------
# The ``r33`` helpers and test names below (``r33_passes``, ``r33_row``, ``r33_table``,
# ``pre_r33_leg``) are named for the same 2026-10-16 change as ``timing_rows.r33_strike``.

#: The strike a ladder wrote, before the between-pass gate took effect (2026-10-16), on
#: a pass the bench made TIMED-NOT-REPORTABLE for its within-pass window spread alone
#: (``timing_records.record_gpu``).
WINDOW_STRIKE = "TIMED-NOT-REPORTABLE: floors failed ['spread_within_gate']"


def r33_passes(rates, windows, struck=None):
    return [{"pass": index + 1, "rate": rate, "spread": window,
             "struck": (struck or {}).get(index + 1)}
            for index, (rate, window) in enumerate(zip(rates, windows))]


def pre_r33_leg(passes):
    """A GPU leg as a ladder collected it before the between-pass gate took effect
    (2026-10-16): struck by any struck pass."""
    reasons = [f"pass {p['pass']}: {p['struck']}" for p in passes if p.get("struck")]
    rates = [p["rate"] for p in passes]
    return {"median": sorted(rates)[len(rates) // 2], "passes": passes,
            "between_pass_spread": TR.spread(rates),
            "status": "struck" if reasons else "passed",
            "gate": ("struck by the 5 % gate: " + "; ".join(reasons)) if reasons
            else "passed the 5 % gate"}


def r33_row(cells, session, gpu, meep=None):
    record = ladder_row(cells, session, "2026-10-20T10:00:00Z")
    record["legs"] = {"gpu": {"cuda": gpu}, "array": None,
                      "meep": {"meep@32": meep or meep_leg((500.0, 505.0, 495.0), 32)}}
    return record


def r33_table(tmp_path, records):
    root = release_tree(tmp_path)
    selection = TR.select(TR.load_store(store_of(tmp_path, records)), root)
    return {row["cells"]: row for row in TR.cut_ladder_table(selection, ARCH)["rows"]}


def test_a_row_whose_windows_jitter_10_percent_but_whose_passes_agree_within_1_percent_enters(
        tmp_path):
    """Passes 2000 / 2010 / 1995 (0.75 % between passes), windows 10 % / 2 % / 10 %.
    The row enters whether its ladder struck the jittery passes (written before the
    between-pass gate took effect, 2026-10-16) or not (written after), and the window
    spreads are printed beside it."""
    rates, windows = (2000.0, 2010.0, 1995.0), (0.10, 0.02, 0.10)
    before = r33_row(512000, "s1", pre_r33_leg(
        r33_passes(rates, windows, struck={1: WINDOW_STRIKE, 3: WINDOW_STRIKE})))
    after = r33_row(2097152, "s1", TR.leg_median(r33_passes(rates, windows), "gpu"))
    rows = r33_table(tmp_path, [before, after])
    for cells in (512000, 2097152):
        leg, ratio = rows[cells]["gpu"]["cuda"], rows[cells]["ratios"]["cuda"]
        assert leg["status"] == "passed" and leg["gate"] == "passed the 5 % gate", cells
        assert leg["between_pass_spread"] == pytest.approx(15.0 / 2000.0, rel=1e-5)
        assert leg["window_spread_max"] == pytest.approx(0.10)
        assert [p["spread"] for p in leg["passes"]] == list(windows)
        assert ratio["enters"] is True and ratio["ratio"] == pytest.approx(2000.0 / 500.0)
    assert rows[512000]["gpu"]["cuda"]["pre_r33_window_strikes_read_unstruck"] == [1, 3]
    assert rows[512000]["gpu"]["cuda"]["passes"][0]["struck_as_recorded"] == WINDOW_STRIKE


def test_a_row_whose_passes_disagree_by_6_percent_is_struck(tmp_path):
    """Passes 2000 / 2120 / 2060 spread 5.8 % between passes, with every window
    within 1 %: the leg is struck by the gate, printed with its median, and its ratio
    does not enter."""
    leg = TR.leg_median(r33_passes((2000.0, 2120.0, 2060.0), (0.01, 0.01, 0.01)), "gpu")
    assert leg["status"] == "struck" and leg["median"] == 2060.0
    assert "between-pass spread 0.0583 > 0.05" in leg["gate"]
    row = r33_table(tmp_path, [r33_row(512000, "s1", leg)])[512000]
    assert row["gpu"]["cuda"]["status"] == "struck"
    assert row["ratios"]["cuda"]["enters"] is False
    assert "struck by the 5 % gate: between-pass spread" in row["ratios"]["cuda"]["why"]


def test_meep_is_never_struck_for_its_spread_under_r33(tmp_path):
    """MEEP's own spread, between passes (12 %) or inside them (up to 20 %), strikes
    nothing: its leg is ungated, it is the row's best, and the ratio over it enters."""
    meep = dict(TR.leg_median(r33_passes((440.0, 500.0, 460.0), (0.15, 0.20, 0.12)),
                              "meep"), ranks=32, control=None)
    assert meep["status"] == "ungated" and TR.MEEP_SPREAD_RULE in meep["gate"]
    gpu = TR.leg_median(r33_passes((2000.0, 2010.0, 1995.0), (0.01,) * 3), "gpu")
    row = r33_table(tmp_path, [r33_row(512000, "s1", gpu, meep=meep)])[512000]
    assert row["meep"]["meep@32"]["status"] == "ungated"
    assert row["meep"]["meep@32"]["between_pass_spread"] == pytest.approx(60.0 / 460.0,
                                                                          rel=1e-5)
    assert row["meep_best"] == "meep@32" and row["ratios"]["cuda"]["enters"] is True
    assert row["ratios"]["cuda"]["ratio"] == pytest.approx(2000.0 / 460.0, rel=1e-4)


def test_the_window_spread_still_appears_in_the_record_and_the_table(tmp_path):
    """Every pass of a record keeps its fused leg's window spread and, since the
    between-pass gate took effect (2026-10-16), every leg's; the cut keeps them per pass
    with their largest, and the table prints them."""
    ladder = tmp_path / "ladder_s1_u01"
    bench = fixtures.bench_row(case="pml_3d", table="cuda")
    bench[TR.IMPORTED_KEY] = imported(ALL_LOADED)
    bench[TR.DATA_KEY] = {}
    ledger = []
    for p, (rate, window) in enumerate(((2000.0, 0.09), (2010.0, 0.03)), 1):
        rows_file = ladder / "gpu_cuda" / f"pass{p}" / "pml_3d_res80" / "rows.jsonl"
        write_jsonl(rows_file, [bench])
        ledger.append({"ledger": "gpu", "index": p, "tier": "priority", "case": "pml_3d",
                       "utc": f"2026-10-20T10:0{p}:00Z", "cells_planned": 512000,
                       "pass": p, "route": "cuda", "rows_file": str(rows_file),
                       "struck": None,
                       "measured": {"legs": {
                           "fused": {"mcell_steps_per_s": rate, "spread": window},
                           "unfused": {"mcell_steps_per_s": 900.0, "spread": 0.07},
                           "array": {"mcell_steps_per_s": 30.0, "spread": 0.02}}}})
    write_jsonl(ladder / "ladder_rows.jsonl", ledger)
    session = {"id": "s1", "started_utc": "2026-10-20T10:00:00Z",
               "ended_utc": "2026-10-20T11:00:00Z"}
    record, = TR.ladder_records(str(ladder), session=session, architecture=ARCH,
                                environment=ENVIRONMENT, gates=GATES,
                                unit={"name": "u", "rc": 0})
    stored = record["legs"]["gpu"]["cuda"]["passes"]
    assert [p["spread"] for p in stored] == [0.09, 0.03]
    assert stored[0]["window_spreads"] == {"array": 0.02, "fused": 0.09, "unfused": 0.07}
    record["legs"]["meep"] = {"meep@32": meep_leg((500.0, 505.0), 32)}
    row = r33_table(tmp_path, [record])[512000]
    leg = row["gpu"]["cuda"]
    assert leg["status"] == "passed" and leg["window_spread_max"] == pytest.approx(0.09)
    assert leg["passes"][1]["window_spreads"]["unfused"] == 0.07
    root = release_tree(tmp_path)
    table = TR.cut_ladder_table(TR.select(TR.load_store(str(tmp_path / "store")), root),
                                ARCH)
    assert TR.GATE_RULE in table["rules"]["gate_rule"] and table["rules"]["pre_r33"]
    text = TR.render_table(table)
    assert "| fused window spreads |" in text and "| 9.00 % / 3.00 % |" in text
    assert "the between-pass gate (2026-10-16)" in text


def test_a_row_cuts_to_one_verdict_whichever_side_of_r33_it_was_written_on(tmp_path):
    """The cut reads the passes, never the status a ladder baked into the record: a row
    collected before the between-pass gate took effect (2026-10-16; window-only strikes,
    its leg struck) and one collected after (no such strike) with the same rates and
    window spreads read the same median, spreads, status, gate and ratio. A strike naming
    another floor keeps that floor; every other strike stands; reading a strike twice
    changes nothing."""
    rates, windows = (1360.6, 1363.2, 1364.7), (0.042, 0.015, 0.007)
    passes = r33_passes(rates, windows, struck={1: WINDOW_STRIKE, 2: WINDOW_STRIKE})
    old, new = pre_r33_leg(passes), TR.leg_median(r33_passes(rates, windows), "gpu")
    assert old["status"] == "struck", "the status the old ladder baked in"
    rows = r33_table(tmp_path, [r33_row(512000, "s1", old), r33_row(2097152, "s1", new)])
    fields = ("median", "between_pass_spread", "window_spread_max", "status", "gate")
    read = {cells: {name: rows[cells]["gpu"]["cuda"][name] for name in fields}
            for cells in rows}
    assert read[512000] == read[2097152] and read[512000]["status"] == "passed"
    ratios = {cells: dict(rows[cells]["ratios"]["cuda"]) for cells in rows}
    assert ratios[512000] == ratios[2097152] and ratios[512000]["enters"] is True
    both = "TIMED-NOT-REPORTABLE: floors failed ['bit_identical', 'spread_within_gate']"
    assert TR.r33_strike(both) == "TIMED-NOT-REPORTABLE: floors failed ['bit_identical']"
    assert TR.r33_strike(TR.r33_strike(both)) == TR.r33_strike(both)
    assert TR.r33_strike(WINDOW_STRIKE) is None
    for other in ("the fused leg is not bit-identical to the array leg",
                  "TIMED-NOT-REPORTABLE: floors failed TypeError: boom",
                  "host conditions: cpu_state_before: on battery"):
        assert TR.r33_strike(other) == other
    standalone = ("the standalone row is TIMED-NOT-REPORTABLE: floors "
                  "{'windows_launched_nothing': True, 'spread_within_gate': False}")
    assert TR.r33_strike(standalone) is None
    struck = TR.leg_median(r33_passes(rates, windows, struck={2: both}), "gpu")
    assert struck["status"] == "struck" and "['bit_identical']" in struck["gate"]
    assert TR.leg_median(struck["passes"], "gpu") == struck, "a read leg reads the same"


def test_a_fused_row_struck_for_its_window_spread_alone_reads_reportable(tmp_path):
    """A bench row whose only failed floor is the window floor: its record's
    legs read as passing the floors that gate, and the dispatch-preference record cut
    from it admits it (the gate of record, ``cut_timing_record``)."""
    store = str(tmp_path / "store")
    path = fused_row(tmp_path, store, case="pml_2d", architecture=ARCH, session="s1",
                     ended="2026-10-21T10:00:00Z")
    line = json.loads(open(path[:-5] + ".rows.jsonl").read())
    old = dict(line, reportable=False, verdict="TIMED-NOT-REPORTABLE",
               floors=dict(line["floors"], spread_within_gate=False))
    assert TR.reportable(old) and not TR.reportable(
        dict(old, floors=dict(old["floors"], bit_identical=False)))
    assert TR.gating_floors(old) == {k: v for k, v in line["floors"].items()
                                     if k != "spread_within_gate"}
    rows_path = tmp_path / "old" / "rows.jsonl"
    rows_path.parent.mkdir(parents=True)
    rows_path.write_text(json.dumps(old) + "\n")
    record, _line = TR.fused_record(str(rows_path), key=TR.fused_key(
        ARCH, "triton", "pml_2d", "res24000", 2), session={
            "id": "s2", "started_utc": "a", "ended_utc": "2026-10-22T10:00:00Z"},
        environment=ENVIRONMENT, gates=GATES, unit={"name": "u", "rc": 0})
    assert {leg["gate"] for leg in record["legs"]["per_leg"].values()} == {
        "passed the bench's floors"}
    assert cut.admission_refusal(old, "triton", False) is None


def test_ladder_records_open_the_row_files_a_neutral_ledger_names(tmp_path):
    """The ladder's ledger leaves the machine, so its recorder writes every path under
    its root's label (``timing_records.append``); the records are cut on the machine that
    ran it, and open each row file through ``neutral_ids.resolve``."""
    import neutral_ids  # noqa: PLC0415
    import timing_records  # noqa: PLC0415

    ladder = tmp_path / "ladder_s1_u01"
    bench = fixtures.bench_row(case="pml_3d", table="cuda")
    bench[TR.IMPORTED_KEY] = imported(ALL_LOADED)
    gpu_rows = ladder / "gpu_cuda" / "pass1" / "pml_3d_res80" / "rows.jsonl"
    write_jsonl(gpu_rows, [bench])
    meep_rows = ladder / "meep" / "rows.jsonl"
    write_jsonl(meep_rows, [{"row": "measurement", "row_id": "r32_p1",
                             TR.IMPORTED_KEY: {"parity/meep_gpu/timing_cases.py": "1" * 64}}])
    common = {"tier": "priority", "case": "pml_3d", "cells_planned": 512000, "pass": 1,
              "run_dir": str(ladder), "struck": None}
    ledger = str(ladder / "ladder_rows.jsonl")
    timing_records.append(ledger, dict(common, ledger="gpu", index=1, route="cuda",
                                       utc="2026-10-20T10:01:00Z", rows_file=str(gpu_rows),
                                       measured={"legs": {"fused": {
                                           "mcell_steps_per_s": 2000.0, "spread": 0.01}}}))
    timing_records.append(ledger, dict(common, ledger="meep", index=2, ranks=32,
                                       row_id="r32_p1", utc="2026-10-20T10:01:10Z",
                                       rows_file=str(meep_rows),
                                       configuration={"build": "reference"},
                                       measured={"mcell_steps_per_s": 500.0,
                                                 "spread": 0.02}))
    written = [json.loads(line) for line in open(ledger, encoding="utf-8")]
    assert written[0]["rows_file"] == neutral_ids.path(str(gpu_rows))
    assert neutral_ids.residue(written) == []
    session = {"id": "s1", "started_utc": "2026-10-20T10:00:00Z",
               "ended_utc": "2026-10-20T11:00:00Z"}
    record, = TR.ladder_records(str(ladder), session=session, architecture=ARCH,
                                environment=ENVIRONMENT, gates=GATES,
                                unit={"name": "u", "rc": 0})
    artifacts = {p["role"]: p["artifact"] for p in record["processes"]}
    assert artifacts == {"gpu:cuda:p1": "gpu_cuda/pass1/pml_3d_res80/rows.jsonl",
                         "meep:meep@32:p1": "meep/rows.jsonl"}
    assert all(p[TR.IMPORTED_KEY] for p in record["processes"]), \
        "every row file the neutral ledger names was opened"


# ---------------------------------------------------------------------------
# The split ladder: one row from a GPU half and a MEEP half
# ---------------------------------------------------------------------------

def write_ladder(ladder, *, gpu=True, meep=True, utc=None, driver="LADDER = 1\n",
                 rates=((2000.0, 500.0), (2020.0, 560.0)), case="pml_3d", tier="priority",
                 cells=512000, architecture=None):
    """A ladder directory as ``timing_ladder.py`` writes one: its ledger, the GPU rows'
    bench lines and the MEEP rows' measurement lines, and its snapshot's ladder driver.
    ``gpu`` and ``meep`` say which sides it timed: a whole ladder times both, a split
    unit's halves one each (``--ranks none``, ``--routes none``)."""
    bench = fixtures.bench_row(case=case, table="cuda")
    for plan in bench["plans"].values():
        if architecture:
            plan["environment"]["device"]["compute_capability"] = architecture
    bench["plans"]["fused"]["launch_counters"] = {
        "step_B": {"arm": "cuda:fused magnetic pair", "backend": "cuda", "dispatches": 6}}
    bench["plans"]["fused"]["families"] = {
        "cuda:fused magnetic pair": {"family": "fm", "gate": "cuda_magnetic_gate"}}
    bench[TR.IMPORTED_KEY] = imported(ALL_LOADED)
    bench[TR.DATA_KEY] = {"meep_gpu/cuda_kernels/fingerprints.json": "2" * 64}
    meep_rows = ladder / "meep" / "rows.jsonl"
    if meep:
        write_jsonl(meep_rows, [{"row": "measurement", "row_id": f"r32_p{p}",
                                 TR.IMPORTED_KEY: {"parity/meep_gpu/timing_cases.py": "1" * 64}}
                                for p in (1, 2)])
    ledger = []
    for p, (gpu_rate, meep_rate) in enumerate(rates, 1):
        when = utc or f"2026-10-20T10:0{p}:00Z"
        common = {"tier": tier, "case": case, "utc": when, "cells_planned": cells, "pass": p}
        if gpu:
            gpu_rows = ladder / "gpu_cuda" / f"pass{p}" / f"{case}_res80" / "rows.jsonl"
            write_jsonl(gpu_rows, [bench])
            ledger += [
                dict(common, ledger="gpu", index=10 * p, route="cuda",
                     rows_file=str(gpu_rows), struck=None,
                     measured={"legs": {"fused": {"mcell_steps_per_s": gpu_rate,
                                                  "spread": 0.01}}}),
                dict(common, ledger="array_control", index=10 * p + 2,
                     mcell_steps_per_s_mean=300.0, sources=[{"spread": 0.03}])]
        if meep:
            ledger.append(dict(common, ledger="meep", index=10 * p + 1, ranks=32,
                               row_id=f"r32_p{p}", rows_file=str(meep_rows), struck=None,
                               configuration={"build": "reference", "threads_per_rank": 1},
                               measured={"mcell_steps_per_s": meep_rate, "spread": 0.02}))
    write_jsonl(ladder / "ladder_rows.jsonl", ledger)
    snapshot = ladder / "tree" / "parity" / "meep_gpu"
    snapshot.mkdir(parents=True, exist_ok=True)
    (snapshot / "timing_ladder.py").write_text(driver)
    return str(ladder)


SPLIT_SESSION = {"id": "s1", "started_utc": "2026-10-20T10:00:00Z",
                 "ended_utc": "2026-10-20T11:00:00Z"}


def held(side, mask):
    """A half as the session records it: run once, exited 0, its mask set, read back and
    the only one its readings saw."""
    return {"side": side, "unit": f"u01_ladder_pml_3d_priority_{side}",
            "ladder": f"ladder_s1_u01_{side}", "attempt": 1, "rc": 0,
            "started_utc": "2026-10-20T10:00:30Z", "ended_utc": "2026-10-20T10:30:00Z",
            "vm_affinity": {"declared": mask, "before": "0xffffff", "set": "ok",
                            "read_back": mask, "read": [mask]}}


def split_block(order=("gpu", "meep")):
    masks = {"gpu": "0x999988", "meep": "0xffffff"}
    return {"rule": "two child runs", "order": list(order),
            "halves": [held(side, masks[side]) for side in order]}


#: The gates of a split session: its declared VM affinity (the Windows side's rule) and
#: the masks it held for its GPU and MEEP halves.
SPLIT_GATES = dict(GATES, windows={"applied": True, "host": "WSL2", "vm_affinity": "0x999988"},
                   split_ladder={"gpu": "0x999988", "meep": "0xffffff",
                                 "meep_source": "cleared: every one of Windows' 24 logical "
                                                "CPUs"})


def split_record(tmp_path, split=None, meep_driver="LADDER = 1\n", meep=True, gates=None):
    gpu = write_ladder(tmp_path / "ladder_s1_u01_gpu", meep=False)
    half = write_ladder(tmp_path / "ladder_s1_u01_meep", gpu=False, meep=meep,
                        driver=meep_driver)
    records = TR.split_ladder_records(gpu, half, session=SPLIT_SESSION, architecture=ARCH,
                                      environment=ENVIRONMENT,
                                      gates=SPLIT_GATES if gates is None else gates,
                                      unit={"name": "u01", "rc": 0},
                                      split=split_block() if split is None else split)
    return records


def test_a_split_unit_composes_one_row_of_the_whole_ladders_shape(tmp_path):
    """The GPU half's legs and the MEEP half's legs, each read where its ladder wrote it,
    make the record one ladder of both sides makes, plus the ``split`` block: the same
    key, legs, launch counters and processes (each artifact under its half's directory),
    and the drivers once."""
    whole, = TR.ladder_records(write_ladder(tmp_path / "ladder_s1_u01"),
                               session=SPLIT_SESSION, architecture=ARCH,
                               environment=ENVIRONMENT, gates=GATES,
                               unit={"name": "u01", "rc": 0})
    record, = split_record(tmp_path)
    assert set(record) == set(whole) | {"split"}
    assert record["key"] == whole["key"] and record["legs"] == whole["legs"]
    assert record["launch_counters"] == whole["launch_counters"]
    assert record["legs"]["gpu"]["cuda"]["median"] == 2010.0
    assert record["legs"]["meep"]["meep@32"]["median"] == 530.0
    assert record["legs"]["array"]["median"] == 300.0
    roles = sorted(p["role"] for p in record["processes"])
    assert roles == sorted(p["role"] for p in whole["processes"])
    artifacts = {p["role"]: p["artifact"] for p in record["processes"]}
    assert artifacts["gpu:cuda:p1"] == "ladder_s1_u01_gpu/gpu_cuda/pass1/pml_3d_res80/rows.jsonl"
    assert artifacts["meep:meep@32:p2"] == "ladder_s1_u01_meep/meep/rows.jsonl"
    assert artifacts["ladder"] == "drivers"
    assert record["split"]["halves_with_rows"] == ["gpu", "meep"]
    assert record["split"]["drivers_differ"] == []
    assert record["split"]["order"] == ["gpu", "meep"]
    assert TR.refusals(record) == []
    path = TR.write_record(str(tmp_path / "store"), record)
    assert TR.load_store(str(tmp_path / "store"))[0]["_path"] == path


def _split_without(field, value):
    split = split_block()
    if field == "order":
        split["halves"] = split["halves"][:1]
        split["not_run"] = {"side": "meep", "why": "the gpu half exited rc 4"}
        return split, "the halves run were ['gpu'], not one GPU half and one MEEP half"
    half = split["halves"][1 if field.startswith("meep") else 0]
    name = field.split(":", 1)[1]
    if name == "rc":
        half["rc"] = value
        return split, f"split: the {half['side']} half exited rc {value}"
    half["vm_affinity"] = dict(half["vm_affinity"], **{name: value})
    if name == "read":
        return split, f"saw the VM affinity {value}, not its declared"
    if name == "declared":
        return split, "the MEEP half ran under the GPU half's VM affinity 0x999988"
    return split, f"split: the {half['side']} half's VM affinity was not held"


@pytest.mark.parametrize("field, value", [
    ("order", None), ("gpu:rc", 4), ("meep:rc", 1),
    ("gpu:set", "error Win32Exception 5"), ("meep:read_back", "0x0"),
    ("gpu:read", ["0x999988", "0xffffff"]), ("meep:read", ["0x999988", "0xffffff"]),
    ("meep:declared", "0x999988")])
def test_a_split_row_is_refused_unless_both_halves_held_their_own_masks(
        tmp_path, field, value):
    """Each clause by name: one half run (the other not, after a failure), a half that
    exited non-zero, a mask whose set failed or did not read back, a reading that saw
    another mask during a half, the MEEP half under the GPU half's mask."""
    split, named = _split_without(field, value)
    if field == "meep:declared":
        split["halves"][1]["vm_affinity"]["read_back"] = value
        split["halves"][1]["vm_affinity"]["read"] = [value]
    record, = split_record(tmp_path, split)
    why = TR.refusals(record)
    assert any(named in reason for reason in why), why


def test_a_split_row_whose_halves_ran_other_drivers_or_wrote_no_row_is_refused(tmp_path):
    record, = split_record(tmp_path / "drivers", meep_driver="LADDER = 2\n")
    assert record["split"]["drivers_differ"] == ["parity/meep_gpu/timing_ladder.py"]
    assert any("ran different ladder drivers: parity/meep_gpu/timing_ladder.py" in why
               for why in TR.refusals(record))
    record, = split_record(tmp_path / "rows", meep=False)
    assert record["split"]["halves_with_rows"] == ["gpu"]
    assert record["legs"]["meep"] == {}
    assert any("the meep half recorded no row of this key" in why
               for why in TR.refusals(record))


def _session_gates(declared="0x999988", gpu="0x999988", meep="0xffffff", masks=True):
    gates = dict(GATES, windows={"applied": True, "host": "WSL2", "vm_affinity": declared})
    if declared is None:
        gates["windows"] = {"applied": False, "skipped": "not a WSL2 host"}
    if masks:
        gates["split_ladder"] = {"gpu": gpu, "meep": meep, "meep_source": "cleared"}
    return gates


def _halves_at(gpu, meep):
    split = split_block()
    for half, mask in zip(split["halves"], (gpu, meep)):
        half["vm_affinity"] = dict(half["vm_affinity"], declared=mask, read_back=mask,
                                   read=[mask])
    return split


@pytest.mark.parametrize("halves, gates, named", [
    # The GPU half under the cleared mask and the MEEP half under the declared one, each
    # set, read back and read alone: the masks swapped between the sides.
    (("0xffffff", "0x999988"), {}, [
        "split: the gpu half ran under VM affinity 0xffffff, not the session's gpu mask "
        "0x999988",
        "split: the meep half ran under VM affinity 0x999988, not the session's meep mask "
        "0xffffff",
        "split: the GPU half ran under VM affinity 0xffffff, not the session's declared VM "
        "affinity 0x999988",
        "split: the MEEP half ran under the session's declared VM affinity 0x999988, under "
        "which MEEP is never timed"]),
    # The session's own masks agree with the halves, and neither GPU mask is the declared
    # one: the MEEP half ran under the declared mask, the GPU half under another.
    (("0x1", "0x999988"), {"gpu": "0x1", "meep": "0x999988"}, [
        "split: the GPU half ran under VM affinity 0x1, not the session's declared VM "
        "affinity 0x999988",
        "split: the MEEP half ran under the session's declared VM affinity 0x999988, under "
        "which MEEP is never timed"]),
    # A MEEP half under a mask that is not the one the session held for MEEP.
    (("0x999988", "0xfffff0"), {}, [
        "split: the meep half ran under VM affinity 0xfffff0, not the session's meep mask "
        "0xffffff"]),
    # No masks recorded by the session, or no declared mask to bind the GPU half to.
    (("0x999988", "0xffffff"), {"masks": False}, [
        "split: the session recorded no split ladder masks (gates.split_ladder), so neither "
        "half's mask is bound to the session's"]),
    (("0x999988", "0xffffff"), {"declared": None}, [
        "split: the GPU half ran under VM affinity 0x999988, not the session's declared VM "
        "affinity None"]),
])
def test_a_split_row_is_refused_unless_each_half_ran_under_the_sessions_mask(
        tmp_path, halves, gates, named):
    """Each half's mask is bound to the session's, not only to itself: the GPU half under
    the declared VM affinity and the session's GPU mask, the MEEP half under the session's
    MEEP mask and never under the declared one. Each clause by name, and no other split
    refusal."""
    record, = split_record(tmp_path, _halves_at(*halves), gates=_session_gates(**gates))
    why = [reason for reason in TR.refusals(record) if reason.startswith("split:")]
    assert why == named, why
    bound, = split_record(tmp_path / "bound", _halves_at("0x999988", "0xffffff"))
    assert TR.refusals(bound) == []


def test_every_ratio_of_a_split_row_is_labelled_split(tmp_path):
    """The table cut from a split row is the table of a whole one, every ratio labelled
    split with the two masks and the order; a table with no split row carries no label
    and no split rule."""
    root = release_tree(tmp_path)
    plain = ladder_row(512000, "s0", "2026-10-20T09:00:00Z")
    record, = split_record(tmp_path)
    record["processes"] = plain["processes"]   # the synthetic release tree's own modules
    table = TR.cut_ladder_table(TR.select(TR.load_store(store_of(tmp_path, [record])), root),
                                ARCH)
    line, = table["rows"]
    label = ("split: the GPU legs under VM affinity 0x999988 and the MEEP legs under "
             "0xffffff, timed as two child runs of one session, gpu then meep")
    assert line["split"] == label and table["rules"]["split"] == TR.SPLIT_RULE
    assert line["ratios"]["cuda"]["ratio"] == pytest.approx(2010.0 / 530.0, abs=1e-4)
    assert line["ratios"]["cuda"]["split"] is True and line["ratios"]["cuda"]["enters"]
    assert line["ratios"]["cuda"]["why"].endswith(label)
    text = TR.render_table(table)
    assert "(split)" in text and "Split rows: " + TR.SPLIT_RULE in text
    whole = TR.cut_ladder_table(TR.select(TR.load_store(store_of(tmp_path / "w", [plain])),
                                          root), ARCH)
    assert "split" not in whole["rules"] and "split" not in whole["rows"][0]
    assert not any("split" in ratio for ratio in whole["rows"][0]["ratios"].values())
    assert "(split)" not in TR.render_table(whole)


# ---------------------------------------------------------------------------
# What a record may name, and a split unit as the ledger builders read it
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text, local", [
    ("/" + "ho" + "me" + "/someone/env/bin/python", True),     # a home, by its shape
    ("-I" + "/" + "Us" + "ers" + "/someone/env/include", True),
    ("C:\\" + "Us" + "ers" + "\\someone\\x", True),               # a Windows profile
    ("~/runs/x", True),
    ("/scratch/project/envs/ref/bin/python", True),               # under no label
    ("x GPU-3f1c2a9e-5b7d-4e21-9a0c-6d8e2f4b1a37 y", True),       # a card's UUID
    ("<home>/miniforge3/envs/ref/bin/python", False),
    ("-L<home>/env/lib -Wl,-rpath,<home>/env/lib", False),
    ("--prefix=<home>//miniforge3/envs/ref", False),             # a labelled path, continued
    ("/usr/lib/x86_64-linux-gnu/libm.so.6", False),               # a system directory
    ("/proc/meminfo", False),
    ("Report bugs to https://www.open-mpi.org/community/help/", False),
    ("ladder_s1_u01_gpu/gpu_cuda/pass1/rows.jsonl", False),       # relative
    ("1695, 9501, P2", False),
])
def test_local_paths_finds_what_names_a_machines_files(text, local):
    """A row record's check (``local_paths``): an absolute path outside the system
    directories, a home directory, a path under ``~/`` and what neutral_ids would rewrite
    are found, as a value and as a key; the labels, the system directories, a URL and a
    relative path are not."""
    assert TR.local_paths({"value": text}) == ([text] if local else [])
    assert TR.local_paths({text: 1}) == ([text] if local else [])
    assert TR.local_paths([[text]]) == ([text] if local else [])


def test_a_host_name_in_a_record_is_a_local_name(monkeypatch):
    import neutral_ids  # noqa: PLC0415

    monkeypatch.setattr(neutral_ids, "_host_names", lambda: ("labdesk7",))
    assert TR.local_paths({"note": "timed on labdesk7"}) == ["timed on labdesk7"]


def _written(tmp_path, record, name="store"):
    return TR.write_record(str(tmp_path / name), record)


def test_split_rows_reads_the_unit_its_records_name(tmp_path):
    """The records of a split unit, any of its keys: the halves' ladders, the masks, the
    order and the label, bound to the session's masks; refused by name otherwise."""
    record, = split_record(tmp_path / "a")
    unit = TR.split_rows([_written(tmp_path, record)])
    assert unit["ladders"] == {"gpu": "ladder_s1_u01_gpu", "meep": "ladder_s1_u01_meep"}
    assert unit["masks"] == {"gpu": "0x999988", "meep": "0xffffff"}
    assert unit["order"] == ["gpu", "meep"] and unit["session"] == "s1"
    assert unit["label"] == TR.split_label(record["split"]) and unit["rule"] == TR.SPLIT_RULE
    assert unit["keys"] == [record["key"]] and unit["record_sha256"] == [
        TR.seal(record)[TR.DIGEST_FIELD]]


def _refusal(paths):
    with pytest.raises(TR.RowRefused) as refused:
        TR.split_rows(paths)
    return refused.value.code, str(refused.value)


def test_split_rows_refuses_by_name(tmp_path):
    record, = split_record(tmp_path / "a")
    good = _written(tmp_path, record)
    assert _refusal([])[0] == "no_split_row"
    # a whole ladder's row
    whole, = TR.ladder_records(write_ladder(tmp_path / "w"), session=SPLIT_SESSION,
                               architecture=ARCH, environment=ENVIRONMENT, gates=GATES,
                               unit={"name": "u01", "rc": 0})
    assert _refusal([_written(tmp_path, whole, "whole")])[0] == "not_a_split_row"
    # the halves not under the session's masks
    swapped, = split_record(tmp_path / "b", _halves_at("0xffffff", "0x999988"))
    code, text = _refusal([_written(tmp_path, swapped, "swapped")])
    assert code == "split_row_refused" and "not the session's gpu mask 0x999988" in text
    # edited after it was written
    edited = json.load(open(good))
    edited["split"]["order"] = ["meep", "gpu"]
    path = tmp_path / "edited.json"
    path.write_text(json.dumps(edited))
    assert _refusal([str(path)])[0] == "record_digest_mismatch"
    # two records of different units
    other, = split_record(tmp_path / "c", split_block(order=("meep", "gpu")))
    code, text = _refusal([good, _written(tmp_path, other, "other")])
    assert code == "split_rows_differ"


@pytest.mark.parametrize("change, named", [
    ({"watcher": dict(SPLIT_GATES["watcher"], intrusions=[
        {"utc": "2026-10-20T10:10:00Z", "why": "a foreign process on the device"}])},
     "intruded: the watcher saw 1 intrusion(s)"),
    ({"quiet": []}, "no quiet window of two clear readings"),
    ({"watcher": {}}, "no watcher readings while the unit ran"),
    ({"session_mismatch": ["ladder_rows.jsonl:3"]},
     "another session: 1 ledger entr(y/ies) of this row were written outside"),
])
def test_split_rows_refuses_what_every_cut_of_the_store_refuses(tmp_path, change, named):
    """A split row whose masks hold but that no cut of the row store would take (an
    intruded watcher, no quiet window, no watcher readings, ledger entries of another
    session) reaches no builder's table either: refused by name, with the cut's reason."""
    record, = split_record(tmp_path / "a")
    assert TR.split_rows([_written(tmp_path, record, "clean")])["keys"] == [record["key"]]
    record["gates"] = dict(record["gates"], **change)
    assert any(named in reason for reason in TR.refusals(record))
    code, text = _refusal([_written(tmp_path, record)])
    assert code == "split_row_refused" and named in text


def test_rule_two_does_not_walk_through_the_table_completion():
    # Through the registry every arm would own every kernel, and any kernel edit would
    # re-time every row of the table.
    attribution = TR.Attribution(REPO_ROOT)
    owned = attribution.kernel_deps(["meep_gpu/metal_kernels/launch.py"])
    assert "meep_gpu/metal_kernels/arms.py" in owned
    assert "meep_gpu/metal_kernels/registry.py" not in owned
    assert "meep_gpu/metal_kernels/ade_update_p.py" not in owned


def test_rule_two_keeps_every_metal_arm_its_own_module():
    # An entry may pin launch.py as its only kernel file, so its family module is reached
    # through its gate script once the walk stops at the registry. Read off the live table,
    # because a row's arm owning its own text is what the rule is for.
    from meep_gpu import metal_dispatch  # noqa: PLC0415
    from meep_gpu.metal_kernels import arms  # noqa: PLC0415

    arms.ensure_registered()
    attribution = TR.Attribution(REPO_ROOT)
    missing, checked = [], 0
    for spec in arms.registered():
        row = metal_dispatch.ARM_CERTIFICATION.get(spec.label)
        if row is None:
            continue
        checked += 1
        owned, _rule = attribution.arm_files("metal", row[1])
        module = spec.coverage.__module__.replace(".", "/") + ".py"
        if owned is not None and module not in owned:
            missing.append((spec.label, module))
    assert checked > 100
    assert missing == []


def test_rule_two_still_walks_a_real_edge_between_two_families():
    # The planted control: the ADE arm imports the shared spine it builds from.
    attribution = TR.Attribution(REPO_ROOT)
    owned = attribution.kernel_deps(["meep_gpu/metal_kernels/fused_ade_chain.py"])
    assert "meep_gpu/metal_kernels/fused_ade_chain.py" in owned
    assert "meep_gpu/metal_kernels/ade_update_p.py" in owned


def test_status_names_the_moved_ledger_digests_no_state_answers_for():
    selection = {"k": {"owed": "a dispatch check", "assessments": [
        {"status": "check_owed", "session": "s", "refused": [], "stale": [], "exempt": {},
         "set_aside": {"meep_gpu/cuda_kernels/fingerprints.json": {"why": "slice_equal",
                                                                   "recorded": "a" * 64},
                       "meep_gpu/cuda_kernels/certification.json": {"why": "unread",
                                                                    "recorded": "c" * 64}},
         "compared_whole": {"meep_gpu/triton_kernels/fingerprints.json": {
             "why": "ledger_state_not_archived", "recorded": "b" * 64}}}]}}
    last = TR._status_lines(selection)[-1]  # noqa: SLF001
    assert last.startswith("ledger states not archived: 1 of 2 moved ledger digests")
    assert "b" * 64 in last and "a" * 64 not in last
