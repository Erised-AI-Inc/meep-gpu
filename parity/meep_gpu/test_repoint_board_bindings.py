"""The boards' gate bindings are re-pointed from a measurement, never by hand (host-only).

``repoint_board_bindings.py`` reads every ``GATE_BOUND`` entry of the Triton board, and
every ``"gate", "metal_..."`` reference of the Metal board, against a round's results, and
rewrites the table from that measurement. These tests drive it on synthetic results: one
admissible run is chosen, a flush twin, a drifted run, an unreleased run and a second run
of the same gate are not, an entry with no run refuses the write (option A1) or is withheld
by name (option A2), a rerun changes nothing, report mode writes nothing, and every entry of
the committed tables is in the shape the rewrite reads.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
for path in (str(HERE.parents[1]), str(HERE)):
    if path not in sys.path:
        sys.path.insert(0, path)

import build_triton_fusion_matrix as triton_board  # noqa: E402
import repoint_board_bindings as tool  # noqa: E402

STAMP = "2099-01-02_10"
OLD = "2099-01-01_a"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture()
def triton(tmp_path, monkeypatch):
    """An API root holding two credited modules and an empty results tree, the board
    pointed at it."""
    api = tmp_path / "api"
    for family in ("fused_hd_pair", "beta_real_fused_hd_pair"):
        module = api / "meep_gpu" / "triton_kernels" / triton_board.PRODUCTS[family]["module"]
        module.parent.mkdir(parents=True, exist_ok=True)
        module.write_text(f"# {family}\n", encoding="utf-8")
    results = api / "parity" / "meep_gpu" / "results"
    results.mkdir(parents=True)
    monkeypatch.setattr(triton_board, "API", api)
    monkeypatch.setattr(triton_board, "RESULTS", results)
    for family, bound in BOUND.items():          # the runs the board names, on cc 8.6
        _artifact(api, results, bound.split("/")[1], bound.split("/", 2)[2], family)
    return api, results


def _artifact(api: Path, results: Path, root: str, tail: str, family: str, *,
              released=True, drift=False, policy="ieee_keep_ftz_stripped",
              capability="8.6") -> str:
    module = f"meep_gpu/triton_kernels/{triton_board.PRODUCTS[family]['module']}"
    digest = "0" * 64 if drift else _sha(api / module)
    path = results / root / tail
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"device_status": "RUN", "release": {"released": released}, "passed": released,
               "source_sha256": {module: digest}, "subnormal_policy": policy}
    if capability:
        payload["stamped_device"] = {"compute_capability": capability}
    path.write_text(json.dumps(payload), encoding="utf-8")
    return f"results/{root}/{tail}"


BOUND = {"fused_hd_pair": f"results/triton_fused_hd_pair_{OLD}/keep/gate.json",
         "beta_real_fused_hd_pair": f"results/triton_beta_real_fused_hd_pair_{OLD}/keep_probe/gate.json"}


def test_one_admissible_run_is_chosen_and_its_flush_twin_is_not(triton):
    api, results = triton
    chosen = _artifact(api, results, f"triton_fused_hd_pair_{STAMP}", "keep/gate.json",
                       "fused_hd_pair")
    _artifact(api, results, f"triton_fused_hd_pair_{STAMP}_flush", "keep/gate.json",
              "fused_hd_pair", policy="meep_x86_flush")
    _artifact(api, results, f"triton_beta_real_fused_hd_pair_{STAMP}", "keep/gate.json",
              "beta_real_fused_hd_pair")
    measured = tool.measure_triton(triton_board, STAMP, BOUND)
    row = measured["rows"]["fused_hd_pair"]
    assert row["state"] == "repointed" and row["chosen"] == chosen
    assert [c["why_not"] for c in row["candidates"] if not c["admissible"]] == [
        "a flush run (the board binds the keep run)"]
    # a keep_probe/ tail is looked for as keep/ (the retired expansion probe run)
    beta = measured["rows"]["beta_real_fused_hd_pair"]
    assert beta["chosen"] == f"results/triton_beta_real_fused_hd_pair_{STAMP}/keep/gate.json"
    assert measured["counts"] == {"repointed": 2}


@pytest.mark.parametrize("planted", ["drift", "unreleased", "flush_policy"])
def test_a_run_the_floor_would_not_credit_leaves_the_entry_unresolved(triton, planted):
    api, results = triton
    _artifact(api, results, f"triton_fused_hd_pair_{STAMP}", "keep/gate.json",
              "fused_hd_pair", drift=planted == "drift", released=planted != "unreleased",
              policy="meep_x86_flush" if planted == "flush_policy" else "keep")
    row = tool.measure_triton(triton_board, STAMP, {"fused_hd_pair": BOUND["fused_hd_pair"]}
                              )["rows"]["fused_hd_pair"]
    assert row["state"] == "unresolved" and row["chosen"] is None
    why = row["candidates"][0]["why_not"]
    assert {"drift": "recorded file(s) moved", "unreleased": "is not a passed device release",
            "flush_policy": "a flush run"}[planted] in why


def test_two_admissible_runs_of_one_gate_are_never_chosen_between(triton):
    api, results = triton
    for root in (f"triton_fused_hd_pair_{STAMP}", f"triton_fused_hd_pair_{STAMP}_r2"):
        _artifact(api, results, root, "keep/gate.json", "fused_hd_pair")
    measured = tool.measure_triton(triton_board, STAMP, {"fused_hd_pair": BOUND["fused_hd_pair"]})
    assert measured["rows"]["fused_hd_pair"]["state"] == "ambiguous"
    with pytest.raises(SystemExit, match="never chooses between runs"):
        tool.rewrite_triton(_builder_text(), measured, withhold=True)


def test_a_run_of_another_device_is_never_re_pointed_to_silently(triton, tmp_path):
    """A second device's round adds ledger records; it never moves a board onto its runs."""
    api, results = triton

    def stamped(relative: str, capability: str) -> None:
        path = results / relative[len("results/"):]
        payload = json.loads(path.read_text()) if path.is_file() else {
            "release": {"released": True}}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(payload, stamped_device={
            "compute_capability": capability})))

    _artifact(api, results, f"triton_fused_hd_pair_{OLD}", "keep/gate.json", "fused_hd_pair")
    stamped(BOUND["fused_hd_pair"], "8.6")
    new = _artifact(api, results, f"triton_fused_hd_pair_{STAMP}", "keep/gate.json",
                    "fused_hd_pair")
    stamped(new, "9.0")
    bound = {"fused_hd_pair": BOUND["fused_hd_pair"]}
    row = tool.measure_triton(triton_board, STAMP, bound)["rows"]["fused_hd_pair"]
    assert row["state"] == "unresolved" and row["device"] == "cc86"
    assert "ran on cc90" in row["candidates"][0]["why_not"]
    row = tool.measure_triton(triton_board, STAMP, bound, device="cc90")["rows"]["fused_hd_pair"]
    assert row["state"] == "repointed" and row["chosen"] == new
    stamped(new, "8.6")
    assert tool.measure_triton(triton_board, STAMP, bound)["rows"]["fused_hd_pair"][
        "state"] == "repointed"
    # Metal: the architecture each stamp's environment record names.
    metal = tmp_path / "metal_results"
    source = api / "meep_gpu" / "metal_kernels" / "one.py"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("# one\n")
    for stamp, architecture in ((OLD, "applegpu_g13s"), (STAMP, "applegpu_g15s")):
        (metal / f"metal_environment_{stamp}").mkdir(parents=True)
        (metal / f"metal_environment_{stamp}" / "start.json").write_text(
            json.dumps({"architecture": architecture}))
        (metal / f"metal_one_gate_{stamp}").mkdir()
        (metal / f"metal_one_gate_{stamp}" / "gate.json").write_text(json.dumps(
            {"release": {"released": True}, "source_sha256": {str(source): _sha(source)}}))
    text = f'{tool.METAL_ANCHOR}    "a": ("gate", "metal_one_gate_{OLD}"),\n}}\n'
    row = tool.measure_metal(text, metal, STAMP, api)["rows"][f"metal_one_gate_{OLD}"]
    assert row["state"] == "unresolved" and "ran on applegpu_g15s" in row["why_not"]
    row = tool.measure_metal(text, metal, STAMP, api, device="applegpu_g15s")["rows"][
        f"metal_one_gate_{OLD}"]
    assert row["state"] == "repointed"


def test_a_device_that_cannot_be_read_is_refused_never_passed(triton, tmp_path):
    """Neither side's device may be assumed: a bound run that records none needs --device,
    and a run that records none is not credited on a board whose device is known."""
    api, results = triton
    bound = {"fused_hd_pair": BOUND["fused_hd_pair"]}
    new = _artifact(api, results, f"triton_fused_hd_pair_{STAMP}", "keep/gate.json",
                    "fused_hd_pair", capability=None)
    row = tool.measure_triton(triton_board, STAMP, bound)["rows"]["fused_hd_pair"]
    assert row["state"] == "unresolved"
    assert "the device it ran on cannot be read" in row["candidates"][0]["why_not"]
    _artifact(api, results, f"triton_fused_hd_pair_{STAMP}", "keep/gate.json",
              "fused_hd_pair")
    _artifact(api, results, f"triton_fused_hd_pair_{OLD}", "keep/gate.json", "fused_hd_pair",
              capability=None)
    row = tool.measure_triton(triton_board, STAMP, bound)["rows"]["fused_hd_pair"]
    assert row["state"] == "unresolved" and "--device" in row["candidates"][0]["why_not"]
    row = tool.measure_triton(triton_board, STAMP, bound, device="cc86")["rows"][
        "fused_hd_pair"]
    assert row["state"] == "repointed" and row["chosen"] == new
    # Metal: a bound stamp with no environment record, and a run whose stamp has none.
    metal = tmp_path / "metal_results"
    source = api / "meep_gpu" / "metal_kernels" / "one.py"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("# one\n")
    (metal / f"metal_one_gate_{STAMP}").mkdir(parents=True)
    (metal / f"metal_one_gate_{STAMP}" / "gate.json").write_text(json.dumps(
        {"release": {"released": True}, "source_sha256": {str(source): _sha(source)}}))
    text = f'{tool.METAL_ANCHOR}    "a": ("gate", "metal_one_gate_{OLD}"),\n}}\n'
    row = tool.measure_metal(text, metal, STAMP, api)["rows"][f"metal_one_gate_{OLD}"]
    assert row["state"] == "unresolved" and "--device" in row["why_not"]
    row = tool.measure_metal(text, metal, STAMP, api, device="applegpu_g13s")["rows"][
        f"metal_one_gate_{OLD}"]
    assert row["state"] == "unresolved" and "cannot be read" in row["why_not"]
    (metal / f"metal_environment_{STAMP}").mkdir()
    (metal / f"metal_environment_{STAMP}" / "start.json").write_text(
        json.dumps({"architecture": "applegpu_g13s"}))
    row = tool.measure_metal(text, metal, STAMP, api, device="applegpu_g13s")["rows"][
        f"metal_one_gate_{OLD}"]
    assert row["state"] == "repointed"


def _builder_text() -> str:
    return Path(triton_board.__file__).read_text(encoding="utf-8")


def _measured(rows) -> dict:
    return {"stamp": STAMP, "rows": rows}


def _row(current, state, chosen):
    return {"current": current, "state": state, "chosen": chosen, "candidates": []}


def test_the_rewrite_writes_the_measured_pointers_into_the_committed_builder():
    text = _builder_text()
    table, _node = tool._assigned(text, "GATE_BOUND")
    families = sorted(table)
    rows = {f: _row(table[f], "repointed", f"results/round_{STAMP}/{f}/gate.json")
            for f in families}
    written, summary = tool.rewrite_triton(text, _measured(rows), withhold=False,
                                           today="2099-01-02")
    assert summary["repointed"] == len(families) and summary["withheld"] == []
    tool.verify_triton(written, _measured(rows), [])
    assert "REPOINTED 2099-01-02 TO THE RUNS OF ROUND ``2099-01-02_10``" in written
    # The written module still parses, and every other assignment is unchanged.
    after, _node = tool._assigned(written, "GATE_BOUND")
    assert after == {f: rows[f]["chosen"] for f in families}
    assert tool._assigned(written, "PRODUCTS")[0] == tool._assigned(text, "PRODUCTS")[0]
    # Measured again on the written table, nothing moves and nothing is written.
    again = {f: _row(after[f], "unchanged", after[f]) for f in families}
    same, summary = tool.rewrite_triton(written, _measured(again), withhold=False)
    assert same == written and summary["changed"] is False


def test_an_entry_with_no_run_refuses_the_write_or_is_withheld_by_name():
    text = _builder_text()
    table, _node = tool._assigned(text, "GATE_BOUND")
    families = sorted(table)
    rows = {f: _row(table[f], "unresolved", None) if "hd_pair" in f
            else _row(table[f], "repointed", f"results/round_{STAMP}/{f}/gate.json")
            for f in families}
    with pytest.raises(SystemExit, match="no admissible run of the round"):
        tool.rewrite_triton(text, _measured(rows), withhold=False)
    written, summary = tool.rewrite_triton(text, _measured(rows), withhold=True)
    unresolved = sorted(f for f in families if "hd_pair" in f)
    assert summary["withheld"] == unresolved
    tool.verify_triton(written, _measured(rows), unresolved)
    withheld, _node = tool._assigned(written, "CREDIT_WITHHELD_STALE_BINDING")
    assert all(why.startswith(tool.WITHHELD_REASON_LEAD) for why in withheld.values())
    # A rerun on the same measurement writes nothing; once every entry resolves, the
    # tool's own withheld entries are removed.
    after, _node = tool._assigned(written, "GATE_BOUND")
    rerun = {f: _row(after[f], "unchanged", after[f]) if rows[f]["chosen"]
             else _row(after[f], "unresolved", None) for f in families}
    same, summary = tool.rewrite_triton(written, _measured(rerun), withhold=True)
    assert same == written and summary["changed"] is False
    resolved = {f: _row(after[f], "unchanged", after[f]) if rows[f]["chosen"]
                else _row(after[f], "repointed", f"results/later_{f}/gate.json")
                for f in families}
    cleared, _summary = tool.rewrite_triton(written, _measured(resolved), withhold=False)
    tool.verify_triton(cleared, _measured(resolved), [])
    assert tool._assigned(cleared, "CREDIT_WITHHELD_STALE_BINDING")[0] == {}


def test_a_withheld_entry_a_person_wrote_is_never_overwritten():
    text = _builder_text().replace(
        tool.WITHHELD_ANCHOR + "}\n",
        tool.WITHHELD_ANCHOR + '    "fused_hd_pair": "measured drift, by hand",\n}\n', 1)
    table, _node = tool._assigned(text, "GATE_BOUND")
    rows = {f: _row(table[f], "unchanged", table[f]) for f in table}
    with pytest.raises(SystemExit, match="did not write"):
        tool.rewrite_triton(text, _measured(rows), withhold=True)


def test_every_committed_gate_bound_entry_is_in_the_shape_the_rewrite_reads():
    text = _builder_text()
    start = text.index(tool.GATE_BOUND_ANCHOR)
    block = text[start:text.index("\n}\n", start) + 3]
    table, _node = tool._assigned(text, "GATE_BOUND")
    for family, value in table.items():
        assert len(tool._literal_pattern(family, value).findall(block)) == 1, family
        parts = value.split("/")
        assert parts[0] == "results" and len(parts) >= 3, value
        assert tool.split_stamp(parts[1])[1], f"{family}: {parts[1]} carries no stamp"


def test_report_mode_writes_nothing(triton, tmp_path, monkeypatch, capsys):
    api, results = triton
    copy = tmp_path / "build_triton_fusion_matrix.py"
    shutil.copyfile(triton_board.__file__, copy)
    monkeypatch.setattr(triton_board, "__file__", str(copy))
    before = _sha(copy)
    assert tool.main(["triton", "--stamp", STAMP]) == 1          # nothing of the round: A1
    assert "would REFUSE" in capsys.readouterr().out
    assert tool.main(["triton", "--stamp", STAMP, "--withhold-unrun",
                      "--json", str(tmp_path / "m.json")]) == 0
    assert _sha(copy) == before
    assert "REPORT ONLY" in capsys.readouterr().out
    assert json.loads((tmp_path / "m.json").read_text())["builder_sha256_after"] == before


# --------------------------------------------------------------------------- Metal ----
def _metal_text() -> str:
    return (f'CENSUS = RESULTS / ("metal_coverage_{OLD}")\n'
            'PRODUCTS = {"b": {"cells": [{"binding": ("gate",\n'
            f'    "metal_cell_gate_{OLD}")}}]}}}}\n'
            f"# a note\n{tool.METAL_ANCHOR}"
            f'    "a": ("gate", "metal_one_gate_{OLD}"),\n'
            f'    "a2": (\n        "gate", "metal_one_gate_{OLD}"),\n'
            f'    "c": ("fingerprint", "metal_c_device_gate"),\n'
            "}\n")


def _metal_run(results: Path, api: Path, name: str, *, released=True, drift=False) -> None:
    source = api / "meep_gpu" / "metal_kernels" / f"{name}.py"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(f"# {name}\n", encoding="utf-8")
    digest = "0" * 64 if drift else _sha(source)
    for stamp in (OLD, STAMP):                   # both stamps ran on the one certified Mac
        environment = results / f"metal_environment_{stamp}"
        environment.mkdir(parents=True, exist_ok=True)
        (environment / "start.json").write_text(json.dumps({"architecture": "applegpu_g13s"}))
    (results / f"{name}_{STAMP}").mkdir(parents=True)
    (results / f"{name}_{STAMP}" / "gate.json").write_text(json.dumps({
        "release": {"released": released}, "source_sha256": {str(source): digest}}))


def test_metal_references_move_to_the_round_and_nothing_else_does(tmp_path):
    results, api = tmp_path / "results", tmp_path
    for name in ("metal_one_gate", "metal_cell_gate"):
        _metal_run(results, api, name)
    text = _metal_text()
    measured = tool.measure_metal(text, results, STAMP, api)
    assert measured["references"] == 3 and measured["counts"] == {"repointed": 2}
    written, summary = tool.rewrite_metal(text, measured, today="2099-01-02")
    tool.verify_metal(written, measured)
    assert summary == {"repointed": 2, "references_moved": 3, "unchanged": 0, "changed": True}
    assert written.count(f"_{STAMP}") == 3 + 1            # three references, one note
    assert f'"metal_coverage_{OLD}"' in written           # not a gate reference
    assert '"metal_c_device_gate"' in written             # a ledger binding
    assert f"# REPOINTED 2099-01-02 from `_{OLD}` to `_{STAMP}`" in written
    again = tool.measure_metal(written, results, STAMP, api)
    assert again["counts"] == {"unchanged": 2}
    assert tool.rewrite_metal(written, again) == (written, {
        "repointed": 0, "references_moved": 0, "unchanged": 2, "changed": False})


@pytest.mark.parametrize("planted", ["absent", "unreleased", "drift"])
def test_a_metal_reference_without_a_released_zero_drift_run_refuses(tmp_path, planted):
    results, api = tmp_path / "results", tmp_path
    _metal_run(results, api, "metal_cell_gate")
    if planted != "absent":
        _metal_run(results, api, "metal_one_gate", released=planted != "unreleased",
                   drift=planted == "drift")
    measured = tool.measure_metal(_metal_text(), results, STAMP, api)
    assert measured["rows"][f"metal_one_gate_{OLD}"]["state"] == "unresolved"
    with pytest.raises(SystemExit, match="have no released, zero-drift run"):
        tool.rewrite_metal(_metal_text(), measured)


def test_every_committed_metal_gate_reference_is_dated_and_the_census_is_not_one():
    text = (HERE / "build_fusion_matrix.py").read_text(encoding="utf-8")
    refs = [m.group(2) for m in tool.METAL_GATE_REF.finditer(text)]
    assert refs and all(tool.split_stamp(ref)[1] for ref in refs)
    assert not [ref for ref in refs if ref.startswith("metal_coverage_")]
    assert text.count(tool.METAL_ANCHOR) == 1
