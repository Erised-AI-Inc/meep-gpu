"""Where a Metal H->D lift child runs, what it reads, and which rows a host cannot lift.

``lift_host`` replaced the census's RECORDED interpreter (an absolute path on the machine
that cut the census) with the gate's own, made the four input files of a lift basis
required by name, and declared, as data, what a few corpus rows need beyond MEEP. These
checks need no Apple GPU and no MEEP:

* the rule: a row runs under this interpreter whatever the census recorded, and keeps the
  recorded path as ``recorded_interpreter``; a row's needs come from its lift record;
* the probe, run for real in a child against stand-in modules: present, absent, a MEEP
  build feature reported False, an unreadable answer, an undeclared need;
* the seven lift sites (the six gates that keep their own ``leg_lift`` and
  ``hd_tail_gate_common``) driven on a synthetic census with ``subprocess.run`` replaced
  so nothing is executed: what each launches, that no path the census recorded reaches a
  child, that a row with an absent need is named ``unlifted_on_this_host`` and counted
  out instead of launched, that an unreadable probe fails the leg by name, and that a
  missing input stops the basis by name; then with every launched child answering a
  passing record: each leg passes on the rows it asked about with one row counted out,
  and fails when every row of its cell is unlifted (a leg that asked about nothing);
* the complex gate, which reads the same four inputs in process, refuses a missing one by
  name;
* the source guard: no Metal gate or kit launches a recorded interpreter, and the one
  file in the harness that still does is named.

The gates set their subnormal policy in the environment when imported, so the gate checks
run in a child process with torch made unimportable.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
API_ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import lift_host  # noqa: E402

STOCK = "corpus_final_numpy_2026-08-09_monitorfix"
GDSII = "corpus_final_numpy_2026-08-09_monitorfix_gdsii"
SIGMA = "corpus_final_numpy_2026-08-09_monitorfix_sigma"
RECORDED = "/nonexistent/census-host/envs/stock/bin/python"
RECORDED_SIDE = "/nonexistent/census-host/envs/side/bin/python"
#: The sites that launch lift children, and the one harness file that still launches the
#: interpreter a census row recorded (a probe no round runs; it is not changed here).
SITES = ("gate_metal_bfast_fused_hd_pair.py", "gate_metal_conductive_fused_hd_pair.py",
         "gate_metal_cylindrical_real_fused_hd_pair.py", "gate_metal_folded_fused_hd_pair.py",
         "gate_metal_fused_hd_pair.py", "gate_metal_fused_hd_pair_nonlinear.py",
         "hd_tail_gate_common.py")
RECORDED_LAUNCHER = "probe_h_to_d_withdraw_order.py"


# ------------------------------------------------------------------------ the rule ----
def test_a_row_runs_under_this_interpreter_whatever_the_census_recorded():
    for record, seam in (({"interpreter": RECORDED, "lift_record": STOCK}, {}),
                         ({"python": RECORDED}, {}),
                         ({}, {"interpreter": RECORDED_SIDE, "lift_record": GDSII}),
                         ({}, {})):
        fields = lift_host.row_fields(record, seam)
        assert fields["interpreter"] == sys.executable, fields
        assert fields["recorded_interpreter"] == (record.get("interpreter")
                                                  or record.get("python")
                                                  or seam.get("interpreter")), fields


def test_a_rows_needs_are_its_lift_records():
    assert lift_host.row_fields({"lift_record": GDSII}, {})["lift_needs"] == [
        "gdspy", "libGDSII"]
    assert lift_host.row_fields({}, {"lift_record": SIGMA})["lift_needs"] == [
        "PyMieScatt", "nlopt", "autograd"]
    for record in (STOCK, STOCK + "_blocked", None, "corpus_some_other_leg"):
        assert lift_host.row_fields({"lift_record": record}, {})["lift_needs"] == []


def test_every_declared_need_has_a_probe():
    declared = {need for needs in lift_host.LIFT_RECORD_NEEDS.values() for need in needs}
    assert declared == set(lift_host.NEEDS)
    assert {kind for kind, _name in lift_host.NEEDS.values()} == {"module", "meep"}


def test_the_four_inputs_are_named_and_a_missing_one_is_found(tmp_path):
    inputs = lift_host.lift_inputs("census_x", "seam_x")
    assert inputs == ["census_x/examples.jsonl", "census_x/tests.jsonl",
                      "census_x/tests_param_matched.jsonl", "seam_x/h_to_d_seam.jsonl"]
    for rel in inputs:
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text("", encoding="utf-8")
    assert lift_host.missing_inputs(tmp_path, "census_x", "seam_x") == []
    (tmp_path / "census_x" / "tests_param_matched.jsonl").unlink()
    assert lift_host.missing_inputs(tmp_path, "census_x", "seam_x") == [
        "census_x/tests_param_matched.jsonl"]
    assert "census_x/tests_param_matched.jsonl" in lift_host.missing_reason(
        tmp_path, ["census_x/tests_param_matched.jsonl"])


# ----------------------------------------------------------------------- the probe ----
def _stand_ins(directory: Path, libgdsii: bool) -> dict:
    """A directory holding a stand-in ``meep`` (with_libGDSII answering ``libgdsii``) and a
    stand-in ``gdspy``, and an environment that puts it first on the path."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "meep.py").write_text(
        f"def with_libGDSII():\n    return {libgdsii!r}\n", encoding="utf-8")
    (directory / "gdspy.py").write_text("", encoding="utf-8")
    return {**os.environ, "PYTHONPATH": str(directory)}


def test_the_probe_answers_present_absent_and_a_build_feature(tmp_path, monkeypatch):
    monkeypatch.setitem(lift_host.NEEDS, "absent_module",
                        ("module", "lift_host_no_such_module"))
    environment = _stand_ins(tmp_path / "site", libgdsii=False)
    probed = lift_host.probe(["gdspy", "libGDSII", "absent_module"],
                             environment=environment, cwd=tmp_path)
    assert probed["unreadable"] is None and probed["interpreter"] == sys.executable
    assert probed["needs"]["gdspy"] is None
    assert probed["needs"]["libGDSII"] == "meep.with_libGDSII() is False in this MEEP build"
    assert probed["needs"]["absent_module"].startswith("ModuleNotFoundError")
    present = lift_host.probe(["gdspy", "libGDSII"],
                              environment=_stand_ins(tmp_path / "site2", libgdsii=True))
    assert present["needs"] == {"gdspy": None, "libGDSII": None}


def test_an_unreadable_probe_and_an_undeclared_need_are_named(tmp_path):
    silent = tmp_path / "python"
    silent.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    silent.chmod(0o755)
    probed = lift_host.probe(["gdspy"], executable=str(silent))
    assert probed["needs"] == {} and "printed no answer" in probed["unreadable"]
    absent = lift_host.probe(["gdspy"], executable=str(tmp_path / "absent"))
    assert "did not finish" in absent["unreadable"]
    undeclared = lift_host.probe(["no_such_need"])
    assert "declared in no probe" in undeclared["unreadable"]


def test_no_child_is_launched_when_no_row_needs_anything(monkeypatch):
    def refuse(*_args, **_kwargs):
        raise AssertionError("a probe was launched for rows that need nothing")

    monkeypatch.setattr(lift_host.subprocess, "run", refuse)
    host = lift_host.host_check([{"label": "examples:a.py", "lift_needs": []}])
    assert host["unlifted_on_this_host"] == {} and host["unreadable"] is None


def test_a_row_with_an_absent_need_is_named_with_the_reason():
    rows = [{"label": "examples:coupler.py", "lift_record": GDSII,
             "lift_needs": ["gdspy", "libGDSII"]},
            {"label": "examples:mie.py", "lift_record": SIGMA,
             "lift_needs": ["PyMieScatt", "nlopt", "autograd"]},
            {"label": "tests:T.t", "lift_record": None, "lift_needs": []}]
    probed = {"interpreter": "/env/bin/python",
              "needs": {"gdspy": None, "libGDSII": "meep.with_libGDSII() is False",
                        "PyMieScatt": None, "nlopt": None, "autograd": None}}
    named = lift_host.unlifted(rows, probed)
    assert list(named) == ["examples:coupler.py"]
    assert "libGDSII (meep.with_libGDSII() is False)" in named["examples:coupler.py"]
    assert GDSII in named["examples:coupler.py"]
    entry = lift_host.unlifted_entry(rows[0], named["examples:coupler.py"])
    assert entry["measured"] is False and entry["unlifted_on_this_host"]


# ---------------------------------------------------------------- the seven sites ----
_NO_TORCH = textwrap.dedent('''
    import importlib.abc, sys
    class _NoTorch(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name == "torch" or name.startswith("torch."):
                raise ImportError("torch is not importable in this host check")
            return None
    sys.meta_path.insert(0, _NoTorch())
    sys.path[:0] = [{here!r}, {root!r}]
''').format(here=str(HERE), root=str(API_ROOT))

_SITES = textwrap.dedent('''
    import importlib, json, os, subprocess, sys
    from pathlib import Path
    from types import SimpleNamespace

    import lift_host

    root = Path(os.environ["LIFT_TEST_ROOT"])
    stock, gdsii, sigma, recorded, recorded_side = {names!r}
    seen = []

    def no_exec(command, *args, **kwargs):
        # Nothing is executed: the command and its environment are recorded and the
        # child "dies" without writing a record.
        seen.append({{"argv": [str(part) for part in command],
                      "env": [f"{{k}}={{v}}" for k, v in (kwargs.get("env") or {{}}).items()]}})
        return subprocess.CompletedProcess(command, 1, None, b"")

    subprocess.run = no_exec
    # A PASSING CHILD, when asked: the record a lifted row's census child writes, carrying
    # every site's block. A row flagged withdraw in the seam (the gdsii row here) is
    # refused under the nonlinear gate's block, which still expects a standing withdraw's
    # refusal, and admitted under every other.
    WRITE = {{"on": False}}
    PASSING = {{"predicate_admits": True, "predicate_reasons": [], "sources": [],
               "driven": True, "passed": True, "bit_identical": True, "identical": True,
               "steps_compared": 10 ** 6, "words_compared": 1,
               "composer_refuses_this_product": True}}
    REFUSED = {{"predicate_admits": False, "predicate_reasons": ["standing withdraw"],
               "sources": []}}
    BLOCKS = ["metal_fused_hd_pair_gate", "metal_bfast_fused_hd_pair_gate",
              "metal_conductive_fused_hd_pair_gate",
              "metal_cylindrical_real_fused_hd_pair_gate", "metal_folded_fused_hd_pair_gate",
              "metal_tail_gate"]
    NONLINEAR = "metal_fused_hd_pair_nonlinear_gate"

    def answering(command, *args, **kwargs):
        done = no_exec(command, *args, **kwargs)
        argv = [str(part) for part in command]
        if not WRITE["on"] or "--out-json" not in argv:
            return done
        row = (Path(argv[argv.index("--child-script") + 1]).name if "--child-script" in argv
               else json.loads(argv[argv.index("--child-cases") + 1])[0])
        record = {{"row": row, "measured": True, **{{key: PASSING for key in BLOCKS}},
                  NONLINEAR: REFUSED if str(row).startswith("gds_") else PASSING}}
        Path(argv[argv.index("--out-json") + 1]).write_text(json.dumps(record),
                                                             encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, None, b"")

    subprocess.run = answering
    ANSWERS = {{
        "present": {{"interpreter": sys.executable, "unreadable": None,
                     "needs": {{n: None for n in lift_host.NEEDS}}}},
        "gdsii_absent": {{"interpreter": sys.executable, "unreadable": None,
                          "needs": {{**{{n: None for n in lift_host.NEEDS}},
                                     "gdspy": "ModuleNotFoundError: No module named 'gdspy'",
                                     "libGDSII": "meep.with_libGDSII() is False"}}}},
        "unreadable": {{"interpreter": sys.executable, "needs": {{}},
                        "unreadable": "the capability probe printed no answer"}},
    }}
    SITES = ["gate_metal_fused_hd_pair", "gate_metal_bfast_fused_hd_pair",
             "gate_metal_conductive_fused_hd_pair", "gate_metal_cylindrical_real_fused_hd_pair",
             "gate_metal_folded_fused_hd_pair", "gate_metal_fused_hd_pair_nonlinear",
             "hd_tail_gate_common"]
    modules = {{name: importlib.import_module(name) for name in SITES}}
    product = SimpleNamespace(census="census_x", seam_record="seam_x",
                              cell_arms={{"plain": ("TAIL H", "TAIL D")}}, family="metal_tail",
                              lift_environment_prefix="LIFT_TEST_TAIL", gate_stem="gate_tail")
    # THE CENSUS: per site's cell, one stock example, one example from the gdsii lift
    # record, one from the sigma record, and one tests row with no lift record; every row
    # records an interpreter path that does not exist here.
    arms = [tuple(modules[name].CELL_ARMS) for name in SITES[:-1]] + [("TAIL H", "TAIL D")]
    results, corpus = root / "results", root / "corpus"
    for directory in (results / "census_x", results / "seam_x", corpus / "examples",
                      corpus / "tests"):
        directory.mkdir(parents=True, exist_ok=True)
    (corpus / "tests" / "test_rows.py").write_text("# rows\\n", encoding="utf-8")
    examples, tests, seam = [], [], []
    for k, (update_h, step_d) in enumerate(arms):
        selected = {{"plan_step": {{"selected": {{"update_H": update_h, "step_D": step_d}}}}}}
        for script, record, path in ((f"plain_{{k}}.py", stock, recorded),
                                     (f"gds_{{k}}.py", gdsii, recorded_side),
                                     (f"mie_{{k}}.py", sigma, recorded_side)):
            (corpus / "examples" / script).write_text("# row\\n", encoding="utf-8")
            examples.append({{"leg": "examples", "row": script, "lift_record": record,
                              "interpreter": path, "python": path, "grid_cells": 8,
                              "measured": True, **selected}})
            seam.append({{"label": f"examples:{{script}}", "leg": "examples", "row": script,
                          "lift_record": record, "interpreter": path,
                          "h_to_d_seam": {{"withdraw_in_seam": record == gdsii}}}})
        tests.append({{"leg": "tests", "row": f"TestRows.test_{{k}}", "module": "test_rows.py",
                       "python": recorded, "grid_cells": 8, "measured": True, **selected}})
    for name, rows in (("census_x/examples.jsonl", examples), ("census_x/tests.jsonl", tests),
                       ("census_x/tests_param_matched.jsonl", []),
                       ("seam_x/h_to_d_seam.jsonl", seam)):
        (results / name).write_text("".join(json.dumps(r) + "\\n" for r in rows),
                                    encoding="utf-8")
    for module in modules.values():
        module.HERE = root
        if hasattr(module, "CENSUS"):
            module.CENSUS, module.SEAM_RECORD = "census_x", "seam_x"

    def run(name, case):
        seen.clear()
        module, out = modules[name], root / "out" / case / name
        try:
            if name == "hd_tail_gate_common":
                result = module.leg_lift(product, out, 2, None, 5.0, False, None)
            elif name == "gate_metal_cylindrical_real_fused_hd_pair":
                result = module.leg_lift(out, 2, 5.0, False)
            else:
                result = module.leg_lift(out, 2, None, 5.0, False, None)
        except SystemExit as exc:
            return {{"exit": str(exc)}}
        rows_path = out / "lift" / "rows.jsonl"
        rows = ([json.loads(l) for l in rows_path.read_text().splitlines() if l.strip()]
                if rows_path.is_file() else [])
        census_paths = sorted({{part for call in seen for part in call["argv"] + call["env"]
                                if "/nonexistent/census-host" in part}})
        unmeasured = result.get("rows_unmeasured", result.get("unmeasured")) or []
        return {{"passed": result.get("passed"), "reason": result.get("reason"),
                 "launched": [call["argv"][0] for call in seen],
                 "launched_scripts": sorted(Path(call["argv"][6]).name for call in seen
                                            if "--child-script" in call["argv"]),
                 "census_paths": census_paths,
                 "unlifted": result.get("unlifted_on_this_host"),
                 "asked": result.get("rows_the_engine_was_asked_about"),
                 "unmeasured": sorted(unmeasured),
                 "rows": [{{k: r.get(k) for k in ("label", "interpreter", "lift_needs",
                                                  "recorded_interpreter",
                                                  "unlifted_on_this_host")}} for r in rows]}}

    def answer_with(answer):
        lift_host.probe = lambda needs, answer=answer, **kw: {{
            **answer, "needs": {{n: answer["needs"].get(n) for n in sorted(set(needs))}}
            if not answer["unreadable"] else {{}}}}

    out = {{}}
    for case, answer in ANSWERS.items():
        answer_with(answer)
        out[case] = {{name: run(name, case) for name in SITES}}
    WRITE["on"] = True
    for case in ("present", "gdsii_absent"):
        answer_with(ANSWERS[case])
        out[f"{{case}}_passing"] = {{name: run(name, f"{{case}}_passing") for name in SITES}}
    # EVERY ROW UNLIFTED: every lift record, and a row with none, declares gdspy.
    declared = dict(lift_host.LIFT_RECORD_NEEDS)
    lift_host.LIFT_RECORD_NEEDS.update({{record: ("gdspy",) for record in
                                         (stock, gdsii, sigma, "")}})
    answer_with(ANSWERS["gdsii_absent"])
    out["all_absent_passing"] = {{name: run(name, "all_absent_passing") for name in SITES}}
    lift_host.LIFT_RECORD_NEEDS.clear()
    lift_host.LIFT_RECORD_NEEDS.update(declared)
    WRITE["on"] = False
    (results / "census_x" / "tests_param_matched.jsonl").unlink()
    out["missing_input"] = {{name: run(name, "missing_input") for name in SITES}}
    print(json.dumps({{"executable": sys.executable, "cases": out}}))
''')


def test_the_seven_lift_sites_launch_this_interpreter_and_count_out_what_the_host_lacks(
        tmp_path):
    environment = {k: v for k, v in os.environ.items() if not k.startswith("MEEP_GPU_")}
    environment.update(LIFT_TEST_ROOT=str(tmp_path),
                       MEEP_GPU_CORPUS_ROOT=str(tmp_path / "corpus"),
                       PYTHONDONTWRITEBYTECODE="1")
    body = _NO_TORCH + _SITES.format(names=(STOCK, GDSII, SIGMA, RECORDED, RECORDED_SIDE))
    done = subprocess.run([sys.executable, "-c", body], cwd=str(API_ROOT), env=environment,
                          capture_output=True, text=True, check=False, timeout=900)
    assert done.returncode == 0, done.stdout[-4000:] + done.stderr[-4000:]
    summary = json.loads(done.stdout.strip().splitlines()[-1])
    parent, cases = summary["executable"], summary["cases"]
    assert len(cases["present"]) == 7
    for case in ("present", "gdsii_absent", "unreadable"):
        for name, site in cases[case].items():
            # No path the census recorded reaches a child: not argv[0], not an argument,
            # not an environment value.
            assert site["census_paths"] == [], (case, name, site["census_paths"])
            assert set(site["launched"]) <= {parent}, (case, name, site["launched"])
    for name, site in cases["present"].items():
        # Four rows per cell, every one launched under this interpreter.
        assert len(site["launched"]) == 4, (name, site)
        assert site["unlifted"] == {} and site["asked"] == 4, (name, site)
        for row in site["rows"]:
            assert row["interpreter"] == parent, (name, row)
            assert row["recorded_interpreter"] in (RECORDED, RECORDED_SIDE), (name, row)
        assert site["passed"] is False, "children that wrote nothing are unmeasured"
    for name, site in cases["gdsii_absent"].items():
        # The gdsii row is named and not launched; the other three are launched.
        assert len(site["launched"]) == 3, (name, site)
        assert not [s for s in site["launched_scripts"] if s.startswith("gds_")], (name, site)
        assert list(site["unlifted"]) == [label for label in site["unlifted"]
                                          if label.startswith("examples:gds_")], (name, site)
        assert len(site["unlifted"]) == 1 and site["asked"] == 3, (name, site)
        assert "gdspy (ModuleNotFoundError" in next(iter(site["unlifted"].values()))
        assert not set(site["unlifted"]) & set(site["unmeasured"]), (name, site)
        [row] = [r for r in site["rows"] if r["unlifted_on_this_host"]]
        assert row["lift_needs"] == ["gdspy", "libGDSII"], (name, row)
    for name, site in cases["unreadable"].items():
        assert site["launched"] == [] and site["passed"] is False, (name, site)
        assert site["reason"] == "the capability probe printed no answer", (name, site)
    for name, site in cases["present_passing"].items():
        # Every child answers a passing record: the leg passes on all four rows.
        assert len(site["launched"]) == 4 and site["asked"] == 4, (name, site)
        assert site["passed"] is True and site["unmeasured"] == [], (name, site)
    for name, site in cases["gdsii_absent_passing"].items():
        # THE DENOMINATOR LESS THE ROW COUNTED OUT: three rows asked, one named, a pass.
        assert len(site["launched"]) == 3 and site["asked"] == 3, (name, site)
        assert len(site["unlifted"]) == 1 and site["unmeasured"] == [], (name, site)
        assert site["passed"] is True, (name, site)
    for name, site in cases["all_absent_passing"].items():
        # A LEG THAT ASKED ABOUT NO ROW measured nothing, and fails.
        assert site["launched"] == [] and site["asked"] == 0, (name, site)
        assert len(site["unlifted"]) == 4, (name, site)
        assert site["passed"] is False, (name, site)
    for name, site in cases["missing_input"].items():
        assert "census_x/tests_param_matched.jsonl" in site.get("exit", ""), (name, site)


# ------------------------------------------------------------------- the source guard --
def test_no_lift_site_launches_an_interpreter_read_from_evidence():
    """The seven sites launch ``sys.executable`` and resolve their rows through
    ``lift_host``; the census-path-first expression is gone from every Metal gate and kit.
    ``probe_h_to_d_withdraw_order.py`` is the one harness file that still launches the
    interpreter a census row recorded: a probe no round runs, left as it is."""
    old = 'or seam_row.get("interpreter") or sys.executable'
    launch = 'command = [row["interpreter"]'
    resolving = sorted(p.name for p in HERE.glob("*.py") if not p.name.startswith("test_")
                       and "lift_host.row_fields(" in p.read_text(encoding="utf-8"))
    assert tuple(resolving) == SITES
    for name in SITES:
        text = (HERE / name).read_text(encoding="utf-8")
        assert old not in text and launch not in text, name
        assert text.count('command = [sys.executable, "-u",') == 2, name
    launchers = sorted(p.name for p in HERE.glob("*.py") if not p.name.startswith("test_")
                       and launch in p.read_text(encoding="utf-8"))
    assert launchers == [RECORDED_LAUNCHER]
    probe = (HERE / RECORDED_LAUNCHER).read_text(encoding="utf-8")
    assert '"interpreter": record.get("interpreter") or sys.executable' in probe


def test_the_complex_gate_requires_the_four_inputs_by_name(tmp_path):
    """``gate_metal_complex_fused_hd_pair.cell_rows`` drives its cell's rows in process
    and reads the lift legs' four inputs: a row found only in
    ``tests_param_matched.jsonl`` is in its cell, and that file missing stops it by name
    (on the standing records the cell used to shrink from 17 rows to 12)."""
    body = _NO_TORCH + textwrap.dedent('''
        import json, sys
        from pathlib import Path
        import gate_metal_complex_fused_hd_pair as gate
        root = Path(sys.argv[1])
        results = root / "results"
        arms = {"plan_step": {"selected": {"update_H": gate.CELL_ARMS[0],
                                           "step_D": gate.CELL_ARMS[1]}}}
        files = {f"{gate.CENSUS}/examples.jsonl": [{"leg": "examples", "row": "a.py", **arms}],
                 f"{gate.CENSUS}/tests.jsonl": [],
                 f"{gate.CENSUS}/tests_param_matched.jsonl": [
                     {"leg": "tests", "row": "T.t_0", **arms}],
                 f"{gate.SEAM_RECORD}/h_to_d_seam.jsonl": []}
        for rel, rows in files.items():
            (results / rel).parent.mkdir(parents=True, exist_ok=True)
            (results / rel).write_text("".join(json.dumps(r) + "\\n" for r in rows),
                                       encoding="utf-8")
        gate.HERE = str(root)
        rows, _facts = gate.cell_rows()
        out = {"rows": sorted(r["label"] for r in rows), "exit": None,
               "census": gate.CENSUS}
        (results / gate.CENSUS / "tests_param_matched.jsonl").unlink()
        try:
            gate.cell_rows()
        except SystemExit as exc:
            out["exit"] = str(exc)
        print(json.dumps(out))
    ''')
    done = subprocess.run([sys.executable, "-c", body, str(tmp_path)], cwd=str(API_ROOT),
                          capture_output=True, text=True, timeout=900, check=False,
                          env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    summary = json.loads(done.stdout.strip().splitlines()[-1])
    assert summary["rows"] == ["examples:a.py", "tests:T.t_0"], summary
    assert f"{summary['census']}/tests_param_matched.jsonl" in (summary["exit"] or ""), summary


# ---------------------------------------------------------------- the real records ----
_CENSUS = HERE / "results" / "metal_coverage_2026-09-04_m0complex"
_SEAM = HERE / "results" / "h_to_d_seam_2026-09-04"


@pytest.mark.requires_resource("evidence_archive")
@pytest.mark.skipif(not all((HERE / "results" / rel).is_file() for rel in
                            lift_host.lift_inputs(_CENSUS.name, _SEAM.name)),
                    reason="the 2026-09-04 census and seam records are not in this checkout")
def test_the_nine_cells_on_the_standing_records():
    """145 rows over the nine launching gates, every one under this interpreter, and the
    five that declare needs are the two gdsii and three sigma rows."""
    body = _NO_TORCH + textwrap.dedent('''
        import importlib, json, sys
        from pathlib import Path
        import hd_tail_gate_common as common
        results = Path(sys.argv[1])
        direct = ["gate_metal_fused_hd_pair", "gate_metal_bfast_fused_hd_pair",
                  "gate_metal_conductive_fused_hd_pair",
                  "gate_metal_cylindrical_real_fused_hd_pair",
                  "gate_metal_folded_fused_hd_pair", "gate_metal_fused_hd_pair_nonlinear"]
        tail = ["gate_metal_beta_real_fused_hd_pair", "gate_metal_beta_complex_fused_hd_pair",
                "gate_metal_folded_complex_fused_hd_pair"]
        rows = []
        for name in direct + tail:
            module = importlib.import_module(name)
            rows += (common.lift_basis(module.PRODUCT, results)[0] if name in tail
                     else module.lift_basis(results)[0])
        print(json.dumps({"rows": len(rows),
                          "parent": sum(r["interpreter"] == sys.executable for r in rows),
                          "needs": sorted(r["label"] for r in rows if r["lift_needs"])}))
    ''')
    done = subprocess.run([sys.executable, "-c", body, str(HERE / "results")],
                          cwd=str(API_ROOT), capture_output=True, text=True, timeout=900,
                          env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    assert done.returncode == 0, done.stderr[-3000:]
    summary = json.loads(done.stdout.strip().splitlines()[-1])
    assert summary["rows"] == summary["parent"] == 145, summary
    assert summary["needs"] == ["examples:coupler.py", "examples:differential_cross_section.py",
                                "examples:mie_scattering.py", "examples:ring_gds.py",
                                "examples:waveguide_crossing.py"], summary
