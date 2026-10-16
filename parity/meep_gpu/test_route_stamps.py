"""The route-stamp registry: one home for the three tables' campaign names, read by no gate.

WHAT THIS HOLDS (``meep_gpu/route_stamps.py``; bit_parity_0_10 DESIGN.md section 8.2, owner
ruling R22). The three route stamps left the dispatch modules so that a re-route edits one
file that no gate's import record holds:

* the registry imports nothing of the package, and no package module imports it;
* no gate or probe script holds it in its run closure (``import_closure.py bill``), and a
  planted import of it in the planner is caught by the same instrument;
* the dispatch path cites the ledger slot that records the campaign, never a dated stamp,
  and each slot it cites exists in its ledger;
* the names the registry spells, and the refusals it names: the placeholder the freeze
  replaces, a spent campaign or stamp, a campaign of another stamp, a licence of another
  campaign or device; a table re-routed alone moves only its own names;
* every route run the three ledgers record is either spent or a campaign of the current
  stamp, so the spent list cannot fall behind the records;
* the two round preflights key on the same names: ``legs.route_campaign_names`` (NVIDIA)
  and ``metal_round.ROUTE_NAMES_PROGRAM`` (Metal) refuse the placeholder and a spent name.

The re-cut's refusals by name are driven end to end in ``test_recut_route_status.py`` and
``test_recut_metal_runs.py``.
"""

from __future__ import annotations

import ast
import json
import re
import sys
import time
import types
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ROUND_DIR = HERE / "rounds" / "1.0"
for _path in (str(HERE), str(ROUND_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import import_closure as ic  # noqa: E402
import legs as L  # noqa: E402
import metal_round  # noqa: E402

from meep_gpu import fastpath, fastpath_cuda, metal_dispatch, metal_runs  # noqa: E402
from meep_gpu import route_stamps  # noqa: E402

REGISTRY = "meep_gpu/route_stamps.py"
STAMP = "2099-01-01_test"
#: A route campaign name, whole: what a stamp constant's value is (prose that names a past
#: campaign inside a sentence is not one).
CAMPAIGN = re.compile(r"^dispatch_(fused_route_(cuda_)?|metal_route_)\S+$")
#: The three dispatch modules and the slot each cites.
POINTERS = {"triton": ("meep_gpu/fastpath.py", fastpath.ROUTE_RECORD),
            "cuda": ("meep_gpu/fastpath_cuda.py", fastpath_cuda.CUDA_ROUTE_RECORD),
            "metal": ("meep_gpu/metal_dispatch.py", metal_dispatch.METAL_ROUTE_RECORD)}
LEDGERS = {"triton": "meep_gpu/triton_kernels/fingerprints.json",
           "cuda": "meep_gpu/cuda_kernels/fingerprints.json",
           "metal": "meep_gpu/metal_kernels/fingerprints.json"}


@pytest.fixture
def stamped(monkeypatch):
    """The registry with a concrete, unspent round stamp and no table re-routed."""
    monkeypatch.setattr(route_stamps, "ROUND", STAMP)
    monkeypatch.setattr(route_stamps, "REROUTED", {})
    return route_stamps


def _imports(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = ("." * node.level) + (node.module or "")
            yield base
            yield from (f"{base}.{alias.name}" for alias in node.names)


# --------------------------------------------------------------- who reads it -----


def test_the_registry_imports_nothing_of_the_package():
    found = sorted(set(_imports(ROOT / REGISTRY)))
    assert found == ["__future__", "__future__.annotations", "sys", "typing",
                     "typing.Mapping", "typing.Optional", "typing.Sequence",
                     "typing.Tuple"], found


def test_no_package_module_imports_the_registry():
    """The package's production modules, the dispatch path among them; its tests may."""
    readers = []
    for path in sorted((ROOT / "meep_gpu").rglob("*.py")):
        rel = path.relative_to(ROOT).as_posix()
        if rel == REGISTRY or path.name.startswith("test_") or path.name == "conftest.py":
            continue
        if any("route_stamps" in name for name in _imports(path)):
            readers.append(rel)
    assert readers == [], readers


def test_no_gate_or_probe_holds_the_registry_in_its_run_closure():
    """The bill of a re-route: ``import_closure.py bill meep_gpu/route_stamps.py`` reads 0."""
    resolver = ic.Resolver(ic.REPO_ROOT)
    scripts = ic.gate_scripts(ic.REPO_ROOT)
    started, held = time.time(), []
    for index, script in enumerate(scripts, 1):
        if REGISTRY in resolver.run_closure(script):
            held.append(script)
        if index % 50 == 0 or index == len(scripts):
            print(f"closure {index}/{len(scripts)} ({time.time() - started:.1f} s)", flush=True)
    assert len(scripts) > 200, len(scripts)
    assert held == [], f"{len(held)} of {len(scripts)} gate and probe scripts hold it: {held}"


def test_a_planted_import_of_the_registry_in_the_planner_is_caught():
    """The instrument above, shown able to fail: the planner importing the registry would
    put it in the closure of every gate that plans a driver."""
    script = "parity/meep_gpu/gate_dispatch_fused_route.py"
    assert REGISTRY not in ic.Resolver(ic.REPO_ROOT).run_closure(script)
    text = (ROOT / "meep_gpu/fastpath.py").read_text(encoding="utf-8")
    planted = ic.Resolver(ic.REPO_ROOT, overrides={
        "meep_gpu/fastpath.py": text + "\nfrom . import route_stamps\n"})
    assert REGISTRY in planted.run_closure(script)


# ------------------------------------------------------------ what product cites -----


@pytest.mark.parametrize("table", sorted(POINTERS))
def test_the_dispatch_path_holds_no_stamp_constant(table):
    """No module-level name of a dispatch module is bound to a campaign name, as the three
    stamps were until they moved; the record and the refusal cite the ledger slot."""
    module, _pointer = POINTERS[table]
    tree = ast.parse((ROOT / module).read_text(encoding="utf-8"), filename=module)
    bound = []
    for node in tree.body:
        targets = (node.targets if isinstance(node, ast.Assign)
                   else [node.target] if isinstance(node, ast.AnnAssign) else [])
        value = getattr(node, "value", None)
        if (targets and isinstance(value, ast.Constant) and isinstance(value.value, str)
                and CAMPAIGN.match(value.value)):
            bound.append((ast.unparse(targets[0]), value.value))
    assert bound == [], f"{module} binds a route stamp: {bound}"


@pytest.mark.parametrize("table", sorted(POINTERS))
def test_each_cited_slot_exists_in_its_ledger(table):
    """``<ledger> at driver_dispatch.runs[<device>].released_fused_arms.gate``: the
    ledger file the pointer names is this table's, and every route run in it carries the
    field the pointer sends a reader to."""
    _module, pointer = POINTERS[table]
    match = re.search(r"recorded in (\S+) at driver_dispatch\.runs\[<[^>]+>\]"
                      r"\.released_fused_arms\.gate$", pointer)
    assert match, pointer
    assert f"meep_gpu/{match.group(1)}" == LEDGERS[table]
    record = json.loads((ROOT / LEDGERS[table]).read_text(encoding="utf-8"))["driver_dispatch"]
    runs = record.get("runs") or {}
    assert runs, f"{LEDGERS[table]} records no route run"
    for device, run in runs.items():
        assert (run.get("released_fused_arms") or {}).get("gate"), (table, device)


# ---------------------------------------------------------------- what it spells -----


def test_the_stamps_spell_each_tables_campaign_names(stamped):
    assert stamped.route_gate("triton") == f"dispatch_fused_route_{STAMP}"
    assert stamped.route_gate("cuda") == f"dispatch_fused_route_cuda_{STAMP}"
    assert stamped.route_gate("metal") == f"dispatch_metal_route_{STAMP}"
    assert stamped.licence_campaign("triton") == f"dispatch_end_to_end_{STAMP}"
    assert fastpath.route_campaign(stamped.route_gate("cuda"), "8.6") \
        == f"dispatch_fused_route_cuda_{STAMP}_cc86"
    assert metal_runs.route_campaign(stamped.route_gate("metal"), "applegpu_g13s") \
        == f"dispatch_metal_route_{STAMP}_applegpu_g13s"
    with pytest.raises(ValueError):
        stamped.licence_campaign("metal")
    with pytest.raises(ValueError):
        stamped.route_gate("cupy")
    for table in route_stamps.TABLES:
        assert stamped.campaign_refusal(table, f"{stamped.route_gate(table)}_x") is None


def test_the_placeholder_is_refused_everywhere_by_name(monkeypatch):
    monkeypatch.setattr(route_stamps, "ROUND", route_stamps.PLACEHOLDER)
    monkeypatch.setattr(route_stamps, "REROUTED", {})
    for table in route_stamps.TABLES:
        assert route_stamps.is_unset(table)
        name = f"{route_stamps.route_gate(table)}_cc86"
        assert route_stamps.campaign_refusal(table, name).startswith("route_stamp_is_unset")
    refused = route_stamps.licence_refusal("triton", route_stamps.licence_campaign("triton"),
                                           "cc86")
    assert refused.startswith("route_stamp_is_unset")
    lines, refusals = route_stamps.report()
    assert len(lines) == 3 and len(refusals) == 3
    assert route_stamps.main([]) == 1


def test_a_spent_campaign_and_a_spent_stamp_are_refused_by_name(monkeypatch):
    """The 1.0 round's names are spent: a freeze that reused its stamp is refused, on
    every device, before a campaign could write into a directory that already ran."""
    monkeypatch.setattr(route_stamps, "REROUTED", {})
    monkeypatch.setattr(route_stamps, "ROUND", "2026-10-13_10")
    for table, device in (("triton", "cc86"), ("cuda", "cc86"), ("metal", "applegpu_g13s")):
        refused = route_stamps.campaign_refusal(
            table, f"{route_stamps.route_gate(table)}_{device}")
        assert refused and refused.startswith("route_campaign_is_spent"), (table, refused)
    # A campaign named by its stamp alone (the names before per-device campaigns) spends
    # the stamp for every device.
    monkeypatch.setattr(route_stamps, "ROUND", "2026-09-27_flip")
    refused = route_stamps.campaign_refusal("triton", f"{route_stamps.route_gate('triton')}_cc90")
    assert refused and refused.startswith("route_campaign_is_spent"), refused
    assert route_stamps.main([]) == 1


def test_a_campaign_of_another_stamp_is_refused_by_name(stamped):
    refused = stamped.campaign_refusal("triton", "dispatch_fused_route_2026-10-13_10_cc86")
    assert refused.startswith("route_campaign_is_not_this_stamp"), refused
    # The CUDA prefix extends the Triton one: a CUDA campaign is not a Triton one.
    refused = stamped.campaign_refusal("triton", f"dispatch_fused_route_cuda_{STAMP}_cc86")
    assert refused.startswith("route_campaign_is_not_this_stamp"), refused


@pytest.mark.parametrize("campaign, refused", [
    (f"dispatch_end_to_end_{STAMP}", False),
    (f"dispatch_end_to_end_{STAMP}_cc86", False),
    (f"dispatch_end_to_end_{STAMP}_cc90", True),
    ("dispatch_end_to_end_2026-10-13_10", True),
    (f"dispatch_fused_route_{STAMP}_cc86", True),
])
def test_a_licence_is_the_rounds_by_name(stamped, campaign, refused):
    reason = stamped.licence_refusal("triton", campaign, "cc86")
    assert (reason is not None) is refused, reason
    if refused:
        assert reason.startswith("licence_is_for_another_campaign"), reason


def test_a_table_rerouted_alone_moves_only_its_own_names(stamped, monkeypatch):
    before = {table: stamped.route_gate(table) for table in stamped.TABLES}
    monkeypatch.setattr(route_stamps, "REROUTED", {"metal": "2099-01-02_test"})
    after = {table: stamped.route_gate(table) for table in stamped.TABLES}
    assert [t for t in stamped.TABLES if before[t] != after[t]] == ["metal"]
    assert after["metal"] == "dispatch_metal_route_2099-01-02_test"
    assert stamped.licence_campaign("triton") == f"dispatch_end_to_end_{STAMP}"
    refused = stamped.campaign_refusal("metal", f"{before['metal']}_applegpu_g13s")
    assert refused.startswith("route_campaign_is_not_this_stamp"), refused


# -------------------------------------------------------- the list cannot lag ---------


def _recorded_campaigns(table: str):
    record = json.loads((ROOT / LEDGERS[table]).read_text(encoding="utf-8"))["driver_dispatch"]
    for device, run in (record.get("runs") or {}).items():
        names = {(run.get("released_fused_arms") or {}).get("gate"),
                 Path(str(run.get("records") or "")).name}
        yield from ((device, name) for name in sorted(n for n in names if n))


@pytest.mark.parametrize("table", sorted(LEDGERS))
def test_every_route_run_the_ledgers_record_is_spent_or_this_stamps(table):
    """A campaign recorded under an earlier stamp must be in SPENT, so the freeze that
    moves ROUND has to move the old names in with it; the current stamp's own runs are
    not spent until the next round moves ROUND."""
    current = f"{route_stamps.route_gate(table)}_"
    stray = [(device, name) for device, name in _recorded_campaigns(table)
             if name not in route_stamps.SPENT[table] and not name.startswith(current)]
    assert stray == [], f"{table}: recorded route runs neither spent nor this stamp's: {stray}"


def test_the_current_stamps_are_not_spent():
    for table in route_stamps.TABLES:
        gate = route_stamps.route_gate(table)
        assert gate not in route_stamps.SPENT[table], gate
        assert not [name for name in route_stamps.SPENT[table] if name.startswith(gate + "_")]
    for table, names in route_stamps.SPENT.items():
        assert all(name.startswith(route_stamps.ROUTE_PREFIXES[table]) for name in names)
        if table == "triton":
            assert not [n for n in names if n.startswith("dispatch_fused_route_cuda_")]
        assert len(set(names)) == len(names), table


# ------------------------------------------------------------ the two preflights -----


def test_the_nvidia_preflight_names_the_registrys_campaigns(stamped, monkeypatch):
    monkeypatch.setattr(fastpath, "primary_table", lambda capability: "triton")
    facts, problems = L.route_campaign_names("8.6")
    assert problems == []
    assert facts == {"triton_route_campaign": f"dispatch_fused_route_{STAMP}_cc86",
                     "cuda_route_campaign": f"dispatch_fused_route_cuda_{STAMP}_cc86",
                     "licence_campaign": f"dispatch_end_to_end_{STAMP}"}
    assert facts["licence_campaign"] in L.round_roots(STAMP, 2, "8.6", facts)


@pytest.mark.parametrize("round_stamp, reason", [
    (route_stamps.PLACEHOLDER, "route_stamp_is_unset"),
    ("2026-10-13_10", "route_campaign_is_spent"),
])
def test_the_nvidia_preflight_refuses_an_unset_or_spent_stamp(monkeypatch, round_stamp,
                                                              reason):
    monkeypatch.setattr(route_stamps, "ROUND", round_stamp)
    monkeypatch.setattr(route_stamps, "REROUTED", {})
    monkeypatch.setattr(fastpath, "primary_table", lambda capability: "triton")
    _facts, problems = L.route_campaign_names("8.6")
    assert len(problems) == 2 and all(reason in line for line in problems), problems


@pytest.mark.parametrize("route", [True, False])
def test_only_the_route_stage_of_the_check_stops_on_an_unset_stamp(monkeypatch, route):
    """``legs.check`` on this checkout: the route stage refuses the placeholder by name; a
    gate stage records the refusal and goes on, so gate chunks can run before the freeze
    sets the stamp (the stamp edit moves a file no gate holds)."""
    monkeypatch.setattr(route_stamps, "ROUND", route_stamps.PLACEHOLDER)
    monkeypatch.setattr(route_stamps, "REROUTED", {})
    problems, facts = L.check(ROOT, "8.6", route)
    assert [line for line in facts["route_campaign_refusals"]
            if "route_stamp_is_unset" in line], facts
    unset = [line for line in problems if "route_stamp_is_unset" in line]
    assert bool(unset) is route, problems


def _metal_names(monkeypatch, capsys) -> dict:
    """Run the route stage's naming program here, on this tree's registry, with the
    fleet's recorded architecture stood in."""
    environment = types.ModuleType("metal_environment")
    environment.read = lambda: {"architecture": "applegpu_g13s"}
    monkeypatch.setitem(sys.modules, "metal_environment", environment)
    exec(compile(metal_round.ROUTE_NAMES_PROGRAM, "ROUTE_NAMES_PROGRAM", "exec"), {})
    return json.loads(capsys.readouterr().out.strip().splitlines()[-1])


def test_the_metal_preflight_names_the_registrys_campaign(stamped, monkeypatch, capsys):
    facts = _metal_names(monkeypatch, capsys)
    assert facts == {"arch": "applegpu_g13s", "stamp": f"dispatch_metal_route_{STAMP}",
                     "campaign": f"dispatch_metal_route_{STAMP}_applegpu_g13s",
                     "refused": None}


@pytest.mark.parametrize("round_stamp, reason", [
    (route_stamps.PLACEHOLDER, "route_stamp_is_unset"),
    ("2026-10-13_20", "route_campaign_is_spent"),
])
def test_the_metal_preflight_refuses_an_unset_or_spent_stamp(monkeypatch, capsys, tmp_path,
                                                             round_stamp, reason):
    """The naming program refuses the name, and the route stage stops on it with nothing
    launched."""
    monkeypatch.setattr(route_stamps, "ROUND", round_stamp)
    monkeypatch.setattr(route_stamps, "REROUTED", {})
    facts = _metal_names(monkeypatch, capsys)
    assert facts["refused"] and facts["refused"].startswith(reason), facts

    class Named:
        stdout = json.dumps(facts) + "\n"
        stderr = ""
        returncode = 0

    launched = []
    monkeypatch.setattr(metal_round.subprocess, "run", lambda *a, **k: Named())
    monkeypatch.setattr(metal_round.Round, "tree_is_the_commit", lambda self, stage: None)
    monkeypatch.setattr(metal_round.Round, "run_item",
                        lambda self, *a, **k: launched.append(a) or 0)
    assert metal_round.main(["route", "--repo", str(tmp_path), "--stamp", "s",
                             "--results", str(tmp_path), "--logs", str(tmp_path / "logs")]) != 0
    assert launched == []
