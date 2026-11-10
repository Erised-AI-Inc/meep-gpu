"""The route-stamp registry: one home for the three tables' campaign names, read by no gate.

What this holds (``meep_gpu/route_stamps.py``, added for the
1.0 re-certification). The three route stamps left the dispatch modules so that a re-route edits one
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
* the 1.0.1 round's first Metal route campaign, which released on a Metal frontend no
  bound weld certifies and can never be recorded, stays spent, and the Metal route stage
  names another campaign that it may launch;
* its second, which released on bytes whose dispatch preference arbitrated the Metal
  table by NVIDIA records, stays spent too, and the Metal table's stamp is a third;
* the round's compute capability 8.6 NVIDIA route campaigns and licence, which released
  on bytes the dispatch preference has since moved and read timing records these bytes
  replace, are refused by name, and the two NVIDIA tables share a new stamp under which
  any compute capability's route and licence may run; the chunk tooling reads that
  stamp as the NVIDIA round.

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

import chunks as C  # noqa: E402
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


SPENT_METAL_ROUTE = "dispatch_metal_route_2026-10-18_101_applegpu_g13s"


def test_the_spent_metal_route_of_the_round_is_refused_and_another_is_named(monkeypatch,
                                                                            capsys):
    """The held Metal route of round ``2026-10-18_101`` released its legs, but every plan
    recorded a Metal frontend that no weld of the bound ledger certifies, so the re-cut
    refuses its rows and it can never be recorded. Its name is spent, the re-cut and the
    route stage refuse it by name, and the route stage names another campaign: a fresh
    stamp (the Metal table's :data:`REROUTED` entry, or a later round's), never the
    placeholder while the round has a stamp, and never a spent one."""
    assert SPENT_METAL_ROUTE in route_stamps.SPENT["metal"]
    refused = route_stamps.campaign_refusal("metal", SPENT_METAL_ROUTE)
    assert refused and refused.split(":")[0] in (
        "route_campaign_is_spent", "route_campaign_is_not_this_stamp",
        "route_stamp_is_unset"), refused
    facts = _metal_names(monkeypatch, capsys)
    assert facts["campaign"] != SPENT_METAL_ROUTE, facts
    assert facts["campaign"] not in route_stamps.SPENT["metal"], facts
    assert facts["refused"] is None or route_stamps.ROUND == route_stamps.PLACEHOLDER, facts


SECOND_SPENT_METAL_ROUTE = "dispatch_metal_route_2026-10-25_101_applegpu_g13s"


def test_the_second_metal_route_of_the_round_is_refused_and_another_is_named():
    """The Metal route re-run under ``2026-10-25_101`` released on bytes whose dispatch
    preference arbitrated the Metal table by a portable prior derived from NVIDIA records;
    the prior is now derived per vendor, so that campaign is never recorded. On the
    registry as committed: both spent Metal routes of the round are spent and refused by
    name, the Metal table's stamp is neither of theirs, the campaign it names may be
    launched, and it is not the NVIDIA tables' stamp (they were re-routed on their own,
    below)."""
    for spent in (SPENT_METAL_ROUTE, SECOND_SPENT_METAL_ROUTE):
        assert spent in route_stamps.SPENT["metal"], spent
        refused = route_stamps.campaign_refusal("metal", spent)
        assert refused and refused.split(":")[0] in (
            "route_campaign_is_spent", "route_campaign_is_not_this_stamp"), refused
    stamp = route_stamps.stamp("metal")
    assert stamp not in ("2026-10-18_101", "2026-10-25_101", route_stamps.PLACEHOLDER), stamp
    campaign = f"{route_stamps.route_gate('metal')}_applegpu_g13s"
    assert campaign == f"dispatch_metal_route_{stamp}_applegpu_g13s"
    assert campaign not in route_stamps.SPENT["metal"]
    assert route_stamps.campaign_refusal("metal", campaign) is None
    assert stamp not in (route_stamps.stamp("triton"), route_stamps.stamp("cuda")), stamp


SPENT_NVIDIA_ROUTES = {"triton": "dispatch_fused_route_2026-10-18_101_cc86",
                       "cuda": "dispatch_fused_route_cuda_2026-10-18_101_cc86"}
SPENT_NVIDIA_LICENCE = "dispatch_end_to_end_2026-10-18_101"


def test_the_nvidia_routes_of_the_round_are_refused_and_both_tables_share_a_new_stamp(
        monkeypatch):
    """The round's compute capability 8.6 route campaigns of both NVIDIA tables and its
    dispatch-by-default licence released on bytes the dispatch-preference priors have
    since moved (``fastpath.py``, which both route records bind), and their default rows
    read timing records these bytes replace, so they are never recorded. On the registry
    as committed: both route campaigns are spent and refused by name, the licence
    directory is refused as another campaign's, and the two NVIDIA tables share one stamp
    that is neither the round's, nor the Metal table's, nor any spent campaign's, under
    which the route campaigns and the licence of any compute capability (8.6, and 8.9
    certified in the same round) may be launched; the NVIDIA preflight names them."""
    for table, spent in SPENT_NVIDIA_ROUTES.items():
        assert spent in route_stamps.SPENT[table], (table, spent)
        refused = route_stamps.campaign_refusal(table, spent)
        assert refused and refused.split(":")[0] in (
            "route_campaign_is_spent", "route_campaign_is_not_this_stamp"), refused
    for licence in (SPENT_NVIDIA_LICENCE, f"{SPENT_NVIDIA_LICENCE}_cc86"):
        refused = route_stamps.licence_refusal("triton", licence, "cc86")
        assert refused and refused.startswith("licence_is_for_another_campaign"), refused
    stamp = route_stamps.stamp("triton")
    assert route_stamps.stamp("cuda") == stamp
    assert route_stamps.REROUTED.get("triton") == route_stamps.REROUTED.get("cuda") == stamp
    assert stamp not in (route_stamps.ROUND, route_stamps.stamp("metal"),
                         route_stamps.PLACEHOLDER), stamp
    for table in route_stamps.TABLES:
        assert not [name for name in route_stamps.SPENT[table] if stamp in name], table
    monkeypatch.setattr(fastpath, "primary_table", lambda capability: "triton")
    for capability, device in (("8.6", "cc86"), ("8.9", "cc89")):
        facts, problems = L.route_campaign_names(capability)
        assert problems == [], problems
        assert facts == {
            "triton_route_campaign": f"dispatch_fused_route_{stamp}_{device}",
            "cuda_route_campaign": f"dispatch_fused_route_cuda_{stamp}_{device}",
            "licence_campaign": f"dispatch_end_to_end_{stamp}"}, facts
        for table in ("triton", "cuda"):
            assert facts[f"{table}_route_campaign"] not in route_stamps.SPENT[table]
            assert route_stamps.campaign_refusal(table, facts[f"{table}_route_campaign"]) \
                is None
        licence = L.licence_directory(facts["licence_campaign"], capability)
        assert licence == f"dispatch_end_to_end_{stamp}_{device}"
        for table in route_stamps.LICENCE_TABLES:
            assert route_stamps.licence_refusal(table, licence, device) is None


def test_the_chunk_tooling_reads_the_rerouted_nvidia_stamp_as_the_round():
    """A chunked round shares its stamp with its route (``chunks.round_problems``), so on
    the registry as committed the NVIDIA round is the two NVIDIA tables' stamp: an NVIDIA
    index of round ``2026-10-18_101`` is refused by name on every device (and, on compute
    capability 8.6, as a round whose route has run), an index of the new stamp is not, and
    a Metal index of the Metal table's stamp is untouched."""
    stamp, _where = C.registry_round_stamp(ROOT, "nvidia")
    assert stamp == route_stamps.stamp("triton") == route_stamps.stamp("cuda"), stamp
    assert stamp != "2026-10-18_101"
    for device, mode in (("cc86", "certified"), ("cc89", "new")):
        old = C.make_index("nvidia", "2026-10-18_101", [], {"rule": "test"}, device=device,
                           device_mode=mode)
        problems = C.round_problems(ROOT, old)
        named = [line for line in problems
                 if f"the index is of round 2026-10-18_101; {C.REGISTRY} names {stamp}" in line]
        assert len(named) == 2, problems
        ran = [line for line in problems if "has already run" in line]
        assert len(ran) == (2 if device == "cc86" else 0), problems
        assert len(problems) == len(named) + len(ran), problems
        new = C.make_index("nvidia", stamp, [], {"rule": "test"}, device=device,
                           device_mode=mode)
        assert C.round_problems(ROOT, new) == []
    metal = C.make_index("metal", route_stamps.stamp("metal"), [], {"rule": "test"},
                         device="applegpu_g13s")
    assert C.round_problems(ROOT, metal) == []
