#!/usr/bin/env python3
"""Chunked certification: any set of a round's gate units runs as a chunk, on any slot.

WHY. Under per-kernel certification (R17) an artifact is valid wherever every file in its
import record, and every curated pin of the ledger entry it binds, is byte-equal in the
release tree. That is a property of the run of ONE UNIT, not of the batch it ran in. So any
subset of units can run as a chunk at any time and on any machine slot, a single unit can run
again alone, the plan can be re-partitioned freely, and nothing about one unit's run can
invalidate another's. This module is the bookkeeping that makes that safe: which units exist,
the one-shot names a chunk writes, readiness, and how the bind takes, per unit, the newest run
that holds on the release tree.

THE INDEX. One JSON file per machine and device (schema ``meep_gpu.cert_chunks/2``), kept
outside the tracked files: ``machine`` (``nvidia``: the CUDA and Triton CERT-3 fleet;
``metal``: the Metal fleet), ``device``, ``device_mode``, ``round``, ``next_k``, the current
plan ``chunks`` and the ``retired`` chunks a re-partition dropped before they launched. Each
chunk is an id (``c<k>`` on NVIDIA, ``m<k>`` on Metal), its stamp and its UNITS. ``partition``
writes an index once; ``add``, ``relaunch``, ``extend`` and ``repartition`` change it under
``<index>.lock``. A chunk id is allocated once from ``next_k`` and never reused, so a stamp is
one-shot. A unit may be in several chunks: each launch is one more run of it.

A UNIT is the smallest piece of evidence that binds on its own, both policies together:

* ``cuda/<subdirectory>``: a whole CUDA campaign subdirectory, keep and flush (a
  certification block claims a whole subdirectory, and the rebind requires the legs the
  entry stands on), and ``cuda/ade_recut``: the ADE ``update_P`` re-cut;
* ``triton/fleet/<gate>``: one Triton fleet gate under keep and flush, in the shard root
  ``legs.triton_units`` gives it; ``triton/composition`` (the composition step, its staged
  manifest and five gates), ``triton/stencil/<family>``, ``triton/bit_identity``, both
  policies each; ``triton/hd/<gate>``: one H->D hand leg (keep only, as ``legs.py`` runs it);
* ``metal/<gate stem>``: one Metal gate (flush only, as ``metal_round.py`` runs it).

THE DEVICE. An index certifies one device: ``cc<major><minor>`` on NVIDIA (the suffix
``fastpath.route_campaign`` gives a route campaign: ``cc86``, ``cc90``) and the GPU
architecture on Metal (``metal_runs.route_campaign``'s suffix: ``applegpu_g13s``).
``device_mode`` is ``certified`` for a device the ledgers already admit and ``new`` for one
they hold no live record for (an H200 at compute capability 9.0, an Apple M3): ``partition
--new-device`` writes it, and the round drivers refuse an index whose mode does not match the
ledgers or the flag they were given (``legs.py check --new-capability``, ``round.py
--new-capability``), so a new device is never admitted silently. A new device's bind adds
records beside the certified ones and re-points no board (:func:`records_commands`).

STAMPS. ``<round>_<device>_<letter><k>``: the round from the route-stamp registry
(``meep_gpu/route_stamps.py``; ``partition --round`` while it still reads its placeholder, and
the freeze then sets ``ROUND`` to that stamp), the device, the chunk id. Every results root a
chunk writes is a function of its stamp alone (:func:`chunk_roots`), so two chunks of any
device never share a root, and the device sits before the chunk id so no device's roots extend
another's by ``_`` (the bind tools treat ``<root>_<suffix>`` as the same run: ``_r<k>``
continuations, ``_flush`` twins, the board re-point's ``<base>_<stamp>_<suffix>``).
:func:`collisions` refuses an equal root and the prefix case, within the index and against the
roots present under ``results/``.

READINESS (``ready``). A chunk is READY when its closure holds none of the changing files.
Its closure is the union, over its units, of the static run closure of every script the unit
runs (``import_closure.py``, read from the tip being certified), the files the unit's
artifacts bind by argument (a ``--module`` kernel file, the ADE re-cut's subject manifest),
the curated pins of every ledger entry whose gate the unit runs, and, where the tip's static
reader still misses a reach through the package root's ``_LAZY`` table, the files the
previous round's artifacts of the same unit recorded (``records``). ``stale`` asks, per unit
of each launched chunk, whether a file of that closure moved since the chunk's launch tip;
``add --units`` re-runs such units alone. Both read CHECKOUT FILES only: an environment
change (a rebuilt reference environment, another MEEP) moves no file they read and reads 0
stale, so it is never their bill; a round in a new environment re-runs under a new round stamp.

THE LAUNCH RECORD. A launched chunk writes ``results/chunk_<stamp>/LAUNCH.json`` (schema
``meep_gpu.cert_chunk_launch/2``: machine, device, round, chunk, stamp, units, tip, tree, utc)
beside its artifacts; a resume must match it. It is the authority on what ran under a stamp:
:func:`verify` reads the units from it, never from the index's plan.

VERIFY (per unit, ``verify``). A unit's CANDIDATES are the launches, of every chunk the index
ever allocated (current or retired), whose record lists it. A candidate HOLDS when it left an
artifact, every artifact's import record names only files the release tree holds at the
recorded bytes (``artifact_drift.py``, the rule the rebind tools apply too) or, with no whole
import record, no file of the unit's closure moved since its launch tip, every verdict
artifact the unit's legs write passes (``round_status.py``), and the device it ran on is the
index's, read and never assumed (:func:`run_devices`: on NVIDIA the compute capability its
artifacts stamp and a Triton campaign's ``device.json``, or for a unit with neither the one
the rest of its launch's artifacts stamp; on Metal the launch's environment record); a run
whose device cannot be read is refused. The unit takes the NEWEST candidate that holds
(launch time, then chunk number): OK. Otherwise REFUSED (it ran, and no run holds), ABSENT
(it ran and left no artifact) or NOT LAUNCHED. The verdict is N of D over the release tree's
units, and nothing in it reads the chunk a unit was planned in, so re-partitioning the index
moves no verdict.
``--selection F`` writes the choice, with every candidate and why it was or was not taken;
``--against F`` judges again, writes nothing and exits 1 when the choice, or the bytes of a
selected run, moved since ``F`` was written.

THE BIND (``bind --selection F``) prints the commands that bind exactly that selection; nothing
here writes a ledger. A tool that takes a key list (``rebind_triton_welds.py --only``) is
called once per selected stamp with the keys of the units selected there that the checkout's
Triton ledger holds: the tool refuses a whole ``--only`` that names a key the ledger lacks,
and ``CAMPAIGN_DIRS`` may type a row before the seed that writes its key, so a selected unit
whose key the ledger lacks is named on a ``# REFUSED`` line and left out, and a stamp left
with no key prints no Triton line. A tool that binds whatever a stamp holds
(``rebind_metal_welds.py --stamp``, ``rebind_cuda_welds.py
--campaign``) is called once per selected stamp in launch order, so a newer selected run
replaces an older one, and a run that tool would bind against the selection is named on a
STOP line with the way to set it aside. ``record_cuda_regate.py`` records whole campaigns and
refuses one holding an unreleased or moved leg, so such a subdirectory of a campaign the
selection records is named the same way, as is a unit's directory under a selected stamp
whose launch record does not list it. While any STOP line stands the bind prints no command
that writes a ledger, ends with ``# BIND WITHHELD`` and exits 1. Its first command is
``verify --against F``. The Triton family re-certification takes, per family, the first
root holding a released artifact: its roots are ordered so that root is the selected one,
or the conflict is named.

usage (from ``parity/meep_gpu``; INDEX is a path outside the checkout's tracked files):
  chunks.py partition --repo R --machine nvidia --capability 8.6 [--new-device] --out INDEX
                      [--chunks N] [--round S] [--defer PATTERN ...] [--prior-records F]
  chunks.py partition --repo R --machine metal --architecture applegpu_g13s [--new-device] ...
  chunks.py show      --index INDEX
  chunks.py check     --index INDEX --repo R [--results DIR]
  chunks.py plan      --index INDEX --chunk c0 --repo R --results DIR --out DIR [--lanes N]
  chunks.py records   --repo R --results DIR --stamp PRIOR --machine M --out F
  chunks.py ready     --index INDEX --repo TIP [--changing P ...] [--changing-file F]
                      [--since COMMIT] [--prior-records F] [--json F]
  chunks.py stale     --index INDEX --repo TIP --results DIR [--prior-records F] [--json F]
  chunks.py verify    --index INDEX --repo TIP --results DIR [--selection F | --against F]
                      [--json F]
  chunks.py bind      --index INDEX --selection F [--results DIR] [--write]
  chunks.py add       --index INDEX --repo R (--units U,.. | --units-file F | --from-verify F)
                      --why TEXT [--chunks N]
  chunks.py relaunch  --index INDEX --chunk c1 --why TEXT
  chunks.py extend    --index INDEX --repo R
  chunks.py repartition --index INDEX --repo R --results DIR --why TEXT [--chunks N]
"""
from __future__ import annotations

import argparse
import ast
import fnmatch
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
import types
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

HERE = Path(__file__).resolve().parent
HARNESS = HERE.parents[1]
for _path in (str(HERE), str(HARNESS)):
    if _path not in sys.path:
        sys.path.insert(0, _path)
import legs as L  # noqa: E402
import metal_round as M  # noqa: E402
import round_status as RS  # noqa: E402
import artifact_drift  # noqa: E402

SCHEMA = "meep_gpu.cert_chunks/2"
#: Index shapes this tool no longer reads, each refused by name.
RETIRED_SCHEMAS = {"meep_gpu.cert_chunks/1": (
    "an index of the first chunk tooling (no device, units tied to one chunk): partition a "
    "new index for the round's device")}
MACHINES = ("nvidia", "metal")
#: The chunk id letter per machine, so an NVIDIA and a Metal chunk never share a stamp.
LETTER = {"nvidia": "c", "metal": "m"}
#: The default chunk counts: four NVIDIA chunks (the account's four GPU locks), and three
#: Metal chunks run one after another on the one Mac GPU.
DEFAULT_CHUNKS = {"nvidia": 4, "metal": 3}
POLICIES = ("keep", "flush")
DEVICE_MODES = ("certified", "new")
#: A chunk id: the machine's letter and a number allocated once from ``next_k``.
CHUNK_ID = re.compile(r"^(?P<letter>[cm])(?P<k>0|[1-9]\d*)$")
#: A round stamp: a date first (``repoint_board_bindings.STAMP_START``), then word characters.
ROUND_STAMP = re.compile(r"^\d{4}-\d\d-\d\d[A-Za-z0-9_.-]*$")
#: The device a stamp carries: ``cc<major><minor>`` (a normalised compute capability, as
#: ``fastpath._CAPABILITY_KEY`` spells it, without the dot) on NVIDIA; the GPU architecture
#: (``metal_runs.ARCHITECTURE_KEY``) on Metal.
DEVICE_KEY = {"nvidia": re.compile(r"^cc[1-9][0-9]*[0-9]$"),
              "metal": re.compile(r"^applegpu_[a-z0-9_]+$")}
#: A gate the run closure of a tip must reach ``from_meep.py`` from through the package root's
#: ``_LAZY`` table (it calls ``meep_gpu.lift_simulation``): when the tip's static reader does
#: not, the reader misses ``_LAZY`` reaches and readiness needs the recorded import records.
LAZY_PROBE = ("parity/meep_gpu/gate_triton_fused_hd_pair.py", "meep_gpu/from_meep.py")
SUBS_KEYS = ("@FLEET@", "@COMPOSITION@", "@STENCIL@", "@BITID@", "@RESULTS@", "@STAMP@")
#: The unit states :func:`verify` reports, and the one that counts towards N.
UNIT_STATES = ("OK", "REFUSED", "ABSENT", "NOT LAUNCHED")


# ----------------------------------------------------------------------- devices -------
def device_of_capability(capability: str) -> str:
    """``9.0`` -> ``cc90`` (``legs.capability_device``, the one rule)."""
    return L.capability_device(capability)


def device_problems(machine: str, device: Any) -> List[str]:
    if machine not in MACHINES:
        return [f"machine {machine!r} is not one of {list(MACHINES)}"]
    if not isinstance(device, str) or not DEVICE_KEY[machine].match(device):
        return [f"device {device!r} is not a {machine} device ("
                + ("cc<major><minor>, e.g. cc86" if machine == "nvidia" else
                   "a GPU architecture, e.g. applegpu_g13s") + ")"]
    return []


# ------------------------------------------------------------------------ stamps -------
#: The route-stamp registry, and the tables whose stamp names each machine's round.
REGISTRY = "meep_gpu/route_stamps.py"
REGISTRY_TABLES = {"nvidia": ("triton", "cuda"), "metal": ("metal",)}


def load_registry(repo: Path):
    """The checkout's own ``meep_gpu/route_stamps.py`` (standard library only), loaded by
    path so no package module is imported; None when the checkout has none."""
    path = Path(repo) / REGISTRY
    if not path.is_file():
        return None
    # Compiled from the source each time, never from a bytecode cache: a same-second edit
    # of the same size would otherwise read the previous stamp.
    module = types.ModuleType("_checkout_route_stamps")
    module.__file__ = str(path)
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), module.__dict__)
    return module


def registry_round_stamp(repo: Path, machine: str) -> Tuple[Optional[str], str]:
    """``(round stamp, where it was read)`` for ``machine`` from the route-stamp registry,
    or ``(None, why not)``: no registry, the stamp still the placeholder the freeze
    replaces, or the machine's tables on different stamps."""
    registry = load_registry(repo)
    if registry is None:
        return None, f"the checkout has no route-stamp registry ({REGISTRY})"
    tables = REGISTRY_TABLES[machine]
    stamps = {table: registry.stamp(table) for table in tables}
    if any(value == registry.PLACEHOLDER for value in stamps.values()):
        return None, (f"{REGISTRY} names no stamp yet (ROUND = {registry.PLACEHOLDER!r}; the "
                      f"freeze sets it)")
    if len(set(stamps.values())) != 1:
        return None, f"the {machine} tables run under different stamps {stamps}"
    return next(iter(stamps.values())), f"{REGISTRY}:stamp({', '.join(tables)})"


def round_problems(repo: Path, index: Mapping[str, Any]) -> List[str]:
    """Why the index's round does not fit the checkout's route-stamp registry.

    Refused: no registry; a registry stamp other than the index's round (once the freeze
    has set one, the chunks and the route campaigns share it); a round whose route
    campaign has already run ON THIS DEVICE (``route_stamps.SPENT``: one route campaign per
    stamp and device, and a chunked round shares its stamp with them). A registry still
    reading the placeholder is not a refusal: the chunks run before the freeze, which must
    set ``ROUND`` to the index's round.
    """
    registry = load_registry(repo)
    if registry is None:
        return [f"the checkout has no route-stamp registry ({REGISTRY})"]
    problems = []
    round_stamp = index["round"]
    device = index.get("device")
    for table in REGISTRY_TABLES[index["machine"]]:
        value = registry.stamp(table)
        if value != registry.PLACEHOLDER and value != round_stamp:
            problems.append(f"the index is of round {round_stamp}; {REGISTRY} names "
                            f"{value} for the {table} table")
        stem = registry.ROUTE_PREFIXES[table] + round_stamp
        spent = [name for name in registry.SPENT.get(table, ())
                 if name == stem or name == f"{stem}_{device}"]
        if spent:
            problems.append(f"round {round_stamp} has already run a {table} route campaign "
                            f"on {device} ({spent[0]}, route_stamps.SPENT): take a new round "
                            f"stamp")
    return problems


def chunk_stamp(round_stamp: str, machine: str, k: int, device: str) -> str:
    """``<round>_<device>_<letter><k>``."""
    return f"{round_stamp}_{device}_{LETTER[machine]}{int(k)}"


def stamp_subs(results: Path, stamp: str) -> Dict[str, str]:
    """Every placeholder the leg tables spell, for one stamp (``round.py`` uses this)."""
    return {"@FLEET@": str(results / f"triton_fleet_{stamp}"),
            "@COMPOSITION@": str(results / f"triton_composition_{stamp}"),
            "@STENCIL@": str(results / f"triton_offdiag_stencil_welds_{stamp}"),
            "@BITID@": str(results / f"triton_bit_identity_{stamp}"),
            "@RESULTS@": str(results), "@STAMP@": stamp}


def _sub(value: Any, subs: Mapping[str, str]) -> Any:
    if isinstance(value, str):
        for key, rep in subs.items():
            value = value.replace(key, rep)
        return value
    if isinstance(value, list):
        return [_sub(v, subs) for v in value]
    if isinstance(value, dict):
        return {k: _sub(v, subs) for k, v in value.items()}
    return value


# ------------------------------------------------------------------------- units -------
def fleet_gate_scripts(repo: Path) -> Dict[str, str]:
    """``fleet gate name -> repo-relative script``, read off ``drive_triton_weld_gates.py``."""
    path = Path(repo) / "parity" / "meep_gpu" / "drive_triton_weld_gates.py"
    out: Dict[str, str] = {}
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "Gate" and len(node.args) >= 2
                and all(isinstance(a, ast.Constant) and isinstance(a.value, str)
                        for a in node.args[:2])):
            out[node.args[0].value] = f"parity/meep_gpu/{node.args[1].value}"
    return out


def _repo_rel(path: str, repo: Path) -> Optional[str]:
    try:
        rel = Path(path).resolve().relative_to(Path(repo).resolve()).as_posix()
    except ValueError:
        return None
    return rel


def nvidia_units(repo: Path) -> List[Dict[str, Any]]:
    """The NVIDIA fleet as units (see the module docstring), in a fixed order."""
    repo = Path(repo)
    units: Dict[str, Dict[str, Any]] = {}
    for leg in L.cuda_legs(repo, repo / "parity" / "meep_gpu" / "results", "@STAMP@"):
        uid = "cuda/ade_recut" if leg["subdir"] is None else f"cuda/{leg['subdir']}"
        unit = units.setdefault(uid, {
            "id": uid, "kind": "cuda_ade" if leg["subdir"] is None else "cuda_subdir",
            "subdir": leg["subdir"], "expect_s": 0, "scripts": set(), "extra": set(),
            "policies": set()})
        unit["expect_s"] += int(leg["expect_s"])
        unit["scripts"].add(_repo_rel(leg["script"], repo))
        unit["policies"].add(leg["policy"])
        for arg in leg["args"]:
            rel = _repo_rel(arg, repo) if arg.endswith(".py") else None
            if rel:
                unit["extra"].add(rel)
        if leg["subdir"] is None:
            unit["extra"].update(L.ADE_SUBJECTS)
    gates = fleet_gate_scripts(repo)
    for tunit in L.triton_units():
        for step in tunit["steps"]:
            policy, kind = step["policy"], step["kind"]
            if kind == "fleet" and step.get("manifest"):
                uid, extra = "triton/composition", {"gates": list(step["gates"])}
                scripts = [gates.get(g) for g in step["gates"]]
                expect = int(step["expect_s"])
            elif kind == "fleet":
                shard = step["id"].split("/", 2)[2]
                for g in step["gates"]:
                    uid = f"triton/fleet/{g}"
                    unit = units.setdefault(uid, {"id": uid, "kind": "triton_fleet", "gate": g,
                                                  "shard": shard, "expect_s": 0,
                                                  "scripts": set(), "extra": set()})
                    if unit["shard"] != shard:
                        raise ValueError(f"{g} runs in shard {unit['shard']} and {shard}")
                    unit["expect_s"] += int(L.TRITON_EXPECT[g])
                    unit["scripts"].add(gates.get(g))
                    unit.setdefault("policies", set()).add(policy)
                continue
            elif kind == "stencil":
                uid, extra = f"triton/stencil/{step['family']}", {"family": step["family"]}
                scripts, expect = ["parity/meep_gpu/gate_triton_offdiag_stencil_welds.py"], \
                    int(step["expect_s"])
            elif kind == "bitid":
                uid, extra = "triton/bit_identity", {}
                scripts, expect = ["parity/meep_gpu/probe_fused_kernel_bit_identity.py"], \
                    int(step["expect_s"])
            elif kind == "hd":
                uid, extra = f"triton/hd/{step['gate']}", {"gate": step["gate"]}
                scripts, expect = [f"parity/meep_gpu/{step['script']}"], int(step["expect_s"])
            else:
                raise ValueError(f"unknown Triton step kind {kind!r}")
            unit = units.setdefault(uid, {"id": uid, "kind": f"triton_{kind}" if kind != "fleet"
                                          else "triton_composition", "expect_s": 0,
                                          "scripts": set(), "extra": set(), **extra})
            unit["expect_s"] += expect
            unit["scripts"].update(scripts)
            unit.setdefault("policies", set()).add(policy)
            if kind == "bitid":
                unit["extra"].add("parity/meep_gpu/track_triton_pml.py")
    missing = sorted(u["id"] for u in units.values() if None in u["scripts"])
    if missing:
        raise ValueError(f"units with a gate the fleet driver does not name: {missing}")
    return [_frozen(u) for u in sorted(units.values(), key=lambda u: u["id"])]


def metal_units(repo: Path) -> List[Dict[str, Any]]:
    """One unit per ``gate_metal_*.py`` of the checkout, priced by ``metal_round.EXPECT``."""
    harness = Path(repo) / "parity" / "meep_gpu"
    out = []
    for script in sorted(harness.glob("gate_metal_*.py")):
        stem = script.stem
        out.append(_frozen({"id": f"metal/{stem}", "kind": "metal_gate", "gate": stem,
                            "family": stem[len("gate_"):],
                            "expect_s": int(M.EXPECT.get(stem, 300)),
                            "scripts": {f"parity/meep_gpu/{stem}.py"}, "extra": set(),
                            "policies": {"flush"}}))
    return out


def _frozen(unit: Dict[str, Any]) -> Dict[str, Any]:
    return {k: (sorted(v) if isinstance(v, set) else v) for k, v in unit.items()}


def units_for(repo: Path, machine: str) -> List[Dict[str, Any]]:
    return nvidia_units(repo) if machine == "nvidia" else metal_units(repo)


# --------------------------------------------------------------------- partition -------
def lpt(units: Sequence[Mapping[str, Any]], n: int) -> List[List[Mapping[str, Any]]]:
    """Longest-processing-time packing into ``n`` bins, deterministic (ties by id)."""
    bins: List[List[Mapping[str, Any]]] = [[] for _ in range(max(1, n))]
    load = [0] * len(bins)
    for unit in sorted(units, key=lambda u: (-int(u["expect_s"]), u["id"])):
        k = min(range(len(bins)), key=lambda i: (load[i], i))
        bins[k].append(unit)
        load[k] += int(unit["expect_s"])
    return bins


def partition(units: Sequence[Mapping[str, Any]], n: int,
              deferred: Optional[Set[str]] = None) -> List[List[Mapping[str, Any]]]:
    """``n`` chunks balanced by the measured walls.

    With ``deferred`` (unit ids whose closure holds a file still changing), the units split
    into two groups first, so the ones that can certify now are not held back by a deferred
    neighbour: the free group comes first and gets chunks in proportion to its wall (at least
    one each while ``n`` allows), and each group is packed by :func:`lpt`.
    """
    if not deferred or n < 2:
        return [b for b in lpt(units, n) if b] if units else []
    free = [u for u in units if u["id"] not in deferred]
    held = [u for u in units if u["id"] in deferred]
    if not free or not held:
        return [b for b in lpt(units, n) if b]
    wall = lambda group: sum(int(u["expect_s"]) for u in group)  # noqa: E731
    n_free = min(n - 1, max(1, round(n * wall(free) / (wall(free) + wall(held)))))
    return [b for b in lpt(free, n_free) if b] + [b for b in lpt(held, n - n_free) if b]


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def new_chunk(index: Dict[str, Any], units: Sequence[Mapping[str, Any]], why: str,
              gpus: int = 1) -> Dict[str, Any]:
    """Allocate the next chunk id of ``index`` (never reused) for ``units``; appends it."""
    k = int(index["next_k"])
    index["next_k"] = k + 1
    chunk = {"id": f"{LETTER[index['machine']]}{k}",
             "stamp": chunk_stamp(index["round"], index["machine"], k, index["device"]),
             "gpus": int(gpus), "expect_s": sum(int(u["expect_s"]) for u in units),
             "units": sorted({u["id"] for u in units}), "why": why, "utc": _utc()}
    index["chunks"].append(chunk)
    return chunk


def make_index(machine: str, round_stamp: str, bins: Sequence[Sequence[Mapping[str, Any]]],
               rule: Dict[str, Any], device: str, device_mode: str = "certified"
               ) -> Dict[str, Any]:
    index: Dict[str, Any] = {"schema": SCHEMA, "machine": machine, "device": device,
                             "device_mode": device_mode, "round": round_stamp,
                             "partition": rule, "next_k": 0, "chunks": [], "retired": []}
    for units in bins:
        new_chunk(index, units, "partition")
    return index


# ------------------------------------------------------------------------- index -------
def index_shape_problems(index: Any) -> List[str]:
    """Why ``index`` is not a chunk index of this tool's schema; empty when it is."""
    if not isinstance(index, dict):
        return ["not a JSON object"]
    schema = index.get("schema")
    if schema in RETIRED_SCHEMAS:
        return [f"retired_index_schema: {schema} is {RETIRED_SCHEMAS[schema]}"]
    if schema != SCHEMA:
        return [f"not a chunk index ({SCHEMA}): schema {schema!r}"]
    problems = device_problems(index.get("machine"), index.get("device"))
    if index.get("device_mode") not in DEVICE_MODES:
        problems.append(f"device_mode {index.get('device_mode')!r} is not one of "
                        f"{list(DEVICE_MODES)}")
    if not isinstance(index.get("next_k"), int) or index["next_k"] < 0:
        problems.append(f"next_k {index.get('next_k')!r} is not a count")
    for key in ("chunks", "retired"):
        if not isinstance(index.get(key), list):
            problems.append(f"{key} is not a list")
    return problems


def read_index(path: Path) -> Dict[str, Any]:
    index = json.loads(Path(path).read_text(encoding="utf-8"))
    problems = index_shape_problems(index)
    if problems:
        raise SystemExit(f"{path}: {'; '.join(problems)}")
    return index


def write_index(path: Path, index: Mapping[str, Any]) -> None:
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(index, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


class IndexLock:
    """``mkdir <index>.lock`` with the holder's pid, for the commands that change an index."""

    def __init__(self, path: Path, label: str):
        self.dir = Path(str(path) + ".lock")
        self.label = label

    def __enter__(self):
        try:
            self.dir.mkdir()
        except FileExistsError:
            pid = (self.dir / "pid").read_text().strip() if (self.dir / "pid").is_file() else ""
            if pid.isdigit() and _alive(int(pid)):
                raise SystemExit(f"{self.dir} is held by pid {pid}; try again when it ends")
            (self.dir / "pid").write_text(f"{os.getpid()}\n")
        (self.dir / "pid").write_text(f"{os.getpid()}\n")
        (self.dir / "label").write_text(f"{self.label}\n")
        return self

    def __exit__(self, *exc):
        for name in ("pid", "label"):
            try:
                (self.dir / name).unlink()
            except OSError:
                pass
        try:
            self.dir.rmdir()
        except OSError:
            pass


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def find_chunk(index: Mapping[str, Any], chunk_id: str) -> Dict[str, Any]:
    for chunk in index["chunks"]:
        if chunk["id"] == chunk_id:
            return chunk
    retired = [c["id"] for c in index.get("retired", ())]
    raise SystemExit(f"the index has no current chunk {chunk_id!r} "
                     f"(current {[c['id'] for c in index['chunks']]}"
                     + (f"; retired {retired}" if retired else "") + ")")


def allocated(index: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Every chunk the index ever allocated, current then retired, in id order."""
    every = list(index["chunks"]) + list(index.get("retired", ()))
    return sorted(every, key=lambda c: _chunk_k(c["id"]))


def _chunk_k(chunk_id: str) -> int:
    match = CHUNK_ID.match(chunk_id or "")
    return int(match.group("k")) if match else -1


def chunk_by_stamp(index: Mapping[str, Any], stamp: str) -> Optional[Dict[str, Any]]:
    for chunk in allocated(index):
        if chunk["stamp"] == stamp:
            return chunk
    return None


# -------------------------------------------------------------------------- plans ------
def _keep_step(step: Mapping[str, Any], ids: Set[str]) -> Optional[Dict[str, Any]]:
    kind = step["kind"]
    if kind == "fleet" and step.get("manifest"):
        return dict(step) if "triton/composition" in ids else None
    if kind == "fleet":
        gates = [g for g in step["gates"] if f"triton/fleet/{g}" in ids]
        if not gates:
            return None
        return dict(step, gates=gates, expect_s=sum(int(L.TRITON_EXPECT[g]) for g in gates))
    if kind == "stencil":
        return dict(step) if f"triton/stencil/{step['family']}" in ids else None
    if kind == "bitid":
        return dict(step) if "triton/bit_identity" in ids else None
    if kind == "hd":
        return dict(step) if f"triton/hd/{step['gate']}" in ids else None
    raise ValueError(f"unknown Triton step kind {kind!r}")


def nvidia_plan(repo: Path, results: Path, chunk: Mapping[str, Any], lanes: int = 1
                ) -> Tuple[List[List[Dict[str, Any]]], Dict[str, Any]]:
    """The chunk's gate lanes: its CUDA campaign (one, ``cuda_regate_<stamp>_g0``) and its
    Triton steps (``legs.triton_units`` with only the chunk's gates), packed onto ``lanes``
    lanes by ``legs.pack_lanes``. Placeholders are left for ``round.py`` to fill."""
    ids = set(chunk["units"])
    stamp = chunk["stamp"]
    subdirs = {u[len("cuda/"):] for u in ids if u.startswith("cuda/") and u != "cuda/ade_recut"}
    ade = "cuda/ade_recut" in ids
    units: List[Dict[str, Any]] = []
    legs = [leg for leg in L.cuda_legs(repo, results, stamp)
            if (leg["subdir"] in subdirs) or (leg["subdir"] is None and ade)]
    if legs:
        lane = L.split_cuda(legs, 1)[0]
        units.append({"name": "cuda_g0", "steps": L.materialise_cuda(
            lane, results / f"cuda_regate_{stamp}_g0", results / f"cuda_ade_recut_{stamp}")})
    for tunit in L.triton_units():
        steps = [kept for kept in (_keep_step(s, ids) for s in tunit["steps"]) if kept]
        if steps:
            units.append({"name": tunit["name"], "steps": steps})
    packed = L.pack_lanes(units, max(1, lanes))
    plan = [[step for unit in lane for step in unit["steps"]] for lane in packed]
    summary = {"chunk": chunk["id"], "stamp": stamp, "lanes": [
        {"units": [u["name"] for u in lane], "steps": len(p),
         "expect_s": sum(int(s["expect_s"]) for s in p)} for lane, p in zip(packed, plan)]}
    summary["wall_s"] = max((lane["expect_s"] for lane in summary["lanes"]), default=0)
    return plan, summary


def metal_gates(chunk: Mapping[str, Any]) -> List[str]:
    """The chunk's Metal gate stems, in ``metal_round.py``'s run order (the three longest
    last)."""
    stems = sorted(u[len("metal/"):] for u in chunk["units"] if u.startswith("metal/"))
    return [s for s in stems if s not in M.LONG] + [s for s in M.LONG if s in stems]


def chunk_roots(chunk: Mapping[str, Any], machine: str, stamp: Optional[str] = None
                ) -> List[str]:
    """Every results root the chunk writes under ``stamp`` (default: its own)."""
    stamp = stamp or chunk["stamp"]
    if machine == "metal":
        names = {f"{stem[len('gate_'):]}_{stamp}" for stem in metal_gates(chunk)}
        names.update({f"metal_environment_{stamp}", f"metal_campaign_{stamp}",
                      f"logs_metal_{stamp}", launch_root(stamp)})
        return sorted(names)
    results = Path("/results")
    plan, _summary = nvidia_plan(Path("/repo"), results, dict(chunk, stamp=stamp), 1)
    subs = stamp_subs(results, stamp)
    names: Set[str] = {launch_root(stamp)}
    for step in (s for lane in plan for s in lane):
        step = _sub(step, subs)
        path = Path(step.get("artifact") or step["root"])
        names.add(path.relative_to(results).parts[0])
    return sorted(names)


# ------------------------------------------------------------------ launch records -----
#: The chunk's own root holding its launch record, beside its artifacts.
LAUNCH_SCHEMA = "meep_gpu.cert_chunk_launch/2"


def launch_root(stamp: str) -> str:
    return f"chunk_{stamp}"


def _git(repo: Path, *args: str) -> Optional[str]:
    done = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
    return done.stdout.strip() if done.returncode == 0 else None


def write_launch_record(results: Path, index: Mapping[str, Any], chunk: Mapping[str, Any],
                        repo: Path) -> Optional[str]:
    """Write ``results/chunk_<stamp>/LAUNCH.json`` (machine, device, round, chunk, stamp,
    units, tip, tree, utc), or find it written by an earlier launch of the same stamp on the
    same tip. Returns why the launch must not go on (another tip or unit set), or None.

    The record is what :func:`verify` reads as the units that ran under the stamp, and the
    tip an artifact with no import record is judged by. A chunk resumes on the tip it
    launched on; on any other tip its units run again in a new chunk (``chunks.py add`` or
    ``relaunch``), and both runs stay evidence.
    """
    tip, tree = _git(repo, "rev-parse", "HEAD"), _git(repo, "rev-parse", "HEAD^{tree}")
    if not tip or not tree:
        return f"{repo} is not a git checkout: a chunk records the tip it ran on"
    record = {"schema": LAUNCH_SCHEMA, "machine": index["machine"], "device": index["device"],
              "round": index["round"], "chunk": chunk["id"], "stamp": chunk["stamp"],
              "units": sorted(chunk["units"]), "tip": tip, "tree": tree}
    path = Path(results) / launch_root(chunk["stamp"]) / "LAUNCH.json"
    if path.is_file():
        old = json.loads(path.read_text(encoding="utf-8"))
        moved = sorted(k for k in record if old.get(k) != record[k])
        if moved:
            return (f"{path} records another launch ({', '.join(f'{k} {old.get(k)!r}' for k in moved[:3])}): "
                    f"a chunk resumes only on the tip and units it launched with; run its "
                    f"units again in a new chunk (chunks.py relaunch or add)")
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    write_index(path, dict(record, utc=_utc()))
    return None


def read_launch_record(results: Path, stamp: str) -> Optional[Dict[str, Any]]:
    """The launch record of ``stamp``, whatever its schema (None when there is none)."""
    path = Path(results) / launch_root(stamp) / "LAUNCH.json"
    if not path.is_file():
        return None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"_unreadable": str(path)}
    return record if isinstance(record, dict) else {"_unreadable": str(path)}


def launch_problems(record: Mapping[str, Any], index: Mapping[str, Any],
                    chunk: Mapping[str, Any]) -> List[str]:
    """Why a launch record under one of the index's stamps is not that chunk's launch."""
    if record.get("_unreadable"):
        return [f"{record['_unreadable']} is not readable JSON"]
    if record.get("schema") != LAUNCH_SCHEMA:
        return [f"{launch_root(chunk['stamp'])}/LAUNCH.json is schema "
                f"{record.get('schema')!r}, not {LAUNCH_SCHEMA}"]
    want = {"machine": index["machine"], "device": index["device"], "round": index["round"],
            "chunk": chunk["id"], "stamp": chunk["stamp"]}
    problems = [f"{launch_root(chunk['stamp'])}/LAUNCH.json names {key} {record.get(key)!r}, "
                f"the index {value!r}" for key, value in want.items() if record.get(key) != value]
    if not isinstance(record.get("units"), list):
        problems.append(f"{launch_root(chunk['stamp'])}/LAUNCH.json names no units")
    return problems


# ----------------------------------------------------------------------- collisions ----
def _roots_of_index(index: Mapping[str, Any]) -> List[Tuple[str, str]]:
    names: List[Tuple[str, str]] = []
    for chunk in allocated(index):
        names += [(name, chunk["stamp"]) for name in chunk_roots(chunk, index["machine"])]
    return names


def collisions(index: Mapping[str, Any], results: Optional[Path] = None,
               repo: Optional[Path] = None) -> List[str]:
    """Why the index's chunks could collide; empty when they cannot.

    Refused: an index of another shape (:func:`index_shape_problems`); a round stamp that
    does not start with a date, or carries the word flush; a chunk id that is not
    ``<letter><k>`` with ``k`` below ``next_k``, or that two chunks share; a stamp that is not
    ``<round>_<device>_<letter><k>``; a unit named twice in one chunk; with ``repo``, a unit of
    the checkout in no current chunk, or a current chunk's unit the checkout does not have;
    two stamps' roots equal, or one followed by ``_`` starting the other; and, with
    ``results``, a root present there that is not the index's own and that one of its roots
    followed by ``_`` starts (a bind tool would read it as the chunk's continuation or twin),
    or that, followed by ``_``, starts one of its roots (the other way round: a run of the
    round's bare stamp, or of another device, that a chunk's root would extend).

    A unit in several chunks is not a collision: each launch is one more run of it.
    """
    problems: List[str] = list(index_shape_problems(index))
    if problems:
        return problems
    machine, round_stamp, device = index["machine"], index["round"], index["device"]
    if not ROUND_STAMP.match(round_stamp or "") or "flush" in round_stamp.split("_"):
        problems.append(f"the round stamp {round_stamp!r} does not start with a date or "
                        f"carries the word flush")
    seen_ids: Dict[str, str] = {}
    for where, chunks in (("current", index["chunks"]), ("retired", index["retired"])):
        for chunk in chunks:
            cid = chunk.get("id", "")
            m = CHUNK_ID.match(cid)
            if not m or m.group("letter") != LETTER[machine]:
                problems.append(f"chunk id {cid!r} is not {LETTER[machine]}<k>")
                continue
            if int(m.group("k")) >= int(index["next_k"]):
                problems.append(f"chunk {cid} is not below next_k {index['next_k']}: an id "
                                f"is allocated once, by chunks.py")
            if cid in seen_ids:
                problems.append(f"chunk id {cid} is {seen_ids[cid]} and {where}")
            seen_ids[cid] = where
            expected = chunk_stamp(round_stamp, machine, int(m.group("k")), device)
            if chunk.get("stamp") != expected:
                problems.append(f"chunk {cid} carries stamp {chunk.get('stamp')!r}, not "
                                f"{expected!r}")
            units = list(chunk.get("units") or ())
            if len(set(units)) != len(units):
                problems.append(f"chunk {cid} names a unit twice")
    if repo is not None:
        have = {u["id"] for u in units_for(repo, machine)}
        placed = {u for chunk in index["chunks"] for u in chunk.get("units", ())}
        missing = sorted(have - placed)
        extra = sorted(placed - have)
        if missing:
            problems.append(f"{len(missing)} unit(s) of the checkout are in no chunk (run "
                            f"chunks.py extend): {missing[:6]}")
        if extra:
            problems.append(f"{len(extra)} chunk unit(s) the checkout does not have: "
                            f"{extra[:6]}")
    names = _roots_of_index(index)
    problems += name_collisions(names)
    if results is not None and Path(results).is_dir():
        own = {name for name, _stamp in names}
        present = sorted(p.name for p in Path(results).iterdir())
        # A chunk's own root, or a resume continuation of one (``<root>_r<k>``).
        mine = lambda p: p in own or re.sub(r"_r\d+$", "", p) in own  # noqa: E731
        foreign = [p for p in present if not mine(p)]
        for name, stamp in names:
            for hit in (p for p in foreign if p.startswith(name + "_")):
                problems.append(f"{Path(results).name}/{hit} starts with the root {name} of "
                                f"stamp {stamp} followed by '_', and is not that chunk's: a "
                                f"bind tool would read it as the same run")
            for hit in (p for p in foreign if f"_{round_stamp}" in p
                        and name.startswith(p + "_")):
                problems.append(f"the root {name} of stamp {stamp} starts with "
                                f"{Path(results).name}/{hit} followed by '_', which is not "
                                f"the index's: a bind tool reading {hit} would read the "
                                f"chunk's root as its continuation or twin")
    return problems


def name_collisions(names: Sequence[Tuple[str, str]]) -> List[str]:
    """Pairs of roots of DIFFERENT stamps that are equal, or one a ``<root>_`` prefix of the
    other."""
    problems = []
    ordered = sorted(names)
    for i, (a, sa) in enumerate(ordered):
        for b, sb in ordered[i + 1:]:
            if not (b == a or b.startswith(a + "_")):
                if b > a + "_\U0010ffff":
                    break
                continue
            if sa != sb:
                problems.append(f"root {b} of stamp {sb} " + ("equals" if a == b else
                                "starts with") + f" root {a} of stamp {sa}")
    return problems


# ----------------------------------------------------------------- closures, ready -----
def load_import_closure(repo: Path):
    """The tip's own ``import_closure.py``, loaded by path (never this tree's)."""
    path = Path(repo) / "parity" / "meep_gpu" / "import_closure.py"
    spec = importlib.util.spec_from_file_location("_tip_import_closure", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def lazy_gap_open(resolver) -> bool:
    """True when the tip's static run closure misses the package root's ``_LAZY`` reaches."""
    gate, target = LAZY_PROBE
    if not resolver.exists(gate):
        return True
    return target not in resolver.run_closure(gate)


def ledger_pins(repo: Path, ic) -> Dict[str, Dict[str, Set[str]]]:
    """``gate script -> {pinned file -> {ledger entry}}`` over the three rebind ledgers."""
    out: Dict[str, Dict[str, Set[str]]] = {}
    for table, rel in ic.REBIND_LEDGERS.items():
        path = Path(repo) / rel
        if not path.is_file():
            continue
        ledger = json.loads(path.read_text(encoding="utf-8"))
        for key, entry in ledger.items():
            try:
                scripts = ic.weld_gate_scripts(table, key, entry)
            except ValueError:
                continue
            for script in scripts:
                pins = out.setdefault(script, {})
                for name in (entry.get("source_sha256") or {}):
                    pins.setdefault(name, set()).add(f"{table}:{key}")
    return out


def unit_closure(unit: Mapping[str, Any], resolver, pins: Mapping[str, Mapping[str, Set[str]]],
                 prior: Optional[Mapping[str, Any]] = None) -> Dict[str, Set[str]]:
    """``file -> why the unit's evidence depends on it``."""
    out: Dict[str, Set[str]] = {}
    for script in unit["scripts"]:
        if resolver.exists(script):
            for name in resolver.run_closure(script):
                out.setdefault(name, set()).add(f"run closure of {script}")
        else:
            out.setdefault(script, set()).add("a script the unit runs, absent from the tip")
        for name, entries in (pins.get(script) or {}).items():
            out.setdefault(name, set()).update(f"pin of {e}" for e in entries)
    for name in unit.get("extra", ()):
        out.setdefault(name, set()).add("named by the unit's arguments")
    for name in ((prior or {}).get(unit["id"]) or {}).get("files", ()):
        out.setdefault(name, set()).add("recorded by the previous round's artifact")
    return out


def matches(name: str, patterns: Sequence[str]) -> bool:
    for pattern in patterns:
        if pattern == name or (pattern.endswith("/") and name.startswith(pattern)):
            return True
        if any(c in pattern for c in "*?[") and fnmatch.fnmatchcase(name, pattern):
            return True
    return False


def changed_since(repo: Path, commit: str) -> List[str]:
    done = subprocess.run(["git", "-C", str(repo), "diff", "--name-only", commit, "HEAD"],
                          capture_output=True, text=True)
    if done.returncode != 0:
        raise SystemExit(f"git diff {commit} HEAD failed in {repo}: {done.stderr.strip()}")
    return sorted(line for line in done.stdout.splitlines() if line.strip())


def _closure_tools(repo: Path, prior: Optional[Mapping[str, Any]],
                   overrides: Optional[Mapping[str, str]] = None):
    """``(resolver, pins, gap)`` of the tip; refuses (SystemExit) a tip whose static reader
    misses ``_LAZY`` reaches when no previous round's records are given."""
    ic = load_import_closure(repo)
    resolver = ic.Resolver(Path(repo), overrides=overrides)
    gap = lazy_gap_open(resolver)
    if gap and not prior:
        raise SystemExit(
            f"the tip's import_closure.py does not follow the package root's _LAZY table "
            f"({LAZY_PROBE[0]} does not reach {LAZY_PROBE[1]} statically), so a static "
            f"closure under-counts; pass --prior-records (chunks.py records over the previous "
            f"round's results)")
    return resolver, ledger_pins(repo, ic), gap


def chunk_closure(chunk_units: Iterable[str], units: Mapping[str, Mapping[str, Any]], resolver,
                  pins, prior: Optional[Mapping[str, Any]]) -> Tuple[Dict[str, Set[str]], List[str]]:
    """``(file -> why, units the tip lacks)`` over a chunk's units."""
    closure: Dict[str, Set[str]] = {}
    chunk_units = list(chunk_units)
    absent = [u for u in chunk_units if u not in units]
    for uid in chunk_units:
        if uid in units:
            for name, why in unit_closure(units[uid], resolver, pins, prior).items():
                closure.setdefault(name, set()).update(f"{uid}: {w}" for w in why)
    return closure, absent


def readiness(repo: Path, index: Mapping[str, Any], changing: Sequence[str],
              prior: Optional[Mapping[str, Any]] = None, *, say=print,
              overrides: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
    """Per current chunk: READY (no changing file in its closure) or NOT READY (which files,
    and why each is in the closure). Refuses (SystemExit) on a tip whose static reader misses
    ``_LAZY`` reaches when no previous round's records are given. ``overrides`` (repo path ->
    source text read instead of the file) plants controls, as ``import_closure`` does."""
    resolver, pins, gap = _closure_tools(repo, prior, overrides)
    units = {u["id"]: u for u in units_for(repo, index["machine"])}
    report: Dict[str, Any] = {"round": index["round"], "machine": index["machine"],
                              "device": index["device"], "lazy_gap_open": gap,
                              "changing": list(changing),
                              "prior_round": (prior or {}).get("_stamp"), "chunks": {}}
    for chunk in index["chunks"]:
        closure, absent = chunk_closure(chunk["units"], units, resolver, pins, prior)
        hits = {name: sorted(why) for name, why in closure.items() if matches(name, changing)}
        state = "READY" if not hits and not absent else "NOT READY"
        report["chunks"][chunk["id"]] = {
            "stamp": chunk["stamp"], "state": state, "expect_s": chunk["expect_s"],
            "units": len(chunk["units"]), "closure_files": len(closure),
            "hits": hits, "units_absent_from_tip": absent}
        detail = "" if state == "READY" else (
            "; ".join(f"{name} ({why[0]}{', ...' if len(why) > 1 else ''})"
                      for name, why in sorted(hits.items())[:4])
            + (f" and {len(hits) - 4} more" if len(hits) > 4 else "")
            + (f"; units absent from the tip {absent[:3]}" if absent else ""))
        say(f"{state:9s} {chunk['id']:4s} {chunk['stamp']}  {chunk['expect_s']} s  "
            f"{len(chunk['units'])} units  closure {len(closure)} files"
            + (f"  hits: {detail}" if detail else ""))
    ready = [cid for cid, row in report["chunks"].items() if row["state"] == "READY"]
    report["ready"] = ready
    say(f"READY_CHUNKS {','.join(ready) if ready else '-'} ({len(ready)} of "
        f"{len(report['chunks'])} chunks)")
    return report


def staleness(repo: Path, index: Mapping[str, Any], results: Path,
              prior: Optional[Mapping[str, Any]] = None, *, say=print) -> Dict[str, Any]:
    """Per allocated chunk, against the tip at ``repo``: NOT LAUNCHED (no launch record under
    ``results``), UNKNOWN (its launch tip is not in this clone, or its record is not the
    chunk's), or the units of its launch record whose closure moved between its launch tip
    and HEAD: STALE (some; run them again alone with ``chunks.py add --units``) or CURRENT.

    It reads checkout files only: an environment change reads 0 stale here."""
    resolver, pins, gap = _closure_tools(repo, prior)
    units = {u["id"]: u for u in units_for(repo, index["machine"])}
    report: Dict[str, Any] = {"round": index["round"], "machine": index["machine"],
                              "device": index["device"], "lazy_gap_open": gap, "chunks": {}}
    stale_units: Set[str] = set()
    for chunk in allocated(index):
        launch = read_launch_record(results, chunk["stamp"])
        row: Dict[str, Any] = {"stamp": chunk["stamp"]}
        if launch is None:
            row["state"] = "NOT LAUNCHED"
        elif launch_problems(launch, index, chunk):
            row.update(state="UNKNOWN", why=launch_problems(launch, index, chunk))
        elif _git(repo, "cat-file", "-e", f"{launch.get('tip')}^{{commit}}") is None:
            row.update(state="UNKNOWN", tip=launch.get("tip"))
        else:
            moved = set(changed_since(repo, launch["tip"]))
            hits: Dict[str, List[str]] = {}
            for uid in launch["units"]:
                closure, absent = chunk_closure([uid], units, resolver, pins, prior)
                files = sorted(name for name in closure if name in moved)
                if files or absent:
                    hits[uid] = files or ["the unit is absent from the tip"]
            stale_units.update(hits)
            row.update(state="STALE" if hits else "CURRENT", tip=launch["tip"],
                       moved_since_launch=len(moved), stale_units=hits)
        report["chunks"][chunk["id"]] = row
        say(f"{row['state']:12s} {chunk['id']:4s} {chunk['stamp']}"
            + (f"  launched on {str(row.get('tip'))[:12]}" if row.get("tip") else "")
            + (f"  stale units: {sorted(row['stale_units'])[:4]}" if row.get("stale_units")
               else ""))
    stale = [cid for cid, row in report["chunks"].items() if row["state"] == "STALE"]
    report["stale"] = stale
    report["stale_units"] = sorted(stale_units)
    say(f"STALE_CHUNKS {','.join(stale) if stale else '-'} ({len(stale)} of "
        f"{len(report['chunks'])} chunks); STALE_UNITS {len(stale_units)}"
        + (f" (chunks.py add --units {','.join(sorted(stale_units))})" if stale_units else ""))
    return report


# -------------------------------------------------------------------- artifacts --------
def _roots(results: Path, name: str) -> List[Path]:
    """``name`` and its resume continuations, newest first (``legs._roots_newest_first``)."""
    return [Path(results) / rel[len("results/"):] for rel in L._roots_newest_first(Path(results),
                                                                                   name)]


def _policy_suffix(policy: str) -> str:
    return "" if policy == "keep" else "_flush"


def _shard_tail(shard: str) -> str:
    return "" if shard == "fleet" else f"_{shard}"


def unit_dirs(unit: Mapping[str, Any], stamp: str, results: Path) -> List[Path]:
    """The directories that hold ``unit``'s artifacts under ``stamp`` (existing ones only).

    The layout is the round's own, so the same function finds a chunk's artifacts under its
    chunk stamp and a previous unchunked round's under that round's stamp.
    """
    results = Path(results)
    kind = unit["kind"]
    out: List[Path] = []
    if kind == "cuda_subdir":
        for campaign in sorted(results.glob(f"cuda_regate_{stamp}_g[0-9]*")):
            out.append(campaign / unit["subdir"])
    elif kind == "cuda_ade":
        out.append(results / f"cuda_ade_recut_{stamp}")
    elif kind == "triton_fleet":
        tail = _shard_tail(unit["shard"])
        for suffix in ("", "_flush"):
            for root in _roots(results, f"triton_fleet_{stamp}{suffix}{tail}"):
                out.append(root / unit["gate"])
    elif kind == "triton_composition":
        for suffix in ("", "_flush"):
            for root in _roots(results, f"triton_composition_{stamp}{suffix}"):
                out += [root / gate for gate in unit["gates"]]
    elif kind == "triton_stencil":
        for suffix in ("", "_flush"):
            out.append(results / f"triton_offdiag_stencil_welds_{stamp}{suffix}" / unit["family"])
    elif kind == "triton_bitid":
        for suffix in ("", "_flush"):
            out.append(results / f"triton_bit_identity_{stamp}{suffix}")
    elif kind == "triton_hd":
        out.append(results / f"triton_{unit['gate']}_{stamp}")
    elif kind == "metal_gate":
        out.append(results / f"{unit['family']}_{stamp}")
    else:
        raise ValueError(f"unknown unit kind {kind!r}")
    return [d for d in out if d.is_dir() and not d.is_symlink()]


def _first_holding(roots: Sequence[Path], name: str) -> Path:
    """``root/name`` in the first root (newest continuation first) whose ``name`` directory
    holds a JSON file, as the bind tools pick it; the first root's when none does."""
    for root in roots:
        candidate = root / name
        if candidate.is_dir() and not candidate.is_symlink() and any(candidate.glob("*.json")):
            return candidate
    return roots[0] / name


def unit_verdict_paths(unit: Mapping[str, Any], stamp: str, results: Path
                       ) -> List[Tuple[str, Path]]:
    """``(leg, artifact or directory)`` per verdict the unit's legs write under ``stamp``: the
    artifacts the bind tools read a released verdict from (``round_status.classify``)."""
    results = Path(results)
    kind = unit["kind"]
    out: List[Tuple[str, Path]] = []
    if kind == "cuda_subdir":
        campaigns = sorted(results.glob(f"cuda_regate_{stamp}_g[0-9]*")) or \
            [results / f"cuda_regate_{stamp}_g0"]
        campaign = next((c for c in campaigns if (c / unit["subdir"]).is_dir()), campaigns[0])
        for leg in L.cuda_legs(Path("/repo"), results, stamp):
            if leg["subdir"] == unit["subdir"]:
                out.append((leg["leg"], campaign / leg["subdir"] / leg["leg"]))
    elif kind == "cuda_ade":
        for policy in unit["policies"]:
            out.append((f"{policy}/gate.json",
                        results / f"cuda_ade_recut_{stamp}" / policy / "gate.json"))
    elif kind == "triton_fleet":
        for policy in unit["policies"]:
            roots = _roots(results, f"triton_fleet_{stamp}{_policy_suffix(policy)}"
                                    f"{_shard_tail(unit['shard'])}")
            out.append((f"{policy}/{unit['gate']}", _first_holding(roots, unit["gate"])))
    elif kind == "triton_composition":
        for policy in unit["policies"]:
            roots = _roots(results, f"triton_composition_{stamp}{_policy_suffix(policy)}")
            out += [(f"{policy}/{gate}", _first_holding(roots, gate)) for gate in unit["gates"]]
    elif kind == "triton_stencil":
        for policy in unit["policies"]:
            out.append((f"{policy}/{unit['family']}", results / (
                f"triton_offdiag_stencil_welds_{stamp}{_policy_suffix(policy)}") / unit["family"]))
    elif kind == "triton_bitid":
        for policy in unit["policies"]:
            out.append((policy, results / f"triton_bit_identity_{stamp}{_policy_suffix(policy)}"))
    elif kind == "triton_hd":
        for policy in unit["policies"]:
            out.append((policy, results / f"triton_{unit['gate']}_{stamp}" / policy / "gate.json"))
    elif kind == "metal_gate":
        out.append(("flush", results / f"{unit['family']}_{stamp}"))
    else:
        raise ValueError(f"unknown unit kind {kind!r}")
    return out


def _rel(path: Path, results: Path) -> str:
    """``path`` relative to ``results`` where it lies under it, else as given."""
    try:
        return Path(path).relative_to(Path(results)).as_posix()
    except ValueError:
        return str(path)


def json_files(directory: Path) -> List[Path]:
    """Every ``*.json`` under ``directory``, never following a symlink."""
    out: List[Path] = []
    stack = [Path(directory)]
    while stack:
        current = stack.pop()
        try:
            entries = list(os.scandir(current))
        except OSError:
            continue
        for entry in entries:
            if entry.is_symlink():
                continue
            if entry.is_dir(follow_symlinks=False):
                stack.append(Path(entry.path))
            elif entry.name.endswith(".json") and entry.is_file(follow_symlinks=False):
                out.append(Path(entry.path))
    return sorted(out)


def unit_records(unit: Mapping[str, Any], stamp: str, results: Path
                 ) -> List[Tuple[Path, Dict[str, Any]]]:
    """``(artifact, payload)`` for every JSON of the unit under ``stamp`` carrying a record."""
    out = []
    for directory in unit_dirs(unit, stamp, results):
        for path in json_files(directory):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if isinstance(payload, dict) and artifact_drift.records(payload):
                out.append((path, payload))
    return out


def prior_records(repo: Path, results: Path, stamp: str, machine: str, *, say=print
                  ) -> Dict[str, Any]:
    """``unit id -> {"files": [...], "artifacts": N}`` from a previous round's artifacts."""
    out: Dict[str, Any] = {"_stamp": stamp, "_machine": machine}
    units = units_for(repo, machine)
    started = time.time()
    for i, unit in enumerate(units, 1):
        files: Set[str] = set()
        found = unit_records(unit, stamp, results)
        for _path, payload in found:
            for record in artifact_drift.records(payload).values():
                files.update(record)
        out[unit["id"]] = {"files": sorted(files), "artifacts": len(found)}
        say(f"records {i}/{len(units)} {unit['id']}: {len(found)} artifact(s), "
            f"{len(files)} file(s) ({time.time() - started:.1f} s)")
    return out


# ------------------------------------------------------------------------ verify -------
def stamped_device(machine: str, payload: Mapping[str, Any]) -> Optional[str]:
    """The device tag an artifact stamps itself, or None where it stamps none readable: an
    NVIDIA artifact's ``stamped_device.compute_capability``; a Metal artifact's
    ``environment.apple_gpu.architecture`` (``metal_runs.artifact_architecture``; some
    Metal gates record it, and every Metal launch is judged by its environment record too,
    :func:`launch_device`)."""
    if machine == "metal":
        environment = payload.get("environment")
        apple = environment.get("apple_gpu") if isinstance(environment, dict) else None
        value = apple.get("architecture") if isinstance(apple, dict) else None
        return value if isinstance(value, str) and value else None
    if machine != "nvidia":
        return None
    block = payload.get("stamped_device")
    return _nvidia_device(block.get("compute_capability") if isinstance(block, dict) else None)


def _nvidia_device(value: Any) -> Optional[str]:
    """``cc86`` from a compute capability as a run records it (``8.6``, ``86``), None for
    none, and ``unreadable:<value>`` for one that is not a capability."""
    if value is None:
        return None
    text = str(value).strip()
    if text.isdigit() and len(text) >= 2:
        text = f"{text[:-1]}.{text[-1]}"
    try:
        return device_of_capability(text)
    except ValueError:
        return f"unreadable:{value!r}"


#: The Triton campaign's own device record: ``drive_triton_weld_gates.py`` writes it into each
#: campaign root, and the Triton rebinds read it (``rebind_triton_welds.DEVICE_STAMP``) in the
#: family directory first and the campaign root second. The CUDA rebind never reads one.
TRITON_DEVICE_RECORD = "device.json"


def campaign_devices(unit: Mapping[str, Any], stamp: str, results: Path) -> List[str]:
    """Every device the Triton device records beside ``unit``'s directories under ``stamp``
    name (each directory's own, and its campaign root's), ``unreadable:<path>`` for a record
    that cannot be read, as the Triton rebinds refuse it; empty for a CUDA or Metal unit,
    and where no record is written."""
    if not str(unit.get("kind", "")).startswith("triton_"):
        return []
    results = Path(results)
    found: Set[str] = set()
    for directory in unit_dirs(unit, stamp, results):
        for place in (directory, directory.parent):
            path = place / TRITON_DEVICE_RECORD
            if place == results or not path.is_file():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                found.add(f"unreadable:{_rel(path, results)}")
                continue
            if not isinstance(payload, dict):
                found.add(f"unreadable:{_rel(path, results)}")
                continue
            device = _nvidia_device(payload.get("compute_capability"))
            if device:
                found.add(device)
    return sorted(found)


def launch_device(machine: str, results: Path, stamp: str) -> Optional[str]:
    """The GPU architecture a Metal launch's environment records
    (``metal_environment_<stamp>/start.json``, the record ``rebind_metal_welds.py`` reads),
    or None where it records none."""
    if machine != "metal":
        return None
    path = Path(results) / f"metal_environment_{stamp}" / "start.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8")).get("architecture")
    except (OSError, ValueError, AttributeError):
        return None
    return value if isinstance(value, str) else None


def launches(index: Mapping[str, Any], results: Path) -> List[Dict[str, Any]]:
    """Every launch of every chunk the index ever allocated, oldest first (launch time, then
    chunk number): ``{"chunk", "stamp", "k", "record", "problems"}``."""
    out = []
    for chunk in allocated(index):
        record = read_launch_record(results, chunk["stamp"])
        if record is None:
            continue
        out.append({"chunk": chunk["id"], "stamp": chunk["stamp"], "k": _chunk_k(chunk["id"]),
                    "record": record, "utc": str(record.get("utc") or ""),
                    "problems": launch_problems(record, index, chunk)})
    return sorted(out, key=lambda launch: (launch["utc"], launch["k"]))


def run_devices(machine: str, found: Sequence[Tuple[Path, Mapping[str, Any]]], results: Path,
                stamp: str, launch_devices=None, unit: Optional[Mapping[str, Any]] = None
                ) -> Tuple[List[str], str]:
    """``(devices, where they were read)`` for one run of one unit; ``devices`` is empty
    when nothing says what it ran on, and such a run is never taken.

    Metal: the launch's environment record (:func:`launch_device`, the record
    ``rebind_metal_welds.py`` keys the run by, and without which it binds nothing), with
    every architecture the unit's own artifacts stamp beside it. NVIDIA: the compute
    capability the unit's artifacts stamp (``stamped_device``), with every one a Triton
    unit's campaign device record names beside it (:func:`campaign_devices`, the record the
    Triton rebinds read; it judges the units whose artifacts stamp none, 2 of 123 at the 1.0
    freeze: ``triton/fleet/cylindrical_complex_composition`` and
    ``triton/fleet/source_seams``, also when one runs alone); a unit with neither is judged
    by what every other artifact of the same launch stamps (``launch_devices``: one launch is
    one host's lanes), and a launch whose artifacts stamp nothing is refused."""
    own = sorted({d for d in (stamped_device(machine, p) for _a, p in found) if d})
    if machine == "metal":
        architecture = launch_device(machine, results, stamp)
        if architecture is None:
            return [], f"metal_environment_{stamp}/start.json (absent or names no architecture)"
        return sorted(set(own) | {architecture}), (
            f"metal_environment_{stamp}/start.json" + (" and the artifacts" if own else ""))
    campaign = campaign_devices(unit, stamp, results) if unit is not None else []
    if own or campaign:
        return sorted(set(own) | set(campaign)), " and ".join(
            where for where, read in (("the artifacts' stamped_device", own),
                                      (f"the campaign's {TRITON_DEVICE_RECORD}", campaign))
            if read)
    seen = sorted(launch_devices(stamp)) if launch_devices is not None else []
    return seen, (f"the other artifacts of launch {stamp} (this unit's stamp no device)"
                  if seen else f"no artifact of launch {stamp} stamps a device")


def judge_candidate(unit: Mapping[str, Any], launch: Mapping[str, Any], repo: Path,
                    results: Path, index: Mapping[str, Any], cache: Dict[str, Optional[str]],
                    closure_of, diff_of, launch_devices=None) -> Dict[str, Any]:
    """One run of one unit, judged on its own: HOLDS, REFUSED, NOT RELEASED or EMPTY.

    ``tool`` says whether a bind tool that applies only the import-record rule and the
    released verdict would bind this run (what :func:`bind_commands` guards against).
    ``launch_devices(stamp)`` is the set of devices the launch's artifacts stamp
    (:func:`run_devices`)."""
    stamp = launch["stamp"]
    row: Dict[str, Any] = {"stamp": stamp, "chunk": launch["chunk"], "utc": launch["utc"]}
    if launch["problems"]:
        return dict(row, state="REFUSED", tool=False, artifacts=0,
                    why="; ".join(launch["problems"]))
    found = unit_records(unit, stamp, results)
    present = bool(found) or any(json_files(d) for d in unit_dirs(unit, stamp, results))
    bad = []
    for path, payload in found:
        moved = artifact_drift.artifact_moved(payload, repo, cache)
        if moved:
            bad.append({"artifact": str(path.relative_to(results)), "moved": moved})
    verdicts = []
    for leg, path in unit_verdict_paths(unit, stamp, results):
        verdict = RS.classify(path)
        verdicts.append({"leg": leg, "state": verdict["state"],
                         "artifact": _rel(Path(verdict.get("artifact") or path), results)})
    failing = [v for v in verdicts if not RS.is_pass(v["state"])]
    row.update(artifacts=len(found), verdicts=verdicts,
               artifact_sha256={_rel(path, results): hashlib.sha256(
                   path.read_bytes()).hexdigest() for path, _payload in found})
    # What a bind tool that applies only the import-record rule and the released verdict
    # would make of this run: it binds it, whatever this module judges beyond those two.
    row["tool"] = present and not bad and not failing
    if not present:
        return dict(row, state="EMPTY", tool=False, why=f"no artifact under {stamp}")
    if bad:
        return dict(row, state="REFUSED", why="; ".join(
            artifact_drift.refusal(b["artifact"], b["moved"], repo) for b in bad[:2])
            + (f"; and {len(bad) - 2} more artifact(s)" if len(bad) > 2 else ""))
    # THE DEVICE IS READ, NEVER ASSUMED: a run that says nothing about what it ran on is
    # not taken, whatever its stamp's name says.
    devices, device_from = run_devices(index["machine"], found, results, stamp, launch_devices,
                                       unit)
    row.update(devices=devices, device_from=device_from)
    if not devices:
        if index["machine"] == "metal":
            row["tool"] = False                  # rebind_metal_welds.py refuses it too
        return dict(row, state="REFUSED", why=(
            f"the device it ran on cannot be read ({device_from}), and the index certifies "
            f"{index['device']}"))
    foreign = [d for d in devices if d != index["device"]]
    if foreign:
        return dict(row, state="REFUSED", why=f"ran on {foreign} ({device_from}), and the "
                                              f"index certifies {index['device']}")
    whole = any("imported_source_sha256" in artifact_drift.records(payload)
                for _path, payload in found)
    if not whole:
        kind = ("carry no import record" if not found else
                "carry no whole import record (a curated source_sha256 only)")
        diff = diff_of(launch)
        if diff is None:
            tip = launch["record"].get("tip")
            return dict(row, state="REFUSED", why=(
                f"its artifacts under {stamp} {kind}, and the launch tip {str(tip)[:12]} is not "
                f"in this clone, so nothing says the bytes it ran are the release tree's"))
        hits = sorted(name for name in closure_of(unit) if name in set(diff))
        if hits:
            return dict(row, state="REFUSED", why=(
                f"its artifacts {kind}, and {len(hits)} file(s) of its closure moved since the "
                f"launch tip {launch['record']['tip'][:12]}: {hits[:4]}"))
    if failing:
        return dict(row, state="NOT RELEASED", why="; ".join(
            f"{v['leg']}: {v['state']}" for v in failing[:3])
            + (f"; and {len(failing) - 3} more" if len(failing) > 3 else ""))
    return dict(row, state="HOLDS", why=f"{len(found)} artifact(s) hold, {len(verdicts)} "
                                        f"verdict(s) pass")


def verify(repo: Path, results: Path, index: Mapping[str, Any], *, say=print
           ) -> Dict[str, Any]:
    """Every unit of the release tree at ``repo``, judged by its own runs (module docstring).

    Returns ``{"ok", "counts", "units", "launches", "orphans", "artifacts"}``: per unit its
    state, its candidates and, when OK, the selected stamp (:func:`selection_of` turns it into
    the bind's record). ``ok`` is True only when every unit of the release tree is OK.
    """
    machine = index["machine"]
    units = {u["id"]: u for u in units_for(repo, machine)}
    cache: Dict[str, Optional[str]] = {}
    tools: Dict[str, Any] = {}
    diffs: Dict[str, Optional[List[str]]] = {}

    def closure_of(unit: Mapping[str, Any]) -> Dict[str, Set[str]]:
        if not tools:
            ic = load_import_closure(repo)
            tools.update(resolver=ic.Resolver(Path(repo)), pins=ledger_pins(repo, ic))
        return unit_closure(unit, tools["resolver"], tools["pins"])

    def diff_of(launch: Mapping[str, Any]) -> Optional[List[str]]:
        tip = launch["record"].get("tip")
        if tip not in diffs:
            known = tip and _git(repo, "cat-file", "-e", f"{tip}^{{commit}}") is not None
            diffs[tip] = changed_since(repo, tip) if known else None
        return diffs[tip]

    stamped: Dict[str, Set[str]] = {}

    def launch_devices(stamp: str) -> Set[str]:
        """Every device the artifacts of the launch of ``stamp`` stamp, over its units."""
        if stamp not in stamped:
            record = next((run["record"] for run in runs if run["stamp"] == stamp), {})
            stamped[stamp] = {d for uid in record.get("units") or () if uid in units
                              for _p, payload in unit_records(units[uid], stamp, results)
                              for d in [stamped_device(machine, payload)] if d}
        return stamped[stamp]

    runs = launches(index, results)
    for launch in runs:
        if launch["problems"]:
            say(f"LAUNCH   {launch['chunk']} {launch['stamp']}: refused: "
                f"{'; '.join(launch['problems'])}")
    ran: Dict[str, List[Mapping[str, Any]]] = {}
    orphans: Dict[str, List[str]] = {}
    for launch in runs:
        for uid in launch["record"].get("units") or ():
            if uid in units:
                ran.setdefault(uid, []).append(launch)
            else:
                orphans.setdefault(uid, []).append(launch["stamp"])
    rows: Dict[str, Any] = {}
    counts = {state: 0 for state in UNIT_STATES}
    started = time.time()
    for i, (uid, unit) in enumerate(sorted(units.items()), 1):
        candidates = [judge_candidate(unit, launch, repo, results, index, cache, closure_of,
                                      diff_of, launch_devices) for launch in ran.get(uid, ())]
        holding = [c for c in candidates if c["state"] == "HOLDS"]
        if holding:
            chosen = holding[-1]
            state, why = "OK", f"{chosen['stamp']}: {chosen['why']}"
        elif not candidates:
            chosen, state, why = None, "NOT LAUNCHED", "in no launched chunk"
        elif all(c["state"] == "EMPTY" for c in candidates):
            chosen, state, why = None, "ABSENT", "; ".join(
                f"{c['stamp']}: {c['why']}" for c in candidates)
        else:
            chosen, state = None, "REFUSED"
            why = "; ".join(f"{c['stamp']}: {c['state']}: {c['why']}" for c in candidates)
        counts[state] += 1
        rows[uid] = {"state": state, "why": why, "candidates": candidates,
                     "selected": chosen["stamp"] if chosen else None,
                     "chunk": chosen["chunk"] if chosen else None}
        if state != "OK":
            say(f"{state:12s} {uid}: {why}")
        if i % 25 == 0:
            say(f"verify {i}/{len(units)} units ({time.time() - started:.1f} s)")
    total = len(units)
    artifacts = sum(c["artifacts"] for row in rows.values() for c in row["candidates"])
    say(f"VERIFY {'OK' if counts['OK'] == total else 'REFUSED'}: {counts['OK']} of {total} "
        f"units hold ({counts['REFUSED']} refused, {counts['ABSENT']} absent, "
        f"{counts['NOT LAUNCHED']} not launched); {len(runs)} launch(es), {artifacts} "
        f"artifact(s) read"
        + (f"; {len(orphans)} launched unit(s) the release tree does not have" if orphans
           else ""))
    return {"ok": counts["OK"] == total, "counts": counts, "units": rows,
            "launches": [{k: v for k, v in launch.items() if k != "record"}
                         | {"units": len(launch["record"].get("units") or ())}
                         for launch in runs],
            "orphans": orphans, "artifacts": artifacts}


SELECTION_SCHEMA = "meep_gpu.cert_chunk_selection/1"


def selection_of(report: Mapping[str, Any], index: Mapping[str, Any], repo: Path
                 ) -> Dict[str, Any]:
    """The bind's record of the choice: per OK unit its stamp and the bytes it binds, and
    per unit every candidate with its state and why."""
    return {"schema": SELECTION_SCHEMA, "machine": index["machine"], "device": index["device"],
            "device_mode": index["device_mode"], "round": index["round"],
            "release_tip": _git(repo, "rev-parse", "HEAD"),
            "release_tree": _git(repo, "rev-parse", "HEAD^{tree}"), "utc": _utc(),
            "counts": report["counts"],
            "launch_order": [launch["stamp"] for launch in report["launches"]],
            "selected": {uid: row["selected"] for uid, row in report["units"].items()
                         if row["state"] == "OK"},
            "units": report["units"]}


def selection_moved(recorded: Mapping[str, Any], now: Mapping[str, Any]) -> List[str]:
    """Why a selection judged again (``now``) is not the one ``recorded``: another release
    tree, another unit selected or a unit's selected stamp moved, or other bytes in a
    selected run's artifacts. Empty when the bind printed from ``recorded`` binds ``now``."""
    problems = []
    for key in ("release_tree", "device", "device_mode", "round"):
        if recorded.get(key) != now.get(key):
            problems.append(f"{key} was {recorded.get(key)!r} and is {now.get(key)!r}")
    before, after = dict(recorded.get("selected") or {}), dict(now.get("selected") or {})
    for uid in sorted(set(before) | set(after)):
        if before.get(uid) != after.get(uid):
            problems.append(f"{uid} was selected under {before.get(uid)} and is under "
                            f"{after.get(uid)}")
            continue
        old = _candidate(recorded, uid, before[uid]) or {}
        new = _candidate(now, uid, after[uid]) or {}
        if old.get("artifact_sha256") != new.get("artifact_sha256"):
            problems.append(f"{uid}: the artifacts of its selected run {after[uid]} are not "
                            f"the bytes the selection recorded")
    return problems


def read_selection(path: Path, index: Mapping[str, Any]) -> Dict[str, Any]:
    selection = json.loads(Path(path).read_text(encoding="utf-8"))
    if selection.get("schema") != SELECTION_SCHEMA:
        raise SystemExit(f"{path} is not a selection ({SELECTION_SCHEMA})")
    for key in ("machine", "device", "round"):
        if selection.get(key) != index[key]:
            raise SystemExit(f"{path} was cut for {key} {selection.get(key)!r}; the index is "
                             f"{index[key]!r}")
    return selection


# ------------------------------------------------------------------------- binds -------
def bind_plan_last(write: bool) -> str:
    """The last command of the unchunked bind (``legs.bind_plan``), which the chunked
    bind ends with too; read from the plan itself, so the two cannot drift apart."""
    return L.bind_plan(Path("results"), "x", 0, write=write)[-1]


def _literal_dict(path: Path, name: str) -> Dict[str, Any]:
    """A module-scope dict literal of the tip's harness file, read, never imported."""
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        target = node.target if isinstance(node, ast.AnnAssign) else (
            node.targets[0] if isinstance(node, ast.Assign) and len(node.targets) == 1
            else None)
        if isinstance(target, ast.Name) and target.id == name:
            return ast.literal_eval(node.value)
    raise SystemExit(f"{path.name} assigns no {name}")


def triton_ledger_keys(repo: Path) -> Tuple[str, Set[str]]:
    """``(path, keys)`` of the checkout's Triton ledger, the file ``rebind_triton_welds.py``
    binds: the path as the checkout's ``import_closure.REBIND_LEDGERS`` names it, the file
    read by path, never imported. The keys are every top-level key, the set the tool's
    ``--only`` guard checks a key list against."""
    harness = Path(repo) / "parity" / "meep_gpu"
    rel = _literal_dict(harness / "import_closure.py", "REBIND_LEDGERS")["triton"]
    path = Path(repo) / rel
    if not path.is_file():
        raise SystemExit(f"the checkout has no Triton ledger at {rel}: no rebind_triton_welds.py "
                         f"--only can be printed against it")
    return rel, set(json.loads(path.read_text(encoding="utf-8")))


def _ordered(selection: Mapping[str, Any], stamps: Iterable[str]) -> List[str]:
    order = {stamp: i for i, stamp in enumerate(selection.get("launch_order") or ())}
    return sorted(set(stamps), key=lambda s: (order.get(s, len(order)), s))


def _candidate(selection: Mapping[str, Any], uid: str, stamp: str) -> Optional[Mapping[str, Any]]:
    for candidate in (selection["units"].get(uid) or {}).get("candidates") or ():
        if candidate["stamp"] == stamp:
            return candidate
    return None


def _later(selection: Mapping[str, Any], a: str, b: Optional[str]) -> bool:
    """True when stamp ``a`` launched after ``b`` (or ``b`` is None)."""
    if b is None:
        return True
    order = list(selection.get("launch_order") or ())
    return (order.index(a) if a in order else len(order)) > (
        order.index(b) if b in order else len(order))


def _set_aside(stamp: str, rel: str) -> str:
    """The command that moves ``results/<rel>`` out of every bind tool's reach, keeping its
    path under ``results/set_aside_<stamp>/`` (nothing is deleted; ``mv`` refuses a target
    that already holds a directory of that name)."""
    parent = Path(rel).parent.as_posix()
    target = f"results/set_aside_{stamp}" + ("" if parent == "." else f"/{parent}")
    return f"mkdir -p {target} && mv results/{rel} {target}/"


def stamp_tool_conflicts(selection: Mapping[str, Any], stamps: Sequence[str],
                         kinds: Sequence[str], units: Mapping[str, Mapping[str, Any]],
                         results: Path, tool: str) -> List[str]:
    """STOP lines for a tool that binds whatever a stamp holds, called once per stamp in
    launch order: a run under a passed stamp that the tool would bind, and that is not the
    selected run of its unit while launching after it (or its unit has none), would replace
    the selection."""
    lines = []
    for stamp in stamps:
        for uid, unit in sorted(units.items()):
            if unit["kind"] not in kinds:
                continue
            candidate = _candidate(selection, uid, stamp)
            chosen = selection["selected"].get(uid)
            if candidate is None and chosen != stamp:
                # A directory of the unit under a stamp whose launch record does not list
                # it is a run of no launch: verify never judged it, and the tool binds it.
                for directory in unit_dirs(unit, stamp, results):
                    rel = directory.relative_to(results).as_posix()
                    lines.append(f"# STOP {tool}: results/{rel} is {uid} under {stamp}, whose "
                                 f"launch record does not list it, so no verify judged it; "
                                 f"set it aside first: {_set_aside(stamp, rel)}")
                continue
            if candidate is None or chosen == stamp or not candidate.get("tool"):
                continue
            if _later(selection, stamp, chosen):
                for directory in unit_dirs(unit, stamp, results):
                    rel = directory.relative_to(results).as_posix()
                    lines.append(f"# STOP {tool}: {uid} under {stamp} would replace "
                                 f"{'its selected run ' + chosen if chosen else 'nothing'} "
                                 f"({candidate['state']}: {candidate['why']}); set it aside "
                                 f"first: {_set_aside(stamp, rel)}")
    return lines


def nvidia_bind_commands(index: Mapping[str, Any], results: Path, index_path: str,
                         selection: Mapping[str, Any], selection_path: str,
                         repo: Optional[Path] = None, write: bool = False) -> List[str]:
    """The NVIDIA bind of exactly ``selection``, in ``legs.bind_plan``'s order.

    First the verify that wrote the selection (re-runnable; a different answer means the
    results moved). Then the CUDA campaigns the selection draws from, recorded and rebound
    one at a time in launch order, and the ADE re-cut from its selected stamp; the Triton
    rebind once per selected stamp with ``--only`` the keys of the units selected there that
    the Triton ledger holds (a selected unit whose key it lacks named on a REFUSED line); the
    bit identity and the composition from their selected stamps; the family
    re-certification over the selected fleet roots in an order that makes each family's
    selected root the first holding a released artifact; and last the label pass.
    """
    repo = Path(repo) if repo is not None else HARNESS.parents[1]
    w = " --write" if write else ""
    units = {u["id"]: u for u in units_for(repo, "nvidia")}
    chosen: Dict[str, str] = dict(selection["selected"])
    out = [f"python rounds/1.0/chunks.py verify --index {index_path} --repo ../.. "
           f"--results results --against {selection_path}"]
    # CUDA: whole campaigns, one per stamp the selection draws a subdirectory from.
    cuda_stamps = _ordered(selection, (s for uid, s in chosen.items()
                                       if units[uid]["kind"] == "cuda_subdir"))
    for stamp in cuda_stamps:
        for uid, unit in sorted(units.items()):
            if unit["kind"] != "cuda_subdir" or chosen.get(uid) == stamp:
                continue
            candidate = _candidate(selection, uid, stamp)
            if candidate is not None and candidate["state"] not in ("HOLDS",) and not \
                    candidate.get("tool"):
                for directory in unit_dirs(unit, stamp, results):
                    rel = directory.relative_to(results).as_posix()
                    out.append(f"# STOP record_cuda_regate.py: results/{rel} is "
                               f"{candidate['state']} ({candidate['why']}) and its campaign "
                               f"is recorded whole for the units selected there; set it "
                               f"aside first: {_set_aside(stamp, rel)}")
    out += stamp_tool_conflicts(selection, cuda_stamps, ("cuda_subdir",), units, results,
                                "rebind_cuda_welds.py")
    campaigns = {stamp: sorted(p.name for p in results.glob(f"cuda_regate_{stamp}_g[0-9]*"))
                 or [f"cuda_regate_{stamp}_g0"] for stamp in cuda_stamps}
    for stamp in cuda_stamps:
        out += [f"python record_cuda_regate.py --campaign results/{name}{w}"
                for name in campaigns[stamp]]
    for stamp in cuda_stamps:
        out += [f"python rebind_cuda_welds.py --campaign results/{name}{w}"
                for name in campaigns[stamp]]
    if "cuda/ade_recut" in chosen:
        out.append(f"python record_cuda_ade_recut.py --campaign "
                   f"results/cuda_ade_recut_{chosen['cuda/ade_recut']}{w}")
    # Triton: once per selected stamp, the keys of the units selected there that the
    # checkout's Triton ledger holds. The tool refuses a whole --only naming a key the ledger
    # lacks, before it binds anything, and CAMPAIGN_DIRS types rows ahead of the seed that
    # writes their keys (seed_triton_welds.py), so such a key is left out and its unit named.
    # The keys are read when the bind is printed and the lines run later, but no line printed
    # here adds a CAMPAIGN_DIRS key to the ledger or removes one: the rebind rewrites the
    # entries it binds, and the keys it retires (RETIRED_WELDS) are disjoint from
    # CAMPAIGN_DIRS by the tool's own assertion.
    campaign_dirs = _literal_dict(repo / "parity" / "meep_gpu" / "rebind_triton_welds.py",
                                  "CAMPAIGN_DIRS")
    by_dir: Dict[str, List[str]] = {}
    for key, directory in campaign_dirs.items():
        by_dir.setdefault(directory, []).append(key)
    ledger_rel, held = triton_ledger_keys(repo)
    welding = {uid: unit for uid, unit in units.items()
               if unit["kind"] in ("triton_fleet", "triton_stencil")}
    for stamp in _ordered(selection, (chosen[uid] for uid in welding if uid in chosen)):
        keep: List[str] = []
        flush: List[str] = []
        keys: List[str] = []
        shards = set()
        stencil = False
        for uid, unit in sorted(welding.items()):
            if chosen.get(uid) != stamp:
                continue
            name = unit.get("gate") or unit.get("family")
            lacking = [key for key in by_dir.get(name, []) if key not in held]
            if lacking:
                out.append(f"# REFUSED rebind_triton_welds.py --only: {uid}, selected under "
                           f"{stamp}, welds {', '.join(lacking)}, which the Triton ledger "
                           f"{ledger_rel} does not hold; the tool refuses an --only naming a "
                           f"key the ledger lacks, so "
                           f"{'it is' if len(lacking) == 1 else 'they are'} left out")
            keys += [key for key in by_dir.get(name, []) if key in held]
            if unit["kind"] == "triton_fleet":
                shards.add(unit["shard"])
            else:
                stencil = True
        if not keys:
            continue
        for shard in ["fleet", "env_probe", "offdiag_probe"] + [n for n, _g in L.TRITON_MOVABLE]:
            if shard in shards:
                keep += L._roots_newest_first(results, f"triton_fleet_{stamp}{_shard_tail(shard)}")
                flush += L._roots_newest_first(
                    results, f"triton_fleet_{stamp}_flush{_shard_tail(shard)}")
        if stencil:
            keep += L._roots_newest_first(results, f"triton_offdiag_stencil_welds_{stamp}")
            flush += L._roots_newest_first(results, f"triton_offdiag_stencil_welds_{stamp}_flush")
        out.append("python rebind_triton_welds.py " + " ".join(f"--campaign {r}" for r in keep)
                   + " " + " ".join(f"--flush-campaign {r}" for r in flush)
                   + f" --only {','.join(sorted(keys))}{w}")
    if "triton/bit_identity" in chosen:
        s = chosen["triton/bit_identity"]
        out.append(f"python rebind_triton_welds.py --bind-bit-identity "
                   f"results/triton_bit_identity_{s}/gate.json --bit-identity-flush "
                   f"results/triton_bit_identity_{s}_flush/gate.json{w}")
    if "triton/composition" in chosen:
        s = chosen["triton/composition"]
        out.append(f"python recut_composition_records.py --run results/triton_composition_{s} "
                   f"--flush-run results/triton_composition_{s}_flush "
                   f"--only {','.join(L.TRITON_COMPOSITION_KEYS)}{w}")
    families = _family_recert_families(repo)
    if families:
        out += family_recert_commands(selection, units, families, results, w)
    # LAST, as in ``legs.bind_plan``: the label leaves no re-bind above replaced get their
    # neutral labels, so the guard's ledger leg reads clean.
    out.append(bind_plan_last(write))
    return out


def _family_recert_families(repo: Path) -> Tuple[str, ...]:
    path = Path(repo) / "parity" / "meep_gpu" / "recut_composition_records.py"
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and getattr(node.targets[0], "id", None) == "FAMILY_RECERT_FAMILIES"):
            return tuple(ast.literal_eval(node.value))
    return ()


def family_recert_commands(selection: Mapping[str, Any], units: Mapping[str, Mapping[str, Any]],
                           families: Sequence[str], results: Path, w: str) -> List[str]:
    """``recut_composition_records.py --family-recert``: it takes, per family, the first root
    holding a released artifact, so the selected stamps' core fleet roots are ordered to make
    that root the selected one; a cycle is named on a STOP line."""
    chosen = selection["selected"]
    out: List[str] = []
    picked: Dict[str, str] = {}
    for family in families:
        uid = f"triton/fleet/{family}"
        if uid not in chosen:
            out.append(f"# STOP recut_composition_records.py --family-recert: {uid} has no "
                       f"selected run ({(selection['units'].get(uid) or {}).get('state')})")
            continue
        if units.get(uid, {}).get("shard") != "fleet":
            out.append(f"# STOP recut_composition_records.py --family-recert: {uid} is not a "
                       f"core fleet gate")
            continue
        picked[family] = chosen[uid]
    if out:
        return out
    stamps = _ordered(selection, picked.values())
    commands = {}
    for policy, suffix in (("keep", ""), ("flush", "_flush")):
        before: Dict[str, Set[str]] = {s: set() for s in stamps}
        for family, own in picked.items():
            for other in stamps:
                if other == own:
                    continue
                roots = _roots(results, f"triton_fleet_{other}{suffix}")
                path = _first_holding(roots, family)
                if path.is_dir() and RS.is_pass(RS.classify(path)["state"]):
                    before[own].add(other)          # ``own`` must come before ``other``
        order = _topological(stamps, before)
        if order is None:
            return [f"# STOP recut_composition_records.py --family-recert: no order of the "
                    f"selected {policy} fleet roots makes every family's selected root the "
                    f"first holding a released artifact ({picked}); set the older released "
                    f"runs aside (results/set_aside_<stamp>/) and print the bind again"]
        commands[policy] = [r for s in order
                            for r in L._roots_newest_first(results, f"triton_fleet_{s}{suffix}")]
    return [f"python recut_composition_records.py --family-recert "
            f"{' '.join(commands['keep'])} --flush-family-recert "
            f"{' '.join(commands['flush'])}{w}"]


def _topological(nodes: Sequence[str], before: Mapping[str, Set[str]]) -> Optional[List[str]]:
    """``nodes`` ordered so every ``a`` precedes each of ``before[a]`` (stable by the given
    order), or None on a cycle."""
    indegree = {n: 0 for n in nodes}
    for a in nodes:
        for b in before.get(a, ()):
            indegree[b] += 1
    ready = [n for n in nodes if indegree[n] == 0]
    out: List[str] = []
    while ready:
        n = ready.pop(0)
        out.append(n)
        for b in sorted(before.get(n, ()), key=list(nodes).index):
            indegree[b] -= 1
            if indegree[b] == 0:
                ready.append(b)
        ready.sort(key=list(nodes).index)
    return out if len(out) == len(nodes) else None


def metal_bind_commands(index: Mapping[str, Any], results: Path, index_path: str,
                        selection: Mapping[str, Any], selection_path: str,
                        repo: Optional[Path] = None, write: bool = False) -> List[str]:
    """The Metal bind of exactly ``selection``: ``rebind_metal_welds.py --stamp`` once per
    selected stamp, in launch order (each reads its own stamp's environment record)."""
    repo = Path(repo) if repo is not None else HARNESS.parents[1]
    w = " --write" if write else ""
    units = {u["id"]: u for u in units_for(repo, "metal")}
    stamps = _ordered(selection, selection["selected"].values())
    out = [f"python rounds/1.0/chunks.py verify --index {index_path} --repo ../.. "
           f"--results results --against {selection_path}"]
    out += stamp_tool_conflicts(selection, stamps, ("metal_gate",), units, results,
                                "rebind_metal_welds.py")
    out += [f"python rebind_metal_welds.py --stamp {stamp}{w}" for stamp in stamps]
    return out


def records_commands(index: Mapping[str, Any], selection: Mapping[str, Any],
                     results: Optional[Path] = None, repo: Optional[Path] = None) -> List[str]:
    """The records-time board re-point over the selected stamps.

    Only an index of a CERTIFIED device re-points a board: the boards credit one run per
    product, and a new device's runs are certified by the records its bind ADDS beside the
    certified device's, so they never replace the runs a board names. ``repoint_board_bindings``
    refuses two admissible runs of one gate among the stamps it is given; a unit the
    selection holds a second such run of is named on a STOP line."""
    board = "triton" if index["machine"] == "nvidia" else "metal"
    if index["device_mode"] == "new":
        return [f"no board re-point: {index['device']} is a new device, certified by the "
                f"records its bind adds; the {board} board stays bound to the certified "
                f"device's runs"]
    stamps = _ordered(selection, selection["selected"].values())
    lines = []
    if results is not None:
        repo = Path(repo) if repo is not None else HARNESS.parents[1]
        kinds = ("triton_hd",) if board == "triton" else ("metal_gate",)
        for uid, unit in sorted({u["id"]: u for u in units_for(repo, index["machine"])}.items()):
            if unit["kind"] not in kinds:
                continue
            admissible = [c["stamp"] for c in (selection["units"].get(uid) or {}).get(
                "candidates") or () if c.get("tool") and c["stamp"] in stamps]
            if len(admissible) > 1:
                lines.append(f"STOP repoint_board_bindings.py: {uid} holds a released run "
                             f"under {admissible}, two of the stamps given; set the "
                             f"unselected ones aside first")
    return lines + [f"python repoint_board_bindings.py {board} --stamps {','.join(stamps)} "
                    f"--device {index['device']}"]


def records_lines(index: Mapping[str, Any], selection: Mapping[str, Any],
                  results: Optional[Path] = None, repo: Optional[Path] = None) -> List[str]:
    """The records-time lines both bind printers end with (``chunks.py bind`` and
    ``legs.py bind --chunks``), commented out: they run at the RECORDS stage."""
    return ["# records time (RECORDS stage), after the bind:"] + [
        f"# {line}" for line in records_commands(index, selection, results, repo)]


def bind_commands(index: Mapping[str, Any], results: Path, index_path: str,
                  selection_path: str, repo: Optional[Path] = None,
                  write: bool = False) -> List[str]:
    selection = read_selection(Path(selection_path), index)
    if index["machine"] == "nvidia":
        lines = nvidia_bind_commands(index, results, index_path, selection, selection_path,
                                     repo, write)
    else:
        lines = metal_bind_commands(index, results, index_path, selection, selection_path,
                                    repo, write)
    digest = hashlib.sha256(Path(selection_path).read_bytes()).hexdigest()
    head = [f"# the bind of selection {selection_path} (sha256 {digest[:16]}): "
            f"{len(selection['selected'])} unit(s) from "
            f"{len(set(selection['selected'].values()))} stamp(s); "
            + ", ".join(f"{k} {v}" for k, v in sorted(selection["counts"].items()))]
    # A STOP WITHHOLDS THE BIND. A STOP line names a run a tool would bind against the
    # selection (or a command that cannot be ordered to bind it); printing the commands
    # after it would leave the selection to whoever reads the comments, and a shell that
    # runs the output top to bottom skips them. So nothing that writes a ledger is printed
    # while one stands: the runs are set aside as each line says, verify writes the
    # selection again, and the bind is printed again.
    stops = [line for line in lines if line.startswith("# STOP")]
    if stops:
        return head + lines[:1] + stops + [
            f"{WITHHELD}: {len(stops)} STOP line(s) above; set each run aside as its line "
            f"says, run chunks.py verify --selection {selection_path} again, and print the "
            f"bind again"]
    return head + lines


#: The line :func:`bind_commands` ends with when it prints no command that writes a ledger.
WITHHELD = "# BIND WITHHELD"


def bind_withheld(lines: Sequence[str]) -> bool:
    return any(line.startswith(WITHHELD) for line in lines)


# --------------------------------------------------------------------------- main ------
def _partition_closure_hits(repo: Path, machine: str, patterns: Sequence[str],
                            prior: Optional[Mapping[str, Any]]) -> Set[str]:
    ic = load_import_closure(repo)
    resolver = ic.Resolver(Path(repo))
    if lazy_gap_open(resolver) and not prior:
        raise SystemExit("--defer on a tip whose static reader misses _LAZY reaches needs "
                         "--prior-records")
    pins = ledger_pins(repo, ic)
    held = set()
    for unit in units_for(repo, machine):
        if any(matches(name, patterns) for name in unit_closure(unit, resolver, pins, prior)):
            held.add(unit["id"])
    return held


def _read_changing(args: argparse.Namespace) -> List[str]:
    out = list(args.changing or [])
    if args.changing_file:
        for line in Path(args.changing_file).read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                out.append(line)
    if args.since:
        out += changed_since(Path(args.repo), args.since)
    return sorted(set(out))


def _units_named(args: argparse.Namespace, index: Mapping[str, Any]) -> List[str]:
    """The units ``add`` names: a list, a file of ids, or a verify report's units in the
    states named by ``--states``."""
    named: List[str] = [u for u in (args.units or "").split(",") if u.strip()]
    if args.units_file:
        for line in Path(args.units_file).read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                named.append(line)
    if args.from_verify:
        report = json.loads(Path(args.from_verify).read_text(encoding="utf-8"))
        states = {s.strip() for s in args.states.split(",") if s.strip()}
        named += [uid for uid, row in (report.get("units") or {}).items()
                  if row.get("state") in states]
    return sorted(set(u.strip() for u in named))


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("partition")
    p.add_argument("--repo", required=True)
    p.add_argument("--machine", choices=MACHINES, required=True)
    p.add_argument("--capability", default=None, help="nvidia: the compute capability (8.6)")
    p.add_argument("--architecture", default=None,
                   help="metal: the GPU architecture (applegpu_g13s)")
    p.add_argument("--new-device", action="store_true",
                   help="the device is one the ledgers hold no live record for (an H200, an "
                        "Apple M3): written into the index, which every driver then checks")
    p.add_argument("--out", required=True)
    p.add_argument("--chunks", type=int, default=None)
    p.add_argument("--round", default=None,
                   help="the round stamp (default: read from the route-stamp registry)")
    p.add_argument("--defer", nargs="*", default=[],
                   help="files (exact, dir/ prefix or glob) still changing: units whose "
                        "closure holds one are packed apart from the rest")
    p.add_argument("--prior-records", default=None)
    s = sub.add_parser("show")
    s.add_argument("--index", required=True)
    c = sub.add_parser("check")
    c.add_argument("--index", required=True)
    c.add_argument("--repo", required=True)
    c.add_argument("--results", default=None)
    pl = sub.add_parser("plan")
    pl.add_argument("--index", required=True)
    pl.add_argument("--chunk", required=True)
    pl.add_argument("--repo", required=True)
    pl.add_argument("--results", required=True)
    pl.add_argument("--out", required=True)
    pl.add_argument("--lanes", type=int, default=None)
    r = sub.add_parser("records")
    r.add_argument("--repo", required=True)
    r.add_argument("--results", required=True)
    r.add_argument("--stamp", required=True, help="the previous round's stamp")
    r.add_argument("--machine", choices=MACHINES, required=True)
    r.add_argument("--out", required=True)
    rd = sub.add_parser("ready")
    rd.add_argument("--index", required=True)
    rd.add_argument("--repo", required=True)
    rd.add_argument("--changing", nargs="*", default=[])
    rd.add_argument("--changing-file", default=None)
    rd.add_argument("--since", default=None, help="add the files changed from COMMIT to HEAD")
    rd.add_argument("--prior-records", default=None)
    rd.add_argument("--json", default=None)
    st = sub.add_parser("stale")
    st.add_argument("--index", required=True)
    st.add_argument("--repo", required=True)
    st.add_argument("--results", required=True)
    st.add_argument("--prior-records", default=None)
    st.add_argument("--json", default=None)
    v = sub.add_parser("verify")
    v.add_argument("--index", required=True)
    v.add_argument("--repo", required=True)
    v.add_argument("--results", required=True)
    v.add_argument("--selection", default=None,
                   help="write the per-unit choice here, which the bind reads")
    v.add_argument("--against", default=None,
                   help="a selection verify wrote: judge again and exit 1 when the choice "
                        "or the bytes it binds moved since (writes nothing; the bind's "
                        "first command)")
    v.add_argument("--json", default=None)
    b = sub.add_parser("bind")
    b.add_argument("--index", required=True)
    b.add_argument("--selection", required=True, help="the file chunks.py verify wrote")
    b.add_argument("--results", default="results")
    b.add_argument("--repo", default=None, help="default: the checkout holding this file")
    b.add_argument("--write", action="store_true",
                   help="print the commands WITH --write (they are printed, never run)")
    ad = sub.add_parser("add")
    ad.add_argument("--index", required=True)
    ad.add_argument("--repo", required=True)
    ad.add_argument("--units", default="")
    ad.add_argument("--units-file", default=None)
    ad.add_argument("--from-verify", default=None, help="a chunks.py verify --json report")
    ad.add_argument("--states", default="REFUSED,ABSENT,NOT LAUNCHED",
                    help="with --from-verify: the unit states to run again")
    ad.add_argument("--chunks", type=int, default=1)
    ad.add_argument("--gpus", type=int, default=1)
    ad.add_argument("--why", required=True)
    rl = sub.add_parser("relaunch")
    rl.add_argument("--index", required=True)
    rl.add_argument("--chunk", required=True)
    rl.add_argument("--why", required=True)
    ex = sub.add_parser("extend")
    ex.add_argument("--index", required=True)
    ex.add_argument("--repo", required=True)
    rp = sub.add_parser("repartition")
    rp.add_argument("--index", required=True)
    rp.add_argument("--repo", required=True)
    rp.add_argument("--results", required=True,
                    help="where the launch records are: a chunk with one is never dropped")
    rp.add_argument("--chunks", type=int, default=None)
    rp.add_argument("--why", required=True)
    args = ap.parse_args(argv)

    if args.cmd == "partition":
        repo = Path(args.repo).resolve()
        try:
            if args.machine == "nvidia":
                if args.architecture or not args.capability:
                    ap.error("an nvidia index takes --capability (8.6, 9.0)")
                device = device_of_capability(args.capability)
            else:
                if args.capability or not args.architecture:
                    ap.error("a metal index takes --architecture (applegpu_g13s)")
                device = args.architecture
        except ValueError as refused:
            ap.error(str(refused))
        bad = device_problems(args.machine, device)
        if bad:
            ap.error(bad[0])
        stamp, where = registry_round_stamp(repo, args.machine)
        if args.round:
            if stamp is not None and stamp != args.round:
                raise SystemExit(f"--round {args.round}: the registry names {stamp} ({where})")
            stamp, where = args.round, ("--round" if stamp is None else where)
        if stamp is None:
            raise SystemExit(f"no round stamp: {where}; pass --round <stamp> (the freeze then "
                             f"sets route_stamps.ROUND to it)")
        prior = json.loads(Path(args.prior_records).read_text()) if args.prior_records else None
        units = units_for(repo, args.machine)
        n = args.chunks or DEFAULT_CHUNKS[args.machine]
        deferred = (_partition_closure_hits(repo, args.machine, args.defer, prior)
                    if args.defer else set())
        bins = partition(units, n, deferred)
        index = make_index(args.machine, stamp, bins, {
            "rule": "lpt by legs.py / metal_round.py measured walls"
                    + (", deferred units packed apart" if args.defer else ""),
            "chunks": n, "defer": sorted(args.defer), "deferred_units": len(deferred),
            "round_stamp_from": where, "units": len(units)},
            device=device, device_mode="new" if args.new_device else "certified")
        problems = collisions(index, repo=repo) + round_problems(repo, index)
        for line in problems:
            print(f"FAIL {line}", flush=True)
        if problems:
            return 1
        out = Path(args.out)
        if out.exists():
            raise SystemExit(f"{out} exists; an index is written once (add, relaunch, extend "
                             f"and repartition change it)")
        out.parent.mkdir(parents=True, exist_ok=True)
        write_index(out, index)
        for chunk in index["chunks"]:
            print(f"{chunk['id']} {chunk['stamp']}: {len(chunk['units'])} units, "
                  f"{chunk['expect_s']} s", flush=True)
        print(f"wrote {out}: {len(index['chunks'])} chunks of round {stamp} on {device} "
              f"({index['device_mode']} device; {where})", flush=True)
        return 0

    if args.cmd == "records":
        found = prior_records(Path(args.repo).resolve(), Path(args.results).resolve(),
                              args.stamp, args.machine,
                              say=lambda line: print(line, flush=True))
        Path(args.out).write_text(json.dumps(found, indent=1, sort_keys=True) + "\n")
        units = [k for k in found if not k.startswith("_")]
        with_records = sum(1 for k in units if found[k]["artifacts"])
        print(f"RECORDS {with_records} of {len(units)} units carry a recorded artifact under "
              f"{args.stamp}; wrote {args.out}", flush=True)
        return 0

    index = read_index(Path(args.index))
    if args.cmd == "show":
        print(f"{index['machine']} {index['device']} ({index['device_mode']} device), round "
              f"{index['round']}, next_k {index['next_k']}", flush=True)
        for chunk in index["chunks"]:
            roots = chunk_roots(chunk, index["machine"])
            print(f"{chunk['id']} {chunk['stamp']}: {len(chunk['units'])} units, "
                  f"{chunk['expect_s']} s, {len(roots)} roots ({chunk.get('why')})", flush=True)
        for chunk in index["retired"]:
            print(f"retired {chunk['id']} {chunk['stamp']}: {chunk.get('retired_why')}",
                  flush=True)
        return 0
    if args.cmd == "check":
        problems = collisions(index, results=Path(args.results) if args.results else None,
                              repo=Path(args.repo).resolve())
        problems += round_problems(Path(args.repo).resolve(), index)
        for line in problems:
            print(f"FAIL {line}", flush=True)
        print(f"CHECK {'OK' if not problems else 'REFUSED'} ({len(problems)} problem(s))",
              flush=True)
        return 1 if problems else 0
    if args.cmd == "plan":
        chunk = find_chunk(index, args.chunk)
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        if index["machine"] == "metal":
            gates = metal_gates(chunk)
            (out / "metal_gates.txt").write_text("\n".join(gates) + "\n")
            print(f"{chunk['id']} {chunk['stamp']}: {len(gates)} Metal gates", flush=True)
            return 0
        plan, summary = nvidia_plan(Path(args.repo).resolve(), Path(args.results).resolve(),
                                    chunk, args.lanes or int(chunk.get("gpus", 1)))
        for k, steps in enumerate(plan):
            if steps:
                (out / f"lane_{k}.json").write_text(json.dumps(steps, indent=1) + "\n")
        (out / "lanes_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
        for k, lane in enumerate(summary["lanes"]):
            print(f"lane_{k}: {lane['steps']} steps, {lane['expect_s']} s expected: "
                  f"{', '.join(lane['units'])}", flush=True)
        return 0
    if args.cmd == "ready":
        prior = json.loads(Path(args.prior_records).read_text()) if args.prior_records else None
        report = readiness(Path(args.repo).resolve(), index, _read_changing(args), prior,
                           say=lambda line: print(line, flush=True))
        if args.json:
            Path(args.json).write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
        return 0
    if args.cmd == "stale":
        prior = json.loads(Path(args.prior_records).read_text()) if args.prior_records else None
        report = staleness(Path(args.repo).resolve(), index, Path(args.results).resolve(), prior,
                           say=lambda line: print(line, flush=True))
        if args.json:
            Path(args.json).write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
        return 0
    if args.cmd == "verify":
        repo = Path(args.repo).resolve()
        report = verify(repo, Path(args.results).resolve(), index,
                        say=lambda line: print(line, flush=True))
        if args.json:
            Path(args.json).write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
        if args.against:
            moved = selection_moved(read_selection(Path(args.against), index),
                                    selection_of(report, index, repo))
            for line in moved:
                print(f"FAIL {line}", flush=True)
            print(f"SELECTION {'MOVED' if moved else 'HOLDS'}: {args.against} "
                  + (f"({len(moved)} difference(s)); stop, and write the selection and print "
                     f"the bind again" if moved else "is the choice the results give now"),
                  flush=True)
            if moved:
                return 1
        if args.selection:
            Path(args.selection).write_text(json.dumps(
                selection_of(report, index, repo), indent=1, sort_keys=True) + "\n")
            print(f"wrote the selection {args.selection}: {report['counts']['OK']} unit(s)",
                  flush=True)
        return 0 if report["ok"] else 1
    if args.cmd == "bind":
        results = Path(args.results).resolve()
        repo = Path(args.repo).resolve() if args.repo else None
        lines = bind_commands(index, results, args.index, args.selection, repo, args.write)
        withheld = bind_withheld(lines)
        if not withheld:
            lines += records_lines(index, read_selection(Path(args.selection), index),
                                   results, repo)
        for line in lines:
            print(line, flush=True)
        return 1 if withheld else 0
    if args.cmd in ("add", "relaunch", "extend", "repartition"):
        path = Path(args.index)
        with IndexLock(path, f"chunks.py {args.cmd}"):
            index = read_index(path)
            repo = Path(getattr(args, "repo", None) or HARNESS.parents[1]).resolve()
            have = {u["id"]: u for u in units_for(repo, index["machine"])}
            if args.cmd == "add":
                named = _units_named(args, index)
                unknown = sorted(set(named) - set(have))
                if unknown:
                    raise SystemExit(f"the checkout has no unit(s) {unknown[:6]}")
                if not named:
                    print("no unit named; nothing added", flush=True)
                    return 0
                made = [new_chunk(index, units, args.why, args.gpus) for units in
                        lpt([have[u] for u in named], max(1, args.chunks)) if units]
            elif args.cmd == "relaunch":
                old = find_chunk(index, args.chunk)
                missing = sorted(set(old["units"]) - set(have))
                if missing:
                    raise SystemExit(f"{args.chunk} names unit(s) the checkout lacks: "
                                     f"{missing[:6]}")
                made = [new_chunk(index, [have[u] for u in old["units"]],
                                  f"relaunch of {old['id']}: {args.why}")]
            elif args.cmd == "extend":
                placed = {u for c in index["chunks"] for u in c["units"]}
                new = sorted(set(have) - placed)
                if not new:
                    print("every unit of the checkout is in a chunk; nothing to extend",
                          flush=True)
                    return 0
                made = [new_chunk(index, [have[u] for u in new], "extend")]
            else:
                results = Path(args.results).resolve()
                dropped = [c for c in index["chunks"]
                           if read_launch_record(results, c["stamp"]) is None]
                if not dropped:
                    print("every current chunk has launched; nothing to re-partition "
                          "(chunks.py add runs units again)", flush=True)
                    return 0
                for chunk in dropped:
                    index["chunks"].remove(chunk)
                    index["retired"].append(dict(chunk, retired_why=args.why,
                                                 retired_utc=_utc()))
                pool = sorted({u for c in dropped for u in c["units"] if u in have})
                n = args.chunks or len(dropped)
                made = [new_chunk(index, units, f"repartition: {args.why}") for units in
                        lpt([have[u] for u in pool], n) if units]
            problems = collisions(index)
            if problems:
                for line in problems:
                    print(f"FAIL {line}", flush=True)
                return 1
            write_index(path, index)
        for chunk in made:
            print(f"{chunk['id']} {chunk['stamp']}: {len(chunk['units'])} unit(s), "
                  f"{chunk['expect_s']} s ({chunk['why']})", flush=True)
        return 0
    raise AssertionError(args.cmd)


if __name__ == "__main__":
    sys.exit(main())
