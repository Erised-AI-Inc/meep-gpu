"""Per-architecture run records for the Metal ledger: the rules every writer obeys.

WHY THIS EXISTS. A Metal weld entry held ONE run: its artifact, its ``host`` line and
its timestamp sat beside the digests of the bytes it certified. That shape can hold
one Apple GPU architecture, so a second Mac's round could only replace the first
one's certification, never add to it. The NVIDIA ledgers solved the same problem with
one record per compute capability (``fastpath.bind_capability`` and its neighbours);
this module is the Metal half of that design, keyed by the GPU architecture Metal
compiles for (``applegpu_g13s``, ...).

WHAT A RUN RECORD IS. Each weld entry keeps its digests and gains
``runs[<architecture>]``: the fields that describe one run of the gate
(:data:`RUN_FIELDS`), the torch and Metal frontend that run recorded, and
``bound_sha256``, the digest :func:`fastpath.bound_digest` takes over
``fastpath.BOUND_FIELDS``. A run is LIVE while its ``bound_sha256`` equals the
entry's, and :func:`fastpath.live_capabilities` is the one liveness answer: the
dispatch admission (``metal_dispatch.cited_environments``) and every tool here ask it.

WHY THE NVIDIA WRITERS ARE NOT REUSED AS THEY ARE. ``fastpath._require_capability``,
``fastpath.bind_capability``, ``fastpath.retired_shape_reasons`` and
``fastpath.route_campaign`` accept only a ``major.minor`` key and refuse
``applegpu_g13s`` by name, and ``fastpath.RUN_FIELDS`` does not list
``environment_read_from``. ``fastpath.py`` is pinned by records a pending NVIDIA bind
needs untouched, so the Metal rules live here, in a module no ledger pins, and borrow
only what works as it is: ``RUNS``, ``BOUND_FIELDS``, ``bound_digest`` and
``live_capabilities``.

WHO READS THIS MODULE. The writers (``parity/meep_gpu/rebind_metal_welds.py``,
``mint_metal_weld.py``, ``migrate_metal_runs.py``,
``recut_driver_dispatch_record.py --backend metal``), the weld contract and the
fusion-matrix builder. The dispatch ladder does NOT import it: the admission reads
``runs`` through ``fastpath.live_capabilities`` inside ``metal_dispatch.py``, the file
the Metal ``driver_dispatch`` record binds, so what decides a dispatch stays in bytes
a record is cut against.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from . import fastpath
from . import metal_dispatch

#: The key, inside a ledger entry, holding one record per GPU architecture. The same
#: key the NVIDIA ledgers use, so the record walker and ``fastpath.live_capabilities``
#: read both.
RUNS = fastpath.RUNS

#: What an architecture key may be: the name ``MTLDevice.architecture`` reports.
#: Checked rather than inferred, so a typo or a device name cannot key a record.
ARCHITECTURE_KEY = re.compile(r"^applegpu_[a-z0-9_]+$")

#: The environment facts a run records beside its host line, under the names
#: ``metal_dispatch.ENVIRONMENT_FACTS`` uses for them. The architecture is the key.
ENVIRONMENT_FIELDS: Tuple[str, ...] = tuple(
    fact for fact in metal_dispatch.ENVIRONMENT_FACTS if fact != "architecture")

#: The fields that describe ONE RUN of a Metal weld's gate. They live in that run's
#: record under :data:`RUNS`, never beside the digests. ``status`` and ``purpose``
#: stay at entry level: they describe the weld, as on the NVIDIA ledgers.
#: ``_subnormal_policy_read_from`` says where a rebind took ``subnormal_policy`` from,
#: because a gate artifact does not always state it.
RUN_FIELDS: Tuple[str, ...] = (
    "artifact_sha256", "environment_read_from", "host", "recorded_utc", "records",
    "subnormal_policy", "verdict_read_from") + ENVIRONMENT_FIELDS + (
    "_subnormal_policy_read_from",)

#: The route-campaign fields of the Metal ``driver_dispatch`` record, kept per
#: architecture. ``residency`` and ``lift`` are read off the legs case by case, so a
#: second architecture's cut must not overwrite the first's; ``subnormal``,
#: ``environments``, the slots and the release predicate are statements about the
#: shipped module and stay at entry level.
DISPATCH_RUN_FIELDS: Tuple[str, ...] = (
    "status", "records", "legs", "recorded_utc", "verdict_read_from",
    "released_fused_arms", "residency", "lift")

#: Entries of the Metal ledger that are not weld records.
NOT_WELDS: Tuple[str, ...] = ("driver_dispatch", "metal_kernels")


class ArchitectureRecordError(fastpath.CapabilityRecordError):
    """A Metal ledger entry whose per-architecture records cannot be read or written."""


class ArchitectureStale(ArchitectureRecordError):
    """A write that would leave other architectures' runs bound to bytes that moved."""

    def __init__(self, names: Sequence[str]) -> None:
        self.names = tuple(sorted(names))
        super().__init__(
            f"this write changes the bytes the entry binds, which would leave the runs "
            f"for {list(self.names)} certifying bytes that no longer ship; re-run those "
            f"architectures on these bytes, or supersede them by name "
            f"(--supersede {','.join(self.names)})")


def require_architecture(value: Any) -> str:
    """``value`` as an architecture key, or a named refusal."""
    if not isinstance(value, str) or not ARCHITECTURE_KEY.match(value):
        raise ArchitectureRecordError(
            f"{value!r} is not a GPU architecture key (an MTLDevice architecture name, "
            f"as {ARCHITECTURE_KEY.pattern} spells it)")
    return value


def environment_of(host: Any) -> Dict[str, Optional[str]]:
    """The architecture, torch and Metal frontend a ``host`` line names.

    ``metal_dispatch.host_environment``, so the frontend comes back in the one
    spelling the admission compares (``frontend_key``: no ``metalfe-`` prefix).
    """
    return metal_dispatch.host_environment(host)


#: The suffix every derived ``subnormal_policy`` line carries: the policy a Metal weld
#: runs under is the table's, MPS flushing float32 subnormals natively.
POLICY_SUFFIX = " - native and uncontrollable on MPS; the oracle flushes too"

#: The keys of a gate's policy REPORT (a mapping) that hold the policy itself, in the
#: order they are read: what was in force, what it resolved to, what was asked for.
POLICY_REPORT_KEYS: Tuple[str, ...] = ("in_force", "resolved", "requested")


def artifact_policy(artifact: Mapping[str, Any]) -> Optional[str]:
    """The subnormal policy a gate artifact states, as its VALUE (``flush``), or ``None``.

    Two generations of gate write it differently: a plain string under
    ``subnormal_policy``, or a report (a mapping) under ``subnormal_policy`` or
    ``subnormal_policy_report``. A report is read by its policy keys
    (:data:`POLICY_REPORT_KEYS`) and NEVER formatted whole: an f-string of a mapping
    writes its Python repr into the ledger, and ``fastpath._certification_for`` quotes
    that line into every dispatched arm's record.
    """
    stated = artifact.get("subnormal_policy") or artifact.get("subnormal_policy_report")
    if isinstance(stated, Mapping):
        for key in POLICY_REPORT_KEYS:
            value = stated.get(key)
            if isinstance(value, str) and value:
                return value
        return None
    return stated if isinstance(stated, str) and stated else None


def artifact_architecture(artifact: Mapping[str, Any]) -> Optional[str]:
    """The GPU architecture a gate artifact records ITSELF, where it records one.

    Some Metal gates stamp ``environment.apple_gpu.architecture`` (14 of 62 on the
    2026-10-04 fleet); the others do not, and a writer then has only the campaign's
    environment records. Where the artifact does say, a writer compares the two, so a
    run cannot be filed under a GPU its own artifact says it did not run on.
    """
    environment = artifact.get("environment")
    apple = environment.get("apple_gpu") if isinstance(environment, Mapping) else None
    value = apple.get("architecture") if isinstance(apple, Mapping) else None
    return value if isinstance(value, str) and value else None


def weld_keys(ledger: Mapping[str, Any]) -> List[str]:
    """Every weld entry of a Metal ledger: a mapping that pins source bytes."""
    return sorted(name for name, entry in ledger.items()
                  if name not in NOT_WELDS and not name.startswith("_")
                  and isinstance(entry, Mapping) and "source_sha256" in entry)


def shape_reasons(entry: Any, run_fields: Sequence[str] = RUN_FIELDS) -> List[str]:
    """Why this entry is not in the per-architecture shape. Empty means it is.

    THE SHAPE IS REFUSED BY NAME rather than read both ways, as on the NVIDIA
    ledgers: a run field beside the digests would make "which bytes did this run
    certify" a question with two answers. A run's ``host`` line must also name the
    architecture it is keyed by and the torch and frontend it records, so the line a
    reader sees and the fields the admission compares cannot disagree.
    """
    reasons: List[str] = []
    if not isinstance(entry, Mapping):
        return [f"not a mapping: {type(entry).__name__}"]
    stranded = sorted(field for field in run_fields if field in entry)
    if stranded:
        reasons.append(f"run fields beside the digests: {stranded}")
    runs = entry.get(RUNS)
    if runs is None:
        reasons.append(f"no {RUNS!r} record")
    elif not isinstance(runs, Mapping):
        reasons.append(f"{RUNS!r} is {type(runs).__name__}, not a mapping")
    else:
        for architecture in sorted(runs, key=str):
            run = runs[architecture]
            where = f"{RUNS}[{architecture!r}]"
            if not ARCHITECTURE_KEY.match(str(architecture)):
                reasons.append(f"{where} is not a GPU architecture key")
                continue
            if not isinstance(run, Mapping):
                reasons.append(f"{where} is not a mapping")
                continue
            if "bound_sha256" not in run:
                reasons.append(f"{where} records no bound_sha256")
            foreign = sorted(set(run) - set(run_fields) - {"bound_sha256"})
            if foreign:
                reasons.append(f"{where} carries fields that are not run fields: "
                               f"{foreign}")
            if "host" in run:
                named = environment_of(run["host"])
                if named["architecture"] != architecture:
                    reasons.append(f"{where}.host names architecture "
                                   f"{named['architecture']!r}")
                for field in ENVIRONMENT_FIELDS:
                    recorded = run.get(field)
                    if field == "metal_frontend":
                        recorded = metal_dispatch.frontend_key(recorded)
                    if recorded != named[field]:
                        reasons.append(f"{where}.{field} is {run.get(field)!r} and its "
                                       f"host line names {named[field]!r}")
    if not any(field in entry for field in fastpath.BOUND_FIELDS):
        reasons.append(f"binds no bytes: none of {list(fastpath.BOUND_FIELDS)}")
    return reasons


def live_architectures(entry: Any) -> Tuple[str, ...]:
    """The architectures whose run still binds this entry's bytes."""
    return fastpath.live_capabilities(entry)


def live_runs(entry: Any) -> Dict[str, Mapping[str, Any]]:
    """``{architecture: run}`` for this entry's live runs, in key order."""
    live = live_architectures(entry)
    return {architecture: entry[RUNS][architecture] for architecture in live}


def live_run(entry: Any, architecture: Optional[str]) -> Optional[Mapping[str, Any]]:
    """This architecture's live run, or ``None``."""
    if architecture is None:
        return None
    return live_runs(entry).get(architecture)


def bind_architecture(entry: Dict[str, Any], *, bound_before: Optional[str],
                      architecture: str, run: Mapping[str, Any],
                      run_fields: Sequence[str] = RUN_FIELDS,
                      supersede: Sequence[str] = ()) -> Tuple[str, ...]:
    """Write one architecture's run record, and refuse to strand the others.

    ``fastpath.bind_capability``'s rule with an architecture key. A write on
    UNCHANGED bytes adds this architecture beside the ones already recorded, which
    is what lets a second Mac join the first. A write on bytes that MOVED leaves
    every other architecture's run certifying code that no longer ships, so it
    refuses and names them unless each is listed in ``supersede``.

    ``bound_before`` is :func:`fastpath.bound_digest` of the entry as it stood
    before the writer refreshed its digests (``None`` for an entry written for the
    first time). Returns the architectures this write superseded.
    """
    require_architecture(architecture)
    for name in supersede:
        require_architecture(name)
    foreign = sorted(set(run) - set(run_fields))
    if foreign:
        raise ArchitectureRecordError(
            f"{foreign} are not run fields; a field that describes the BYTES belongs "
            f"beside the digests, not inside {RUNS}[{architecture!r}]")
    current = fastpath.bound_digest(entry)
    runs = entry.setdefault(RUNS, {})
    if not isinstance(runs, Mapping):
        raise ArchitectureRecordError(f"{RUNS!r} is {type(runs).__name__}, not a mapping")
    staled: Tuple[str, ...] = ()
    if bound_before is not None and current != bound_before:
        staled = tuple(sorted(
            name for name, record in runs.items()
            if name != architecture and isinstance(record, Mapping)
            and record.get("bound_sha256") == bound_before))
        unnamed = sorted(set(staled) - set(supersede))
        if unnamed:
            raise ArchitectureStale(unnamed)
    runs[architecture] = {key: run[key] for key in sorted(run)}
    runs[architecture]["bound_sha256"] = current
    entry[RUNS] = {name: runs[name] for name in sorted(runs)}
    return staled


def admission_report(ledger: Mapping[str, Any], keys: Sequence[str]) -> Dict[str, Any]:
    """Which architectures every one of ``keys`` has a live run for, and what blocks.

    THE INTERSECTION IS THE ANSWER, as in ``fastpath.capability_report``: the table
    dispatches an arm from any cited weld, so an architecture one cited weld never
    ran on is one the table cannot claim. ``admitted`` is ``None`` for a ledger that
    could not be read and ``()`` for one whose cited welds share no live
    architecture.
    """
    by_key: Dict[str, Dict[str, Any]] = {}
    if not isinstance(ledger, Mapping) or not ledger:
        return {"admitted": None, "by_key": by_key}
    admitted: Optional[set] = None
    for key in sorted(set(keys)):
        entry = ledger.get(key)
        live = live_architectures(entry)
        problem = None
        if not isinstance(entry, Mapping):
            problem = f"no entry under {key!r}"
        else:
            reasons = shape_reasons(entry)
            if reasons:
                problem = "; ".join(reasons)
            elif not live:
                problem = ("every recorded run binds bytes that have since moved; "
                           "re-run this gate")
        runs = entry.get(RUNS) if isinstance(entry, Mapping) else None
        by_key[key] = {
            "live": list(live),
            "stale": sorted(set(runs) - set(live)) if isinstance(runs, Mapping) else [],
            "problem": problem,
        }
        admitted = set(live) if admitted is None else (admitted & set(live))
    return {"admitted": tuple(sorted(admitted or ())), "by_key": by_key}


def cited_keys() -> List[str]:
    """The weld keys ``metal_dispatch.ARM_CERTIFICATION`` cites."""
    return sorted({gate for _family, gate in metal_dispatch.ARM_CERTIFICATION.values()})


def route_campaign(stamp: str, architecture: str) -> str:
    """The route campaign directory for one architecture: ``<stamp>_<architecture>``.

    One stamp per release (``metal_dispatch.METAL_DRIVER_ROUTE_GATE``), one campaign
    per architecture under it, so the directory name says which GPU a route record
    describes, as ``fastpath.route_campaign`` spells ``<stamp>_cc86`` on the NVIDIA
    tables.
    """
    return f"{stamp}_{require_architecture(architecture)}"
