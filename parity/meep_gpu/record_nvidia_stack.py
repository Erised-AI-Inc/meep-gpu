"""Record the NVIDIA stack a compute capability was certified on, from the lanes that certified it.

    python record_nvidia_stack.py --stack results/chunk_<stamp>/stack [--stack ...]
                                  [--capability 9.0] [--restate]            # report
    python record_nvidia_stack.py ... --write

Why. ``MEEP_GPU_HARDWARE=certified`` compares the stack the NVIDIA kernels run beside (CuPy,
NVRTC, the CUDA runtime and driver, the host's OS and architecture, NumPy, the MEEP build)
against the stack the device's own compute capability was certified on
(``meep_gpu.fastpath.certified_nvidia_stack``, rung 4c). That stack is a fact a
certification measures, not one a person types: each gate lane of a certification round
reads it on its own card in its toolchain step (``rounds/1.0/run_lane.py``) and writes
``<lane>.json`` into the stage's stack directory, and this tool, run by the bind, records it
in the Triton ledger under ``certified_nvidia_stack.by_capability[<capability>]``. So a
capability first certified on a newer driver is certified on that driver, and every other
capability keeps the stack its own certification measured.

What it refuses, writing nothing:

* a stack directory that is missing or holds no lane record, a record that is not of
  schema ``meep_gpu.nvidia_stack/1``, names no ``major.minor`` capability, or leaves an axis
  unread: an axis nobody read cannot be certified;
* lanes that disagree on the capability or on any axis but the CUDA driver: binding either
  answer would certify a stack some lanes did not run beside;
* a capability other than ``--capability``, when it is given;
* a MEEP build that is not one of ``fastpath.CERTIFIED_MEEP_BUILDS`` on that host;
* a lane record outside the repository, which the ledger could only name by an absolute
  path;
* a capability whose recorded stack differs from the one the lanes read, unless
  ``--restate`` says so (below).

Several CUDA drivers. A compute capability may be certified on several CUDA drivers, and
the record lists every driver its certified runs used (``stack.cuda_driver``, ordered by
version) with the stamps of the stages whose lanes ran on each
(``stamps_by_cuda_driver``). A driver is admitted only from lanes that agree on every other
axis, with each other and with the values the capability already records, so a second
driver adds one whole stack, never a union of unrelated axes: dispatch admits a host when
each axis it reads is among that axis's values, and only the driver axis holds more than
one. Each listed driver is certified by lane records read on it; a driver no lane read is
never listed.

``--restate``. Re-stating a certified capability's stack is a decision. Without
``--restate`` the tool writes only what keeps every certified stack certified: the values
already recorded (re-recording ``read_from``, ``stamps_by_cuda_driver`` and
``recorded_utc``, the earliest lane's), or those values with drivers added whose lanes
agree with them on every other axis. Anything else is refused by name: another value on any
axis but the driver, or lanes that ran on none of a recorded driver, which would drop it.
With ``--restate`` the record becomes exactly the stack these lanes read, so the drivers no
lane ran on are dropped. Exit: 0 done or nothing to refuse, 1 refused, 2 usage.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional, Sequence, Tuple

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
for _path in (str(_ROOT), str(_HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from meep_gpu import fastpath  # noqa: E402

#: The ledger the record lives in, relative to the repository root.
LEDGER = "meep_gpu/triton_kernels/fingerprints.json"
#: The schema of a lane's stack record (``rounds/1.0/run_lane.py`` writes it).
STACK_SCHEMA = "meep_gpu.nvidia_stack/1"
#: The serialisation every Triton ledger writer uses.
LEDGER_FORMAT = {"indent": 2, "sort_keys": True}
#: The one axis a capability's record may hold several values of.
DRIVER_AXIS = "cuda_driver"
#: Where the record lists, per CUDA driver, the stamps of the stages whose lanes ran on it.
STAMPS_BY_DRIVER = "stamps_by_cuda_driver"


class Refused(Exception):
    """A reason this tool writes nothing."""


def read_lanes(directories: Sequence[Path], root: Path) -> List[Tuple[str, Dict[str, Any]]]:
    """``(repository-relative path, record)`` for every lane record under ``directories``."""
    lanes: List[Tuple[str, Dict[str, Any]]] = []
    for directory in directories:
        directory = Path(directory).resolve()
        if not directory.is_dir():
            raise Refused(f"{directory} is not a directory: a stage records its lanes' "
                          f"stacks there, and without them there is nothing to certify")
        files = sorted(directory.glob("*.json"))
        if not files:
            raise Refused(f"{directory} holds no lane stack record")
        for path in files:
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise Refused(f"{path}: unreadable ({type(exc).__name__}: {exc})") from exc
            try:
                name = path.relative_to(root).as_posix()
            except ValueError as exc:
                raise Refused(f"{path.name} in {directory.name} lies outside the repository: "
                              f"the ledger names its records by repository-relative paths "
                              f"only; bind from a results tree inside the checkout") from exc
            lanes.append((name, record))
    return lanes


def lane_stack(name: str, record: Any) -> Tuple[str, Dict[str, str], Optional[str]]:
    """One lane record's capability, stack and time, or a named refusal."""
    if not isinstance(record, dict) or record.get("schema") != STACK_SCHEMA:
        raise Refused(f"{name} is not a lane stack record (schema "
                      f"{record.get('schema') if isinstance(record, dict) else None!r}, "
                      f"not {STACK_SCHEMA})")
    try:
        capability = fastpath._require_capability(record.get("capability"))  # noqa: SLF001
    except fastpath.CapabilityRecordError as exc:
        raise Refused(f"{name}: {exc}") from exc
    if capability != record.get("capability"):
        raise Refused(f"{name}: {record.get('capability')!r} is not spelled as the ledger "
                      f"keys a capability ({capability!r} is)")
    stack = record.get("stack")
    if not isinstance(stack, dict):
        raise Refused(f"{name} records no stack")
    missing = [axis for axis in fastpath.NVIDIA_STACK_AXES
               if not isinstance(stack.get(axis), str) or not stack.get(axis)]
    if missing:
        raise Refused(f"{name} leaves {missing} unread: an axis nobody read cannot be "
                      f"certified")
    extra = sorted(set(stack) - set(fastpath.NVIDIA_STACK_AXES))
    if extra:
        raise Refused(f"{name} records {extra}, which are not axes of the NVIDIA stack")
    return capability, {axis: stack[axis] for axis in fastpath.NVIDIA_STACK_AXES}, \
        record.get("utc")


def stage_stamp(name: str) -> str:
    """The stamp of the stage a lane record belongs to, read off its repository path.

    ``results/chunk_<stamp>/stack/<lane>.json`` for a chunk, ``results/stack_<stamp>/
    <lane>.json`` for an unchunked stage; any other directory is named by its path."""
    parent = PurePosixPath(name).parent
    if parent.name == "stack" and parent.parent.name.startswith("chunk_"):
        return parent.parent.name[len("chunk_"):]
    if parent.name.startswith("stack_"):
        return parent.name[len("stack_"):]
    return parent.as_posix()


def driver_order(value: str) -> Tuple[int, int, str]:
    """Drivers in version order (``cuDriverGetVersion`` numbers), anything else after them."""
    return (0, int(value), value) if value.isdigit() else (1, 0, value)


def judge(lanes: Sequence[Tuple[str, Dict[str, Any]]], capability: Optional[str],
          recorded: Optional[Dict[str, Any]], restate: bool) -> Dict[str, Any]:
    """The record the lanes certify, and what writing it changes. Raises :class:`Refused`."""
    if not lanes:
        raise Refused("no lane stack record was given")
    read = [(name,) + lane_stack(name, record) for name, record in lanes]
    capabilities = sorted({cap for _name, cap, _stack, _utc in read})
    if len(capabilities) > 1:
        raise Refused(f"the lanes ran on {capabilities}: one capability's certification is "
                      f"recorded at a time")
    found = capabilities[0]
    normalise = fastpath._normalized_capability  # noqa: SLF001
    if capability is not None and normalise(capability) != found:
        raise Refused(f"the lanes ran on compute capability {found}, not {capability}")
    # Every axis but the driver: one value, read alike by every lane. So each driver the
    # lanes ran on comes with the same value on every other axis, and the record adds whole
    # stacks, never a union of unrelated axes.
    stack: Dict[str, List[str]] = {}
    for axis in fastpath.NVIDIA_STACK_AXES:
        values: Dict[str, List[str]] = {}
        for name, _cap, lane, _utc in read:
            values.setdefault(lane[axis], []).append(name)
        if axis != DRIVER_AXIS and len(values) > 1:
            raise Refused(f"the lanes disagree on the {fastpath.NVIDIA_STACK_NAMES[axis]}: "
                          + "; ".join(f"{value} in {names}" for value, names
                                      in sorted(values.items())))
        stack[axis] = sorted(values, key=driver_order)
    meep, host = stack["meep"][0], stack["host"][0]
    if not fastpath.meep_build_is_certified(meep, host):
        raise Refused(f"MEEP {meep} on {host} is not a certified MEEP build "
                      f"({sorted(fastpath.CERTIFIED_MEEP_BUILDS)}): a capability is certified "
                      f"against the release's reference build")
    stamps: Dict[str, List[str]] = {driver: [] for driver in stack[DRIVER_AXIS]}
    for name, _cap, lane, _utc in read:
        stamp = stage_stamp(name)
        if stamp not in stamps[lane[DRIVER_AXIS]]:
            stamps[lane[DRIVER_AXIS]].append(stamp)
    times = sorted(utc for _name, _cap, _stack, utc in read if isinstance(utc, str) and utc)
    new = {"read_from": sorted(name for name, _cap, _stack, _utc in read),
           "recorded_by": "parity/meep_gpu/record_nvidia_stack.py",
           "stack": {axis: stack[axis] for axis in fastpath.NVIDIA_STACK_AXES},
           STAMPS_BY_DRIVER: {driver: sorted(names) for driver, names in stamps.items()}}
    if times:
        new["recorded_utc"] = times[0]
    before = (recorded or {}).get("stack") if isinstance(recorded, dict) else None
    if before is None:
        change = f"adds the certified stack of compute capability {found}"
    elif before == new["stack"]:
        change = (f"keeps the certified stack of compute capability {found} and re-records "
                  f"where it was read from")
    else:
        moved = [f"{fastpath.NVIDIA_STACK_NAMES[axis]} {before.get(axis)} -> "
                 f"{new['stack'][axis]}" for axis in fastpath.NVIDIA_STACK_AXES
                 if before.get(axis) != new["stack"][axis]]
        recorded_drivers = before.get(DRIVER_AXIS)
        added = [driver for driver in new["stack"][DRIVER_AXIS]
                 if not isinstance(recorded_drivers, list) or driver not in recorded_drivers]
        widens = (isinstance(recorded_drivers, list)
                  and all(before.get(axis) == new["stack"][axis]
                          for axis in fastpath.NVIDIA_STACK_AXES if axis != DRIVER_AXIS)
                  and set(recorded_drivers) <= set(new["stack"][DRIVER_AXIS]))
        if widens and not added:
            change = (f"keeps the certified stack of compute capability {found} and re-records "
                      f"where it was read from")
        elif widens:
            change = (f"adds CUDA driver {', '.join(added)} to the certified stack of compute "
                      f"capability {found}, beside {', '.join(recorded_drivers)}: the lanes "
                      f"that ran on it read the recorded value on every other axis")
        elif not restate:
            raise Refused(f"compute capability {found} is certified on another stack "
                          f"({'; '.join(moved)}): re-stating a certified capability's stack "
                          f"is a decision; pass --restate to record the stack these lanes "
                          f"ran beside")
        else:
            change = f"re-states the certified stack of compute capability {found}: " \
                     + "; ".join(moved)
    return {"capability": found, "record": new, "change": change, "lanes": len(read)}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--stack", action="append", required=True, type=Path, metavar="DIR",
                        help="a stage's stack directory (repeatable): results/chunk_<stamp>/"
                             "stack for a chunk, results/stack_<stamp> unchunked")
    parser.add_argument("--capability", default=None,
                        help="the compute capability the lanes must have run on")
    parser.add_argument("--restate", action="store_true",
                        help="record exactly the lanes' stack over a different recorded "
                             "one, dropping any recorded driver no lane ran on")
    parser.add_argument("--write", action="store_true", help="apply (default: report)")
    parser.add_argument("--root", type=Path, default=_ROOT, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    path = args.root / LEDGER
    try:
        text = path.read_text(encoding="utf-8")
        ledger = json.loads(text)
        if json.dumps(ledger, **LEDGER_FORMAT) + "\n" != text:
            raise Refused(f"{LEDGER} does not round-trip through {LEDGER_FORMAT}; writing it "
                          f"would rewrite bytes this tool does not own")
        entry = ledger.get(fastpath.CERTIFIED_STACK_ENTRY)
        records = entry.get(fastpath.STACK_BY_CAPABILITY) if isinstance(entry, dict) else None
        if not isinstance(records, dict):
            raise Refused(f"{LEDGER} holds no {fastpath.CERTIFIED_STACK_ENTRY}."
                          f"{fastpath.STACK_BY_CAPABILITY} record to add to")
        lanes = read_lanes(args.stack, args.root.resolve())
        capability = None
        if lanes:
            capability = lane_stack(*lanes[0])[0]
        outcome = judge(lanes, args.capability, records.get(capability), args.restate)
    except Refused as exc:
        print(f"REFUSED: {exc}", flush=True)
        return 1
    found = outcome["capability"]
    for axis in fastpath.NVIDIA_STACK_AXES:
        print(f"  {fastpath.NVIDIA_STACK_NAMES[axis]}: "
              f"{', '.join(outcome['record']['stack'][axis])}", flush=True)
    for driver, stamps in outcome["record"][STAMPS_BY_DRIVER].items():
        print(f"  CUDA driver {driver} ran in: {', '.join(stamps)}", flush=True)
    print(f"{outcome['lanes']} lane record(s) agree on compute capability {found}; the record "
          f"{outcome['change']}", flush=True)
    if not args.write:
        print(f"REPORT ONLY: {LEDGER} {fastpath.CERTIFIED_STACK_ENTRY}."
              f"{fastpath.STACK_BY_CAPABILITY}[{found!r}] would be written; add --write",
              flush=True)
        return 0
    records[found] = outcome["record"]
    entry[fastpath.STACK_BY_CAPABILITY] = {key: records[key] for key in sorted(records)}
    path.write_text(json.dumps(ledger, **LEDGER_FORMAT) + "\n", encoding="utf-8")
    print(f"WROTE {LEDGER} {fastpath.CERTIFIED_STACK_ENTRY}."
          f"{fastpath.STACK_BY_CAPABILITY}[{found!r}]", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
