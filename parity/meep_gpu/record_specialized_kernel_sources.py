"""Re-record a module's digest in the Triton ledger's specialization index.

    python record_specialized_kernel_sources.py --module no_pml.py
    python record_specialized_kernel_sources.py --module no_pml.py --write

WHAT IT WRITES. ``specialized_kernel_sources[<module>].sha256`` in
``meep_gpu/triton_kernels/fingerprints.json``: the index of which kernel each
specialised Triton module holds, and the sha256 of that module. The module tests
(``test_triton_no_pml.py``, ``test_triton_conductivity.py``) hold the digest against
the file in the tree. The index is not a device run: no gate produces it, and no
rebind or recut tool writes it.

THE RULE. The digest is derived from the welds that ran the module, never typed.
A module's digest is written only when BOTH of its welds, named in
:data:`WELDS_OF_MODULE`, certify the module's live bytes:

* the STANDALONE device weld (``triton_<family>_device_gate``), and
* the COMPOSITION weld (``<family>_composition_gate``), the record the module test
  reads beside the index.

Each must have at least one live per-capability record
(:func:`fastpath.live_capabilities`), and each must pin the module's live sha256 in
its ``source_sha256`` map, under either spelling the ledger uses: the repository
path (``meep_gpu/triton_kernels/<module>``) or the bare file name, which resolves to
``triton_kernels/`` first. Where a weld carries both spellings, both must agree.
A module whose standalone weld still pins older bytes waits for that weld's device
re-run and rebind; deriving its digest from the composition weld alone would
certify the standalone product from a run that did not execute it.

The table is typed rather than derived from the file name, as
``rebind_triton_welds.CAMPAIGN_DIRS`` is: a name rule that lands on the wrong weld
would derive the digest from a run of something else, and nothing downstream could
tell. A module without a row is refused by name.

WHAT ELSE IT CHECKS. The index entry keeps its fields (``kernel``, ``sha256``), and
its ``kernel`` must still be defined in the module (``def <kernel>(``). The ledger
must round-trip byte for byte through the binders' serialisation
(``json.dumps(indent=2, sort_keys=True)`` plus a newline). After the change, in
memory, only the requested ``sha256`` leaves may differ, and every entry's
:func:`fastpath.bound_digest` and :func:`fastpath.live_capabilities` must be
unchanged: the index is not a weld and carries no per-capability record. A request
naming several modules is all or nothing: one refusal writes none of them.

It reports by default and writes only with ``--write``.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

_HERE = Path(__file__).resolve().parent


def _find_api_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "parity" / "meep_gpu").is_dir() and (candidate / "meep_gpu").is_dir():
            return candidate
    raise SystemExit(f"cannot locate the repository root above {start}")


_API = _find_api_root(_HERE)
if str(_API) not in sys.path:
    sys.path.insert(0, str(_API))

from meep_gpu import fastpath  # noqa: E402

#: The ledger and the modules' directory, relative to the repository root.
LEDGER_RELATIVE = Path("meep_gpu") / "triton_kernels" / "fingerprints.json"
MODULE_DIR = Path("meep_gpu") / "triton_kernels"

#: The index this tool writes into, and the fields each of its entries has.
INDEX_KEY = "specialized_kernel_sources"
INDEX_FIELDS = ("kernel", "sha256")

#: Module -> (standalone device weld, composition weld). Typed, not derived.
WELDS_OF_MODULE: Dict[str, Tuple[str, str]] = {
    "no_pml.py": ("triton_no_pml_device_gate", "no_pml_composition_gate"),
    "conductivity.py": ("triton_conductivity_device_gate", "conductivity_composition_gate"),
    "cylindrical_triton.py": ("triton_cylindrical_device_gate", "cylindrical_composition_gate"),
}


class Refusal(Exception):
    """A named reason not to write. Nothing has been written when it is raised."""


def canonical(ledger: Mapping[str, Any]) -> str:
    """The binders' serialisation of a ledger."""
    return json.dumps(ledger, indent=2, sort_keys=True) + "\n"


def leaves(value: Any, prefix: Tuple[str, ...] = ()) -> Dict[Tuple[str, ...], Any]:
    """Every scalar of a JSON document, keyed by its path."""
    if isinstance(value, Mapping):
        out: Dict[Tuple[str, ...], Any] = {}
        if not value:
            out[prefix] = {}
        for key, item in value.items():
            out.update(leaves(item, prefix + (str(key),)))
        return out
    if isinstance(value, list):
        out = {}
        if not value:
            out[prefix] = []
        for index, item in enumerate(value):
            out.update(leaves(item, prefix + (f"[{index}]",)))
        return out
    return {prefix: value}


def changed_leaves(before: Mapping[str, Any], after: Mapping[str, Any]) -> List[Tuple[str, ...]]:
    old, new = leaves(before), leaves(after)
    return sorted(path for path in set(old) | set(new) if old.get(path, ...) != new.get(path, ...))


def entry_bindings(ledger: Mapping[str, Any]) -> Dict[str, Tuple[Optional[str], Tuple[str, ...]]]:
    """Per top-level entry: its bound digest (None if it binds nothing) and live capabilities."""
    out: Dict[str, Tuple[Optional[str], Tuple[str, ...]]] = {}
    for key, entry in ledger.items():
        if not isinstance(entry, Mapping):
            continue
        try:
            bound: Optional[str] = fastpath.bound_digest(entry)
        except fastpath.CapabilityRecordError:
            bound = None
        out[key] = (bound, fastpath.live_capabilities(entry))
    return out


def load_ledger(ledger_path: Path) -> Dict[str, Any]:
    """The ledger; refused unless its text is the canonical form."""
    text = ledger_path.read_text(encoding="utf-8")
    ledger = json.loads(text)
    if canonical(ledger) != text:
        raise Refusal(f"{ledger_path} does not round-trip through json.dumps(indent=2, "
                      "sort_keys=True); writing it would reformat more than the index")
    return ledger


def module_pins(entry: Mapping[str, Any], module: str) -> Dict[str, Any]:
    """The weld's pins of ``module``, under both spellings the ledger uses."""
    sources = entry.get("source_sha256")
    if not isinstance(sources, Mapping):
        return {}
    spellings = ((MODULE_DIR / module).as_posix(), module)
    return {spelling: sources[spelling] for spelling in spellings if spelling in sources}


def weld_reasons(ledger: Mapping[str, Any], role: str, key: str, module: str,
                 live: str) -> Tuple[List[str], Tuple[str, ...]]:
    """Why ``key`` does not certify the live ``module`` (empty: it does), and its live capabilities."""
    entry = ledger.get(key)
    if not isinstance(entry, Mapping):
        return [f"the {role} weld {key} is not in the ledger"], ()
    capabilities = fastpath.live_capabilities(entry)
    reasons = []
    if not capabilities:
        reasons.append(f"the {role} weld {key} has no live per-capability record")
    pins = module_pins(entry, module)
    if not pins:
        reasons.append(f"the {role} weld {key} does not pin {module}")
    for spelling, digest in sorted(pins.items()):
        if digest != live:
            reasons.append(f"the {role} weld {key} pins {spelling} at {str(digest)[:12]}, "
                           f"not the live {live[:12]}")
    return reasons, capabilities


def derive(api_root: Path, ledger: Mapping[str, Any], module: str) -> Dict[str, Any]:
    """The verdict for one module: the live digest and its welds' evidence, or the reasons not."""
    if module not in WELDS_OF_MODULE:
        raise Refusal(f"{module} has no row in WELDS_OF_MODULE (rows: "
                      f"{sorted(WELDS_OF_MODULE)}); name its standalone and composition "
                      "welds there before its digest can be derived")
    index = ledger.get(INDEX_KEY)
    if not isinstance(index, Mapping) or not isinstance(index.get(module), Mapping):
        raise Refusal(f"the ledger has no {INDEX_KEY}[{module!r}] entry; this tool does not "
                      "add one")
    item = index[module]
    if set(item) != set(INDEX_FIELDS):
        raise Refusal(f"{INDEX_KEY}[{module!r}] has fields {sorted(item)}, expected "
                      f"{sorted(INDEX_FIELDS)}; this tool does not reshape it")
    path = api_root / MODULE_DIR / module
    if not path.is_file():
        raise Refusal(f"{(MODULE_DIR / module).as_posix()} is not in the tree")
    source = path.read_bytes()
    live = hashlib.sha256(source).hexdigest()
    reasons = []
    kernel = item["kernel"]
    if not re.search(rf"^\s*def {re.escape(str(kernel))}\s*\(",
                     source.decode("utf-8"), flags=re.MULTILINE):
        reasons.append(f"{module} no longer defines the indexed kernel {kernel!r}")
    standalone, composition = WELDS_OF_MODULE[module]
    evidence = {}
    for role, key in (("standalone", standalone), ("composition", composition)):
        found, capabilities = weld_reasons(ledger, role, key, module, live)
        reasons += found
        evidence[key] = list(capabilities)
    return {"module": module, "kernel": kernel, "before": item["sha256"], "after": live,
            "live_capabilities": evidence, "reasons": reasons}


def record(api_root: Path = _API, *, modules: Sequence[str], write: bool = False) -> Dict[str, Any]:
    """Derive each module's digest and (with ``write``) move its index leaf. Returns a report."""
    if not modules:
        raise Refusal("name at least one module")
    ledger_path = api_root / LEDGER_RELATIVE
    ledger = load_ledger(ledger_path)
    verdicts = [derive(api_root, ledger, module) for module in dict.fromkeys(modules)]
    report: Dict[str, Any] = {"ledger": str(ledger_path), "modules": verdicts, "written": False}
    refused = [verdict for verdict in verdicts if verdict["reasons"]]
    if refused:
        raise Refusal("; ".join(f"{verdict['module']}: " + "; ".join(verdict["reasons"])
                                for verdict in refused))
    changed = copy.deepcopy(ledger)
    intended = []
    for verdict in verdicts:
        if verdict["before"] != verdict["after"]:
            changed[INDEX_KEY][verdict["module"]]["sha256"] = verdict["after"]
            intended.append((INDEX_KEY, verdict["module"], "sha256"))
    diff = changed_leaves(ledger, changed)
    if diff != sorted(intended):
        raise Refusal(f"the change would move {diff}, not {sorted(intended)}")
    if entry_bindings(changed) != entry_bindings(ledger):
        raise Refusal("the change would move a bound digest or a live capability")
    report["changed"] = [".".join(path) for path in diff]
    if diff and write:
        ledger_path.write_text(canonical(changed), encoding="utf-8")
        if ledger_path.read_text(encoding="utf-8") != canonical(changed):
            raise Refusal(f"{ledger_path} does not read back as written")
        report["written"] = True
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--module", action="append", required=True,
                        help="a module of meep_gpu/triton_kernels/ (repeatable)")
    parser.add_argument("--write", action="store_true",
                        help="write the index (default: report only)")
    args = parser.parse_args(argv)
    try:
        report = record(modules=args.module, write=args.write)
    except Refusal as exc:
        print(f"REFUSED: {exc}", flush=True)
        return 2
    for verdict in report["modules"]:
        welds = ", ".join(f"{key} live {tuple(caps)}"
                          for key, caps in verdict["live_capabilities"].items())
        state = ("already current" if verdict["before"] == verdict["after"]
                 else f"{verdict['before']} -> {verdict['after']}")
        print(f"{INDEX_KEY}[{verdict['module']}] ({verdict['kernel']}): {state}; "
              f"both welds pin the live bytes ({welds})", flush=True)
    if not report["changed"]:
        print("nothing to write", flush=True)
        return 0
    count = len(report["changed"])
    print(f"{count} {'leaf changes' if count == 1 else 'leaves change'}: {report['changed']}; "
          "every entry's bound digest and live capabilities are unchanged", flush=True)
    print(f"written: {report['ledger']}" if report["written"]
          else "report only; pass --write to write", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
