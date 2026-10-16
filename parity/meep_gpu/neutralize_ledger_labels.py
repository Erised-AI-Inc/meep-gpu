"""Write the neutral labels into the machine stamps an earlier bind left in the ledgers.

    python neutralize_ledger_labels.py                       # report (from parity/meep_gpu)
    python neutralize_ledger_labels.py --write
    python neutralize_ledger_labels.py --write --staging-root DIR

Since owner ruling R22 every bind and record tool writes the neutral labels of
``neutral_labels.py``. A record a round writes replaces the one before it, but an
entry no round re-binds keeps what its bind wrote: the dated certification blocks
of ``cuda_kernels/certification.json`` accumulate, one per campaign, and a block cut
before the tools were fixed names its machine. This tool rewrites those LABEL leaves
in place, after the round's binds, so the guard's ledger leg
(``tools/ci/check_hygiene.py --legs ledgers``) reads clean without a hand edit.

WHAT IT REWRITES, and nothing else:

* a ``host`` or ``hostname`` value written as a raw machine stamp (one word alone,
  before ``:``, or before ``(machine-stamped`` or ``(NOT machine-stamped``): the word
  becomes ``the GPU host``, the rest of the value is kept;
* every whole-word occurrence of one of those words in another string leaf (the
  ``records`` lines that quote a manifest's machine): the same label;
* a path under the staging root in an ``environment_at_start`` block:
  ``<staging-root>/...`` (``neutral_labels.neutral_environment``; ``--staging-root``,
  else the directory the block's local paths share).

WHAT IT PROVES before anything is written (any failure refuses the whole run, exit 1):
the document's shape is unchanged (the same keys in the same order, the same list
lengths, the same type at every leaf); only string leaves under ``host``,
``hostname``, ``records`` and an ``environment_at_start`` block changed, none under a
``*sha256*`` key; every entry's ``fastpath.bound_digest`` and
``fastpath.live_capabilities`` are unchanged; each file still serialises the way its
writer writes it; and the guard's ledger leg finds nothing in the result. The raw
stamps stay in the artifacts under ``results/``, which never ship.

Exit: 0 done (or nothing to do), 1 refused, 2 usage.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
for _path in (str(_ROOT), str(_HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import neutral_labels  # noqa: E402
from meep_gpu import fastpath  # noqa: E402

LEDGERS = ("meep_gpu/triton_kernels/fingerprints.json",
           "meep_gpu/cuda_kernels/fingerprints.json",
           "meep_gpu/cuda_kernels/certification.json",
           "meep_gpu/metal_kernels/fingerprints.json")
#: The writers' serialisations, tried in order: the file must round-trip through one.
FORMATS = ({"indent": 2, "sort_keys": True}, {"indent": 2, "ensure_ascii": False},
           {"indent": 2, "sort_keys": True, "ensure_ascii": False}, {"indent": 1},
           {"indent": 2})
HOST_KEYS = ("host", "hostname")
LABEL_KEYS = HOST_KEYS + ("records",)
ENVIRONMENT_KEY = "environment_at_start"
#: A raw machine stamp: one word alone, or before a colon or a machine-stamp note.
_STAMP = re.compile(r"^(?P<word>[A-Za-z0-9._-]+)(?P<rest>\s*$|\s*:.*|\s+\((?:NOT )?machine-stamped.*)$",
                    re.S)


class Refused(Exception):
    """The rewrite cannot be shown to touch labels alone; nothing is written."""


def _format_of(text: str, document: Any) -> Dict[str, Any]:
    for kwargs in FORMATS:
        if json.dumps(document, **kwargs) + "\n" == text:
            return kwargs
    raise Refused("the file does not round-trip through any writer's serialisation, so "
                  "rewriting it would change bytes that are not labels")


def stamped_words(document: Any) -> Set[str]:
    """The raw machine words the document's host fields carry."""
    words: Set[str] = set()

    def walk(value: Any, key: Optional[str]) -> None:
        if isinstance(value, dict):
            for inner_key, inner in value.items():
                walk(inner, str(inner_key))
        elif isinstance(value, list):
            for inner in value:
                walk(inner, key)
        elif isinstance(value, str) and key in HOST_KEYS:
            match = _STAMP.match(value.strip())
            if match and not value.startswith(neutral_labels.GPU_HOST):
                words.add(match.group("word"))

    walk(document, None)
    return words


def rewrite(document: Any, words: Set[str], staging_root: Optional[str]
            ) -> Tuple[Any, List[str], List[str]]:
    """``(document, changed key paths, problems)`` with every label leaf neutral."""
    out = copy.deepcopy(document)
    changed: List[str] = []
    problems: List[str] = []

    def label(value: str, key: Optional[str]) -> str:
        if key in HOST_KEYS:
            match = _STAMP.match(value.strip())
            if match and match.group("word") in words:
                value = neutral_labels.GPU_HOST + match.group("rest")
        return neutral_labels.neutral_text(value, words) if key in LABEL_KEYS else value

    def walk(value: Any, path: str, key: Optional[str]) -> Any:
        if isinstance(value, dict):
            if key == ENVIRONMENT_KEY:
                neutral, _root, residue = neutral_labels.neutral_environment(value, staging_root)
                problems.extend(f"{path}.{name}: still a local path after the staging root "
                                f"is written as {neutral_labels.STAGING_ROOT}; pass "
                                f"--staging-root" for name in residue)
                for name in value:
                    if neutral[name] != value[name]:
                        changed.append(f"{path}.{name}")
                return neutral
            return {k: walk(v, f"{path}.{k}" if path else str(k), str(k))
                    for k, v in value.items()}
        if isinstance(value, list):
            return [walk(v, f"{path}[{i}]", key) for i, v in enumerate(value)]
        if isinstance(value, str):
            new = label(value, key)
            if new != value:
                changed.append(path)
            return new
        return value

    out = walk(out, "", None)
    return out, changed, problems


def prove(before: Any, after: Any, changed: List[str]) -> List[str]:
    """Why ``after`` is not ``before`` with label leaves alone rewritten."""
    problems: List[str] = []

    def shape(a: Any, b: Any, path: str) -> None:
        if type(a) is not type(b):
            problems.append(f"{path}: a {type(a).__name__} became a {type(b).__name__}")
        elif isinstance(a, dict):
            if list(a) != list(b):
                problems.append(f"{path}: the keys changed")
            else:
                for k in a:
                    shape(a[k], b[k], f"{path}.{k}" if path else str(k))
        elif isinstance(a, list):
            if len(a) != len(b):
                problems.append(f"{path}: the list length changed")
            else:
                for i, (x, y) in enumerate(zip(a, b)):
                    shape(x, y, f"{path}[{i}]")
        elif a != b and not isinstance(a, str):
            problems.append(f"{path}: a non-string leaf changed")

    shape(before, after, "")
    for path in changed:
        parts = re.split(r"\.|\[", path)
        if any("sha256" in part for part in parts):
            problems.append(f"{path}: lies under a digest")
        if any(part in fastpath.BOUND_FIELDS for part in parts):
            problems.append(f"{path}: lies under a bound field")
        if not (parts[-1].rstrip("]") in LABEL_KEYS or ENVIRONMENT_KEY in parts
                or (len(parts) > 1 and parts[-2] in LABEL_KEYS)):
            problems.append(f"{path}: is not a label leaf")
    if isinstance(before, dict):
        for name, entry in before.items():
            if not isinstance(entry, dict) or not isinstance(entry.get(fastpath.RUNS), dict):
                continue
            try:
                same = fastpath.bound_digest(entry) == fastpath.bound_digest(after[name])
            except fastpath.CapabilityRecordError:
                continue
            if not same or (fastpath.live_capabilities(entry)
                            != fastpath.live_capabilities(after[name])):
                problems.append(f"{name}: its bound digest or live capabilities moved")
    return problems


def _guard():
    path = _ROOT / "tools" / "ci" / "check_hygiene.py"
    spec = importlib.util.spec_from_file_location("check_hygiene", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--write", action="store_true", help="apply (default: report)")
    parser.add_argument("--staging-root", default=None, metavar="DIR",
                        help="the directory a round was staged in, for recorded paths")
    parser.add_argument("--root", type=Path, default=_ROOT, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    planned: Dict[str, Tuple[str, Any]] = {}
    refusals: List[str] = []
    for name in LEDGERS:
        path = args.root / name
        text = path.read_text(encoding="utf-8")
        before = json.loads(text)
        try:
            kwargs = _format_of(text, before)
        except Refused as exc:
            refusals.append(f"{name}: {exc}")
            continue
        words = stamped_words(before)
        after, changed, problems = rewrite(before, words, args.staging_root)
        problems += prove(before, after, changed)
        refusals.extend(f"{name}: {problem}" for problem in problems)
        print(f"  {name}: {len(changed)} label leaf(s) to rewrite, {len(words)} raw "
              f"machine word(s)", flush=True)
        if changed:
            planned[name] = (json.dumps(after, **kwargs) + "\n", after)
    if not refusals and planned:
        guard = _guard()
        rules = guard.load_rules(args.root / "tools" / "ci" / "hygiene_rules.json")
        cache: Dict[str, str] = {}
        for name, (_text, after) in planned.items():
            for where, value, key in guard._walk(after, ""):
                if guard.scan_text(rules, value, cache) or (
                        key in guard._HOST_KEYS and guard._HOST_STAMP.match(value)):
                    refusals.append(f"{name}:{where}: the guard still finds a hit")
    for refusal in refusals:
        print(f"  REFUSED   {refusal}", flush=True)
    if refusals:
        print("  nothing written", flush=True)
        return 1
    if not planned:
        print("  nothing to rewrite", flush=True)
        return 0
    if not args.write:
        print(f"  would rewrite {sorted(planned)} (report mode; --write applies)", flush=True)
        return 0
    for name, (text, _after) in planned.items():
        (args.root / name).write_text(text, encoding="utf-8")
        if (args.root / name).read_text(encoding="utf-8") != text:
            print(f"  REFUSED   {name} did not read back as written", flush=True)
            return 1
        print(f"  wrote {name}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
