#!/usr/bin/env python
"""The hygiene guard: no host name, account name, local home path or personal e-mail
address in what this repository tracks, in its certification ledgers, or in the
coverage board's subject.

    python tools/ci/check_hygiene.py [--root DIR] [--legs tracked,subject,ledgers]
                                     [--names-file FILE] [--json OUT]
    python tools/ci/check_hygiene.py --patch CHANGE.patch [--names-file FILE]
    python tools/ci/check_hygiene.py digest --class host_name TEXT [TEXT ...]

Exit 0 when every leg asked for is clean, 1 on any hit, 2 when a leg cannot be read.

THE LEGS.

* ``tracked``: every file ``git ls-files`` lists under ``--root`` (a directory walk
  when the root is not a git work tree), except the four ledgers, which are a leg of
  their own. A file holding a NUL byte is binary: only its printable runs of 8 or more
  characters are read.
* ``ledgers``: :data:`LEDGERS`, parsed as JSON. Every key and every string value is
  read, and a hit names its key path. These files ship inside the wheel and dispatch
  reads them, so a leak here cannot be scrubbed at export: the tools that write them
  write neutral labels (``parity/meep_gpu/neutral_labels.py``).
* ``subject``: ``build_parity_board.board_subject_files`` (the coverage board's
  subject), minus the four ledgers. The board digests these bytes, so they are read
  where the board reads them, tracked or not.
* ``--patch``: only the lines a unified diff ADDS, by the file and line they land on.
  This is the form that fails the change that introduces a leak, whatever its base
  carries.

THE RULES, one per class. The patterns name no machine, account or person: they are
built from neutral pieces, and the concrete names are held as digests.

* ``host_name`` and ``account_name``: a word of the text, lower-cased, whose salted
  sha256 is listed in ``hygiene_rules.json``. The words are the runs of letters (digits
  and punctuation separate them) and the parts of a camel-cased run, so a name inside
  ``status_name``, ``nameSuffix`` or ``/dir/name/`` is found and one fused into a longer
  run of lower-case letters is not. A listed name of several words is matched on
  consecutive runs. ``--names-file`` adds names from an untracked local file (JSON:
  ``{"host_name": [...], "account_name": [...]}``); ``digest`` prints the entries to
  list a new name without writing it into a tracked file.
* ``home_path``: an absolute path under a home root (``/home/<account>``,
  ``/Users/<account>``) or a macOS temporary root (``/private/var/<dir>``,
  ``/private/tmp/<dir>``). A path must start the token (``x/home/y`` is a relative path),
  and must name something under the root.
* ``email``: an address shape whose local part does not follow a backslash (so the
  ``\\n@triton.jit`` of a source string is not one), other than the allowed ones.
* ``host_stamp`` (ledgers): a ``host`` or ``hostname`` value written as a raw machine
  stamp, one bare word alone, before a colon, or before ``(machine-stamped``. The
  neutral labels (``the GPU host``, Metal's ``this machine: ...``) are phrases, so
  this catches a host this file has no digest for.

THE ALLOWLIST is ``hygiene_rules.json``'s ``allow`` list. Every entry names its rule
and carries its reason; nothing is allowed by path alone.

WHAT IS PRINTED. A hit's matched text is masked by its class (a public CI log is
public): ``LEG path:LINE: CLASS (RULE): <context>``, then a count per leg and class.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = HERE.parent.parent
DEFAULT_RULES = HERE / "hygiene_rules.json"

LEDGERS = ("meep_gpu/triton_kernels/fingerprints.json",
           "meep_gpu/cuda_kernels/fingerprints.json",
           "meep_gpu/cuda_kernels/certification.json",
           "meep_gpu/metal_kernels/fingerprints.json")
LEGS = ("tracked", "subject", "ledgers")
NAME_CLASSES = ("host_name", "account_name")
CLASSES = NAME_CLASSES + ("home_path", "email", "host_stamp")

#: Runs of letters, and the parts of a camel-cased run.
_RUN = re.compile(r"[A-Za-z]+")
_CAMEL = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+")
#: Home roots and temporary roots, from neutral pieces.
_HOME_ROOTS = ("ho" + "me", "Us" + "ers")
_TMP_ROOTS = ("var", "tmp")
_PATH_CHAR = r"A-Za-z0-9._\-"
_HOME = re.compile(rf"(?<![{_PATH_CHAR}~])/(?P<root>{'|'.join(_HOME_ROOTS)})/"
                   rf"(?P<account>[{_PATH_CHAR}]+)")
_TMP = re.compile(rf"(?<![{_PATH_CHAR}~])/priv" + rf"ate/(?:{'|'.join(_TMP_ROOTS)})/"
                  rf"[{_PATH_CHAR}]")
_EMAIL = re.compile(r"(?<![A-Za-z0-9._%+\\-])[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+"
                    r"(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")
_HOST_KEYS = ("host", "hostname")
_HOST_STAMP = re.compile(r"^\s*[A-Za-z0-9._-]+(?:\s*$|\s*:|\s+\(machine-stamped)")
_MIN_BINARY_RUN = 8
_PRINTABLE = re.compile(rb"[\x20-\x7e]{%d,}" % _MIN_BINARY_RUN)


def digest(salt: str, words: Sequence[str]) -> str:
    """The listed form of a name: sha256 over the salt and its lower-cased words."""
    text = salt + " ".join(word.lower() for word in words)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def words_of(text: str) -> List[str]:
    """The runs of letters of ``text``: the words a listed name is made of."""
    return _RUN.findall(text)


@dataclass(frozen=True)
class Hit:
    leg: str
    path: str
    where: str  # a line number, or a ledger key path
    klass: str
    rule: str
    context: str

    def line(self) -> str:
        return f"{self.leg} {self.path}:{self.where}: {self.klass} ({self.rule}): {self.context}"


@dataclass
class Rules:
    salt: str
    names: Dict[str, str] = field(default_factory=dict)      # digest -> class
    phrases: Dict[str, Tuple[str, int]] = field(default_factory=dict)  # digest -> (class, n)
    leads: Set[str] = field(default_factory=set)              # digest of a phrase's first word
    allowed_addresses: Set[str] = field(default_factory=set)  # sha256 of lower-cased address
    allowed_address_literals: Set[str] = field(default_factory=set)
    allowed_domain_suffixes: Tuple[str, ...] = ()
    allowed_home_accounts: Set[str] = field(default_factory=set)

    def add_name(self, klass: str, words: Sequence[str]) -> None:
        if klass not in NAME_CLASSES:
            raise ValueError(f"unknown name class {klass!r}; one of {NAME_CLASSES}")
        if not words:
            raise ValueError("a name needs at least one word of letters")
        if len(words) == 1:
            self.names[digest(self.salt, words)] = klass
        else:
            self.phrases[digest(self.salt, words)] = (klass, len(words))
            self.leads.add(digest(self.salt, words[:1]))

    def add_entry(self, entry: dict) -> None:
        klass, words = entry["class"], int(entry.get("words", 1))
        if klass not in NAME_CLASSES:
            raise ValueError(f"unknown name class {klass!r}")
        if not entry.get("reason"):
            raise ValueError(f"a listed name carries no reason: {entry}")
        if words == 1:
            self.names[entry["sha256"]] = klass
        else:
            self.phrases[entry["sha256"]] = (klass, words)
            self.leads.add(entry["lead_sha256"])


def load_rules(path: Path = DEFAULT_RULES, names_file: Optional[Path] = None) -> Rules:
    document = json.loads(path.read_text(encoding="utf-8"))
    rules = Rules(salt=document["salt"])
    for entry in document["names"]:
        rules.add_entry(entry)
    suffixes = []
    for entry in document["allow"]:
        if not entry.get("reason") or not entry.get("rule"):
            raise ValueError(f"an allowlist entry carries no rule or no reason: {entry}")
        rule = entry["rule"]
        if rule == "email":
            rules.allowed_addresses.update(entry.get("address_sha256", ()))
            rules.allowed_address_literals.update(a.lower() for a in entry.get("addresses", ()))
            suffixes.extend(s.lower() for s in entry.get("domain_suffixes", ()))
        elif rule == "home_path":
            rules.allowed_home_accounts.update(entry.get("accounts", ()))
        else:
            raise ValueError(f"an allowlist entry names an unknown rule {rule!r}")
    rules.allowed_domain_suffixes = tuple(suffixes)
    if names_file is not None:
        local = json.loads(names_file.read_text(encoding="utf-8"))
        for klass in NAME_CLASSES:
            for name in local.get(klass, ()):
                rules.add_name(klass, words_of(name))
    return rules


# ------------------------------------------------------------------ matching one text
def _mask(text: str, spans: Sequence[Tuple[int, int, str]]) -> str:
    out, last = [], 0
    for start, end, klass in sorted(spans):
        if start < last:
            continue
        out.append(text[last:start])
        out.append(f"[{klass}]")
        last = end
    out.append(text[last:])
    return "".join(out)


def _address_allowed(rules: Rules, address: str) -> bool:
    low = address.lower()
    if low in rules.allowed_address_literals:
        return True
    if hashlib.sha256(low.encode("utf-8")).hexdigest() in rules.allowed_addresses:
        return True
    domain = low.rsplit("@", 1)[1]
    return any(domain == s.lstrip(".") or domain.endswith(s) for s in rules.allowed_domain_suffixes)


def scan_text(rules: Rules, text: str, cache: Dict[str, str]) -> List[Tuple[int, int, str, str]]:
    """Every hit in ``text`` as (start, end, class, rule), allowed forms removed."""
    hits: List[Tuple[int, int, str, str]] = []
    allowed_spans: List[Tuple[int, int]] = []
    for match in _EMAIL.finditer(text):
        if _address_allowed(rules, match.group()):
            allowed_spans.append(match.span())
        else:
            hits.append((match.start(), match.end(), "email", "email"))

    def inside_allowed(start: int, end: int) -> bool:
        return any(a <= start and end <= b for a, b in allowed_spans)

    def word_digest(word: str) -> str:
        low = word.lower()
        found = cache.get(low)
        if found is None:
            found = cache[low] = digest(rules.salt, [low])
        return found

    runs = list(_RUN.finditer(text))
    for index, run in enumerate(runs):
        word = run.group()
        candidates = [(run.start(), run.end(), word)]
        if not _plain_case(word):
            candidates += [(run.start() + part.start(), run.start() + part.end(), part.group())
                           for part in _CAMEL.finditer(word)]
        for start, end, piece in candidates:
            klass = rules.names.get(word_digest(piece))
            if klass and not inside_allowed(start, end):
                hits.append((start, end, klass, "listed_name"))
        if rules.leads and word_digest(word) in rules.leads:
            for phrase, (klass, count) in rules.phrases.items():
                window = runs[index:index + count]
                if len(window) == count and digest(rules.salt, [r.group() for r in window]) == phrase:
                    start, end = window[0].start(), window[-1].end()
                    if not inside_allowed(start, end):
                        hits.append((start, end, klass, "listed_name"))
    for match in _HOME.finditer(text):
        if match.group("account") in rules.allowed_home_accounts:
            continue
        hits.append((match.start(), match.end(), "home_path", "home_root"))
    for match in _TMP.finditer(text):
        hits.append((match.start(), match.end(), "home_path", "temporary_root"))
    return hits


def _plain_case(word: str) -> bool:
    """A run its camel parts would not split: all small, all capital, or Capitalised."""
    return word.islower() or word.isupper() or (word[:1].isupper() and word[1:].islower())


def _quick_reject(rules: Rules, text: str, cache: Dict[str, str]) -> bool:
    """True when ``text`` certainly holds no hit: no listed word, no address, no root."""
    if "@" in text and _EMAIL.search(text):
        return False
    if (any(f"/{root}/" in text for root in _HOME_ROOTS) and _HOME.search(text)) \
            or ("/priv" + "ate/" in text and _TMP.search(text)):
        return False
    for word in set(_RUN.findall(text)):
        pieces = [word] if _plain_case(word) else [word] + _CAMEL.findall(word)
        for piece in pieces:
            low = piece.lower()
            found = cache.get(low)
            if found is None:
                found = cache[low] = digest(rules.salt, [low])
            if found in rules.names or found in rules.leads:
                return False
    return True


def _hits_by_line(leg: str, path: str, rules: Rules, text: str,
                  cache: Dict[str, str], line_offset: Optional[Dict[int, int]] = None) -> List[Hit]:
    if _quick_reject(rules, text, cache):
        return []
    found: List[Hit] = []
    for number, line in enumerate(text.splitlines(), 1):
        spans = scan_text(rules, line, cache)
        if not spans:
            continue
        masked = _mask(line, [(s, e, k) for s, e, k, _ in spans]).strip()
        where = str(line_offset.get(number, number) if line_offset else number)
        for _start, _end, klass, rule in spans:
            found.append(Hit(leg, path, where, klass, rule, masked[:200]))
    return found


def read_text(path: Path) -> str:
    data = path.read_bytes()
    if b"\0" in data:
        return "\n".join(run.decode("ascii") for run in _PRINTABLE.findall(data))
    return data.decode("utf-8", errors="replace")


# ------------------------------------------------------------------------- the legs
def tracked_files(root: Path) -> List[str]:
    probe = subprocess.run(["git", "-C", str(root), "ls-files", "-z"],
                           capture_output=True)
    if probe.returncode == 0 and probe.stdout:
        return sorted(name for name in probe.stdout.decode("utf-8").split("\0") if name)
    skip = {".git", "__pycache__", ".pytest_cache", "results", "reports", "build", "site"}
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*")
                  if p.is_file() and not (set(p.relative_to(root).parts) & skip))


def leg_files(rules: Rules, root: Path, leg: str, names: Iterable[str]) -> List[Hit]:
    cache: Dict[str, str] = {}
    hits: List[Hit] = []
    for name in names:
        if name in LEDGERS:
            continue
        path = root / name
        if not path.is_file():
            continue
        hits.extend(_hits_by_line(leg, name, rules, read_text(path), cache))
    return hits


def subject_files(root: Path) -> List[str]:
    sys.path.insert(0, str(root / "parity" / "meep_gpu"))
    try:
        import build_parity_board  # noqa: PLC0415
        return list(build_parity_board.board_subject_files(root))
    finally:
        sys.path.pop(0)


def _walk(value, path: str) -> Iterator[Tuple[str, str, Optional[str]]]:
    """(key path, text, key) for every key and every string value of a JSON document."""
    if isinstance(value, dict):
        for key, inner in value.items():
            here = f"{path}.{key}" if path else str(key)
            yield here, str(key), None
            if isinstance(inner, str):
                yield here, inner, str(key)
            elif isinstance(inner, list) and all(isinstance(v, str) for v in inner):
                for index, item in enumerate(inner):
                    yield f"{here}[{index}]", item, str(key)
            else:
                yield from _walk(inner, here)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            here = f"{path}[{index}]"
            if isinstance(item, str):
                yield here, item, None
            else:
                yield from _walk(item, here)


def leg_ledgers(rules: Rules, root: Path) -> List[Hit]:
    cache: Dict[str, str] = {}
    hits: List[Hit] = []
    for name in LEDGERS:
        path = root / name
        document = json.loads(path.read_text(encoding="utf-8"))
        for where, text, key in _walk(document, ""):
            spans = scan_text(rules, text, cache)
            masked = _mask(text, [(s, e, k) for s, e, k, _ in spans]).strip()
            for _start, _end, klass, rule in spans:
                hits.append(Hit("ledgers", name, where, klass, rule, masked[:200]))
            if key in _HOST_KEYS and _HOST_STAMP.match(text):
                hits.append(Hit("ledgers", name, where, "host_stamp", "raw_machine_stamp",
                                "[a raw machine stamp: one bare word, before ':' or "
                                "'(machine-stamped']"))
    return hits


def added_lines(patch_text: str) -> Iterator[Tuple[str, int, str]]:
    """(file, new line number, text) for every line a unified diff adds."""
    path, number = None, 0
    for line in patch_text.splitlines():
        if line.startswith("+++ "):
            target = line[4:].strip()
            path = None if target == "/dev/null" else target[2:] if target.startswith("b/") else target
            continue
        if line.startswith("@@"):
            match = re.match(r"@@ -\d+(?:,\d+)? \+(\d+)", line)
            number = int(match.group(1)) if match else 0
            continue
        if path is None or line.startswith(("--- ", "diff --git", "index ")):
            continue
        if line.startswith("+"):
            yield path, number, line[1:]
            number += 1
        elif line.startswith(" "):
            number += 1


def leg_patch(rules: Rules, patch: Path) -> List[Hit]:
    cache: Dict[str, str] = {}
    hits: List[Hit] = []
    for name, number, text in added_lines(patch.read_text(encoding="utf-8", errors="replace")):
        for start, end, klass, rule in scan_text(rules, text, cache):
            masked = _mask(text, [(start, end, klass)]).strip()
            hits.append(Hit("patch", name, str(number), klass, rule, masked[:200]))
        if name in LEDGERS:
            for key in _HOST_KEYS:
                match = re.match(rf'^\s*"{key}":\s*"(.*)",?\s*$', text)
                if match and _HOST_STAMP.match(match.group(1)):
                    hits.append(Hit("patch", name, str(number), "host_stamp",
                                    "raw_machine_stamp", "[a raw machine stamp]"))
    return hits


def run_legs(rules: Rules, root: Path, legs: Sequence[str]) -> Tuple[List[Hit], List[str]]:
    hits: List[Hit] = []
    problems: List[str] = []
    for leg in legs:
        try:
            if leg == "tracked":
                hits.extend(leg_files(rules, root, leg, tracked_files(root)))
            elif leg == "subject":
                hits.extend(leg_files(rules, root, leg, subject_files(root)))
            elif leg == "ledgers":
                hits.extend(leg_ledgers(rules, root))
            else:
                problems.append(f"unknown leg {leg!r}; one of {LEGS}")
        except (OSError, ValueError, ImportError) as exc:
            problems.append(f"{leg}: cannot be read: {exc!r}")
    return hits, problems


def summary(hits: Sequence[Hit], legs: Sequence[str]) -> List[str]:
    lines = []
    for leg in legs:
        counts = {klass: 0 for klass in CLASSES}
        for hit in hits:
            if hit.leg == leg:
                counts[hit.klass] += 1
        total = sum(counts.values())
        lines.append(f"{leg}: {total} hit(s): "
                     + ", ".join(f"{klass} {n}" for klass, n in counts.items()))
    return lines


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["digest"]:
        parser = argparse.ArgumentParser(prog="check_hygiene.py digest")
        parser.add_argument("--class", dest="klass", required=True, choices=NAME_CLASSES)
        parser.add_argument("--rules", type=Path, default=DEFAULT_RULES)
        parser.add_argument("text", nargs="+")
        args = parser.parse_args(argv[1:])
        salt = json.loads(args.rules.read_text(encoding="utf-8"))["salt"]
        for text in args.text:
            words = words_of(text)
            entry = {"class": args.klass, "words": len(words), "sha256": digest(salt, words),
                     "reason": "<why this name may not appear>"}
            if len(words) > 1:
                entry["lead_sha256"] = digest(salt, words[:1])
            print(json.dumps(entry, sort_keys=True))
        return 0
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--legs", default=",".join(LEGS))
    parser.add_argument("--rules", type=Path, default=DEFAULT_RULES)
    parser.add_argument("--names-file", type=Path, default=None,
                        help="an untracked JSON file of further names, by class")
    parser.add_argument("--patch", type=Path, default=None,
                        help="read only the lines this unified diff adds")
    parser.add_argument("--json", type=Path, default=None, help="write every hit here")
    args = parser.parse_args(argv)
    rules = load_rules(args.rules, args.names_file)
    if args.patch is not None:
        legs, problems = ["patch"], []
        hits = leg_patch(rules, args.patch)
    else:
        legs = [leg.strip() for leg in args.legs.split(",") if leg.strip()]
        hits, problems = run_legs(rules, args.root.resolve(), legs)
    for hit in hits:
        print(hit.line(), flush=True)
    for line in summary(hits, legs):
        print(line, flush=True)
    for problem in problems:
        print(f"UNREADABLE {problem}", flush=True)
    if args.json is not None:
        args.json.write_text(json.dumps({"legs": legs, "problems": problems,
                                         "hits": [hit.__dict__ for hit in hits]},
                                        indent=1) + "\n", encoding="utf-8")
    if problems:
        return 2
    print("CLEAN" if not hits else f"{len(hits)} HIT(S)", flush=True)
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main())
