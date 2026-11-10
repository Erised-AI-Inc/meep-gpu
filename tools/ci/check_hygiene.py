#!/usr/bin/env python
"""The hygiene guard: no host name, account name, local home path, personal e-mail
address or private evidence path in what this repository tracks, in its certification
ledgers, or in the coverage board's subject; and no development jargon (a citation of a
private design document, a work-item label, a ruling number) in what it exports.

    python tools/ci/check_hygiene.py [--root DIR] [--legs tracked,subject,ledgers,ledger_states]
                                     [--names-file FILE] [--json OUT]
    python tools/ci/check_hygiene.py --legs records --records STORE [--records STORE ...]
    python tools/ci/check_hygiene.py --patch CHANGE.patch [--names-file FILE]
    python tools/ci/check_hygiene.py digest --class host_name TEXT [TEXT ...]

Exit 0 when every leg asked for is clean, 1 on any hit, 2 when a leg cannot be read.

THE LEGS.

* ``tracked``: every file ``git ls-files`` lists under ``--root`` (a directory walk
  when the root is not a git work tree), except the four ledgers, which are a leg of
  their own. A file holding a NUL byte is binary: only its printable runs of 8 or more
  characters are read. ``dev_jargon`` reads the exported files only: it skips the files
  ``tools/release/export_manifest.json`` excludes from the public tree.
* ``ledgers``: :data:`LEDGERS`, parsed as JSON. Every key and every string value is
  read, and a hit names its key path. These files ship inside the wheel and dispatch
  reads them, so a leak here cannot be scrubbed at export: the tools that write them
  write neutral labels (``parity/meep_gpu/neutral_labels.py``).
* ``ledger_states``: every archived ledger state (``parity/meep_gpu/ledger_states/<table>/
  <sha256>.json.gz``), decompressed and read as the ``ledgers`` leg reads that table's
  ledger, under its path, so a pending entry applies and ``dev_jargon`` does not. Gzip hides
  these bytes from ``tracked``, and neutralizing one would change its digest, so a state
  that fails here is left out of the archive: its rows compare whole.
* ``subject``: ``build_parity_board.board_subject_files`` (the coverage board's
  subject), minus the four ledgers. The board digests these bytes, so they are read
  where the board reads them, tracked or not.
* ``records`` (asked for by name, never by default): the timing row records of each
  ``--records`` store (``parity/meep_gpu/timing_rows.py``: every ``*.json`` below it whose
  schema is a timing row), parsed and read as the ledgers are, every key and string value,
  a hit naming its key path. A row record leaves the machine that timed it, so its writers
  spell a path under its root's label. The rules are the ledgers' (a record's ``host`` is
  the neutral ``WSL2`` of its Windows side, which the host stamp admits), and one more
  class: ``local_path``, every string the records' own check
  (``timing_rows.local_paths``) finds that no other rule did: an absolute path outside the
  system directories (a project or scratch filesystem, a mount, ``/tmp/<dir>``), a Windows
  profile, a path under ``~/``, a card's UUID or this machine's host name. A store with no
  row record cannot be read: the leg never passes on nothing. The bench line a fused row
  keeps verbatim beside its record (``*.rows.jsonl``) is the bench's own output and is not
  read.
* ``--patch``: only the lines a unified diff ADDS, by the file and line they land on.
  This is the form that fails the change that introduces a leak, whatever its base
  carries. ``dev_jargon`` reads every added line but a ledger's.

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
  or follow a one-letter option glued to it, as a compiler or linker flag spells a
  directory (``-I/home/<account>/include``), and must name something under the root.
* ``email``: an address shape whose local part does not follow a backslash (so the
  ``\\n@triton.jit`` of a source string is not one), other than the allowed ones.
* ``host_stamp`` (ledgers): a ``host`` or ``hostname`` value written as a raw machine
  stamp, one bare word alone, before a colon, or before ``(machine-stamped``. The
  neutral labels (``the GPU host``, Metal's ``this machine: ...``) are phrases, so
  this catches a host this file has no digest for.
* ``private_evidence``: a path into the development repository's own evidence, which
  never ships. Three rules, the patterns built from neutral pieces:

  - ``private_tree``: the name of the development evidence tree, or of the staging
    directory a round's lanes ran in, anywhere in the text (a path into either, or a
    citation of a document inside one);
  - ``lane_path``: a lane or record directory of that tree, ``dev/`` followed by its lane
    or record directory, or that directory name followed by a capitalised lane or record
    name at the start of a relative path;
  - ``home_relative`` (the ledgers leg, and the ledger lines of ``--patch``): a path under
    ``~/``. A ledger records where a run lay on the machine that ran it, so it writes
    ``<staging-root>/...``. A tracked file is not held to this rule: there a ``~/`` path
    is an instruction about the reader's own home (``~/.local``, a clone or a stub
    directory), and no measured form of it tells a recorded run from an instruction.

* ``dev_jargon`` (the tracked and subject legs and ``--patch``; never the ledgers, whose
  prose their bind and record tools write): text that only the development process
  can read. An exported file says what was decided, measured or required, and why, or
  points to an in-tree document that does. Six rules, the patterns built from neutral
  pieces:

  - ``private_document``: the file name of one of the private design documents the
    kernel tables and the array path were developed from (the kernel-table design, the
    decision log, the array-path and the test-and-certification designs). A file of
    that name inside a record directory, ``results/<campaign>/``, is a record and is
    not refused;
  - ``decision_number``: a decision number of the decision log, a capital D and a
    number from 4 up, standing alone. MEEP's dimension names D1, D2, D3 and Dcyl, the
    displacement field D and the D-seams are not decision numbers;
  - ``ruling_number``: a ruling cited by its number (a capital R and a number after
    the word ruling, after per, under, since, pre- or post-, in parentheses, with a
    possessive, or before round, freeze, corpus or re-certification), or a decision
    credited to the owner by the word ruling. A number with a leading zero in
    parentheses is an off-diagonal row-mask name (row, partner), never a ruling;
  - ``release_label``: the 1.0 release's working name, 0.10, used as a label: after
    until, since or before, after pre- or post-, after the evidence tree's prose name,
    before a development noun (lane, work package, phase, batch, round, step, series,
    design, forms, engine, freeze, a work-item label, and the like), or opening a
    parenthesis that holds a work-item label or ends the line. A number 0.10 is not a
    label, and a bare parenthesised 0.10 is read as a number and not refused;
  - ``process_label``: a work-package or lane label (``WP`` and a capitalised name,
    work package and an upper-case name, lane and a work-item label: upper-case letters
    with a number, or three to five upper-case letters that are not an English word a
    GPU lane's comment uses), an integration-batch directory name, or the label of
    the prototype phase (the word phase with the number zero, and its legs: a capital
    P, a zero, a hyphen and a small letter);
  - ``private_section``: an orphaned section number of the kernel-table design: its
    mutation sections 7.1 and 7.2 cited as a section or as a battery, and its
    appendix A. The IEEE 754 standard's own section numbers are not refused.

THE ALLOWLIST is ``hygiene_rules.json``'s ``allow`` list. Every entry names its rule
and carries its reason; nothing is allowed by path alone. A ``dev_jargon`` entry names
a file and the sha256 of each line it allows (the line with its leading and trailing
blanks removed), so an edit of that line ends the allowance.

A DECLARED PENDING HIT is one ``hygiene_rules.json``'s ``pending`` list names exactly: its
leg, file, rule, key path and the sha256 of the value it is in, with the number of hits
and what clears them. Only a tool may clear a ledger, so a hit there that a release's
bind re-cuts is declared until that bind: it is printed ``PENDING``, not counted, and
the exit is 0. A hit the list does not name is a hit. A declaration that names a value
no longer there is ``STALE``: the clearing tool has run, and the declaration is
deleted in the same change, after which the same text is refused. A file a tool writes
(a board, a record, a fixture copied from a round) is declared the same way on the
tracked leg, for ``dev_jargon`` only: by its path, the sha256 of its whole content and
its number of hits. Any change to the file ends the declaration, so the tool that
re-cuts it must leave it clean.

WHAT IS PRINTED. A hit's matched text is masked by its class (a public CI log is
public): ``LEG path:LINE: CLASS (RULE): <context>``, then a count per leg and class,
then each declared pending entry: its hits found against its hits declared.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import subprocess
import sys
import zlib
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
STATES_DIR = "parity/meep_gpu/ledger_states"
STATE_TABLES = ("triton", "cuda", "metal")
#: The desktop's own file in a folder: git ignores it, so no tracked tree holds it.
STATE_FOLDER_LITTER = ".DS_Store"
LEGS = ("tracked", "subject", "ledgers", "ledger_states")
#: The leg of timing row records, run only when asked for with its stores (``--records``).
RECORDS_LEG = "records"
#: What a timing row record is stamped (``timing_rows.SCHEMA``).
ROW_SCHEMA = "meep_gpu.timing_row/1"
#: The records leg's own class: what ``timing_rows.local_paths`` finds and no rule did.
LOCAL_PATH = "local_path"
#: The host stamps a row record writes: its Windows side's neutral label.
RECORD_HOST_STAMPS = ("WSL2",)
NAME_CLASSES = ("host_name", "account_name")
CLASSES = NAME_CLASSES + ("home_path", "email", "host_stamp", "private_evidence", "dev_jargon")
PRIVATE = "private_evidence"
JARGON = "dev_jargon"
EXPORT_MANIFEST = "tools/release/export_manifest.json"

#: Runs of letters, and the parts of a camel-cased run.
_RUN = re.compile(r"[A-Za-z]+")
_CAMEL = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+")
#: Home roots and temporary roots, from neutral pieces.
_HOME_ROOTS = ("ho" + "me", "Us" + "ers")
_TMP_ROOTS = ("var", "tmp")
_PATH_CHAR = r"A-Za-z0-9._\-"
#: A path starts a token, or follows a one-letter option glued to it (``-I/x``, ``-L/x``), as
#: compiler and linker flags spell include and library directories.
_START = rf"(?:(?<![{_PATH_CHAR}~])|(?<=(?<![A-Za-z0-9_])-[A-Za-z]))"
_HOME = re.compile(rf"{_START}/(?P<root>{'|'.join(_HOME_ROOTS)})/"
                   rf"(?P<account>[{_PATH_CHAR}]+)")
_TMP = re.compile(rf"{_START}/priv" + rf"ate/(?:{'|'.join(_TMP_ROOTS)})/"
                  rf"[{_PATH_CHAR}]")
_EMAIL = re.compile(r"(?<![A-Za-z0-9._%+\\-])[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+"
                    r"(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")
#: The development repository's private evidence, from neutral pieces: the evidence
#: tree and the staging directory a round's lanes ran in; the tree's lane and record
#: directories; a home-relative path (ledgers only).
_PRIVATE_TREES = ("bit" + "_parity_0_10", "bit" + "parity_phase0")
_PRIVATE_TREE = re.compile(r"(?<![A-Za-z0-9])(?:" + "|".join(_PRIVATE_TREES)
                           + r")(?![A-Za-z0-9])")
_LANE_DIRS = ("w" + "p", "ce" + "rt")
_LANE_PATH = re.compile(r"(?<![A-Za-z0-9_])de" + r"v/(?:" + "|".join(_LANE_DIRS) + r")/"
                        r"|(?<![A-Za-z0-9._/~-])(?:" + "|".join(_LANE_DIRS)
                        + r")/[A-Z][A-Za-z0-9_-]*")
_HOME_RELATIVE = re.compile(r"(?<![A-Za-z0-9._$/~-])~/[A-Za-z0-9._<>-][A-Za-z0-9._<>/-]*")
_HOST_KEYS = ("host", "hostname")
_HOST_STAMP = re.compile(r"^\s*[A-Za-z0-9._-]+(?:\s*$|\s*:|\s+\(machine-stamped)")
_MIN_BINARY_RUN = 8
_PRINTABLE = re.compile(rb"[\x20-\x7e]{%d,}" % _MIN_BINARY_RUN)
#: Development jargon, from neutral pieces. The private design documents' file names.
_PRIVATE_DOCS = ("KER" + "NELS", "DES" + "IGN", "ARRAY" + "_PATH", "TESTS" + "_AND_CERT")
_PRIVATE_DOCUMENT = re.compile(r"(?<![A-Za-z0-9_])(?:" + "|".join(_PRIVATE_DOCS)
                               + r")\.md(?![A-Za-z0-9_])")
#: A record directory, ``results/<campaign>/``, at the start of the token a name is in.
_RECORD_DIR = re.compile(r"(?:^|/)results/[^/\s]+/")
_TOKEN_START = re.compile(r"[^\s`'\"(\[{<]*$")
#: A decision number: D and a number from 4 up (MEEP's D1, D2, D3 are dimensions).
_DECISION = re.compile(r"(?<![A-Za-z0-9_.$/-])" + "D" + r"(?:[4-9]|[1-9][0-9])(?![A-Za-z0-9_])")
#: A ruling cited by number, or credited to the owner by the word.
_R = "R" + r"[0-9]+"
_RULING = re.compile(
    r"\b(?:[Rr]ulings?|per|under|since) " + _R + r"\b"
    r"|\b(?:pre|post)-" + _R + r"\b"
    r"|\((?!R0)" + _R + r"\)"
    r"|\b" + _R + r"'s\b"
    r"|\b" + _R + r" (?:round|freeze|corpus|re-certification|ruling|ruled)\b"
    r"|\b[Oo]wn" + r"er(?:'s)? (?:design )?rul(?:ing|ings|ed)\b"
    r"|\b[Tt]he own" + r"er rul(?:ed|es)\b")
#: The 1.0 release's working name used as a label.
_TEN = "0" + r"\.10"
_WORK_ITEM = r"(?:AP|K[CTM]|X)[0-9][A-Za-z0-9]*"
_NOT_A_NUMBER = r"(?![0-9]|\.[0-9])"
_RELEASE_LABEL = re.compile(
    r"\b(?:[Uu]ntil|[Ss]ince|[Bb]efore)\s+" + _TEN + _NOT_A_NUMBER
    + r"|\b(?:pre|post)-" + _TEN + _NOT_A_NUMBER
    + r"|\bbit[_-]parity " + _TEN + _NOT_A_NUMBER
    + r"|\b" + _TEN + r" (?:lanes?|WPs?|work packages?|phase|batch|round|step|series|design"
    r"|development|forms?|FORMS?|text|tree|seam|source|constitutive|refusals|REFUSALS"
    r"|certification|engine|freeze|array|start|mutation|inversion|sweep|" + _WORK_ITEM + r")\b"
    r"|(?<![A-Za-z0-9_])\(" + _TEN + r"(?: (?:" + _WORK_ITEM + r"|lanes?|WPs?|certification"
    r"|development)\b|\s*$)")
#: Work-package and lane labels, batch directories, the prototype phase.
_LANE_WORDS = ("ALL", "AND", "ARE", "CAN", "DOES", "END", "GETS", "HAS", "HOLDS", "MUST",
               "NOT", "ONE", "ONLY", "OWNS", "READS", "SEES", "THE", "TWO", "WAIT", "WAS")
_PROCESS_LABEL = re.compile(
    r"\b" + "W" + r"Ps? [A-Z][A-Za-z0-9]*"
    r"|\b[Ww]ork packages? [A-Z]{1,4}[0-9]*[a-z]?\b"
    r"|\blanes? (?:[A-Z]{1,4}[0-9]+[a-z]*|(?!(?:" + "|".join(_LANE_WORDS)
    + r")(?![A-Za-z0-9]))[A-Z]{3,5})(?![A-Za-z0-9])"
    r"|\bbatch_" + r"r[0-9]+\b"
    r"|\b[Pp]hase[- ]" + r"0(?![0-9.])"
    r"|\b" + "P" + r"0-[a-z][0-9]*\b")
#: The kernel-table design's mutation sections and appendix, orphaned.
_PRIVATE_SECTION = re.compile(r"\bApp" + r"endix A\b|\bsection " + r"7\.[12]\b"
                              r"|§" + r"7\.[12] battery\b")
_JARGON_RULES = (("private_document", _PRIVATE_DOCUMENT), ("decision_number", _DECISION),
                 ("ruling_number", _RULING), ("release_label", _RELEASE_LABEL),
                 ("process_label", _PROCESS_LABEL), ("private_section", _PRIVATE_SECTION))
#: Any jargon in a whole text (multi-line, so a form anchored at a line's end is seen).
_JARGON_ANY = re.compile("|".join(f"(?:{pattern.pattern})" for _rule, pattern in _JARGON_RULES),
                         re.MULTILINE)


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
    pending: List[dict] = field(default_factory=list)         # declared pending hits
    pending_found: Dict[str, int] = field(default_factory=dict)  # id -> hits found
    allowed_jargon: Dict[str, Set[str]] = field(default_factory=dict)  # path -> line sha256
    allowed_jargon_found: Set[Tuple[str, str]] = field(default_factory=set)
    legs_run: Set[str] = field(default_factory=set)

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
        elif rule == JARGON:
            lines = entry.get("lines", ())
            if not entry.get("path") or not lines or not all(
                    re.fullmatch(r"[0-9a-f]{64}", line) for line in lines):
                raise ValueError(f"a {JARGON} allowlist entry names a file and the sha256 of "
                                 f"each line it allows: {entry.get('id')}")
            rules.allowed_jargon.setdefault(entry["path"], set()).update(lines)
        else:
            raise ValueError(f"an allowlist entry names an unknown rule {rule!r}")
    rules.allowed_domain_suffixes = tuple(suffixes)
    for entry in document.get("pending", ()):
        rules.pending.append(_checked_declaration(entry))
    if names_file is not None:
        local = json.loads(names_file.read_text(encoding="utf-8"))
        for klass in NAME_CLASSES:
            for name in local.get(klass, ()):
                rules.add_name(klass, words_of(name))
    return rules


def _checked_declaration(entry: dict) -> dict:
    """A pending declaration, refused unless it names its leg, file, rule, hits and
    clearing tool: a ledger value by key path and digest, or a tracked file by digest."""
    leg = entry.get("leg")
    keys = ("id", "leg", "path", "rule", "hits", "until", "reason") + (
        ("file_sha256",) if leg == "tracked" else ("values",))
    missing = [key for key in keys if not entry.get(key)]
    if missing:
        raise ValueError(f"a pending declaration lacks {missing}: {entry.get('id')}")
    if leg == "ledgers":
        if entry["path"] not in LEDGERS:
            raise ValueError(f"a ledgers declaration names a ledger only: {entry['id']}")
        if entry["rule"] not in ("private_tree", "lane_path", "home_relative"):
            raise ValueError(f"a pending declaration names an unknown rule: {entry['id']}")
    elif leg == "tracked":
        if entry["path"] in LEDGERS:
            raise ValueError(f"a tracked declaration names a file, not a ledger: {entry['id']}")
        if entry["rule"] != JARGON:
            raise ValueError(f"a tracked declaration names an unknown rule; it declares "
                             f"{JARGON} hits only: {entry['id']}")
        if not re.fullmatch(r"[0-9a-f]{64}", str(entry["file_sha256"])):
            raise ValueError(f"a tracked declaration names its file's sha256: {entry['id']}")
    else:
        raise ValueError(f"a pending declaration names the ledgers or the tracked leg: "
                         f"{entry.get('id')}")
    return entry


def line_digest(line: str) -> str:
    """The allowlisted form of a line: sha256 of the line without its outer blanks."""
    return hashlib.sha256(line.strip().encode("utf-8")).hexdigest()


def export_excluded(root: Path) -> Set[str]:
    """The files the export manifest keeps out of the public tree (none without one)."""
    path = root / EXPORT_MANIFEST
    if not path.is_file():
        return set()
    document = json.loads(path.read_text(encoding="utf-8"))
    return {entry["path"] for entry in document.get("excluded", ())}


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
    for match in _PRIVATE_TREE.finditer(text):
        hits.append((match.start(), match.end(), PRIVATE, "private_tree"))
    for match in _LANE_PATH.finditer(text):
        hits.append((match.start(), match.end(), PRIVATE, "lane_path"))
    return hits


def scan_dev_jargon(text: str) -> List[Tuple[int, int, str, str]]:
    """Every development-jargon hit in ``text``: a class of its own, read on the exported
    files and kept out of :func:`scan_text`, which other tools call on ledger values."""
    hits: List[Tuple[int, int, str, str]] = []
    for rule, pattern in _JARGON_RULES:
        for match in pattern.finditer(text):
            if rule == "private_document":
                token = _TOKEN_START.search(text[max(0, match.start() - 400):match.start()]).group()
                if _RECORD_DIR.search(token):
                    continue
            hits.append((match.start(), match.end(), JARGON, rule))
    return hits


def scan_home_relative(text: str) -> List[Tuple[int, int, str, str]]:
    """Every home-relative path in ``text``: the ledger-only rule of ``private_evidence``."""
    return [(match.start(), match.end(), PRIVATE, "home_relative")
            for match in _HOME_RELATIVE.finditer(text)]


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
    if any(name in text for name in _PRIVATE_TREES) or (
            any(f"{name}/" in text for name in _LANE_DIRS) and _LANE_PATH.search(text)):
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


def _jargon_spans(rules: Rules, path: str, line: str) -> List[Tuple[int, int, str, str]]:
    """The jargon hits of one line of ``path``, none when the allowlist names the line."""
    spans = scan_dev_jargon(line)
    if spans:
        digest_ = line_digest(line)
        if digest_ in rules.allowed_jargon.get(path, ()):
            rules.allowed_jargon_found.add((path, digest_))
            return []
    return spans


def _hits_by_line(leg: str, path: str, rules: Rules, text: str,
                  cache: Dict[str, str], line_offset: Optional[Dict[int, int]] = None,
                  jargon: bool = True) -> List[Hit]:
    jargon = jargon and _JARGON_ANY.search(text) is not None
    if not jargon and _quick_reject(rules, text, cache):
        return []
    found: List[Hit] = []
    for number, line in enumerate(text.splitlines(), 1):
        spans = scan_text(rules, line, cache)
        if jargon:
            spans += _jargon_spans(rules, path, line)
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


def _tracked_declaration(rules: Rules, name: str, data: bytes) -> Optional[dict]:
    """The tracked declaration that names this file at exactly these bytes, if one does."""
    digest_ = None
    for entry in rules.pending:
        if entry["leg"] == "tracked" and entry["path"] == name:
            digest_ = digest_ or hashlib.sha256(data).hexdigest()
            if entry["file_sha256"] == digest_:
                return entry
    return None


def leg_files(rules: Rules, root: Path, leg: str, names: Iterable[str]) -> List[Hit]:
    cache: Dict[str, str] = {}
    hits: List[Hit] = []
    excluded = export_excluded(root)
    for name in names:
        if name in LEDGERS:
            continue
        path = root / name
        if not path.is_file():
            continue
        found = _hits_by_line(leg, name, rules, read_text(path), cache,
                              jargon=name not in excluded)
        entry = _tracked_declaration(rules, name, path.read_bytes()) if any(
            hit.klass == JARGON for hit in found) else None
        if entry is not None:
            jargon = [hit for hit in found if hit.klass == JARGON]
            if leg == entry["leg"]:
                rules.pending_found[entry["id"]] = len(jargon)
            found = [hit for hit in found if hit.klass != JARGON]
        hits.extend(found)
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


def _pending_entry(rules: Rules, name: str, where: str, rule: str, text: str
                   ) -> Optional[dict]:
    """The pending declaration that names this ledger hit exactly, if one does."""
    value = hashlib.sha256(text.encode("utf-8")).hexdigest()
    for entry in rules.pending:
        if entry["leg"] == "ledgers" and entry["path"] == name and entry["rule"] == rule and any(
                item.get("where") == where and item.get("sha256") == value
                for item in entry["values"]):
            return entry
    return None


def pending_status(rules: Rules) -> List[Tuple[str, int, int, str]]:
    """``(id, hits found, hits declared, state)`` per declaration whose leg has run:
    ``pending`` while every hit it declares is found, ``partial`` while some are,
    ``STALE`` when none is (the clearing tool has run: delete the declaration)."""
    out = []
    for entry in rules.pending:
        if entry["leg"] not in rules.legs_run:
            continue
        found = rules.pending_found.get(entry["id"], 0)
        state = ("STALE" if found == 0 else "pending" if found == int(entry["hits"])
                 else "partial")
        out.append((entry["id"], found, int(entry["hits"]), state))
    return out


def _scan_ledger(rules: Rules, leg: str, name: str, shown: str, document: object,
                 cache: Dict[str, str], count_pending: bool = True) -> List[Hit]:
    """One parsed ledger's hits. ``name`` is the path a pending entry declares; ``shown``
    is the file a hit names."""
    hits: List[Hit] = []
    for where, text, key in _walk(document, ""):
        spans = scan_text(rules, text, cache) + scan_home_relative(text)
        masked = _mask(text, [(s, e, k) for s, e, k, _ in spans]).strip()
        for _start, _end, klass, rule in spans:
            entry = (_pending_entry(rules, name, where, rule, text)
                     if klass == PRIVATE else None)
            if entry is not None:
                if count_pending:
                    rules.pending_found[entry["id"]] = rules.pending_found.get(entry["id"], 0) + 1
                continue
            hits.append(Hit(leg, shown, where, klass, rule, masked[:200]))
        if key in _HOST_KEYS and _HOST_STAMP.match(text):
            hits.append(Hit(leg, shown, where, "host_stamp", "raw_machine_stamp",
                            "[a raw machine stamp: one bare word, before ':' or "
                            "'(machine-stamped']"))
    return hits


def leg_ledgers(rules: Rules, root: Path) -> List[Hit]:
    cache: Dict[str, str] = {}
    hits: List[Hit] = []
    for name in LEDGERS:
        path = root / name
        document = json.loads(path.read_text(encoding="utf-8"))
        hits += _scan_ledger(rules, "ledgers", name, name, document, cache)
    return hits


def leg_ledger_states(rules: Rules, root: Path) -> List[Hit]:
    """A tree with no archive has nothing to read: its rows compare whole. A pending entry
    is declared for the live ledger, so an old state does not add to its count."""
    cache: Dict[str, str] = {}
    hits: List[Hit] = []
    folder = root / STATES_DIR
    for path in sorted(folder.glob("*/*")) if folder.is_dir() else ():
        table = path.parent.name
        if path.name == STATE_FOLDER_LITTER:
            continue
        if table not in STATE_TABLES or not path.name.endswith(".json.gz"):
            raise ValueError(f"{path.relative_to(root)} is no ledger state")
        document = json.loads(gzip.decompress(path.read_bytes()).decode("utf-8"))
        hits += _scan_ledger(rules, "ledger_states", f"meep_gpu/{table}_kernels/fingerprints.json",
                             path.relative_to(root).as_posix(), document, cache,
                             count_pending=False)
    return hits


def _local_spans(neutral, text: str, spans: Sequence[Tuple[int, int, str, str]]
                 ) -> List[Tuple[int, int, str, str]]:
    """``local_path`` spans of ``text`` that no span of ``spans`` overlaps: each absolute
    path outside the system directories, Windows profile and ``~/`` path, as
    ``timing_rows.local_paths`` reads them (``neutral`` is ``neutral_ids``)."""
    found = [(m.start(), m.end()) for m in neutral._ABSOLUTE.finditer(text)
             if not m.group().startswith(("//",) + neutral._SYSTEM_ROOTS)
             and text[m.start() - 1:m.start()] != "/"]
    found += [(m.start(), m.end()) for m in neutral._HOME_SHAPE.finditer(text)]
    found += [(m.start(), m.end()) for m in re.finditer(
        r"(?<![A-Za-z0-9._$/~-])~/[^\s'\"<>,;:()\[\]{}=]*", text)]
    out: List[Tuple[int, int, str, str]] = []
    for start, end in sorted(found, key=lambda span: (span[0], -span[1])):
        if not any(start < e and s < end for s, e, _k, _r in list(spans) + out):
            out.append((start, end, LOCAL_PATH, "local_paths"))
    return out


def leg_records(rules: Rules, stores: Sequence[Path], root: Path = DEFAULT_ROOT) -> List[Hit]:
    """Every timing row record below each store, read as a ledger is (``records``), and
    held to the check the records' writers are held to (``timing_rows.local_paths``)."""
    sys.path.insert(0, str(root / "parity" / "meep_gpu"))
    try:
        import neutral_ids  # noqa: PLC0415
        import timing_rows  # noqa: PLC0415
    finally:
        sys.path.pop(0)
    cache: Dict[str, str] = {}
    hits: List[Hit] = []
    for store in stores:
        found = 0
        for path in sorted(Path(store).rglob("*.json")):
            document = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(document, dict) or document.get("schema") != ROW_SCHEMA:
                continue
            found += 1
            name = f"{Path(store).name}/{path.relative_to(store).as_posix()}"
            for where, text, key in _walk(document, ""):
                spans = scan_text(rules, text, cache) + scan_home_relative(text)
                if timing_rows.local_paths(text):
                    # What the writers' check finds and no rule did, by its span where it
                    # has one (a path), else the string (a host name, a card's UUID).
                    spans += _local_spans(neutral_ids, text, spans) or (
                        [] if spans else [(0, len(text), LOCAL_PATH, "local_paths")])
                masked = _mask(text, [(s, e, k) for s, e, k, _ in spans]).strip()
                # A record's key can be a path (a build record's map of images), so the key
                # path is printed masked as well, read with its joining dots as spaces (a
                # path right after one would not start a token).
                probe = where.replace(".", " ")
                named = scan_text(rules, probe, cache) + scan_home_relative(probe)
                place = _mask(where, [(s, e, k) for s, e, k, _ in
                                      named + _local_spans(neutral_ids, probe, named)])
                for _start, _end, klass, rule in spans:
                    hits.append(Hit(RECORDS_LEG, name, place, klass, rule, masked[:200]))
                if key in _HOST_KEYS and _HOST_STAMP.match(text) and \
                        text.strip() not in RECORD_HOST_STAMPS:
                    hits.append(Hit(RECORDS_LEG, name, place, "host_stamp",
                                    "raw_machine_stamp", "[a raw machine stamp: one bare "
                                    "word, before ':' or '(machine-stamped']"))
        if not found:
            raise ValueError(f"{Path(store).name}: no timing row record below it")
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


def leg_patch(rules: Rules, patch: Path, root: Path = DEFAULT_ROOT) -> List[Hit]:
    cache: Dict[str, str] = {}
    hits: List[Hit] = []
    excluded = export_excluded(root)
    for name, number, text in added_lines(patch.read_text(encoding="utf-8", errors="replace")):
        spans = scan_text(rules, text, cache)
        if name in LEDGERS:
            spans += scan_home_relative(text)
        elif name not in excluded:
            spans += _jargon_spans(rules, name, text)
        for start, end, klass, rule in spans:
            masked = _mask(text, [(start, end, klass)]).strip()
            hits.append(Hit("patch", name, str(number), klass, rule, masked[:200]))
        if name in LEDGERS:
            for key in _HOST_KEYS:
                match = re.match(rf'^\s*"{key}":\s*"(.*)",?\s*$', text)
                if match and _HOST_STAMP.match(match.group(1)):
                    hits.append(Hit("patch", name, str(number), "host_stamp",
                                    "raw_machine_stamp", "[a raw machine stamp]"))
    return hits


def run_legs(rules: Rules, root: Path, legs: Sequence[str],
             records: Sequence[Path] = ()) -> Tuple[List[Hit], List[str]]:
    hits: List[Hit] = []
    problems: List[str] = []
    rules.pending_found = {}
    rules.legs_run = set()
    for leg in legs:
        rules.legs_run.add(leg)
        try:
            if leg == "tracked":
                hits.extend(leg_files(rules, root, leg, tracked_files(root)))
            elif leg == "subject":
                hits.extend(leg_files(rules, root, leg, subject_files(root)))
            elif leg == "ledgers":
                hits.extend(leg_ledgers(rules, root))
            elif leg == "ledger_states":
                hits.extend(leg_ledger_states(rules, root))
            elif leg == RECORDS_LEG:
                if not records:
                    problems.append(f"{leg}: no store given (--records STORE)")
                else:
                    hits.extend(leg_records(rules, records, root))
            else:
                problems.append(f"unknown leg {leg!r}; one of {LEGS + (RECORDS_LEG,)}")
        except (OSError, ValueError, EOFError, RecursionError, zlib.error,
                ImportError) as exc:
            problems.append(f"{leg}: cannot be read: {exc!r}")
    return hits, problems


def summary(hits: Sequence[Hit], legs: Sequence[str]) -> List[str]:
    lines = []
    for leg in legs:
        counts = {klass: 0 for klass in CLASSES + ((LOCAL_PATH,) if leg == RECORDS_LEG
                                                   else ())}
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
    parser.add_argument("--records", type=Path, action="append", default=[],
                        metavar="STORE", help="a timing row store the records leg reads "
                                              "(repeatable; with --legs records)")
    args = parser.parse_args(argv)
    rules = load_rules(args.rules, args.names_file)
    if args.patch is not None:
        legs, problems = ["patch"], []
        hits = leg_patch(rules, args.patch, args.root.resolve())
    else:
        legs = [leg.strip() for leg in args.legs.split(",") if leg.strip()]
        hits, problems = run_legs(rules, args.root.resolve(), legs, args.records)
    for hit in hits:
        print(hit.line(), flush=True)
    for line in summary(hits, legs):
        print(line, flush=True)
    for ident, found, declared, state in pending_status(rules):
        print(f"{state.upper() if state == 'STALE' else 'PENDING'} {ident}: {found} of "
              f"{declared} declared hit(s) found ({state}); cleared by the tool its "
              f"declaration names", flush=True)
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
