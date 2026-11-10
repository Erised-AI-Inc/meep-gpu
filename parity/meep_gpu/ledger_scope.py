"""Which kernel-table ledgers a run can read, and whether what it read has moved.

A ledger decides what its own table's ladder admits, so a change to it can change a plan
only on a process that reads it. A process reads the ledgers of the tables its device can
serve and no others: ``fastpath.candidate_tables`` answers the Triton and hand-CUDA tables
for a CuPy engine and the Metal table for a NumPy engine on an Apple GPU, and each reader
(``fastpath._fingerprints``, ``metal_dispatch.fingerprints``) is reached only from its own
platform's rungs: ``fastpath.certified_nvidia_stacks`` reads the Triton ledger for both
NVIDIA tables, so the scope is the platform and not the table. ``fastpath_cuda._read_json``
opens ``certification.json`` for ``fastpath_cuda.certification()``, which no dispatch path
calls, so that file is read by no device (:data:`UNREAD_LEDGERS`).

The recorders state more than that. ``plan_key.plan_inputs`` digests the ledgers of all
three tables on every run, and ``bench_fused_products.admission_data`` digests the ledgers
of every table whose module the process imported, which on an Apple GPU includes the
NVIDIA tables: the Metal arm table imports ``triton_kernels/launch.py``, and importing a
table's module opens no ledger. So the comparison is narrowed where it is read, over the
map each run recorded: no run is rewritten or re-run. A run that records a launch or a plan
on a table of the other platform is compared on that table's ledgers too: the record
outranks the label.

THE SLICE. Within a ledger a device reads, a decision reads one small part of it
(:func:`ledger_slice`, cut per file because records hold per-file digests):

* ``bound``: the digest of the bound fields of every cited weld, ``fastpath.bound_digest``;
* ``run_bound`` and the weld's ``runs`` shape, and per table the fields of the device's own
  run a verdict reads: the keys of ``certified_policies`` on NVIDIA, ``torch`` and
  ``metal_frontend`` on Metal. The other devices' runs are not in it;
* the device's own stack and its Triton version's membership of the validated list, on the
  Triton ledger;
* ``other_bound``: the bound digest of every other entry, and none of its run fields. A
  kernel module can read a sibling as text at emission (:data:`TEXT_READS`), so that sibling
  is in no import record and may be pinned only by a weld no arm cites. Its re-gate moves
  this digest, and only that shows it; an edit nothing re-gated moves no digest, so the file's
  live bytes are held to its pins (:func:`ledger_moved`). That asks only when the ledger's
  own digest moved: with no ledger move, such an edit shows nowhere, as it never did. A pin
  counts only from an entry whose other package pins hold at the tree's bytes, and the file
  must sit at the one set of bytes those entries name: a weld left behind by an older tree can
  still bind its digest, so a live run does not show it current.

``driver_dispatch`` is out, as it is out of ``plan_key``'s view: it is the route record,
which the planner does not read.

A bind of another device on unchanged bytes writes only that device's runs and stack, so a
slice stays equal and the whole-file digest moves: :func:`ledger_moved` then sets the file
aside. The slice carries digest values rather than a liveness bit, so a device's own re-gate
of a cited weld keeps owing what it owes, even for a row that did not import the moved file.
Every case that cannot be proved compares the whole digest, as a run always was, under a
reason of its own. A ledger is parsed as the package's readers parse it (:func:`parse_ledger`),
so a file they refuse is not one this module can call unmoved.

THE OLD BYTES. A record holds a digest, not bytes, and neither git nor a lab's clone can be
relied on to hold them, so ``ledger_states/<table>/<raw sha256>.json.gz`` keeps the ledger
states records can still carry on (:class:`LedgerStates`, and ``python ledger_scope.py
archive``). The directory is the index, every read re-hashes the bytes against the file's
name, and a tree with no archive compares as it always did. The slice is computed on both
sides at compare time and stored nowhere, so its definition can change without a re-cut.

THE TIMING RECORDS. :func:`timing_record_moved`: a planner reads its own architecture's
record, and the prior only on silence, from the nearest timed architecture of its own
vendor. So another vendor's record is not compared, nor another architecture's while an
own-architecture record holds at one digest on both sides. The own architecture's is always
compared, whichever card or table it names: those records pool into one prior.

The prior opens every record of every vendor (``fastpath._apply_preference``, through
``dispatch_preference.record_files`` and ``load_record``), so an NVIDIA run opens the Apple
record, and the converse. An Apple record reaches no NVIDIA decision: ``portable_prior``
files it under its table's vendor and ``consult_prior`` asks only its own. The file's bytes
are another matter, since the reader takes them with no table expected: a record of an
NVIDIA table in that file pools into the NVIDIA prior and moves a plan, and a sealed record
that ``validate`` raises on refuses the planner outright. So what is set aside is the
record's shape on the live tree, never the file's name, and
``test_ledger_scope.py`` moves that file's contents under the real planner. The recorded
bytes are not held, so a row's old file is taken to have been an Apple record, as a shipped
one always is.

``test_ledger_scope.py`` holds the claims. Its audit-hook half runs the planning path on a
host with no device and does not step a field.

Standard library and ``selected_plan``; imported by no gate.
"""
from __future__ import annotations

import argparse
import ast
import gzip
import hashlib
import io
import json
import os
import re
import sys
import types
import zlib
from pathlib import Path
from typing import Any, Collection, Dict, Iterable, List, Mapping, NamedTuple, Optional, Tuple

import selected_plan

_NVIDIA_DEVICE = re.compile(r"(cc\d+|\d+\.\d+)")
_APPLE_DEVICE = re.compile(r"applegpu_\w+")
_LEDGER = re.compile(r"meep_gpu/(?P<table>\w+)_kernels/(fingerprints|certification)\.json")
_FINGERPRINTS = re.compile(r"meep_gpu/(?P<table>\w+)_kernels/fingerprints\.json")
_TIMING_RECORD = re.compile(r"meep_gpu/(?P<table>\w+)_kernels/timing_[^/]+\.json")
#: The one spelling a capability keys ``runs`` under (``fastpath._CAPABILITY_KEY``, and the
#: Apple family ``metal_runs.ARCHITECTURE_KEY``; a test holds them equal, because this
#: module imports no package module).
_CAPABILITY_KEY = re.compile(r"^[1-9][0-9]*\.[0-9]$")
_APPLE_KEY = re.compile(r"^applegpu_[a-z0-9_]+$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
#: What a recorder writes for an imported Triton. Anything else names no version, so no
#: membership of the list can be read for it.
_TRITON_VERSION = re.compile(r"[0-9]+\.[0-9]+[-+.0-9a-zA-Z]*")

#: The ledgers no dispatch path opens (``test_ledger_scope.py``, under an audit hook).
UNREAD_LEDGERS = ("meep_gpu/cuda_kernels/certification.json",)
#: The entry that holds the route record: outside ``plan_key``'s view and every slice.
ROUTE_ENTRY = "driver_dispatch"
#: ``fastpath.BOUND_FIELDS``, ``fastpath.NVIDIA_STACK_AXES`` and the names they sit under,
#: spelled here for the same reason.
BOUND_FIELDS = ("source_sha256", "source_sha256_at_recert", "probe", "probe_sha256",
                "adapter", "adapter_sha256")
NVIDIA_STACK_AXES = ("cupy", "nvrtc", "cuda_runtime", "cuda_driver", "host", "numpy", "meep")
STACK_ENTRY = "certified_nvidia_stack"
STACK_BY_CAPABILITY = "by_capability"
VERSIONS_ENTRY = "validated_triton_versions"
#: A ledger the reader could not take: ``fastpath._fingerprints`` writes it, and a reader
#: then answers unknown whatever the rest of the document says.
UNREADABLE_KEY = "_unreadable"

SLICE_FORM = "meep_gpu.ledger_slice/1"
LEDGER_OF_TABLE = {table: f"meep_gpu/{table}_kernels/fingerprints.json"
                   for table in ("triton", "cuda", "metal")}
#: Per table, the file and the constant that say which welds its arms cite.
CITED_CONSTANTS = {"triton": ("meep_gpu/fastpath.py", "ARM_CERTIFICATION"),
                   "cuda": ("meep_gpu/fastpath_cuda.py", "CUDA_ARM_CERTIFICATION"),
                   "metal": ("meep_gpu/metal_dispatch.py", "ARM_CERTIFICATION")}
STATES_DIR = "parity/meep_gpu/ledger_states"

#: Kernel modules that read a sibling's source as text at emission, because importing it
#: would pull CuPy or Triton in. The sibling is then in no import record unless something
#: else imported it, so the closure cannot see it move. ``test_ledger_scope.py`` holds this
#: table to an ast scan of the package.
TEXT_READS: Mapping[str, Tuple[str, ...]] = {
    "meep_gpu/cuda_kernels/bfast_curl.py": ("step_curl_kernels.py",),
    "meep_gpu/cuda_kernels/conductive_bfast_fused_hd_pair.py": ("constitutive_kernels.py",
                                                                "step_curl_kernels.py"),
    "meep_gpu/cuda_kernels/conductive_kernels.py": ("step_curl_kernels.py",),
    "meep_gpu/cuda_kernels/fused_polarization_pair.py": ("ade_kernels.py",),
    "meep_gpu/cuda_kernels/no_pml_curl.py": ("step_curl_kernels.py",),
    "meep_gpu/cuda_kernels/special_kz_curl.py": ("step_curl_kernels.py",),
    "meep_gpu/triton_kernels/beta_complex_fused_hd_pair.py": ("folded_complex.py",
                                                              "special_kz.py"),
    "meep_gpu/triton_kernels/beta_real_fused_hd_pair.py": ("folded_complex.py",
                                                           "special_kz.py"),
    "meep_gpu/triton_kernels/complex_fused_hd_pair.py": ("complex_fields.py",),
    "meep_gpu/triton_kernels/conductive_bfast_fused_hd_pair.py": ("bfast_curl.py",
                                                                  "conductivity.py"),
    "meep_gpu/triton_kernels/folded_complex_fused_hd_pair.py": ("folded_complex.py",),
    "meep_gpu/triton_kernels/folded_fused_hd_pair.py": ("symmetry.py",),
    "meep_gpu/triton_kernels/fused_hd_pair.py": ("kernels.py",),
}


class Verdict(NamedTuple):
    """One file's comparison: whether it moved for this run, the reason by name, the proof."""
    moved: bool
    why: str
    proof: Mapping[str, Any]


class SliceUnavailable(Exception):
    """A weld list could not be read from a table's constant."""


class LedgerStateRefused(Exception):
    """An archived state whose bytes are not what its name says."""


def platform_of_device(device: Any) -> Optional[str]:
    """``nvidia`` for a compute capability (``cc86``, ``8.6``), ``metal`` for an Apple GPU
    architecture (``applegpu_g13s``), ``None`` for anything else."""
    text = str(device or "")
    if _NVIDIA_DEVICE.fullmatch(text):
        return "nvidia"
    if _APPLE_DEVICE.fullmatch(text):
        return "metal"
    return None


def platform_of_table(table: Any) -> Optional[str]:
    """The platform of a kernel table's name, ``None`` for anything else."""
    return selected_plan.PLATFORM_OF_TABLE.get(str(table))


def reads_ledger(name: str, device: Any, recorded: Collection[Optional[str]] = ()) -> bool:
    """Whether a run on ``device`` can read the file ``name``. False only for a ledger of
    a table of the other platform, when the run records no launch or plan on that platform
    (``recorded``, platform names): a device that cannot be placed, or a name that is no
    table's ledger, is compared as it always was."""
    platform = platform_of_device(device)
    match = _LEDGER.fullmatch(str(name))
    table = match.group("table") if match else None
    if platform is None or table not in selected_plan.PLATFORM_OF_TABLE:
        return True
    own = selected_plan.PLATFORM_OF_TABLE[table]
    return own == platform or own in recorded


def policies_dropped(previous: Any, run: Mapping[str, Any], policies_field: str) -> List[str]:
    """The policies a same-capability bind would lose: runs are keyed by capability, not
    card, so a second card's single-policy campaign would replace the first's flush
    evidence, and dispatch reads those keys. ``previous`` is the run being replaced."""
    held = previous.get(policies_field) if isinstance(previous, Mapping) else None
    kept = run.get(policies_field)
    return sorted(set(held if isinstance(held, Mapping) else ())
                  - set(kept if isinstance(kept, Mapping) else ()))


def holds_own_ledger(names: Iterable[str], device: Any) -> bool:
    """Whether ``names`` holds a ledger of the platform of ``device`` that a process reads.
    True for a device that cannot be placed: nothing is narrowed on it."""
    platform = platform_of_device(device)
    if platform is None:
        return True
    tables = (match.group("table") for match in map(_LEDGER.fullmatch, map(str, names))
              if match and match.group() not in UNREAD_LEDGERS)
    return any(platform_of_table(table) == platform for table in tables)


def architecture_of_device(device: Any) -> Optional[str]:
    """The key a device's runs sit under: ``cc86`` and ``8.6`` are ``8.6``, ``cc100`` is
    ``10.0``, ``applegpu_g13s`` is itself. ``None`` for a spelling that keys no run, which
    is then compared as it always was: ``8.60``, ``08.6``, ``cc086`` and ``cc8`` here, and
    ``cc860``, whose ``86.0`` is well formed, where a ledger is read (:func:`ledger_moved`)."""
    text = str(device or "")
    match = re.fullmatch(r"cc(\d+)", text)
    if match:
        digits = match.group(1)
        text = f"{digits[:-1]}.{digits[-1]}"
    if _CAPABILITY_KEY.match(text) or _APPLE_KEY.match(text):
        return text
    return None


def parse_ledger(raw: bytes) -> Any:
    """The bytes as every package reader takes a ledger, ``json.load(open(path,
    encoding="utf-8"))``. ``json.loads`` on bytes also accepts a byte-order mark, UTF-16 and
    UTF-32, which those readers refuse."""
    return json.loads(raw.decode("utf-8"))


def view_sha256(document: Any) -> str:
    """``plan_key.ledger_view_sha256`` of a parsed ledger."""
    if isinstance(document, dict):
        document = {k: v for k, v in document.items() if k != ROUTE_ENTRY}
    text = json.dumps(document, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def bound_digest(entry: Mapping[str, Any]) -> Optional[str]:
    """``fastpath.bound_digest``, or ``None`` for an entry that binds no bytes, where the
    package raises: that weld's own contract refuses it, and here it is one more value."""
    bound = {field: entry[field] for field in BOUND_FIELDS if field in entry}
    if not bound:
        return None
    text = json.dumps(bound, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def frontend_key(value: Any) -> Optional[str]:
    """``metal_dispatch.frontend_key``: one spelling of a Metal frontend version."""
    if value is None:
        return None
    text = str(value).strip()
    if text.startswith("metalfe-"):
        text = text[len("metalfe-"):]
    return text or None


def _runs_shape(entry: Mapping[str, Any]) -> str:
    """A verdict tells these apart (``metal_dispatch``: no ``runs`` reads unknown, an empty
    one reads uncertified), so a slice must."""
    if "runs" not in entry:
        return "absent"
    runs = entry["runs"]
    if not isinstance(runs, Mapping):
        return "malformed"
    return "held" if runs else "empty"


def _weld_slice(entry: Any, table: str, device: str) -> Dict[str, Any]:
    if not isinstance(entry, Mapping):
        return {"entry": False}
    out: Dict[str, Any] = {"entry": True, "bound": bound_digest(entry),
                           "runs": _runs_shape(entry)}
    runs = entry.get("runs")
    run = runs.get(device) if isinstance(runs, Mapping) else None
    if not isinstance(run, Mapping):
        out["run_bound"] = None
        return out
    out["run_bound"] = run.get("bound_sha256")
    if table == "metal":
        torch = run.get("torch")
        out["torch"] = None if torch in (None, "?", "unknown") else str(torch)
        out["metal_frontend"] = frontend_key(run.get("metal_frontend"))
    else:
        policies = run.get("certified_policies")
        out["policies"] = sorted(policies) if isinstance(policies, Mapping) else None
    return out


def _stack_slice(document: Mapping[str, Any], device: str) -> Optional[Dict[str, Any]]:
    """``fastpath.certified_nvidia_stacks`` for one capability: a record out of shape
    certifies nothing and reads as no record."""
    entry = document.get(STACK_ENTRY)
    records = entry.get(STACK_BY_CAPABILITY) if isinstance(entry, Mapping) else None
    if not isinstance(records, Mapping):
        return {"unreadable": True}
    record = records.get(device)
    stack = record.get("stack") if isinstance(record, Mapping) else None
    if not isinstance(stack, Mapping) or set(stack) != set(NVIDIA_STACK_AXES):
        return None
    values = {axis: stack[axis] for axis in NVIDIA_STACK_AXES}
    if all(isinstance(listed, list) and listed
           and all(isinstance(value, str) and value for value in listed)
           for listed in values.values()):
        return values
    return None


def _versions_slice(document: Mapping[str, Any], triton_version: Any) -> Dict[str, Any]:
    """Membership of the version the row ran on; the whole list when no version is known,
    or the label is none, so an append then counts as a move."""
    try:
        listed = [str(value) for value in document.get(VERSIONS_ENTRY) or ()]
    except TypeError:
        listed = []
    if not (isinstance(triton_version, str) and _TRITON_VERSION.fullmatch(triton_version)):
        return {"validated_list": sorted(listed)}
    return {"read": triton_version, "validated": triton_version in listed}


def ledger_slice(document: Mapping[str, Any], table: str, device: str,
                 cited: Mapping[str, Iterable[str]], *,
                 triton_version: Any = None) -> Dict[str, Any]:
    """The part of one ledger file a decision on ``device`` reads (the module docstring)."""
    welds = set(cited[table])
    out: Dict[str, Any] = {
        "form": SLICE_FORM, "device": device,
        "welds": {key: _weld_slice(document.get(key), table, device) for key in sorted(welds)},
        "other_bound": {key: bound_digest(entry) for key, entry in sorted(document.items())
                        if key != ROUTE_ENTRY and key not in welds
                        and isinstance(entry, Mapping)
                        and any(field in entry for field in BOUND_FIELDS)}}
    if table == "triton":
        out["stack"] = _stack_slice(document, device)
        out["triton"] = _versions_slice(document, triton_version)
    return out


def slice_sha256(document: Mapping[str, Any], table: str, device: str,
                 cited: Mapping[str, Iterable[str]], *, triton_version: Any = None) -> str:
    text = json.dumps(ledger_slice(document, table, device, cited,
                                   triton_version=triton_version),
                      sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _differences(old: Any, new: Any, path: str = "") -> List[str]:
    if isinstance(old, Mapping) and isinstance(new, Mapping):
        found: List[str] = []
        for key in sorted(set(old) | set(new)):
            found += _differences(old.get(key), new.get(key), f"{path}.{key}" if path else key)
        return found
    return [] if old == new else [path]


def _pins_moved(old: Mapping[str, Any], new: Mapping[str, Any], keys: Iterable[str]) -> List[str]:
    """The bound fields that moved on ``keys``, by file where the field names files."""
    found = set()
    for key in keys:
        before, after = old.get(key), new.get(key)
        before = before if isinstance(before, Mapping) else {}
        after = after if isinstance(after, Mapping) else {}
        for field in BOUND_FIELDS:
            a, b = before.get(field), after.get(field)
            if a == b:
                continue
            if isinstance(a, Mapping) and isinstance(b, Mapping):
                found.update(pin for pin in set(a) | set(b) if a.get(pin) != b.get(pin))
            else:
                found.add(field)
    return sorted(found)


def stated_drift(document: Mapping[str, Any]) -> List[str]:
    """The entries that say a pinned file left their artifact's bytes (``rebind_cuda_welds``
    writes ``source_drift`` and ``status: DRIFTED``), or whose status is anything but
    ``PASS``. The bound digest comes from the artifact, so it does not show this; the ledger's
    own statement does."""
    return sorted(key for key, entry in document.items()
                  if key != ROUTE_ENTRY and isinstance(entry, Mapping)
                  and (entry.get("source_drift") or entry.get("status") not in (None, "PASS")))


def _keys_device(document: Mapping[str, Any], device: str, welds: Iterable[str],
                 table: str) -> bool:
    """Whether this ledger state knows ``device`` at all. One that does not has only
    ``None`` for its run fields on both sides, which proves nothing: a spelling no run
    uses reads like any device the ledger has never heard of."""
    for key in welds:
        entry = document.get(key)
        runs = entry.get("runs") if isinstance(entry, Mapping) else None
        if isinstance(runs, Mapping) and device in runs:
            return True
    if table == "triton":
        block = document.get(STACK_ENTRY)
        records = block.get(STACK_BY_CAPABILITY) if isinstance(block, Mapping) else None
        return isinstance(records, Mapping) and device in records
    return False


def cited_welds(root: Any) -> Dict[str, Tuple[str, ...]]:
    """The weld keys each table's arms cite, by ast over the three constants: this module
    imports no package module. One module-level Name is resolved; anything else raises
    :class:`SliceUnavailable` by name."""
    out: Dict[str, Tuple[str, ...]] = {}
    for table, (rel, constant) in CITED_CONSTANTS.items():
        try:
            tree = ast.parse(Path(root, rel).read_text(encoding="utf-8"))
        except (OSError, SyntaxError, ValueError) as exc:
            raise SliceUnavailable(f"{rel}: {exc!r}") from exc
        names: Dict[str, str] = {}
        node = None
        for item in tree.body:
            target = (item.target if isinstance(item, ast.AnnAssign) else
                      item.targets[0] if isinstance(item, ast.Assign) and len(item.targets) == 1
                      else None)
            if not isinstance(target, ast.Name) or item.value is None:
                continue
            if isinstance(item.value, ast.Constant) and isinstance(item.value.value, str):
                names[target.id] = item.value.value
            if target.id == constant:
                node = item.value
        if not isinstance(node, ast.Dict):
            raise SliceUnavailable(f"{rel}: {constant} is not a literal mapping")
        gates = set()
        for value in node.values:
            last = value.elts[-1] if isinstance(value, ast.Tuple) and value.elts else None
            if isinstance(last, ast.Constant) and isinstance(last.value, str):
                gates.add(last.value)
            elif isinstance(last, ast.Name) and last.id in names:
                gates.add(names[last.id])
            else:
                raise SliceUnavailable(f"{rel}: {constant} holds a value that names no weld")
        out[table] = tuple(sorted(gates))
    return out


# ---------------------------------------------------------------------------------------
# The archive of old ledger bytes
# ---------------------------------------------------------------------------------------

class LedgerStates:
    """``ledger_states/<table>/<raw sha256>.json.gz`` of a judged tree. The directory is the
    index: no manifest exists that could lie. A state is looked up by its raw digest, or by
    ``plan_key``'s view of it, and answers only for its own table."""

    def __init__(self, root: Any):
        self.directory = Path(root, STATES_DIR)
        self._raw: Dict[Tuple[str, str], Mapping[str, Any]] = {}
        self._views: Dict[str, Dict[str, Mapping[str, Any]]] = {}

    def _load(self, path: Path) -> Mapping[str, Any]:
        raw = gzip.decompress(path.read_bytes())
        digest = hashlib.sha256(raw).hexdigest()
        if digest != path.name.split(".")[0]:
            raise LedgerStateRefused(f"{path.name}: its bytes digest to {digest}")
        document = parse_ledger(raw)
        if not isinstance(document, dict):
            raise LedgerStateRefused(f"{path.name}: not a mapping")
        return document

    def raw(self, table: str, digest: str) -> Optional[Mapping[str, Any]]:
        """The state whose file digests to ``digest``, None when it is not archived."""
        key = (table, digest)
        if key not in self._raw:
            path = self.directory / table / f"{digest}.json.gz"
            if not path.is_file():
                return None
            try:
                self._raw[key] = self._load(path)
            except (OSError, ValueError, EOFError, RecursionError, zlib.error) as exc:
                raise LedgerStateRefused(f"{path.name}: {exc!r}") from exc
        return self._raw[key]

    def view(self, table: str, digest: str) -> Optional[Mapping[str, Any]]:
        """A state with this view, None when none is archived. Two raw states can share a
        view (they differ in ``driver_dispatch``); the slice leaves that entry out, so
        either answers the same."""
        if table not in self._views:
            views: Dict[str, Mapping[str, Any]] = {}
            folder = self.directory / table
            for path in sorted(folder.glob("*.json.gz")) if folder.is_dir() else ():
                document = self.raw(table, path.name.split(".")[0])
                views.setdefault(view_sha256(document), document)
            self._views[table] = views
        return self._views[table].get(digest)


def archive_state(root: Any, table: str, source: Any) -> Path:
    """File the bytes of ``source`` under their digest in the tree at ``root``. Idempotent:
    gzip level 9 with no time or name in its header, so one state is one file. The bytes'
    digest is their proof, so they may come from any copy. Refuses what a reader would not
    parse (:func:`parse_ledger`)."""
    if table not in LEDGER_OF_TABLE:
        raise ValueError(f"{table!r} is not one of {sorted(LEDGER_OF_TABLE)}")
    raw = Path(source).read_bytes()
    document = parse_ledger(raw)
    if not isinstance(document, dict):
        raise ValueError(f"{source}: not a mapping, so no ledger")
    digest = hashlib.sha256(raw).hexdigest()
    buffer = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", compresslevel=9, fileobj=buffer, mtime=0) as out:
        out.write(raw)
    path = Path(root, STATES_DIR, table, f"{digest}.json.gz")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file() or path.read_bytes() != buffer.getvalue():
        path.write_bytes(buffer.getvalue())
    return path


# ---------------------------------------------------------------------------------------
# One tree, as a comparison reads it
# ---------------------------------------------------------------------------------------

class _Tree:
    """What the comparisons read from the judged tree, once per ``cache``."""

    def __init__(self, root: Any):
        self.root = Path(root)
        self.states = LedgerStates(self.root)
        self._documents: Dict[str, Any] = {}
        self._sha: Dict[str, Optional[str]] = {}
        self._cited: Any = None
        self._preference: Any = None
        self._holds: Dict[str, List[str]] = {}
        self._pins: Dict[Optional[str], Tuple[set, Dict[str, set]]] = {}
        self.slices: Dict[Tuple[Any, ...], Verdict] = {}

    def sha256(self, rel: str) -> Optional[str]:
        if rel not in self._sha:
            try:
                self._sha[rel] = hashlib.sha256((self.root / rel).read_bytes()).hexdigest()
            except OSError:
                self._sha[rel] = None
        return self._sha[rel]

    def document(self, rel: str) -> Any:
        """The parsed file, ``None`` when it is absent or does not parse."""
        if rel not in self._documents:
            try:
                self._documents[rel] = parse_ledger((self.root / rel).read_bytes())
            except (OSError, ValueError, RecursionError):
                self._documents[rel] = None
        return self._documents[rel]

    def digest(self, rel: str, form: str) -> Optional[str]:
        if form == "raw":
            return self.sha256(rel)
        document = self.document(rel)
        return None if document is None else view_sha256(document)

    def cited(self) -> Dict[str, Tuple[str, ...]]:
        if self._cited is None:
            try:
                self._cited = cited_welds(self.root)
            except SliceUnavailable as exc:
                self._cited = exc
        if isinstance(self._cited, SliceUnavailable):
            raise self._cited
        return self._cited

    def keys_device_live(self, device: str, platform: Optional[str]) -> bool:
        """Whether any ledger of ``platform`` on this tree runs ``device`` on any entry."""
        for table, rel in LEDGER_OF_TABLE.items():
            document = self.document(rel)
            if platform_of_table(table) != platform or not isinstance(document, Mapping):
                continue
            if _keys_device(document, device, list(document), table):
                return True
        return False

    def pins_holding(self, platform: Optional[str]) -> Tuple[set, Dict[str, set]]:
        """``(every file pinned, {file: digests})`` over the platform's ledgers. The digests
        of a file come only from entries whose other package pins hold at this tree's bytes:
        those entries describe this tree, whatever their own pin of that file says. A value
        that is no digest pins nothing, and it moves the bound digest anyway."""
        if platform not in self._pins:
            known: set = set()
            holding: Dict[str, set] = {}
            for table, rel in LEDGER_OF_TABLE.items():
                document = self.document(rel)
                if platform_of_table(table) != platform or not isinstance(document, Mapping):
                    continue
                for key, entry in document.items():
                    bound = entry.get("source_sha256") if isinstance(entry, Mapping) else None
                    if key == ROUTE_ENTRY or not isinstance(bound, Mapping):
                        continue
                    left = {pin for pin, digest in bound.items()
                            if pin.startswith("meep_gpu/") and self.sha256(pin) != digest}
                    for pin, digest in bound.items():
                        if isinstance(digest, str):
                            known.add(pin)
                            if not left - {pin}:
                                holding.setdefault(pin, set()).add(digest)
            self._pins[platform] = (known, holding)
        return self._pins[platform]

    def preference(self) -> Any:
        """``meep_gpu/dispatch_preference.py`` of this tree (standard library only), run by
        path so that nothing imports the package; ``None`` when it will not load. Compiled
        here rather than imported, so judging a tree leaves no bytecode in it."""
        if self._preference is None:
            path = self.root / "meep_gpu" / "dispatch_preference.py"
            name = f"_judged_dispatch_preference_{hashlib.sha256(str(path).encode()).hexdigest()}"
            module = types.ModuleType(name)
            module.__file__ = str(path)
            sys.modules[name] = module
            try:
                exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)  # noqa: S102
                self._preference = module
            except Exception:  # noqa: BLE001 - any failure is "the rule is unavailable"
                sys.modules.pop(name, None)
                self._preference = False
        return self._preference or None

    def own_records_holding(self, architecture: str) -> List[str]:
        """Repository-relative records of ``architecture`` that time it on this tree: stamped
        so, loaded, and past ``timing_refusal``."""
        if architecture in self._holds:
            return self._holds[architecture]
        held: List[str] = []
        dp = self.preference()
        if dp is not None:
            package = str(self.root / "meep_gpu")
            try:
                route = dp.repair_route_sha256(os.path.join(package, "deposit_repair.py"))
                subjects = {table: dp.live_subject_sha256(table, package) for table in dp.TABLES}
                files = dp.record_files(package)
            except Exception:  # noqa: BLE001 - no barrier can be read: nothing holds
                files = []
            for path in files:
                parts = dp.record_name_parts(path)
                if not parts or parts[0] != architecture:
                    continue
                try:
                    record = dp.load_record(path)
                    good = (dp.record_capabilities(record) == (architecture,)
                            and dp.timing_refusal(record, repair_route_sha256=route,
                                                  subject_sha256_by_table=subjects) is None)
                except Exception:  # noqa: BLE001 - a record that refuses holds nothing
                    good = False
                if good:
                    held.append(Path(path).relative_to(self.root).as_posix())
        self._holds[architecture] = held
        return held


_TREE_KEY = "\0ledger_scope"


def _tree(root: Any, cache: Optional[Dict[str, Any]]) -> _Tree:
    """The tree's reads, kept in the comparison's own ``cache`` (the map a tool already
    threads through ``artifact_drift``, keyed by file name) and so never older than it."""
    if cache is None:
        return _Tree(root)
    held = cache.get(_TREE_KEY)
    if not isinstance(held, _Tree) or held.root != Path(root):
        held = cache[_TREE_KEY] = _Tree(root)
    return held


# ---------------------------------------------------------------------------------------
# The comparison
# ---------------------------------------------------------------------------------------

def _whole(why: str, **proof: Any) -> Verdict:
    return Verdict(True, why, proof)


def _text_read_gaps(tree: _Tree, imports: Mapping[str, Any], platform: Optional[str]
                    ) -> Tuple[List[str], List[str]]:
    """``(unpinned, left_its_pin)``: files this process's closure read as text, that its import
    record does not name, and that no entry of the platform's ledgers pins, or that are not at
    the one set of bytes :meth:`_Tree.pins_holding` answers. Nothing else can show them moving:
    the bound digest moves on a re-gate, and an edit that was never re-gated moves no entry."""
    unpinned, left = set(), set()
    for reader, targets in TEXT_READS.items():
        # A process imports modules of a table it never emits from: the Metal arm table
        # imports Triton's. Only its own platform's emitters read anything for it.
        table = reader.split("/")[1][:-len("_kernels")]
        if reader not in imports or platform_of_table(table) != platform:
            continue
        known, holding = tree.pins_holding(platform)
        for name in targets:
            target = f"{reader.rsplit('/', 1)[0]}/{name}"
            if target in imports:
                continue
            if target not in known:
                unpinned.add(target)
            elif holding.get(target) != {tree.sha256(target)}:
                left.add(target)
    return sorted(unpinned), sorted(left)


def ledger_moved(name: str, recorded: Any, *, form: str, device: Any,
                 imports: Optional[Mapping[str, Any]], triton_version: Any, root: Any,
                 cache: Optional[Dict[str, Any]] = None) -> Verdict:
    """Whether the ledger file ``name`` moved for a process of ``device`` that recorded
    ``recorded`` for it (``form``: ``view`` is ``plan_key``'s, ``raw`` the bench's). The one
    comparison ``plan_carry`` and ``timing_rows`` call. Whatever cannot be proved moves under
    a reason of its own, on the whole digest, as a run always was compared, and that includes
    a comparison that fails: one bad file must not end a whole cut."""
    try:
        return _ledger_moved(name, recorded, form, device, imports, triton_version, root, cache)
    except Exception as exc:  # noqa: BLE001 - any failure compares whole, by name
        return _whole("comparison_failed", name=name, detail=repr(exc))


def _ledger_moved(name: str, recorded: Any, form: str, device: Any,
                  imports: Optional[Mapping[str, Any]], triton_version: Any, root: Any,
                  cache: Optional[Dict[str, Any]]) -> Verdict:
    if not isinstance(recorded, str) or not _SHA256.match(recorded):
        return _whole("ledger_digest_unrecorded", recorded=recorded)
    tree = _tree(root, cache)
    live = tree.digest(name, form)
    if live == recorded:
        return Verdict(False, "digest_equal", {"recorded": recorded})
    architecture = architecture_of_device(device)
    platform = platform_of_device(device)
    if architecture is None:
        return _whole("device_not_placed", device=device)
    if name in UNREAD_LEDGERS:
        if not tree.keys_device_live(architecture, platform):
            return _whole("device_names_no_run", device=device)
        return Verdict(False, "unread", {"recorded": recorded})
    match = _FINGERPRINTS.fullmatch(str(name))
    table = match.group("table") if match else None
    if table not in LEDGER_OF_TABLE:
        return _whole("not_a_kernel_ledger", name=name)
    if platform_of_table(table) != platform:
        return _whole("other_platform_recorded", table=table)
    constant_file = CITED_CONSTANTS[table][0]
    imported = (imports or {}).get(constant_file)
    if imported is None or imported != tree.sha256(constant_file):
        return _whole("cited_source_not_in_closure", file=constant_file)
    unpinned, left_its_pin = _text_read_gaps(tree, imports or {}, platform)
    if unpinned:
        return _whole("text_read_unpinned", files=unpinned)
    if left_its_pin:
        return _whole("text_read_left_its_pin", files=left_its_pin)
    memo = (name, recorded, form, architecture,
            None if triton_version is None else str(triton_version))
    held = tree.slices.get(memo)
    if held is None:
        held = tree.slices[memo] = _compare_slices(
            tree, name, table, recorded, form, architecture, triton_version)
    return held


def _compare_slices(tree: _Tree, name: str, table: str, recorded: str, form: str,
                    architecture: str, triton_version: Any) -> Verdict:
    try:
        welds = tree.cited()
    except SliceUnavailable as exc:
        return _whole("cited_unreadable", detail=str(exc))
    try:
        old = (tree.states.view if form == "view" else tree.states.raw)(table, recorded)
    except LedgerStateRefused as exc:
        return _whole("ledger_state_refused", detail=str(exc))
    if old is None:
        return _whole("ledger_state_not_archived", recorded=recorded, table=table)
    new = tree.document(name)
    if not isinstance(new, Mapping):
        return _whole("live_ledger_unreadable", file=name)
    if UNREADABLE_KEY in old or UNREADABLE_KEY in new:
        return _whole("ledger_marked_unreadable", file=name)
    if not _keys_device(old, architecture, welds[table], table):
        return _whole("device_names_no_run", device=architecture, table=table)
    drift = sorted(set(stated_drift(old)) | set(stated_drift(new)))
    if drift:
        return _whole("pin_drift_stated", entries=drift)
    before = ledger_slice(old, table, architecture, welds, triton_version=triton_version)
    after = ledger_slice(new, table, architecture, welds, triton_version=triton_version)
    if before != after:
        fields = _differences(before, after)
        moved_keys = {field.split(".")[1] for field in fields
                      if field.startswith(("welds.", "other_bound."))}
        pins = _pins_moved(old, new, moved_keys)
        return _whole("slice_moved", fields=fields, pins_moved=pins,
                      harness_only=bool(pins) and not any(
                          pin.startswith("meep_gpu/") for pin in pins))
    digest = slice_sha256(new, table, architecture, welds, triton_version=triton_version)
    return Verdict(False, "slice_equal", {"recorded": recorded, "device": architecture,
                                          "slice_sha256": digest})


def _vendor_of(dp: Any, architecture: str) -> Optional[str]:
    if dp.CAPABILITY.match(architecture):
        return "nvidia"
    if dp.APPLE_FAMILY.match(architecture):
        return "apple"
    return None


def timing_record_moved(name: str, device: Any, recorded_records: Any, live_records: Any,
                        root: Any, platforms: Collection[Optional[str]] = (),
                        cache: Optional[Dict[str, Any]] = None) -> Verdict:
    """Whether a moved timing record ``name`` can move a plan of ``device``'s (the module
    docstring). The prior loads every file by path and pools by what it holds, not by its
    name or folder, so nothing is taken on the name's word: a record is set aside only when
    it loads, holds its own folder's table and stamps its own name's architecture, the one
    shape another vendor's plan never reads. Anything else is compared, under a reason of its
    own, and that includes a comparison that fails."""
    try:
        return _timing_record_moved(name, device, recorded_records, live_records, root,
                                    platforms, cache)
    except Exception as exc:  # noqa: BLE001 - any failure compares whole, by name
        return _whole("comparison_failed", name=name, detail=repr(exc))


def _timing_record_moved(name: str, device: Any, recorded_records: Any, live_records: Any,
                         root: Any, platforms: Collection[Optional[str]],
                         cache: Optional[Dict[str, Any]]) -> Verdict:
    architecture = architecture_of_device(device)
    if architecture is None:
        return _whole("device_not_placed", device=device)
    tree = _tree(root, cache)
    own_platform = platform_of_device(device)
    if not tree.keys_device_live(architecture, own_platform):
        return _whole("device_names_no_run", device=device)
    dp = tree.preference()
    if dp is None:
        return _whole("timing_rule_unavailable")
    match = _TIMING_RECORD.fullmatch(str(name))
    parts = dp.record_name_parts(str(name)) if match else None
    if parts is None:
        return _whole("timing_name_unread", name=name)
    named = parts[0]
    table = match.group("table")
    path = tree.root / name
    if not path.is_file():
        return _whole("timing_record_absent", name=name)
    if dp.architecture_refusal(table, named) is not None:
        return _whole("timing_name_on_another_table", name=name)
    try:
        record = dp.load_record(str(path))
    except Exception as exc:  # noqa: BLE001 - validate also raises unnamed
        return _whole("timing_record_refused", detail=repr(exc))
    if record.get("table") != table:
        return _whole("timing_record_of_another_table", name=name, table=record.get("table"))
    stamped = tuple(dp.record_capabilities(record))
    if stamped != (named,):
        return _whole("timing_name_disagrees_with_stamp", name=name, stamped=list(stamped))
    if named == architecture:
        return _whole("own_architecture", name=name)
    vendor, own_vendor = _vendor_of(dp, named), _vendor_of(dp, architecture)
    if vendor is None or own_vendor is None:
        return _whole("timing_architecture_unreadable", name=name)
    if vendor != own_vendor:
        other = {"nvidia": "nvidia", "apple": "metal"}[vendor]
        if other in platforms:
            return _whole("other_platform_recorded", name=name)
        return Verdict(False, "other_vendor", {"name": name})
    if (dp.architecture_number(vendor, named) is None
            or dp.architecture_number(vendor, architecture) is None):
        return _whole("timing_architecture_unreadable", name=name)
    recorded = recorded_records if isinstance(recorded_records, Mapping) else {}
    live = live_records if isinstance(live_records, Mapping) else {}
    for held in tree.own_records_holding(architecture):
        if recorded.get(held) is not None and recorded.get(held) == live.get(held):
            return Verdict(False, "own_architecture_holds", {"name": name, "held_by": held})
    return _whole("own_architecture_not_held", name=name, device=architecture)


def states_not_archived(comparisons: Iterable[Tuple[Mapping[str, Any], Mapping[str, Any]]]
                        ) -> Tuple[List[str], int]:
    """``(digests no archived state answers for, moved ledger digests in all)`` over
    ``(set_aside, compared_whole)`` pairs: what ``status`` and ``carry.py`` print, because a
    row naming such a digest compares whole, permanently, until its state is added."""
    missing, seen = set(), set()
    for set_aside, compared_whole in comparisons:
        for holder in (set_aside, compared_whole):
            for name, entry in holder.items():
                if str(name) in UNREAD_LEDGERS or not _FINGERPRINTS.fullmatch(str(name)):
                    continue
                recorded = entry.get("recorded")
                if isinstance(recorded, str):
                    seen.add(recorded)
                    if entry.get("why") == "ledger_state_not_archived":
                        missing.add(recorded)
    return sorted(missing), len(seen)


# ---------------------------------------------------------------------------------------
# Adding a state
# ---------------------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", maxsplit=1)[0])
    sub = parser.add_subparsers(dest="command", required=True)
    archive = sub.add_parser("archive", help="file a ledger state under its digest")
    archive.add_argument("--table", required=True, choices=sorted(LEDGER_OF_TABLE))
    archive.add_argument("--root", default=str(Path(__file__).resolve().parents[2]))
    archive.add_argument("file", help="the ledger bytes, from any copy")
    args = parser.parse_args(argv)
    try:
        path = archive_state(args.root, args.table, args.file)
    except (OSError, ValueError, RecursionError) as exc:
        print(f"REFUSED {args.file}: {exc}", flush=True)
        return 1
    print(f"archived {Path(path).relative_to(args.root).as_posix()}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
