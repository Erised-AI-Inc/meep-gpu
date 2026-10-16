"""MEEP identity: the digest schema the three test tiers share, and the Tier A replay.

MEEP IDENTITY is the property that every stored float32 word of the twelve field
components (real and imaginary parts, the sign of zero included) on owned and wall cells
equals the reference MEEP's word at every compared step, and that every word
``get_array`` returns equals MEEP's. It is tested in three tiers
(``docs/development/testing.md``, "MEEP identity"):

* **Tier C**, the MEEP lane (``parity/meep_gpu/meep_identity_tier_c.py``): where a
  verified identity-reference MEEP is installed, it lifts a case, steps the lifted driver
  and MEEP in pair order, compares every word in process, and writes what Tier A replays.
* **Tier A**, the replay (this module, judged in the default package suite): a capsule of
  the lift (:mod:`.capsule`), the currents the lifted run's sources returned, and the
  digests of MEEP's words. :func:`replay` steps the capsule with no MEEP and judges its
  digests against MEEP's. It needs NumPy only.
* **Tier B**, archive words, lives in the evidence archive and is not read here.

THE DIGEST of one (case, compared step, component) is the SHA-256 of the component's
words as little-endian 32-bit words, in the stored array's C order (x, then y, then z),
restricted to the COMPARED cells (:func:`cell_classes`): every stored cell except the
mirror ghosts. MEEP stores no ghost cell; its ghost word is a connection of an owned word,
and the package's ghost storage is never read back (``readback.chunk_words``), so a ghost
is reported, never hashed. A complex component's words are its real and imaginary parts
interleaved, as ``complex64`` lays them out. The READER digest is the SHA-256 of the words
``get_array`` returns for the whole cell (``Fields.to_meep_array``, MEEP's
``sim.get_array``), in its own C order. For every compared step up to
:data:`PLANE_STEPS` the record also carries one 8-byte PLANE digest per x-plane of every
component, so a failing replay names the first step, component and plane without stored
words.

THE SCHEDULE of a case with N steps is every step 0 to 20, Table 2's ten checkpoints
``floor(k N / 10 + 1/2)`` for k = 1..10, and N (:func:`compared_steps`).

THE LEDGER ``meep_identity.json`` (beside the fixtures, ``parity/meep_gpu/fixtures/
meep_identity/``, outside the wheel) lists every minted case per subnormal mode with the
sha256 of each fixture file, the Tier C counts it was minted from, the reference MEEP's
identity, and the array-path oracle fingerprint (:mod:`.array_path_oracle`) of the tree it
was minted on. Only ``parity/meep_gpu/record_meep_identity.py`` writes it.

SUBNORMAL MODE. A digest set is minted in one mode, ``keep`` or ``flush``, and is judged
only in that mode: :func:`host_subnormal_mode` drives this thread's FPU to the mode
(``subnormal_policy._drive_host_fpu``: MEEP's own switch when MEEP is imported; with no
MEEP imported, Darwin's ``FE_DFL_DISABLE_DENORMS_ENV`` on Apple silicon and MXCSR
FTZ|DAZ through glibc's ``fenv_t`` on x86-64 Linux), measures it before the first step
and at every compared step, refuses by name when it is not the mode, and restores the
entry state. A ``flush`` set is minted only from an x86 flush reference at
``OMP_NUM_THREADS=1``; its replay runs the array path under FTZ|DAZ on x86-64 and under
FZ on Apple silicon, where it is judged row by row (owner ruling R13).

COMPLEX MULTIPLIES. :func:`watching_complex_multiplies` views the driver's state arrays as
an ndarray subclass whose ``__array_ufunc__`` and ``__array_function__`` record every
multiply of two complex operands (a complex scalar counts). It sees only operations on
arrays that descend from the viewed state: an array the step makes with ``numpy.empty``
and fills by assignment is not watched, so the count is a lower bound and the watch reports
how many multiplies it observed.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Tuple

import numpy

__all__ = [
    "COMPONENTS", "DIGESTS_SCHEMA", "DIGESTS_VERSION", "FIXTURE_BUDGET_BYTES",
    "FIXTURES_ENVIRONMENT", "FIXTURES_RELATIVE", "IdentityRefusal", "LEDGER_FILE",
    "LEDGER_SCHEMA", "LEDGER_VERSION", "MODES", "PLANE_DIGEST_BYTES", "PLANE_STEPS",
    "ReplayResult", "StepDigests", "cell_classes", "compare_planes", "compared_steps",
    "compared_words", "digest_words", "fixtures_absence", "fixtures_root", "host_subnormal_mode",
    "plane_digests", "read_digests", "read_ledger", "reader_words", "replay", "step_digests",
    "subnormal_state", "table2_checkpoints", "watching_complex_multiplies", "write_digests",
    "WatchedArray", "flip_word", "flipped_planes", "word_cell",
]

#: The twelve stored components, in the order every record lists them.
COMPONENTS: Tuple[str, ...] = ("Ex", "Ey", "Ez", "Dx", "Dy", "Dz",
                               "Hx", "Hy", "Hz", "Bx", "By", "Bz")
LEDGER_SCHEMA = "meep_gpu.meep_identity"
LEDGER_VERSION = 1
DIGESTS_SCHEMA = "meep_gpu.meep_identity_digests"
DIGESTS_VERSION = 1
MODES: Tuple[str, ...] = ("keep", "flush")
#: Plane digests are recorded for every compared step up to and including this one.
PLANE_STEPS = 20
PLANE_DIGEST_BYTES = 8
#: The fixtures directory, relative to the repository root, and the ledger in it.
FIXTURES_RELATIVE = "parity/meep_gpu/fixtures/meep_identity"
LEDGER_FILE = "meep_identity.json"
#: Names another fixtures directory (a set minted to a staging directory, or an absent
#: one): read by :func:`fixtures_root`.
FIXTURES_ENVIRONMENT = "MEEP_IDENTITY_FIXTURES"
#: The Tier A fixture budget in the repository (owner ruling R6), measured at mint.
FIXTURE_BUDGET_BYTES = 5_000_000
#: The files of one minted case, by role, relative to the fixtures directory.
CASE_FILES: Tuple[str, ...] = ("capsule", "currents", "digests", "planes")


class IdentityRefusal(ValueError):
    """A record, a fixture or a host state that a MEEP-identity tier refuses, by name."""


# ---------------------------------------------------------------------------------
# The schedule
# ---------------------------------------------------------------------------------

def table2_checkpoints(n_steps: int) -> List[int]:
    """Table 2's checkpoints: ``{20} | {floor(k N / 10 + 1/2) : k = 1..10}`` within 1..N.

    ``floor(k N / 10 + 1/2)`` is computed in integers as ``(2 k N + 10) // 20``.
    """
    n_steps = int(n_steps)
    points = {(2 * k * n_steps + 10) // 20 for k in range(1, 11)}
    if n_steps >= 20:
        points.add(20)
    return sorted(point for point in points if 1 <= point <= n_steps)


def compared_steps(n_steps: int) -> List[int]:
    """Every step 0..min(20, N), Table 2's checkpoints, and N."""
    n_steps = int(n_steps)
    if n_steps < 0:
        raise IdentityRefusal(f"a run of {n_steps} steps has no schedule")
    steps = set(range(0, min(PLANE_STEPS, n_steps) + 1))
    steps.update(table2_checkpoints(n_steps))
    steps.add(n_steps)
    return sorted(steps)


# ---------------------------------------------------------------------------------
# Cells and words
# ---------------------------------------------------------------------------------

def _face(axis: int, index: int) -> Tuple[Any, ...]:
    return (slice(None),) * axis + (index,)


def cell_classes(grid: Any, component: str) -> Dict[str, numpy.ndarray]:
    """Boolean masks over the stored array of one component: ``ghost``, ``wall``, ``compared``.

    * ``ghost``: a mirror ghost. On a folded axis, stored cell 0 of a component whose Yee
      shift there is 0 (doubled -2, connected to its image at stored cell 2), and on a
      folded PERIODIC axis the last stored slot of a component whose shift is 1 (MEEP's
      not-owned far slot, ``stepping._stored_past_owned``).
    * ``wall``: a cell the owned-cell rule drops (``stepping._mask_non_owned_cells``) that
      is not a ghost: a metallic low wall, the cylindrical axis row. MEEP stores it.
    * ``compared``: every stored cell that is not a ghost (owned and wall cells).
    """
    from .fields import IYEE_SHIFTS  # noqa: PLC0415
    from .stepping import _mask_non_owned_cells, _stored_past_owned  # noqa: PLC0415

    if component not in IYEE_SHIFTS:
        raise IdentityRefusal(f"unknown component {component!r}; the components are {COMPONENTS}")
    shape = (int(grid.nx), int(grid.ny), int(grid.nz))
    shifts = IYEE_SHIFTS[component]
    ghost = numpy.zeros(shape, dtype=bool)
    for axis in range(3):
        if not grid.is_mirrored(axis):
            continue
        if shifts[axis] == 0:
            ghost[_face(axis, 0)] = True
        elif _stored_past_owned(grid, axis):
            ghost[_face(axis, -1)] = True
    ones = numpy.ones(shape, dtype=numpy.float32)
    _mask_non_owned_cells(ones, grid, shifts)
    dropped = ones == 0
    return {"ghost": ghost, "wall": dropped & ~ghost, "compared": ~ghost}


def _word_view(array: Any) -> numpy.ndarray:
    """The stored words of an array: ``uint32`` of shape ``array.shape + (parts,)``."""
    values = numpy.ascontiguousarray(array)
    if values.dtype == numpy.complex64:
        return values.view(numpy.float32).reshape(values.shape + (2,)).view(numpy.uint32)
    if values.dtype == numpy.float32:
        return values.view(numpy.uint32).reshape(values.shape + (1,))
    raise IdentityRefusal(f"stored words are float32 or complex64, not {values.dtype}")


def compared_words(array: Any, mask: numpy.ndarray) -> numpy.ndarray:
    """The words of ``array`` on the cells ``mask`` selects, in C order, parts interleaved."""
    words = _word_view(array)
    if tuple(mask.shape) != tuple(words.shape[:-1]):
        raise IdentityRefusal(f"a mask of shape {tuple(mask.shape)} cannot select the cells "
                              f"of an array of shape {tuple(words.shape[:-1])}")
    return numpy.ascontiguousarray(words[mask]).reshape(-1)


def reader_words(array: Any) -> numpy.ndarray:
    """Every word of a ``get_array`` result, in its C order, parts interleaved."""
    return _word_view(array).reshape(-1)


def digest_words(words: numpy.ndarray) -> str:
    """SHA-256 of 32-bit words, little-endian."""
    return hashlib.sha256(numpy.ascontiguousarray(words, dtype="<u4").tobytes()).hexdigest()


def _planes_of_view(words: numpy.ndarray, mask: numpy.ndarray) -> numpy.ndarray:
    out = numpy.empty((words.shape[0], PLANE_DIGEST_BYTES), dtype=numpy.uint8)
    for plane in range(words.shape[0]):
        selected = numpy.ascontiguousarray(words[plane][mask[plane]], dtype="<u4")
        out[plane] = numpy.frombuffer(
            hashlib.sha256(selected.tobytes()).digest()[:PLANE_DIGEST_BYTES], dtype=numpy.uint8)
    return out


def plane_digests(array: Any, mask: numpy.ndarray) -> numpy.ndarray:
    """One :data:`PLANE_DIGEST_BYTES`-byte digest per x-plane: ``uint8[nx, 8]``."""
    return _planes_of_view(_word_view(array), mask)


def word_cell(mask: numpy.ndarray, parts: int, index: int) -> Tuple[Tuple[int, ...], int]:
    """The stored cell and part of compared word ``index`` (C order, parts interleaved)."""
    cells = numpy.argwhere(mask)
    cell, part = divmod(int(index), int(parts))
    if not 0 <= cell < len(cells):
        raise IdentityRefusal(f"compared word {index} does not exist ({len(cells) * parts} words)")
    return tuple(int(v) for v in cells[cell]), part


def flipped_planes(array: Any, mask: numpy.ndarray, index: int) -> numpy.ndarray:
    """The plane digests of ``array`` with compared word ``index`` flipped by one ulp."""
    view = _word_view(array).copy()
    cell, part = word_cell(mask, view.shape[-1], index)
    view[cell + (part,)] ^= numpy.uint32(1)
    return _planes_of_view(view, mask)


def compare_planes(recorded: numpy.ndarray, found: numpy.ndarray) -> List[int]:
    """The x-planes whose digests differ."""
    if recorded.shape != found.shape:
        raise IdentityRefusal(f"plane digests of shape {found.shape} cannot be compared with "
                              f"the record's {recorded.shape}")
    return [int(plane) for plane in numpy.flatnonzero(numpy.any(recorded != found, axis=1))]


@dataclasses.dataclass
class StepDigests:
    """The digests of one compared step, and the words they were taken from."""

    step: int
    stored: Dict[str, str]
    reader: Dict[str, str]
    planes: Optional[numpy.ndarray]  # uint8[components, nx, 8] when step <= PLANE_STEPS
    words: Dict[str, numpy.ndarray]  # the compared words, kept for a planted flip


def step_digests(driver: Any, masks: Mapping[str, numpy.ndarray], *, planes: bool,
                 reader: bool = True) -> StepDigests:
    """The digests of the driver's words at its current step."""
    stored, read, words = {}, {}, {}
    plane_rows = []
    alias = driver.h_alias_rule() if reader else None
    for name in COMPONENTS:
        array = driver.get_field(name, cell_centered=False)
        words[name] = compared_words(array, masks[name])
        stored[name] = digest_words(words[name])
        if planes:
            plane_rows.append(plane_digests(array, masks[name]))
        if reader:
            read[name] = digest_words(reader_words(driver.fields.to_meep_array(name, alias)))
    return StepDigests(step=int(driver.step_count), stored=stored, reader=read,
                       planes=numpy.stack(plane_rows) if planes else None, words=words)


def flip_word(words: numpy.ndarray, index: int) -> numpy.ndarray:
    """A copy with the least significant bit of word ``index`` flipped: one ulp."""
    flipped = numpy.array(words, dtype=numpy.uint32, copy=True)
    flipped[int(index)] ^= numpy.uint32(1)
    return flipped


# ---------------------------------------------------------------------------------
# The digest record of one case
# ---------------------------------------------------------------------------------

def write_digests(directory: Path, document: Dict[str, Any], planes: numpy.ndarray) -> Dict[str, Dict[str, Any]]:
    """Write ``digests.json`` and ``planes.npy`` into ``directory``; ``{role: {file, sha256, bytes}}``."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    plane_path = directory / "planes.npy"
    with open(plane_path, "wb") as handle:
        numpy.save(handle, numpy.ascontiguousarray(planes, dtype=numpy.uint8), allow_pickle=False)
    plane_bytes = plane_path.read_bytes()
    document = dict(document)
    document["planes"] = dict(document["planes"], sha256=hashlib.sha256(plane_bytes).hexdigest(),
                              shape=list(planes.shape))
    text = json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest_path = directory / "digests.json"
    digest_path.write_bytes(text)
    return {"digests": {"sha256": hashlib.sha256(text).hexdigest(), "bytes": len(text)},
            "planes": {"sha256": hashlib.sha256(plane_bytes).hexdigest(), "bytes": len(plane_bytes)}}


def read_digests(digest_path: Path, plane_path: Path) -> Tuple[Dict[str, Any], numpy.ndarray]:
    """A case's digest record and its plane digests, after the record's checks."""
    document = json.loads(Path(digest_path).read_bytes().decode("utf-8"))
    if document.get("schema") != DIGESTS_SCHEMA or document.get("version") != DIGESTS_VERSION:
        raise IdentityRefusal(
            f"{digest_path} is {document.get('schema')!r} version {document.get('version')!r}; "
            f"this tree reads {DIGESTS_SCHEMA} version {DIGESTS_VERSION} only")
    if tuple(document.get("components") or ()) != COMPONENTS:
        raise IdentityRefusal(f"{digest_path} lists components {document.get('components')}, "
                              f"not {list(COMPONENTS)}")
    steps = [int(step) for step in document["steps"]]
    if steps != compared_steps(int(document["n_steps"])):
        raise IdentityRefusal(f"{digest_path}: its steps are not the schedule of "
                              f"{document['n_steps']} steps")
    for kind in ("stored", "reader"):
        if sorted(document[kind], key=int) != [str(step) for step in steps]:
            raise IdentityRefusal(f"{digest_path}: its {kind} digests do not cover its steps")
        for step in steps:
            if sorted(document[kind][str(step)]) != sorted(COMPONENTS):
                raise IdentityRefusal(f"{digest_path}: step {step} lacks {kind} digests of "
                                      f"some components")
    plane_bytes = Path(plane_path).read_bytes()
    if hashlib.sha256(plane_bytes).hexdigest() != document["planes"]["sha256"]:
        raise IdentityRefusal(f"{plane_path} fails the sha256 its digest record holds")
    import io  # noqa: PLC0415

    planes = numpy.load(io.BytesIO(plane_bytes), allow_pickle=False)
    plane_steps = [int(step) for step in document["planes"]["steps"]]
    if plane_steps != [step for step in steps if step <= PLANE_STEPS]:
        raise IdentityRefusal(f"{digest_path}: its plane digests do not cover steps 0..{PLANE_STEPS}")
    if list(planes.shape) != [len(plane_steps), len(COMPONENTS)] + list(planes.shape[2:]) \
            or planes.dtype != numpy.uint8 or planes.shape[-1] != PLANE_DIGEST_BYTES:
        raise IdentityRefusal(f"{plane_path} has shape {planes.shape} {planes.dtype}, not "
                              f"[{len(plane_steps)}, {len(COMPONENTS)}, planes, {PLANE_DIGEST_BYTES}] uint8")
    return document, planes


# ---------------------------------------------------------------------------------
# The fixtures and the ledger
# ---------------------------------------------------------------------------------

def fixtures_root() -> Path:
    """The fixtures directory: :data:`FIXTURES_ENVIRONMENT` when set, else the repository's."""
    named = os.environ.get(FIXTURES_ENVIRONMENT)
    if named:
        return Path(named)
    return Path(__file__).resolve().parent.parent / FIXTURES_RELATIVE


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_ledger(root: Optional[Path] = None, *, check_files: bool = True) -> Dict[str, Any]:
    """The ledger under ``root`` (default :func:`fixtures_root`), after its checks.

    Every case entry must name its four files; with ``check_files`` each file must exist
    with the size and sha256 the ledger records, and the files must total the recorded
    ``fixture_bytes``, within :data:`FIXTURE_BUDGET_BYTES`.
    """
    root = Path(root) if root is not None else fixtures_root()
    path = root / LEDGER_FILE
    try:
        ledger = json.loads(path.read_bytes().decode("utf-8"))
    except FileNotFoundError:
        raise IdentityRefusal(f"{path} does not exist") from None
    except (OSError, ValueError) as exc:
        raise IdentityRefusal(f"{path} is not readable JSON ({exc})") from None
    if not isinstance(ledger, dict) or ledger.get("schema") != LEDGER_SCHEMA \
            or ledger.get("version") != LEDGER_VERSION:
        found = (ledger.get("schema"), ledger.get("version")) if isinstance(ledger, dict) else None
        raise IdentityRefusal(f"{path} is {found!r}; this tree reads {LEDGER_SCHEMA} version "
                              f"{LEDGER_VERSION} only")
    sets = ledger.get("sets")
    if not isinstance(sets, dict) or not sets:
        raise IdentityRefusal(f"{path} holds no digest set")
    unknown = sorted(set(sets) - set(MODES))
    if unknown:
        raise IdentityRefusal(f"{path} holds digest sets of unknown modes {unknown}")
    total = 0
    for mode, entry in sets.items():
        cases = entry.get("cases")
        if not isinstance(cases, dict) or not cases:
            raise IdentityRefusal(f"{path}: the {mode} set holds no case")
        for case, files in cases.items():
            for role in CASE_FILES:
                spec = (files.get("files") or {}).get(role)
                if not isinstance(spec, dict) or not {"file", "sha256", "bytes"} <= set(spec):
                    raise IdentityRefusal(f"{path}: {mode}/{case} names no {role} file")
                if not check_files:
                    continue
                target = root / spec["file"]
                if not target.is_file():
                    raise IdentityRefusal(f"{path}: {mode}/{case} {role} file {spec['file']} "
                                          f"does not exist")
                size = target.stat().st_size
                if size != int(spec["bytes"]) or _sha256_file(target) != spec["sha256"]:
                    raise IdentityRefusal(f"{path}: {mode}/{case} {role} file {spec['file']} is "
                                          f"not the file the ledger records (size or sha256)")
                total += size
    if check_files:
        if total != int(ledger.get("fixture_bytes", -1)):
            raise IdentityRefusal(f"{path}: the files total {total} bytes; the ledger records "
                                  f"{ledger.get('fixture_bytes')}")
        if total > FIXTURE_BUDGET_BYTES:
            raise IdentityRefusal(f"{path}: the fixtures take {total} bytes, over the "
                                  f"{FIXTURE_BUDGET_BYTES}-byte budget")
    return ledger


def fixtures_absence(root: Optional[Path] = None) -> Optional[str]:
    """Why there is no Tier A fixture set to replay, or ``None`` when there is one.

    An installed wheel carries no ``parity/``, so the directory is absent there; a
    checkout carries it. A ledger that is present but does not read is a reason too, so a
    fixture set that went wrong is named rather than skipped silently by the replay.
    """
    root = Path(root) if root is not None else fixtures_root()
    if not root.is_dir():
        return (f"{root} does not exist: the MEEP-identity fixtures are part of the "
                f"repository checkout ({FIXTURES_RELATIVE}), not of an installed wheel")
    try:
        read_ledger(root, check_files=False)
    except IdentityRefusal as exc:
        return f"the MEEP-identity fixtures do not read: {exc}"
    return None


# ---------------------------------------------------------------------------------
# The host FPU
# ---------------------------------------------------------------------------------

def subnormal_state() -> Dict[str, Any]:
    """This thread's subnormal handling, measured in float32 and float64.

    Flush-to-zero from a product of two normal operands whose exact result is subnormal;
    denormals-are-zero from a subnormal operand times a power of two whose exact result is
    normal. ``mode`` is ``keep`` when none flushes, ``flush`` when all four do, ``mixed``
    otherwise.
    """
    tiny32 = numpy.frombuffer(b"\x01\x00\x00\x00", dtype=numpy.float32)[0]
    tiny64 = numpy.frombuffer(b"\x01" + b"\x00" * 7, dtype=numpy.float64)[0]
    state = {
        "float32_flush_to_zero": bool(numpy.float32(1e-38) * numpy.float32(0.25) == 0),
        "float32_denormals_are_zero": bool(tiny32 * numpy.float32(2.0 ** 100) == 0),
        "float64_flush_to_zero": bool(numpy.float64(1e-300) * numpy.float64(1e-16) == 0),
        "float64_denormals_are_zero": bool(tiny64 * numpy.float64(2.0 ** 600) == 0),
    }
    flushed = sum(state.values())
    state["mode"] = "keep" if flushed == 0 else "flush" if flushed == 4 else "mixed"
    return state


@contextlib.contextmanager
def host_subnormal_mode(mode: str) -> Iterator[Dict[str, Any]]:
    """Run with this thread's FPU in ``mode``; refuse by name when it cannot be; restore.

    The report carries the state at entry, the mechanism that drove it (or ``None``), and
    a ``check`` callable that re-measures and refuses if the mode moved.
    """
    if mode not in MODES:
        raise IdentityRefusal(f"unknown subnormal mode {mode!r}; the modes are {MODES}")
    from . import subnormal_policy  # noqa: PLC0415

    entry = subnormal_state()
    report: Dict[str, Any] = {"mode": mode, "entry": entry, "driven_by": None, "checks": 0}
    if entry["mode"] != mode:
        _module, mechanism = subnormal_policy._drive_host_fpu(mode == "flush")  # noqa: SLF001
        report["driven_by"] = mechanism
    measured = subnormal_state()
    report["measured"] = measured

    def check(where: str) -> None:
        state = subnormal_state()
        report["checks"] += 1
        if state["mode"] != mode:
            raise IdentityRefusal(f"the host FPU is {state['mode']} at {where}, not {mode}: "
                                  f"something in the step changed it")

    report["check"] = check
    try:
        if measured["mode"] != mode:
            raise IdentityRefusal(
                f"the host FPU is {measured['mode']} and could not be driven to {mode} "
                f"({report['driven_by'] or 'no mechanism tried'}); a {mode} digest set is "
                f"judged only in {mode} (the levers are MEEP's set_zero_subnormals when "
                f"MEEP is imported and, without MEEP, Darwin's FE_DFL_DISABLE_DENORMS_ENV "
                f"or MXCSR FTZ|DAZ on x86-64 glibc; on any other ABI a MEEP-less {mode} "
                f"is refused)")
        yield report
    finally:
        if report["driven_by"] is not None and subnormal_state()["mode"] != entry["mode"]:
            subnormal_policy._drive_host_fpu(entry["mode"] == "flush")  # noqa: SLF001
        report.pop("check", None)


# ---------------------------------------------------------------------------------
# The complex-multiply interceptor
# ---------------------------------------------------------------------------------

_MULTIPLY_FUNCTIONS = ("dot", "vdot", "inner", "outer", "tensordot", "einsum", "kron", "cross")


def _complex_operand(value: Any) -> bool:
    if isinstance(value, complex):
        return True
    dtype = getattr(value, "dtype", None)
    return dtype is not None and numpy.dtype(dtype).kind == "c"


def _site() -> str:
    for frame in reversed(traceback.extract_stack()[:-3]):
        if frame.filename != __file__ and "numpy" not in Path(frame.filename).parts:
            return f"{Path(frame.filename).name}:{frame.lineno}"
    return "unknown"


class _Watch:
    """What one :func:`watching_complex_multiplies` block observed."""

    def __init__(self):
        self.multiplies = 0
        self.complex_multiplies = 0
        self.sites: Dict[str, int] = {}
        self.functions: Dict[str, int] = {}

    def saw(self, complex_pair: bool) -> None:
        self.multiplies += 1
        if complex_pair:
            self.complex_multiplies += 1
            site = _site()
            self.sites[site] = self.sites.get(site, 0) + 1

    def report(self) -> Dict[str, Any]:
        return {"multiplies_observed": self.multiplies,
                "complex_times_complex": self.complex_multiplies,
                "sites": dict(sorted(self.sites.items())),
                "functions_observed": dict(sorted(self.functions.items()))}


_ACTIVE: List[_Watch] = []


def _plain(value: Any) -> Any:
    return value.view(numpy.ndarray) if isinstance(value, WatchedArray) else value


class WatchedArray(numpy.ndarray):
    """An ndarray whose multiplies of two complex operands are recorded while a watch is open."""

    def __array_ufunc__(self, ufunc: Any, method: str, *inputs: Any, **kwargs: Any) -> Any:
        if _ACTIVE and ufunc in (numpy.multiply, numpy.matmul) and method in ("__call__", "outer"):
            _ACTIVE[-1].saw(len(inputs) == 2 and all(_complex_operand(x) for x in inputs))
        plain_inputs = tuple(_plain(x) for x in inputs)
        outs = kwargs.get("out")
        if outs is not None:
            kwargs["out"] = tuple(_plain(x) for x in outs)
        result = getattr(ufunc, method)(*plain_inputs, **kwargs)
        if outs is not None:
            return outs[0] if len(outs) == 1 else outs
        if isinstance(result, tuple):
            return tuple(r.view(WatchedArray) if isinstance(r, numpy.ndarray) else r for r in result)
        if isinstance(result, numpy.ndarray):
            return result.view(WatchedArray)
        return result

    def __array_function__(self, func: Any, types: Any, args: Any, kwargs: Any) -> Any:
        name = getattr(func, "__name__", "")
        if _ACTIVE and name in _MULTIPLY_FUNCTIONS:
            watch = _ACTIVE[-1]
            watch.functions[name] = watch.functions.get(name, 0) + 1
            operands = [a for a in args if isinstance(a, (numpy.ndarray, complex))]
            watch.saw(len(operands) >= 2 and all(_complex_operand(a) for a in operands[:2]))
        return super().__array_function__(func, types, args, kwargs)


def _package_object(value: Any) -> bool:
    module = type(value).__module__ or ""
    return module == "meep_gpu" or module.startswith("meep_gpu.")


def _view_arrays(value: Any, seen: Dict[int, Any], restore: List[Callable[[], None]]) -> Any:
    """``value`` with every plain ndarray below it viewed as :class:`WatchedArray`.

    One array reached under two paths gets one view, so an alias stays an alias.
    """
    if type(value) is numpy.ndarray:
        view = seen.get(id(value))
        if view is None:
            view = seen[id(value)] = value.view(WatchedArray)
        return view
    if id(value) in seen:
        return value
    if isinstance(value, list):
        seen[id(value)] = value
        for index, item in enumerate(value):
            replaced = _view_arrays(item, seen, restore)
            if replaced is not item:
                restore.append(lambda v=value, i=index, o=item: v.__setitem__(i, o))
                value[index] = replaced
        return value
    if isinstance(value, dict):
        seen[id(value)] = value
        for key in list(value):
            item = value[key]
            replaced = _view_arrays(item, seen, restore)
            if replaced is not item:
                restore.append(lambda v=value, k=key, o=item: v.__setitem__(k, o))
                value[key] = replaced
        return value
    if _package_object(value) and hasattr(value, "__dict__"):
        seen[id(value)] = value
        for name, item in list(vars(value).items()):
            replaced = _view_arrays(item, seen, restore)
            if replaced is not item:
                restore.append(lambda v=value, n=name, o=item: object.__setattr__(v, n, o))
                object.__setattr__(value, name, replaced)
        return value
    return value


@contextlib.contextmanager
def watching_complex_multiplies(driver: Any) -> Iterator[_Watch]:
    """View the driver's arrays as :class:`WatchedArray` and record complex multiplies.

    The views share memory with the arrays they replace, so the run's words are the run's;
    the originals are put back when the block ends. ``watch.report()`` names every site
    that multiplied two complex operands and counts the multiplies observed.
    """
    seen: Dict[int, Any] = {}
    restore: List[Callable[[], None]] = []
    watch = _Watch()
    _view_arrays(driver, seen, restore)
    _ACTIVE.append(watch)
    try:
        yield watch
    finally:
        _ACTIVE.remove(watch)
        for undo in reversed(restore):
            undo()


# ---------------------------------------------------------------------------------
# The Tier A replay
# ---------------------------------------------------------------------------------

@dataclasses.dataclass
class ReplayResult:
    """What one replay of one minted case measured."""

    case: str
    mode: str
    steps: List[int]
    stored_equal: int
    reader_equal: int
    digests: int
    first_difference: Optional[Dict[str, Any]]
    planted_flip: Dict[str, Any]
    fpu: Dict[str, Any]
    load_report: Optional[Dict[str, Any]]
    seconds: float
    multiply_watch: Optional[Dict[str, Any]] = None

    @property
    def holds(self) -> bool:
        """Every stored and reader digest equal, and the planted flip caught and placed."""
        return (self.stored_equal == self.digests and self.reader_equal == self.digests
                and bool(self.planted_flip.get("caught")) and bool(self.planted_flip.get("placed")))

    def as_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def replay(case: str, *, mode: str = "keep", root: Optional[Path] = None,
           ledger: Optional[Dict[str, Any]] = None, watch: bool = False,
           progress: Optional[Callable[[str], None]] = None) -> ReplayResult:
    """Step a minted case from its capsule and judge every digest; no MEEP is needed.

    The capsule is loaded with the recorded currents injected, so the run's source terms
    are the minting run's, not this host's libm's. At every compared step the stored and
    reader digests of the twelve components are compared with MEEP's; at steps up to
    :data:`PLANE_STEPS` a differing component is localised to its x-planes. At the
    record's planted-flip step one word (the record's) of one component is flipped in a
    copy and must change that component's digest and exactly the plane it lies in.
    """
    from . import capsule  # noqa: PLC0415

    root = Path(root) if root is not None else fixtures_root()
    ledger = ledger if ledger is not None else read_ledger(root)
    entry = ((ledger.get("sets") or {}).get(mode) or {}).get("cases", {}).get(case)
    if entry is None:
        raise IdentityRefusal(f"the ledger under {root} holds no {mode} case {case!r}")
    files = entry["files"]
    document, recorded_planes = read_digests(root / files["digests"]["file"], root / files["planes"]["file"])
    if document["case"] != case or document["mode"] != mode:
        raise IdentityRefusal(f"{files['digests']['file']} is case {document['case']!r} "
                              f"mode {document['mode']!r}, not {case!r} {mode!r}")
    started = time.time()
    stored = capsule.read_capsule(root / files["capsule"]["file"])
    table = capsule.CurrentsTable.read(root / files["currents"]["file"])
    steps = [int(step) for step in document["steps"]]
    plane_steps = [int(step) for step in document["planes"]["steps"]]
    flip_spec = document["planted_flip"]
    stored_equal = reader_equal = 0
    first: Optional[Dict[str, Any]] = None
    flip: Dict[str, Any] = {"step": flip_spec["step"], "component": flip_spec["component"],
                            "word": flip_spec["word"], "caught": False, "placed": False}
    watch_report = None
    with host_subnormal_mode(mode) as fpu:
        driver = stored.build_driver(prefer_gpu=False, currents=table)
        try:
            if driver.gpu is not None:
                raise IdentityRefusal(f"the replay built a GPU driver ({driver.gpu!r}); Tier A "
                                      f"judges the array path")
            masks = {name: cell_classes(driver.grid, name)["compared"] for name in COMPONENTS}
            cells = {name: int(masks[name].sum()) for name in COMPONENTS}
            if cells != {name: int(document["compared_cells"][name]) for name in COMPONENTS}:
                raise IdentityRefusal(f"the loaded driver compares {cells} cells; the record "
                                      f"was minted over {document['compared_cells']}")
            with contextlib.ExitStack() as stack:
                watcher = stack.enter_context(watching_complex_multiplies(driver)) if watch else None
                for number, target in enumerate(steps, 1):
                    while driver.step_count < target:
                        driver.step()
                    fpu["check"](f"step {target}")
                    with_planes = target in plane_steps
                    found = step_digests(driver, masks, planes=with_planes)
                    for name in COMPONENTS:
                        same_stored = found.stored[name] == document["stored"][str(target)][name]
                        same_reader = found.reader[name] == document["reader"][str(target)][name]
                        stored_equal += same_stored
                        reader_equal += same_reader
                        if first is None and not (same_stored and same_reader):
                            first = {"step": target, "component": name,
                                     "stored_equal": same_stored, "reader_equal": same_reader}
                            if with_planes:
                                row = plane_steps.index(target)
                                first["planes_differing"] = compare_planes(
                                    recorded_planes[row, COMPONENTS.index(name)],
                                    found.planes[COMPONENTS.index(name)])[:16]
                    if target == flip_spec["step"]:
                        name = flip_spec["component"]
                        words = flip_word(found.words[name], flip_spec["word"])
                        flip["caught"] = digest_words(words) != document["stored"][str(target)][name]
                        flip["word_plane"] = int(flip_spec["plane"])
                        if with_planes:
                            differing = compare_planes(
                                recorded_planes[plane_steps.index(target), COMPONENTS.index(name)],
                                flipped_planes(driver.get_field(name, cell_centered=False),
                                               masks[name], flip_spec["word"]))
                            flip["planes_differing"] = differing
                            flip["placed"] = differing == [int(flip_spec["plane"])]
                    if progress is not None:
                        progress(f"{case} {mode} step {target} ({number}/{len(steps)}): stored "
                                 f"{stored_equal} / reader {reader_equal} of "
                                 f"{number * len(COMPONENTS)} digests equal "
                                 f"({time.time() - started:.1f} s)")
                if watcher is not None:
                    watch_report = watcher.report()
        finally:
            driver.close()
        fpu_report = {key: value for key, value in fpu.items() if key != "check"}
    return ReplayResult(case=case, mode=mode, steps=steps, stored_equal=stored_equal,
                        reader_equal=reader_equal, digests=len(steps) * len(COMPONENTS),
                        first_difference=first, planted_flip=flip, fpu=fpu_report,
                        load_report=stored.load_report, seconds=round(time.time() - started, 3),
                        multiply_watch=watch_report)
