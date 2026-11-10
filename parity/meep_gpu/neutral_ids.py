"""The neutral spellings of what a run records about its machine: ids, card ids and paths.

A record a run writes leaves the machine that ran it: certification results are sent
back, archived and read on other hosts. So nothing a run WRITES names the machine or the
account it ran under. Every writer of a run record spells its machine stamps through this
module, and every reader that compares or follows one reads the same spellings.

THE RULE

* No host name, login, account, e-mail address, or absolute or home path is written.
* "Same machine": :func:`machine_id`, an anonymous id that is stable per machine and
  cannot be turned back into the machine's name.
* "Same card": a GPU UUID is still the key a check compares in process, but what is
  written is :func:`card_id`, its salted hash.
* A path is written under a named root (:func:`path`): ``<staging-root>`` (the directory
  the checkout lies in, the label of ``neutral_labels.STAGING_ROOT``), ``<home>`` (a home
  directory), ``<tmp>`` (a temporary root, a per-account directory under one included); an
  account's own path component outside those roots is written ``<account>``, and so is its
  name as the leading word of a component or of a hyphen-joined word
  (``<account>_runs``, ``pytest-of-<account>``).
* A host name in free text is written as the machine id (:func:`text`).
* What stays: the GPU model, compute capability, driver and CUDA versions, the CPU model
  and the OS release.

THE MACHINE ID, exactly. The machine's own random 128-bit id is read, first of:
``/etc/machine-id``, ``/var/lib/dbus/machine-id`` (systemd's ``machine-id(5)``; WSL2
distributions carry one), the ``IOPlatformUUID`` of ``ioreg -rd1 -c IOPlatformExpertDevice``
(macOS), else a random 128-bit id this module creates once in
``$XDG_STATE_HOME/meep-gpu/machine-id`` (default ``~/.local/state``). It is never written.
The id written is

    "m-" + HMAC-SHA256(key = the 16 bytes of that id, message = MACHINE_APP_ID)[:16 hex]

the construction systemd recommends for an application-specific machine id
(``sd_id128_get_machine_app_specific``: keyed by the machine id, over a fixed
application id), truncated to 64 bits. The same machine always gives the same id; the
id gives back neither the machine id nor the host name, and no dictionary of host names
can be tried against it. ``m-unknown`` is written only where no source could be read.

THE CARD ID, exactly. ``"c-" + HMAC-SHA256(key = CARD_KEY, message = the UUID, stripped
and lower-cased)[:16 hex]``: the same card always gives the same id, on any machine.

Readers. :func:`resolve` turns a recorded path back into a path on this machine (the
labels expanded), for a reader that executes or opens what a record names: an
interpreter a census row ran under, say. A recorded host field holds the machine id, so a
reader comparing two records' hosts compares machine ids, as before.

Standard library only: drivers, gates and the round's tools import it before anything
else, and the round's tools load it by its path.
"""

from __future__ import annotations

import functools
import getpass
import hashlib
import hmac
import os
import re
import secrets
import socket
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import IO, Any, Dict, Iterable, List, Mapping, Optional, Tuple

#: The application id of :func:`machine_id` (16 fixed random bytes, drawn once for this
#: module). Changing it changes every machine id.
MACHINE_APP_ID = bytes.fromhex("e08a1bd2f460cafc1c7cbee1e76bd0eb")
#: The key of :func:`card_id`.
CARD_KEY = b"meep-gpu card id v1"

STAGING_ROOT = "<staging-root>"
HOME = "<home>"
TMP = "<tmp>"
ACCOUNT = "<account>"
UNKNOWN_MACHINE = "m-unknown"

_HERE = Path(__file__).resolve().parent
#: The checkout this file lies in (``<checkout>/parity/meep_gpu/neutral_ids.py``).
CHECKOUT = _HERE.parents[1]

#: Environment variables whose value names the account, the machine or a connection.
ACCOUNT_VARIABLES = ("USER", "LOGNAME", "USERNAME")
MACHINE_VARIABLES = ("HOSTNAME", "HOST", "COMPUTERNAME", "NAME", "SLURMD_NODENAME",
                     "SLURM_NODELIST", "SLURM_JOB_NODELIST", "SLURM_STEP_NODELIST",
                     "SLURM_SUBMIT_HOST", "SLURM_TOPOLOGY_ADDR", "PBS_O_HOST", "PBS_NODEFILE",
                     "WSL_DISTRO_NAME")
CONNECTION_VARIABLES = ("SSH_CONNECTION", "SSH_CLIENT", "SSH_TTY", "SSH_AUTH_SOCK")

#: Words records carry as values their readers match on (kernel tables and backends, legs,
#: roles, platforms). A host name equal to one of them is not rewritten in free text: the
#: word stands in every record, so rewriting it would break those readers, and the name
#: carries nothing a record does not already say. The writers never write a host name
#: themselves; this rule bounds only the safety net of :func:`text`.
RECORD_WORDS = frozenset({
    "apple", "array", "card", "census", "check", "control", "cpu", "cuda", "cupy", "darwin",
    "default", "device", "examples", "flush", "fused", "gate", "gpu", "host", "identity",
    "keep", "ladder", "linux", "local", "machine", "main", "meep", "metal", "mps", "node",
    "numpy", "nvidia", "other", "own", "probe", "python", "reference", "release", "route",
    "stage", "table", "test", "tests", "torch", "triton", "unfused", "variance"})

_UUID = re.compile(r"(?<![0-9A-Za-z-])(?:GPU|MIG)-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}"
                   r"-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}(?![0-9A-Za-z])")
#: What may stand right before a path a record names: anything but a character of a word or
#: a path, or a one-letter option glued to it (``-I/opt/x``, ``-L/opt/lib``), as compiler and
#: linker flags spell include and library directories.
_BEFORE = r"(?:(?<![A-Za-z0-9._~<>-])|(?<=(?<![A-Za-z0-9_])-[A-Za-z]))"
#: Home and temporary roots by shape, for a path under no root this process knows.
_HOME_SHAPE = re.compile(_BEFORE + r"/(?:home|Users)/(?!user(?:/|$|[^A-Za-z0-9._-]))"
                         r"[A-Za-z0-9._-]+"
                         r"|" + _BEFORE + r"(?:/mnt/[a-z]|\b[A-Za-z]:)[/\\]Users[/\\]"
                         r"[^/\\\s'\"]+")
_TMP_SHAPE = re.compile(_BEFORE + r"(?:/private)?/var/folders/[^/\s'\"]+/[^/\s'\"]+"
                        r"/[A-Za-z]+(?=/|$|[^A-Za-z0-9._-])"
                        r"|" + _BEFORE + r"/private/(?:var/)?tmp(?=/|$|[^A-Za-z0-9._-])")
#: A character that cannot continue a path or a host-name token.
_END = r"(?=$|/|[^A-Za-z0-9._-])"
#: An absolute path token, and the system roots under which no component is an account's.
_ABSOLUTE = re.compile(_BEFORE + r"/[^\s'\"<>,;:()\[\]{}=]+")
#: An absolute path token, or one that begins at a root label this module writes.
_PATH_TOKEN = re.compile(r"(?:<(?:staging-root|home|tmp)>|" + _BEFORE
                         + r")/[^\s'\"<>,;:()\[\]{}=]+")
_SYSTEM_ROOTS = ("/usr/", "/opt/", "/etc/", "/lib", "/bin/", "/sbin/", "/proc/", "/sys/",
                 "/dev/", "/System/", "/Library/", "/Applications/", "/nix/")


def _read_id(text: str) -> Optional[bytes]:
    text = text.strip().replace("-", "").lower()
    if re.fullmatch(r"[0-9a-f]{32}", text) and set(text) != {"0"}:
        return bytes.fromhex(text)
    return None


def _state_file() -> Path:
    base = os.environ.get("XDG_STATE_HOME") or os.path.join(os.path.expanduser("~"), ".local",
                                                            "state")
    return Path(base) / "meep-gpu" / "machine-id"


def _machine_secret() -> Tuple[Optional[bytes], str]:
    """The machine's own 128-bit id and where it was read. Never written anywhere."""
    for name in ("/etc/machine-id", "/var/lib/dbus/machine-id"):
        try:
            found = _read_id(Path(name).read_text(encoding="ascii", errors="replace"))
        except OSError:
            continue
        if found is not None:
            return found, name
    if sys.platform == "darwin":
        try:
            out = subprocess.run(["/usr/sbin/ioreg", "-rd1", "-c", "IOPlatformExpertDevice"],
                                 capture_output=True, text=True, timeout=20).stdout
        except (OSError, subprocess.SubprocessError):
            out = ""
        match = re.search(r'"IOPlatformUUID"\s*=\s*"([0-9A-Fa-f-]{36})"', out)
        found = _read_id(match.group(1)) if match else None
        if found is not None:
            return found, "IOPlatformUUID"
    state = _state_file()
    try:
        found = _read_id(state.read_text(encoding="ascii", errors="replace"))
        if found is not None:
            return found, "state file"
    except OSError:
        pass
    try:
        state.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(state), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="ascii") as handle:
            handle.write(secrets.token_hex(16) + "\n")
    except FileExistsError:
        pass
    except OSError:
        return None, "unavailable"
    try:
        found = _read_id(state.read_text(encoding="ascii", errors="replace"))
    except OSError:
        found = None
    return (found, "state file") if found is not None else (None, "unavailable")


@functools.lru_cache(maxsize=None)
def _machine() -> Tuple[str, str]:
    secret, source = _machine_secret()
    if secret is None:
        return UNKNOWN_MACHINE, source
    digest = hmac.new(secret, MACHINE_APP_ID, hashlib.sha256).hexdigest()
    return "m-" + digest[:16], source


def machine_id() -> str:
    """This machine's anonymous id (module docstring, THE MACHINE ID)."""
    return _machine()[0]


def machine_id_source() -> str:
    """Where :func:`machine_id` read the machine's own id (a file name or a mechanism)."""
    return _machine()[1]


def card_id(uuid: Any) -> str:
    """A GPU UUID's salted hash (module docstring, THE CARD ID); ``""`` for no UUID."""
    text = str(uuid or "").strip().lower()
    if not text:
        return ""
    return "c-" + hmac.new(CARD_KEY, text.encode("utf-8"), hashlib.sha256).hexdigest()[:16]


def login() -> str:
    """This process's login, for the comparison :func:`path` makes; never written."""
    try:
        return getpass.getuser()
    except Exception:  # noqa: BLE001 - no account name is a fact, not a failure
        return ""


@functools.lru_cache(maxsize=None)
def _host_names() -> Tuple[str, ...]:
    names = set()
    for name in (socket.gethostname(), _node()):
        name = str(name or "").strip()
        if len(name) >= 3 and name.lower() not in ("localhost", "localhost.localdomain"):
            if name.lower() not in RECORD_WORDS:
                names.add(name)
            first = name.split(".")[0]
            if len(first) >= 3 and first.lower() not in RECORD_WORDS:
                names.add(first)
    return tuple(sorted(names, key=len, reverse=True))


def host_names() -> List[str]:
    """This machine's host names (and their first labels), for :func:`text`; never written."""
    return list(_host_names())


def reset() -> None:
    """Forget what was read about this machine (a test that changes the environment)."""
    for cached in (_machine, _host_names, _roots_for):
        cached.cache_clear()


def _node() -> str:
    try:
        return os.uname().nodename
    except AttributeError:
        return ""


def _spellings(value: str) -> List[str]:
    out = []
    for item in (value, os.path.realpath(value)):
        item = item.rstrip("/")
        if item and item not in out:
            out.append(item)
    return out


def staging_root(checkout: Any = None) -> Optional[str]:
    """The directory a checkout lies in: what ``<staging-root>`` stands for. ``None`` when
    that directory is the file system's root or a directory directly under it."""
    root = Path(checkout).resolve() if checkout else CHECKOUT
    parent = root.parent
    if len(parent.parts) < 3:
        return None
    return str(parent)


@functools.lru_cache(maxsize=None)
def _roots_for(checkout: Optional[str]) -> Tuple[Tuple[str, str], ...]:
    pairs: List[Tuple[str, str]] = []
    stage = staging_root(checkout)
    if stage:
        pairs += [(s, STAGING_ROOT) for s in _spellings(stage)]
    for home in {os.path.expanduser("~"), os.environ.get("HOME") or ""}:
        if home and len(Path(home).parts) >= 2:
            pairs += [(s, HOME) for s in _spellings(home)]
    for tmp in {tempfile.gettempdir(), os.environ.get("TMPDIR") or ""}:
        if tmp and Path(tmp).resolve() not in (Path("/tmp").resolve(), Path("/")):
            pairs += [(s, TMP) for s in _spellings(tmp)]
    seen, out = set(), []
    for root, label in sorted(pairs, key=lambda p: (-len(p[0]), p[1] != STAGING_ROOT)):
        if root not in seen:
            seen.add(root)
            out.append((root, label))
    return tuple(out)


def roots(checkout: Any = None) -> Tuple[Tuple[str, str], ...]:
    """``((absolute root, label), ...)``, longest first: the roots :func:`path` writes."""
    return _roots_for(str(Path(checkout).resolve()) if checkout else None)


def _account_components(token: str, name: str) -> str:
    if not name:
        return token
    return re.sub(r"(?<=/)" + re.escape(name) + r"(?=/|$)", ACCOUNT, token)


def _account_words(token: str, name: str) -> str:
    """``name`` written ``<account>`` where it leads a path component or a hyphen-joined word
    (``/<name>_runs``, ``pytest-of-<name>``, ``-Users-<name>-<project>``). A word that ends in
    the name (``metal_gate_runner.py`` for an account named ``runner``) is left alone."""
    return re.sub(r"(?:(?<=/)|(?<=-))" + re.escape(name) + r"(?![A-Za-z0-9.])", ACCOUNT, token)


@functools.lru_cache(maxsize=None)
def _scratch_shape(uid: str) -> "re.Pattern[str]":
    """A per-account directory under a temporary root, named for the account's numeric id
    (``/tmp/<tool>-<uid>``, ``/private/tmp/<tool>-<uid>``), with the directory under it when
    that one spells a whole path as one name (``-Users-<login>-<project>``)."""
    return re.compile(_BEFORE + r"(?:/private)?(?:/var)?/tmp/[A-Za-z][A-Za-z0-9._]*-"
                      + re.escape(uid) + r"(?:/-[^/\s'\"]+)?" + _END)


def _uid() -> str:
    try:
        return str(os.getuid())
    except AttributeError:  # no numeric account ids on this platform
        return ""


def path(value: Any, checkout: Any = None, generic: bool = True) -> Any:
    """``value`` with every absolute path written under its root's label. Anything that is
    not a string comes back unchanged; so does a string that names no local path.

    ``generic=False`` writes only this process's own roots and account (no home or
    temporary root recognised by its shape alone), so that :func:`resolve` gives the value
    back exactly: the spelling of a record this machine reads again, such as a lane's plan."""
    if not isinstance(value, str) or not value or ("/" not in value and "\\" not in value):
        return value
    out = value
    for root, label in roots(checkout):
        if root in out:
            out = re.sub(_BEFORE + re.escape(root) + _END, label, out)
    if generic:
        out = _HOME_SHAPE.sub(HOME, out)
        uid = _uid()
        if uid:
            out = _scratch_shape(uid).sub(TMP, out)
        out = _TMP_SHAPE.sub(TMP, out)
    name = login()
    if name and len(name) >= 2 and name in out:
        out = _ABSOLUTE.sub(lambda m: m.group() if m.group().startswith(_SYSTEM_ROOTS)
                            else _account_components(m.group(), name), out)
        # The name inside a component, where it leads it or a hyphen-joined word, also under
        # a root label; not for a name of two letters, nor for one of RECORD_WORDS (a login
        # spelled like the package or a table would rewrite every path naming that word).
        if len(name) >= 3 and name.lower() not in RECORD_WORDS and name in out:
            out = _PATH_TOKEN.sub(lambda m: m.group() if m.group().startswith(_SYSTEM_ROOTS)
                                  else _account_words(m.group(), name), out)
    return out


def text(value: Any, checkout: Any = None) -> Any:
    """Free text as a record may hold it: paths under their labels (:func:`path`), GPU
    UUIDs as card ids, and this machine's host names as its machine id."""
    if not isinstance(value, str) or not value:
        return value
    out = path(value, checkout)
    if "GPU-" in out or "MIG-" in out:
        out = _UUID.sub(lambda m: card_id(m.group()), out)
    for name in host_names():
        if name in out:
            out = re.sub(r"(?<![A-Za-z0-9_-])" + re.escape(name) + r"(?![A-Za-z0-9_-])",
                         machine_id(), out)
    return out


def process_label(command: Any, matched: Any = None) -> str:
    """Another process as a record names it: its program's base name, then ``matched`` (the
    harness script or word a reader's pattern found in its command line) when given. The
    arguments are never written: they can carry a remote account, an e-mail address, a
    repository URL or a path, which :func:`text` does not recognise."""
    parts = str(command or "").split()
    program = os.path.basename(parts[0]) if parts else ""
    found = str(matched).strip().lstrip("/") if matched else ""
    return text(f"{program} {found}" if found else program)


def _variable(name: Any, item: Any) -> Any:
    """The written value of an environment variable that names the account, the machine or a
    connection, or :data:`None` for any other variable."""
    if not isinstance(name, str) or not isinstance(item, str) or not item:
        return None
    if name in ACCOUNT_VARIABLES:
        return ACCOUNT
    if name in MACHINE_VARIABLES:
        return machine_id()
    if name in CONNECTION_VARIABLES:
        return "<connection>"
    return None


def value(obj: Any, checkout: Any = None) -> Any:
    """A JSON-shaped value with every string (dictionary keys included) through :func:`text`,
    and an environment variable naming the account, the machine or a connection, wherever it
    sits, written as :func:`environment` writes it. A path object is written as its text."""
    if isinstance(obj, str):
        return text(obj, checkout)
    if isinstance(obj, os.PathLike):
        return text(os.fspath(obj), checkout)
    if isinstance(obj, Mapping):
        out = {}
        for k, v in obj.items():
            named = _variable(k, v)
            out[text(k, checkout) if isinstance(k, str) else k] = (
                named if named is not None else value(v, checkout))
        return out
    if isinstance(obj, (list, tuple)):
        return [value(v, checkout) for v in obj]
    return obj


def argv(arguments: Optional[Iterable[Any]], checkout: Any = None) -> Optional[List[str]]:
    """A command line as a record writes it: every argument through :func:`text`."""
    if arguments is None:
        return None
    return [text(str(part), checkout) for part in arguments]


def environment(env: Optional[Mapping[str, Any]], checkout: Any = None) -> Optional[Dict[str, Any]]:
    """An environment block as a record writes it: the account variables as ``<account>``,
    the machine variables as the machine id, the connection variables as ``<connection>``,
    and every other value through :func:`text`. Variable names are kept."""
    if env is None:
        return None
    out: Dict[str, Any] = {}
    for name, item in env.items():
        named = _variable(name, item)
        out[name] = named if named is not None else text(item, checkout)
    return out


def resolve(recorded: Any, checkout: Any = None) -> Any:
    """A recorded path back on this machine: each label expanded to this machine's root.
    A value with no label comes back unchanged."""
    if not isinstance(recorded, str):
        return recorded
    out = recorded
    stage = staging_root(checkout)
    if stage:
        out = out.replace(STAGING_ROOT, stage)
    out = out.replace(HOME, os.path.expanduser("~"))
    out = out.replace(TMP, tempfile.gettempdir())
    name = login()
    if name:
        out = out.replace(ACCOUNT, name)
    return out


def resolve_value(obj: Any, checkout: Any = None) -> Any:
    """:func:`resolve` over a JSON-shaped value (every string, dictionary keys kept)."""
    if isinstance(obj, str):
        return resolve(obj, checkout)
    if isinstance(obj, Mapping):
        return {k: resolve_value(v, checkout) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [resolve_value(v, checkout) for v in obj]
    return obj


def path_value(obj: Any, checkout: Any = None) -> Any:
    """:func:`path` with ``generic=False`` over a JSON-shaped value: the spelling
    :func:`resolve_value` inverts exactly (dictionary keys kept)."""
    if isinstance(obj, str):
        return path(obj, checkout, generic=False)
    if isinstance(obj, Mapping):
        return {k: path_value(v, checkout) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [path_value(v, checkout) for v in obj]
    return obj


def neutralize_in_place(obj: Any, checkout: Any = None) -> Any:
    """:func:`value`'s spellings written INTO ``obj`` (a dict or a list), keeping every
    container the same object: a writer that goes on filling a list it handed to a record
    still fills the record's list. Returns ``obj``."""
    if isinstance(obj, dict):
        for key in list(obj):
            item = obj[key]
            named = _variable(key, item)
            if named is not None:
                item = named
            elif isinstance(item, str):
                item = text(item, checkout)
            elif isinstance(item, os.PathLike):
                item = text(os.fspath(item), checkout)
            elif isinstance(item, (dict, list)):
                neutralize_in_place(item, checkout)
            elif isinstance(item, tuple):
                item = tuple(value(list(item), checkout))
            new_key = text(key, checkout) if isinstance(key, str) else key
            if new_key != key:
                del obj[key]
            obj[new_key] = item
    elif isinstance(obj, list):
        for index, item in enumerate(obj):
            if isinstance(item, str):
                obj[index] = text(item, checkout)
            elif isinstance(item, os.PathLike):
                obj[index] = text(os.fspath(item), checkout)
            elif isinstance(item, (dict, list)):
                neutralize_in_place(item, checkout)
            elif isinstance(item, tuple):
                obj[index] = tuple(value(list(item), checkout))
    return obj


def residue(obj: Any, checkout: Any = None) -> List[str]:
    """What in ``obj`` still names this machine or account: every string (keys included)
    holding a local root, a home or temporary path by shape, a GPU UUID or a host name.
    The writers' tests call it on what they wrote; an empty list is the pass."""
    found: List[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, str):
            if text(node, checkout) != node:
                found.append(node)
        elif isinstance(node, Mapping):
            for key, item in node.items():
                walk(key)
                walk(item)
        elif isinstance(node, (list, tuple)):
            for item in node:
                walk(item)

    walk(obj)
    return found


#: The line a log carries in place of a child's line that could not be written neutrally.
UNWRITTEN_LINE = "[a line of the child's output was not written: it could not be written neutrally]\n"


def copy_lines(source: IO[bytes], handle: IO[str], checkout: Any = None) -> threading.Thread:
    """Copy a child's output into a log, every line through :func:`text`, each flushed as it
    lands (the progress rule: the log is readable while the child runs). ``source`` is the
    child's binary stdout pipe; the returned thread is started: join it once the child exits.

    The pipe always drains: a line that cannot be written neutrally is replaced by
    :data:`UNWRITTEN_LINE` (never written raw), and a log that cannot be written any more
    is read to the end and dropped, so the child never blocks on a full pipe."""

    def pump() -> None:
        writable = True
        for raw in iter(source.readline, b""):
            if not writable:
                continue
            try:
                line = text(raw.decode("utf-8", errors="replace"), checkout)
            except Exception:  # noqa: BLE001 - the drain outlives any one line
                line = UNWRITTEN_LINE
            try:
                handle.write(line)
                handle.flush()
            except Exception:  # noqa: BLE001 - a closed or full log: keep draining
                writable = False
        source.close()

    thread = threading.Thread(target=pump, name="neutral-log", daemon=True)
    thread.start()
    return thread


def rewrite_file(file_path: Any, checkout: Any = None) -> bool:
    """A log a child wrote directly (a timed process's, whose output must not pass through
    this process while it runs), rewritten once the child has exited, every line through
    :func:`text`. Bytes that are not UTF-8 are kept as they are. Returns whether it changed;
    a file that cannot be read or written is left alone (False)."""
    try:
        raw = Path(file_path).read_bytes()
    except OSError:
        return False
    before = raw.decode("utf-8", errors="surrogateescape")
    after = text(before, checkout)
    if after == before:
        return False
    try:
        Path(file_path).write_bytes(after.encode("utf-8", errors="surrogateescape"))
    except OSError:
        return False
    return True
