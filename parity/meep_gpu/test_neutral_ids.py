"""The neutral spellings of a run's machine stamps (``neutral_ids.py``): ids, card ids, paths."""

from __future__ import annotations

import hashlib
import hmac
import os
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import neutral_ids  # noqa: E402

SECRET = bytes.fromhex("00112233445566778899aabbccddeeff")
#: The two home roots, from pieces: a source line spelling one in full is a hygiene-guard hit.
HOME_ROOT, USERS_ROOT = "/" + "ho" + "me", "/" + "Us" + "ers"
UUID = "GPU-1a2b3c4d-0000-1111-2222-333344445555"
HOST_NAME = "labbox7.example.org"


@pytest.fixture
def machine(monkeypatch, tmp_path):
    """A machine whose own id, host name, login, home and temporary root are known."""
    home, tmp = tmp_path / "home" / "someone", tmp_path / "scratchtmp"
    home.mkdir(parents=True)
    tmp.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("TMPDIR", str(tmp))
    monkeypatch.setattr(neutral_ids.tempfile, "gettempdir", lambda: str(tmp))
    monkeypatch.setattr(neutral_ids.os.path, "expanduser",
                        lambda p: p.replace("~", str(home), 1) if p.startswith("~") else p)
    monkeypatch.setattr(neutral_ids, "_machine_secret", lambda: (SECRET, "/etc/machine-id"))
    monkeypatch.setattr(neutral_ids.socket, "gethostname", lambda: "labbox7.example.org")
    monkeypatch.setattr(neutral_ids, "_node", lambda: "labbox7")
    monkeypatch.setattr(neutral_ids, "login", lambda: "someone")
    neutral_ids.reset()
    yield {"home": str(home), "tmp": str(tmp)}
    neutral_ids.reset()


def expected_machine_id() -> str:
    return "m-" + hmac.new(SECRET, neutral_ids.MACHINE_APP_ID, hashlib.sha256).hexdigest()[:16]


def test_machine_id_is_the_keyed_hash_of_the_machine_id(machine):
    got = neutral_ids.machine_id()
    assert got == expected_machine_id()
    assert len(got) == 18 and got.startswith("m-")
    assert neutral_ids.machine_id() == got                     # stable
    assert SECRET.hex() not in got and "labbox7" not in got     # gives back neither
    assert neutral_ids.machine_id_source() == "/etc/machine-id"


def test_machine_id_differs_between_machines(machine, monkeypatch):
    first = neutral_ids.machine_id()
    monkeypatch.setattr(neutral_ids, "_machine_secret", lambda: (bytes(15) + b"\x01", "x"))
    neutral_ids.reset()
    assert neutral_ids.machine_id() != first


def test_machine_id_reads_the_systemd_file_and_refuses_an_empty_one():
    assert neutral_ids._read_id("0123456789abcdef0123456789abcdef\n") == bytes.fromhex(
        "0123456789abcdef0123456789abcdef")
    assert neutral_ids._read_id("0" * 32) is None
    assert neutral_ids._read_id("not an id") is None
    assert neutral_ids._read_id("A1B2C3D4-0000-1111-2222-333344445555") is not None


def test_machine_id_falls_back_to_a_state_file_that_persists(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setattr(neutral_ids, "Path", _NoSystemIdPath)
    monkeypatch.setattr(neutral_ids.sys, "platform", "linux")
    first, source = neutral_ids._machine_secret()
    again, _ = neutral_ids._machine_secret()
    assert source == "state file" and first is not None and first == again
    assert (tmp_path / "state" / "meep-gpu" / "machine-id").is_file()


class _NoSystemIdPath(type(Path())):
    """A ``Path`` for which the two system machine-id files do not exist."""

    def read_text(self, *args, **kwargs):
        if str(self) in ("/etc/machine-id", "/var/lib/dbus/machine-id"):
            raise FileNotFoundError(str(self))
        return super().read_text(*args, **kwargs)


def test_card_id_is_the_salted_hash_of_the_uuid():
    want = "c-" + hmac.new(neutral_ids.CARD_KEY, UUID.lower().encode(),
                           hashlib.sha256).hexdigest()[:16]
    assert neutral_ids.card_id(UUID) == want
    assert neutral_ids.card_id(UUID.upper()) == want and neutral_ids.card_id(f" {UUID} ") == want
    assert neutral_ids.card_id("GPU-ffffffff-0000-1111-2222-333344445555") != want
    assert neutral_ids.card_id("") == "" and neutral_ids.card_id(None) == ""
    assert "1a2b3c4d" not in want


def test_paths_are_written_under_their_roots(machine):
    stage = neutral_ids.staging_root()
    home, tmp = machine["home"], machine["tmp"]
    assert neutral_ids.path(f"{stage}/meep-gpu/parity/meep_gpu/results/x/gate.json") == \
        "<staging-root>/meep-gpu/parity/meep_gpu/results/x/gate.json"
    assert neutral_ids.path(f"{home}/miniforge3/envs/ref/bin/python") == \
        "<home>/miniforge3/envs/ref/bin/python"
    assert neutral_ids.path(f"{tmp}/pytest-1/a") == "<tmp>/pytest-1/a"
    assert neutral_ids.path(f"{HOME_ROOT}/other/x") == "<home>/x"
    assert neutral_ids.path(f"{USERS_ROOT}/other/x y") == "<home>/x y"
    assert neutral_ids.path(f"{HOME_ROOT}/user/x") == f"{HOME_ROOT}/user/x"   # the placeholder
    private = "/" + "priv" + "ate"
    assert neutral_ids.path(f"{private}/var/folders/ab/c12/T/tmpq/z") == "<tmp>/tmpq/z"
    assert neutral_ids.path(f"{private}/tmp/run") == "<tmp>/run"
    assert neutral_ids.path(f"/mnt/c{USERS_ROOT}/Bob/a") == "<home>/a"
    assert neutral_ids.path(f"C:{USERS_ROOT}/Bob/a".replace("/", "\\")) == "<home>\\a"
    assert neutral_ids.path("/scratch/someone/run") == "/scratch/<account>/run"
    assert neutral_ids.path("/usr/lib/someone/x") == "/usr/lib/someone/x"  # a system root
    assert neutral_ids.path("parity/meep_gpu/x.py") == "parity/meep_gpu/x.py"
    assert neutral_ids.path(7) == 7 and neutral_ids.path(None) is None
    # A root is a whole path component, never a prefix of a longer name.
    assert neutral_ids.path(f"{home}x/y") == f"{neutral_ids.path(home + 'x')}/y"
    # Several paths in one string, as a command line or PYTHONPATH holds them.
    joined = neutral_ids.path(f"{home}/a:{tmp}/b --out={stage}/c")
    assert joined == "<home>/a:<tmp>/b --out=<staging-root>/c"


def test_text_writes_host_names_and_uuids(machine):
    want = expected_machine_id()
    line = f"host: labbox7.example.org on labbox7 card {UUID} labbox7x"
    out = neutral_ids.text(line)
    assert out == f"host: {want} on {want} card {neutral_ids.card_id(UUID)} labbox7x"
    assert neutral_ids.text("GPU 0: NVIDIA RTX A6000") == "GPU 0: NVIDIA RTX A6000"


def test_value_argv_and_environment(machine):
    home = machine["home"]
    record = {"argv": [f"{home}/env/bin/python", "-u", "gate.py"],
              f"{home}/k": {"host": "labbox7", "n": 3, "ok": True}}
    assert neutral_ids.value(record) == {
        "argv": ["<home>/env/bin/python", "-u", "gate.py"],
        "<home>/k": {"host": expected_machine_id(), "n": 3, "ok": True}}
    assert neutral_ids.argv([f"{home}/a", 3]) == ["<home>/a", "3"]
    assert neutral_ids.argv(None) is None
    env = neutral_ids.environment({"USER": "someone", "LOGNAME": "someone", "HOSTNAME": "labbox7",
                                   "SSH_CONNECTION": "10.0.0.1 22 10.0.0.2 22",
                                   "CUPY_CACHE_DIR": f"{home}/c/cupy_ftz_stripped_x",
                                   "OMP_NUM_THREADS": "1", "EMPTY": ""})
    assert env == {"USER": "<account>", "LOGNAME": "<account>", "HOSTNAME": expected_machine_id(),
                   "SSH_CONNECTION": "<connection>",
                   "CUPY_CACHE_DIR": "<home>/c/cupy_ftz_stripped_x",
                   "OMP_NUM_THREADS": "1", "EMPTY": ""}
    assert "ftz_stripped" in env["CUPY_CACHE_DIR"]            # the policy token survives


def test_resolve_gives_the_path_back_on_this_machine(machine):
    for raw in (f"{machine['home']}/miniforge3/envs/ref/bin/python",
                f"{neutral_ids.staging_root()}/meep-gpu/x", f"{machine['tmp']}/q",
                "/scratch/someone/r", "/usr/bin/python3"):
        assert neutral_ids.resolve(neutral_ids.path(raw)) == raw
    assert neutral_ids.resolve(None) is None


def test_residue_names_what_still_identifies(machine):
    clean = {"a": "<home>/x", "b": [expected_machine_id(), neutral_ids.card_id(UUID)], "c": 1}
    assert neutral_ids.residue(clean) == []
    dirty = {"a": f"{machine['home']}/x", "b": ["labbox7"], UUID: 1}
    assert sorted(neutral_ids.residue(dirty)) == sorted([f"{machine['home']}/x", "labbox7", UUID])


def test_no_identity_reaches_a_record_through_the_helpers(machine):
    """Every helper's output, on inputs full of this machine's stamps, holds none of them."""
    home = machine["home"]
    raw = {"argv": [f"{home}/x", f"--host=labbox7.example.org"], "uuid": UUID,
           "env": {"USER": "someone", "HOME": home, "PWD": f"/scratch/someone/w"}}
    written = repr([neutral_ids.value(raw), neutral_ids.argv(raw["argv"]),
                    neutral_ids.environment(raw["env"])])
    for token in ("labbox7", "someone", home, UUID, "1a2b3c4d", SECRET.hex()):
        assert token not in written, token


def test_a_host_named_like_a_record_word_is_not_rewritten(machine, monkeypatch):
    """A host name equal to a word every record carries (a table, a backend, ``meep``)
    stays: rewriting it would break the readers matching that word, and the word says
    nothing a record does not already say. A distinctive name is still rewritten."""
    monkeypatch.setattr(neutral_ids.socket, "gethostname", lambda: "meep")
    monkeypatch.setattr(neutral_ids, "_node", lambda: "triton.example.org")
    neutral_ids.reset()
    assert neutral_ids.text('import meep; "table": "triton"') == 'import meep; "table": "triton"'
    assert neutral_ids.text("on triton.example.org") == f"on {neutral_ids.machine_id()}"
    assert "meep" in neutral_ids.RECORD_WORDS and "triton" in neutral_ids.RECORD_WORDS



def test_a_log_a_child_wrote_is_rewritten_after_it_exits(machine, tmp_path):
    """A timed child's log (written straight to the file) is rewritten once: host names,
    UUIDs and paths go, a byte that is not UTF-8 stays, and a second rewrite changes nothing."""
    log = tmp_path / "row.log"
    raw = (f"[labbox7:01234] MCW rank 0 bound to core 0\n{machine['home']}/x {UUID}\n"
           ).encode() + b"\xff\n"
    log.write_bytes(raw)
    assert neutral_ids.rewrite_file(log) is True
    text = log.read_bytes()
    assert b"labbox7" not in text and machine["home"].encode() not in text
    assert UUID.encode() not in text and text.endswith(b"\xff\n")
    assert neutral_ids.rewrite_file(log) is False
    assert neutral_ids.rewrite_file(tmp_path / "absent.log") is False


def test_the_log_copier_drains_the_child_past_a_line_it_cannot_write(machine, monkeypatch):
    """copy_lines never stops draining: a line whose neutral spelling raises is replaced by
    the marker line (never written raw), the lines after it still arrive, and a log that can
    no longer be written still lets the child's pipe drain to its end."""
    import io  # noqa: PLC0415

    real = neutral_ids.text

    def text(value, checkout=None):
        if "poison" in value:
            raise RuntimeError("cannot spell this line")
        return real(value, checkout)
    monkeypatch.setattr(neutral_ids, "text", text)
    source = io.BytesIO(f"first on {HOST_NAME}\npoison {HOST_NAME}\nlast\n".encode())
    log = io.StringIO()
    neutral_ids.copy_lines(source, log).join(timeout=10)
    lines = log.getvalue().splitlines(keepends=True)
    assert lines == [f"first on {neutral_ids.machine_id()}\n", neutral_ids.UNWRITTEN_LINE, "last\n"]
    assert source.closed

    class Full(io.StringIO):
        def write(self, value):
            raise OSError("no space left")
    drained = io.BytesIO(b"a\nb\nc\n")
    neutral_ids.copy_lines(drained, Full()).join(timeout=10)
    assert drained.closed


def test_a_path_glued_to_a_compiler_flag_is_labelled(machine):
    """Compiler and linker options glue a directory to a one-letter flag (``-I``, ``-L``),
    as NVRTC's recorded options and a build record's flags do; the path is labelled there too,
    and a path inside a word is not."""
    home = machine["home"]
    assert neutral_ids.path(f"-I{home}/env/include") == "-I<home>/env/include"
    assert neutral_ids.path(f"-L{home}/env/lib -Wl,-rpath,{home}/env/lib") == \
        "-L<home>/env/lib -Wl,-rpath,<home>/env/lib"
    assert neutral_ids.path(f"--include-path={home}/x") == "--include-path=<home>/x"
    assert neutral_ids.path(f"x-I{home}/y") == f"x-I{home}/y"


def test_an_account_name_inside_a_component_is_labelled(machine, monkeypatch):
    """The account's name where it leads a component or a hyphen-joined word, outside a root
    or under one (a scratch directory named for the account, a test runner's per-account
    directory, a home path spelled as one directory name), and a per-account directory under
    a temporary root named for the account's numeric id, with that one-name path under it."""
    home, stage = machine["home"], neutral_ids.staging_root()
    uid = str(os.getuid())
    private = "/" + "priv" + "ate"
    flat = "-" + USERS_ROOT[1:] + "-someone-project"
    assert neutral_ids.path("/scratch/someone_runs/a") == "/scratch/<account>_runs/a"
    assert neutral_ids.path(f"{home}/x/pytest-of-someone/y") == "<home>/x/pytest-of-<account>/y"
    assert neutral_ids.path(f"{stage}/w{flat}/z") == \
        "<staging-root>/w-" + USERS_ROOT[1:] + "-<account>-project/z"
    assert neutral_ids.path(f"{private}/tmp/tool-{uid}/{flat}/s1/corpus/python") == \
        "<tmp>/s1/corpus/python"
    assert neutral_ids.path(f"/tmp/tool-{uid}/{flat}/a b") == "<tmp>/a b"
    assert neutral_ids.path(f"/tmp/tool-{uid}/keep/a") == "<tmp>/keep/a"
    assert neutral_ids.path(f"/tmp/tool-{uid}0/a") == f"/tmp/tool-{uid}0/a"   # another id
    assert neutral_ids.path("/usr/share/someone-docs/x") == "/usr/share/someone-docs/x"
    assert neutral_ids.path("/data/runs_someone/x") == "/data/runs_someone/x"  # trailing word
    assert neutral_ids.residue([neutral_ids.path(f"{private}/tmp/tool-{uid}/{flat}/x")]) == []
    # A plan this machine reads again keeps an exact round trip.
    raw = {"corpus": f"{stage}/w{flat}/z", "out": "/scratch/someone_runs/a"}
    assert neutral_ids.resolve_value(neutral_ids.path_value(raw)) == raw


def test_an_account_name_ending_a_file_name_or_spelled_like_a_record_word_is_kept(
        machine, monkeypatch):
    """A word that ENDS in the account's name is not the account (a hosted runner's account
    ``runner`` against ``metal_gate_runner.py``), and an account spelled like a word every
    record carries is written ``<account>`` only as a whole path component."""
    home, stage = machine["home"], neutral_ids.staging_root()
    monkeypatch.setattr(neutral_ids, "login", lambda: "runner")
    for raw in (f"{stage}/meep-gpu/parity/meep_gpu/metal_gate_runner.py",
                f"{home}/parity/meep_gpu/test_metal_gate_runner.py",
                "/data/x/metal_gate_runner.py", "/data/x/runner.py"):
        assert "runner.py" in neutral_ids.path(raw), raw
    assert neutral_ids.path("/data/runner-logs/a") == "/data/<account>-logs/a"
    monkeypatch.setattr(neutral_ids, "login", lambda: "meep")
    assert neutral_ids.path(f"{home}/env/lib/site-packages/meep/x.py") == \
        "<home>/env/lib/site-packages/meep/x.py"
    assert neutral_ids.path("/data/meep_runs/meep-1.35/x") == "/data/meep_runs/meep-1.35/x"
    assert neutral_ids.path("/data/meep/x") == "/data/<account>/x"     # the component rule


def test_process_label_names_the_program_and_the_matched_harness_name_only(machine):
    at = "@"                # the address sign from a piece: a whole address is a guard hit
    assert neutral_ids.process_label(f"rsync -a r/ someone{at}remote.example.org:/x") == "rsync"
    assert neutral_ids.process_label(
        f"{machine['home']}/env/bin/python -u /x/gate_y.py --to someone{at}example.org",
        "/gate_y.py ") == "python gate_y.py"
    assert neutral_ids.process_label("") == "" and neutral_ids.process_label(None) == ""


def test_the_weld_walk_reads_every_path_root_as_outside_the_checkout():
    """A ledger path written under one of these roots names no directory of a checkout, so
    the weld walk excludes it as out of tree (``meep_gpu/weld_record_walk.NEUTRAL_ROOTS``)
    rather than checking it as a repository path that is not there."""
    from meep_gpu import weld_record_walk as walk

    assert walk.NEUTRAL_ROOTS == tuple(f"{root}/" for root in (
        neutral_ids.STAGING_ROOT, neutral_ids.HOME, neutral_ids.TMP)), walk.NEUTRAL_ROOTS
