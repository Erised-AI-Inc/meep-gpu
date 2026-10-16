"""The hygiene guard (``tools/ci/check_hygiene.py``): what it must catch, what it must
let through, and the three legs on this tree.

Every leak planted here is built at run time from neutral pieces and judged against
rules made for the test (``_rules``), so this file names no machine, account, path or
address of its own; the guard reading this file is part of the tracked leg.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
GUARD_PATH = ROOT / "tools" / "ci" / "check_hygiene.py"
_spec = importlib.util.spec_from_file_location("check_hygiene", GUARD_PATH)
guard = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = guard
_spec.loader.exec_module(guard)

#: Fake names, made of letters so the guard's word rule applies to them.
HOST = "qx" + "nodefour"
ACCOUNT = "qx" + "acct"
PHRASE = "Qx Lab Desk"
HOME = "/" + "ho" + "me"
MAC_HOME = "/" + "Us" + "ers"
TMP_ROOT = "/" + "priv" + "ate"


def _rules():
    rules = guard.load_rules()
    rules.add_name("host_name", [HOST])
    rules.add_name("account_name", [ACCOUNT])
    rules.add_name("host_name", guard.words_of(PHRASE))
    return rules


def _classes(text, rules=None):
    return sorted(klass for _s, _e, klass, _r in guard.scan_text(rules or _rules(), text, {}))


# --------------------------------------------------------------- the tracked rules file
def test_the_rules_list_every_name_as_a_digest_with_a_reason():
    document = json.loads((ROOT / "tools" / "ci" / "hygiene_rules.json").read_text())
    assert document["names"], "the rules list no name, so the name rule checks nothing"
    for entry in document["names"]:
        assert set(entry) <= {"class", "words", "sha256", "lead_sha256", "reason"}, entry
        assert entry["class"] in guard.NAME_CLASSES and entry["reason"], entry
        assert re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]), entry
    for entry in document["allow"]:
        assert entry.get("id") and entry.get("rule") and entry.get("reason"), entry


def test_the_guard_and_its_rules_carry_nothing_the_guard_catches():
    rules = guard.load_rules()
    for path in (GUARD_PATH, ROOT / "tools" / "ci" / "hygiene_rules.json", Path(__file__)):
        hits = guard._hits_by_line("self", path.name, rules, path.read_text(), {})
        assert not hits, [hit.line() for hit in hits]


def test_the_digest_command_prints_the_entry_the_rules_read(capsys):
    assert guard.main(["digest", "--class", "host_name", PHRASE]) == 0
    entry = json.loads(capsys.readouterr().out.strip())
    rules = guard.load_rules()
    rules.add_entry({**entry, "reason": "test"})
    assert _classes(f"ran on {PHRASE.lower()}-3 today", rules) == ["host_name"]


# ---------------------------------------------------------------- must catch, per class
@pytest.mark.parametrize("text", [
    f"measured on {HOST} CPU, two threads",
    f"status_{ACCOUNT} = 1",
    f"value{ACCOUNT.capitalize()}Seen = 1",
    f"cite numpy_check_{HOST}.txt",
    f"{HOST.upper()}: the card",
    f"scripts/{HOST}/check.py",
])
def test_a_listed_name_is_caught_inside_text_identifiers_and_paths(text):
    assert _classes(text) in (["host_name"], ["account_name"]), text


def test_a_listed_phrase_is_caught_across_punctuation():
    assert _classes("the " + PHRASE.replace(" ", "-") + "-2 laptop") == ["host_name"]
    assert _classes("a " + PHRASE.split()[0] + " desk") == []


def test_a_name_fused_into_a_longer_word_is_not_a_word_of_the_text():
    assert _classes(f"pre{HOST}post") == []


@pytest.mark.parametrize("text", [
    f"CUPY_CACHE_DIR={HOME}/someone/stage/cache",
    f"'{MAC_HOME}/someone/repo/x'",
    f"tmp is {TMP_ROOT}/var/folders/ab/T/x",
    f"{TMP_ROOT}/tmp/run1",
    f"see `{MAC_HOME}/someone/notes.md`",
    f"file://{MAC_HOME}/someone/page.html",
    f"{{{HOME}/someone}}|{HOME}/other",
])
def test_a_local_path_is_caught(text):
    assert set(_classes(text)) == {"home_path"}, text


@pytest.mark.parametrize("text", [
    f"{HOME}/user/meep_gpu_validation/x",       # the placeholder the fixtures ship
    f"repo/sub{HOME}/x",                        # a relative path
    f're.escape("{TMP_ROOT}/" + part + "/")',   # a pattern built from a root, no path
    f"<staging-root>/round/cache",
])
def test_a_placeholder_relative_path_or_bare_root_is_not_a_local_path(text):
    assert _classes(text) == [], text


def test_an_address_is_caught_and_the_allowed_forms_are_not():
    assert _classes("write to some.one" + "@" + "example-mail.org") == ["email"]
    assert _classes('assert "\\n' + "@" + 'triton.jit\\ndef f(" in source') == []
    assert _classes("user.email round" + "@" + "test.invalid") == []
    assert _classes("git clone git" + "@" + "github.com:org/repo.git") == []
    credited = re.search(r'email: "([^"]+)"', (ROOT / "CITATION.cff").read_text()).group(1)
    assert _classes(f"contact {credited}") == []


def test_a_name_inside_a_credited_address_is_part_of_the_credit():
    rules = guard.load_rules()
    address = "12345+" + ACCOUNT + "@users.noreply.github.com"
    rules.add_name("account_name", [ACCOUNT])
    assert _classes(f"by {address}", rules) == ["account_name", "email"]
    rules.allowed_addresses.add(hashlib.sha256(address.lower().encode()).hexdigest())
    assert _classes(f"by {address}", rules) == []
    assert _classes(f"by {address} and {ACCOUNT}", rules) == ["account_name"]


def test_a_hit_is_printed_masked():
    hits = guard._hits_by_line("tracked", "x.md", _rules(), f"on {HOST} CPU\n", {})
    assert [hit.where for hit in hits] == ["1"]
    assert HOST not in hits[0].line() and "[host_name]" in hits[0].line()


# ------------------------------------------------------------------- the ledger rule
def _ledger_tree(tmp_path, host_values, environment=None):
    for name in guard.LEDGERS:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n")
    entry = {"runs": {"8.6": {"host": value} for value in host_values[:1]},
             "extra": [{"host": value} for value in host_values[1:]]}
    if environment:
        entry["environment_at_start"] = environment
    (tmp_path / guard.LEDGERS[0]).write_text(json.dumps({"weld": entry}))
    return tmp_path


@pytest.mark.parametrize("value", ["node7: NVIDIA RTX A6000, cc 8.6", "node7",
                                   "node7 (machine-stamped: flush/gate.json:host)"])
def test_a_raw_machine_stamp_in_a_ledger_host_field_is_caught_without_its_name(tmp_path, value):
    hits, problems = guard.run_legs(_rules(), _ledger_tree(tmp_path, [value]), ["ledgers"])
    assert not problems and [hit.klass for hit in hits] == ["host_stamp"], hits
    assert hits[0].where == "weld.runs.8.6.host"


def test_the_neutral_host_labels_pass_the_ledger_rule(tmp_path):
    values = ["the GPU host: NVIDIA RTX A6000, cc 8.6; CuPy 13.5.1; Triton 3.1.0 (GPU 1)",
              "the GPU host (machine-stamped: flush/gate.json:host)", "the GPU host",
              "this machine: Apple MPS device applegpu_g13s, torch 2.10.0, metalfe-1"]
    hits, problems = guard.run_legs(_rules(), _ledger_tree(tmp_path, values), ["ledgers"])
    assert not problems and not hits, [hit.line() for hit in hits]


def test_a_home_path_in_a_recorded_environment_is_caught(tmp_path):
    tree = _ledger_tree(tmp_path, ["the GPU host"],
                        {"TRITON_CACHE_DIR": f"{HOME}/{ACCOUNT}/stage/cache",
                         "CUPY_CACHE_DIR": "<staging-root>/stage/cache"})
    hits, _ = guard.run_legs(_rules(), tree, ["ledgers"])
    assert sorted(hit.klass for hit in hits) == ["account_name", "home_path"]
    assert {hit.where for hit in hits} == {"weld.environment_at_start.TRITON_CACHE_DIR"}


# --------------------------------------------------------------- patches and binaries
def test_a_patch_is_judged_by_the_lines_it_adds(tmp_path):
    patch = tmp_path / "change.patch"
    patch.write_text("\n".join([
        "diff --git a/notes.md b/notes.md", "--- a/notes.md", "+++ b/notes.md",
        "@@ -10,3 +10,3 @@", " kept line", f"-removed on {HOST}", f"+added on {HOST}",
        " kept", "+clean addition", ""]))
    hits = guard.leg_patch(_rules(), patch)
    assert [(hit.path, hit.where, hit.klass) for hit in hits] == [("notes.md", "11", "host_name")]


def test_a_binary_file_is_read_by_its_printable_runs(tmp_path):
    (tmp_path / "blob.bin").write_bytes(b"\0\x01\x02" + f"{HOME}/someone/x.txt".encode()
                                        + b"\0" * 7 + b"\xff\xfe")
    (tmp_path / "noise.bin").write_bytes(bytes(range(256)) * 64)
    hits = guard.leg_files(_rules(), tmp_path, "tracked", ["blob.bin", "noise.bin"])
    assert [(hit.path, hit.klass) for hit in hits] == [("blob.bin", "home_path")]


# ------------------------------------------------------------------- this tree's legs
def test_the_tracked_files_and_the_board_subject_are_clean():
    hits, problems = guard.run_legs(guard.load_rules(), ROOT, ["tracked", "subject"])
    assert not problems, problems
    assert not hits, (f"{len(hits)} hit(s); a host name, account name, local path or "
                      f"personal address in a tracked file: "
                      + "; ".join(hit.line() for hit in hits[:12]))


def test_the_four_ledgers_carry_no_machine_stamp_account_or_local_path():
    hits, problems = guard.run_legs(guard.load_rules(), ROOT, ["ledgers"])
    assert not problems, problems
    assert not hits, (f"{len(hits)} hit(s) in the ledgers; the bind and record tools write "
                      f"neutral labels (parity/meep_gpu/neutral_labels.py): "
                      + "; ".join(hit.line() for hit in hits[:12]))


# ------------------------------------------------------------ the round's preflight
def _legs_module():
    rounds = ROOT / "parity" / "meep_gpu" / "rounds" / "1.0"
    spec = importlib.util.spec_from_file_location("legs_for_hygiene", rounds / "legs.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_the_round_preflight_refuses_a_checkout_the_guard_finds_a_leak_in(tmp_path):
    legs = _legs_module()
    assert legs.hygiene_problems(tmp_path, ("tracked",)) == [
        "tools/ci/check_hygiene.py is not in the checkout"]
    (tmp_path / "tools" / "ci").mkdir(parents=True)
    for name in ("check_hygiene.py", "hygiene_rules.json"):
        (tmp_path / "tools" / "ci" / name).write_bytes((ROOT / "tools" / "ci" / name).read_bytes())
    (tmp_path / "notes.md").write_text("clean\n")
    assert legs.hygiene_problems(tmp_path, ("tracked",)) == []
    (tmp_path / "notes.md").write_text(f"cache in {HOME}/someone/stage\n")
    problems = legs.hygiene_problems(tmp_path, ("tracked",))
    assert len(problems) == 1 and "notes.md:1 home_path" in problems[0], problems


def test_the_bind_plan_ends_by_writing_the_neutral_labels(tmp_path):
    lines = _legs_module().bind_plan(tmp_path, "2000-01-01_t", 2, write=True)
    assert lines[-1] == "python neutralize_ledger_labels.py --write"
