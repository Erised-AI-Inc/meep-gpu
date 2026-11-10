"""The hygiene guard (``tools/ci/check_hygiene.py``): what it must catch, what it must
let through, and the three legs on this tree.

Every leak planted here is built at run time from neutral pieces and judged against
rules made for the test (``_rules``), so this file names no machine, account, path or
address of its own, and spells no development jargon; the guard reading this file is
part of the tracked leg.
"""

from __future__ import annotations

import gzip
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
sys.path.insert(0, str(ROOT / "parity" / "meep_gpu"))

import ledger_scope  # noqa: E402

#: Fake names, made of letters so the guard's word rule applies to them.
HOST = "qx" + "nodefour"
ACCOUNT = "qx" + "acct"
PHRASE = "Qx Lab Desk"
HOME = "/" + "ho" + "me"
MAC_HOME = "/" + "Us" + "ers"
TMP_ROOT = "/" + "priv" + "ate"
#: The development repository's private evidence, from the same neutral pieces the guard
#: builds its patterns from: the evidence tree, a round's staging directory, the lane and
#: record directories under ``dev/``.
TREE = "bit" + "_parity_0_10"
STAGE = "bit" + "parity_phase0"
WP, CERT, DEV = "w" + "p", "ce" + "rt", "de" + "v"
#: Development jargon, from the same kind of pieces: a private design document's file
#: name, a decision number, a ruling number, the 1.0 release's working name, a lane, a
#: work package, the owner, the prototype phase.
KERN, DSGN = "KER" + "NELS.md", "DES" + "IGN.md"
DNUM, RUL, TEN = "D" + "13", "R" + "33", "0" + ".10"
LANE, PKG, OWN, PHASE = "la" + "ne", "W" + "P", "own" + "er", "pha" + "se"


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


# ------------------------------------------------------------ private evidence paths
@pytest.mark.parametrize("text", [
    f"probe in parity/meep_gpu/results/{TREE}/{DEV}/{WP}/RG/scripts/cpow_check.py",
    f"decoded ONCE for both halves ({TREE} {DSGN} D{10}).",
    f"the inversion (``{TREE}/{DSGN}`` D{12})",
    f"(``results/{TREE}/design/{KERN}`` section 8)",
    f"MEASURED ({TREE} {LANE} FXHD): the conductive family's control",
    f'"CUPY_CACHE_DIR": "<staging-root>/{STAGE}/NR27/route_1/caches/cupy"',
    f"keep and flush ({WP}/FXAn/REPORT.txt).",
    f"as the control ({WP}/A13, the same tree)",
    f"the control ({CERT}/C1, tree afb53f5d)",
    f"R{27} corpus gap ({CERT}/CORPUS5/ALERT_EXPLAINED.txt): c1",
    f"see {DEV}/{CERT}/HYG30.txt",
])
def test_a_private_evidence_path_or_citation_is_caught(text):
    # A citation inside the private tree is often jargon as well; this rule is the path.
    classes = set(_classes(text))
    assert "private_evidence" in classes and classes <= {"private_evidence", "dev_jargon"}, text


@pytest.mark.parametrize("text", [
    "apps/api/parity/meep_gpu/results/cuda_round_b_compose_2026-09-25/texts/x.cu",
    "parity/meep_gpu/results/<campaign>/gate.json",
    "meep_gpu/cuda_kernels/certification.json and cert_chunks/2",
    f"rounds/1.0/{CERT}/INDEX.json and parity/{WP}/X",
    f"{WP}/lowercase and {CERT}/lowercase",
    "export CUPY_CACHE_DIR=~/.cupy; mkdir -p ~/triton_libcuda_stub",
    "tail -f ~/meep_gpu_validation/timing/ladder_<stamp>/progress.log",
    "THE PER-CHUNK FORMS, 1.0 (docs/development/kernel-tables.md)",
    "the record C1 of the 1.0 round, cut from hoist_reference/x.cu",
])
def test_the_record_format_neutral_names_and_instructions_are_not_private_evidence(text):
    assert _classes(text) == [], text


# ------------------------------------------------------------------ development jargon
def _jargon(text):
    return sorted(rule for _s, _e, klass, rule in guard.scan_dev_jargon(text)
                  if klass == guard.JARGON)


@pytest.mark.parametrize("text, rule", [
    (f"the form contract ({KERN} section 1)", "private_document"),
    (f"the inversion ({DSGN} {DNUM})", "private_document"),
    ("the row (" + "ARRAY" + "_PATH.md)", "private_document"),
    ("see " + "TESTS" + "_AND_CERT.md", "private_document"),
    (f"the form-coverage rule ({DNUM})", "decision_number"),
    (f"{DNUM}'s rule", "decision_number"),
    ("D" + "4: one record", "decision_number"),
    (f"per {RUL}", "ruling_number"),
    (f"({OWN} ruling {RUL})", "ruling_number"),
    (f"refused under {RUL}", "ruling_number"),
    (f"the {RUL} round", "ruling_number"),
    (f"timed ({RUL})", "ruling_number"),
    (f"a pre-{RUL} ladder", "ruling_number"),
    (f"{RUL}'s gate", "ruling_number"),
    (f"is an {OWN}'s design ruling", "ruling_number"),
    (f"which the {OWN} ruled out", "ruling_number"),
    (f"the grouping until {TEN}", "release_label"),
    (f"since {TEN} the kernel", "release_label"),
    (f"a pre-{TEN} tree", "release_label"),
    (f"THE {TEN} FORMS moved the emitter", "release_label"),
    (f"the {TEN} development", "release_label"),
    (f"words differ ({TEN} AP6b)", "release_label"),
    (f"Until {TEN} the array path accumulated twice", "release_label"),
    (f"Since {TEN} every PML form is chosen per chunk", "release_label"),
    (f"the arithmetic the package carried until {TEN}.", "release_label"),
    (f"the aligned forms (bit_parity {TEN}): one product per part", "release_label"),
    (f"on the {TEN} engine no withdraw does work", "release_label"),
    (f"SINCE MEEP'S PER-CHUNK FORMS ({TEN}", "release_label"),
    (f"{PKG} AP6c", "process_label"),
    ("the readback work " + "package RG", "process_label"),
    (f"the {LANE} KC8 record", "process_label"),
    (f"({LANE} FXHD)", "process_label"),
    (f"{LANE}s KC1, KT1, KM1", "process_label"),
    ("batch_" + "r13 (A11)", "process_label"),
    (f"measured in {PHASE} 0", "process_label"),
    (f"the {PHASE}-0 leg", "process_label"),
    ("P" + "0-h measured it", "process_label"),
    ("the App" + "endix A inversion", "private_section"),
    ("the needle constants (section " + "7.2)", "private_section"),
    ("the \u00a7" + "7.1 battery", "private_section"),
])
def test_development_jargon_is_caught_by_its_rule(text, rule):
    assert rule in _jargon(text), (text, _jargon(text))
    assert set(_classes(text)) == set(), text  # its own class, never scan_text's


@pytest.mark.parametrize("text", [
    "MEEP's D1, D2 and D3 lattices; a Dcyl grid; mp.D1",
    "the D-seams and B-seams; the D field, Dz and Dsqr",
    "the radial R, rinv and R = 1.5; the matrix R",
    "R7_REFUSAL, r7_reasons, RULES_R33 and REQUIRED_FLOORS_R33 are code names",
    "the row masks (R01) and (R02), and R11, R12, R21, R22 as constexpr flags",
    "the records MR27, FROZEN_R27 and round_chunks_r27",
    "CERT-3, the device-gate stage of a certification round",
    f"the record results/prefer_gpu_semantics_2026-09-27/{DSGN}",
    "the owner volume of a residency, the chunk's owner, ownership masks",
    "a destination lane READS its own D; the two lanes END in reverse",
    "the wrapped-lane SELECT; each lane WAIT; the lane ONLY",
    "pytest.approx(0.10); near_pole (0.10, 0.16); MIRROR_ROW_Y = 0.10",
    "the two correlate at 0.10 and agree in sign; a 1.0f + 0.10f * x term; (0.10)",
    "IEEE 754-2019 \u00a76.2.3 and \u00a77.2 leave the payload unspecified",
    "the 1.0 certification round, round-to-nearest, round(x)",
    "entry 0 to entry 2 of the table; the 2026-09-02 release decision",
    "the per-chunk form battery and the inverted grouping mutations",
    "a phase 0.5 offset; the Bloch phase; phase0 is a local name",
])
def test_technical_uses_are_not_development_jargon(text):
    assert _jargon(text) == [], (text, _jargon(text))


def test_a_form_anchored_at_a_line_end_is_found_inside_a_whole_file(tmp_path):
    (tmp_path / "notes.md").write_text(f"clean line\nSINCE THE PER-CHUNK FORMS ({TEN}\nKC2): more\n")
    hits = guard.leg_files(_rules(), tmp_path, "tracked", ["notes.md"])
    assert [(h.where, h.rule) for h in hits] == [("2", "release_label")]


def test_a_jargon_allowance_names_the_line_and_ends_when_it_changes(tmp_path):
    lines = [f"arm one  # label ({RUL[0]}5)", f"arm two  # label ({RUL[0]}6)"]
    (tmp_path / "gate.py").write_text("\n".join(lines) + "\n")
    rules = _rules()
    rules.allowed_jargon = {"gate.py": {guard.line_digest(lines[0])}}
    hits = guard.leg_files(rules, tmp_path, "tracked", ["gate.py"])
    assert [(h.where, h.klass, h.rule) for h in hits] == [("2", "dev_jargon", "ruling_number")]
    assert rules.allowed_jargon_found == {("gate.py", guard.line_digest(lines[0]))}
    (tmp_path / "gate.py").write_text(lines[0] + " edited\n" + lines[1] + "\n")
    assert [h.where for h in guard.leg_files(rules, tmp_path, "tracked", ["gate.py"])] == [
        "1", "2"]


def test_a_jargon_allowance_names_a_file_and_line_digests(tmp_path):
    document = json.loads((ROOT / "tools" / "ci" / "hygiene_rules.json").read_text())
    document["allow"].append({"id": "t", "rule": "dev_jargon", "path": "x.py",
                              "lines": ["not-a-digest"], "reason": "test"})
    path = tmp_path / "rules.json"
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="sha256 of each line"):
        guard.load_rules(path)


def test_a_tool_written_file_is_declared_whole_and_goes_stale_when_it_changes(tmp_path):
    text = f'{{"note": "{DNUM}: the record", "why": "per {RUL}"}}\n'
    (tmp_path / "rows.json").write_text(text)
    rules = _rules()
    rules.pending = [{"id": "test.rows", "leg": "tracked", "path": "rows.json",
                      "rule": "dev_jargon", "hits": 2,
                      "file_sha256": hashlib.sha256(text.encode()).hexdigest(),
                      "until": "the rows' tool", "reason": "a tool-written file"}]
    hits, problems = guard.run_legs(rules, tmp_path, ["tracked"])
    assert not problems and hits == [], [h.line() for h in hits]
    assert guard.pending_status(rules) == [("test.rows", 2, 2, "pending")]
    (tmp_path / "rows.json").write_text(text.replace("the record", "the records"))
    hits, _ = guard.run_legs(rules, tmp_path, ["tracked"])
    assert sorted(h.rule for h in hits) == ["decision_number", "ruling_number"]
    assert guard.pending_status(rules) == [("test.rows", 0, 2, "STALE")]
    # A host name in a declared file is a hit whatever the declaration says.
    (tmp_path / "rows.json").write_text(text + f"ran on {HOST}\n")
    rules.pending[0]["file_sha256"] = hashlib.sha256(
        (tmp_path / "rows.json").read_bytes()).hexdigest()
    hits, _ = guard.run_legs(rules, tmp_path, ["tracked"])
    assert [h.klass for h in hits] == ["host_name"]


def test_jargon_reads_only_the_files_the_export_keeps(tmp_path):
    (tmp_path / "tools" / "release").mkdir(parents=True)
    (tmp_path / "tools" / "release" / "export_manifest.json").write_text(json.dumps(
        {"excluded": [{"path": "dev/runbook.html"}]}))
    (tmp_path / "dev").mkdir()
    for name in ("dev/runbook.html", "notes.md"):
        (tmp_path / name).write_text(f"per {RUL}, ran on {HOST}\n")
    hits = guard.leg_files(_rules(), tmp_path, "tracked", ["dev/runbook.html", "notes.md"])
    assert sorted((h.path, h.klass) for h in hits) == [
        ("dev/runbook.html", "host_name"), ("notes.md", "dev_jargon"), ("notes.md", "host_name")]


def test_a_patch_is_refused_jargon_on_every_added_line_but_a_ledger_s(tmp_path):
    patch = tmp_path / "change.patch"
    ledger = guard.LEDGERS[0]
    patch.write_text("\n".join([
        f"diff --git a/{ledger} b/{ledger}", f"--- a/{ledger}", f"+++ b/{ledger}",
        "@@ -1,1 +1,2 @@", " {", f'+  "why": "per {RUL}",',
        "diff --git a/notes.md b/notes.md", "--- a/notes.md", "+++ b/notes.md",
        "@@ -1,1 +1,3 @@", " kept", f"+the grouping until {TEN}", "+the grouping until 1.0", ""]))
    hits = guard.leg_patch(_rules(), patch, tmp_path)
    assert [(h.path, h.where, h.rule) for h in hits] == [("notes.md", "2", "release_label")]


def test_this_trees_jargon_allowances_are_each_still_needed():
    rules = guard.load_rules()
    hits, problems = guard.run_legs(rules, ROOT, ["tracked"])
    assert not problems
    wanted = {(path, line) for path, lines in rules.allowed_jargon.items() for line in lines}
    assert wanted and wanted == rules.allowed_jargon_found, sorted(
        wanted - rules.allowed_jargon_found)


def test_a_home_relative_path_is_refused_in_a_ledger_and_nowhere_else(tmp_path):
    value = "~/meep_gpu_validation/run_2026-08-13/launchers/a.sh"
    hits, problems = guard.run_legs(_rules(), _ledger_tree(tmp_path, ["the GPU host"],
                                                           {"LAUNCHER": value}), ["ledgers"])
    assert not problems and [(h.klass, h.rule) for h in hits] == [
        ("private_evidence", "home_relative")], hits
    assert "meep_gpu_validation" not in hits[0].line()
    (tmp_path / "notes.md").write_text(f"run {value}\n")
    assert guard.leg_files(_rules(), tmp_path, "tracked", ["notes.md"]) == []


def test_a_patch_is_refused_a_home_relative_path_only_on_a_ledger_line(tmp_path):
    patch = tmp_path / "change.patch"
    ledger = guard.LEDGERS[0]
    patch.write_text("\n".join([
        f"diff --git a/{ledger} b/{ledger}", f"--- a/{ledger}", f"+++ b/{ledger}",
        "@@ -1,1 +1,2 @@", " {", '+  "run_root": "~/stage/run_2026-08-13",',
        "diff --git a/notes.md b/notes.md", "--- a/notes.md", "+++ b/notes.md",
        "@@ -1,1 +1,3 @@", " kept", "+clone into ~/meep-gpu", f"+cite {TREE} lane X", ""]))
    hits = guard.leg_patch(_rules(), patch)
    assert [(h.path, h.where, h.rule) for h in hits] == [
        (ledger, "2", "home_relative"), ("notes.md", "3", "private_tree")], hits


def _declaration(name, where, value, hits=1, rule="home_relative"):
    return {"id": "test.pending", "leg": "ledgers", "path": name, "rule": rule, "hits": hits,
            "values": [{"where": where, "sha256": hashlib.sha256(value.encode()).hexdigest()}],
            "until": "the test's clearing tool", "reason": "a declared ledger hit"}


def test_a_declared_pending_ledger_hit_is_reported_not_counted_and_goes_stale(tmp_path):
    old, other = "~/stage/run_2026-08-13/a.sh", "~/stage/run_2026-08-14/b.sh"
    tree = _ledger_tree(tmp_path, ["the GPU host"], {"OLD": old, "OTHER": other})
    rules = _rules()
    # This test's own declaration only, in place of the tree's.
    rules.pending = [_declaration(guard.LEDGERS[0], "weld.environment_at_start.OLD", old)]
    hits, problems = guard.run_legs(rules, tree, ["ledgers"])
    # Only the declared value is pending; the undeclared one at a sibling key is a hit.
    assert not problems and [h.where for h in hits] == ["weld.environment_at_start.OTHER"]
    assert guard.pending_status(rules) == [("test.pending", 1, 1, "pending")]
    # The clearing tool rewrote the value: nothing is found, the declaration is stale, and
    # the same text written back at another key is a hit.
    _ledger_tree(tmp_path, ["the GPU host"], {"OLD": "<staging-root>/run_2026-08-13/a.sh",
                                              "MOVED": old})
    hits, _ = guard.run_legs(rules, tmp_path, ["ledgers"])
    assert [h.where for h in hits] == ["weld.environment_at_start.MOVED"]
    assert guard.pending_status(rules) == [("test.pending", 0, 1, "STALE")]


@pytest.mark.parametrize("change, why", [
    ({"until": ""}, "lacks"), ({"path": "notes.md"}, "ledger only"),
    ({"rule": "email"}, "unknown rule"), ({"leg": "subject"}, "ledgers or the tracked leg"),
    ({"leg": "tracked"}, "lacks"),
    ({"leg": "tracked", "file_sha256": "0" * 64}, "not a ledger"),
    ({"leg": "tracked", "path": "notes.md", "file_sha256": "0" * 64}, "dev_jargon hits only"),
    ({"leg": "tracked", "path": "notes.md", "rule": "dev_jargon", "file_sha256": "f"},
     "file's sha256")])
def test_a_pending_declaration_names_its_ledger_rule_and_clearing_tool(tmp_path, change, why):
    document = json.loads((ROOT / "tools" / "ci" / "hygiene_rules.json").read_text())
    entry = {**_declaration(guard.LEDGERS[0], "k", "~/x"), **change}
    document["pending"] = [entry]
    path = tmp_path / "rules.json"
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match=why):
        guard.load_rules(path)


def test_this_trees_pending_declarations_are_each_found_whole():
    """The declared hits are pending until the tool that writes their file clears them
    (the bind for a ledger; a board, record or fixture re-cut for a tracked file); then the
    declaration is STALE and this fails until it is deleted, after which the same text is
    refused (the re-cut instruction's last step)."""
    rules = guard.load_rules()
    hits, problems = guard.run_legs(rules, ROOT, ["tracked", "ledgers"])
    assert not problems
    status = guard.pending_status(rules)
    assert len(status) == len(rules.pending), status
    assert all(state == "pending" for *_n, state in status), status
    for entry in rules.pending:
        if entry["leg"] == "ledgers":
            assert entry["path"] in guard.LEDGERS and entry["rule"] in (
                "private_tree", "lane_path", "home_relative"), entry["id"]
        else:
            assert entry["leg"] == "tracked" and entry["rule"] == guard.JARGON, entry["id"]
            assert (ROOT / entry["path"]).is_file(), entry["id"]


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


@pytest.mark.parametrize("text", [
    f"-I{HOME}/someone/env/include",
    f"-L{HOME}/someone/env/lib -Wl,-rpath,<home>/env/lib",
    f"-L{TMP_ROOT}/tmp/build/lib",
])
def test_a_local_path_glued_to_a_compiler_flag_is_caught(text):
    """A build record spells its include and library directories as ``-I`` and ``-L``
    glued to the path; the path is still a local one."""
    assert set(_classes(text)) == {"home_path"}, text


def _row_store(tmp_path, environment, bench_line=None, host="WSL2"):
    """A timing row store of one record (``timing_rows.write_record``'s layout)."""
    folder = tmp_path / "store" / "ladder" / "ladder__8.6__pml_3d__512000__priority__x"
    folder.mkdir(parents=True)
    record = {"schema": guard.ROW_SCHEMA, "kind": "ladder", "gates": {"windows": {
        "host": host, "vm_affinity": "0x999988"}}, "environment": environment}
    (folder / "s1__t.json").write_text(json.dumps(record))
    if bench_line is not None:
        (folder / "s1__t.rows.jsonl").write_text(bench_line + "\n")
    (tmp_path / "store" / "other.json").write_text(json.dumps({"x": f"{HOME}/{ACCOUNT}"}))
    return tmp_path / "store"


def test_the_records_leg_reads_every_row_record_of_a_store(tmp_path):
    """A row record leaves the machine: a home path in it, a key holding one included,
    is a hit with its key path printed masked; the store's other files and the bench line
    kept verbatim beside a record are not row records; the neutral spellings pass."""
    image = f"{HOME}/{ACCOUNT}/env/lib/libmeep.so"
    leaky = _row_store(tmp_path / "a", {"reference": {"record": {
        "python": f"{HOME}/{ACCOUNT}/env/bin/python", "sha256": {image: "ab" * 32},
        "config_log": "~/env/share/config.log"}}},
        bench_line=json.dumps({"cache": f"{HOME}/<account>/x"}))
    hits, problems = guard.run_legs(_rules(), ROOT, [guard.RECORDS_LEG], [leaky])
    assert not problems
    assert sorted({(hit.leg, hit.klass) for hit in hits}) == [
        ("records", "account_name"), ("records", "home_path"), ("records", "private_evidence")]
    assert {hit.path for hit in hits} == {
        "store/ladder/ladder__8.6__pml_3d__512000__priority__x/s1__t.json"}
    assert {hit.where for hit in hits} == {
        "environment.reference.record.python", "environment.reference.record.config_log",
        "environment.reference.record.sha256.[home_path]/env/lib/libmeep.so"}
    assert all(ACCOUNT not in hit.line() and image not in hit.line() for hit in hits)
    clean = _row_store(tmp_path / "b", {"reference": {"record": {
        "python": "<home>/env/bin/python", "libm": "/usr/lib/libm.so.6",
        "sha256": {"<home>/env/lib/libmeep.so": "ab" * 32}}}})
    assert guard.run_legs(_rules(), ROOT, [guard.RECORDS_LEG], [clean]) == ([], [])


def test_the_records_leg_runs_only_when_asked_and_never_on_nothing(tmp_path):
    """The records leg is no default leg; asked for without a store, or on a store with no
    row record, it cannot be read and does not pass."""
    assert guard.RECORDS_LEG not in guard.LEGS
    _hits, problems = guard.run_legs(_rules(), ROOT, [guard.RECORDS_LEG])
    assert problems == ["records: no store given (--records STORE)"]
    (tmp_path / "empty").mkdir()
    _hits, problems = guard.run_legs(_rules(), ROOT, [guard.RECORDS_LEG], [tmp_path / "empty"])
    assert problems and "no timing row record below it" in problems[0]
    assert guard.main(["--legs", "records", "--records", str(tmp_path / "empty")]) == 2


@pytest.mark.parametrize("text", [
    "C:\\" + "Us" + "ers" + "\\someone\\AppData\\Local\\Temp\\x.log",  # a Windows profile
    "/mnt/c/" + "Us" + "ers" + "/someone/AppData/x",                  # the same, under WSL2
    "/tmp/pytest-of-someone/x",                                       # a Linux temporary root
    "/scratch/proj42/envs/ref/bin/python",                            # a site's filesystem
    "./configure --prefix=/project/m4242/envs/ref --enable-single",
])
def test_the_records_leg_finds_what_the_records_own_check_finds(tmp_path, text):
    """Every string ``timing_rows.local_paths`` finds in a row record (the check its writers
    are held to) is a hit of its own class where no rule found it, as a value and as a key,
    printed masked; the guard then never reads CLEAN what the writers' check refuses."""
    image = "/scratch/proj42/envs/ref/lib/libmeep.so"
    store = _row_store(tmp_path, {"reference": {"record": {"note": text,
                                                             "sha256": {image: "ab" * 32}}}})
    hits, problems = guard.run_legs(_rules(), ROOT, [guard.RECORDS_LEG], [store])
    assert not problems
    assert {(hit.where, hit.klass, hit.rule) for hit in hits} == {
        ("environment.reference.record.note", guard.LOCAL_PATH, "local_paths"),
        ("environment.reference.record.sha256.[local_path].so", guard.LOCAL_PATH,
         "local_paths")}
    assert len(hits) == 2, "one hit a string"
    assert all("someone" not in hit.line() and "/scratch" not in hit.line()
               and "/project" not in hit.line() for hit in hits)
    if text.startswith("./configure"):
        note, = [hit for hit in hits if hit.where.endswith(".note")]
        assert note.context == "./configure --prefix=[local_path] --enable-single", \
            "the path masked, the rest of the string kept"
    assert guard.main(["--legs", "records", "--records", str(store)]) == 1


def test_a_path_continued_after_a_label_is_no_hit_beside_one_that_is(tmp_path):
    """In a string the records' check flags, the hit is the path under no root; the rest of a
    labelled path continued after a doubled slash (``<home>//miniforge3``) is not masked or
    counted as one."""
    text = "./configure --prefix=<home>//miniforge3/envs/ref CC=/scratch/proj42/bin/cc"
    store = _row_store(tmp_path, {"reference": {"record": {"configure_line": text}}})
    hits, problems = guard.run_legs(_rules(), ROOT, [guard.RECORDS_LEG], [store])
    assert not problems and len(hits) == 1
    assert hits[0].context == "./configure --prefix=<home>//miniforge3/envs/ref CC=[local_path]"


def test_the_records_leg_admits_the_neutral_host_stamp_and_no_other(tmp_path):
    """A record's ``host`` is its Windows side's neutral ``WSL2``; any other bare word there
    is a raw machine stamp."""
    clean = _row_store(tmp_path / "a", {"reference": {"record": {
        "configure_line": "./configure --prefix=<home>//miniforge3/envs/ref"}}})
    assert guard.run_legs(_rules(), ROOT, [guard.RECORDS_LEG], [clean]) == ([], []), \
        "a labelled path continued after a doubled slash names nothing"
    stamped = _row_store(tmp_path / "b", {"reference": None}, host="labdesk7")
    hits, _problems = guard.run_legs(_rules(), ROOT, [guard.RECORDS_LEG], [stamped])
    assert [(hit.where, hit.klass) for hit in hits] == [("gates.windows.host", "host_stamp")]
    assert "labdesk7" not in hits[0].line()


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


# ---------------------------------------------------------------- the archived ledger states ----

def _state(root, table, document):
    source = root / "ledger.json"
    source.write_text(json.dumps(document))
    return ledger_scope.archive_state(root, table, source)


def test_the_archived_states_are_a_default_leg_and_this_tree_s_are_clean():
    assert "ledger_states" in guard.LEGS
    assert guard.leg_ledger_states(_rules(), ROOT) == []
    assert list((ROOT / guard.STATES_DIR).glob("*/*.json.gz"))


def test_a_planted_local_path_in_an_archived_state_fails_under_its_own_name(tmp_path):
    path = _state(tmp_path, "cuda", {"w": {"note": "/" + "Users/someone/project"}})
    hits, problems = guard.run_legs(_rules(), tmp_path, ["ledger_states"])
    assert problems == [] and hits, (hits, problems)
    assert {hit.klass for hit in hits} == {"home_path"}
    assert hits[0].path == path.relative_to(tmp_path).as_posix() and hits[0].leg == "ledger_states"
    assert guard.main(["--root", str(tmp_path), "--legs", "ledger_states"]) == 1


def test_a_raw_machine_stamp_in_an_archived_state_fails(tmp_path):
    _state(tmp_path, "metal", {"w": {"host": "somebox"}})
    hits, _problems = guard.run_legs(_rules(), tmp_path, ["ledger_states"])
    assert [hit.klass for hit in hits] == ["host_stamp"]


def test_a_tree_with_no_archive_has_nothing_to_read(tmp_path):
    assert guard.run_legs(_rules(), tmp_path, ["ledger_states"]) == ([], [])


@pytest.mark.parametrize("name", ["other/x.json.gz", "cuda/x.json", "cuda/broken.json.gz"])
def test_what_is_no_ledger_state_cannot_be_read(tmp_path, name):
    path = tmp_path / guard.STATES_DIR / name
    path.parent.mkdir(parents=True)
    path.write_bytes(b"not gzip")
    _hits, problems = guard.run_legs(_rules(), tmp_path, ["ledger_states"])
    assert problems and problems[0].startswith("ledger_states: cannot be read")


@pytest.mark.parametrize("packed", [
    gzip.compress(b'{"a": 1}' * 100, mtime=0)[:10] + b"\xff" * 8,
    gzip.compress(b"\xef\xbb\xbf{}", mtime=0),
    gzip.compress("{}".encode("utf-16"), mtime=0),
    gzip.compress(b"[" * 200000, mtime=0)],
    ids=["will_not_inflate", "bom", "utf16", "nested_past_the_parser"])
def test_a_state_the_package_could_not_read_cannot_be_read_here(tmp_path, packed):
    path = tmp_path / guard.STATES_DIR / "cuda" / ("a" * 64 + ".json.gz")
    path.parent.mkdir(parents=True)
    path.write_bytes(packed)
    _hits, problems = guard.run_legs(_rules(), tmp_path, ["ledger_states"])
    assert problems and problems[0].startswith("ledger_states: cannot be read"), problems


def test_a_file_the_desktop_leaves_in_the_archive_is_no_state(tmp_path):
    """Git ignores it, so no tracked tree holds it: it must not fail a local run. Any other
    stray name still does."""
    path = tmp_path / guard.STATES_DIR / "cuda" / ".DS_Store"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"\0\0\0\1Bud1")
    assert guard.run_legs(_rules(), tmp_path, ["ledger_states"]) == ([], [])
    (path.parent / ".hidden.json.gz").write_bytes(b"not gzip")
    _hits, problems = guard.run_legs(_rules(), tmp_path, ["ledger_states"])
    assert problems and problems[0].startswith("ledger_states: cannot be read")
