"""The Metal environment: which GPU, torch and Metal frontend, and whether each is certified.

A Metal environment is three facts, each of which changes the code that runs: the
GPU architecture Metal compiles for, the torch that supplies ``compile_shader``,
and the Metal frontend of the macOS build. Each cited weld keeps one run record per
GPU architecture (``runs[<architecture>]``), so the GPU architecture is certified when
every weld the Metal table cites has a live run for it, and torch and the frontend
when every such run recorded this host's. A fact is uncertified when a readable weld
contradicts it, and unjudged when it was not read or a weld's record cannot judge it.
These tests need no GPU except the one that reads this host's architecture.
"""

from __future__ import annotations

import pathlib
import re
import sys

import pytest

from conftest import requires_resource_skip

from meep_gpu import backends, fastpath, metal_dispatch, metal_runs
from meep_gpu.metal_kernels import device
from meep_gpu.test_capability_records import NUMBER_WORDS, _documentation

STAMPED = "this machine: Apple MPS device applegpu_g13s, torch 2.10.0, metalfe-32023.850.10"


@pytest.mark.parametrize("host,expected", [
    (STAMPED, {"architecture": "applegpu_g13s", "torch": "2.10.0",
               "metal_frontend": "32023.850.10"}),
    ("this machine: Apple MPS device, torch 2.10.0, metalfe-32023.850.10",
     {"architecture": None, "torch": "2.10.0", "metal_frontend": "32023.850.10"}),
    ("this machine: Apple MPS device, torch 2.10.0",
     {"architecture": None, "torch": "2.10.0", "metal_frontend": None}),
    ("this machine: Apple MPS device, torch ?, ?",
     {"architecture": None, "torch": None, "metal_frontend": None}),
    ("this Mac (Apple silicon, MPS); metal",
     {"architecture": None, "torch": None, "metal_frontend": None}),
    (None, {"architecture": None, "torch": None, "metal_frontend": None}),
])
def test_a_host_line_records_the_facts_it_names_and_none_it_does_not(host, expected):
    assert metal_dispatch.host_environment(host) == expected


def _cite(monkeypatch, *welds):
    """Substitute the cited welds' LIVE runs: one argument per weld.

    A weld is an environment (one live run), a tuple of environments (one live run
    per architecture) or ``None`` (an entry that cannot be read as runs).
    """
    def runs(weld):
        if weld is None:
            return None
        return tuple(dict(environment) for environment in
                     (weld if isinstance(weld, tuple) else (weld,)))

    monkeypatch.setattr(metal_dispatch, "cited_environments", lambda ledger=None: tuple(
        (f"metal_{index}_device_gate", runs(weld)) for index, weld in enumerate(welds)))


FULL = {"architecture": "applegpu_g13s", "torch": "2.10.0", "metal_frontend": "32023.850.10"}
SECOND = {"architecture": "applegpu_g15s", "torch": "2.10.0",
          "metal_frontend": "32023.850.10"}


@pytest.mark.parametrize("recorded,fact,value,verdict", [
    ([FULL, FULL], "architecture", "applegpu_g13s", True),
    ([FULL, FULL], "architecture", "applegpu_g15p", False),
    ([FULL, dict(FULL, architecture="applegpu_g15p")], "architecture", "applegpu_g13s",
     False),
    ([FULL, None], "architecture", "applegpu_g13s", None),
    ([FULL, None], "architecture", "applegpu_g15p", False),
    ([FULL, ()], "architecture", "applegpu_g13s", False),
    ([FULL], "architecture", None, None),
    ([], "torch", "2.10.0", None),
    ([FULL], "metal_frontend", "metalfe-32023.850.10", True),
    ([FULL], "metal_frontend", "32023.850.10", True),
    ([FULL], "metal_frontend", "metalfe-32023.851.1", False),
])
def test_a_fact_is_certified_only_where_every_cited_weld_recorded_it(
        monkeypatch, recorded, fact, value, verdict):
    """``()`` is a readable weld with no live run: it certifies no architecture."""
    _cite(monkeypatch, *recorded)
    architecture = None if fact == "architecture" else "applegpu_g13s"
    assert metal_dispatch.environment_verdict(fact, value, architecture) is verdict


def test_a_second_architecture_on_every_weld_is_certified_beside_the_first(monkeypatch):
    _cite(monkeypatch, (FULL, SECOND), (FULL, SECOND))
    for architecture in ("applegpu_g13s", "applegpu_g15s"):
        assert metal_dispatch.environment_verdict("architecture", architecture) is True
        assert metal_dispatch.environment_verdict("torch", "2.10.0", architecture) is True


def test_a_partial_round_certifies_nothing_new_and_takes_nothing_away(monkeypatch):
    """The intersection rule: one cited weld short and the second GPU is not certified."""
    _cite(monkeypatch, (FULL, SECOND), FULL)
    assert metal_dispatch.environment_verdict("architecture", "applegpu_g15s") is False
    assert metal_dispatch.environment_verdict("architecture", "applegpu_g13s") is True
    judged = metal_dispatch.environment_judgements("architecture", "applegpu_g15s")
    assert [verdict for _gate, verdict in judged] == [True, False]


def test_a_partial_round_does_not_name_the_second_architecture_as_certified(monkeypatch):
    """What a NOTE line calls certified obeys the intersection rule, as the verdict does.

    Every architecture a live run recorded is a candidate; only one every cited weld
    has a run for is listed. Otherwise a GPU one weld short of certified would read
    ``(certified: applegpu_g13s, applegpu_g15s)`` beside its own NOT CERTIFIED.
    """
    _cite(monkeypatch, (FULL, SECOND), FULL)
    assert metal_dispatch.environment_verdict("architecture", "applegpu_g15s") is False
    assert metal_dispatch.certified_values("architecture") == ["applegpu_g13s"]
    assert metal_dispatch.certified_values("torch", "applegpu_g15s") == []
    assert metal_dispatch.certified_values("torch", "applegpu_g13s") == ["2.10.0"]
    _cite(monkeypatch, (FULL, SECOND), (FULL, SECOND))
    assert metal_dispatch.certified_values("architecture") == [
        "applegpu_g13s", "applegpu_g15s"]
    assert metal_dispatch.certified_values("torch", "applegpu_g15s") == ["2.10.0"]
    # Runs that disagree certify neither value: each is contradicted by the other.
    _cite(monkeypatch, FULL, dict(FULL, torch="2.11.0"))
    assert metal_dispatch.certified_values("torch", "applegpu_g13s") == []


def test_torch_and_the_frontend_are_judged_against_this_architectures_runs(monkeypatch):
    """Each architecture's run records its own torch: one GPU's update is not the other's."""
    newer = dict(SECOND, torch="2.11.0")
    _cite(monkeypatch, (FULL, newer), (FULL, newer))
    assert metal_dispatch.environment_verdict("torch", "2.11.0", "applegpu_g15s") is True
    assert metal_dispatch.environment_verdict("torch", "2.11.0", "applegpu_g13s") is False
    assert metal_dispatch.environment_verdict("torch", "2.10.0", "applegpu_g13s") is True
    # An architecture no weld ran on has no run to read torch from: not judged, and
    # its own verdict is the one that says it is not certified.
    assert metal_dispatch.environment_verdict("torch", "2.10.0", "applegpu_g16x") is None
    # An unread architecture is judged against every live run.
    assert metal_dispatch.environment_verdict("torch", "2.10.0") is False
    _cite(monkeypatch, (FULL, SECOND), FULL)
    assert metal_dispatch.environment_verdict("torch", "2.10.0") is True


def test_the_recorded_environments_count_their_welds_most_common_first(monkeypatch):
    unreadable = {"architecture": None, "torch": None, "metal_frontend": None}
    _cite(monkeypatch, None, FULL, (FULL, SECOND))
    assert metal_dispatch.recorded_environments() == [
        dict(FULL, welds=2), dict(SECOND, welds=1), dict(unreadable, welds=1)]


def _ledger_with_runs(**runs_by_gate):
    """A ledger holding every cited weld, each with the named live runs."""
    ledger = {}
    for gate in metal_runs.cited_keys():
        entry = {"source_sha256": {"meep_gpu/metal_kernels/launch.py": "a" * 64},
                 "status": "PASS"}
        for environment in runs_by_gate.get(gate, (FULL,)):
            metal_runs.bind_architecture(
                entry, bound_before=None, architecture=environment["architecture"],
                run={"torch": environment["torch"],
                     "metal_frontend": environment["metal_frontend"]})
        ledger[gate] = entry
    return ledger


def test_a_run_whose_bytes_moved_is_not_offered_to_the_admission():
    """One liveness answer: ``fastpath.live_capabilities`` over the entry's bound."""
    ledger = _ledger_with_runs()
    assert metal_dispatch.environment_verdict(
        "architecture", "applegpu_g13s", ledger=ledger) is True
    gate = metal_runs.cited_keys()[0]
    ledger[gate]["source_sha256"]["meep_gpu/metal_kernels/launch.py"] = "b" * 64
    assert dict(metal_dispatch.cited_environments(ledger=ledger))[gate] == ()
    assert metal_dispatch.environment_verdict(
        "architecture", "applegpu_g13s", ledger=ledger) is False


def test_an_entry_in_the_one_run_shape_is_not_read_both_ways():
    """A weld still holding its host line beside its digests is unreadable, not certified."""
    ledger = _ledger_with_runs()
    gate = metal_runs.cited_keys()[0]
    ledger[gate] = {"source_sha256": {"x.py": "a" * 64}, "host": STAMPED}
    assert dict(metal_dispatch.cited_environments(ledger=ledger))[gate] is None
    assert metal_dispatch.environment_verdict(
        "architecture", "applegpu_g13s", ledger=ledger) is None


def test_the_shipped_ledger_cites_a_readable_entry_for_every_gate():
    cited = metal_dispatch.cited_environments()
    gates = {gate for _family, gate in metal_dispatch.ARM_CERTIFICATION.values()}
    assert [gate for gate, _runs in cited] == sorted(gates)
    assert metal_dispatch.unresolved_certification_rows() == ()
    unreadable = sorted(gate for gate, runs in cited if not runs)
    assert not unreadable, (
        f"{len(unreadable)} cited Metal welds hold no live per-architecture run: "
        f"{unreadable[:4]} (parity/meep_gpu/migrate_metal_runs.py moves a one-run "
        f"ledger; a stale run is repaired by re-running the gate)")


@pytest.mark.parametrize("value,restricted", [
    (None, False), ("supported", False), ("any", False), ("certified", True),
    ("0", False), ("1", False), ("Certified", False), ("", False)])
def test_only_certified_restricts_the_metal_kernels_to_certified_environments(
        monkeypatch, value, restricted):
    """The Metal record's ``certified_only`` reads the one scope reader; a value rung 3b
    refuses (the retired ``0`` included) restricts nothing, because it never runs."""
    monkeypatch.delenv(fastpath.RETIRED_HARDWARE_SWITCH, raising=False)
    if value is None:
        monkeypatch.delenv(fastpath.HARDWARE_SWITCH, raising=False)
    else:
        monkeypatch.setenv(fastpath.HARDWARE_SWITCH, value)
    assert metal_dispatch.environment_block(None, {})["certified_only"] is restricted


@pytest.mark.parametrize("value,certified", [
    (None, True), ("0", True), ("1", False), ("true", False)])
def test_fast_math_is_certified_only_unset_or_zero(value, certified):
    """Every weld ran with it unset; ``0`` measured identical, ``1`` did not."""
    block = metal_dispatch.environment_block(None, {"fast_math": value})
    assert block["fast_math_certified"] is certified


@pytest.mark.parametrize("architecture", ["applegpu_g13s", "applegpu_g15s", None])
def test_the_architecture_is_written_as_the_key_the_certification_quote_reads(
        architecture):
    """Option B: ``fastpath._certification_for`` reads ``device.compute_capability``.

    The Metal ledger keys its runs by GPU architecture where the NVIDIA ledgers key
    theirs by compute capability, so the device block offers the architecture under
    that key, and a dispatched arm quotes the run its weld recorded on THIS GPU. An
    unread architecture is not spelled as one: the quote then names no run and says
    why.
    """
    block = metal_dispatch.environment_block(None, {"architecture": architecture})
    if architecture is None:
        assert "compute_capability" not in block["device"], block["device"]
    else:
        assert block["device"]["compute_capability"] == architecture
        assert block["device"]["architecture"] == architecture


@pytest.mark.parametrize("architecture,name,supported", [
    ("applegpu_g15p", "Apple M3 Pro", True),
    ("applegpu_g13s", None, True),
    ("other_gpu_arch", "Apple M1 Max", False),
    (None, "Apple M1 Max", True),
    (None, "AMD Radeon Pro 5500M", False),
    (None, None, None),
])
def test_every_apple_gpu_is_supported_whether_or_not_it_is_certified(
        architecture, name, supported):
    """Supported reads the GPU's maker; certified is what the welds measured."""
    toolchain = {"architecture": architecture, "device_name": name}
    assert metal_dispatch.device_supported(toolchain) is supported
    assert metal_dispatch.environment_block(None, toolchain)["device_supported"] is supported


@pytest.fixture
def _fresh_identity(monkeypatch):
    monkeypatch.setattr(device, "_APPLE_GPU", None)
    yield
    monkeypatch.setattr(device, "_APPLE_GPU", None)


def test_the_gpu_identity_off_macos_is_unread_and_says_why(monkeypatch, _fresh_identity):
    monkeypatch.setattr(sys, "platform", "linux")
    assert device.apple_gpu_identity() == {
        "name": None, "architecture": None, "error": "not macOS (linux)"}


def test_the_gpu_identity_is_read_once_and_a_caller_cannot_change_it(_fresh_identity):
    first = device.apple_gpu_identity()
    first["architecture"] = "edited"
    assert device.apple_gpu_identity()["architecture"] != "edited"
    assert set(device.apple_gpu_identity()) == {"name", "architecture", "error"}


def test_this_apple_gpu_names_its_architecture(_fresh_identity):
    if backends.available_gpu() != "metal":
        requires_resource_skip("mps_device", "the architecture is read from an Apple GPU")
    identity = device.apple_gpu_identity()
    assert identity["error"] is None, identity
    assert re.fullmatch(r"applegpu_[a-z0-9_]+", identity["architecture"]), identity
    assert identity["name"], identity


# --- the fastpath hooks the Metal half reports through --------------------------------

def _admitted(*entries):
    record = {}
    for table, because in entries:
        fastpath._admit_uncertified(record, table, "torch", "0.0.0", ["2.10.0"],  # noqa: SLF001
                                    because=because)
    return record


def test_an_admission_under_any_records_and_says_what_it_always_did():
    record = _admitted(("triton", None))
    assert record["uncertified"]["admitted"] == [
        {"table": "triton", "what": "torch", "read": "0.0.0", "certified": ["2.10.0"]}]
    note = fastpath._uncertified_note(record)  # noqa: SLF001
    assert note.endswith(f". They were dispatched because {fastpath.HARDWARE_SWITCH}=any; "
                         "compare the results with a prefer_gpu=False run of the same "
                         "simulation before relying on them"), note


@pytest.mark.parametrize("reason", ["SUPPORTED_BECAUSE", "UNCERTIFIED_BECAUSE"])
def test_a_table_that_admits_by_default_says_why_in_the_note(reason):
    because = getattr(metal_dispatch, reason).format(switch=fastpath.HARDWARE_SWITCH)
    note = fastpath._uncertified_note(_admitted(("metal", because)))  # noqa: SLF001
    assert f"They were dispatched because {because};" in note, note
    assert f"{fastpath.HARDWARE_SWITCH}=certified" in because, because
    assert f"{fastpath.HARDWARE_SWITCH}=any" not in note, note


@pytest.mark.parametrize("device_certified,verdict", [
    ("absent", True), (True, True), (None, None)])
def test_the_metal_verdict_reads_the_gpu_where_the_record_names_one(device_certified,
                                                                    verdict):
    environment = {"torch_certified": True, "frontend_certified": True}
    if device_certified != "absent":
        environment["device_certified"] = device_certified
    record = {"environment": environment, "uncertified": {"admitted": []}}
    assert fastpath._certified_verdict(record, ["metal"]) is verdict  # noqa: SLF001


# --- what the documentation says the records certify ------------------------------------

DOCS_ROOT = pathlib.Path(__file__).resolve().parents[1]
APPLE_ARCHITECTURE = re.compile(r"\bapplegpu_[a-z0-9]+\b")


def _doc(name):
    return (DOCS_ROOT / name).read_text(encoding="utf-8")


def _apple_readme_row():
    """The README's Apple row: ``(identity cell, certified-on cell)``."""
    rows = [line for line in _doc("README.md").splitlines()
            if line.startswith("| Apple, Metal kernels |")]
    assert len(rows) == 1, rows
    cells = rows[0].split("|")
    assert len(cells) == 5, rows[0]
    return cells[2], cells[3]


def test_the_readme_states_the_apple_environments_the_metal_ledger_certifies():
    """The README's Apple row read against the records, as its NVIDIA rows are
    (``test_capability_records``). The identity cell names every GPU architecture the cited
    welds certify and no other, with the PyTorch each one's runs certify; the certified-on
    cell gives each architecture, by name, the Metal frontend its own runs certify. A second
    Apple GPU's records would otherwise land beside a row that names one, and an
    architecture re-certified on a later macOS build would leave the row naming the old
    frontend: both pass every other test.
    """
    identity, machines = _apple_readme_row()
    certified = metal_dispatch.certified_values("architecture")
    assert certified, "the Metal ledger certifies no GPU architecture"
    assert sorted(set(APPLE_ARCHITECTURE.findall(identity))) == certified, identity
    by_architecture = {}
    for part in machines.split(";"):
        named = re.match(r"\s*`(applegpu_[a-z0-9]+)`:", part)
        if named:
            by_architecture[named.group(1)] = part
    assert sorted(by_architecture) == certified, machines
    for architecture in certified:
        for torch in metal_dispatch.certified_values("torch", architecture):
            assert f"PyTorch {torch}" in identity, (architecture, torch, identity)
        frontends = metal_dispatch.certified_values("metal_frontend", architecture)
        assert frontends, f"no Metal frontend is certified for {architecture}"
        printed = set(re.findall(r"`metalfe-([0-9][0-9.]*)`", by_architecture[architecture]))
        assert printed == set(frontends), (architecture, sorted(printed), frontends)


#: The documents that quote what the package prints on an Apple GPU it does not certify.
QUOTED_NOTES = ("README.md", "INSTALL.md", "docs/guides/kernel-dispatch.md")
NOTE_QUOTE = re.compile(r"GPU architecture (applegpu_[a-z0-9]+) \(certified: ([^)]*)\)")
REFUSAL_QUOTE = re.compile(r"GPU architecture (applegpu_[a-z0-9]+) is not the one every Metal "
                           r"weld this table cites ran on \(certified: (\[[^\]]*\])\)")


@pytest.mark.parametrize("name", QUOTED_NOTES)
def test_a_quoted_apple_note_or_refusal_lists_what_the_records_certify(name):
    """A quoted NOTE line or refusal is text the package prints, so the certified list in
    it is :func:`metal_dispatch.certified_values` in the form each prints it: joined with
    commas in a NOTE line (``fastpath._uncertified_note``), as a list in a refusal. The
    quoted GPU is one the records do not certify, or the quote shows a line no run prints.
    """
    text = _doc(name)
    certified = metal_dispatch.certified_values("architecture")
    notes, refusals = NOTE_QUOTE.findall(text), REFUSAL_QUOTE.findall(text)
    assert notes or refusals, f"{name} quotes no Apple NOTE line or refusal"
    for quoted, listed in notes:
        assert quoted not in certified, (name, quoted)
        assert listed == ", ".join(certified), (name, listed, certified)
    for quoted, listed in refusals:
        assert quoted not in certified, (name, quoted)
        assert listed == str(certified), (name, listed, certified)


#: One-line statements of the certified Apple set, each found by how its line starts.
CERTIFIED_SET_LINES = (
    ("INSTALL.md", "| Metal (Apple) | the certified environment"),
    ("docs/guides/kernel-dispatch.md", "| GPU architecture | <code>applegpu_"),
    ("docs/development/certification.md", "| GPU architecture | <code>MTLDevice.architecture"),
    ("docs/development/certify-a-new-gpu.md", "If `ARCH` reads "),
    ("docs/reference/glossary.md", "| **Certified environment** |"),
    ("docs/reference/glossary.md", "| **Certified identity** |"),
    ("docs/guides/kernel-dispatch.md", "certified side by side; 1.0.1 certifies "),
    ("docs/development/execution-backends.md", "| Metal kernel table |"),
    # The new-GPU guide's Apple after-bind example: the runs a new architecture's run is
    # written beside, and the admission it is listed beside, are the certified set's.
    ("docs/development/certify-a-new-gpu.md", "beside the `applegpu_"),
    ("docs/development/certify-a-new-gpu.md", "lists your architecture beside "),
    # The certified-environment tables' frontend rows, which give each architecture its own.
    ("docs/development/certification.md", "| Metal frontend | "),
    ("docs/guides/kernel-dispatch.md", "| Metal frontend | "),
)

#: A Metal frontend as the documentation writes one: ``metalfe-<version>``, or INSTALL.md's
#: bare ``Metal frontend <version>``.
METAL_FRONTEND = re.compile(r"(?:metalfe-|Metal frontend )([0-9]+(?:\.[0-9]+)+)")


def _named_set(statement):
    """The Apple GPU architectures ``statement`` names, sorted, and for each the first Metal
    frontend written after it and before the next architecture: that is the frontend the
    statement certifies it with. A later version in the same stretch (the M1 Max's newer
    macOS, supported and uncertified) is not read."""
    names = list(APPLE_ARCHITECTURE.finditer(statement))
    frontends = {}
    for index, name in enumerate(names):
        stop = names[index + 1].start() if index + 1 < len(names) else len(statement)
        frontend = METAL_FRONTEND.search(statement, name.end(), stop)
        if frontend:
            frontends.setdefault(name.group(0), frontend.group(1))
    return sorted({name.group(0) for name in names}), frontends


def _assert_names_the_certified_set(where, statement, *, names_required=True):
    """``statement`` names exactly the architectures the cited welds certify (when it names
    any, or always when ``names_required``), and, when it names a Metal frontend, names
    each architecture's own certified frontend."""
    certified = metal_dispatch.certified_values("architecture")
    named, frontends = _named_set(statement)
    if names_required or named:
        assert named == certified, (where, named, certified, statement)
    if frontends:
        wrong = {architecture: frontends.get(architecture) for architecture in certified
                 if frontends.get(architecture) not in
                 metal_dispatch.certified_values("metal_frontend", architecture)}
        assert not wrong, (where, "frontend stated per architecture", wrong, statement)


@pytest.mark.parametrize("name, start", CERTIFIED_SET_LINES)
def test_a_one_line_statement_of_the_certified_apple_set_names_what_the_records_certify(
        name, start):
    """Each of these lines states which Apple GPU architectures are certified, and nothing
    else in the tree checks it: it is read against the cited welds, so a certification that
    adds or retires an architecture fails here until the line says so. A line that names a
    Metal frontend gives each architecture the one its own runs certify, so two frontends
    swapped between architectures, or an older one left beside a re-certified one, fail too.
    """
    lines = [line for line in _doc(name).splitlines() if line.startswith(start)]
    assert len(lines) == 1, (name, start, lines)
    _assert_names_the_certified_set((name, start), lines[0])


#: The end of a sentence in flattened text, or of a table cell.
SENTENCE_END = re.compile(r"\.(?:\*\*)?\s+(?=[A-Z])|\s\|(?:\s|$)|$")

#: The sentences that count the certified Apple environments. Each is read from the count to
#: the end of its sentence, wrapped lines joined.
APPLE_COUNT_STATEMENTS = (
    r"\bcompute capabilit(?:y|ies) and (\w+) Apple environments?\b",
    r"\bon Apple, (\w+) certified environments?\b",
    r"\bcertified (?:on|in) (\w+) environments?: (?:the )?GPU architecture\b",
    r"\bEvery Apple GPU supported, (\w+) environments? certified\b",
    r"\band (\w+) (?:is|are) certified \(`applegpu_",
)

#: The pages a count of the certified Apple environments must be read on.
APPLE_COUNT_PAGES = ("README.md", "INSTALL.md", "AGENTS.md",
                     "docs/development/certify-a-new-gpu.md",
                     "docs/getting-started/installation.md",
                     "docs/getting-started/reading-what-ran.md",
                     "docs/guides/kernel-dispatch.md", "docs/guides/will-it-help.md")


def _sentence(text, start):
    return text[start:SENTENCE_END.search(text, start).start()]


def test_every_count_of_certified_apple_environments_is_the_metal_ledgers():
    """A sentence that counts the certified Apple environments counts the architectures the
    cited welds certify, and the architectures and frontends it goes on to name are those.
    The NVIDIA count in the same sentence is ``test_capability_records``'s; this is the
    Apple half. Each sentence is read with its wrapped lines joined, and the count must be
    read on every page that states one, so a rewording this no longer reads fails here.
    """
    certified = metal_dispatch.certified_values("architecture")
    found, wrong = set(), []
    for name, text in _documentation().items():
        for pattern in APPLE_COUNT_STATEMENTS:
            for match in re.finditer(pattern, text):
                word = match.group(1).lower()
                count = int(word) if word.isdigit() else NUMBER_WORDS.get(word)
                found.add(name)
                if count != len(certified):
                    wrong.append((name, match.group(0)))
                _assert_names_the_certified_set(name, _sentence(text, match.start()),
                                                names_required=False)
    missing = sorted(set(APPLE_COUNT_PAGES) - found)
    assert not missing, (
        f"no count of the certified Apple environments was read on {missing}; a reworded "
        "count must still be read here")
    assert wrong == [], (
        f"the Metal welds certify {len(certified)} architectures ({certified}) and these "
        f"sentences count otherwise: {wrong}")


#: Statements of the certified Apple set that wrap and count nothing, each found by how it
#: starts in the flattened page and read to the end of its sentence.
CERTIFIED_SET_SENTENCES = (
    ("README.md", "and other Macs whose GPU reports "),
)


@pytest.mark.parametrize("name, start", CERTIFIED_SET_SENTENCES)
def test_a_wrapped_statement_of_the_certified_apple_set_names_what_the_records_certify(
        name, start):
    """As the one-line statements, for a sentence wrapped over several lines."""
    text = _documentation()[name]
    assert text.count(start) == 1, (name, start, text.count(start))
    _assert_names_the_certified_set((name, start), _sentence(text, text.index(start)))


def _apple_machines():
    """The machine each certified architecture was measured on, as the README's Apple row
    names it in its certified-on cell (``applegpu_g13s: one M1 Max, ...``)."""
    _identity, machines = _apple_readme_row()
    named = {}
    for part in machines.split(";"):
        machine = re.match(r"\s*`(applegpu_[a-z0-9]+)`: one ([^,]+),", part)
        if machine:
            named[machine.group(1)] = machine.group(2).strip()
    return named


#: The sentences that count the machines measured for the records.
MACHINE_STATEMENTS = (
    r"\b\w+ machines have been measured for these records:",
    r"\bon \w+ machines:",
)


def test_every_count_of_the_machines_measured_names_each_certified_apple_machine():
    """A sentence that counts the machines measured for the records lists them, and the list
    names the machine each certified Apple architecture was measured on (the README row's,
    which its own test holds to the records). Its number is not derived: one such sentence
    counts the RTX A6000 whose runs are kept and the other does not."""
    machines = _apple_machines()
    assert sorted(machines) == metal_dispatch.certified_values("architecture"), machines
    found, wrong = set(), []
    for name, text in _documentation().items():
        for pattern in MACHINE_STATEMENTS:
            for match in re.finditer(pattern, text):
                found.add(name)
                sentence = _sentence(text, match.start())
                unnamed = sorted(machine for machine in machines.values()
                                 if not re.search(rf"\b(?:the|one) {re.escape(machine)}\b",
                                                  sentence))
                if unnamed:
                    wrong.append((name, unnamed, sentence))
    assert {"README.md", "INSTALL.md"} <= found, (
        f"no count of the machines measured was read in README.md and INSTALL.md (read in "
        f"{sorted(found)})")
    assert wrong == [], f"these sentences leave out a certified Apple machine: {wrong}"


#: The statement that a certified Apple architecture has no timing record of its own, in the
#: shape of the compute-capability one (``test_capability_records``): the architecture, the
#: one whose prior its default reads, and whether that prior gives way anywhere. Markdown
#: pages quote an architecture in backticks, the HTML tables in ``<code>``.
UNTIMED_APPLE_STATEMENT = re.compile(
    r"GPU architecture (?:`|<code>)(applegpu_[a-z0-9]+)(?:`|</code>) has no record of its own: "
    r"its default reads (?:`|<code>)(applegpu_[a-z0-9]+)(?:`|</code>)'s, the nearest timed "
    r"Apple architecture(, whose prior gives way nowhere)?")


def _shipped_prior():
    """The portable prior the shipped timing records give on this tree, derived as the
    dispatch path derives it (``fastpath``'s ``the_prior``): every record file loaded, judged
    against the live deposit-repair route and each table's live subjects."""
    from meep_gpu import dispatch_preference as dp  # noqa: PLC0415

    root = pathlib.Path(dp.__file__).resolve().parent
    loaded, refused = [], []
    for path in dp.record_files(str(root)):
        name = pathlib.Path(path).relative_to(root).as_posix()
        try:
            loaded.append((name, dp.load_record(path)))
        except dp.RecordRefused as exc:
            refused.append((name, str(exc)))
    subjects = {}
    for table in dp.TABLES:
        try:
            subjects[table] = dp.live_subject_sha256(table)
        except OSError:
            subjects[table] = {}
    return dp.portable_prior(loaded, repair_route_sha256=dp.repair_route_sha256(),
                             subject_sha256_by_table=subjects, refused=refused)


def test_a_certified_apple_architecture_no_record_times_is_stated_as_reading_the_nearest_one():
    """Certified is not timed. A certified Apple architecture that no shipped Metal timing
    record times reads the bracketed-veto prior of the nearest timed one
    (``dispatch_preference.nearest_architecture``), never its record. The README says so for
    each such architecture, by architecture as the compute-capability sentence is
    (``test_capability_records`` holds that half), and every page that says it names the
    architecture the rule reads. The sentence says the prior gives way nowhere exactly when
    the shipped prior of that architecture has no ceiling (its records measure no bracketed
    veto below their keeps), so a record cut later for the untimed architecture, or a re-cut
    that gives the nearest one a ceiling, fails here by name. A page that names the nearest
    timed Apple architecture in a form this does not read fails too.
    """
    from meep_gpu import dispatch_preference as dp  # noqa: PLC0415

    root = pathlib.Path(dp.__file__).resolve().parent
    timed = sorted({architecture for record in dp.timed_vendors()["apple"]["records"]
                    for architecture in dp.record_capabilities(dp.load_record(str(root / record)))})
    assert timed, "no shipped Metal timing record times its architecture on this tree"
    reads = {architecture: dp.nearest_architecture("apple", architecture, timed)[0]
             for architecture in metal_dispatch.certified_values("architecture")
             if architecture not in timed}
    priors = _shipped_prior()["by_vendor"]["apple"]["by_architecture"]
    stated, wrong = {}, []
    for name, text in _documentation().items():
        found = UNTIMED_APPLE_STATEMENT.findall(text)
        if len(found) != text.count("the nearest timed Apple architecture"):
            wrong.append((name, "names the nearest timed Apple architecture in a form not read",
                          text.count("the nearest timed Apple architecture"), len(found)))
        for architecture, read, nowhere in found:
            stated.setdefault(architecture, set()).add(name)
            if reads.get(architecture) != read:
                wrong.append((name, architecture, read,
                              reads.get(architecture, "a record of its own")))
                continue
            ceiling = (priors.get(read) or {}).get("bracketed_veto_up_to_cells", "no prior")
            gives_way_nowhere = read in priors and ceiling is None
            if bool(nowhere) != gives_way_nowhere:
                wrong.append((name, architecture, read, "prior gives way nowhere stated "
                              f"{bool(nowhere)}, shipped prior {priors.get(read)}"))
    assert wrong == [], (
        "these sentences name the prior an Apple architecture reads wrongly (page, "
        f"architecture, stated, what the rule and the shipped records give): {wrong}")
    unstated = sorted(architecture for architecture in reads
                      if "README.md" not in stated.get(architecture, ()))
    assert unstated == [], (
        f"Apple architectures {unstated} are certified with no timing record of their own, "
        "and the README does not say whose prior their default reads")
