"""Every CUDA certification block says WHEN it ran, WHERE, on WHICH BYTES, under WHICH POLICY.

WHAT THIS FILE IS FOR. ``certification.json`` is stronger than a
``fingerprints.json`` weld on the axis that matters most: it pins the device
source STRINGS the compiler is handed, evaluated from the syntax tree with no
CuPy import, and it carries a certified/dead partition over all fourteen shipped
kernels. ``test_certification_record.py`` enforces all of that. What it never
asked for is the provenance of the verdicts themselves.

MEASURED 2026-08-19, before this file existed, over the four blocks in the record
that name a device:

    block                     when            where             which bytes  policy
    bit_identity_gate         started_utc     environment.hostname  -        -
    recut_2026-08-15          -               -                 -            both
    constitutive_2026-08-15   -               null              -            both
    offdiag_2026-08-16        -               -                 -            both

Zero of four carried a ``recorded_utc``; one of four carried a host, and a second
carried the KEY with ``null`` in it. Meanwhile the Metal track had just made all
four fields a merge-bar requirement (``test_metal_weld_contract.py``) and the
Triton weld test had been rewritten to ENUMERATE every passing entry rather than
name gates by hand — because naming them by hand is how six welds shipped as
orphans that nothing recomputed.

So this file ports both lessons to the CUDA record, and adds the one the other two
tracks do not have. Metal requires ``artifact_sha256`` to be PRESENT; nothing
recomputes it. Here the artifact digest, the timestamp, the host and the policy are
each checked AGAINST THE ARTIFACT wherever the directory is on the machine — which
is what catches a value transcribed from the wrong run, the failure
``gate_provenance.py`` was built after a weld took a digest from the wrong one of
the GPU host's 67 staged trees.

WHAT WAS FIXED IN THE RECORD RATHER THAN DECLARED. Three blocks gained
``recorded_utc``, ``host`` and ``artifact_sha256``, and the constitutive block's
``"hostname": null`` became ``"host": "the GPU host"`` — every value read out of the
artifacts in this checkout, none typed from memory. The null had a cause worth
recording: ``summarize_constitutive_recut.py:215`` reads ``environment.hostname``
off the gate payload, and NO CUDA gate payload has ever carried that key (measured
on ``keep/cgate.json`` and on the 2026-08-09 ``gate.json`` — identical shape, both
hostname-free). The host is stamped only in the probe's ``PROVENANCE.txt`` sidecar.

WHAT IS DECLARED DEBT. One thing, and it is not one of the four facts:
``offdiag_2026-08-16``'s host exists only in a hand-written PROVENANCE.md, because
that gate writes no sidecar and its payload carries the device without the
hostname. It is carried in ``HOST_NOT_MACHINE_STAMPED`` — the host is REAL and
recorded, it is the EVIDENCE CLASS that is weaker — and closing it means the gate
stamping its own hostname and a re-cut.

WHAT WAS DISCHARGED, 2026-08-20. ``METADATA_DEBT`` is now EMPTY, and
``DEBT_BUDGET`` is zero on all four facts. Its one entry said ``bit_identity_gate``
could not record a subnormal policy because the 2026-08-09 run predated the policy
machinery. The folded-PERIODIC top-plane branch then moved both certified curl
device strings, so that verdict described bytes that no longer shipped and the
gate was RE-CUT on the shipped ones — a full leg under each policy, in a private
CUPY_CACHE_DIR named for the resolved policy
(``results/fused_pml_bit_identity_hand_2026-08-22_rename/{keep,flush}/``). The
debt was therefore discharged by a MEASUREMENT and the entry deleted, which is the
only way an entry here is allowed to leave.

A zero budget on every fact is the intended terminal state and it is a hard one:
the budget is DERIVED from the debt list rather than asserted beside it, so a block
that cannot record one of the four can only be parked here by writing an entry that
names it and says why — and the artifact clauses below then check that the RUN
really lacks the fact. There is no allowance to spend. That is deliberate: the
thing such a block needs is a device run, not a dictionary entry.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import re

import pytest

HERE = pathlib.Path(__file__).parent
RECORD = HERE / "certification.json"

#: the repository root, from which the record's artifact paths are written.
REPO = HERE.parents[1]

#: The prefix every block's artifact path carries. Blocks spell the key
#: ``artifact`` or ``artifacts`` and some append " (gitignored)"; the path is
#: extracted rather than listed, for the same reason the blocks are.
ARTIFACT_PREFIX = "parity/meep_gpu/results/"

#: The four facts. Names chosen to match the Metal weld contract so one reader
#: serves both tracks; the per-block spellings the CUDA record actually uses are
#: resolved in :func:`_fact`.
REQUIRED = ("recorded_utc", "host", "artifact_sha256", "subnormal_policy")

#: DECLARED DEBT: (block, fact) the record cannot carry, and why. Closing an entry
#: means a device run, never an edit to this dict.
#:
#: EMPTY since 2026-08-20. The single entry — ``bit_identity_gate`` owing a
#: subnormal policy, because the 2026-08-09 cut predated the policy machinery —
#: was discharged by the re-cut on the shipped device strings, which ran a full leg
#: under ``ieee_keep_ftz_stripped`` and one under ``meep_x86_flush``. The entry was
#: deleted rather than re-worded, which is what the test below requires of a debt
#: whose fact has arrived.
METADATA_DEBT: dict[tuple[str, str], str] = {}

#: Per fact, how many blocks may owe it. MAY ONLY GO DOWN. All four reached zero
#: on 2026-08-20; the floor is where they stay.
DEBT_BUDGET = {"recorded_utc": 0, "host": 0, "artifact_sha256": 0,
               "subnormal_policy": 0}

#: Blocks whose host is real but was read off PROSE rather than a stamped field.
#: A weaker kind of evidence, so it is named rather than passed over in silence.
HOST_NOT_MACHINE_STAMPED = {
    "offdiag_2026-08-16":
        "gate_cuda_offdiag.py writes no PROVENANCE.txt and its payload's "
        "environment block carries device_name but no hostname, so the only "
        "record of the host is a sentence in the directory's PROVENANCE.md. "
        "Closing it means the gate stamping its own hostname and a re-cut.",
    "cuda_folded_offdiag_2026-08-21":
        "gate_cuda_folded_offdiag_kernel.py writes no PROVENANCE.txt and stamps "
        "no top-level host; its payload's environment block carries device_name, "
        "compute capability and the CUDA_VISIBLE_DEVICES UUID but no hostname. "
        "The host is recorded only in folded_offdiag_kernels.FOLDED_OFFDIAG_"
        "ADMISSION's prose. Closing it means the gate stamping its own hostname "
        "and a re-cut.",
    "cuda_dispersive_offdiag_2026-08-20":
        "gate_cuda_dispersive_offdiag.py has the same gap as its folded sibling: "
        "no PROVENANCE.txt, no top-level host, an environment block without a "
        "hostname. The host is a sentence in the directory's PROVENANCE.md. "
        "Closing it means the gate stamping its own hostname and a re-cut.",
}

#: The COUNT of the dict above rather than an allowance beside it — the test
#: derives it, so it moves only when an entry is added or discharged.
#:
#: 1 -> 3 on 2026-08-28, and it is an EXPANSION OF THE RECORD rather than a
#: loosening of it. The two folded/dispersive off-diagonal verdicts were RELEASED
#: on a device in August and had no block here at all, so nothing checked their
#: timestamps, their policies or their artifact digests. Transcribing them brings
#: three of the four facts under machine check for the first time and leaves the
#: fourth — the host — declared as the prose it has always been. Both entries close
#: the same way: the gate stamps its own hostname and the leg is re-cut.
HOST_TRANSCRIPTION_BUDGET = 3

#: THE RELEASE SPELLING OF A MACHINE'S NAME. The release export replaces the name of
#: the machine a run stamped with this phrase, in every record it ships, and a bind
#: of a new round applies the same rule before the record is committed. A record
#: holding it therefore says "the one machine the artifact stamped", and that is what
#: :func:`test_the_recorded_host_is_the_one_the_artifact_stamped` checks: the artifact
#: must stamp a hostname, and exactly one, because one placeholder cannot stand for two
#: machines. Any other recorded value must still be one the artifact stamped.
RELEASE_HOST_PLACEHOLDER = "the GPU host"


def record() -> dict:
    return json.loads(RECORD.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# Enumeration, not a hand list.
# --------------------------------------------------------------------------

def blocks(entry: dict | None = None) -> dict:
    """Every top-level object in the record that claims a DEVICE measurement.

    ENUMERATED. The Triton weld test named three gates by hand and six welds
    added the same day were therefore orphans — recorded digests that nothing
    recomputed. The discriminator here is the claim itself: a block that says it
    ran on a device is a block that has to say when, where, on what and under
    which policy. ``mutations``, ``lifted_cases``, ``harness_2026-08-15`` and
    ``dispatch`` name no device and are legs or descriptions, not verdicts.
    """
    entry = record() if entry is None else entry
    found = {}
    for name, value in entry.items():
        if not isinstance(value, dict):
            continue
        environment = value.get("environment")
        device = value.get("device") or (
            environment.get("device_name") if isinstance(environment, dict) else None)
        if device:
            found[name] = value
    return found


def artifact_directory(block: dict) -> pathlib.Path | None:
    """The results directory this block's verdict lives in, off its own text."""
    for key in ("artifact", "artifacts"):
        text = block.get(key)
        if not isinstance(text, str) or ARTIFACT_PREFIX not in text:
            continue
        relative = text[text.index(ARTIFACT_PREFIX):].split()[0].rstrip("/")
        return REPO / relative
    return None


def _fact(block: dict, name: str):
    """One of the four facts, in whichever spelling this block uses.

    The spellings are not sloppiness. ``bit_identity_gate``'s ``environment`` is a
    verbatim transcription of the probe's own payload block plus the ``hostname``
    the probe stamps only in its ``PROVENANCE.txt`` sidecar, and both halves are
    checked against those two files below, so renaming ``hostname`` inside it would
    break the one thing that makes the transcription auditable.
    ``policies_cut_under`` is a list because these blocks were cut under BOTH
    policies, which a scalar cannot say. A reader that knows the spellings is the price of a record whose
    fields still mean what their source meant — the same trade the Metal track's
    ``read_verdict`` makes across ten gate spellings.
    """
    environment = block.get("environment")
    environment = environment if isinstance(environment, dict) else {}
    if name == "recorded_utc":
        return block.get("recorded_utc") or block.get("started_utc")
    if name == "host":
        return block.get("host") or environment.get("hostname")
    if name == "artifact_sha256":
        return block.get("artifact_sha256")
    if name == "subnormal_policy":
        return block.get("subnormal_policy") or block.get("policies_cut_under")
    raise AssertionError(f"unknown fact {name!r}")


def test_the_blocks_are_enumerated_from_the_record_not_listed_here():
    found = blocks()
    assert len(found) >= 4, (
        f"only {len(found)} device blocks discovered — the device/environment "
        f"shape changed and this file would now enforce almost nothing")
    for name, block in sorted(found.items()):
        directory = artifact_directory(block)
        assert directory is not None, (
            f"{name} claims a device measurement and names no artifact directory, "
            f"so nothing it says can ever be checked against what ran")
        assert ARTIFACT_PREFIX in str(directory.as_posix()), name


def test_a_block_added_without_metadata_is_caught():
    """The blindness control: this file must be able to FAIL.

    A requirement whose discovery step quietly misses new work enforces nothing,
    which is exactly how the six orphaned Triton welds happened. So a synthetic
    block is injected and every one of the four facts must come back missing —
    including the ``null`` spelling, since a key present and empty is what the
    constitutive block carried while reading as a block that recorded its host.
    """
    entry = record()
    entry["synthetic_device_block"] = {"device": "NVIDIA RTX A6000", "host": None}
    found = blocks(entry)
    assert "synthetic_device_block" in found, "the discriminator missed a new block"
    for name in REQUIRED:
        assert not _fact(found["synthetic_device_block"], name), name


# --------------------------------------------------------------------------
# The requirement, and the debt it is allowed to carry.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("fact", REQUIRED)
def test_every_certification_block_records_its_provenance(fact):
    owing = sorted(name for name, block in blocks().items() if not _fact(block, fact))
    undeclared = [name for name in owing if (name, fact) not in METADATA_DEBT]
    assert not undeclared, (
        f"{undeclared} claim a device measurement and do not record {fact}. Read "
        f"it out of the block's own artifact directory — do not add it to "
        f"METADATA_DEBT to make this pass unless the RUN genuinely lacks it.")
    assert len(owing) <= DEBT_BUDGET[fact], (
        f"{len(owing)} blocks owe {fact} and the budget is {DEBT_BUDGET[fact]}: "
        f"{owing}. The budget may only shrink.")


def test_the_declared_debt_still_describes_real_debt():
    """A cleared entry must be REMOVED, so the list cannot rot into fiction.

    AND THE BUDGET IS DERIVED FROM THE LIST, not asserted beside it. MEASURED
    2026-08-20 by mutation: with the budget merely commented "MAY ONLY GO DOWN",
    putting ``subnormal_policy`` back from 0 to 1 — undoing the reduction the
    re-cut earned — was caught by NOTHING. Every other clause here reads the debt
    LIST, and the budget passed at 1 because zero blocks owed the fact. So the two
    are welded: a budget larger than the debts that account for it is a parking
    space, and lowering the list without lowering the budget re-opens it.
    """
    present = blocks()
    stale = [(name, fact) for (name, fact) in METADATA_DEBT
             if name in present and _fact(present[name], fact)]
    assert not stale, (
        f"METADATA_DEBT names {stale}, which the record now carries — delete the "
        f"entry; a debt list that outlives its debt stops being read")
    unknown = [(name, fact) for (name, fact) in METADATA_DEBT if name not in present]
    assert not unknown, f"METADATA_DEBT names blocks that no longer exist: {unknown}"
    assert all(reason.strip() for reason in METADATA_DEBT.values()), \
        "a debt entry with no reason is an exemption wearing a debt's clothes"

    stale_hosts = [name for name in HOST_NOT_MACHINE_STAMPED if name not in present]
    assert not stale_hosts, stale_hosts

    for fact in REQUIRED:
        declared = sum(1 for (_, owed) in METADATA_DEBT if owed == fact)
        assert DEBT_BUDGET[fact] == declared, (
            f"DEBT_BUDGET[{fact!r}] is {DEBT_BUDGET[fact]} and {declared} block(s) "
            f"declare that debt. The budget is not an allowance — it is a count of "
            f"the entries above, so it goes down when one is discharged and can only "
            f"go up by adding an entry that names a block and a reason.")
    assert HOST_TRANSCRIPTION_BUDGET == len(HOST_NOT_MACHINE_STAMPED), (
        f"HOST_TRANSCRIPTION_BUDGET is {HOST_TRANSCRIPTION_BUDGET} and "
        f"{sorted(HOST_NOT_MACHINE_STAMPED)} is declared; same rule")


def test_the_manifest_rule_is_written_where_someone_holding_the_artifact_can_use_it():
    """A digest nobody can recompute is a number, not evidence.

    ``artifact_sha256`` covers a directory the repository does not track, so the
    rule that produced it has to travel with the record rather than living only
    in this file — otherwise the only way to check a block is to already have
    this test.
    """
    rule = record()["_artifact_sha256_rule"]
    for clause in ("*.json", "sort by", "relative", "newline", "sha256"):
        assert clause in rule, f"the manifest rule does not state {clause!r}"
    convention = record()["_metadata_convention"]
    for fact in REQUIRED:
        assert fact in convention, f"the convention does not name {fact}"


# --------------------------------------------------------------------------
# The half that needs the artifact: recorded AGAINST what ran.
# --------------------------------------------------------------------------
#
# results/ is untracked, so on most hosts these are declared skips and the
# presence requirements above still hold. Where the directories ARE present this
# is the part that catches a value taken from the wrong run — which is a real
# failure mode on this project and not a hypothetical one.


def _artifact_stamps(directory: pathlib.Path) -> dict:
    """What the artifact itself says about when, where and under which policy.

    Read from BOTH homes because the CUDA fleet uses both: the probe's
    ``PROVENANCE.txt`` sidecar, and the gate payloads' own keys.

    THE HOSTNAME HAS TWO SPELLINGS, and reading only one is how a machine-stamped
    fact gets recorded as prose. The 2026-08-09 probe writes ``environment.hostname``
    into its sidecar; ``gate_cuda_ade.py`` and ``gate_cuda_folded_constitutive.py``
    stamp a TOP-LEVEL ``host`` in the payload and leave ``environment`` to the
    device. MEASURED 2026-08-19 across every artifact directory the record names:
    the four pre-existing ones carry NEITHER key in any payload (their hosts come
    from sidecars and prose), and the two added that day carry only the top-level
    one. So adding this read strengthens the check and cannot retire the
    ``HOST_NOT_MACHINE_STAMPED`` entry that is still real.
    """
    utcs, hosts, policies = set(), set(), set()
    for path in sorted(directory.rglob("PROVENANCE.txt")):
        text = path.read_text(encoding="utf-8", errors="replace")
        for key, sink in (("started_utc", utcs), ("hostname", hosts),
                          ("subnormal_policy", policies)):
            match = re.search(rf"^{key}\s+(\S+)\s*$", text, re.M)
            if match:
                sink.add(match.group(1))
    for path in sorted(directory.rglob("*.json")):
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
        if not isinstance(document, dict):
            continue
        if isinstance(document.get("started_utc"), str):
            utcs.add(document["started_utc"])
        environment = document.get("environment")
        if isinstance(environment, dict) and environment.get("hostname"):
            hosts.add(environment["hostname"])
        if isinstance(document.get("host"), str) and document["host"]:
            hosts.add(document["host"])
        for key in ("subnormal_policy_install", "subnormal_policy_stamp"):
            value = document.get(key)
            if not isinstance(value, dict):
                continue
            if isinstance(value.get("policy"), str):
                policies.add(value["policy"])
            stamp = value.get("stamp")
            if isinstance(stamp, dict) and isinstance(stamp.get("policy"), str):
                policies.add(stamp["policy"])
    return {"started_utc": utcs, "hostname": hosts, "subnormal_policy": policies}


def _manifest(directory: pathlib.Path):
    """(digest, {relative path: sha256}) by the rule stated in the record."""
    table = {path.relative_to(directory).as_posix():
             hashlib.sha256(path.read_bytes()).hexdigest()
             for path in directory.rglob("*.json")}
    lines = [f"{name} {digest}" for name, digest in sorted(table.items())]
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest(), table


def _present_blocks():
    """Blocks whose artifact directory is on this machine."""
    out = {}
    for name, block in sorted(blocks().items()):
        directory = artifact_directory(block)
        if directory is not None and directory.is_dir():
            out[name] = (block, directory)
    return out


def _require_artifacts():
    present = _present_blocks()
    if not present:
        pytest.skip("no CUDA gate artifact directory is on this host")
    return present


@pytest.mark.requires_resource("cuda-gate-artifact")
def test_the_recorded_time_is_a_stamp_the_artifact_actually_carries():
    """Not merely well-formed: PRESENT in the run it claims to describe.

    Membership rather than equality, because a block is several device legs and
    each stamps its own start. Every block holds the EARLIEST stamp in its tree —
    when its first device leg began. ``bit_identity_gate`` used to be the exception,
    holding the single gate leg's own stamp because it had only one; the 2026-08-20
    re-cut gave it a leg per float32 policy, so it now follows the same rule as the
    rest and records the earlier of the two.
    """
    for name, (block, directory) in _require_artifacts().items():
        recorded = _fact(block, "recorded_utc")
        stamps = _artifact_stamps(directory)["started_utc"]
        assert stamps, f"{name}: the artifact stamps no start time at all"
        assert recorded in stamps, (
            f"{name}: records {recorded!r}, which no leg in {directory.name} "
            f"stamped (earliest is {min(stamps)!r}). A timestamp from another run "
            f"is how a record starts describing a run that never happened.")
        assert recorded.endswith("Z"), f"{name}: ISO-8601 UTC expected, got {recorded!r}"


@pytest.mark.requires_resource("cuda-gate-artifact")
def test_the_recorded_host_is_the_one_the_artifact_stamped():
    """A recorded host is a name the artifact stamped, or the release placeholder over
    an artifact that stamps exactly one hostname; a block whose artifact stamps none
    must be declared in ``HOST_NOT_MACHINE_STAMPED``."""
    for name, (block, directory) in _require_artifacts().items():
        recorded = _fact(block, "host")
        stamped = _artifact_stamps(directory)["hostname"]
        if not stamped:
            assert name in HOST_NOT_MACHINE_STAMPED, (
                f"{name}: nothing in {directory.name} stamps a hostname, so the "
                f"recorded {recorded!r} came from prose. Declare it in "
                f"HOST_NOT_MACHINE_STAMPED with what would close it.")
            continue
        assert name not in HOST_NOT_MACHINE_STAMPED, (
            f"{name} is declared as prose-only and {directory.name} stamps "
            f"{sorted(stamped)} — delete the entry")
        if recorded == RELEASE_HOST_PLACEHOLDER:
            # The release spelling of the stamped machine's name (see the constant):
            # it can stand for one stamped machine and no more.
            assert len(stamped) == 1, (
                f"{name}: records the release placeholder {recorded!r}, which names one "
                f"machine, and {directory.name} stamps {len(stamped)} hostnames")
            continue
        assert recorded in stamped, (
            f"{name}: records host {recorded!r}; the artifact stamps {sorted(stamped)}")


@pytest.mark.requires_resource("cuda-gate-artifact")
def test_the_recorded_policies_are_the_ones_the_artifact_stamped():
    """Including the block that records none — its artifact must record none too.

    This is what stops METADATA_DEBT becoming a parking space. A block may owe a
    fact only when the RUN lacks it; if the artifact stamps a policy and the
    record does not, that is an untranscribed fact wearing a debt's clothes.
    """
    for name, (block, directory) in _require_artifacts().items():
        recorded = _fact(block, "subnormal_policy")
        stamped = _artifact_stamps(directory)["subnormal_policy"]
        if recorded is None:
            assert (name, "subnormal_policy") in METADATA_DEBT, name
            assert not stamped, (
                f"{name} is declared as owing a subnormal policy and its artifact "
                f"stamps {sorted(stamped)} — that is a transcription gap, not a "
                f"gap in the run; transcribe it and delete the debt entry")
            continue
        recorded = [recorded] if isinstance(recorded, str) else list(recorded)
        assert stamped, f"{name}: records {recorded} and the artifact stamps none"
        assert set(recorded) == stamped, (
            f"{name}: records {sorted(recorded)}, artifact stamps {sorted(stamped)}")


@pytest.mark.requires_resource("cuda-gate-artifact")
def test_the_recorded_artifact_digest_recomputes_over_the_directory():
    """THE one the sibling tracks do not have.

    Metal requires ``artifact_sha256`` to be present and nothing recomputes it, so
    an artifact could be replaced under a weld that still reads as bound to it.
    Here the digest is recomputed from the directory, and a mismatch names the
    files that differ rather than reporting two hex strings — because the useful
    question after a mismatch is which leg moved.
    """
    for name, (block, directory) in _require_artifacts().items():
        recorded = _fact(block, "artifact_sha256")
        live, table = _manifest(directory)
        assert table, f"{name}: no verdict payloads under {directory}"
        assert live == recorded, (
            f"{name}: the artifact under {directory.name} no longer digests to the "
            f"recorded {recorded}; it digests to {live} over {len(table)} payloads. "
            f"Either a verdict file changed or the recorded digest describes a "
            f"different run — do not re-cut the digest without finding out which.")


@pytest.mark.requires_resource("cuda-gate-artifact")
def test_the_manifest_covers_the_verdicts_and_not_the_scripts_beside_them():
    """The rule has to bite on evidence and not on prose, or nobody keeps it.

    A manifest over every file would move whenever a summary script or a log was
    touched, which trains the next person to re-cut the digest on sight — and a
    digest re-cut on sight is a digest that no longer means anything. What is left
    out has to be logs, drivers and prose, never a verdict payload; the one piece
    of evidence among them, the probe's ``PROVENANCE.txt``, is covered by the
    host and timestamp clauses above, which read it directly.
    """
    excluded_somewhere = False
    for name, (_, directory) in _require_artifacts().items():
        _, table = _manifest(directory)
        assert all(key.endswith(".json") for key in table), name
        excluded = sorted(path.relative_to(directory).as_posix()
                          for path in directory.rglob("*")
                          if path.is_file() and path.suffix != ".json")
        excluded_somewhere |= bool(excluded)
        unexpected = [path for path in excluded
                      if pathlib.Path(path).suffix not in
                      {".log", ".md", ".txt", ".sh", ".py"}]
        assert not unexpected, (
            f"{name}: {unexpected} are excluded from the manifest and are not "
            f"logs, drivers or prose — if one of them is evidence the rule is "
            f"wrong, not the file")
    assert excluded_somewhere, (
        "no artifact directory holds a non-JSON file, so the JSON-only rule "
        "excludes nothing and this clause is asserting a distinction that has "
        "stopped existing")
