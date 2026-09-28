"""The re-gate writers refuse what they say they refuse — and their output stays true.

WHY THIS FILE EXISTS. On 2026-08-30 two records were re-cut from fresh device
runs by three tools under ``parity/meep_gpu/``: ``record_cuda_regate.py`` (writes
a re-gate campaign into ``cuda_kernels/certification.json``),
``rebind_cuda_welds.py --campaign`` (binds ``cuda_kernels/fingerprints.json`` to
that campaign) and ``recut_composition_records.py`` (re-cuts a Triton
composition weld, including — newly — its curated measurement). Each carries
refusals that are the whole reason it is safe. NONE of those refusals had ever
been watched fire.

That is the shape this project keeps finding: ``rebind_cuda_welds.py``'s sole
reader was itself until ``test_cuda_weld_contract.py`` was written, and a check
nobody has seen FAIL is a check that cannot fail. A writer is worse than a
record in this respect, because its refusals are what stand between a fresh
measurement and a record that quietly stops describing one.

WHAT IS ARMED HERE, and each is one half of a failure that has already happened:

* THE RECORD MUST AGREE WITH ITS OWN DIRECTORIES. The first attempt to clear the
  CUDA drift installed fresh legs INTO each block's dated directory: the ledger
  reached zero drift and four certification tests went red, because every block
  carries a manifest digest over its own tree. Both CUDA writers now recompute
  every present block's digest before writing a byte. Driven here against a
  planted mismatch AND against the honest pairing.
* A CAMPAIGN THE RECORD DOES NOT NAME MAY NOT BE BOUND. Otherwise a weld points
  at evidence the narrative record has no entry for.
* THE JOIN IS THE RECORD'S OWN CLAIM. Which certification block a campaign
  subdirectory re-gated comes out of the block's declared artifact directory,
  never out of string surgery — the hazard ``rebind_triton_welds.CAMPAIGN_DIRS``
  is typed out to avoid. A subdirectory no block claims is a REFUSAL, not a row
  that gets dropped.
* A CURATED MEASUREMENT MAY NOT BE SILENTLY OVERWRITTEN, and once re-cut it must
  agree with the run it names. The second half is the one with no home before
  today: 38 Triton entries state ``n/m`` counters beside a ``records`` line that
  resolves to a payload in this checkout, and nothing compared them at the merge
  bar. ``dispersive_fused_pair_gate`` carried ``36/36`` against a run measuring
  ``48/48`` for most of a day.

Stdlib and pytest. Loads the writers by path the way
``test_gate_provenance.py`` does, because they live under ``parity/`` and are
not importable as a package.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import pathlib
import re
import sys

import pytest

PACKAGE = pathlib.Path(__file__).resolve().parent
REPO = PACKAGE.parent
PARITY = REPO / "parity" / "meep_gpu"
CUDA_RECORD = PACKAGE / "cuda_kernels" / "certification.json"
CUDA_LEDGER = PACKAGE / "cuda_kernels" / "fingerprints.json"
TRITON_LEDGER = PACKAGE / "triton_kernels" / "fingerprints.json"

#: A ``records`` line's pointer at the run it describes. Directory or payload:
#: the Triton record uses both spellings and neither is wrong.
RESULTS_POINTER = re.compile(r"apps/api/(parity/meep_gpu/results/[^\s,;]+)")

#: Non-vacuity floors, measured 2026-08-30. THE DENOMINATOR IS THE CHECK: a
#: comparison that quietly stops finding anything to compare passes an empty loop
#: while reading as strict, which is the defect this whole campaign is about.
CUDA_WELDS_BOUND_TO_A_RECORDED_RUN = 20
#: 38 -> 40 ON 2026-08-31: ``dispersive_composition_gate`` gained a resolving
#: ``records`` line when it was re-cut from its fresh GPU-host run, and left
#: :data:`NO_COMPARABLE_ARTIFACT` in the same change. Raised to the measured value so
#: the coverage this round gained cannot quietly drain back out.
#:
#: 40 -> 42 LATER THE SAME DAY: the two ADE-chain entries left
#: :data:`NO_COMPARABLE_ARTIFACT` in the same change as this raise. Their reasons
#: stopped being true the moment ``rebind_triton_welds.py`` re-ran their gates on
#: the GPU host and bound the ``records`` lines to the fresh artifacts
#: (``triton_regate_2026-08-31_retry2/complex_fused_ade_chain/`` and
#: ``triton_regate_2026-08-31_final3/fused_ade_chain/``, both released); both now
#: resolve and both compare with zero counter disagreements, so the honest edit is
#: the one this test's own failure message names — delete the entries — plus this
#: raise to the measured value so the gained coverage cannot quietly drain out.
#:
#: 42 -> 43 ON 2026-09-01: ``bit_identity_gate`` gained a ``records`` line for the
#: first time, written by ``rebind_triton_welds.py --bind-bit-identity`` from the
#: released identity-leg re-run
#: (``results/triton_bit_identity_2026-09-01/gate.json``) that cleared the
#: ledger's last two drifted pins. Its curated legs carry no field name any run
#: summary shares — the artifact writes ``verdict``, not ``summary`` — so it adds
#: comparability without adding compared counters, and the counter floor below
#: deliberately does not move with this raise.
TRITON_ENTRIES_WITH_A_COMPARABLE_ARTIFACT = 43

#: How many curated ``n/m`` counters actually get compared, across those 40.
#: STATED RATHER THAN IMPLIED, because it is far smaller than "every curated
#: claim" and a reader of the entry count above would assume otherwise: the join
#: is the FIELD NAME, and most curated fields share no name with any run summary,
#: so they are compared by nothing. That is a fact about what these gates report,
#: not a filter — but an unstated small denominator is how a check comes to read
#: stronger than it is.
#:
#: 13 -> 17 ON 2026-08-31, and the four are all one entry's:
#: ``dispersive_composition_gate.real_engine_route`` contributed ZERO before and now
#: contributes ``covered_curl_identical``, ``covered_whole_step_identical``,
#: ``builder_matches_its_predicate`` and ``refusals_returned_none``. TWO of those four
#: were previously uncomparable for a second reason worth stating, because it is the
#: hazard this constant exists to name: the probe's summary spelled them
#: ``covered_identical`` and ``whole_step_identical`` while the record carried
#: ``covered_curl_identical`` and ``covered_whole_step_identical``, so they shared a
#: name with nothing and the join found them by NOTHING. Renaming the probe's counters
#: to the record's own field names is what brought them under comparison — and the
#: first thing the comparison reported was that one of them had been stale since the
#: whole-step leg started running on eight cases rather than four.
TRITON_CURATED_COUNTERS_COMPARED = 17

#: Triton entries whose ``records`` line resolves to no payload in this checkout,
#: named with the reason. BOTH DIRECTIONS below: an entry that becomes comparable
#: must leave this list in the same change, and a new one must be argued for
#: here rather than silently reducing the denominator.
#:
#: 3 -> 2 ON 2026-08-31. ``dispersive_composition_gate`` left this list because its
#: reason stopped being true, which is the only way an entry may leave it. The reason
#: had been: "its records line names
#: results/triton_dispersive_composition_2026-08-11/, which holds engine_route.json
#: rather than a gate.json — the engine-route probe writes neither the gate spelling
#: nor a verdict." Both halves are now false. ``probe_triton_engine_route.py`` states
#: a release condition, writes ``summary.status`` and stamps through
#: ``gate_provenance``, and its 2026-08-31 the GPU host re-run
#: (``results/triton_engine_route_regate_2026-08-31/dispersive_composition/gate.json``)
#: released; ``recut_composition_records.py`` then bound the entry and moved its
#: ``records`` line to that payload. So this entry's curated counters are compared
#: against a run for the FIRST TIME, which is why the counter floor below rises with
#: this edit rather than staying put.
#: 2 -> 0 ON 2026-08-31: the two ADE-chain entries left the way the previous two
#: did — their reasons stopped being true. The misattributed records lines were
#: cleared exactly as the reason predicted ("by re-running that tool against the
#: tree the artifact is actually in"): both gates were re-run on the GPU host and
#: rebound, both lines now resolve to released payloads, and both entries compare
#: clean. The dict stays, empty, so the next entry that stops resolving has a
#: place to be argued for rather than silently shrinking the denominator.
NO_COMPARABLE_ARTIFACT = {}


def _load(name):
    """Load a ``parity/meep_gpu`` writer by path, under its own module name.

    Its OWN name, not a ``*_under_test`` alias: ``record_cuda_regate`` imports
    ``rebind_cuda_welds`` as a flat module, which is how it guarantees the two
    tools share one ``manifest_digest`` rather than growing a second one. An
    alias here would make that import miss and the shared-helper assertion below
    vacuous.
    """
    if str(PARITY) not in sys.path:
        sys.path.insert(0, str(PARITY))
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, PARITY / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _planted_root(tmp_path):
    """A tree shaped like ``the repository root``, so a block's own path spelling resolves.

    The writers build ``_API / "parity/meep_gpu/results/<name>"`` out of the
    block's ``artifacts`` string. A control that planted its directories anywhere
    else would exercise the ``not directory.is_dir()`` branch and report zero
    problems while looking like a refusal that fired.
    """
    root = tmp_path / "parity" / "meep_gpu" / "results"
    root.mkdir(parents=True, exist_ok=True)
    return root


@pytest.fixture(scope="module")
def rebind():
    return _load("rebind_cuda_welds")


@pytest.fixture(scope="module")
def transcribe():
    return _load("record_cuda_regate")


@pytest.fixture(scope="module")
def recut():
    return _load("recut_composition_records")


def _plant_campaign(root, families):
    """A campaign tree: ``{subdir: {leg relative path: payload}}``."""
    for subdirectory, legs in families.items():
        for leg, payload in legs.items():
            path = root / subdirectory / leg
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload), encoding="utf-8")
    return root


def _released_leg(**extra):
    payload = {"canonical_verdict": {"released": True, "reasons": []},
               "started_utc": "2026-08-30T15:00:00Z", "host": "a-host",
               "imported_source_sha256": {}}
    payload.update(extra)
    return payload


# ---------------------------------------------------------------------------
# THE RECORD MUST AGREE WITH ITS OWN DIRECTORIES
# ---------------------------------------------------------------------------


def test_the_cuda_record_agrees_with_every_directory_it_names(rebind):
    """The live check, on this checkout, before anything else is trusted.

    Every ``certification.json`` block whose artifact directory is on this
    machine must still digest to the ``artifact_sha256`` it records. This is the
    same rule ``test_certification_metadata.py`` enforces from the record's side;
    it is asserted again HERE because it is the precondition both CUDA writers
    refuse on, and a precondition that only one file checks is a precondition that
    can be true in one place and false in the other.
    """
    record = json.loads(CUDA_RECORD.read_text(encoding="utf-8"))
    assert not rebind.record_disagrees_with_its_own_directories(record)


def test_the_agreement_guard_reports_a_directory_that_has_moved(rebind, tmp_path):
    """THE ARMED CONTROL. Without it, the assertion above passing is consistent
    with the guard returning an empty list unconditionally — and an empty list is
    exactly what it returns on a machine holding no artifacts, so the live check
    alone cannot tell "agrees" from "found nothing".

    Both directions, over a real directory: the honest digest is silent, one
    changed byte in one payload is reported by block name.
    """
    directory = _planted_root(tmp_path) / "a_campaign"
    directory.mkdir()
    (directory / "gate.json").write_text('{"canonical_verdict": {"released": true}}',
                                         encoding="utf-8")
    honest = rebind.manifest_digest(directory)
    block = {"artifacts": f"parity/meep_gpu/results/{directory.name} (gitignored)",
             "artifact_sha256": honest}

    # The block's path is resolved against the real repo root, so point the
    # resolver at the planted tree the way the tool does: by patching the root.
    original = rebind._API
    try:
        rebind._API = tmp_path
        assert rebind.record_disagrees_with_its_own_directories({"a_block": block}) == []
        moved = dict(block, artifact_sha256="f" * 64)
        reported = rebind.record_disagrees_with_its_own_directories({"a_block": moved})
        assert len(reported) == 1 and reported[0].startswith("a_block: "), reported
        assert honest[:12] in reported[0] and "ffffffffffff" in reported[0], reported
        # AND a directory that is not on this machine is NOT evidence of anything.
        absent = {"artifacts": "parity/meep_gpu/results/nowhere_at_all",
                  "artifact_sha256": "f" * 64}
        assert rebind.record_disagrees_with_its_own_directories({"b": absent}) == []
    finally:
        rebind._API = original


# ---------------------------------------------------------------------------
# A CAMPAIGN THE RECORD DOES NOT NAME MAY NOT BE BOUND
# ---------------------------------------------------------------------------


def test_a_campaign_the_record_does_not_name_is_not_bindable(rebind, tmp_path):
    """``campaign_block`` answers only for a tree the record actually records.

    Three legs, because the refusal has three ways to be wrong: a directory no
    block names, a block that names it but is not a re-gate record (a family
    block's own dated directory is not a campaign), and the honest case.
    """
    campaign = _planted_root(tmp_path) / "a_regate"
    campaign.mkdir()
    original = rebind._API
    try:
        rebind._API = tmp_path
        assert rebind.campaign_block({}, campaign) is None
        family = {"f": {"artifacts": "parity/meep_gpu/results/a_regate",
                        "artifact_sha256": "a" * 64}}
        assert rebind.campaign_block(family, campaign) is None, (
            "a block naming the tree but carrying no re_gated table was accepted "
            "as a campaign — a family's own directory is not a re-gate")
        recorded = {"c": dict(family["f"], **{rebind.RE_GATE_KEY: []})}
        found = rebind.campaign_block(recorded, campaign)
        assert found is not None and found[0] == "c"
    finally:
        rebind._API = original


#: Ledger entries that are NOT welds and so have no certification campaign to join.
#: A weld names the RUN its digests came from and ``certification.json`` records that
#: run; the dispatch record is cut from a five-leg ROUTE campaign by
#: ``recut_driver_dispatch_record.py``, which the certification record has never
#: described and should not — it describes device gates, one verdict each. Named here
#: with the reason, so a future entry cannot join the exemption by accident, and
#: matching the exemption the sibling contracts already carry for the same block
#: (``test_triton_weld_contract`` NOT_A_WELD, ``test_cuda_weld_contract`` NOT_A_WELD).
NOT_A_WELD = {
    "driver_dispatch":
        "the dispatch WIRING record, cut from a route campaign rather than from a "
        "device gate. Its SOURCE DIGESTS are checked on the same terms as a weld's "
        "-- recut_driver_dispatch_record refuses unless every bound file carries a "
        "digest a LEG recorded -- but it names no certification campaign because "
        "none describes it.",
}


def test_every_cuda_weld_names_a_run_the_certification_record_records(rebind):
    """THE JOIN BETWEEN THE TWO FILES, which nothing checked until today.

    A weld's ``records`` line is the only handle a later reader gets on the run
    its digests came from — ``results/`` is gitignored, so the line is the
    evidence, not a convenience. Before the re-gate every line named the block's
    own dated directory and the join was true by construction. It is not any
    more: twenty welds now name a subdirectory of a RE-GATE campaign, and the
    thing that makes that honest rather than a dangling pointer is that
    ``certification.json`` records the campaign AND its ``re_gated`` table says
    which block each subdirectory re-gated.

    So both halves are required here: the directory a weld names must lie inside
    a tree the record names, and the campaign block's own row for that entry must
    name the SAME subdirectory. A weld pointing into a recorded campaign at a
    family the campaign says it did not re-gate would satisfy the first and fail
    the second.
    """
    record = json.loads(CUDA_RECORD.read_text(encoding="utf-8"))
    ledger = json.loads(CUDA_LEDGER.read_text(encoding="utf-8"))
    campaigns = {name: {row["block"]: row["directory"]
                        for row in block[rebind.RE_GATE_KEY]}
                 for name, block in record.items()
                 if isinstance(block, dict) and rebind.RE_GATE_KEY in block}
    own = {name: rebind.artifact_directory(block).name
           for name, block in record.items()
           if isinstance(block, dict) and rebind.artifact_directory(block) is not None}

    bound, wrong = 0, {}
    for name, entry in sorted(ledger.items()):
        if not isinstance(entry, dict) or name in NOT_A_WELD:
            continue
        line = str(entry.get("records", ""))
        found = RESULTS_POINTER.search(line)
        if found is None:
            wrong[name] = f"records names no results path: {line!r}"
            continue
        parts = pathlib.PurePosixPath(found.group(1)).parts[3:]   # after results/
        if not parts:
            wrong[name] = f"records names the results root: {line!r}"
            continue
        head = parts[0]
        if head == own.get(name):
            bound += 1
            continue
        table = campaigns.get(head)
        if table is None:
            wrong[name] = (f"records names {head}/, which is neither this block's "
                           f"own artifact directory nor a campaign certification."
                           f"json records")
            continue
        expected = table.get(name)
        if expected is None:
            wrong[name] = (f"records points into campaign {head}/, whose "
                           f"{rebind.RE_GATE_KEY} table does not mention this entry")
            continue
        if len(parts) < 2 or parts[1] != expected:
            wrong[name] = (f"records points at {head}/{parts[1:2]}, and the campaign "
                           f"says this entry was re-gated in {expected}/")
            continue
        bound += 1
    assert not wrong, wrong
    assert bound >= CUDA_WELDS_BOUND_TO_A_RECORDED_RUN, (
        f"only {bound} CUDA welds resolve to a run the record records, was "
        f"{CUDA_WELDS_BOUND_TO_A_RECORDED_RUN} — the join SHRANK")


# ---------------------------------------------------------------------------
# THE JOIN IS THE RECORD'S OWN CLAIM, AND A SUBDIRECTORY NOBODY CLAIMS REFUSES
# ---------------------------------------------------------------------------


def test_the_regate_survey_refuses_what_it_cannot_attribute(transcribe, tmp_path):
    """Five refusals and two acceptances, each planted.

    The dangerous one is the first: a re-gate tree holding a family the record
    does not describe. Dropping that row would produce a campaign block that
    reads complete and silently covers less than the tree it digests — the
    defect class this whole round exists to remove, arriving through a writer
    instead of a reader.
    """
    campaign = _plant_campaign(_planted_root(tmp_path) / "a_regate", {
        "known_family": {"keep/gate.json": _released_leg(),
                         "flush/gate.json": _released_leg()},
    })
    record = {"a_block": {"artifacts": "parity/meep_gpu/results/known_family",
                          "artifact_sha256": "a" * 64}}
    original = transcribe._API
    try:
        transcribe._API = tmp_path
        rows, refusals = transcribe.survey(campaign, record)
        assert not refusals and [row["block"] for row in rows] == ["a_block"], (
            rows, refusals)
        assert rows[0]["legs"] == ["flush/gate.json", "keep/gate.json"]

        # 1. A SUBDIRECTORY NO BLOCK CLAIMS.
        _plant_campaign(campaign, {"a_stranger": {"keep/gate.json": _released_leg(),
                                                  "flush/gate.json": _released_leg()}})
        rows, refusals = transcribe.survey(campaign, record)
        assert len(refusals) == 1 and "re-gates nothing this record names" in refusals[0]
        assert [row["block"] for row in rows] == ["a_block"]

        # 2. TWO BLOCKS CLAIMING ONE DIRECTORY AND NEITHER SCOPING: still refused,
        #    with the reason the 2026-09-11 widening made precise. Until then the
        #    message was "the join is ambiguous" and the shape was refused outright;
        #    now the refusal falls on the part that is actually a guess -- WHICH LEGS
        #    belong to which block -- because a shared directory is legitimate when
        #    each claimant names its own gate_family (case 2b).
        ambiguous = dict(record, a_twin={"artifacts":
                                         "parity/meep_gpu/results/known_family",
                                         "artifact_sha256": "b" * 64})
        _, refusals = transcribe.survey(campaign, ambiguous)
        assert any("name no gate_family" in why for why in refusals), refusals
        assert any("which legs belong to which block is a guess" in why
                   for why in refusals), refusals

        # 2b. TWO BLOCKS CLAIMING ONE DIRECTORY, EACH SCOPED: accepted, one row per
        #     block, and each row carries ONLY its own family's legs. This is the
        #     shape a multi-family gate script writes (`<family>_<policy>` legs), and
        #     refusing it is what made the complex fused pairs unrecordable.
        shared = _plant_campaign(_planted_root(tmp_path) / "shared", {
            "known_family": {"alpha_keep/gate.json": _released_leg(),
                             "alpha_flush/gate.json": _released_leg(),
                             "beta_keep/gate.json": _released_leg(),
                             "beta_flush/gate.json": _released_leg()}})
        scoped = {
            "alpha_block": {"artifacts": "parity/meep_gpu/results/known_family",
                            "artifact_sha256": "a" * 64, "gate_family": "alpha"},
            "beta_block": {"artifacts": "parity/meep_gpu/results/known_family",
                           "artifact_sha256": "b" * 64, "gate_family": "beta"},
        }
        rows, refusals = transcribe.survey(shared, scoped)
        assert not refusals, refusals
        assert sorted(row["block"] for row in rows) == ["alpha_block", "beta_block"]
        by_block = {row["block"]: row for row in rows}
        assert by_block["alpha_block"]["legs"] == ["alpha_flush/gate.json",
                                                   "alpha_keep/gate.json"]
        assert by_block["beta_block"]["legs"] == ["beta_flush/gate.json",
                                                  "beta_keep/gate.json"]
        assert by_block["alpha_block"]["gate_family"] == "alpha"

        # 2c. TWO BLOCKS CLAIMING ONE DIRECTORY UNDER THE SAME gate_family: refused.
        #     Checking that every claimant NAMES a family, without checking the names
        #     are DISTINCT, would let two blocks stand on one measurement and would
        #     report twice the legs the tree holds.
        collided = {
            "alpha_block": {"artifacts": "parity/meep_gpu/results/known_family",
                            "artifact_sha256": "a" * 64, "gate_family": "alpha"},
            "alpha_twin": {"artifacts": "parity/meep_gpu/results/known_family",
                           "artifact_sha256": "b" * 64, "gate_family": "alpha"},
        }
        rows, refusals = transcribe.survey(shared, collided)
        assert any("the same gate_family" in why for why in refusals), refusals
        assert not rows, rows

        # 3. A LEG THAT DID NOT RELEASE.
        (campaign / "known_family" / "keep" / "gate.json").write_text(
            json.dumps({"canonical_verdict": {"released": False,
                                              "reasons": ["a divergence"]}}),
            encoding="utf-8")
        rows, refusals = transcribe.survey(campaign, record)
        assert any("is not released" in why for why in refusals), refusals
        assert not rows

        # 4. A DIRECTORY WITH NO CANONICAL POLICY LEG AT ALL.
        bare = _plant_campaign(_planted_root(tmp_path) / "bare", {"known_family": {}})
        (bare / "known_family").mkdir(parents=True, exist_ok=True)
        (bare / "known_family" / "notes.json").write_text("{}", encoding="utf-8")
        _, refusals = transcribe.survey(bare, record)
        assert any("holds no canonical policy leg" in why for why in refusals), refusals
    finally:
        transcribe._API = original


def test_the_live_regate_campaign_surveys_clean_or_is_not_on_this_machine(transcribe):
    """The real tree, when it is here: every family attributed, nothing refused."""
    record = json.loads(CUDA_RECORD.read_text(encoding="utf-8"))
    campaigns = [name for name, block in record.items()
                 if isinstance(block, dict) and transcribe.RE_GATE_KEY in block]
    assert campaigns, "certification.json records no re-gate campaign at all"
    for name in campaigns:
        directory = transcribe.artifact_directory(record[name])
        if directory is None or not directory.is_dir():
            continue
        rows, refusals = transcribe.survey(directory, record)
        assert not refusals, (name, refusals)
        assert {row["block"] for row in rows} == \
            {row["block"] for row in record[name][transcribe.RE_GATE_KEY]}, name
        assert sum(len(row["legs"]) for row in rows) == \
            record[name]["canonical_policy_legs"], name


# ---------------------------------------------------------------------------
# THE CURATED MEASUREMENT
# ---------------------------------------------------------------------------


def _comparable(recut):
    """``{entry: [payloads]}`` for every Triton entry whose run is on this machine."""
    ledger = json.loads(TRITON_LEDGER.read_text(encoding="utf-8"))
    out = {}
    for name, entry in sorted(ledger.items()):
        if not isinstance(entry, dict) or not isinstance(entry.get("records"), str):
            continue
        payloads = []
        for pointer in RESULTS_POINTER.findall(entry["records"]):
            path = REPO / pointer.rstrip("/.,")
            if path.is_file() and path.suffix == ".json":
                payloads.append(path)
            elif path.is_dir():
                for candidate in ("gate.json", "gate_keep.json"):
                    if (path / candidate).is_file():
                        payloads.append(path / candidate)
                        break
        if payloads:
            out[name] = (entry, payloads)
    return ledger, out


def test_every_curated_triton_claim_still_describes_the_run_it_names(recut):
    """THE STALE-CLAIM CHECK, at the merge bar for the first time.

    A weld's curated block is the measurement a reader takes away — ``product``,
    ``real_engine_route``, ``single_launch``, ``composition``. Its digests are
    compared to the tree by ``test_triton_weld_contract.py``; its NUMBERS were
    compared to the run by nothing except an operator choosing to run
    ``recut_composition_records.py``. That is how ``dispersive_fused_pair_gate``
    stood at ``complete_steps_exact 36/36`` while the run its own records line
    names measured 48/48 over four cases rather than three.

    COUNTERS ONLY, for the reason the tool states: a shared key name is not a
    shared meaning, and a check that reports prose rewording as a defect is a
    check someone deletes.

    Measured 2026-08-30: 38 entries comparable, 0 disagreeing.
    """
    ledger, comparable = _comparable(recut)
    wrong, compared = {}, 0
    for name, (entry, payloads) in comparable.items():
        for path in payloads:
            payload = json.loads(path.read_text(encoding="utf-8"))
            summary = payload.get("summary") or {}
            for trail, field, _ in recut._counters(entry):
                if recut._is_historical(trail):
                    continue
                measured = summary.get(field)
                if isinstance(measured, str) and recut.COUNTER.match(measured):
                    compared += 1
            for block, field, recorded, measured in recut.claim_disagreements(
                    entry, payload):
                wrong[f"{name}.{block}[{field}]"] = (
                    f"record says {recorded!r}, {path.name} measured {measured!r}")
    assert compared >= TRITON_CURATED_COUNTERS_COMPARED, (
        f"only {compared} curated counters were actually compared against a run, "
        f"was {TRITON_CURATED_COUNTERS_COMPARED} — the entry count above can hold "
        f"while this falls to zero, and then this test enforces nothing")
    assert not wrong, (
        f"{len(wrong)} curated counters no longer describe the run their records "
        f"line names: {wrong} — re-run the gate, then re-cut the claim with "
        f"parity/meep_gpu/recut_composition_records.py after declaring the block "
        f"in CLAIM_RECUTS; do not hand-edit the number")
    assert len(comparable) >= TRITON_ENTRIES_WITH_A_COMPARABLE_ARTIFACT, (
        f"only {len(comparable)} entries have a run to compare against, was "
        f"{TRITON_ENTRIES_WITH_A_COMPARABLE_ARTIFACT} — an entry does not leave "
        f"this check by having its records line stop resolving")

    # BOTH DIRECTIONS on the declared gap: an entry that becomes comparable must
    # leave NO_COMPARABLE_ARTIFACT in the same change.
    resolved = sorted(set(NO_COMPARABLE_ARTIFACT) & set(comparable))
    assert not resolved, (
        f"{resolved} are declared to have no comparable artifact and now do — "
        f"delete the entry rather than leaving a reason that has stopped being "
        f"true")
    dict_entries = {name for name, entry in ledger.items() if isinstance(entry, dict)}
    stale = sorted(set(NO_COMPARABLE_ARTIFACT) - dict_entries)
    assert not stale, f"declared gaps naming entries the record no longer has: {stale}"
    assert all(reason.strip() for reason in NO_COMPARABLE_ARTIFACT.values())


def test_the_claim_check_reports_a_planted_disagreement(recut):
    """ARMED, both directions, and on the SHAPE that actually bit.

    ``36/36`` against a run measuring ``48/48`` must be reported; the same value
    with the record's extra prose after it must not; and a prose-only field whose
    wording differs must not, because that is the false positive the counter
    narrowing exists to avoid.
    """
    payload = {"summary": {"complete_steps_exact": "48/48",
                           "oracles_per_step": "array path + separate products"}}
    stale = {"product": {"complete_steps_exact": "36/36"}}
    found = recut.claim_disagreements(stale, payload)
    assert found == [("product", "complete_steps_exact", "36/36", "48/48")], found

    honest = {"product": {"complete_steps_exact": "48/48 against both oracles"}}
    assert recut.claim_disagreements(honest, payload) == []

    prose = {"product": {"oracles_per_step": "CuPy array path and separate plans"}}
    assert recut.claim_disagreements(prose, payload) == [], (
        "a prose field whose wording differs was reported as a measurement "
        "disagreement — the check now cries wolf and will be deleted")


def test_the_curated_claim_walk_is_structural_and_excludes_history(recut):
    """The enumerator is a WALK, and the one thing it drops is named.

    THE COINCIDENCE THIS REMOVES. ``CLAIM_BLOCKS`` names four block spellings and
    the Triton record carries thirty-three distinct key names holding an ``n/m``
    counter, several at an entry's own top level. Both enumerators happen to find
    the same 13 comparable counters today, which is precisely the shape that goes
    wrong silently: a denominator right by accident stops being right without
    saying so. So the walk recurses, and this pins that it reaches a counter a
    block-name list cannot — nested under an unanticipated key, and inside a list.

    AND THE ONE EXCLUSION IS ARMED. A counter inside a superseded run or a recert
    snapshot describes a run that is over; requiring it to match today's summary
    would demand that the past change, and the only way to clear that red would be
    to delete the evidence. It fires ZERO times on the record today — no
    historical counter shares a field name with any summary — so without a planted
    control there would be no way to tell the guard from a no-op.
    """
    deep = {"a_key_nobody_wrote": {"nested": [{"deeper": {"cases_exact": "3/4"}}]},
            "complete_steps_exact": "7/8"}
    found = {".".join(str(p) for p in trail): value
             for trail, _, value in recut._counters(deep)}
    assert found == {
        "a_key_nobody_wrote.nested.0.deeper.cases_exact": "3/4",
        "complete_steps_exact": "7/8"}, found

    payload = {"summary": {"cases_exact": "4/4", "complete_steps_exact": "8/8"}}
    reported = recut.claim_disagreements(deep, payload)
    assert sorted(f for _, f, _, _ in reported) == ["cases_exact",
                                                    "complete_steps_exact"], reported
    assert ("a_key_nobody_wrote.nested.0.deeper", "cases_exact", "3/4", "4/4") \
        in reported, reported

    # HISTORY IS DROPPED, and only history.
    for container in ("recert_2026_08_13", "superseded_runs", "kernels_at_recert",
                      "a_recut_log", "source_drift_since_recert"):
        historical = {container: {"cases_exact": "3/4"}}
        assert recut.claim_disagreements(historical, payload) == [], container
        assert recut._is_historical((container, "cases_exact")), container
    assert not recut._is_historical(("product", "cases_exact"))
    assert recut.claim_disagreements({"product": {"cases_exact": "3/4"}},
                                     payload), "a LIVE block was dropped as history"

    # The declared spellings must still describe the record: a name nothing
    # matches is an exclusion attached to nothing.
    ledger = json.loads(TRITON_LEDGER.read_text(encoding="utf-8"))
    text = json.dumps(ledger)
    stale = [name for name in recut.HISTORICAL_CONTAINERS if name not in text]
    assert not stale, (
        f"HISTORICAL_CONTAINERS names {stale}, which appear nowhere in the record "
        f"— delete the spelling rather than leaving a rule nothing exercises")


def test_the_claim_recut_is_mechanical_and_cross_checked(recut):
    """The re-cut takes its numbers from the artifact, and refuses when it cannot.

    Four legs. The third is the load-bearing one: a per-case derivation that did
    not agree with the run's own total would be arithmetic dressed as evidence,
    so it refuses rather than writing either number.
    """
    payload = {
        "summary": {"complete_steps_exact": "48/48", "cases_exact": "4/4",
                    "in_seam_deposit_cases": "1/1 (two_pole_electric_source)"},
        "cases": [{"case": name,
                   "per_step": [{"a": {"bit_identical": True}}] * 12}
                  for name in ("one_pole_periodic", "two_pole_anisotropic",
                               "three_pole_metallic_x", "two_pole_electric_source")],
    }
    entry = {"product": {"complete_steps_exact": "36/36",
                         "one_pole_periodic": "12/12 complete steps exact",
                         "two_pole_anisotropic": "12/12 with component-specific poles",
                         "three_pole_metallic_x": "12/12 with the wall wipe inline",
                         "state_scope": "prose that must survive"}}

    fresh, moved, why = recut.recut_claim(entry, "product", payload)
    assert why == "" and fresh is not None, why
    # 1. the shared counter follows the run.
    assert fresh["complete_steps_exact"] == "48/48"
    # 2. counters the run reports and the block lacked are ADDED.
    assert fresh["cases_exact"] == "4/4"
    assert fresh["in_seam_deposit_cases"] == "1/1 (two_pole_electric_source)"
    # 3. the case the gate gained gets a line, derived from the artifact.
    assert fresh["two_pole_electric_source"] == "12/12 complete steps exact"
    # ... and the record's prose survives, on both the per-case lines and the
    # fields the run says nothing about.
    assert fresh["two_pole_anisotropic"] == "12/12 with component-specific poles"
    assert fresh["state_scope"] == "prose that must survive"
    assert any("36/36" in note and "48/48" in note for note in moved), moved
    # THE POST-CONDITION: nothing left disagreeing.
    assert recut.claim_disagreements({"product": fresh}, payload, block="product") == []

    # 4a. THE CROSS-CHECK REFUSES when the cases and the summary disagree.
    lying = json.loads(json.dumps(payload))
    lying["summary"]["complete_steps_exact"] = "60/60"
    fresh, _, why = recut.recut_claim(entry, "product", lying)
    assert fresh is None and "cases sum to 48/48" in why, why

    # 4b. AND A CLAIM WITH NO LEADING COUNTER CANNOT BE SPLIT, so it refuses
    #     rather than guessing which half of the sentence is the measurement.
    unsplittable = {"product": {"complete_steps_exact": "all of them"}}
    fresh, _, why = recut.recut_claim(unsplittable, "product", payload)
    assert fresh is None and "carries no leading counter" in why, why


def test_every_declared_claim_recut_names_a_real_block_and_carries_a_reason(recut):
    """A DECLARED RE-CUT IS AN EXEMPTION, and it decays like every other one.

    ``CLAIM_RECUTS`` is the one place a curated measurement is allowed to be
    overwritten from a run. Both directions: every declaration must name an entry
    and a block the ledger actually has, and must carry a reason. A declaration
    whose block has gone is a standing licence attached to nothing, which is
    where the next silent overwrite would live.
    """
    ledger = json.loads(TRITON_LEDGER.read_text(encoding="utf-8"))
    assert recut.CLAIM_RECUTS, (
        "no claim re-cut is declared — if the table is genuinely empty, delete "
        "this test with it rather than leaving a check over nothing")
    for (key, block), reason in sorted(recut.CLAIM_RECUTS.items()):
        entry = ledger.get(key)
        assert isinstance(entry, dict), f"CLAIM_RECUTS names absent entry {key!r}"
        assert isinstance(entry.get(block), dict), (
            f"CLAIM_RECUTS names {key}.{block}, which the record does not carry "
            f"as a block")
        assert block in recut.CLAIM_BLOCKS, (
            f"{block!r} is declared re-cuttable and is not in CLAIM_BLOCKS. "
            f"Reading is structural, so it WOULD be compared — but overwriting a "
            f"measurement is the narrower act and stays bounded by that list")
        assert reason.strip(), f"{key}.{block} is declared with no reason"


def test_the_frozen_key_rule_does_not_leak_past_the_declaration(recut):
    """A declared block moves; every other key on the entry stays frozen.

    The write step widens its immovable set by the blocks THIS RUN re-cut, and
    only those. Driven here as arithmetic on the same expression the tool uses,
    because the alternative is trusting that ``MUTABLE | {...}`` was spelled with
    the right operand — and a licence that leaked would let any key ride out on a
    re-cut.
    """
    key = "dispersive_fused_pair_gate"
    fresh = {"source_sha256": {}, "product": {}}
    may_move = recut.MUTABLE | {block for block in fresh
                                if (key, block) in recut.CLAIM_RECUTS}
    assert "product" in may_move
    assert "composition" not in may_move and "purpose" not in may_move
    # A block nobody declared does not become movable by appearing in the update.
    undeclared = recut.MUTABLE | {block for block in {"real_engine_route": {}}
                                  if (key, block) in recut.CLAIM_RECUTS}
    assert "real_engine_route" not in undeclared


def test_the_manifest_rule_is_one_rule_across_both_cuda_writers(rebind, transcribe):
    """Two tools, one digest, or a directory gets two answers and no way to choose.

    ``record_cuda_regate.py`` writes ``artifact_sha256`` and
    ``rebind_cuda_welds.py`` refuses on it. They must compute it the same way,
    and the way ``certification.json``'s own ``_artifact_sha256_rule`` states —
    which is why the transcriber imports the function rather than restating it.
    Asserted as identity, so a second implementation cannot appear unnoticed.
    """
    assert transcribe.manifest_digest is rebind.manifest_digest
    assert transcribe.artifact_directory is rebind.artifact_directory
    assert transcribe.canonical_legs is rebind.canonical_legs
    assert transcribe.read_verdict is rebind.read_verdict
    rule = json.loads(CUDA_RECORD.read_text(encoding="utf-8"))["_artifact_sha256_rule"]
    for clause in ("*.json", "sort by", "relative", "newline", "sha256"):
        assert clause in rule


def test_the_regate_block_digests_the_tree_it_names(rebind):
    """Recomputed here, not only inside the writer that produced it.

    ``test_certification_metadata.py`` already recomputes every device block's
    artifact digest, and this is deliberately the same comparison from the other
    side: the campaign block is the ONE thing standing between twenty welds and
    an unrecorded run, so it is checked by the reader that depends on it as well
    as by the record's own suite.
    """
    record = json.loads(CUDA_RECORD.read_text(encoding="utf-8"))
    checked = 0
    for name, block in sorted(record.items()):
        if not isinstance(block, dict) or rebind.RE_GATE_KEY not in block:
            continue
        directory = rebind.artifact_directory(block)
        assert directory is not None, f"{name} records no artifact directory"
        if not directory.is_dir():
            continue
        assert rebind.manifest_digest(directory) == block["artifact_sha256"], name
        # THE CLAIM IS ABOUT CANONICAL POLICY LEGS, so it is counted with the
        # same function the writers use — not as "every .json under here". The
        # two happen to be equal on this campaign, and a control that leant on
        # that would start failing the day a summary payload lands beside the
        # legs, which is a fact about the directory rather than about the claim.
        legs = rebind.canonical_legs(directory)
        assert len(legs) == block["canonical_policy_legs"], (
            f"{name}: {len(legs)} canonical policy legs under {directory.name}/ "
            f"and the block claims {block['canonical_policy_legs']}")
        assert len(legs) == sum(len(row["legs"])
                                for row in block[rebind.RE_GATE_KEY]), (
            f"{name}: the campaign total and its per-family rows disagree")
        checked += 1
    if not checked:
        pytest.skip("no re-gate campaign directory is on this host")


def test_a_weld_and_its_run_agree_on_which_bytes_were_imported(rebind):
    """The digests in the ledger are the ones the run's own payload recorded.

    Not the same claim as ``test_the_cuda_welds_are_bound_to_the_live_sources``,
    which compares the ledger to the TREE. This compares the ledger to the RUN:
    a rebind that took a digest from the checkout instead of from the artifact
    would pass that test and fail this one, and the two together are what make
    "these bytes ran" mean something rather than "these bytes are here now".
    """
    ledger = json.loads(CUDA_LEDGER.read_text(encoding="utf-8"))
    compared, wrong = 0, {}
    for name, entry in sorted(ledger.items()):
        if (not isinstance(entry, dict) or name in NOT_A_WELD
                or not isinstance(entry.get("records"), str)):
            continue
        found = RESULTS_POINTER.search(entry["records"])
        if found is None:
            continue
        directory = REPO / found.group(1).rstrip("/")
        if not directory.is_dir():
            continue
        legs = entry.get("legs") or []
        if not all((directory / leg).is_file() for leg in legs):
            continue
        agreed = rebind.agreed_imports(rebind.leg_payloads(directory, list(legs)))
        for path, digest in sorted((entry.get("source_sha256") or {}).items()):
            compared += 1
            if agreed.get(path) != digest:
                wrong[f"{name}:{path}"] = (
                    f"the weld pins {digest[:12]} and the run it names imported "
                    f"{str(agreed.get(path))[:12]}")
    assert not wrong, wrong
    if not compared:
        pytest.skip("no CUDA gate artifact directory is on this host")
    assert compared >= 79, (
        f"only {compared} weld pins were compared against their own run, was 79 "
        f"— the comparison SHRANK")


def test_the_hashing_helpers_agree_with_the_stdlib(rebind, tmp_path):
    """The floor under everything above: one file, one digest, computed twice.

    Cheap and worth it — every refusal in this file rests on ``manifest_digest``
    being a sha256 over the payloads, and a helper that returned a constant would
    make every comparison here pass.
    """
    directory = tmp_path / "one"
    directory.mkdir()
    (directory / "gate.json").write_text('{"a": 1}', encoding="utf-8")
    line = ("gate.json " +
            hashlib.sha256(b'{"a": 1}').hexdigest()).encode("utf-8")
    assert rebind.manifest_digest(directory) == hashlib.sha256(line).hexdigest()
    (directory / "notes.txt").write_text("not a payload", encoding="utf-8")
    assert rebind.manifest_digest(directory) == hashlib.sha256(line).hexdigest(), (
        "a non-JSON file changed the manifest digest — the JSON-only rule the "
        "record states is not the rule this computes")


# ---------------------------------------------------------------------------
# --only IS A NARROWING, AND ONLY A NARROWING
# ---------------------------------------------------------------------------


def _plant_device_block(rebind, results_root, name, imports, *, released=True):
    """One certification-style device block with both policy legs on disk.

    The block names a device (``device_blocks`` clause 1), an artifact directory
    under the planted ``parity/meep_gpu/results/``, the module it certifies, and
    an ``artifact_sha256`` computed by the writer's OWN rule so the agreement guard
    that runs before the main loop has nothing to refuse. ``released=False`` turns
    ONE policy leg's verdict false, which is the shape the rules must refuse.
    """
    legs = {f"{policy}/gate.json": _released_leg(imported_source_sha256=dict(imports))
            for policy in ("keep", "flush")}
    if not released:
        legs["flush/gate.json"]["canonical_verdict"] = {
            "released": False, "reasons": ["planted: the flush leg did not release"]}
    _plant_campaign(results_root, {name: legs})
    return {
        "artifacts": f"parity/meep_gpu/results/{name}",
        "device": "planted device",
        "kernel_module": list(imports),
        "certified_kernels": ["planted_kernel"],
        "artifact_sha256": rebind.manifest_digest(results_root / name),
    }


def test_only_narrows_the_cuda_rebind_and_never_widens_it(rebind, tmp_path,
                                                          monkeypatch, capsys):
    """``--only`` skips by name, refuses an unknown name, and admits nothing the rules refuse.

    WHY THE FLAG EXISTS, and therefore what this pins: seeding ONE newly released
    block must not also seed every other absent block from its dated directory,
    several of which would seed DRIFTED -- a decision, not a side effect. So the
    flag may only ever SHRINK what a run touches. Three clauses, each of which is
    a way the flag could quietly become a widening:

    1. **A block outside ``--only`` is reported SKIPPED by name and never bound.**
       Two released blocks planted, one selected: the ledger gains exactly that one,
       and the report names the other as ``not selected by --only``.
    2. **An unknown name refuses**, before anything is written -- the same shape
       ``rebind_triton_welds.py --only`` has had since 2026-08-30 (``--only names
       [...], absent from the ledger``), and the last clause drives that tool too.
    3. **Selection is not admission.** A block whose flush leg did not release is
       selected by name and still refused by the verdict rule; the ledger does not
       move.
    """
    api = tmp_path
    results_root = _planted_root(tmp_path)
    kernel = api / "meep_gpu" / "cuda_kernels" / "planted_kernel.py"
    kernel.parent.mkdir(parents=True, exist_ok=True)
    kernel.write_text("PLANTED = 1\n", encoding="utf-8")
    imports = {"meep_gpu/cuda_kernels/planted_kernel.py":
               hashlib.sha256(kernel.read_bytes()).hexdigest()}
    record = {
        "alpha": _plant_device_block(rebind, results_root, "alpha", imports),
        "beta": _plant_device_block(rebind, results_root, "beta", imports),
        "gamma": _plant_device_block(rebind, results_root, "gamma", imports,
                                     released=False),
    }
    ledger_path = tmp_path / "planted_fingerprints.json"
    ledger_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(rebind, "_API", api)
    monkeypatch.setattr(rebind, "LEDGER", ledger_path)
    monkeypatch.setattr(rebind, "_record", lambda: json.loads(json.dumps(record)))
    assert set(rebind.device_blocks(record)) == {"alpha", "beta", "gamma"}, (
        "the planted blocks are not what device_blocks enumerates; every clause "
        "below would then be exercising the wrong refusal")

    # 2. an unknown name refuses, and refuses BEFORE the ledger is touched
    with pytest.raises(SystemExit) as refusal:
        rebind.main(["--seed", "--only", "no_such_block", "--write"])
    assert "no_such_block" in str(refusal.value) and "--only" in str(refusal.value)
    assert json.loads(ledger_path.read_text(encoding="utf-8")) == {}
    capsys.readouterr()

    # 1. the selected block is seeded; the unselected released one is skipped BY NAME
    rebind.main(["--seed", "--only", "alpha", "--write"])
    out = capsys.readouterr().out
    assert "created  alpha" in out
    assert "SKIPPED  beta: not selected by --only" in out
    assert "SKIPPED  gamma: not selected by --only" in out
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert set(ledger) == {"alpha"}, (
        f"--only alpha wrote {sorted(ledger)}; a block outside --only was bound")
    assert ledger["alpha"]["status"] == "PASS"
    assert ledger["alpha"]["source_sha256"] == imports

    # 3. selection is not admission: the rules still refuse an unreleased block
    rebind.main(["--seed", "--only", "gamma", "--write"])
    out = capsys.readouterr().out
    # Per-entry lines, not the summary's "0 created, 0 rebound" (which spells both words).
    assert "  created  " not in out and "  rebound  " not in out
    assert "  0 created, 0 rebound," in out
    assert "SKIPPED  gamma:" in out and "not selected by --only" not in out.split(
        "SKIPPED  gamma:", 1)[1].splitlines()[0], (
        "gamma was selected, so its refusal must be the verdict rule's and not the "
        "flag's")
    assert set(json.loads(ledger_path.read_text(encoding="utf-8"))) == {"alpha"}, (
        "--only admitted a block the verdict rule refuses")

    # the Triton precedent: same flag, same refusal shape, before any write
    triton = _load("rebind_triton_welds")
    campaign = _planted_root(tmp_path) / "a_triton_regate"
    campaign.mkdir()
    triton_ledger = tmp_path / "planted_triton_fingerprints.json"
    triton_ledger.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(triton, "_API", api)
    monkeypatch.setattr(triton, "LEDGER", triton_ledger)
    with pytest.raises(SystemExit) as precedent:
        triton.main(["--campaign", str(campaign), "--only", "no_such_key", "--write"])
    assert "--only" in str(precedent.value) and "no_such_key" in str(precedent.value)
    assert json.loads(triton_ledger.read_text(encoding="utf-8")) == {}
