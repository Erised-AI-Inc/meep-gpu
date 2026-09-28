"""The two 2026-08-20 admission records, and whether they still describe the files.

WHY A SEPARATE FILE. ``special_kz_curl.SPECIAL_KZ_ADMISSION`` and
``bfast_curl.BFAST_ADMISSION`` are the ONLY places either family states what a
device measured. A record is worth exactly as much as the guarantee that it was
not written about a different program, and there are two ways that guarantee goes
silently:

* THE FILE CHANGES AFTER THE GATE RAN. The gate stamps
  ``imported_source_sha256`` with the sha256 of every repo module the process
  imported (``gate_provenance``), so the artifact carries the bytes it measured.
  Comparing that against the file on disk is the whole check, and it is the one
  that caught a mis-transcribed digest on the sibling track.

  A FILE HASH IS TOO WIDE TO BE THE ONLY PIN, and it was the only one here until
  2026-08-28. It stops matching the moment anyone adds a comment, which trains the
  next reader either to re-cut a device gate over a docstring or to stop believing
  the clause. What decides the verdict is the string NVRTC compiles, and these two
  gates record exactly that: ``nvrtc_binary_report.binary_sha256_by_source`` is
  keyed by the sha256 of every source string the observer saw handed to the
  compiler, installed at the innermost seam before any policy strip. So the file
  hash is still the DEFAULT and preferred answer, and a file that has moved is
  admitted only on a MEASUREMENT -- every shipped device string is one the gate
  itself compiled, on every leg -- plus a declared edit in the module's own
  admission record saying so. A declaration alone admits nothing.
* THE RECORD SAYS SOMETHING THE ARTIFACT DOES NOT. A hand-copied case count that
  drifts from the JSON is a claim with no measurement behind it, which is worse
  than no claim.

Both are read off the artifacts rather than restated here.
"""

from __future__ import annotations

import hashlib
import json
import pathlib

import pytest

from . import bfast_curl, special_kz_curl

_API = pathlib.Path(__file__).resolve().parents[2]

FAMILIES = (
    pytest.param(special_kz_curl, special_kz_curl.SPECIAL_KZ_ADMISSION,
                 "special_kz_curl.py", "curl", 72, id="special_kz"),
    pytest.param(bfast_curl, bfast_curl.BFAST_ADMISSION,
                 "bfast_curl.py", "curl", 48, id="bfast"),
)

#: The kernel-name tuple each family spells, so the device-string clause below asks
#: about EVERY kernel the module ships rather than a hand-picked one.
SHIPPED_KERNELS = {
    "special_kz_curl.py": special_kz_curl.SPECIAL_KZ_KERNELS,
    "bfast_curl.py": bfast_curl.BFAST_KERNELS,
}


def _artifacts(record):
    for relative in record["artifacts"]:
        path = _API / relative
        assert path.exists(), f"the record names an artifact that is not here: {relative}"
        yield relative, json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("module,record,filename,_kind,_cases", FAMILIES)
def test_the_gate_measured_the_bytes_that_ship(module, record, filename,
                                               _kind, _cases):
    """The artifact's own provenance stamp against the file on disk.

    THE DIGEST IS THE GATE'S, not a transcription: ``gate_provenance.stamp``
    records ``module.__file__`` for every repo module the gate process imported,
    which is the one description of "the bytes that ran" that cannot be inferred
    wrongly from a path or a directory name.

    A MOVED FILE HASH IS NOT AUTOMATICALLY A MOVED PROGRAM, and the fall-back is a
    measurement rather than an allowance --
    :func:`test_a_declared_edit_that_moves_a_device_string_is_still_caught` is the
    control that shows it can still fail.
    """
    live = hashlib.sha256(
        pathlib.Path(module.__file__).read_bytes()).hexdigest()
    for relative, payload in _artifacts(record):
        stamped = payload.get("imported_source_sha256") or {}
        matching = [value for key, value in stamped.items()
                    if key.endswith(filename)]
        assert matching, f"{relative} recorded no digest for {filename}"
        if matching[0] == live:
            continue
        declared = record.get("post_gate_record_edits")
        assert declared and all(isinstance(entry, str) and entry.strip()
                                for entry in declared), (
            f"{filename} has changed since {relative} was written and the record "
            f"declares no post-gate edit; the gate measured different bytes from "
            f"the ones that ship, and the record describes a program that is no "
            f"longer here")
        compiled = payload["nvrtc_binary_report"]["binary_sha256_by_source"]
        for name in SHIPPED_KERNELS[filename]:
            digest = hashlib.sha256(
                module.kernel_source(name).encode("utf-8")).hexdigest()
            assert digest in compiled, (
                f"{filename}'s {name} now hashes to {digest}, which {relative}'s "
                f"NVRTC observer never saw. The edit moved a DEVICE STRING, so the "
                f"declaration is wrong and the verdict does not survive it: re-cut "
                f"the gate, do not widen this clause")


@pytest.mark.parametrize("module,record,_filename,_kind,cases", FAMILIES)
def test_the_record_reproduces_the_artifacts_numbers(module, record, _filename,
                                                     _kind, cases):
    """Every count in the record, read back off the JSON it claims to summarise.

    Both policy legs must agree with it: a family released on one policy and
    quietly failing on the other would otherwise be indistinguishable from one
    released on both.
    """
    seen_policies = set()
    for relative, payload in _artifacts(record):
        summary = payload["summary"]
        assert summary["released"] is True, f"{relative} did not release"
        assert summary["scored_cases"] == record["curl_cases_scored"] == cases
        identical = sum(arm["single_identical"] for arm in summary["arms"].values())
        multi = sum(arm["multi_identical"] for arm in summary["arms"].values())
        assert identical == record["curl_single_launch_identical"]
        assert multi == record["curl_multi_step_identical"]
        for side, count in record["constitutive_identical"].items():
            entry = summary["constitutive_per_side"][side]
            assert entry["cases"] == record["constitutive_cases_scored"][side]
            assert entry["single"] == count
            assert entry["multi"] == count
        # TWO BLOCKS, TWO FACTS. ``subnormal_policy_stamp`` is what the process
        # RESOLVED to; ``subnormal_policy_install`` is what installing it actually
        # attained, per executor. A run whose stamp names a policy that some
        # executor did not reach measured a different arithmetic from the one the
        # record claims, so both are read.
        stamp = payload["subnormal_policy_stamp"]
        install = payload["subnormal_policy_install"]
        seen_policies.add(stamp["policy"])
        assert install["installed"] is True, relative
        assert install["unattained"] == [], (
            f"{relative} ran with an UNATTAINED executor: the policy the record "
            f"names was not the policy in force")
        assert all(entry["attained"] for entry in install["executors"].values()), (
            f"{relative}: an executor did not attain the requested policy")
    assert seen_policies == set(record["policies"]), (
        f"the record names {record['policies']} and the artifacts carry "
        f"{sorted(seen_policies)}")


@pytest.mark.parametrize("module,record,_filename,_kind,_cases", FAMILIES)
def test_the_mutation_battery_scores_match(module, record, _filename, _kind,
                                           _cases):
    """A record claiming N caught mutations must have N in the JSON.

    And the NULL count is checked with it, deliberately: a battery whose every leg
    must be caught scores identically whether the comparator works or has
    degenerated into failing everything, so the nulls are the half that says the
    number means something.
    """
    for relative, payload in _artifacts(record):
        source = payload["source_mutations"]
        host = payload["host_mutations"]
        caught = sum(1 for leg in source.values() if leg.get("verdict") == "CAUGHT")
        null = sum(1 for leg in source.values()
                   if leg.get("verdict") == "NULL CONFIRMED")
        assert caught == record["source_mutations_caught"], relative
        assert null == record["source_mutations_null_confirmed"], relative
        assert sum(1 for leg in host.values()
                   if leg["verdict"] == "CAUGHT") == record["host_mutations_caught"]
        assert sum(1 for leg in host.values()
                   if leg["verdict"] == "NULL CONFIRMED") == \
            record["host_mutations_null_confirmed"]
        # EVERY leg must be armed and must have landed on one of the two good
        # verdicts; a NOT ARMED or NO LEGS leg is a defect nobody demonstrated
        # was visible, and it must not hide inside a total.
        for key, leg in source.items():
            assert leg.get("armed"), f"{relative}: {key} was not armed"
            assert leg["verdict"] in ("CAUGHT", "NULL CONFIRMED"), f"{relative}: {key}"
        for key, leg in host.items():
            assert leg["verdict"] in ("CAUGHT", "NULL CONFIRMED"), f"{relative}: {key}"


@pytest.mark.parametrize("module,record,_filename,_kind,_cases", FAMILIES)
def test_the_contraction_guard_claim_is_the_artifacts(module, record, _filename,
                                                      _kind, _cases):
    """``--fmad=false`` is claimed LOAD-BEARING; the artifact must say so.

    The gate scores the unguarded build only where the guarded one was identical,
    and reports DECORATIVE when the unguarded leg agreed too. A record claiming
    load-bearing over a decorative measurement would be claiming evidence the run
    does not carry.
    """
    claim = record["fmad_false_is_load_bearing"]
    for relative, payload in _artifacts(record):
        control = payload["summary"]["guard_control"]
        assert control["comparable_where_guarded_leg_was_identical"] == \
            claim["comparable"], relative
        assert control["diverged"] == claim["diverged"], relative
        assert "load-bearing" in control["reading"], relative


def test_the_two_families_do_not_claim_a_dispatch():
    """Nothing in ``meep_gpu`` imports ``cuda_kernels``; a True from either
    predicate licenses a MEASUREMENT. Both records must say so in as many words,
    because that sentence is the difference between this work and a wiring."""
    for record in (special_kz_curl.SPECIAL_KZ_ADMISSION,
                   bfast_curl.BFAST_ADMISSION):
        assert any("dispatch" in entry
                   for entry in record["what_it_does_not_license"])


def test_the_bfast_record_still_refuses_the_fold():
    """The refusal that costs nothing is the one most likely to be widened by
    argument; the record names it and the predicate must agree."""
    assert any("MIRROR FOLD" in entry
               for entry in bfast_curl.BFAST_ADMISSION["what_it_does_not_license"])


def test_the_special_kz_record_carries_the_measured_inert_mutation():
    """The one MEASURED REFUSAL this round produced: swapping the beta partner for
    its shifted companion is undetectable on any grid the family can be handed,
    because that companion is the neighbour along the invariant axis and every
    beta grid has nz == 1. Recorded as a fact, not deleted as a nuisance."""
    record = special_kz_curl.SPECIAL_KZ_ADMISSION
    assert record["beta_partner_shifted_is_structurally_inert"] is True
    for _relative, payload in _artifacts(record):
        legs = [leg for key, leg in payload["source_mutations"].items()
                if key.endswith("beta_partner_shifted")]
        assert legs, "the null leg is not in the artifact"
        for leg in legs:
            assert leg["verdict"] == "NULL CONFIRMED"
            assert leg["caught"] == 0
            assert leg["ran"] > 0, "a null that was never scored is not a null"


@pytest.mark.parametrize("module,record,filename,_kind,_cases", FAMILIES)
def test_a_declared_edit_that_moves_a_device_string_is_still_caught(
        module, record, filename, _kind, _cases):
    """The control on the fall-back above: a declaration must not be able to pass.

    Both halves of that clause are exercised against the real artifacts. A ONE-BYTE
    change to a shipped kernel's device text must produce a digest the gate's NVRTC
    observer never saw -- on every leg -- because that observer is keyed by the
    source string itself. If this ever stops failing, the fall-back has become an
    allowance and the file hash is the only pin left.
    """
    for relative, payload in _artifacts(record):
        compiled = payload["nvrtc_binary_report"]["binary_sha256_by_source"]
        assert compiled, f"{relative} recorded no compiled sources"
        for name in SHIPPED_KERNELS[filename]:
            shipped = module.kernel_source(name)
            assert hashlib.sha256(shipped.encode("utf-8")).hexdigest() in compiled, (
                f"{relative} never compiled the {name} that ships today")
            mutated = shipped.replace("float", "float ", 1)
            assert mutated != shipped, "the mutation did not apply"
            assert hashlib.sha256(
                mutated.encode("utf-8")).hexdigest() not in compiled, (
                f"a mutated {name} matches a source {relative} compiled, so the "
                f"device-string clause cannot tell the two apart")
