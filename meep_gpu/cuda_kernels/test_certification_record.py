"""``certification.json`` against the shipped source — with no GPU in the room.

WHAT THIS FILE IS FOR. Everything the hand-CUDA track has certified was measured
on one device and written into a ``parity/meep_gpu/results/`` directory — for the
curl pair, ``fused_pml_bit_identity_hand_2026-08-22_rename/``, which replaced the
2026-08-09 run beside it when the folded-PERIODIC top-plane branch moved both
certified device strings. That
directory is not tracked, and until this file existed nothing anywhere read it:
the probe WRITES ``module_sha256`` into ``PROVENANCE.txt``
(``probe_fused_kernel_bit_identity.py:3202``) and a grep over ``<repo>/**/*.py``
found the writer and no reader. A verdict nothing checks is a verdict that
drifts away from the file it describes, silently, on the first edit.

So the record is transcribed into ``certification.json``, beside the kernels, and
this file welds the two together at the merge bar:

  * the shipped module cannot change without the record changing with it;
  * the DEVICE SOURCE the gate actually compiled cannot change at all without a
    failure here, which is the part a comment edit must not be able to disturb
    and a kernel edit must not be able to sneak past;
  * the record names exactly the certified kernels, the file marks every other
    one dead, and the two sets together are all fourteen — so the file and the
    record cannot disagree about what is certified;
  * and the gaps the record does NOT close stay written down in it.

The sibling record is ``triton_kernels/fingerprints.json`` and the sibling tests
are in ``meep_gpu/test_triton_kernels.py``; the clauses here are the ones that
round converged on, ported to a track that had none of them.
"""

from __future__ import annotations

import ast
import hashlib
import json
import pathlib
import re

import pytest

HERE = pathlib.Path(__file__).parent
MODULE = HERE / "step_curl_kernels.py"
RECORD = HERE / "certification.json"

#: The gate artifact the record transcribes. Written on the GPU host, never tracked.
#:
#: RE-POINTED 2026-08-20. It used to name ``…_hand_2026-08-09``, whose layout was
#: FLAT — one ``PROVENANCE.txt`` and one ``gate.json`` at the top. The
#: folded-PERIODIC top-plane branch moved both certified curl device strings, so
#: that verdict certified bytes that no longer ship; the gate was re-run on the
#: shipped ones and the new artifact is PER POLICY: ``keep/`` and ``flush/``, each
#: holding the probe's ``PROVENANCE.txt`` sidecar and its ``probe.json`` payload.
GATE_DIR = (HERE.parents[1] / "parity" / "meep_gpu" / "results"
            / "fused_pml_bit_identity_hand_2026-08-22_rename")

#: Leg subdirectory -> the float32 subnormal policy that leg must stamp.
#:
#: BOTH ARE REQUIRED, and that is the whole reason this mapping exists rather than
#: a single path. "Cut under both policies" is the fact that discharged this
#: record's last piece of metadata debt (``test_certification_metadata.py``'s
#: METADATA_DEBT, emptied on 2026-08-20), so a test that checked one leg would
#: report green over a re-cut that left the debt real.
GATE_POLICY_LEGS = {"keep": "ieee_keep_ftz_stripped",
                    "flush": "meep_x86_flush"}

KERNEL_DECLARATION = re.compile(r'extern "C" __global__ void (\w+)\(')


def record() -> dict:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def module_source() -> str:
    return MODULE.read_text(encoding="utf-8")


def shipped_kernel_names(source: str) -> set:
    """Every kernel NVRTC could be asked to compile, straight out of the text."""
    return set(KERNEL_DECLARATION.findall(source))


def module_level_literal(source: str, name: str):
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name) \
                and node.targets[0].id == name:
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} is not assigned at module level in {MODULE.name}")


def device_sources(source: str) -> dict:
    """The exact CUDA strings ``_get_kernel`` hands ``cp.RawKernel``.

    Evaluated from the syntax tree rather than by importing the module, because
    importing it needs CuPy — which is the whole reason this file exists.
    Concatenations (``_REAL_PML_PRELUDE + r'''…'''``) are evaluated in order
    against the constants already seen, so the certified pair's shared prelude is
    included exactly as it is at compile time.
    """
    environment: dict = {}
    for node in ast.parse(source).body:
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)):
            continue
        name = node.targets[0].id
        if not (name.endswith("_code") or name.endswith("PRELUDE")):
            continue
        expression = ast.Expression(node.value)
        ast.fix_missing_locations(expression)
        environment[name] = eval(compile(expression, "<device source>", "eval"),
                                 {}, dict(environment))
    return {name: text for name, text in environment.items() if name.endswith("_code")}


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# The record against the file.
# --------------------------------------------------------------------------

def test_the_record_matches_the_shipped_kernel_module():
    """Edit the module without touching the record and this fails.

    ``revision_sha256`` is the whole-file digest as of the record. It is not the
    digest the gate ran — that is ``certified_module_sha256``, and the two differ
    — but it is what makes any edit to the module a visible event rather than a
    silent one.
    """
    live = digest(module_source())
    entry = record()
    assert entry["revision_sha256"] == live, (
        "step_curl_kernels.py changed and certification.json did not. Update "
        "revision_sha256, add a post_certification_edits entry saying what moved "
        "and whether it touched device code, and re-run this file.")


def test_a_module_that_has_drifted_from_the_certified_bytes_says_so():
    """The honest half: the record may not quietly imply it describes this file.

    If the shipped module is no longer the bytes the gate compiled, the record
    has to carry the delta — what changed, and whether it reached device code.
    An empty list beside a changed digest would be a record claiming a
    certification it does not have.
    """
    entry = record()
    if entry["revision_sha256"] == entry["certified_module_sha256"]:
        return  # nothing has been edited since the gate; nothing to declare
    edits = entry["post_certification_edits"]
    assert edits, (
        "the shipped module differs from the certified bytes and "
        "post_certification_edits is empty; the record is describing bytes that "
        "no longer exist")
    for edit in edits:
        assert set(edit) >= {"date", "what", "touches_device_code"}, edit
        if edit["touches_device_code"]:
            pytest.fail(
                f"a post-certification edit reached device code ({edit['what']!r}); "
                f"the gate verdict in this record does not describe the shipped "
                f"kernel any more and has to be re-cut on a CUDA host")


def test_a_no_device_edit_that_names_a_kernel_module_says_why_the_verdict_survives():
    """``touches_device_code: False`` has to mean something on a module that HAS device code.

    The field is what every block's release clause reads, and it is the obvious
    place for a verdict to be kept alive by an assertion. An entry naming a module
    that carries kernel source strings and declaring it did not touch device code
    is either (a) true and worth stating why, or (b) the loophole that lets an
    edited kernel keep a stale certification. Requiring the justification costs
    the honest case one sentence and closes the dishonest one.

    MEASURED 2026-08-20, and this is why the rule exists rather than the taste:
    step_curl_kernels.py's certified curl strings grew the BC_MIRROR_PERIODIC
    top-plane mask, and that module is a SUBJECT of the off-diagonal block, whose
    release clause requires every edit to be touches_device_code False. The
    off-diagonal verdict really does survive -- offdiag_emitter.py imports from
    coverage alone and embeds no curl text -- but nothing in the record said so
    until it had to.
    """
    import json as _json

    #: SHIPPED modules that carry NVRTC source strings. A file with no device
    #: text cannot touch device code, so naming it is not a claim that needs
    #: defending -- and a TEST file's device text is fixture material that
    #: nothing certifies, so test_*.py is out.
    device_modules = tuple(
        path.name for path in sorted(HERE.glob("*.py"))
        if not path.name.startswith("test_")
        and "__global__" in path.read_text(encoding="utf-8"))
    assert device_modules, "no device-carrying module found; the scan is broken"

    blocks = _json.loads((HERE / "certification.json").read_text(encoding="utf-8"))
    offenders = []
    for name, block in blocks.items():
        if not isinstance(block, dict) or "post_certification_edits" not in block:
            continue
        for edit in block["post_certification_edits"]:
            if edit.get("touches_device_code"):
                continue
            # THE SUBJECT OF THE EDIT, not every module its prose mentions. Every
            # entry in this record opens with the file it changed ("coverage.py
            # gained ...", "step_curl_kernels.py's certified curl device strings
            # grew ..."), and a rule that matched anywhere in the sentence flagged
            # three entries whose subject was coverage.py for naming a kernel
            # module in passing -- which is a rule about prose, not about bytes.
            named = [module for module in device_modules
                     if edit["what"].startswith(module)]
            if named and not edit.get("device_code_touched_elsewhere"):
                offenders.append((name, named, edit["what"][:90]))
    assert not offenders, (
        "these edits declare touches_device_code False while naming a module "
        "that carries kernel source, and none says why the verdict survives; add "
        "a device_code_touched_elsewhere field naming exactly which strings moved "
        f"and why this block does not compile them: {offenders}")


def test_the_recorded_device_sources_are_the_shipped_device_sources():
    """THE load-bearing one — and it is deliberately narrower than the file hash.

    ``certified_module_sha256`` stops matching the moment anyone fixes a typo in
    a docstring, which makes it useless as a pin on the thing that actually
    decides the verdict. What NVRTC compiles is the kernel source STRINGS, and
    those are pinned here individually: the certified pair by name, and all
    fourteen together so the twelve dead ones are frozen too. A kernel edit
    fails here on a laptop, which is the only place it can fail before a device
    run hours later.
    """
    entry = record()["device_source_sha256"]
    live = device_sources(module_source())
    for name in ("_step_B_pml_real_kernel_code",
                 "_step_D_pml_real_kernel_code"):
        assert name in live, f"{name} is gone from the module"
        assert digest(live[name]) == entry[name], (
            f"{name} changed since the bit-identity gate certified it. Re-run "
            f"parity/meep_gpu/probe_fused_kernel_bit_identity.py --track hand on "
            f"a CUDA host and re-cut certification.json with the new verdict.")
    # THE ALL-FOURTEEN DIGEST IS RETIRED (2026-08-26) AND SO IS THIS CLAUSE. It
    # existed to freeze the twelve INHERITED kernels, which had no gate of their own;
    # they were deleted, so there is nothing left for it to freeze and a hash over two
    # strings under a name that says fourteen would be worse than none. The binding it
    # backed up is UNAFFECTED and is asserted above: the two certified strings are
    # pinned INDIVIDUALLY, which is what NVRTC actually compiles. Both were re-checked
    # against the shipped module after the deletion and both matched, so the pair still
    # compiles the exact bytes the 2026-08-22 re-cut launched.
    assert "all_fourteen_concatenated" not in entry, (
        "the retired digest is back; if the module ever ships uncertified kernels "
        "again they need a freeze of their own, not this name")
    assert len(live) == 2, sorted(live)


@pytest.mark.requires_resource("cuda-gate-artifact")
def test_the_certified_digest_is_the_one_the_gate_wrote():
    """The transcription, checked against the artifact it was transcribed from.

    ``results/`` is untracked, so on most hosts this is a declared skip — the
    record's own transcription is still pinned to the shipped source by the
    tests above. Where the artifact IS present, this is what catches a digest
    typed in by hand from the wrong run.

    BOTH POLICY LEGS, not one. The old artifact was a single flat run and a single
    pair of files was all there was to check. The 2026-08-20 re-cut is two device
    legs — one per float32 subnormal policy — and "cut under both" is the fact that
    let ``test_certification_metadata.py`` delete its last METADATA_DEBT entry. So
    each leg is required to exist, to stamp the policy the record names it by, to
    have compiled THE SAME module bytes as the other, and to carry the verdict
    numbers the record transcribes from it. A re-cut that ran one leg and copied
    the other's numbers fails here.
    """
    if not GATE_DIR.is_dir():
        pytest.skip(f"the 2026-08-20 gate artifact is not on this host ({GATE_DIR})")

    entry = record()
    gate = entry["bit_identity_gate"]
    assert gate["artifact"].rstrip("/").endswith(GATE_DIR.name), (
        f"the record's bit_identity_gate names {gate['artifact']!r} and this test "
        f"reads {GATE_DIR.name}; they have to be the same directory or the check "
        f"is against a run the record does not claim")
    assert set(gate["policies_cut_under"]) == set(GATE_POLICY_LEGS.values())

    compiled = {}
    for leg, policy in GATE_POLICY_LEGS.items():
        provenance = GATE_DIR / leg / "PROVENANCE.txt"
        payload = GATE_DIR / leg / "probe.json"
        assert provenance.exists() and payload.exists(), (
            f"{leg}/ is missing from {GATE_DIR.name}: the record claims a cut under "
            f"both float32 policies and only part of that run is on this host")
        written = dict(line.split(None, 1)
                       for line in provenance.read_text().splitlines() if line.split())
        document = json.loads(payload.read_text(encoding="utf-8"))
        verdict = document["verdict"]

        # THE DIGEST, from both of the artifact's homes for it.
        assert entry["certified_module_sha256"] == written["module_sha256"].strip(), leg
        assert entry["certified_module_sha256"] == document["kernel_source_sha256"], leg

        # THE LEG IS THE POLICY THE RECORD FILES IT UNDER.
        assert written["subnormal_policy"].strip() == policy, leg
        assert verdict["subnormal_policy"] == policy, leg

        # AND IT IS A PASS AT THE NUMBERS THE RECORD READS OFF IT.
        recorded = gate["per_policy"][leg]
        assert verdict["pass"] is True, leg
        assert verdict["curl_single_launch"] == \
            recorded["single_launch_normal_numbers"] == gate["single_launch_guarded"], leg
        assert verdict["curl_single_launch_subnormal_band"] == \
            recorded["single_launch_subnormal_band"] == \
            gate["single_launch_subnormal_band"], leg
        assert verdict["curl_multi_step_budget"] == recorded["multi_step_budget"] == \
            gate["multi_step"]["steps_budget"], leg
        assert verdict["curl_multi_step_all_identical"] is True, leg
        assert recorded["policy_stamp"] == policy, leg
        compiled[leg] = document["kernel_source_sha256"]

    assert len(set(compiled.values())) == 1, (
        f"the two policy legs compiled different module bytes, so 'identical under "
        f"both policies' is two runs on two different subjects: {compiled}")


# --------------------------------------------------------------------------
# The partition: what is certified, what is dead, and nothing in between.
# --------------------------------------------------------------------------

def test_the_record_and_the_file_partition_every_shipped_kernel():
    """A kernel added without a gate verdict must fail here, not ship unmeasured.

    Deliberately a PARTITION and not a coverage claim. The record names two of
    the fourteen; the file's ``UNCERTIFIED_KERNELS`` names the other twelve as
    dead, each with the certified Triton family that already covers the surface
    it targets. What is forbidden is a kernel in neither set — shipped, wired-
    able, and described by nothing.
    """
    source = module_source()
    shipped = shipped_kernel_names(source)
    certified = set(module_level_literal(source, "CERTIFIED_KERNELS"))
    dead = set(module_level_literal(source, "UNCERTIFIED_KERNELS"))

    # TWO, not fourteen, since the twelve inherited kernels were deleted on
    # 2026-08-26 (see certification.json post_certification_edits). The partition
    # rule below is unchanged and is the point: a kernel in neither set would be
    # shipped, wireable and described by nothing.
    assert len(shipped) == record()["shipped_kernel_count"] == 2, sorted(shipped)
    assert not certified & dead, sorted(certified & dead)
    assert certified | dead == shipped, {
        "in neither set": sorted(shipped - certified - dead),
        "named but not shipped": sorted((certified | dead) - shipped),
    }
    assert set(record()["certified_kernels"]) == certified, (
        "certification.json and step_curl_kernels.py disagree about which "
        "kernels are certified")
    # ZERO dead, since the twelve inherited kernels were deleted on 2026-08-26. The
    # partition is now trivially satisfied by the certified pair alone -- which is the
    # point of the cut: every kernel this module ships is one a gate has measured.
    assert len(certified) == 2 and len(dead) == 0


def test_the_record_does_not_claim_the_twelve():
    """The record may name only what it certifies.

    The temptation, when a test asks whether the record names every shipped
    kernel, is to make it pass by listing all fourteen. That would turn a
    two-kernel gate verdict into a directory-wide one. The record names two; the
    other twelve appear nowhere in it as certified.
    """
    entry = record()
    text = json.dumps(entry)
    dead = set(module_level_literal(module_source(), "UNCERTIFIED_KERNELS"))
    named_anyway = sorted(name for name in dead if f'"{name}"' in text)
    assert not named_anyway, (
        f"certification.json names uncertified kernels {named_anyway}; the "
        f"record covers 2 of 14 and inflating it is the failure this prevents")


def test_every_uncertified_kernel_carries_the_dead_marker_and_its_evidence():
    """The marker has to carry the numbers, or it is an opinion.

    A reader who reaches ``UNCERTIFIED_KERNELS`` should find, right there: that
    the plain kernels are 0/16 against the array path, that repairing all twelve
    would serve 26 rows of 186 which a certified Triton family already covers,
    and which family covers each one.
    """
    source = module_source()
    start = source.index("CERTIFIED, AND DEAD")
    block = source[start:source.index("CERTIFIED_KERNELS = (", start)]
    for fact in ("0/16", "26 rows of the 186", "no fail-closed refusal"):
        assert fact in block, f"the dead-kernel block does not state {fact!r}"

    dead = module_level_literal(source, "UNCERTIFIED_KERNELS")
    for name, covered_by in dead.items():
        assert covered_by.startswith("Triton "), (name, covered_by)
        assert "@" in covered_by, (
            f"{name} does not name the sub-step the covering family serves: "
            f"{covered_by!r}")


# --------------------------------------------------------------------------
# The verdict shape, and the gaps the verdict does not close.
# --------------------------------------------------------------------------

def test_the_recorded_gate_verdict_is_pass_shaped():
    """A record that degenerated into "whatever we measured" is not a record.

    The guarded leg must be N/N and the unguarded control 0/N. The unguarded
    half is what stops the guarded half from being a comparison of two identical
    things: if ``--fmad=false`` did nothing, both columns would read N/N.

    AND THE RE-CUT MAY NOT BE WEAKER THAN WHAT IT REPLACED. The 2026-08-09 cut
    this block used to carry ran ONE leg, at normal numbers only, 8 steps deep.
    Its replacement runs a leg per float32 policy, adds a subnormal-band class the
    old one did not have at all, and goes 60 deep. Every one of those is asserted
    per leg, so a future re-cut cannot quietly drop a class or a policy and still
    read as a re-cut: this block is the only place the 120/120 comes from.
    """
    gate = record()["bit_identity_gate"]
    ran, total = gate["single_launch_guarded"].split("/")
    assert ran == total and int(total) > 0, gate["single_launch_guarded"]
    assert gate["single_launch_unguarded"].startswith("0/"), \
        gate["single_launch_unguarded"]
    assert gate["single_launch_unguarded"].split("/")[1] == total, (
        "the guarded and unguarded legs ran different numbers of cases, so one "
        "is not the control for the other")
    assert gate["guard"] == ["--fmad=false"]
    assert record()["mutations"]["legs_as_required"] == "9/9"
    assert record()["mutations"]["harness_blindness_controls"]

    assert gate["value_classes"] == ["uniform", "subnormal_band"]
    assert set(gate["per_policy"]) == {"keep", "flush"}
    for policy, leg in gate["per_policy"].items():
        assert leg["single_launch_normal_numbers"] == gate["single_launch_guarded"], policy
        assert leg["single_launch_subnormal_band"] == \
            gate["single_launch_subnormal_band"], policy
        assert leg["subnormal_band_is_non_vacuous"] is True, (
            f"{policy}: a band leg whose operands hold no subnormal and no signed "
            f"zero exercised the policy exactly as much as the uniform class did")
        assert leg["subnormal_band_operands"]["subnormals"] > 0, policy
        assert leg["normal_class_operands"]["subnormals"] == 0, (
            f"{policy}: the normal-number class drew subnormals, so it is not the "
            f"class the band leg is being contrasted with")
        assert leg["every_primary_case_moved_a_target"] is True, (
            f"{policy}: THE NON-VACUITY FLOOR. zero == zero is bit-identical, so a "
            f"leg in which the array-path reference moved nothing would report a "
            f"perfect pass while measuring nothing at all")
        assert leg["unguarded_normal_numbers"] == gate["single_launch_unguarded"], policy
        assert leg["pass"] is True, policy

    # The two legs must be two RUNS, not one number written twice: the policy that
    # keeps subnormals must not have reached NVRTC with -ftz=true, and the one that
    # flushes them must have.
    assert gate["per_policy"]["keep"]["ftz_true_reached_nvrtc"] is False
    assert gate["per_policy"]["flush"]["ftz_true_reached_nvrtc"] is True


#: The six the 2026-08-14 assessment opened. All six were closed by the
#: 2026-08-15 re-cut and moved to ``resolved_gaps`` — kept, not deleted, so a
#: reader can tell a question that was ANSWERED from one never asked.
GAPS_CLOSED_BY_THE_RECUT = {
    "subnormal-policy-not-recorded",
    "operand-classes-cannot-see-the-policy",
    "multi-step-budget-is-8",
    "compile-memo-cannot-see-a-policy",
    "no-mutation-that-must-be-uncaught",
    "no-armed-mutation-accounting",
}


def test_the_record_states_the_gaps_it_does_not_close():
    """Every gap is either open with a reason, or resolved with a MEASUREMENT.

    The partition is the point. A gap may leave ``open_gaps`` only by acquiring
    the number that closed it — deleting one, or flipping its status on the
    strength of code having changed, is the "fixed means edited" half-truth the
    sibling clause below exists to prevent. So the six are asserted to be
    resolved AND to carry their evidence, and whatever remains open is asserted
    to carry its reason.
    """
    entry = record()
    gaps = {gap["id"]: gap for gap in entry["open_gaps"]}
    resolved = {gap["id"]: gap for gap in entry["resolved_gaps"]}

    assert set(resolved) == GAPS_CLOSED_BY_THE_RECUT, sorted(resolved)
    assert not (set(gaps) & set(resolved)), (
        "a gap is listed as both open and resolved")
    for gap in resolved.values():
        assert gap["status"].startswith("resolved"), gap
        assert gap["resolved_2026-08-15"], gap["id"]
        assert "what_still_needs_the_device" not in gap, (
            f"{gap['id']} is resolved and still names device work as outstanding")
    for gap in gaps.values():
        assert gap["status"].startswith("open"), gap
        assert gap["why_it_matters"] and gap["reference"], gap

    # The re-cut did not close everything, and the one it opened is the residual
    # of the one it closed: flush's host FPU flushes the graded band at
    # generation, so that leg covers the needle constants only (section 7.2).
    assert "graded-subnormal-band-not-covered-under-flush" in gaps


def test_the_recut_records_both_policies_and_the_control_that_scales_them():
    """A 120/120 with no unguarded control beside it is a number with no scale.

    And "identical under both policies" is worth nothing if the two legs were one
    binary served twice, which is exactly what a shared CuPy cache directory
    produces — so the distinct-binary evidence is required to sit here too.
    """
    recut = record()["recut_2026-08-15"]
    assert recut["pml_multi_step_budget"] == 60
    assert recut["pml_value_classes"] == ["uniform", "subnormal_band"]
    assert set(recut["per_policy"]) == {"keep", "flush"}

    for policy, leg in recut["per_policy"].items():
        assert leg["single_launch_normal_numbers"] == "120/120", policy
        assert leg["single_launch_subnormal_band"] == "120/120", policy
        assert leg["subnormal_band_is_non_vacuous"] is True, (
            f"{policy}: a band leg whose operands hold no subnormal and no signed "
            f"zero exercised the policy exactly as much as the uniform class did")
        assert leg["subnormal_band_operands"]["subnormals"] > 0, policy
        assert leg["multi_step_all_identical"] is True, policy
        assert leg["lifted_cases_at_inexact_dtdx"] == "6/6", (
            f"{policy}: the lifted leg is the only evidence for the arithmetic "
            f"guard at corpus scale, and only its inexact-dtdx half is that")
        accounting = leg["armed_mutation_accounting"]
        assert accounting["legs_with_a_source_mutation"] == \
            accounting["legs_whose_mutated_bytes_reached_the_compiler"], policy
        assert accounting[
            "legs_reporting_a_verdict_for_a_mutation_never_compiled"] == [], policy
        # The control. Without it the guarded number has nothing to be measured
        # against; it must be far below the guarded one on the same cases.
        guards = leg["per_guard"]
        assert guards["fmad_false"]["identical"] == guards["fmad_false"]["ran"], policy
        assert guards["default_no_options"]["identical"] < \
            guards["fmad_false"]["identical"] / 100, (
            f"{policy}: the unguarded control did not diverge, so this gate is not "
            f"measuring the arithmetic guard at all")

    binaries = recut["distinct_binaries_per_policy"]
    assert binaries["two_policies_produced_distinct_binaries"] is True
    assert binaries["same_source_both_legs"] is True, (
        "if the two legs compiled different SOURCE, distinct binaries would be "
        "trivial and would say nothing about the policy")
    assert binaries["shared_between_policies"] == []
    assert binaries["warm_cache_nvrtc_calls"] == 0, (
        "the warm-cache control compiled something, so it does not show that "
        "CuPy's disk cache answers above the strip seam")


def test_a_gap_whose_harness_half_landed_says_what_still_needs_the_device():
    """The dangerous half-truth this prevents: "fixed" meaning "changed in code".

    Every one of the six gaps had harness or code work land on 2026-08-15 and NOT
    ONE of them was closed by it, because closing them means re-cutting the gate
    on a device. A gap that describes its code change without naming what is still
    unmeasured reads, at a glance, as a gap that went away — so both fields are
    required together and the gap stays open regardless.
    """
    for gap in record()["open_gaps"]:
        landed = [key for key in gap if key.startswith("harness_landed_")]
        if not landed:
            continue
        assert gap.get("what_still_needs_the_device"), (
            f"{gap['id']} records a harness change and does not say what is still "
            f"unmeasured; a reader would take it for closed")
        assert gap["status"].startswith("open"), gap["id"]


#: Every key in the record that names a MULTI-STEP LAUNCH BUDGET, in any block, in
#: any spelling the gates use. All of them must read 60.
#:
#: What is deliberately NOT in here: ``multi_step_runs`` (how many runs, not how
#: deep each goes) and ``lifted_whole_steps_per_case`` / ``sub_steps_per_case``
#: (whole steps of a lifted corpus case, a different quantity that legitimately
#: reads 8). Sweeping every integer named "step" would have made this clause fail
#: on facts it is not about, which is how a clause gets loosened until it means
#: nothing.
MULTI_STEP_BUDGET_KEYS = {
    "steps_budget", "multi_step_budget", "pml_multi_step_budget",
    "constitutive_multi_step_budget", "multi_step_launches_each",
    "identical_steps_per_run", "multi_step_identical_steps_per_run",
}

#: The depth every certified CUDA family is now cut at.
MULTI_STEP_BUDGET = 60


def multi_step_budgets(node, path="") -> dict:
    """Every recorded launch budget in the record, by the path it sits at."""
    found = {}
    if isinstance(node, dict):
        for key, value in node.items():
            if key in MULTI_STEP_BUDGET_KEYS and isinstance(value, int) \
                    and not isinstance(value, bool):
                found[f"{path}.{key}"] = value
            found.update(multi_step_budgets(value, f"{path}.{key}"))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            found.update(multi_step_budgets(value, f"{path}[{index}]"))
    return found


def test_the_harness_description_and_the_measured_verdict_stay_separate_blocks():
    """The two blocks may now AGREE — and must still be two different kinds of thing.

    THE OLD CLAUSE AND WHY IT WENT. ``harness_2026-08-15`` says what the gate
    harness DOES; ``bit_identity_gate`` says what a device measured. Until
    2026-08-20 the loudest sign the two had not collapsed was that their step
    budgets disagreed — 60 in the harness, 8 in the verdict — so this test pinned
    the 8 and forbade the gate block to name a subnormal policy. Both premises are
    gone, and they went the only way they were ever allowed to go: the gate was
    RE-CUT, at 60, once under each float32 policy, on the shipped device strings.
    The verdict EARNED the harness's number instead of inheriting it, which is
    exactly the event the old clause was there to distinguish from the other one.

    WHAT NOW PROTECTS THE SAME PROPERTY. Not a disagreeing number — a difference in
    kind that survives the numbers agreeing:

      * the verdict block is CHECKABLE AGAINST A RUN. It names an artifact
        directory, a device, both policy stamps, and a per-policy leg for every
        aggregate it reports; ``test_certification_metadata.py`` recomputes its
        digest over that directory and asserts its timestamp, host and policies are
        stamps the tree actually carries.
      * the harness block is checkable against CODE and nothing else. It names no
        device, no environment and no artifact — which is precisely why the
        metadata file's enumerator does not treat it as a verdict — and
        ``parity/meep_gpu/test_pml_gate_harness.py`` welds it to the probe's own
        constants.

    So the collapse this guards against is no longer "the two numbers became one
    field". It is "the verdict block started reporting a number no leg of its own
    ran", and that is what the per-leg agreement below asserts.
    """
    entry = record()
    harness = entry["harness_2026-08-15"]
    assert harness["pml_multi_step_budget"] == MULTI_STEP_BUDGET
    assert harness["pml_value_classes"] == ["uniform", "subnormal_band"]
    assert set(harness["pml_null_mutation_discriminators"]) == \
        set(harness["pml_null_mutations"]), (
        "a null mutation with no discriminating leg is a claim of inertness "
        "nobody showed could fail")
    assert 0.35 in harness["lifted_case_courants"]
    assert entry["lifted_cases"]["dtdx_on_every_case"] == 0.5

    # The harness block is a DESCRIPTION. If it acquired any of these it would be
    # claiming a measurement, and the metadata file would start enumerating it as
    # a verdict that owes four provenance facts — which is the collapse, arriving
    # from the other direction.
    for claim in ("device", "environment", "artifact", "artifacts", "per_policy",
                  "pass"):
        assert claim not in harness, (
            f"harness_2026-08-15 has acquired {claim!r}; it describes what the "
            f"harness does and has no run behind it, so it may not carry the "
            f"shape of a verdict")

    # The verdict block is a MEASUREMENT, and every aggregate it reports has to be
    # one its own legs ran. Raise the block's budget without the legs and this is
    # where it fails.
    gate = entry["bit_identity_gate"]
    assert gate["artifact"] and gate["environment"]["device_name"]
    assert set(gate["per_policy"]) == {"keep", "flush"}
    for policy, leg in gate["per_policy"].items():
        assert leg["multi_step_budget"] == gate["multi_step"]["steps_budget"], policy
        assert leg["multi_step_identical_steps_per_run"] == \
            gate["multi_step"]["identical_steps_per_run"], (
            f"{policy}: the block reports {gate['multi_step']['identical_steps_per_run']} "
            f"identical steps per run and this leg ran "
            f"{leg['multi_step_identical_steps_per_run']}")
        assert leg["multi_step_all_identical"] is True, policy
        assert leg["multi_step_runs"] == gate["multi_step"]["runs"], policy
    assert gate["multi_step"]["all_steps_identical"] is True

    # And the 8 is gone from the whole record, not merely from the block that
    # carried it: the gap it was named after is resolved on the primary block too.
    budgets = multi_step_budgets(entry)
    assert budgets, "no multi-step budget found in the record; the scan is broken"
    wrong = {path: value for path, value in budgets.items()
             if value != MULTI_STEP_BUDGET}
    assert not wrong, (
        f"these recorded launch budgets are not {MULTI_STEP_BUDGET}: {wrong}. Every "
        f"certified CUDA family is cut at {MULTI_STEP_BUDGET}; a lower one is either "
        f"a verdict that was never re-cut or a budget somebody lowered.")
    assert "multi-step-budget-is-8" in {gap["id"] for gap in entry["resolved_gaps"]}
    assert gate["_multi_step_supersedes_the_recorded_8"], (
        "the block no longer says what its 60 replaced, so a reader comparing it "
        "against the 2026-08-14 disposition finds the 8 simply missing")


def test_the_lifted_case_leg_records_the_regime_it_could_not_test():
    """The six corpus cases all ran at dtdx = 0.5, and that hides the guard.

    A power-of-two dtdx scales exactly, so a contracted and an uncontracted
    stencil round identically — the module's own docstring says so. The leg is
    real evidence for the ghost rules, the metallic mask and the coefficient
    lattice at 2560²; it is not independent evidence for ``--fmad=false``, and
    the record has to say which it is.
    """
    lifted = record()["lifted_cases"]
    assert lifted["dtdx_on_every_case"] == 0.5
    assert lifted["dtdx_float32_exact_on_every_case"] is True
    assert "not independent evidence" in \
        lifted["_what_this_leg_does_not_establish"].lower()
    assert set(record()["bit_identity_gate"]["swept"]["dtdx_values"]) == {"0.5", "0.35"}, (
        "the synthetic leg is where the guard IS established, because it sweeps "
        "an inexact dtdx; if that stopped being true the lifted-case caveat is "
        "no longer a caveat, it is the whole story")


def test_the_record_does_not_claim_the_kernels_are_dispatched():
    """Nothing here runs in production, and the record says so out loud."""
    assert record()["dispatch"]["wired"] is False


def test_the_test_coverage_index_reports_as_many_open_gaps_as_the_record_carries():
    """The index row that names the gaps may not under-report them.

    ``the test coverage notes``'s row for this file said "the four open gaps" and
    enumerated four while the record carried six and the test above pinned six —
    the two omitted being exactly the pair §1.5 singles out (a mutation that must
    be UNCAUGHT, and armed-mutation accounting). A stale count in the file the
    round patched *for* stale counts, so the count is welded rather than trusted:
    the row must spell the number the record actually carries.
    """
    words = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
             7: "seven", 8: "eight", 9: "nine", 10: "ten"}

    def phrase(n: int) -> str:
        # Pluralised, because the count reached one on 2026-08-15 and "the one
        # open gaps" is not a sentence a reader would write or trust.
        return f"the {words[n]} open gap" + ("" if n == 1 else "s")

    count = len(record()["open_gaps"])
    narrative = HERE.parents[0] / "TEST_COVERAGE.md"
    if not narrative.exists():
        from conftest import requires_resource_skip  # noqa: PLC0415

        requires_resource_skip(
            "coverage_narrative",
            f"{narrative.name} is a development document and is not part of "
            f"this repository")
    index = narrative.read_text(encoding="utf-8")
    assert phrase(count) in index, (
        f"certification.json carries {count} open gaps and TEST_COVERAGE.md does "
        f"not say {phrase(count)!r}")
    for wrong in words:
        if wrong != count:
            assert phrase(wrong) not in index, (
                f"TEST_COVERAGE.md still says {phrase(wrong)!r}")

    # And the resolved ones may not simply vanish from the index either: the row
    # has to say they were closed by a measurement, or a reader comparing the
    # index against the 2026-08-14 disposition finds five questions missing.
    assert "resolved_gaps" in index or "re-cut" in index, (
        "TEST_COVERAGE.md does not mention the re-cut that closed the other "
        "gaps, so the drop from six to one reads as gaps going quiet")


def test_the_record_does_not_call_the_compile_log_an_nvrtc_compile_count():
    """The log counts ``cp.RawKernel`` constructions, and the record must say so.

    They are not the same quantity, by this project's own measurement: the
    ``cuda_policy_reach`` artifact records a leg where the memo was cleared, a
    kernel was constructed, and the bytes did not change — CuPy's disk cache
    answered above the strip seam and no compiler ran. The distinction does not
    bite for an armed mutation (mutated source misses that source-keyed cache),
    but an auditor checking a leg's accounting would be reading the wrong number.
    """
    text = json.dumps(record())
    claims = re.findall(r"(?:logs|records|carries) one entry per ([A-Za-z. -]+)", text)
    assert claims, "the record no longer describes what the compile log counts"
    for claimed in claims:
        assert "NVRTC" not in claimed, (
            f"the record says the log carries one entry per {claimed.strip()!r}; "
            f"it carries one per cp.RawKernel construction, which is an upper "
            f"bound on NVRTC calls and equal to it only for mutated sources")
    assert "upper bound on NVRTC" in text, (
        "the record states the quantity without stating how it relates to the "
        "one a reader will assume, which is what made the old wording wrong")


LINE_REFERENCE = re.compile(r"see :\d+")


def test_the_kernel_module_carries_no_bare_line_number_reference():
    """A pointer that goes stale on the next insertion is worse than no pointer.

    The ``_COMPILE_OPTIONS`` docstring — the one place this file states the
    CONDITION its bit-identity claim rests on — pointed at ":1445-1453" for the
    parenthesization argument. That was correct in the certified blob and wrong
    the moment ~118 lines went in above it: the same range now lands in the
    launch wrapper of ``step_D_pml_sym_complex``, one of the twelve kernels
    this file marks DEAD. Cross-references here name the thing, not its line.
    """
    source = module_source()
    found = LINE_REFERENCE.findall(source)
    assert not found, (
        f"{MODULE.name} carries line-number references {found}; they go stale on "
        f"the next insertion above them and nothing recomputes them. Name the "
        f"symbol or the section heading instead.")
    assert "THE STENCIL IS PARENTHESISED TO" in source, (
        "the note the _COMPILE_OPTIONS docstring points at by name is gone")
