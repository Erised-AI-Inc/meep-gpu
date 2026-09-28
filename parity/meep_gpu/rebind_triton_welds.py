"""Rebind ``triton_kernels/fingerprints.json`` to a fresh round of gate artifacts.

WHY THIS EXISTS. A weld entry names the exact bytes its gate executed. One edit
to ``triton_kernels/launch.py`` -- pinned by 28 of the 30 ``status == "PASS"``
entries -- puts every one of them in drift: the ledger still points at the
previous round's artifacts, so the welds describe bytes that no longer ship.
Re-running the gates is only half the repair. Until the ledger is rebound to the
NEW artifacts the record keeps its old digests, and the weld tests correctly
refuse the tree. This is the second half, and it is the Metal track's
``rebind_metal_welds.py`` brought across unchanged in intent.

WHAT IT REFUSES TO DO. It does not invent a weld, widen one, or promote a gate
that did not pass. An entry is rebound only when the fresh artifact for that
family says ``released``; anything else is reported and skipped, leaving the
stale entry in place to keep failing. It keeps each entry's CURATED key set --
both which PATHS that entry chose to pin and which FIELDS it carries -- because
either is a claim someone made rather than a thing to regenerate. Only the
digests, the artifact pointer, the ``records`` line and the timestamp move.

It is NOT a normalizer. Every bespoke hand-authored key on an entry
(``slurm_job_id``, ``recert_*``, ``probe_sha256``, ``real_engine_route``,
``standalone_product_gate``, the ``_why``/``_dispatch_note`` prose, ...) is
asserted byte-identical after the rewrite, and an entry the tool does not
understand is skipped rather than reshaped.

    python rebind_triton_welds.py --campaign results/triton_regate_2026-08-27_routing
    python rebind_triton_welds.py --campaign results/triton_regate_2026-08-27_routing --write
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).resolve().parent


def _find_api_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "parity" / "meep_gpu").is_dir() and (candidate / "meep_gpu").is_dir():
            return candidate
    raise SystemExit(f"cannot locate the repository root above {start}")


_API = _find_api_root(_HERE)
if str(_API) not in sys.path:
    sys.path.insert(0, str(_API))

from meep_gpu.code_identity import code_digest_of_path  # noqa: E402
from meep_gpu.device_identity import device_digests      # noqa: E402

LEDGER = _API / "meep_gpu" / "triton_kernels" / "fingerprints.json"

#: Ledger key -> the campaign subdirectory whose gate measured it.
#:
#: AN EXPLICIT TABLE IS THE HONEST CHOICE, and not because deriving the name is
#: hard. Stripping ``triton_``/``_device_gate`` happens to reproduce all twelve
#: rows below -- which is exactly the hazard. The stem rule is a coincidence of
#: this round, and the ledger already contains keys it would get wrong in both
#: directions: ``triton_dispersive_fused_pair_device_gate`` is welded by
#: ``probe_triton_dispersive_fused_pair/dispersive_fused_pair.json`` and
#: ``triton_fused_ade_chain_device_gate`` by ``gate_triton_fused_ade_chain/
#: gate_keep.json``, neither of which the stem produces. A stem match that lands
#: on the WRONG family binds a weld to a run that measured something else, and
#: nothing downstream can detect it: the digests would all verify, because the
#: artifacts of a full campaign import the same closure. A missing table row
#: fails closed and is reported; a wrong stem match is silent. So the mapping is
#: typed once, here, where a reader can check it against the gate scripts.
CAMPAIGN_DIRS = {
    # 2026-08-30, the complex D->E weld. The campaign directory is named for the
    # BINDS key propose_triton_welds uses, which is also the board's product name;
    # it is typed here for the same reason every other row is -- a stem rule that
    # happened to reproduce it would be a coincidence of this round.
    "triton_complex_fused_electric_pair_device_gate": "complex_fused_electric_pair",
    # 2026-09-17. This row was MISSING, and the machinery above did exactly what its
    # own comment promises: the entry was reported UNCOVERED -- "no row in
    # CAMPAIGN_DIRS -- needs a device re-run, not a guessed directory" -- rather than
    # stem-matched onto something that happened to be near. The stem rule WOULD have
    # produced this one correctly, which is precisely why it is typed instead: a rule
    # that is right by coincidence is not a rule. The directory is the one the entry's
    # own `records` line has always named (triton_fleet_2026-09-15_gapclose_ccpair/
    # complex_conductive_fused_pair/), and the producer is
    # probe_triton_complex_conductive_fused_pair.py, whose --out is a DIRECTORY it
    # writes gate.json into.
    "triton_complex_conductive_fused_pair_device_gate": "complex_conductive_fused_pair",
    "triton_complex_device_gate": "complex",
    # 2026-09-23. The two STENCIL welds were seeded on 2026-09-17 and reported UNCOVERED
    # on every rebind since, so their pins on driver.py and stepping.py drifted through
    # two rounds that had re-gated them. Their producer is
    # gate_triton_offdiag_stencil_welds.py --family <family> --out-root <root>, which
    # writes <root>/<family>/gate.json, so the campaign tree to pass is
    # triton_offdiag_stencil_welds_<stamp>/ and the directory is the family name.
    "triton_offdiag_fused_electric_pair_device_gate": "offdiag_fused_electric_pair",
    "triton_folded_offdiag_fused_electric_pair_device_gate": "folded_offdiag_fused_electric_pair",
    "triton_conductivity_device_gate": "conductivity",
    "triton_cylindrical_complex_device_gate": "cylindrical_complex",
    "triton_cylindrical_device_gate": "cylindrical",
    "triton_folded_complex_device_gate": "folded_complex",
    "triton_fused_electric_device_gate": "fused_electric",
    "triton_no_pml_constitutive_device_gate": "no_pml_constitutive",
    "triton_no_pml_device_gate": "no_pml",
    "triton_nonlinear_device_gate": "nonlinear",
    "triton_offdiag_device_gate": "offdiag",
    "triton_symmetry_device_gate": "symmetry",
    "triton_unified_expansion_device_gate": "unified_expansion",

    # TRANCHE 2+, 2026-08-28. The families the 2026-08-27 campaign did not run and
    # this tool reported as UNCOVERED. Their artifacts live in the tranche2/5/6
    # campaign trees, so pass --campaign once per tree; the first tree that holds a
    # released artifact for a key wins, and a key present in none stays UNCOVERED.
    "triton_complex_ade_device_gate": "complex_ade",
    "triton_complex_no_pml_curl_device_gate": "complex_no_pml_curl",
    "triton_complex_no_pml_conductive_device_gate": "complex_no_pml_conductive",
    "triton_complex_no_pml_stored_e_device_gate": "complex_no_pml_stored_e",
    "triton_complex_offdiag_device_gate": "complex_offdiag",
    "triton_no_pml_conductive_device_gate": "no_pml_conductive",
    "triton_no_pml_stored_e_device_gate": "no_pml_stored_e",
    "triton_folded_offdiag_device_gate": "folded_offdiag",
    "triton_folded_offdiag_dispersive_device_gate": "folded_offdiag_dispersive",
    "triton_fused_ade_chain_device_gate": "fused_ade_chain",
    "triton_complex_fused_ade_chain_device_gate": "complex_fused_ade_chain",
    # 2026-09-10, the folded off-diagonal dispersive E->P chain. FOR REFRESHES
    # ONLY: this table maps an EXISTING ledger key to the campaign subdirectory a
    # re-run writes, and this tool never invents a key. The first entry is minted
    # by seed_triton_welds.py from the released keep artifact.
    "triton_folded_offdiag_fused_ade_chain_device_gate":
        "folded_offdiag_fused_ade_chain",

    # THE FUSED-PAIR FAMILIES, 2026-08-28. These ship NO gate_triton_*.py of their
    # own -- they are gated by probe_triton_*.py scripts run directly, so their
    # campaign subdirectory is named for the PROBE, not the family. That is why the
    # table cannot be derived from the ledger key by string surgery and is written
    # out instead.
    "triton_folded_fused_pair_device_gate": "probe_triton_folded_fused_pair",
    "triton_folded_fused_magnetic_pair_device_gate":
        "probe_triton_folded_fused_magnetic_pair",
    "triton_folded_complex_fused_magnetic_pair_device_gate":
        "probe_triton_folded_complex_fused_magnetic_pair",
    "triton_complex_fused_magnetic_pair_device_gate":
        "probe_triton_complex_fused_magnetic_pair",
    "triton_cylindrical_real_fused_magnetic_pair_device_gate":
        "probe_triton_cylindrical_real_fused_magnetic_pair",
    "triton_dispersive_fused_pair_device_gate": "probe_triton_dispersive_fused_pair",
    "triton_fused_ade_state_device_gate": "probe_triton_fused_ade_state",
    "triton_fused_dispersive_chain_device_gate": "probe_triton_fused_dispersive_chain",

    # THE ARMS RELEASED 2026-09-11. Typed here in the SAME change that pre-writes
    # their `fastpath.ARM_CERTIFICATION` rows, not after their entries exist: these
    # three keys are SEEDED by seed_triton_welds.py from the 2026-09-11_realarms
    # fleet, and a row here for a key the ledger does not yet carry reports UNCOVERED
    # and binds nothing -- which is the correct reading until the seed runs, and the
    # right place to be caught if it never does. Their fleet subdirectory is the gate
    # name in drive_triton_weld_gates.GATES, which for these three is the FAMILY and
    # not the probe (contrast the block above, where it is the probe).
    "triton_folded_dispersive_fused_pair_device_gate": "folded_dispersive_fused_pair",
    "triton_conductive_fused_electric_pair_device_gate":
        "conductive_fused_electric_pair",
    "triton_cylindrical_real_fused_electric_pair_device_gate":
        "cylindrical_real_fused_electric_pair",

    # THE DEPOSIT IMAGE CLOSURE, 2026-08-30. Not a product weld like the rest of
    # this table: it binds the gate that shows the two FOLDED fused products
    # REQUIRE deposit_repair's image closure, which is the evidence the
    # 2026-08-30 CARRIES_DEPOSIT_REPAIR flip was missing. Its directory holds
    # gate_keep.json and gate_flush.json and no gate.json, so ``_artifact_for``
    # selects the KEEP record -- the policy every other weld here was cut under.
    "triton_folded_deposit_closure_device_gate":
        "probe_triton_folded_deposit_closure",
    # THE FIFTEEN ARMS PHASE B RELEASED, 2026-09-13. Seeded by seed_triton_welds.py
    # from the 2026-09-12 identity fleet and refreshed here from the batch fleet
    # (shard C, and the cylindrical pair's own shards). Their fleet subdirectory is
    # the gate name in drive_triton_weld_gates.GATES, which for these is the FAMILY
    # (the probe-named rows above are the 2026-08-28 exceptions, kept as they were).
    "triton_beta_fused_electric_pair_device_gate": "beta_fused_electric_pair",
    "triton_beta_fused_magnetic_pair_device_gate": "beta_fused_magnetic_pair",
    "triton_bfast_fused_electric_pair_device_gate": "bfast_fused_electric_pair",
    "triton_bfast_fused_magnetic_pair_device_gate": "bfast_fused_magnetic_pair",
    "triton_nonlinear_fused_magnetic_pair_device_gate": "nonlinear_fused_magnetic_pair",
    "triton_folded_beta_fused_electric_pair_device_gate": "folded_beta_fused_electric_pair",
    "triton_folded_beta_fused_magnetic_pair_device_gate": "folded_beta_fused_magnetic_pair",
    "triton_no_pml_fused_electric_pair_device_gate": "no_pml_fused_electric_pair",
    "triton_complex_beta_fused_electric_pair_device_gate": "complex_beta_fused_electric_pair",
    "triton_complex_beta_fused_magnetic_pair_device_gate": "complex_beta_fused_magnetic_pair",
    "triton_folded_beta_complex_fused_magnetic_pair_device_gate":
        "folded_beta_complex_fused_magnetic_pair",
    "triton_folded_beta_complex_fused_pair_device_gate": "folded_beta_complex_fused_pair",
    "triton_folded_complex_fused_pair_device_gate": "folded_complex_fused_pair",
    "triton_cylindrical_fused_magnetic_pair_device_gate": "cylindrical_fused_magnetic_pair",
    "triton_cylindrical_fused_electric_pair_device_gate": "cylindrical_fused_electric_pair",
}

#: The fields this tool owns. Everything else on an entry survives untouched,
#: and the assertion below proves it did.
#:
#: ``host`` is owned CONDITIONALLY and only under :data:`HOST_PENDING` -- see
#: :func:`_host_line`. It is listed here because the survival proof compares
#: every field NOT in this tuple byte for byte, and a field that may legitimately
#: move cannot also be asserted immovable.
#:
#: ``subnormal_policy`` is owned on the same terms and only under
#: :data:`POLICY_UNNAMED`. Because listing it here takes it out of the byte-for-byte
#: comparison for EVERY entry, the survival proof asserts it separately: an entry
#: that did not read the sentinel must come out with the policy it went in with.
REFRESHED = ("source_sha256", "code_sha256", "device_sha256", "artifact_sha256",
             "records", "recorded_utc", "status", "verdict_read_from", "host",
             "subnormal_policy")

#: The ONE value of ``subnormal_policy`` this tool will overwrite.
#:
#: WHY IT EXISTS (2026-09-15). ``triton_complex_offdiag_device_gate`` has read
#: ``"see artifact"`` since 2026-08-19, because ``gate_triton_complex_offdiag.py``
#: wrote no policy stamp, and it is the one name in
#: ``test_triton_weld_contract.POLICY_UNNAMED_BUDGET``. The gate stamps the policy
#: now, but a rebind could not carry the stamp into the ledger: this tool owned no
#: policy field, and ``seed_triton_welds.py`` refuses a key that already exists. So
#: the debt could only be cleared by a hand edit, which is the record forgery the
#: ledger forbids. The rule is :data:`HOST_PENDING`'s, and deliberately as narrow:
#: derive the policy only where the entry says it points elsewhere, derive it with
#: ``seed_triton_welds.policy_line`` so a rebound entry and a seeded one spell it
#: alike, skip the entry when the artifact cannot name one, and never restate any
#: other entry's policy.
POLICY_UNNAMED = "see artifact"

#: The ONE value of ``host`` this tool will overwrite, and it is a sentinel no
#: real host string can be.
#:
#: WHY A SENTINEL AND NOT A GENERAL RULE. A new weld has to get its ``host`` from
#: somewhere, and the alternative is a human typing a machine description into the
#: ledger beside digests that were derived from a measurement -- the one
#: hand-authored factual claim in an otherwise derived entry. Deriving it from the
#: artifact removes that. But REWRITING the host of an already-bound weld is a
#: different act: those strings were authored against runs whose artifacts are not
#: all resolvable from a checkout, and a tool that silently restates them would be
#: replacing a standing claim with a reconstruction. So the rule is narrow and
#: NAMED: derive the host only where the entry says it is waiting for one, refuse
#: to bind at all if it says that and the artifact cannot answer, and never touch
#: any other entry's host. This is deliberately not a predicate that decides which
#: host strings are "equivalent" -- that shape is what let a comment-only edit keep
#: twenty-one Triton welds green while they described bytes that no longer shipped.
HOST_PENDING = "PENDING REBIND"

#: RECORDS CARVE-OUT. ``test_triton_folded_fused_pair.py:672`` asserts the weld's
#: ``records`` line still names ``run_farcarryD5``/``run_farcarryD6`` -- the runs
#: that measured ``fill_folded_far_ghosts_D`` carried inline and released it.
#: That release is EARLIER than any re-pin and is not superseded by one, so a
#: generated ``records`` line would delete a standing claim rather than refresh
#: it. Where a sentinel is listed, the old line is kept and the fresh one is
#: appended; if the sentinel is not in the result the entry is skipped instead.
RECORDS_SENTINELS = {
    "triton_folded_fused_pair_device_gate": ("run_farcarryD5", "run_farcarryD6"),
}


def _host_line(fresh: dict) -> str | None:
    """The weld's ``host``, built from the artifact's OWN environment block.

    Returns ``None`` when the artifact cannot answer, and the caller then SKIPS
    the entry rather than binding it with a half-known host. The four parts are
    the four an existing weld's host string carries, and each is load-bearing
    rather than decorative:

    * the machine, so a later reader can go back to it;
    * the device AND its compute capability -- ``test_triton_weld_contract.py``
      binds both to the record's ``validated_triton_versions`` /
      ``validated_compute_capabilities``, because Triton generates PTX for an
      ARCHITECTURE and a weld cut on an unvalidated one must fail until the
      declaration is widened deliberately;
    * the Triton and CuPy versions, for the same reason;
    * the physical GPU index, which is what makes a placement claim checkable.

    ONE SPELLING FOR BOTH TOOLS. ``seed_triton_welds.py`` calls this rather than
    composing its own line: a seeded weld and a rebound one must be
    indistinguishable, and two f-strings drift. The alternative spellings read below
    (``host``/``device_name``/``cc``, and ``env.CUDA_VISIBLE_DEVICES`` nested one
    level down) are the ones measured across the 2026-09-09 fleet artifacts, whose
    ``environment`` blocks were authored per probe and do not agree.

    The capability is normalised through the ENGINE's own rule rather than compared
    raw. CuPy's ``Device().compute_capability`` is ``"86"`` and
    ``getDeviceProperties`` gives ``"8.6"``; the contract tests a substring of this
    line against ``validated_compute_capabilities`` (``["8.6"]``), so the undotted
    spelling would bind a weld that reports "names no validated cc" for a run on a
    validated architecture. Imported lazily: this tool must still list and report on
    a checkout where the engine package will not import.
    """
    from meep_gpu import fastpath  # noqa: PLC0415

    env = fresh.get("environment")
    if not isinstance(env, dict):
        return None
    host = env.get("hostname") or env.get("host")
    device = env.get("device") or env.get("device_name")
    capability = fastpath._normalized_capability(
        env.get("compute_capability") or env.get("cc") or "")
    triton = env.get("triton")
    cupy = env.get("cupy")
    if not (host and device and capability and triton and cupy):
        return None
    # A VERSION, NOT A REPORT OF ITS ABSENCE. A probe whose `environment()` fell to
    # its except branch writes `cupy = "unavailable: TypeError(...)"`, which is
    # truthy and would compose a host line reading "CuPy unavailable: TypeError(...)"
    # -- a weld naming an exception where its toolchain belongs, and one the contract
    # test would pass because it greps only for the Triton and cc substrings.
    for name, version in (("CuPy", cupy), ("Triton", triton)):
        text = str(version)
        if not text[:1].isdigit() or "unavailable" in text or "Error" in text:
            return None
    gpu = (env.get("cuda_visible_devices")
           or (env.get("env") or {}).get("CUDA_VISIBLE_DEVICES"))
    return (f"{host}: {device}, cc {capability}; CuPy {cupy}; Triton {triton}"
            + (f" (GPU {gpu})" if gpu else ""))


def _artifact_for(directory: Path) -> Path | None:
    for candidate in ("gate.json", "gate_keep.json"):
        if (directory / candidate).is_file():
            return directory / candidate
    found = sorted(p for p in directory.glob("*.json") if p.name != "provenance.json")
    return found[0] if found else None


def _verdict(fresh: dict) -> tuple[dict, str]:
    """``canonical_verdict`` first, then ``release``. A NULL value is not a verdict.

    Both keys are present-but-empty on some artifacts in this tree -- the Metal
    whole-step gate writes ``canonical_verdict.released = null`` and puts the
    real answer in ``release`` -- so falling through on absence alone reads the
    empty one and refuses a gate that passed.
    """
    verdict = fresh.get("canonical_verdict") or {}
    if verdict.get("released") is not None:
        return verdict, "canonical_verdict"
    return (fresh.get("release") or {}), "release"


#: The one ledger entry bound by :func:`bind_bit_identity` — the sibling-pin
#: shape (``adapter``/``adapter_sha256``, ``probe``/``probe_sha256``) that the
#: campaign loop cannot reach: the entry carries no ``status: PASS``, no
#: ``source_sha256`` map and no artifact pointer, so nothing in the loop above
#: has anywhere to read a fresh digest from. Its six legs are curated numbers
#: from the 2026-08-09/10 campaign; what the two sibling pins attest is the
#: HARNESS — the probe and its Triton adapter — and what re-earns them is a
#: fresh RELEASED run of that probe's identity legs against the current bytes.
BIT_IDENTITY_KEY = "bit_identity_gate"

#: The experiments a bit-identity re-run must have executed for the binder to
#: accept it — every RUNTIME leg family the entry curates (``pml_curl_step``,
#: ``constitutive_step``, ``fused_curl_constitutive_B``, ``whole_step`` /
#: ``engine_route``, with each family's multi-step arm). An artifact missing
#: any is refused: it re-earned part of the verdict and the pins bind the
#: whole harness.
#:
#: ``compile`` (the ``compile_time_census_A1`` leg) is DELIBERATELY not here:
#: that experiment drives the HAND-CUDA kernel-string contract
#: (``module.cp``/``module._clear_kernel_cache``) and does not run on the
#: Triton adapter at all — the curated A1 numbers are a compile-time census of
#: ``kernels.py``'s PTX, whose bytes the package's other welds pin, not a
#: behaviour of the probe/adapter pair these two pins name.
BIT_IDENTITY_EXPERIMENTS = frozenset({
    "pml", "multistep", "constitutive", "constitutive_multistep",
    "fused_pair", "fused_pair_multistep", "wholestep",
})


def bind_bit_identity(artifact_argument: str, write: bool) -> int:
    """Refresh ``bit_identity_gate``'s two sibling pins from a RELEASED re-run.

    THE BINDER THE 2026-08-30 CONTRACT ROUND SAID WAS MISSING
    (``test_triton_weld_contract.py``: "CLEARING IT IS A TOOL, NOT A TYPING
    JOB... the red stands, deliberately, until somebody writes the binder").
    Everything it writes is DERIVED from the fresh artifact, and every refusal
    fails closed:

    * the artifact must carry a verdict that READS as released
      (``canonical_verdict``/``verdict.pass`` — ``pml_verdict`` refuses an
      empty run, so a vacuous artifact cannot release);
    * it must have run EVERY family in :data:`BIT_IDENTITY_EXPERIMENTS` — a
      partial re-run re-earns part of the verdict while the pins bind the
      whole harness;
    * it must have driven the TRITON track through the recorded adapter
      (both files present in ``imported_source_sha256``);
    * every digest the run recorded for a file in this checkout must MATCH the
      checkout — the proof the run executed these exact bytes — before any of
      them is written into the ledger.

    Only ``adapter_sha256`` and ``probe_sha256`` are refreshed, from the
    artifact's own import record; ``records``, ``artifact_sha256`` and
    ``recorded_utc`` are bound beside them so the run is followable from the
    entry. The six curated legs and the mutation tables are NOT rewritten:
    they are the 2026-08-09/10 campaign's measurements and remain attributed
    to it — what this binds is that the CURRENT harness bytes reproduce the
    released identity verdict those legs curate.
    """
    artifact = Path(artifact_argument)
    if not artifact.is_absolute():
        artifact = (_HERE / artifact).resolve()
    if artifact.is_dir():
        found = _artifact_for(artifact)
        if found is None:
            raise SystemExit(f"{artifact} holds no gate artifact")
        artifact = found
    if not artifact.is_file():
        raise SystemExit(f"no such artifact: {artifact}")
    if _API not in artifact.parents:
        raise SystemExit(
            f"{artifact} lives outside {_API}; the records line must name an "
            "apps/api/... path or it points nowhere resolvable")

    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    before = json.loads(json.dumps(ledger))
    entry = ledger.get(BIT_IDENTITY_KEY)
    if not isinstance(entry, dict):
        raise SystemExit(f"the ledger has no {BIT_IDENTITY_KEY} entry")
    for field in ("adapter", "adapter_sha256", "probe", "probe_sha256"):
        if field not in entry:
            raise SystemExit(
                f"{BIT_IDENTITY_KEY} carries no {field}; this binder only "
                f"understands the sibling-pin shape and will not invent one")

    fresh = json.loads(artifact.read_text(encoding="utf-8"))
    verdict, source_field = _verdict(fresh)
    if verdict.get("released") is not True:
        raise SystemExit(
            f"{artifact.name} is not released "
            f"({source_field}.released={verdict.get('released')!r}: "
            f"{verdict.get('reasons') or 'no reason given'}); a pin is "
            f"re-earned by a released run, never by a run that exists")
    ran = set(fresh.get("experiments") or ())
    missing = sorted(BIT_IDENTITY_EXPERIMENTS - ran)
    if missing:
        raise SystemExit(
            f"{artifact.name} did not run {missing} (it ran {sorted(ran)}); "
            f"the pins bind the whole harness and a partial re-run re-earns "
            f"part of the verdict")

    imported = fresh.get("imported_source_sha256") or {}
    adapter_path, probe_path = entry["adapter"], entry["probe"]
    for name in (adapter_path, probe_path):
        if name not in imported:
            raise SystemExit(
                f"{artifact.name} records no digest for {name}; the run did "
                f"not demonstrably execute the file this pin names")
    disagreeing = [name for name, digest in sorted(imported.items())
                   if (_API / name).is_file()
                   and hashlib.sha256(
                       (_API / name).read_bytes()).hexdigest() != digest]
    if disagreeing:
        raise SystemExit(
            f"the run disagrees with this checkout on {disagreeing[:5]}; it "
            f"describes a different tree and nothing here may be bound from it")

    relative = artifact.relative_to(_API).as_posix()
    entry["adapter_sha256"] = imported[adapter_path]
    entry["probe_sha256"] = imported[probe_path]
    entry["artifact_sha256"] = hashlib.sha256(artifact.read_bytes()).hexdigest()
    entry["records"] = (
        f"apps/api/{relative} - {source_field}.released=true "
        f"(read_from {verdict.get('read_from', '?')}); the identity-leg "
        f"families {sorted(BIT_IDENTITY_EXPERIMENTS)} re-run against the "
        f"current tree through the Triton adapter, {len(imported)} imported "
        f"digests all matching this checkout. The six curated legs and the "
        f"mutation tables remain the 2026-08-09/10 campaign's measurements "
        f"(results/triton_fusedpair_2026-08-09/); this run is what binds the "
        f"CURRENT probe and adapter bytes to that released verdict.")
    entry["recorded_utc"] = datetime.now(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ")

    # THE SURVIVAL PROOF: only the fields this binder owns may move.
    owned = {"adapter_sha256", "probe_sha256", "artifact_sha256", "records",
             "recorded_utc"}
    old = before[BIT_IDENTITY_KEY]
    assert set(entry) - set(old) <= {"artifact_sha256", "records"}, sorted(
        set(entry) - set(old))
    assert set(old) - set(entry) == set(), sorted(set(old) - set(entry))
    for field in old:
        if field in owned:
            continue
        assert json.dumps(entry[field], sort_keys=True) == json.dumps(
            old[field], sort_keys=True), field

    print(f"  {BIT_IDENTITY_KEY}:", flush=True)
    print(f"    adapter_sha256 {old['adapter_sha256'][:12]} -> "
          f"{entry['adapter_sha256'][:12]}  ({adapter_path})", flush=True)
    print(f"    probe_sha256   {old['probe_sha256'][:12]} -> "
          f"{entry['probe_sha256'][:12]}  ({probe_path})", flush=True)
    print(f"    records        -> apps/api/{relative}", flush=True)
    if write:
        LEDGER.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n",
                          encoding="utf-8")
        print(f"  wrote {LEDGER}", flush=True)
    else:
        print("  (report only -- pass --write to apply)", flush=True)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    # REPEATABLE. One re-gate round is not necessarily one campaign directory: the
    # 2026-08-27 routing round covered fourteen families, and the three families whose
    # gates need a specific expansion probe had to be re-run into their own trees once
    # the right probe was identified (see drive_triton_weld_gates.py's notes on
    # complex_offdiag and complex_fused_ade_chain). Pass --campaign once per tree; the
    # FIRST tree holding a RELEASED artifact for a key wins, so later trees supersede
    # nothing and order is the operator's declaration of precedence.
    parser.add_argument("--campaign", action="append", default=None,
                        help="a campaign directory; repeat to search several in order")
    parser.add_argument("--bind-bit-identity", default=None, metavar="ARTIFACT",
                        help="bind bit_identity_gate's two sibling pins "
                             "(adapter_sha256/probe_sha256) from a RELEASED "
                             "re-run of probe_fused_kernel_bit_identity.py's "
                             "identity legs; mutually exclusive with "
                             "--campaign. See bind_bit_identity for what it "
                             "refuses.")
    parser.add_argument("--only", default="",
                        help="comma-separated ledger keys to rebind; the rest are "
                             "left exactly as they are. WHY THIS EXISTS: a campaign "
                             "tree can hold a family that some OTHER, later tree "
                             "already bound, and rebinding the whole tree would walk "
                             "that key backwards onto the older artifact. Measured "
                             "2026-08-30 on triton_regate_2026-08-28_tranche2, which "
                             "holds 10 families: 1 (fused_ade_chain) carries the "
                             "artifact its entry is bound to and 9 carry DIFFERENT, "
                             "older ones. Repairing that entry's records line without "
                             "this flag would have regressed nine welds.")
    parser.add_argument("--write", action="store_true", help="apply (default: report)")
    args = parser.parse_args(argv)

    if args.bind_bit_identity is not None:
        if args.campaign:
            raise SystemExit("--bind-bit-identity and --campaign are separate "
                             "modes; run them separately")
        return bind_bit_identity(args.bind_bit_identity, args.write)
    if not args.campaign:
        raise SystemExit("one of --campaign or --bind-bit-identity is required")

    campaigns = []
    for raw in args.campaign:
        root = Path(raw)
        if not root.is_absolute():
            root = (_HERE / root).resolve()
        if not root.is_dir():
            raise SystemExit(f"no such campaign directory: {root}")
        campaigns.append(root)
    # IN-TREE ONLY. ``records`` names the run directory as an ``...``
    # path, which is what makes a weld followable from a checkout; a campaign
    # living outside the package root cannot be named that way and would produce
    # a pointer nobody can resolve. Refuse it here rather than crash on the
    # relative path below.
    for root in campaigns:
        if _API not in root.parents:
            raise SystemExit(
                f"campaign {root} lives outside {_API}; a weld's records line must "
                "name an apps/api/... path or it points nowhere resolvable")

    # PER TREE, NOT PER INVOCATION. This used to be a single ``campaign_rel``
    # and a single ``host``, both taken from ``campaigns[0]``, while the artifact
    # was chosen by scanning ALL the trees a few lines below. Whenever those
    # disagreed, the weld got a records line naming a tree the artifact did not
    # come from and a host read off a manifest that did not describe the run.
    #
    # MEASURED 2026-08-30, on the record this tool wrote: two welds carry a
    # records line pointing at
    # ``parity/meep_gpu/results/triton_fused_regate_2026-08-28/<family>/``,
    # a directory that does not exist and never did — that tree holds only the
    # eight ``probe_triton_*`` families. Their artifacts, identified by matching
    # the recorded ``artifact_sha256`` rather than by guessing, are:
    #
    #   triton_fused_ade_chain_device_gate         d03305d8ed85
    #       -> triton_regate_2026-08-28_tranche2/fused_ade_chain/gate.json
    #   triton_complex_fused_ade_chain_device_gate 4176be0a6a8a
    #       -> triton_regate_2026-08-28_tranche6/complex_fused_ade_chain/gate.json
    #
    # The second is the sharper one: tranche2 ALSO holds a
    # complex_fused_ade_chain/gate.json, with a different digest (fee3b33fea32),
    # so a reader following the record to a plausible-looking directory would
    # have found the wrong run rather than no run. Both lines are cleared by
    # re-running this tool against the trees the artifacts are in; they are NOT
    # cleared by editing the strings.
    rel_by_root = {root: root.relative_to(_API).as_posix() for root in campaigns}
    host_by_root = {}
    for root in campaigns:
        host_by_root[root] = ""
        manifest = root / "campaign.json"
        if manifest.is_file():
            loaded = json.loads(manifest.read_text(encoding="utf-8"))
            gpu = loaded.get("gpu")
            host_by_root[root] = (f"{loaded.get('host', '?')}"
                                  + (f" GPU {gpu}" if gpu is not None else ""))

    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    before = json.loads(json.dumps(ledger))     # deep copy, for the survival proof
    rebound, skipped, uncovered = [], [], []
    taken_from_tree: dict = {}   # key -> keys whose digest came from the checkout

    selected = {k for k in args.only.split(",") if k} if args.only else None
    if selected is not None:
        unknown = selected - set(ledger)
        if unknown:
            raise SystemExit(f"--only names {sorted(unknown)}, absent from the ledger")
    for key in sorted(ledger):
        entry = ledger[key]
        if not isinstance(entry, dict) or entry.get("status") != "PASS":
            continue
        if selected is not None and key not in selected:
            continue
        curated = list(entry.get("source_sha256") or {})
        if not curated:
            skipped.append((key, "PASS but pins no source_sha256"))
            continue
        # The entry must weld SHIPPED package bytes. A weld over harness scripts
        # alone certifies nothing a user runs.
        if not any(name.startswith("meep_gpu/") for name in curated):
            skipped.append((key, "pins no path under meep_gpu/"))
            continue
        if key not in CAMPAIGN_DIRS:
            uncovered.append(key)
            continue

        # THE FIRST TREE HOLDING AN ARTIFACT WINS -- not the first holding a
        # DIRECTORY. A gate that failed leaves its directory behind with a log and
        # an empty cases.jsonl and no gate.json, so matching on is_dir() picks the
        # failed run over the good one in a later tree and reports "holds no gate
        # artifact" about a family that was successfully re-gated. Measured on
        # complex_offdiag, whose tranche2 run died on a probe mispairing and whose
        # tranche5 run released.
        directory = None
        chosen_root = None
        for root in campaigns:
            candidate = root / CAMPAIGN_DIRS[key]
            if candidate.is_dir() and any(candidate.glob("*.json")):
                directory, chosen_root = candidate, root
                break
        if directory is None:
            # Nothing held an artifact. Report against the first tree, which is
            # where the operator said to look first; the refusal below is the
            # only thing this branch can produce, so no records line is written
            # from it and the attribution cannot be wrong.
            directory, chosen_root = campaigns[0] / CAMPAIGN_DIRS[key], campaigns[0]
        if not directory.is_dir():
            skipped.append((key, f"the table names {CAMPAIGN_DIRS[key]}/, "
                                 "which this campaign does not contain"))
            continue
        artifact = _artifact_for(directory)
        if artifact is None:
            skipped.append((key, f"{CAMPAIGN_DIRS[key]}/ holds no gate artifact"))
            continue

        fresh = json.loads(artifact.read_text(encoding="utf-8"))
        verdict, source_field = _verdict(fresh)
        if verdict.get("released") is None:
            skipped.append((key, f"{CAMPAIGN_DIRS[key]}/{artifact.name} carries neither "
                                 "canonical_verdict.released nor release.released"))
            continue
        if not verdict["released"]:
            skipped.append((key, f"not released ({source_field}): "
                                 f"{verdict.get('reasons') or 'no reason given'}"))
            continue

        # TWO SPELLINGS, MEASURED NOT ASSUMED TO AGREE. Older probes record every file
        # the run imported as ``imported_source_sha256``; the probes written on
        # 2026-08-31 record only the set the gate PINS, as ``source_sha256``. Measured
        # over the 2026-08-31 campaigns: 32 artifacts carry one or the other, 15 carry
        # BOTH, and on every one of those the two AGREE on every digest they share --
        # zero disagreements. They are NOT the same set (one artifact reads
        # imported=31, source=13, overlap=11), so this is a FALLBACK and never a merge:
        # the curated pin set is used only where the run recorded no import set at all.
        # Conflating them would let a weld claim a digest no gate leg attested.
        imported = fresh.get("imported_source_sha256") or {}
        if not imported:
            imported = fresh.get("source_sha256") or {}
        unknown = [name for name in curated if name not in imported]

        # A WELD MAY PIN A FILE THE GATE NEVER IMPORTS -- its own test file is the
        # normal case, since the gate is the subject and the test is the checker.
        # The artifact therefore CANNOT record it, and refusing outright left six
        # welds permanently unrebindable, two of which had nothing stale at all.
        #
        # The safe rule is not "trust the tree" but "trust the tree ONLY where the
        # run demonstrably executed it". Every digest the artifact DOES record must
        # already equal the checkout; that is the proof this gate ran against these
        # exact bytes, and it is exactly the check the device-digest block below
        # makes for the same reason. Once that holds, the checkout's digest for an
        # unrecorded file is the right value -- it is the same tree, and no other
        # source for it exists. If ANY recorded digest disagrees, the run describes
        # a different tree and nothing here may be taken from the checkout.
        if unknown:
            disagreeing = [name for name, digest in imported.items()
                           if (_API / name).is_file()
                           and hashlib.sha256((_API / name).read_bytes()).hexdigest() != digest]
            if disagreeing:
                skipped.append((key, f"artifact does not record {unknown}, and the tree "
                                     f"cannot stand in for them because the run disagrees "
                                     f"with the checkout on {disagreeing[:3]}"))
                continue
            missing = [name for name in unknown if not (_API / name).is_file()]
            if missing:
                skipped.append((key, f"artifact does not record {missing}, and they are "
                                     "not in the checkout either"))
                continue
            taken_from_tree[key] = list(unknown)

        # DEVICE DIGESTS ARE RECOMPUTED FROM THE TREE, so they are only honest
        # where the tree IS what the run imported. Compare the run's own byte
        # digest for the file against the checkout first and refuse otherwise --
        # otherwise a kernel-body edit made after the run would be re-pinned from
        # a laptop and the weld would go green over bytes nobody measured.
        device_recut, stale = {}, []
        for name in (entry.get("device_sha256") or {}):
            path = _API / name
            live = hashlib.sha256(path.read_bytes()).hexdigest()
            if imported.get(name) != live:
                stale.append(name)
                continue
            kind, digests = device_digests(path)
            device_recut[name] = {"kind": kind, "digests": digests}
        if stale:
            skipped.append((key, f"device_sha256 pins {stale}, whose bytes differ "
                                 "from the ones this run imported"))
            continue

        # THE TREE THE ARTIFACT ACTUALLY CAME FROM, not the first one passed.
        campaign_rel, host = rel_by_root[chosen_root], host_by_root[chosen_root]
        assert (_API / campaign_rel / CAMPAIGN_DIRS[key]) == directory, (
            f"{key}: records would name {campaign_rel}/{CAMPAIGN_DIRS[key]} but "
            f"the artifact was read from {directory}")
        records = (f"apps/api/{campaign_rel}/{CAMPAIGN_DIRS[key]}/ - {source_field}."
                   f"released=true (read_from {verdict.get('read_from', '?')}); "
                   f"{len(imported)} imported digests, every pinned path taken from "
                   f"the run's own imported_source_sha256"
                   + (f"; {host}" if host else "") + ".")
        sentinels = RECORDS_SENTINELS.get(key)
        if sentinels:
            previous = entry.get("records") or ""
            records = f"{records} PRIOR RECORD, NOT SUPERSEDED: {previous}"
            if not any(token in records for token in sentinels):
                skipped.append((key, f"the records carve-out lost {sentinels}"))
                continue

        # THE CONDITIONAL HOST, resolved BEFORE the first mutation of ``entry``.
        # Every refusal above this point leaves the entry untouched, and this one
        # must too: ``ledger`` is the object that gets written, so an entry that
        # is half-rewritten and then skipped would be PERSISTED in that state the
        # moment any other key binds. It fails CLOSED -- an entry that says it is
        # waiting for a host and gets an artifact that cannot supply one is
        # skipped, keeping its unbound digests and the weld tests red, rather than
        # being bound with a host nobody measured.
        derived_host = None
        if entry.get("host") == HOST_PENDING:
            derived_host = _host_line(fresh)
            if derived_host is None:
                skipped.append((key, f"host is {HOST_PENDING!r} and "
                                     f"{CAMPAIGN_DIRS[key]}/{artifact.name} carries no "
                                     "environment block naming hostname, device, "
                                     "compute_capability, triton and cupy"))
                continue
        # THE CONDITIONAL POLICY, resolved here for the same reason the host is: before
        # the first mutation, and failing closed. An entry that points at its artifact
        # for a policy the artifact does not name keeps its unbound digests and stays
        # red, rather than being rebound with the debt still in it.
        derived_policy = None
        if entry.get("subnormal_policy") == POLICY_UNNAMED:
            from seed_triton_welds import policy_line  # noqa: PLC0415
            derived_policy = policy_line(fresh)
            if derived_policy is None:
                skipped.append((key, f"subnormal_policy is {POLICY_UNNAMED!r} and "
                                     f"{CAMPAIGN_DIRS[key]}/{artifact.name} carries no "
                                     "subnormal_policy stamp naming a resolved policy"))
                continue

        # Keys the artifact records come FROM THE ARTIFACT. The handful it cannot
        # record (see the tree-fallback rule above) come from the checkout, which
        # the guard has already proved is the tree this run executed.
        entry["source_sha256"] = {
            name: (imported[name] if name in imported
                   else hashlib.sha256((_API / name).read_bytes()).hexdigest())
            for name in curated}
        if taken_from_tree.get(key):
            entry["_digests_taken_from_the_checkout"] = (
                f"{sorted(taken_from_tree[key])} are pinned by this weld but are not in "
                f"the gate's imported_source_sha256 -- a gate does not import its own "
                f"test file. Their digests are the CHECKOUT's, which is licensed here "
                f"only because every digest the run DID record already matched the "
                f"checkout byte for byte, so the run and the checkout are the same tree.")
        entry["code_sha256"] = {name: code_digest_of_path(_API / name)
                                for name in curated}
        if device_recut:
            entry["device_sha256"] = device_recut
        if derived_host is not None:
            entry["host"] = derived_host
        if derived_policy is not None:
            entry["subnormal_policy"] = derived_policy
        entry["artifact_sha256"] = hashlib.sha256(artifact.read_bytes()).hexdigest()
        entry["records"] = records
        entry["recorded_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        entry["status"] = "PASS"
        entry["verdict_read_from"] = (
            f"{CAMPAIGN_DIRS[key]}/{artifact.name}:{source_field}.released"
            + (f" (read_from={verdict['read_from']})" if verdict.get("read_from") else ""))

        # THE SURVIVAL PROOF, in the tool rather than in a reviewer's head. The
        # key set may not move, and every field this tool does not own must come
        # out byte-identical to what went in.
        old = before[key]
        assert set(entry) - set(old) <= {"verdict_read_from",
                                         "_digests_taken_from_the_checkout"}, \
                (key, sorted(set(entry) - set(old)))
        assert set(old) - set(entry) == set(), (key, sorted(set(old) - set(entry)))
        assert set(entry["source_sha256"]) == set(old["source_sha256"]), key
        for field in old:
            if field in REFRESHED:
                continue
            assert json.dumps(entry[field], sort_keys=True) == json.dumps(
                old[field], sort_keys=True), (key, field)
        if derived_policy is None:
            assert entry.get("subnormal_policy") == old.get("subnormal_policy"), (
                key, "subnormal_policy")
        rebound.append((key, rel_by_root[chosen_root]))

    # THE REPORT NAMES THE TREE, not just the family directory. With several
    # --campaign trees in play the family name alone does not say which run bound
    # the key, so the two misattributed records lines this tool wrote on
    # 2026-08-28 were invisible in its own output. Printing the tree makes the
    # attribution reviewable before --write rather than after.
    for key, tree in rebound:
        print(f"  rebound   {key}  <- {tree}/{CAMPAIGN_DIRS[key]}/", flush=True)
    for key, why in skipped:
        print(f"  SKIPPED   {key}: {why}", flush=True)
    for key in uncovered:
        print(f"  UNCOVERED {key}: no row in CAMPAIGN_DIRS -- needs a device re-run, "
              "not a guessed directory", flush=True)
    print(f"\n  {len(rebound)} rebound, {len(skipped)} skipped, "
          f"{len(uncovered)} PASS entries the table does not cover", flush=True)

    if args.write and rebound:
        LEDGER.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n",
                          encoding="utf-8")
        print(f"  wrote {LEDGER}", flush=True)
    elif not args.write:
        print("  (report only -- pass --write to apply)", flush=True)
    return 0 if not (skipped or uncovered) else 1


if __name__ == "__main__":
    raise SystemExit(main())
