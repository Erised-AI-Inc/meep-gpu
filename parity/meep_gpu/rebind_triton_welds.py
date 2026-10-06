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
(``recert_*``, ``probe_sha256``, ``real_engine_route``,
``standalone_product_gate``, the ``_why``/``_dispatch_note`` prose, ...) is
asserted byte-identical after the rewrite, and an entry the tool does not
understand is skipped rather than reshaped.

ONE RECORD PER COMPUTE CAPABILITY (2026-09-30). The digests stay beside the
entry; the facts about the RUN -- its artifact, its records line, its timestamp,
its host, its toolchain and its subnormal policy -- go into
``runs[<capability>]``, and the capability is READ OFF THE RUN
(:func:`run_capability`), never passed in. That makes a second architecture
additive: a re-gate on unchanged bytes adds a record beside the ones already
there, and the admitted set a table dispatches on is the intersection of the
capabilities its cited welds all have a LIVE record for. A rebind on bytes that
MOVED cannot be additive -- it leaves the other architectures' records
certifying bytes that no longer ship -- so it refuses and names them, and
``--supersede`` is how a round says those certifications are being dropped.

    python rebind_triton_welds.py --campaign results/triton_regate_2026-08-27_routing
    python rebind_triton_welds.py --campaign results/triton_regate_2026-08-27_routing --write
    python rebind_triton_welds.py --campaign results/<fresh> --supersede 8.6 --write
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

from meep_gpu import fastpath                            # noqa: E402
from meep_gpu.code_identity import code_digest_of_path  # noqa: E402
from meep_gpu.device_identity import device_digests      # noqa: E402

LEDGER = _API / "meep_gpu" / "triton_kernels" / "fingerprints.json"

#: The device stamp a run leaves in its own directory, written by
#: ``triton_device_identity.py --write <dir>`` in the run's environment.
#:
#: WHY A SEPARATE FILE IS READ AT ALL. Measured across the 2026-09-25 fleet: of the
#: artifacts this tool binds, several carry no ``environment`` block whatsoever
#: (``complex_no_pml_curl/gate.json`` and ``complex_no_pml_conductive/gate.json``
#: record ``environment: null``) and others carry a partial one -- the
#: ``complex_fused_ade_chain`` gate records ``device_name``, ``cupy`` and ``triton``
#: and no ``compute_capability`` or ``hostname``. A record kept PER COMPUTE
#: CAPABILITY cannot be written from an artifact that never says which one it ran
#: on, and the alternative is a person typing the architecture beside the digests,
#: which is the one thing this tool exists not to do. The campaign writes the stamp
#: once at its root, so the file is looked for in the family directory first and the
#: campaign root second.
DEVICE_STAMP = "device.json"

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

    # THE TWO FLEET FAMILIES THE LEDGER NEVER CARRIED, 2026-09-30. Seven of the nine
    # families whose gates the fleet runs have a ``triton_<family>_device_gate`` row
    # above; these two have no ledger entry at all, so a round that runs their gates
    # had nothing to bind them to. They are SEEDED by ``seed_triton_welds.py`` from the
    # round's own fleet; until that runs, a row here for a key the ledger does not
    # carry reports UNCOVERED and binds nothing, which is the correct reading and the
    # right place to be caught if the seed never happens. Their fleet subdirectory is
    # the gate name in ``drive_triton_weld_gates.GATES``, as for the 2026-09-11 block
    # above.
    #
    # THE 22 ARMS THAT CITE ``family_recert_2026-08-14`` STILL CITE IT. Re-pointing
    # them here was considered and reversed on 2026-09-30: the reason to re-point was
    # that a writer rebuilding that entry's nine family records would refuse whenever a
    # family gate states no step budget (three of the nine state none, two state a
    # singular key), but ``recut_composition_records.py --family-recert`` records what
    # the artifact states and an explicit "not stated" otherwise, which
    # ``fastpath._certification_for`` already handles for every other absent budget. So
    # the entry stays bindable in every round and no arm changed what certifies it.
    "triton_bfast_device_gate": "bfast",
    "triton_special_kz_device_gate": "special_kz",
}

#: The ENTRY-level fields this tool owns -- the digests that say which BYTES the
#: weld certifies, plus the status that admits it. Everything else outside
#: ``fastpath.RUNS`` survives untouched, and the survival proof below shows it did.
#:
#: SPLIT FROM ``REFRESHED`` 2026-09-30, when the run's own facts moved into
#: ``runs[<capability>]``. The old single tuple could not express the rule any more:
#: a field inside the capability's record is not "immovable" and not "owned at entry
#: level" either -- the record is REPLACED wholesale by
#: :func:`fastpath.bind_capability`, and what the proof has to assert instead is that
#: every OTHER capability's record came out byte-identical.
ENTRY_REFRESHED = ("source_sha256", "code_sha256", "device_sha256", "status")

#: The fields of the capability's own record this tool writes. Every one is DERIVED
#: from the run being bound -- there is no carry-forward branch left (the
#: ``HOST_PENDING``/``POLICY_UNNAMED`` sentinels that used to gate the host and the
#: policy were deleted in the same change). A run that cannot answer one of them is
#: SKIPPED by name, so a record never describes a machine or a toolchain that no run
#: reported.
SLOT_FIELDS = ("artifact_sha256", "records", "recorded_utc", "verdict_read_from",
               "host", "subnormal_policy", "cupy_version", "triton_version",
               "_digests_taken_from_the_checkout", "step_budget")

#: The one field carried from the record being REPLACED, and only within the same
#: capability. 5 of the 55 PASS entries carry a ``step_budget``
#: (``triton_{complex_fused_ade_chain,complex_no_pml_curl,fused_ade_chain,
#: no_pml_conductive,no_pml_stored_e}_device_gate``) and ``fastpath.py`` quotes it
#: into a plan's certification block, so dropping it on a re-bind would delete
#: something a reader is shown.
#:
#: WHY CARRYING IT IS NOT THE DEFECT ``HOST_PENDING`` WAS. Those strings are curated
#: accounts of what the gate MEASURES -- "8 launches per row; 8 product rows,
#: byte-identical over 49,152 compared uint32 words; 3 of 3 mutations divergent" --
#: and no artifact states them in that form: the gates write a ``step_budgets`` map
#: per case, not this summary. They describe the gate's structure, which a re-run of
#: the same gate reproduces, and they are not a label about the machine or the
#: toolchain, which is the class this tool now refuses to carry. A FIRST record for
#: another architecture does not inherit one: it was written about a run on a
#: different device, and ``fastpath`` names the absence rather than quoting it.
CARRIED_WITHIN_CAPABILITY = ("step_budget",)


def carried_run_facts(entry: dict, capability: str, fields) -> dict:
    """The facts a rebind may carry from ``capability``'s previous record: only ``fields``,
    and only while that record is LIVE on the bytes being bound now.

    Call it after the entry's digests are refreshed from the run and before
    :func:`fastpath.bind_capability` writes the new record, so ``live_capabilities``
    compares the previous record's ``bound_sha256`` with the bytes this run certified.

    IT USED TO ASK WHETHER THE BYTES MOVED IN THIS PASS, which is a different question
    once a second architecture exists. Bind 9.0 first (superseding 8.6) and then re-bind
    8.6 on the same commit: the 8.6 pass moves nothing, yet the 8.6 record it would
    carry from was bound to the PREVIOUS bytes, so a measurement of other code landed in
    the new record and the two binding orders left different ledgers. Asking whether the
    previous record is live answers the question the carry depends on in every order,
    and agrees with the old rule wherever the old rule was right. It is the rule
    ``recut_driver_dispatch_record.py`` already applies to the licence it keeps.
    """
    if capability not in fastpath.live_capabilities(entry):
        return {}
    previous = entry[fastpath.RUNS][capability]
    return {field: previous[field] for field in fields
            if previous.get(field) is not None}

#: RECORDS CARVE-OUT. ``test_triton_folded_fused_pair.py:672`` asserts the weld's
#: ``records`` line still names ``run_farcarryD5``/``run_farcarryD6`` -- the runs
#: that measured ``fill_folded_far_ghosts_D`` carried inline and released it.
#: That release is EARLIER than any re-pin and is not superseded by one, so a
#: generated ``records`` line would delete a standing claim rather than refresh
#: it. Where a sentinel is listed, the old line is kept and the fresh one is
#: appended; if the sentinel is not in the result the entry is skipped instead.
#:
#: IT READS THE CAPABILITY'S OWN RECORD, not the entry, and only where that record
#: exists. The claim being preserved was made by the 8.6 run; appending 8.6's line to
#: a FIRST record for another architecture would make that record quote a run on a
#: different device, and refusing instead would hold the admitted set where it is --
#: this key is one of the 44 the Triton arms cite, so no round could widen it. A
#: first record is written clean; every re-bind of a capability that has one carries.
RECORDS_SENTINELS = {
    "triton_folded_fused_pair_device_gate": ("run_farcarryD5", "run_farcarryD6"),
}


#: The alternative key spellings an identity block is written in, mapped onto the
#: one this tool reads. MEASURED, not defensive: ``device_name``/``cupy_version``/
#: ``CUDA_VISIBLE_DEVICES`` are ``probe_fused_kernel_bit_identity.device_info``'s
#: names, ``host`` and ``cc`` appear across the 2026-09-09 fleet artifacts whose
#: ``environment`` blocks were authored per probe, and ``device``/``cupy``/``triton``
#: are what the Triton probes and :data:`DEVICE_STAMP` write.
#:
#: NORMALISING BEFORE THE MERGE IS WHAT MAKES PRECEDENCE MEAN ANYTHING. Merging raw
#: and then reading ``device or device_name`` inverts it: a sibling device stamp's
#: ``device`` wins over the artifact's own ``device_name``, so the weld's host line
#: would name the stamp's card while the run's own block named another. Measured on a
#: fixture where the two deliberately disagreed -- the line read the stamp's device.
IDENTITY_SPELLINGS = {"host": "hostname", "device_name": "device", "cc":
                      "compute_capability", "cupy_version": "cupy",
                      "triton_version": "triton",
                      "CUDA_VISIBLE_DEVICES": "cuda_visible_devices"}


def _identity_sources(fresh: dict,
                      dirs) -> tuple[list[tuple[str, dict]], list[str]]:
    """Every block in or beside this run that reports which device it used.

    Returns the blocks, canonically spelled, and the reasons any candidate could not
    be read at all.

    FOUR SOURCES, MEASURED NOT ASSUMED. ``environment`` is what the Triton probes
    write (through :mod:`triton_device_identity`); ``device_info`` is the
    bit-identity probe's own block name; ``provenance.device`` is what
    ``gate_provenance.stamp()`` writes; and :data:`DEVICE_STAMP` is the file a
    campaign leaves beside its artifacts for the gates that write none of the three.
    They are returned in that order because the earlier a block is, the closer it is
    to the process that ran the kernels: the artifact's own stamp beats a sibling
    file written afterwards, and the caller's merge keeps the first answer.
    """
    found: list[tuple[str, dict]] = []
    unreadable: list[str] = []

    def canonical(block: dict) -> dict:
        out = {}
        for key, value in block.items():
            out.setdefault(IDENTITY_SPELLINGS.get(key, key), value)
        return out

    for name in ("environment", "device_info"):
        block = fresh.get(name)
        if isinstance(block, dict):
            found.append((name, canonical(block)))
    provenance = fresh.get("provenance")
    if isinstance(provenance, dict) and isinstance(provenance.get("device"), dict):
        found.append(("provenance.device", canonical(provenance["device"])))
    for directory in dirs:
        stamp = Path(directory) / DEVICE_STAMP
        if not stamp.is_file():
            continue
        where = f"{DEVICE_STAMP} in {Path(directory).name}"
        # A STAMP THAT CANNOT BE PARSED IS A NAMED REFUSAL, not a source to skip.
        # Ignoring it would let a truncated or half-written stamp -- the shape a
        # campaign killed mid-write leaves -- pass as though the run had left none,
        # and the capability would then be decided by whatever else happened to
        # answer.
        try:
            loaded = json.loads(stamp.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            unreadable.append(f"{where} cannot be read: {exc!r}")
            continue
        if not isinstance(loaded, dict):
            unreadable.append(f"{where} is {type(loaded).__name__}, not an object")
            continue
        found.append((where, canonical(loaded)))
    return found, unreadable


def run_capability(fresh: dict, dirs) -> tuple[str | None, object]:
    """Which compute capability THIS RUN used, and the identity that reported it.

    Returns ``(capability, identity)`` or ``(None, reason)``. The identity is one
    merged block, canonically spelled (:data:`IDENTITY_SPELLINGS`), which is what
    :func:`_host_line` composes a weld's ``host`` from; the merge is ``setdefault``
    in the order :func:`_identity_sources` returns, so the artifact's own stamp wins
    key by key and a sibling :data:`DEVICE_STAMP` only fills what the gate left out.

    WHY THE CAPABILITY IS READ AND NEVER PASSED IN. A record kept per compute
    capability is only worth keeping if the key is a measurement: a tool that took
    the architecture from its command line would let one typed flag file a run on one
    device under another, and every digest in the entry would still verify. So the
    architecture comes out of the run, every source that names one must AGREE, and a
    run that names none is refused by name rather than defaulted to the one this
    ledger happens to already carry.

    THE HOSTNAME IS CROSS-CHECKED FOR THE SAME REASON. A :data:`DEVICE_STAMP` is
    written by a separate process, so it is weaker evidence than an in-process stamp;
    what makes it usable is that it cannot silently describe a different machine than
    the artifact beside it. Where two sources both name a hostname they must match.
    The remaining identity keys are not compared -- the device name, the CuPy and the
    Triton version are all facts about the same process in practice, and the one that
    decides the weld, the capability, is the one checked.

    WHY IT LIVES HERE AND NOT IN :mod:`triton_device_identity`. That module is
    stdlib-only on purpose: it is imported by a gate that may already have refused,
    possibly because ``meep_gpu`` itself would not import, and it WRITES identity at
    run time. Reading a finished artifact back, normalising it through the engine's
    own rule and deciding whether a weld may be bound from it are this tool's job.
    """
    sources, unreadable = _identity_sources(fresh, dirs)
    if unreadable:
        return None, "; ".join(unreadable)
    capabilities: dict[str, list[str]] = {}
    hostnames: dict[str, list[str]] = {}
    identity: dict = {}
    for name, block in sources:
        raw = block.get("compute_capability")
        if raw:
            capabilities.setdefault(fastpath._normalized_capability(raw), []).append(name)
        host = block.get("hostname")
        if isinstance(host, str) and host:
            hostnames.setdefault(host, []).append(name)
        for key, value in block.items():
            if value is not None:
                identity.setdefault(key, value)
    if not capabilities:
        return None, ("no block in or beside this run names a compute capability "
                      f"(looked at environment, device_info, provenance.device and a "
                      f"{DEVICE_STAMP} beside the artifact); a per-capability record "
                      f"may not be filed under an architecture nobody measured")
    if len(capabilities) > 1:
        return None, ("this run reports two compute capabilities -- "
                      + "; ".join(f"{cc} from {sorted(where)}"
                                  for cc, where in sorted(capabilities.items()))
                      + " -- so which architecture it certifies is not decidable")
    if len(hostnames) > 1:
        return None, ("this run reports two hostnames -- "
                      + "; ".join(f"{host!r} from {sorted(where)}"
                                  for host, where in sorted(hostnames.items()))
                      + " -- so the device stamp may not describe this artifact's run")
    capability = next(iter(capabilities))
    if not capability:
        return None, "the reported compute capability normalises to the empty string"
    identity["compute_capability"] = capability
    return capability, identity


def _host_line(identity: dict) -> str | None:
    """The weld's ``host``, built from the identity THIS RUN reported.

    Returns ``None`` when the run cannot answer, and the caller then SKIPS the entry
    rather than binding it with a half-known host. The four parts are the four an
    existing weld's host string carries, and each is load-bearing rather than
    decorative:

    * the machine, so a later reader can go back to it;
    * the device AND its compute capability -- ``test_triton_weld_contract.py``
      binds both to the record's ``validated_triton_versions`` and to the admitted
      capabilities, because Triton generates PTX for an ARCHITECTURE and a weld cut
      on an unvalidated one must fail until the declaration is widened deliberately;
    * the Triton and CuPy versions, for the same reason;
    * the physical GPU index, which is what makes a placement claim checkable.

    ONE SPELLING FOR BOTH TOOLS. ``seed_triton_welds.py`` calls this rather than
    composing its own line: a seeded weld and a rebound one must be
    indistinguishable, and two f-strings drift.

    IT TAKES AN IDENTITY, NOT AN ARTIFACT (2026-09-30). The host used to be composed
    from ``fresh["environment"]`` alone, which is why the welds whose gates write no
    environment block could never have one derived and kept whatever string they
    were authored with. The caller merges every block the run left
    (:func:`run_capability`) and hands the result here, so one rule serves a probe
    that stamps itself and a gate that only has a sibling device stamp. The merge has
    already put every measured spelling onto one name
    (:data:`IDENTITY_SPELLINGS`), which is why this reads single keys: a second
    spelling read HERE would silently reverse the merge's precedence.
    """
    host = identity.get("hostname")
    device = identity.get("device")
    capability = fastpath._normalized_capability(
        identity.get("compute_capability") or "")
    triton = identity.get("triton")
    cupy = identity.get("cupy")
    if not (host and device and capability and triton and cupy):
        return None
    # A VERSION, NOT A REPORT OF ITS ABSENCE. A probe whose `environment()` fell to
    # its except branch writes `cupy = "unavailable: TypeError(...)"`, which is
    # truthy and would compose a host line reading "CuPy unavailable: TypeError(...)"
    # -- a weld naming an exception where its toolchain belongs, and one the contract
    # test would pass because it greps only for the Triton and cc substrings.
    for version in (cupy, triton):
        text = str(version)
        if not text[:1].isdigit() or "unavailable" in text or "Error" in text:
            return None
    gpu = (identity.get("cuda_visible_devices")
           or (identity.get("env") or {}).get("CUDA_VISIBLE_DEVICES"))
    return (f"{host}: {device}, cc {capability}; CuPy {cupy}; Triton {triton}"
            + (f" (GPU {gpu})" if gpu else ""))


def recorded_cupy_versions(ledger: dict) -> tuple[str, ...]:
    """The CuPy versions this ledger's live records were cut on. DERIVED, never typed.

    THE OTHER HALF OF THE TOOLCHAIN AXIS. ``fastpath.validated_triton_versions()``
    declares the Triton versions the table stands on, and the weld contract holds
    every record's ``host`` string to it -- but nothing held the RUN to it, so a run
    on another Triton or another CuPy could be bound and labelled with whatever the
    entry already said. There is no CuPy counterpart of that declaration to read, so
    it is derived from the records themselves: the versions the live records of this
    ledger name, which measures ``('13.5.1',)`` across 56 records today.

    An empty answer is a refusal and not a licence: a ledger that declares no CuPy
    gives this tool nothing to check a run against, and widening the toolchain is a
    deliberate act in both directions.
    """
    found = set()
    for entry in ledger.values():
        if not isinstance(entry, dict):
            continue
        for capability in fastpath.live_capabilities(entry):
            record = entry[fastpath.RUNS][capability]
            if record.get("cupy_version"):
                found.add(str(record["cupy_version"]))
            for token in str(record.get("host") or "").split(";"):
                token = token.strip()
                if token.startswith("CuPy "):
                    found.add(token[len("CuPy "):].strip())
    return tuple(sorted(found))


def toolchain_refusal(identity: dict, ledger: dict) -> str | None:
    """Why this run's toolchain may not be welded, or ``None``. Checked at BIND time.

    THE DEFECT THIS CLOSES, measured 2026-09-29: a weld's ``host`` string was the
    only place the Triton and CuPy versions were recorded, and the only thing that
    ever compared them with the declaration was a contract test reading that string.
    Since a rebind carried the string forward rather than deriving it, a run on
    Triton 3.2 could be bound into an entry labelled 3.1.0 and the test would pass on
    the label. Deriving the label fixes the labelling; refusing here is what stops
    the run from being welded at all.
    """
    triton = str(identity.get("triton") or "")
    cupy = str(identity.get("cupy") or "")
    validated = fastpath.validated_triton_versions()
    if not validated:
        return ("the ledger declares no validated Triton version, so this run's "
                "toolchain cannot be checked against one")
    if triton not in validated:
        return (f"the run used Triton {triton!r}, which is outside the record's "
                f"validated_triton_versions {list(validated)}; Triton generates the "
                f"PTX this weld is a claim about, so widening that declaration is a "
                f"deliberate act and not a side effect of a rebind")
    recorded = recorded_cupy_versions(ledger)
    if not recorded:
        return ("this ledger's live records name no CuPy version, so this run's "
                "CuPy cannot be checked against the one the table stands on")
    if cupy not in recorded:
        return (f"the run used CuPy {cupy!r}, which is outside the {list(recorded)} "
                f"this ledger's live records were cut on; the compiler that built "
                f"these kernels is part of the claim")
    return None


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


def bind_bit_identity(artifact_argument: str, write: bool,
                      supersede: tuple[str, ...] = ()) -> int:
    """Refresh ``bit_identity_gate``'s pins from a RELEASED re-run, per capability.

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

    THE KERNEL BYTES ARE BOUND TOO, since 2026-09-30. This entry certifies the
    ``PML``, ``ordinary``, ``nonlinear run PML`` and ``nonlinear run`` arms
    (``fastpath.ARM_CERTIFICATION``), and until this change its bound set was the
    harness alone — ``probe``/``probe_sha256`` and ``adapter``/``adapter_sha256``.
    ``kernels.py`` was pinned only at the ledger's TOP level, outside every entry, so
    a per-capability record stayed live across a kernel edit: after an edit and a
    re-gate on one architecture, another architecture's record would still read as
    live while resting on an identity run of the PREVIOUS kernels. So the entry now
    carries a ``source_sha256`` over the run's own ``meep_gpu/`` imports, which is
    what ``fastpath.bound_digest`` hashes. The adapter does import ``kernels.py``
    (``track_triton_pml.py:49``), so these are bytes the run demonstrably executed,
    and the digest guard above is what licenses writing them.

    ``adapter_sha256``, ``probe_sha256`` and ``source_sha256`` are refreshed from the
    artifact's own import record. The facts about the RUN — the artifact digest, the
    records line, the timestamp, the host, the toolchain and the policy — go into
    ``runs[<capability>]``, every one of them DERIVED from this run rather than
    carried from the record it replaces: a re-bind used to leave the host string, the
    CuPy and the Triton version exactly as they were, which made a record's toolchain
    a label nothing checked. The six curated legs and the mutation tables are NOT
    rewritten: they are the 2026-08-09/10 campaign's measurements and remain
    attributed to it — what this binds is that the CURRENT harness and kernel bytes
    reproduce the released identity verdict those legs curate.
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
    retired = fastpath.retired_shape_reasons(entry)
    if retired:
        raise SystemExit(
            f"{BIT_IDENTITY_KEY} is not in the per-capability shape "
            f"({'; '.join(retired)}); run parity/meep_gpu/"
            f"migrate_capability_records.py before binding a run into it")
    if "validated_compute_capabilities" in ledger:
        raise SystemExit(
            "the ledger still carries the typed validated_compute_capabilities key; "
            "the admitted set is DERIVED from the records this tool writes now, and "
            "a typed one beside them is a second answer to the same question")

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

    # THE KERNEL BYTES THIS RUN EXECUTED. Every ``meep_gpu/`` path the probe
    # imported, which is what the bound hashes; a path the run recorded and the
    # checkout does not have would pin a digest for bytes that do not ship.
    package = sorted(name for name in imported if name.startswith("meep_gpu/"))
    if not package:
        raise SystemExit(
            f"{artifact.name} records no import under meep_gpu/; this entry "
            f"certifies the PML and ordinary arms and may not bind a harness alone")
    absent = [name for name in package if not (_API / name).is_file()]
    if absent:
        raise SystemExit(
            f"the run imported {absent[:5]}, which this checkout does not have; "
            f"a pin may not name bytes that do not ship")

    # THE RUN'S OWN DEVICE AND TOOLCHAIN, resolved BEFORE the first mutation of the
    # entry so every refusal leaves the ledger as it was.
    capability, identity = run_capability(fresh, [artifact.parent])
    if capability is None:
        raise SystemExit(f"{artifact.name}: {identity}")
    host = _host_line(identity)
    if host is None:
        raise SystemExit(
            f"{artifact.name} and the {DEVICE_STAMP} beside it cannot compose a host "
            f"line (hostname / device / compute_capability / triton / cupy must all "
            f"be present); the probe's device_info records no hostname and no Triton "
            f"version, so this run needs a {DEVICE_STAMP} in its own directory")
    refused = toolchain_refusal(identity, ledger)
    if refused is not None:
        raise SystemExit(f"{artifact.name}: {refused}")
    from seed_triton_welds import PolicyStampConflict, policy_line  # noqa: PLC0415
    try:
        policy = policy_line(fresh, entry)
    except PolicyStampConflict as exc:
        raise SystemExit(f"{artifact.name}: policy refused: {exc}") from None
    if policy is None:
        raise SystemExit(
            f"{artifact.name} carries no subnormal_policy stamp naming a resolved "
            f"policy; a record that does not say `keep` or `flush` does not identify "
            f"its own result")

    relative = artifact.relative_to(_API).as_posix()
    bound_before = fastpath.bound_digest(entry)
    entry["adapter_sha256"] = imported[adapter_path]
    entry["probe_sha256"] = imported[probe_path]
    entry["source_sha256"] = {name: imported[name] for name in package}
    run = {
        "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "records": (
            f"apps/api/{relative} - {source_field}.released=true "
            f"(read_from {verdict.get('read_from', '?')}); the identity-leg "
            f"families {sorted(BIT_IDENTITY_EXPERIMENTS)} re-run against the "
            f"current tree through the Triton adapter, {len(imported)} imported "
            f"digests all matching this checkout. The six curated legs and the "
            f"mutation tables remain the 2026-08-09/10 campaign's measurements "
            f"(results/triton_fusedpair_2026-08-09/); this run is what binds the "
            f"CURRENT probe, adapter and meep_gpu/ bytes to that released verdict."),
        "recorded_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "host": host,
        # ONE SPELLING EACH: the merge canonicalised them, and _host_line returning
        # a line is the proof both are present.
        "cupy_version": str(identity["cupy"]),
        "triton_version": str(identity["triton"]),
        "subnormal_policy": policy,
    }
    # THE ONE FIELD CARRIED, AND ONLY WITHIN ONE CAPABILITY. ``_this_recut`` is the
    # curated account of WHAT this certification covers -- which kernel bodies were
    # unchanged when the D/E pair was added, and why num_warps is a gate axis. It is
    # not a label about the machine, which is the class of field this tool now
    # refuses to carry; and it belongs to the architecture it was written about, so a
    # FIRST record for another one is written without it rather than inheriting a
    # narrative of a run on a different device.
    # ...and only while the bytes are the same: a narrative about which kernel bodies
    # were unchanged is a statement about one tree, so an edit to the pinned files
    # drops it rather than carrying it onto a record about different bytes.
    run.update(carried_run_facts(entry, capability, ("_this_recut",)))
    # ``device_policy`` is NOT carried. It read "one device
    # (CUDA_VISIBLE_DEVICES=6), every leg serial": the first half is the GPU mask,
    # which the host line above now names from the run itself, and the second is a
    # property of the probe's own structure that the artifact nowhere states -- so
    # restating it would be this tool authoring a measurement.
    try:
        staled = fastpath.bind_capability(
            entry, bound_before=bound_before, capability=capability, run=run,
            supersede=supersede)
    except fastpath.CapabilityRecordError as exc:
        raise SystemExit(f"{BIT_IDENTITY_KEY}: {exc}") from None

    # THE SURVIVAL PROOF: only the fields this binder owns may move, and no OTHER
    # capability's record may move at all unless it was superseded by name.
    owned = {"adapter_sha256", "probe_sha256", "source_sha256", fastpath.RUNS}
    old = before[BIT_IDENTITY_KEY]
    assert set(entry) - set(old) <= {"source_sha256"}, sorted(set(entry) - set(old))
    assert set(old) - set(entry) == set(), sorted(set(old) - set(entry))
    for field in old:
        if field in owned:
            continue
        assert json.dumps(entry[field], sort_keys=True) == json.dumps(
            old[field], sort_keys=True), field
    old_runs = old.get(fastpath.RUNS) or {}
    for name, record in old_runs.items():
        if name in (capability, *staled):
            continue
        assert json.dumps(entry[fastpath.RUNS][name], sort_keys=True) == json.dumps(
            record, sort_keys=True), ("other capability moved", name)

    print(f"  {BIT_IDENTITY_KEY}:  runs[{capability}]", flush=True)
    print(f"    adapter_sha256 {old['adapter_sha256'][:12]} -> "
          f"{entry['adapter_sha256'][:12]}  ({adapter_path})", flush=True)
    print(f"    probe_sha256   {old['probe_sha256'][:12]} -> "
          f"{entry['probe_sha256'][:12]}  ({probe_path})", flush=True)
    print(f"    source_sha256  {len(package)} paths under meep_gpu/", flush=True)
    print(f"    host           {host}", flush=True)
    print(f"    records        -> apps/api/{relative}", flush=True)
    if staled:
        print(f"    SUPERSEDED     {list(staled)} -- their records now bind bytes "
              f"that no longer ship", flush=True)
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
    parser.add_argument("--supersede", default="",
                        help="comma-separated compute capabilities whose records "
                             "this round DISCARDS. WHY IT IS NEEDED AND WHY IT IS "
                             "NOT A DEFAULT: a record is live only while the bytes "
                             "the entry binds are unchanged, so a rebind on MOVED "
                             "bytes leaves every other capability's record "
                             "certifying bytes that no longer ship. That is a "
                             "decision about evidence, not a detail of a rewrite, so "
                             "the tool refuses and names them; listing one here says "
                             "it is understood that the architecture loses its "
                             "certification until it is re-run. A rebind on "
                             "UNCHANGED bytes needs none of this and is additive.")
    parser.add_argument("--write", action="store_true", help="apply (default: report)")
    args = parser.parse_args(argv)
    supersede = tuple(s.strip() for s in args.supersede.split(",") if s.strip())

    if args.bind_bit_identity is not None:
        if args.campaign:
            raise SystemExit("--bind-bit-identity and --campaign are separate "
                             "modes; run them separately")
        return bind_bit_identity(args.bind_bit_identity, args.write, supersede)
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
    # THE TYPED DECLARATION IS GONE AND MAY NOT COME BACK. The capabilities this
    # table admits are the intersection of the capabilities its cited welds have a
    # LIVE record for (``fastpath.capability_admission``), so they are now a
    # consequence of what this tool writes. A typed key beside those records is a
    # second answer to the same question, and the one that used to be there is the
    # defect: no tool wrote it, so a round on another architecture changed every
    # weld's evidence and left the declaration saying what it had always said.
    if "validated_compute_capabilities" in ledger:
        raise SystemExit(
            "the ledger still carries the typed validated_compute_capabilities key; "
            "run parity/meep_gpu/migrate_capability_records.py first")
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
        # THE SHAPE IS REFUSED BY NAME, not read both ways. An entry still carrying
        # its run's facts beside the digests would make "which bytes did this run
        # certify" a question with two answers, which is the whole point of the
        # container; and a tool that accepted both shapes would quietly write a
        # record into an entry nothing reads a record from.
        #
        # ASKED ONLY OF THE KEYS THIS TOOL OWNS, which is why it sits after the
        # table lookup rather than before it. The ledger also holds entries that are
        # NOT welds -- the 08-13/14 planner sweeps and ``pending_host_recut`` carry a
        # ``host`` and a ``recorded_utc`` at their top level and no gate behind them
        # -- and a shape question asked of those would report a defect in something
        # this container was never about.
        retired = fastpath.retired_shape_reasons(entry)
        if retired:
            skipped.append((key, f"not in the per-capability shape "
                                 f"({'; '.join(retired)}) -- run "
                                 f"migrate_capability_records.py first"))
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
        # THE DEVICE, THE TOOLCHAIN, THE HOST AND THE POLICY, resolved BEFORE the
        # first mutation of ``entry``. Every refusal above this point leaves the entry
        # untouched and these must too: ``ledger`` is the object that gets written, so
        # an entry that is half-rewritten and then skipped would be PERSISTED in that
        # state the moment any other key binds.
        #
        # ALL FOUR ARE DERIVED ON EVERY WRITE, with no carry-forward branch left.
        # Until 2026-09-30 the host was re-derived only where the entry read the
        # sentinel "PENDING REBIND" and the policy only where it read "see artifact",
        # so re-binding a standing weld kept whatever machine and toolchain string it
        # had been authored with: a run on another Triton could be bound into an entry
        # labelled 3.1.0, and the contract test -- which greps that label -- would
        # pass on the label. Both sentinels are deleted. A run that cannot answer is
        # SKIPPED, keeping the entry's old digests and the weld tests red, which is
        # how every other refusal here fails.
        capability, identity = run_capability(fresh, [directory, chosen_root])
        if capability is None:
            skipped.append((key, f"{CAMPAIGN_DIRS[key]}/{artifact.name}: {identity}"))
            continue
        derived_host = _host_line(identity)
        if derived_host is None:
            skipped.append((key, f"{CAMPAIGN_DIRS[key]}/{artifact.name} and the "
                                 f"{DEVICE_STAMP} beside it cannot compose a host "
                                 "line (hostname, device, compute_capability, triton "
                                 "and cupy must all be present)"))
            continue
        refused = toolchain_refusal(identity, ledger)
        if refused is not None:
            skipped.append((key, f"{CAMPAIGN_DIRS[key]}/{artifact.name}: {refused}"))
            continue
        # THE ENTRY IS PASSED so a declared ``_mixed_policy`` exception is honoured in
        # the one spelling its existing records use, and refused if the run contradicts
        # it; see ``seed_triton_welds.policy_line``.
        from seed_triton_welds import PolicyStampConflict, policy_line  # noqa: PLC0415
        try:
            derived_policy = policy_line(fresh, entry)
        except PolicyStampConflict as exc:
            skipped.append((key, f"{CAMPAIGN_DIRS[key]}/{artifact.name}: policy "
                                 f"refused: {exc}"))
            continue
        if derived_policy is None:
            skipped.append((key, f"{CAMPAIGN_DIRS[key]}/{artifact.name} carries no "
                                 "subnormal_policy stamp naming a resolved policy"))
            continue

        records = (f"apps/api/{campaign_rel}/{CAMPAIGN_DIRS[key]}/ - {source_field}."
                   f"released=true (read_from {verdict.get('read_from', '?')}); "
                   f"{len(imported)} imported digests, every pinned path taken from "
                   f"the run's own imported_source_sha256"
                   + (f"; {host}" if host else "") + ".")
        # THE CARVE-OUT READS THE CAPABILITY'S OWN RECORD, and only where that
        # capability HAS one. The claim being preserved was made by a run on one
        # architecture, so appending it to a FIRST record for another would make that
        # record quote a run on a different device. Refusing instead would be worse
        # than either: this key is one of the 44 the Triton arms cite, so a rule that
        # skipped it whenever a new architecture had no prior line to carry would
        # hold the intersection at the architectures already certified and no round
        # could ever widen it. A first record is therefore written clean, and the
        # carve-out still binds every re-bind of a capability that has one.
        previous_record = (entry.get(fastpath.RUNS) or {}).get(capability) or {}
        sentinels = RECORDS_SENTINELS.get(key)
        if sentinels and previous_record.get("records"):
            records = (f"{records} PRIOR RECORD, NOT SUPERSEDED: "
                       f"{previous_record['records']}")
            if not any(token in records for token in sentinels):
                skipped.append((key, f"the records carve-out lost {sentinels}: "
                                     f"runs[{capability}]'s prior line does not "
                                     "hold them"))
                continue

        # THE BOUND BEFORE THE REWRITE, read while the entry still holds the digests
        # its existing records were cut against. This is the comparison that decides
        # whether another architecture's record survives this write, so it cannot be
        # taken after the pins move.
        bound_before = fastpath.bound_digest(entry)
        # Keys the artifact records come FROM THE ARTIFACT. The handful it cannot
        # record (see the tree-fallback rule above) come from the checkout, which
        # the guard has already proved is the tree this run executed.
        entry["source_sha256"] = {
            name: (imported[name] if name in imported
                   else hashlib.sha256((_API / name).read_bytes()).hexdigest())
            for name in curated}
        entry["code_sha256"] = {name: code_digest_of_path(_API / name)
                                for name in curated}
        if device_recut:
            entry["device_sha256"] = device_recut
        entry["status"] = "PASS"

        run = {
            "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            "records": records,
            "recorded_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "verdict_read_from": (
                f"{CAMPAIGN_DIRS[key]}/{artifact.name}:{source_field}.released"
                + (f" (read_from={verdict['read_from']})"
                   if verdict.get("read_from") else "")),
            "host": derived_host,
            "subnormal_policy": derived_policy,
            # ONE SPELLING EACH: the merge canonicalised them, and _host_line
            # returning a line is the proof both are present.
            "cupy_version": str(identity["cupy"]),
            "triton_version": str(identity["triton"]),
        }
        # CARRIED ONLY WHILE THE BYTES ARE THE SAME. A step budget is a MEASUREMENT
        # of the run that produced it ("8 launches per row; 8 product rows
        # byte-identical over 49,152 compared uint32 words"), so carrying one onto a
        # record about other bytes would put a previous tree's number in it. When the
        # previous record is not live on these bytes it is dropped, and ``fastpath``
        # names the absence rather than quoting a figure nothing measured here.
        run.update(carried_run_facts(entry, capability, CARRIED_WITHIN_CAPABILITY))
        if taken_from_tree.get(key):
            run["_digests_taken_from_the_checkout"] = (
                f"{sorted(taken_from_tree[key])} are pinned by this weld but are not in "
                f"the gate's imported_source_sha256 -- a gate does not import its own "
                f"test file. Their digests are the CHECKOUT's, which is licensed here "
                f"only because every digest the run DID record already matched the "
                f"checkout byte for byte, so the run and the checkout are the same tree.")
        try:
            staled = fastpath.bind_capability(
                entry, bound_before=bound_before, capability=capability, run=run,
                supersede=supersede)
        except fastpath.CapabilityRecordError as exc:
            # THE ENTRY IS PUT BACK. ``ledger`` is the object that gets written, and a
            # refusal here comes AFTER the pins moved, so leaving the half-rewritten
            # entry in place would persist new digests with no record behind them the
            # moment another key binds.
            ledger[key] = json.loads(json.dumps(before[key]))
            skipped.append((key, str(exc)))
            continue

        # THE SURVIVAL PROOF, in the tool rather than in a reviewer's head. The key
        # set may not move; every entry-level field this tool does not own must come
        # out byte-identical to what went in; and no OTHER capability's record may
        # move at all unless this round superseded it by name.
        old = before[key]
        assert set(entry) - set(old) == set(), (key, sorted(set(entry) - set(old)))
        assert set(old) - set(entry) == set(), (key, sorted(set(old) - set(entry)))
        assert set(entry["source_sha256"]) == set(old["source_sha256"]), key
        for field in old:
            if field in ENTRY_REFRESHED or field == fastpath.RUNS:
                continue
            assert json.dumps(entry[field], sort_keys=True) == json.dumps(
                old[field], sort_keys=True), (key, field)
        assert set(entry[fastpath.RUNS][capability]) <= set(SLOT_FIELDS) | {
            "bound_sha256"}, (key, sorted(entry[fastpath.RUNS][capability]))
        for name, record in (old.get(fastpath.RUNS) or {}).items():
            if name in (capability, *staled):
                continue
            assert json.dumps(entry[fastpath.RUNS][name],
                              sort_keys=True) == json.dumps(
                record, sort_keys=True), (key, "other capability moved", name)
        rebound.append((key, rel_by_root[chosen_root], capability, staled))

    # THE REPORT NAMES THE TREE, not just the family directory. With several
    # --campaign trees in play the family name alone does not say which run bound
    # the key, so the two misattributed records lines this tool wrote on
    # 2026-08-28 were invisible in its own output. Printing the tree makes the
    # attribution reviewable before --write rather than after.
    for key, tree, capability, staled in rebound:
        print(f"  rebound   {key}  runs[{capability}]  <- "
              f"{tree}/{CAMPAIGN_DIRS[key]}/"
              + (f"  SUPERSEDED {list(staled)}" if staled else ""), flush=True)
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
