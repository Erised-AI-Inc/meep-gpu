#!/usr/bin/env python3
"""Re-cut ``fingerprints.json['driver_dispatch']`` FROM a driver-route gate run.

WHY THIS EXISTS, and it is a defect report rather than a convenience. Measured
2026-08-29: three sha256s for ``meep_gpu/fastpath.py`` were live at once --
``87f740ee`` shipped, this record held ``06cdbf37``, and all four legs of the
gate the release cites had executed ``29ac1216``. The record was maintained by
hand, so it could drift from the tree AND from the run independently, and
nothing compared it to the run at all. The fix is not a better habit: it is that
every digest in the block is READ, from the artifact or from the tree, and the
tool refuses when the two disagree.

WHAT IT REFUSES TO DO. It does not release, widen, or invent anything. It
refuses unless

* the run directory names at least four legs, each carrying ``gate.json``;
* every leg reports ``release.released`` true;
* every leg's ``provenance.source_sha256`` agrees with every other leg's on each
  key the record binds -- one campaign, one tree;
* those digests equal the LIVE tree's, so the record cannot be cut against bytes
  that have already moved on;
* every ``(arm, case)`` pair in :data:`fastpath.RELEASED_FUSED_ARMS` appears in
  the ``arms_driven`` of a leg that dispatched;
* every leg stamped the SAME compute capability, and the run directory is the one
  :func:`fastpath.route_campaign` spells for it (``<stamp>_cc86``);
* every dispatching row planned against a device the tables it dispatched through
  certify (``plan.environment.device_certified_by_table``) and, for the Triton
  table, a certified Triton version (``plan.environment.triton_certified``), so a
  row admitted uncertified (a supported identity by default, or any identity
  under ``MEEP_GPU_ALLOW_UNCERTIFIED=1``) cannot enter a record.

Any of those failing prints the disagreement and exits non-zero with nothing
written. A stale record is then fixed the only way it can be: re-run the gate.

TWO BACKENDS, TWO RECORDS, ONE RULE. ``--backend triton`` (the default) cuts
``triton_kernels/fingerprints.json['driver_dispatch']`` from a
``dispatch_fused_route_*`` campaign; ``--backend metal`` cuts
``metal_kernels/fingerprints.json['driver_dispatch']`` from a
``dispatch_metal_route_*`` one. The refusals above are the same refusals on both;
what differs is the record path, which files it binds, which module owns the
release table, and THE POLARITY OF THE HARNESS LEG. Triton certifies under
``keep``, so its harness leg installs ``flush``; the Metal table certifies under
``flush`` — MPS flushes float32 denormals natively and exposes no lever — so its
harness leg installs ``keep`` and is the REFUSAL leg, the one whose licence is
rung 8bM declining rather than a dispatch succeeding.

ONE RECORD PER COMPUTE CAPABILITY, ON BOTH NVIDIA RECORDS. The route run's facts
(:data:`fastpath.DISPATCH_RUN_FIELDS`) live in ``runs[<capability>]`` beside a
``bound_sha256`` of the bytes the block binds, never beside those digests: a second
architecture is then certified by ADDING a record, and a record whose bytes have
moved reads stale instead of reading as evidence for a device it never ran on. The
capability is READ from the legs' ``provenance.device``, and the campaign directory
must be the one :func:`fastpath.route_campaign` spells for it, so a reader can tell
which architecture a record describes from the directory name alone.
``--supersede 8.6,9.0`` is how a cut on MOVED bytes says, by name, which other
capabilities it is retiring; without it such a cut is refused.

ONE RECORD PER GPU ARCHITECTURE ON THE METAL RECORD, by the same rule
(``meep_gpu.metal_runs``). The architecture is READ from every leg's
``provenance.apple_gpu.architecture``, the legs must agree, and the campaign directory
must be ``metal_runs.route_campaign(metal_dispatch.METAL_DRIVER_ROUTE_GATE,
<architecture>)``. The route run's facts (``metal_runs.DISPATCH_RUN_FIELDS``: the
status, legs, records, timestamp, the run's subkeys of ``released_fused_arms``, and the
``residency`` and ``lift`` blocks read off the legs) go into
``runs[<architecture>]``; ``--supersede applegpu_g13s`` retires an architecture by
name on a cut whose bytes moved.

THE DEFAULT AND ITS LICENCE LIVE IN THE PRIMARY TABLE'S RECORD.
``dispatch_by_default`` is written from ``fastpath.DISPATCH_BY_DEFAULT`` on every
cut, never typed. While it reads True the record of
``fastpath.primary_table(<capability>)`` — the first table in the shipped precedence
that admits it, which is the table whose plan composes first — must carry
``runs[<capability>].dispatch_by_default_licence``: the transcription of
``gate_dispatch_end_to_end``'s SHIP LEG (enable unset, harness installing no policy,
released) on the same device and the same bytes the route campaign ran.
``--end-to-end <dir>`` is the only thing that writes one, it is refused on any other
table, and an existing licence is carried forward only while the slot it sits in
still binds this cut's bytes — so the default and its evidence move together or the
cut stops.

    python recut_driver_dispatch_record.py --run dispatch_fused_route_2026-10-05_092_cc86
    python recut_driver_dispatch_record.py --run ... --write
    python recut_driver_dispatch_record.py --backend metal --run dispatch_metal_route_<stamp>
    python recut_driver_dispatch_record.py --run <stamp>_cc86 \\
        --end-to-end dispatch_end_to_end_<stamp>_cc86 --write
    python recut_driver_dispatch_record.py --backend cuda --run <stamp>_cc90 \\
        --supersede 8.6 --write
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

from meep_gpu import fastpath                          # noqa: E402
from meep_gpu import metal_runs                        # noqa: E402

#: The substitution verdicts that count as a measured drop: the two the route gate
#: spells in ``gate_dispatch_fused_route.EXACT_SUBSTITUTION_VERDICTS``, and its
#: ``TRITONLESS_SUBSTITUTION_VERDICT``, which only the ``cuda_alone`` leg writes (the
#: absorbed-slot proof against a Triton-less baseline, launch arithmetic kept).
EXACT_SUBSTITUTION_VERDICTS = ("EXACT", "EXACT-ABSORBED-ARRAY-SLOT",
                               "EXACT-ABSORBED-TRITONLESS-BASELINE")
from meep_gpu.code_identity import code_digest_of_path  # noqa: E402

RECORD = _API / "meep_gpu" / "triton_kernels" / "fingerprints.json"
RESULTS = _HERE / "results"

#: The legs a driver-route campaign runs. Named rather than counted so a campaign
#: that silently dropped one is a refusal instead of a shorter list.
EXPECTED_LEGS = ("flush", "harness_keep", "shipped", "shipped_expansion_probe")

#: Per-backend: the record to cut, the legs the campaign must carry, and the files
#: the block binds.
#:
#: ``optional_legs`` EXISTS FOR TWO LEGS A CAMPAIGN MAY OR MAY NOT CARRY and is empty
#: on Triton, so that backend's behaviour is byte-identical to what it was. The Metal
#: driver writes a fifth leg — ``band_witness``, the array path alone under a
#: keeping host, censusing the subnormal band — and an exact four-name comparison
#: would refuse every real run of it. The CUDA campaign gained ``cuda_alone`` on
#: 2026-09-27 (see that table's entry). The floor is not weakened by admitting
#: either: every REQUIRED leg must carry ``release.released`` true, and so must a
#: band witness, so one that found an empty band still refuses the record. The one
#: exception is ``cuda_alone``, which is NON-BLOCKING by release decision (2026-09-27):
#: present and not released, it is recorded as exactly that, it witnesses nothing
#: (no arms_driven, no arbitration, no measured drop), and the five standard legs
#: are cut without it.
#:
#: ``bound`` MIXES TWO ROOTS on Metal and that is deliberate. Eight keys are
#: package-relative (``meep_gpu/...``) and the ninth is repo-relative
#: (``parity/meep_gpu/gate_dispatch_metal_route.py``), because
#: ``test_every_metal_weld_pins_the_script_that_gated_it`` requires a Metal weld to
#: pin the script that gated it and the ledger spells parity paths that way.
#: :func:`_resolve` is the one place that knows which root a key belongs to.
BACKENDS = {
    "triton": {
        "record": _API / "meep_gpu" / "triton_kernels" / "fingerprints.json",
        "required_legs": ("flush", "harness_keep", "shipped",
                          "shipped_expansion_probe"),
        "optional_legs": (),
        "run_prefix": "dispatch_fused_route",
        "bound": None,   # read from the existing block: Triton's is already cut
    },
    "metal": {
        "record": _API / "meep_gpu" / "metal_kernels" / "fingerprints.json",
        # THE POLARITY INVERSION, named here rather than left to a reader: Triton's
        # harness leg installs FLUSH because that table certifies under keep; this
        # one installs KEEP because this table certifies under flush.
        "required_legs": ("harness_flush", "harness_keep", "shipped",
                          "shipped_expansion_probe"),
        "optional_legs": ("band_witness",),
        "run_prefix": "dispatch_metal_route",
        "bound": ("meep_gpu/driver.py", "meep_gpu/fastpath.py",
                  "meep_gpu/fields.py", "meep_gpu/metal_dispatch.py",
                  "meep_gpu/metal_kernels/launch.py",
                  "meep_gpu/metal_kernels/device.py",
                  "meep_gpu/metal_kernels/subnormal.py",
                  "meep_gpu/subnormal_policy.py",
                  "parity/meep_gpu/gate_dispatch_metal_route.py"),
    },
    # THE THIRD TABLE, AND THE ONE WHOSE LEG NAMES ARE PREFIXED. The CUDA campaign
    # runs inside the SAME gate driver as the Triton one
    # (``gate_dispatch_fused_route.py --backend cuda``), so its legs carry a
    # ``cuda_`` prefix and cannot be confused with a Triton run's in a shared
    # results tree. Five of them, and the fifth is the one that makes this table's
    # complex arms measurable at all: their expansion licence comes from the
    # unified probe record and nothing else offers it.
    #
    # ``cuda_default_precedence`` IS NOT AN OPTIONAL LEG. It is the measured
    # statement of the shipped arbitration — "Triton first, hand-CUDA fills its
    # refusals" — and a campaign that dropped it would leave the default-precedence
    # number an assertion rather than a measurement.
    #
    # ``cuda_alone`` IS OPTIONAL AND NON-BLOCKING, AND THE RECORD SAYS WHICH WAY IT
    # WENT. It is the composition a host with no validated Triton gets — the
    # preference unset and Triton made unimportable inside the gate's process, so
    # rung 4 drops that table by name — and no campaign before 2026-09-27 drove it.
    # Optional because the campaigns already on file (five legs) must stay
    # cuttable; non-blocking because a Triton-less composition no earlier round drove
    # must not hold back the five standard legs (release decision, 2026-09-27). Recorded
    # in ``cuda_alone_leg`` as "absent", "present, released" or "present, not
    # released" (with its reasons), so a reader never infers it from the leg list.
    # Released, it dispatches and is a witness for ``arms_driven``, so it is NOT in
    # :data:`NON_DISPATCHING_LEGS`; not released, it witnesses nothing. Its bytes are
    # still required to agree with the other legs': a leg cut on another tree is
    # another campaign, released or not.
    "cuda": {
        "record": _API / "meep_gpu" / "cuda_kernels" / "fingerprints.json",
        "required_legs": ("cuda_default_precedence", "cuda_flush",
                          "cuda_harness_keep", "cuda_shipped",
                          "cuda_shipped_expansion_probe"),
        "optional_legs": ("cuda_alone",),
        "run_prefix": "dispatch_fused_route_cuda",
        "bound": None,   # derived below: the released arms' modules move per Tier
    },
}


def _cuda_bound_files() -> tuple:
    """Every file the CUDA driver_dispatch block binds, DERIVED from the release.

    Four groups, and the fourth is why this is a function rather than a constant:

    * the seam — ``driver.py``, ``fastpath.py``, ``fields.py`` — the same three the
      Triton block binds, because the same nine consults answer for both tables;
    * ``fastpath_cuda.py``, which holds this table's rows, its release and the
      merge rule;
    * the composer — ``cuda_kernels/{arms,fused_pairs,registry}.py``;
    * EVERY RELEASED ARM'S OWN PRODUCT MODULE. Those move whenever the release
      does (Tier 2 adds nine labels), so a typed list would be a second place to
      forget an edit and a record could end up binding a module no released arm
      uses — or worse, not binding one it does.

    Plus the gate script, which ``test_cuda_weld_contract`` requires every weld to
    pin.
    """
    from meep_gpu import fastpath_cuda  # noqa: PLC0415
    from meep_gpu.cuda_kernels import fused_pairs  # noqa: PLC0415

    released = set(fastpath_cuda.CUDA_RELEASED_FUSED_ARMS)
    modules = sorted({
        f"meep_gpu/cuda_kernels/{product['module']}.py"
        for product in fused_pairs.FUSED_PRODUCTS.values()
        if fastpath_cuda.namespaced(product["label"]) in released})
    return tuple(["meep_gpu/driver.py", "meep_gpu/fastpath.py",
                  "meep_gpu/fastpath_cuda.py", "meep_gpu/fields.py",
                  "meep_gpu/cuda_kernels/arms.py",
                  "meep_gpu/cuda_kernels/fused_pairs.py",
                  "meep_gpu/cuda_kernels/registry.py",
                  "parity/meep_gpu/gate_dispatch_fused_route.py"] + modules)

#: Legs whose licence is a REFUSAL rather than a dispatch, so their artifacts are
#: not consulted for ``arms_driven``. Asking a refusal leg which arms it drove and
#: finding none is not evidence that a release row is wrong.
#:
#: THE CUDA CAMPAIGN'S REFUSAL LEG IS ``cuda_flush``, not its ``cuda_harness_keep``,
#: and the polarity is the table's: this table certifies under ``keep`` like Triton,
#: so a harness that installs KEEP is installing the certified policy and its legs
#: DO dispatch, while the flush leg is the one every dispatch case must be
#: PASS-POLICY-REFUSED on by name at rung 8b.
NON_DISPATCHING_LEGS = ("harness_keep", "band_witness", "cuda_flush")

#: The CUDA campaign's optional Triton-less leg (``gate_dispatch_fused_route``'s
#: ``TRITON_ABSENT_LEG``), named once for the record fields that describe it.
CUDA_ALONE_LEG = "cuda_alone"

#: The three ``exclusions.released_fused_arms`` subkeys that describe the ROUTE RUN
#: rather than the release predicate: which campaign drove the arms, its artifact and
#: what the drive measured. They move into ``runs[<capability>].released_fused_arms``,
#: because they are the only part of that block that changes when the same release is
#: driven on a second architecture. The rest — the arms, their cases, the shared
#: envelope and the per-arm axes — is read off the shipped module and is the same
#: statement on every device, so it stays at entry level.
RELEASED_RUN_SUBKEYS = ("gate", "artifact", "what_was_measured")


def _resolve(backend: str, name: str) -> Path:
    """The file a bound key names. Two roots on Metal; one on Triton.

    A key carrying a ``parity/`` prefix is repo-relative and everything else is
    relative to the package, which is how both ledgers already spell their pins.
    """
    if name.startswith("parity/") or name.startswith("meep_gpu/"):
        return _API / name
    return _API / "meep_gpu" / name


def _leg_key(backend: str, name: str) -> str:
    """The key a LEG's provenance spells for a bound path.

    The Triton gate records ``fastpath.py``; the Metal one records
    ``fastpath.py`` too but the ledger binds ``meep_gpu/fastpath.py``, and the gate
    records its own parity path under the full repo-relative key. One translation,
    here, rather than a second spelling in the artifact.
    """
    if name.startswith("parity/"):
        return name
    return name[len("meep_gpu/"):] if name.startswith("meep_gpu/") else name


def _legs(run: Path):
    return sorted(p for p in run.glob("*/gate.json"))


# ---------------------------------------------------------------------------
# The default's licence: gate_dispatch_end_to_end's ship leg, transcribed
# ---------------------------------------------------------------------------

#: The gate the licence is cut from, by module name (what the record says) and by
#: path (what a reader opens).
LICENCE_GATE = "gate_dispatch_end_to_end"
LICENCE_SCRIPT = "parity/meep_gpu/gate_dispatch_end_to_end.py"

#: The two verdicts a licensing case may carry. Every other verdict — DIVERGENCE,
#: VACUOUS-PASS, CONTROL-FAILURE, COMPARATOR-BLIND, HARNESS-FAILURE, ERROR,
#: PLAN-ONLY — refuses the transcription.
LICENCE_VERDICTS = ("PASS-DISPATCHED", "PASS-FELL-BACK")

#: The fewest PASS-DISPATCHED cases a licensing ship leg may carry: 7 of the gate's 9.
#: ONE CONSTANT, READ BY EVERY SITE THAT DECIDES IT — this tool's transcription,
#: ``run_e2e_ship.sh``'s GO / NO-GO and ``test_dispatch_contract``'s licence weld all
#: import it — because three sites that each typed their own floor disagreed: the
#: transcription and the verdict asked for one dispatched case and the weld for
#: seven, so a ship leg dispatching three of nine transcribed cleanly, read GO, and
#: failed only at the test. Seven is the 2026-08-15 closing re-run's count on a tree
#: that predates the fused and folded releases; neither release removed a
#: dispatching case, so a leg below it has lost one and licenses nothing.
LICENCE_MIN_DISPATCHED = 7

#: The one package variable a PLAIN ship leg may carry: where the dispatch records
#: are appended. It changes what is written down, not what runs.
LICENCE_ENVIRONMENT_ALLOWED = ("MEEP_GPU_DISPATCH_LOG",)

#: What a plain install never sets. Any of these in the ship leg's
#: ``environment_at_start`` means the run measured a configuration a user does not
#: get: the package's own switches (``MEEP_GPU_*``), a retired spelling rung 2
#: refuses (``*_FDTD_*``), and CuPy's accelerator list, which the drivers
#: export EMPTY for their flush legs and a plain install leaves unset.
LICENCE_ENVIRONMENT_REFUSED_PREFIXES = ("MEEP_GPU_", "TRIDENT_FDTD_")
LICENCE_ENVIRONMENT_REFUSED_NAMES = ("CUPY_ACCELERATORS",)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stamped_capabilities(artifacts: dict) -> dict:
    """``leg -> compute capability`` as each leg's artifact stamped it, normalised.

    Read from ``provenance.device``, the key the route gate's ``_provenance`` writes.
    An earlier spelling read ``device_identity``, which no gate writes, so every leg
    read ``None``: the record's capability list came out empty and the refusal of a
    campaign whose legs ran on two architectures could never fire. A leg whose
    artifact names no device maps to ``None``.

    A stamp that is not ``major.minor`` is a NAMED refusal against its own leg rather
    than a traceback three steps later: ``_normalized_capability`` leaves a spelling it
    does not recognise alone (``sm_86`` stays ``sm_86``), so such a value would
    otherwise pass the agreement checks below and become a record key.
    """
    stamped = {}
    for leg, artifact in artifacts.items():
        device = (artifact.get("provenance") or {}).get("device") or {}
        value = device.get("compute_capability") if isinstance(device, dict) else None
        if not value:
            stamped[leg] = None
            continue
        try:
            stamped[leg] = fastpath._require_capability(value)  # noqa: SLF001
        except fastpath.CapabilityRecordError as refused:
            raise SystemExit(
                f"REFUSING: leg {leg!r} stamps compute capability {value!r}, which is "
                f"not one a record can be keyed by: {refused}") from refused
    return stamped


def _uncertified_dispatch_reasons(case: str, plan: dict, tables) -> list:
    """Why this dispatching row's device was not certified by a table that served it.

    THE RECORD MAY NOT REST ON A ROW ADMITTED UNCERTIFIED. A supported device or
    Triton version no cited weld ran on dispatches by default, and
    ``MEEP_GPU_ALLOW_UNCERTIFIED=1`` takes an unsupported one past rungs 4 and 4b too;
    the plan says so: ``environment.device_certified_by_table[t]`` and, for the Triton
    table, ``environment.triton_certified`` stay False (``fastpath._environment_block``)
    while the kernels dispatch anyway. Without this check a campaign run that way would
    cut a record — and a licence — for an architecture or a compiler nothing had
    certified, which is exactly the hole the per-capability container exists to close.

    A row that dispatched through no table is not a dispatching row and is skipped;
    a dispatching row whose plan records no answer refuses rather than passing,
    because a run predating the field cannot show what it planned against.
    """
    if not tables:
        return []
    environment = plan.get("environment")
    if not isinstance(environment, dict):
        return [f"{case} dispatched through {sorted(tables)} and its plan recorded no "
                f"environment; re-run the current gate"]
    by_table = environment.get("device_certified_by_table")
    if not isinstance(by_table, dict):
        return [f"{case} dispatched through {sorted(tables)} and its plan recorded no "
                f"device_certified_by_table; re-run the current gate"]
    device = (environment.get("device") or {}).get("compute_capability")
    reasons = [f"{case} dispatched through {table!r} on compute capability "
               f"{device or 'unrecorded'}, which that table's ledger does not certify "
               f"(device_certified_by_table[{table!r}] = {by_table.get(table)!r})"
               for table in sorted(tables) if by_table.get(table) is not True]
    if "triton" in tables and environment.get("triton_certified") is not True:
        reasons.append(f"{case} dispatched through 'triton' on Triton "
                       f"{environment.get('triton') or 'unrecorded'}, which that "
                       "table's ledger does not certify (triton_certified = "
                       f"{environment.get('triton_certified')!r})")
    return reasons


#: The three verdicts a Metal plan records for its environment, in the order a
#: refusal names them (``metal_dispatch.environment_block``).
METAL_ENVIRONMENT_VERDICTS = (("device_certified", "GPU architecture", "device"),
                              ("torch_certified", "torch", "torch"),
                              ("frontend_certified", "Metal frontend", "metal_frontend"))


def _uncertified_metal_dispatch_reasons(case: str, plan: dict, tables) -> list:
    """Why this dispatching Metal row ran on an environment the cited welds do not certify.

    THE METAL TABLE RUNS AN UNCERTIFIED ENVIRONMENT BY DEFAULT, and says so in the
    plan: its device verdict is ``True`` only when every cited weld has a live run for
    the GPU architecture, and its torch and frontend verdicts only when every such run
    recorded this host's (``metal_dispatch.environment_verdict``). A route campaign run on any
    other Mac would otherwise cut a record for an environment no weld ran on, as an
    NVIDIA campaign run on an identity admitted uncertified would
    (:func:`_uncertified_dispatch_reasons`).
    ``None`` refuses as ``False`` does: a record cannot rest on a fact its welds did
    not stamp, so a Metal route campaign runs after the welds it cites are bound.
    """
    if not tables:
        return []
    environment = plan.get("environment")
    if not isinstance(environment, dict):
        return [f"{case} dispatched through {sorted(tables)} and its plan recorded no "
                f"environment; re-run the current gate"]
    reasons = []
    for key, what, read_key in METAL_ENVIRONMENT_VERDICTS:
        if key not in environment:
            reasons.append(f"{case} dispatched through {sorted(tables)} and its plan "
                           f"recorded no {key}; re-run the current gate")
            continue
        if environment[key] is not True:
            read = environment.get(read_key)
            if isinstance(read, dict):
                read = read.get("architecture")
            reasons.append(f"{case} dispatched on {what} {read or 'unrecorded'}, which "
                           f"the Metal welds this table cites do not certify "
                           f"({key} = {environment[key]!r})")
    return reasons


def _end_to_end_dirs(argument: str):
    """``(ship, null)`` directories for ``--end-to-end``, or a reason it cannot read.

    Accepts the round's directory (holding ``ship/`` and ``null/``, as
    ``run_e2e_ship.sh`` writes them) or the ship leg's own directory with a ``null``
    sibling; relative names are read under ``results/`` first, as ``--run`` is.
    """
    base = Path(argument)
    if not base.is_absolute() and not base.exists():
        base = RESULTS / argument
    if (base / "ship" / "summary.json").is_file():
        return base / "ship", base / "null", None
    if (base / "summary.json").is_file():
        return base, base.parent / "null", None
    return None, None, (f"--end-to-end {argument}: neither {base}/ship/summary.json "
                        f"nor {base}/summary.json exists")


def _run_label(path: Path) -> str:
    """A leg directory as the ledger spells artifacts: repo-relative when it can be."""
    try:
        return f"apps/api/parity/meep_gpu/results/{path.resolve().relative_to(RESULTS.resolve())}/"
    except ValueError:
        return str(path)


def _gate_cases():
    """The gate's own case list, so a partial run cannot license the default."""
    if str(_HERE) not in sys.path:
        sys.path.insert(0, str(_HERE))
    import gate_dispatch_end_to_end as e2e  # noqa: PLC0415

    return tuple(e2e.CASES)


def transcribe_end_to_end(argument: str, ran: dict, bound: list, backend: str,
                          capability: str):
    """``(licence block, problems)`` from a ship leg and its null control.

    REFUSES — every reason collected, nothing written — unless the ship leg:

    * is a stepped gate run (not ``--smoke``, not ``--plan-only``, not the null
      control) over the gate's WHOLE case list;
    * ran with the enable REMOVED (``enable_from_default``) and the harness
      installing no policy, in an environment carrying no dispatch-shaping variable
      (:data:`LICENCE_ENVIRONMENT_REFUSED_PREFIXES`), and every case's dispatch leg
      recorded the enable at ``value`` None, ``effective`` True;
    * released every case as PASS-DISPATCHED or PASS-FELL-BACK with no divergent
      checkpoint, at least :data:`LICENCE_MIN_DISPATCHED` of them dispatched;
    * ran THESE bytes: every package file it digested equals the live tree, and
      every file the record binds equals what the route campaign ran;
    * ran on THIS capability — the one the route legs stamped — because the licence
      is filed under it and a licence cut on another architecture is evidence about
      a device this record does not describe;
    * planned every dispatching case against a device the tables that served it
      certify, so a run reaching the arms through ``MEEP_GPU_ALLOW_UNCERTIFIED``
      cannot license the default;

    and its null control, beside it, is clean on the same ``fastpath.py`` over the
    SAME case list — a one-case null beside a nine-case ship leg attributes eight of
    its "identical" verdicts to nothing.
    """
    problems = []
    ship, null, missing = _end_to_end_dirs(argument)
    if missing:
        return None, [missing]
    summary_path = ship / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    provenance = summary.get("provenance") or {}
    rows = summary.get("rows") or []
    verdicts = dict(summary.get("verdicts") or {})
    label = f"--end-to-end {_run_label(ship)}"

    if not fastpath.DISPATCH_BY_DEFAULT:
        problems.append(f"{label}: a licence is for DISPATCH_BY_DEFAULT = True and "
                        "fastpath.py reads False")
    if summary.get("smoke_run_measures_nothing_about_dispatch"):
        problems.append(f"{label}: a --smoke run measures nothing about dispatch")
    if summary.get("null_control"):
        problems.append(f"{label}: this is a null control, which licenses nothing; "
                        "pass the ship leg")
    if not (summary.get("enable_from_default")
            or provenance.get("enable_from_default")):
        problems.append(f"{label}: the leg SET the enable (no --enable-from-default), "
                        "so it measured a run that asked for dispatch, not the "
                        "default a user gets")
    if provenance.get("install_policy_requested") is not None:
        problems.append(f"{label}: the harness installed "
                        f"{provenance.get('install_policy_requested')!r}; the licence "
                        "is the leg where it installs nothing")
    environment = provenance.get("environment_at_start")
    if environment is None:
        problems.append(f"{label}: the run recorded no environment_at_start, so a "
                        "plain environment cannot be shown; re-run the current gate")
    else:
        shaping = sorted(
            name for name in environment
            if name not in LICENCE_ENVIRONMENT_ALLOWED
            and (name.startswith(LICENCE_ENVIRONMENT_REFUSED_PREFIXES)
                 or name in LICENCE_ENVIRONMENT_REFUSED_NAMES))
        if shaping:
            problems.append(f"{label}: the run inherited {shaping}; a plain install "
                            "sets none of them")

    expected = set(_gate_cases())
    if set(verdicts) != expected:
        problems.append(f"{label}: the run covers {sorted(verdicts)}; the gate's case "
                        f"list is {sorted(expected)}")
    bad = {case: verdict for case, verdict in verdicts.items()
           if verdict not in LICENCE_VERDICTS}
    if bad:
        problems.append(f"{label}: cases not released: {bad}")
    dispatched = sorted(case for case, verdict in verdicts.items()
                        if verdict == "PASS-DISPATCHED")
    fell_back = sorted(case for case, verdict in verdicts.items()
                       if verdict == "PASS-FELL-BACK")
    if len(dispatched) < LICENCE_MIN_DISPATCHED:
        problems.append(f"{label}: {len(dispatched)} of {len(verdicts)} cases "
                        f"dispatched ({dispatched}); the licence needs at least "
                        f"{LICENCE_MIN_DISPATCHED} of {len(expected)} "
                        "(LICENCE_MIN_DISPATCHED)")
    divergent_map = dict(summary.get("first_divergent_checkpoint") or {})
    divergent = {case: checkpoint for case, checkpoint in divergent_map.items()
                 if checkpoint is not None}
    if divergent or set(divergent_map) != set(verdicts):
        problems.append(f"{label}: first_divergent_checkpoint must be null on every "
                        f"case; read {divergent_map}")

    enable, tables = {}, {}
    for row in rows:
        case = row.get("case")
        leg = row.get("dispatch_leg") or {}
        block = leg.get("enable")
        if not block:
            problems.append(f"{label}: {case}'s dispatch leg carries no enable block "
                            "(the run predates it); re-run the current gate")
            continue
        enable[case] = dict(block)
        if (block.get("variable") != fastpath.DISPATCH_ENABLE
                or block.get("value") is not None or block.get("effective") is not True):
            problems.append(f"{label}: {case}'s dispatch leg read the enable as "
                            f"{block}; the licence leg measures {fastpath.DISPATCH_ENABLE} "
                            "unset and effective")
        tables[case] = list(((row.get("evidence") or {}).get("tables_dispatched"))
                            or (leg.get("composition") or {}).get("tables_dispatched")
                            or [])
        problems.extend(f"{label}: licence_row_dispatched_on_an_uncertified_device: "
                        f"{reason}" for reason in
                        _uncertified_dispatch_reasons(case, leg, tables[case]))

    # THE CAPABILITY THE LICENCE IS FILED UNDER IS THE ONE IT RAN ON. The licence
    # sits in ``runs[<capability>]``, whose key comes from the ROUTE legs; a ship leg
    # from another architecture would file an 8.6 measurement under 9.0 and the slot
    # would read as that device's evidence.
    licence_device = (provenance.get("device") or {})
    licence_capability = licence_device.get("compute_capability")
    normalised = (fastpath._normalized_capability(licence_capability)  # noqa: SLF001
                  if licence_capability else None)
    if normalised != capability:
        problems.append(f"{label}: licence_ran_another_capability: the ship leg ran "
                        f"compute capability {normalised or 'none recorded'} and this "
                        f"record is being cut for {capability}; the licence is filed "
                        f"under the capability it measured")

    # THE BYTES. Every package SOURCE file the run digested against the live tree,
    # and every file the record binds against what the route campaign ran. Two
    # classes are reported rather than required: harness files (``parity/``), which
    # move no shipped byte, and the ledgers (``*.json``), which the post-round
    # rebind and this very tool rewrite after the run by design.
    digests = dict(provenance.get("source_sha256") or {})
    if "fastpath.py" not in digests:
        problems.append(f"{label}: the run recorded no fastpath.py digest")
    harness_moved = {}
    for key, value in sorted(digests.items()):
        path = _resolve(backend, key)
        if not path.is_file():
            problems.append(f"{label}: it digested {key}, which the tree no longer has")
            continue
        live = _sha256_file(path)
        if live == value:
            continue
        if key.startswith("parity/") or key.endswith(".json"):
            harness_moved[key] = {"ran": value, "tree": live}
            continue
        problems.append(f"{label}: it ran {key} at {value[:12]} and the tree ships "
                        f"{live[:12]}; re-run the ship leg on the bytes that ship")
    # THE BYTES ARE CHECKED HERE AND PINNED BY THE SLOT, not copied into the block.
    # Every file the record binds must have been digested by this run and must equal
    # what the route campaign ran — one tree, one cut — but the licence CARRIES no
    # digest map of its own. The slot it is written into holds ``bound_sha256`` over
    # the record's own ``source_sha256``, so the moment those bytes move the whole
    # slot reads stale, licence included; a second copy of the map inside it would
    # also be raw-checked against the tree by ``weld_record_walk.census`` forever
    # after, which is a pin a superseded licence can never satisfy. The run's whole
    # digest set stays in the artifact, which ``artifact_sha256`` pins.
    for name in bound:
        key = _leg_key(backend, name)
        if name.startswith("parity/"):
            continue
        if key not in digests:
            problems.append(f"{label}: the record binds {name} and the run recorded no "
                            "digest for it, so the licence cannot vouch for it")
            continue
        if name in ran and digests[key] != ran[name]:
            problems.append(f"{label}: it ran {key} at {digests[key][:12]} and the "
                            f"route campaign ran {ran[name][:12]}; one tree, one cut")

    null_run = None
    null_summary_path = null / "summary.json"
    if not null_summary_path.is_file():
        problems.append(f"{label}: no null control beside it ({null_summary_path}); "
                        "the ship leg's 'identical' is attributable to dispatch only "
                        "when the array path reproduces itself")
    else:
        null_summary = json.loads(null_summary_path.read_text(encoding="utf-8"))
        null_verdicts = dict(null_summary.get("verdicts") or {})
        null_divergent = dict(null_summary.get("first_divergent_checkpoint") or {})
        # CLEAN MEANS CLEAN OVER THE GATE'S WHOLE CASE LIST. Checked on its own
        # verdicts alone, a null control that ran one case passed beside a
        # nine-case ship leg, and the eight cases it never ran had nothing
        # attributing their "identical" to dispatch rather than to the array path
        # reproducing itself.
        covers = set(null_verdicts) == expected
        clean = (bool(null_summary.get("null_control"))
                 and covers
                 and all(v == "PASS-FELL-BACK" for v in null_verdicts.values())
                 and all(v is None for v in null_divergent.values()))
        null_fastpath = ((null_summary.get("provenance") or {})
                         .get("source_sha256") or {}).get("fastpath.py")
        if not covers:
            problems.append(f"--end-to-end {_run_label(null)}: the null control covers "
                            f"{sorted(null_verdicts)}; the gate's case list is "
                            f"{sorted(expected)}")
        if not clean:
            problems.append(f"--end-to-end {_run_label(null)}: the null control is not "
                            f"clean ({null_verdicts}, {null_divergent})")
        if null_fastpath != digests.get("fastpath.py"):
            problems.append(f"--end-to-end {_run_label(null)}: the null control ran "
                            f"fastpath.py {str(null_fastpath)[:12]}, the ship leg "
                            f"{str(digests.get('fastpath.py'))[:12]}")
        null_run = {"run": _run_label(null), "verdicts": null_verdicts,
                    "first_divergent_checkpoint": null_divergent, "clean": clean,
                    "artifact": f"{_run_label(null)}summary.json",
                    "artifact_sha256": _sha256_file(null_summary_path)}

    device = provenance.get("device") or {}
    served = sorted({table for names in tables.values() for table in names})
    block = {
        "gate": LICENCE_GATE,
        "script": LICENCE_SCRIPT,
        "run": _run_label(ship),
        "recorded_utc": provenance.get("utc"),
        "host": provenance.get("host"),
        "device": device,
        "toolchain": {name: provenance.get(name)
                      for name in ("python", "numpy", "meep", "cupy", "triton")},
        "cuda_visible_devices": provenance.get("cuda_visible_devices"),
        "launch_witnesses": provenance.get("launch_witnesses"),
        "null_control": bool(summary.get("null_control")),
        "enable_from_default": bool(summary.get("enable_from_default")
                                    or provenance.get("enable_from_default")),
        "install_policy_requested": provenance.get("install_policy_requested"),
        "environment_at_start": environment,
        "verdicts": verdicts,
        "dispatched_cases": dispatched,
        "fell_back_cases": fell_back,
        "first_divergent_checkpoint": divergent_map,
        "enable": enable,
        "tables_dispatched": tables,
        "harness_and_ledger_files_moved_since_the_run": sorted(harness_moved),
        "artifact": f"{_run_label(ship)}summary.json",
        "artifact_sha256": _sha256_file(summary_path),
        "null_control_run": null_run,
        "what_this_licenses": (
            f"fastpath.DISPATCH_BY_DEFAULT = True on the bytes this record binds, "
            f"which this run digested and the slot's bound_sha256 pins: "
            f"{len(verdicts)} cases, each dispatch leg run with "
            f"{fastpath.DISPATCH_ENABLE} UNSET and the harness installing no "
            f"policy, byte-identical to the {fastpath.FUSED_KILL_SWITCH}=0 array "
            "path at every checkpoint and in the flux spectra; "
            f"{len(dispatched)} dispatched ({', '.join(dispatched)}) through the "
            f"table(s) {served or 'none'}, and {len(fell_back)} fell back with a "
            "named reason"),
        "what_this_does_not_license": (
            f"a compute capability other than {capability} (measured on "
            f"{device.get('name', 'the device recorded')}), which is the capability "
            f"this licence is filed under and the only one it measured; "
            f"a Triton other than {provenance.get('triton')}; a composition of "
            f"tables other than {served or 'the one recorded'} — in particular the "
            "hand-CUDA table composing alone, unless it is the table recorded here; "
            "complex-storage rows under an expansion licence (the run exported none); "
            "and the Metal table, whose default is licensed by the Metal route "
            "campaign's shipped leg"),
        "transcribed_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "transcribed_by": ("parity/meep_gpu/recut_driver_dispatch_record.py "
                           "--end-to-end"),
    }
    return block, problems


def _enable_statement() -> str:
    """The enable's semantics as the record states them, read off ``fastpath``.

    Written from the constants on every cut: the hand-kept string it replaces named
    a switch retired months earlier (``*_FDTD_TRITON``) beside a record whose
    digests were current.
    """
    enable = fastpath.DISPATCH_ENABLE
    values = getattr(fastpath, "DISPATCH_ENABLE_VALUES", None)
    other = ("any other value is refused by name at rung 2 and the run takes the "
             "array path" if values is not None else
             "any other non-empty value opts in")
    kill = fastpath.FUSED_KILL_SWITCH
    # Read with a fallback, as the enable's values are: this tool runs on staged
    # trees, and a tree whose kill switch predates the strict reader still gets a
    # true sentence rather than an AttributeError.
    kill_values = getattr(fastpath, "FUSED_KILL_SWITCH_VALUES", None)
    kill_other = (f"; {kill}=1 or unset leaves dispatch to {enable}, and any other "
                  f"{kill} value is refused by name at rung 1 and the run takes the "
                  "array path" if kill_values is not None else "")
    return (f"{enable}=1 opts in; {enable}=0 opts out; unset takes "
            f"dispatch_by_default ({fastpath.DISPATCH_BY_DEFAULT}); {other}; "
            f"{kill}=0 vetoes dispatch whatever the enable says{kill_other}")


def _pending_dispatch_claim() -> str:
    """``pending_host_recut['dispatch']`` on the Triton ledger, read off the constants.

    The claim that bounds the blast radius of any declared host drift, and
    ``test_triton_planner_composition`` binds its opening words to
    ``fastpath.DISPATCH_BY_DEFAULT``. It was hand-kept, so the flip left it reading
    "WIRED BUT OFF BY DEFAULT ... DISPATCH_BY_DEFAULT is False" beside a constant
    that read True, and only a hand edit of a ledger field could clear the test.
    Written whole on every Triton cut — replaced, never prefixed — so the ledger's
    sentence is the constants' sentence.
    """
    state = ("WIRED AND ON BY DEFAULT" if fastpath.DISPATCH_BY_DEFAULT
             else "WIRED BUT OFF BY DEFAULT")
    return (f"{state} (see driver_dispatch.{fastpath.RUNS}[<compute capability>]"
            f".dispatch_by_default_licence, on the primary table for that "
            f"capability): {_enable_statement()}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True,
                        help="results/ directory name of the campaign, e.g. "
                             "dispatch_fused_route_2026-10-05_092_cc86. On the NVIDIA "
                             "tables it must be fastpath.route_campaign(<the table's "
                             "route gate constant>, <the capability the legs stamped>)")
    parser.add_argument("--write", action="store_true", help="apply (default: report)")
    parser.add_argument("--backend", default="triton", choices=sorted(BACKENDS),
                        help="which kernel table's driver_dispatch record to cut")
    parser.add_argument("--end-to-end", default=None, metavar="DIR",
                        help="gate_dispatch_end_to_end's ship leg (the round's "
                             "directory holding ship/ and null/, or the ship "
                             "directory itself), transcribed into "
                             "runs[<capability>].dispatch_by_default_licence. The "
                             "primary table for that capability only "
                             "(fastpath.primary_table); required there while "
                             "fastpath.DISPATCH_BY_DEFAULT is True unless the "
                             "capability's existing record still binds these bytes")
    parser.add_argument("--supersede", default="", metavar="KEY[,KEY...]",
                        help="compute capabilities (NVIDIA) or GPU architectures "
                             "(Metal) whose records this cut retires. A cut on bytes "
                             "that MOVED leaves every other record certifying code "
                             "that no longer ships; those are named in the refusal and "
                             "must be re-run on these bytes or listed here")
    arguments = parser.parse_args(argv)
    supersede = tuple(sorted({name.strip() for name in arguments.supersede.split(",")
                              if name.strip()}))

    backend = BACKENDS[arguments.backend]
    record_path = backend["record"]
    required = tuple(sorted(backend["required_legs"]))
    optional = tuple(backend["optional_legs"])
    # THE TABLE WHOSE RELEASE ROWS THIS RECORD QUOTES. The Metal rows live in
    # meep_gpu/metal_dispatch.py and NOT in fastpath.py, which is the whole cost
    # argument for that file existing: a Metal release edit must not re-drift the
    # Triton driver_dispatch record. So the module is chosen here rather than
    # `fastpath` being read for both.
    if arguments.backend == "metal":
        from meep_gpu import metal_dispatch as table  # noqa: PLC0415
        released_arms = dict(table.METAL_RELEASED_FUSED_ARMS)
        envelope = table.METAL_FUSED_RELEASE_ENVELOPE
        per_arm_axes = table.METAL_FUSED_RELEASE_ARM_AXES
        route_gate = table.METAL_DRIVER_ROUTE_GATE
    elif arguments.backend == "cuda":
        from meep_gpu import fastpath_cuda as table  # noqa: PLC0415
        released_arms = dict(table.CUDA_RELEASED_FUSED_ARMS)
        envelope = table.CUDA_FUSED_RELEASE_ENVELOPE
        per_arm_axes = table.CUDA_FUSED_RELEASE_ARM_AXES
        route_gate = table.CUDA_DRIVER_ROUTE_FUSED_GATE
        if backend["bound"] is None:
            backend = dict(backend, bound=_cuda_bound_files())
    else:
        table = fastpath
        released_arms = dict(fastpath.RELEASED_FUSED_ARMS)
        envelope = fastpath.FUSED_RELEASE_ENVELOPE
        per_arm_axes = fastpath.FUSED_RELEASE_ARM_AXES
        route_gate = fastpath.DRIVER_ROUTE_FUSED_GATE

    run = RESULTS / arguments.run
    if not run.is_dir():
        print(f"REFUSING: {run} is not a directory", file=sys.stderr)
        return 2
    # THE CONSTANT AND THE ARTIFACT MUST NAME EACH OTHER. The release table cites a
    # run directory by name and the campaign writes into one; a record cut from a
    # directory the table does not cite would be provenance for a measurement the
    # shipped code never claims.
    #
    # THE NVIDIA TABLES ARE CHECKED BELOW INSTEAD, against the per-capability
    # campaign name their legs' stamp decides
    # (``run_is_not_this_capability_campaign``); the constant alone cannot be the
    # whole name once one release is driven on more than one architecture. Triton
    # used to only PRINT this disagreement, which is how records came to be cut from
    # re-run and superseded directories.
    # THE METAL TABLE IS CHECKED BELOW TOO, against the per-architecture campaign name
    # its legs' recorded GPU decides (``run_is_not_this_architecture_campaign``).
    legs = _legs(run)
    names = tuple(sorted(p.parent.name for p in legs))
    allowed = tuple(sorted(set(required) | (set(names) & set(optional))))
    if names != allowed:
        print(f"REFUSING: {arguments.run} carries legs {names}; the campaign is "
              f"{required}"
              + (f" plus optionally {optional}" if optional else ""),
              file=sys.stderr)
        return 2

    document = json.loads(record_path.read_text(encoding="utf-8"))
    if "driver_dispatch" not in document:
        # THE FIRST CUT OF A BACKEND'S BLOCK. `bound` is then the constant above
        # rather than the previous block's key set, which is the only honest source
        # for a record that does not exist yet. Every later cut reads the block.
        if backend["bound"] is None:
            print(f"REFUSING: {record_path} has no driver_dispatch block and this "
                  f"backend declares no bound file set to create one from",
                  file=sys.stderr)
            return 2
        document["driver_dispatch"] = {"source_sha256": {name: None for name
                                                         in backend["bound"]},
                                       "exclusions": {"released_fused_arms": {}}}
        # THE CONTAINER IS PART OF THE EMPTY SHAPE on every table, so a first cut is
        # in the per-capability (per-architecture, on Metal) shape before it writes a
        # record rather than after.
        document["driver_dispatch"][fastpath.RUNS] = {}
    record = document["driver_dispatch"]
    record.setdefault("exclusions", {}).setdefault("released_fused_arms", {})
    bound = sorted(record["source_sha256"])

    problems, artifacts = [], {}
    if arguments.backend in ("triton", "cuda"):
        # THE SHAPE IS REFUSED BY NAME, never read both ways. A record carrying the
        # route run's fields beside its digests, or the typed capability list the
        # ledger used to declare, has two answers to "which device certified this" --
        # the one a tool wrote and the one a hand edit left -- and the whole point of
        # the per-capability container is that there is one. Such a record is
        # repaired by migrate_capability_records.py, not by this tool.
        if "validated_compute_capabilities" in record:
            print("REFUSING: record_carries_a_typed_capability_list: "
                  "driver_dispatch.validated_compute_capabilities is the hand-typed "
                  "declaration the per-capability records replace; the admitted set "
                  "is now DERIVED from the welds (fastpath.capability_admission). "
                  "Migrate the record first.", file=sys.stderr)
            return 2
        stranded = fastpath.retired_shape_reasons(
            record, run_fields=fastpath.DISPATCH_RUN_FIELDS)
        if stranded:
            print("REFUSING: record_carries_route_run_fields_at_entry_level: "
                  f"{'; '.join(stranded)}. The route run's facts belong in "
                  f"{fastpath.RUNS}[<capability>]; migrate the record first.",
                  file=sys.stderr)
            return 2
    if arguments.backend == "metal":
        # THE SAME REFUSAL ON THE METAL RECORD, keyed by GPU architecture: a record
        # still holding its route run beside its digests is migrated by
        # migrate_metal_runs.py, never read both ways here.
        stranded = metal_runs.shape_reasons(
            record, run_fields=metal_runs.DISPATCH_RUN_FIELDS)
        if stranded:
            print("REFUSING: record_carries_route_run_fields_at_entry_level: "
                  f"{'; '.join(stranded)}. The route run's facts belong in "
                  f"{metal_runs.RUNS}[<architecture>]; run migrate_metal_runs.py first.",
                  file=sys.stderr)
            return 2
    # THE BOUND BEFORE THE REFRESH, which is the only moment it can be read: the
    # digests below are rewritten in place, and ``bind_capability`` needs to know
    # whether the bytes MOVED to decide whether the other capabilities' records
    # survive this cut.
    bound_before = fastpath.bound_digest(record)
    # THE ONE NON-BLOCKING LEG (see BACKENDS['cuda']): an unreleased cuda_alone is
    # held here and recorded below, never added to ``problems``.
    unreleased_optional: dict = {}
    for leg in legs:
        artifact = json.loads(leg.read_text(encoding="utf-8"))
        artifacts[leg.parent.name] = artifact
        if artifact.get("release", {}).get("released") is not True:
            if arguments.backend == "cuda" and leg.parent.name == CUDA_ALONE_LEG:
                unreleased_optional[leg.parent.name] = dict(artifact.get("release")
                                                            or {})
                continue
            problems.append(f"{leg.parent.name}: released="
                            f"{artifact.get('release', {}).get('released')!r} "
                            f"{artifact.get('release', {}).get('reasons')}")

    # WHICH ARCHITECTURE THIS RECORD IS ABOUT, READ OFF THE LEGS AND NEVER TYPED.
    # It is the key the run's facts are filed under, so a wrong answer files one
    # device's measurement as another's; and the legs must agree, because a campaign
    # that ran on two architectures is a bit-identity claim about neither. Both
    # NVIDIA tables are under the rule: the CUDA branch alone used to refuse a mixed
    # campaign, and the Triton record simply had no device field at all.
    capability = None
    if arguments.backend in ("triton", "cuda"):
        stamped = stamped_capabilities(artifacts)
        values = sorted({value for value in stamped.values() if value})
        if len(values) > 1:
            print(f"REFUSING: legs_stamp_different_compute_capabilities: {stamped}",
                  file=sys.stderr)
            return 1
        if not values:
            print("REFUSING: no_leg_stamped_a_compute_capability: no leg's gate.json "
                  "records provenance.device.compute_capability, so the run's facts "
                  f"cannot be filed under one ({stamped}). Re-run the campaign with a "
                  "gate that stamps the device.", file=sys.stderr)
            return 1
        unstamped = sorted(leg for leg, value in stamped.items() if not value)
        if unstamped:
            print(f"REFUSING: no_leg_stamped_a_compute_capability: {unstamped} record "
                  f"no provenance.device.compute_capability beside legs that stamped "
                  f"{values[0]}; a leg whose device cannot be read is not evidence "
                  f"about this one.", file=sys.stderr)
            return 1
        capability = values[0]
        # ONE STAMP PER RELEASE, ONE CAMPAIGN PER CAPABILITY UNDER IT. The directory
        # name is the only place a reader of the results tree sees which architecture
        # a route campaign drove, so it is derived here rather than compared by eye.
        expected_run = fastpath.route_campaign(route_gate, capability)
        if arguments.run != expected_run:
            print(f"REFUSING: run_is_not_this_capability_campaign: the legs stamp "
                  f"compute capability {capability}, so {arguments.backend}'s release "
                  f"cites {expected_run!r} (fastpath.route_campaign of {route_gate!r}) "
                  f"and this run is {arguments.run!r}. The constant edit belongs "
                  f"BEFORE the campaign, never after.", file=sys.stderr)
            return 2

    # WHICH APPLE GPU THIS RECORD IS ABOUT, read off the legs the same way: the route
    # gate stamps ``provenance.apple_gpu`` on every leg, so the pinned gate does not
    # move for this. A campaign whose legs name two architectures, or a leg that names
    # none, refuses -- a route record filed under the wrong GPU is the label/evidence
    # mismatch the per-architecture records exist to close.
    architecture = None
    if arguments.backend == "metal":
        stamped = {leg: ((artifact.get("provenance") or {}).get("apple_gpu") or {})
                   .get("architecture") for leg, artifact in artifacts.items()}
        values = sorted({value for value in stamped.values() if value})
        unstamped = sorted(leg for leg, value in stamped.items() if not value)
        if len(values) != 1 or unstamped:
            print(f"REFUSING: legs_do_not_name_one_gpu_architecture: {stamped}; every "
                  f"leg's gate.json must record provenance.apple_gpu.architecture, and "
                  f"all the same one", file=sys.stderr)
            return 1
        try:
            architecture = metal_runs.require_architecture(values[0])
            expected_run = metal_runs.route_campaign(route_gate, architecture)
        except metal_runs.ArchitectureRecordError as refused:
            print(f"REFUSING: legs_do_not_name_one_gpu_architecture: {refused}",
                  file=sys.stderr)
            return 1
        if arguments.run != expected_run:
            print(f"REFUSING: run_is_not_this_architecture_campaign: the legs ran on "
                  f"{architecture}, so the Metal release cites {expected_run!r} "
                  f"(metal_runs.route_campaign of {route_gate!r}) and this run is "
                  f"{arguments.run!r}. The constant edit belongs BEFORE the campaign, "
                  f"never after.", file=sys.stderr)
            return 2
        for name in supersede:
            try:
                metal_runs.require_architecture(name)
            except metal_runs.ArchitectureRecordError as refused:
                print(f"REFUSING: --supersede: {refused}", file=sys.stderr)
                return 2

    # ONE CAMPAIGN, ONE TREE, AND THE TREE THAT SHIPS. Three comparisons, because
    # the defect had three digests: leg against leg, leg against the file on disk.
    ran = {}
    for name in bound:
        live = hashlib.sha256(_resolve(arguments.backend, name).read_bytes()).hexdigest()
        key = _leg_key(arguments.backend, name)
        seen = {leg: (artifact.get("provenance") or {}).get("source_sha256", {}).get(key)
                for leg, artifact in artifacts.items()}
        if len(set(seen.values())) != 1:
            problems.append(f"{name}: the legs disagree {seen}")
            continue
        value = next(iter(seen.values()))
        if value is None:
            problems.append(f"{name}: no leg recorded a digest for it under "
                            f"provenance.source_sha256[{key!r}]; a record may not "
                            f"bind bytes no leg is recorded as having run")
            continue
        if value != live:
            problems.append(f"{name}: the campaign ran {value} and the tree ships "
                            f"{live}; re-run the gate against the bytes that ship")
            continue
        ran[name] = value

    # THE CASE LIST IS THE RUN'S, not the constant's. The release names the cases
    # each arm was driven on; this proves each pair against a leg that actually
    # dispatched, so a case nobody drove cannot enter the record.
    #
    # A REFUSAL LEG IS NOT A WITNESS. The Triton campaign identifies one by
    # `uncertified_policy_installed`; the Metal campaign additionally carries a
    # `band_witness` leg that installs keep and runs the array path alone, and
    # asking it which arms it drove would find none and refuse every row.
    dispatching = {leg: artifact for leg, artifact in artifacts.items()
                   if not artifact.get("uncertified_policy_installed")
                   and leg not in NON_DISPATCHING_LEGS
                   and leg not in unreleased_optional}
    driven_by_case = {}
    driven_by_leg = {}
    for leg_name, artifact in dispatching.items():
        for case, arms in (artifact.get("arms_driven") or {}).items():
            driven_by_case.setdefault(case, set()).update((arms or {}).values())
            driven_by_leg.setdefault((leg_name, case), set()).update(
                (arms or {}).values())
    for arm, cases in released_arms.items():
        for case in cases:
            if arm not in driven_by_case.get(case, set()):
                problems.append(f"the release says {arm!r} was driven on "
                                f"{case!r}; no dispatching leg's artifact shows it")

    # THE DEVICE EVERY DISPATCHING ROW PLANNED AGAINST, read off that row's own plan.
    # This is what keeps a capability admitted uncertified (a supported one by
    # default, any one under ``MEEP_GPU_ALLOW_UNCERTIFIED=1``) from producing a
    # record: the admission takes a run past rung 4b, the kernels launch, and
    # the plan keeps answering ``device_certified_by_table[t]`` False -- so a campaign
    # driven that way would otherwise cut a record for an architecture no cited weld
    # had run on, and the record would then be the evidence admitting it. Checked per
    # ROW rather than per leg because the answer is per table and a row's composition
    # names the tables that served it.
    screen = (_uncertified_metal_dispatch_reasons if arguments.backend == "metal"
              else _uncertified_dispatch_reasons)
    if arguments.backend in ("triton", "cuda", "metal"):
        for leg_name in sorted(dispatching):
            rows_path = run / leg_name / "cases.jsonl"
            if not rows_path.is_file():
                problems.append(
                    f"{leg_name}: no cases.jsonl, so no row's plan can be read for the "
                    f"device it dispatched against; re-run the current gate")
                continue
            for line in rows_path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                plan = ((row.get("legs") or {}).get("fused") or {}).get("plan") or {}
                tables = (plan.get("composition") or {}).get("tables_dispatched") or []
                problems.extend(
                    f"{leg_name}: row_dispatched_on_an_uncertified_device: {reason}"
                    for reason in screen(row.get("case"), plan, tables))

    if arguments.backend == "cuda":
        # THE WALK IS ``(arm, case, LEG)`` ON THIS TABLE, and the third dimension is
        # not bookkeeping. A complex-storage arm's expansion licence comes from the
        # unified probe record, offered on ONE leg; the same arm appearing in the
        # shipped leg's ``arms_driven`` would mean the licence had been consumed
        # WITHOUT the probe, which is the one thing that leg exists to keep apart
        # and the one failure a case-only walk cannot see.
        for arm, legs in table.CUDA_RELEASED_ARM_LEGS.items():
            for case in released_arms.get(arm, ()):
                if not any(arm in driven_by_leg.get((leg, case), set())
                           for leg in legs):
                    problems.append(
                        f"the release says {arm!r} was driven on {case!r} from one "
                        f"of {list(legs)}; no such leg's artifact shows it")
        # THE ARBITRATION IS PART OF THE RELEASE, because the number this record
        # publishes is conditioned on it. Every dispatching case must carry the
        # preference leg's INCUMBENT-YIELDED and the default leg's
        # ARBITRATION-DEFAULT-HELD (or YIELD-MEASURED, where a yield case was
        # driven), or the record would be asserting an arbitration nothing measured.
        # THE ARTIFACT CARRIES A LIST PER CASE, one entry per arbitration control,
        # and a bare string is refused rather than accepted as a one-element list.
        # A campaign whose summary reports a single verdict per case measured three
        # and published one, and which one survived depended on control ORDER -- so
        # the scalar spelling cannot be read as evidence about either verdict this
        # block requires. Re-run the leg against the current gate instead.
        verdicts = {}
        for leg_name, artifact in artifacts.items():
            if leg_name in unreleased_optional:
                continue
            for case, verdict in (artifact.get("arbitration") or {}).items():
                if isinstance(verdict, str):
                    problems.append(
                        f"{leg_name}'s artifact reports arbitration for {case!r} as "
                        f"a single verdict {verdict!r}; the campaign runs three "
                        "controls per case and the summary must carry all of them")
                    continue
                verdicts.setdefault(case, set()).update(
                    v for v in (verdict or []) if v is not None)
        dispatching_cases = sorted({case for cases in released_arms.values()
                                    for case in cases})
        for case in dispatching_cases:
            seen = verdicts.get(case, set())
            if "INCUMBENT-YIELDED" not in seen:
                problems.append(
                    f"{case} carries no INCUMBENT-YIELDED verdict; the preference "
                    "leg has to show the CUDA unit taking the slots and the Triton "
                    "twin's kernels reading zero on them")
            if not seen & {"ARBITRATION-DEFAULT-HELD", "YIELD-MEASURED"}:
                problems.append(
                    f"{case} carries neither ARBITRATION-DEFAULT-HELD nor "
                    "YIELD-MEASURED; the default-precedence number would be an "
                    "assertion rather than a measurement")

    # THE DEFAULT'S LICENCE, decided before anything is written, and FILED UNDER ONE
    # CAPABILITY ON ONE TABLE. ``primary_table`` is the first table in the shipped
    # precedence that admits this capability -- the table whose plan composes first,
    # and therefore the composition the ship leg actually measured. A second licence
    # on the other table would be a second answer to "what licenses the default" that
    # no run distinguishes.
    primary = fastpath.primary_table(capability) if capability else None
    licence = None
    licence_note = None
    previous_run = (record.get(fastpath.RUNS) or {}).get(capability) or {}
    # THE KEEP RULE IS THE BOUND, now that the licence carries no digest map of its
    # own: the existing licence travels forward only while the slot it sits in still
    # binds the bytes this campaign ran, which is exactly what ``bound_sha256`` says.
    # Computed from the refresh that happens below rather than after it, because
    # everything in this block has to be decided before a field is written; the guard
    # keeps a digest disagreement above from also reading as a stale licence.
    bound_after = (fastpath.bound_digest(dict(record, source_sha256=dict(ran)))
                   if len(ran) == len(bound) else None)
    kept = previous_run.get("dispatch_by_default_licence")
    if capability and primary is None:
        # NOTHING ADMITS IT, so there is nowhere for the licence to live and no arm
        # the shipped code would dispatch on this device: rung 4b refuses it. A
        # record cut here would be provenance for a dispatch the package declines,
        # and the admission is DERIVED, so the repair is to certify the welds (step 2
        # of the protocol), never to cut this record first.
        problems.append(
            f"capability_is_admitted_by_no_table: no NVIDIA table's cited welds have a "
            f"live record for compute capability {capability} "
            f"(fastpath.capability_admission names the keys that hold it back), so "
            f"this route run certifies nothing the package would dispatch")
    elif arguments.end_to_end:
        if arguments.backend == "metal":
            problems.append("--end-to-end transcribes gate_dispatch_end_to_end, an "
                            "NVIDIA gate; the Metal table's default is licensed by its "
                            "route campaign's shipped leg")
        elif arguments.backend != primary:
            problems.append(
                f"licence_belongs_to_the_primary_table: compute capability "
                f"{capability} is licensed on the {primary!r} record (the first table "
                f"in {list(fastpath.NVIDIA_TABLE_PRECEDENCE)} that admits it, so the "
                f"one whose plan composes first) and this cut is {arguments.backend!r}")
        else:
            licence, licence_problems = transcribe_end_to_end(
                arguments.end_to_end, ran, bound, arguments.backend, capability)
            problems.extend(licence_problems)
    elif arguments.backend in ("triton", "cuda"):
        if kept and arguments.backend != primary:
            # NOT CARRIED FORWARD, and that is not a loss: one capability has one
            # licence, on the table whose plan composes first. A copy here would be a
            # second block nothing distinguishes from the first once the bytes move.
            licence_note = (
                f"NOTE: the licence this record carried for {capability} is not "
                f"carried forward; compute capability {capability} is licensed on the "
                f"{primary or 'primary'} record")
            kept = None
        elif kept and bound_after is not None and previous_run.get(
                "bound_sha256") != bound_after:
            problems.append(
                f"licence_was_cut_on_other_bytes: {fastpath.RUNS}[{capability!r}] "
                f"binds {str(previous_run.get('bound_sha256'))[:12]} and this campaign "
                f"binds {bound_after[:12]}, so its licence is evidence about code that "
                f"no longer ships. Re-run gate_dispatch_end_to_end's ship leg on these "
                f"bytes and pass --end-to-end")
            kept = None
        elif not kept and fastpath.DISPATCH_BY_DEFAULT:
            if arguments.backend == primary:
                problems.append(
                    f"licence_is_missing: fastpath.DISPATCH_BY_DEFAULT is True and "
                    f"{fastpath.RUNS}[{capability!r}] carries no "
                    f"dispatch_by_default_licence on the primary table for this "
                    f"capability: run gate_dispatch_end_to_end's ship leg on these "
                    f"bytes and pass --end-to-end")
            else:
                licence_note = (
                    f"NOTE: dispatch_by_default is True and this record carries no "
                    f"dispatch_by_default_licence for {capability}; the "
                    f"{primary or 'primary'} record's is the one the tests bind, and "
                    f"--end-to-end is refused here by name")

    if problems:
        print("REFUSING to re-cut:", file=sys.stderr)
        for line in problems:
            print(f"  {line}", file=sys.stderr)
        return 1

    # The measured substitution, read per case off the legs rather than restated.
    # EXACT-ABSORBED-ARRAY-SLOT is the route gate's spelling for a proof on a row
    # whose drive entry declares a slot the baseline left on the array path; the
    # counted drop there is smaller by one per such slot, and it is listed here the
    # same way so the record shows what was measured.
    drops = {}
    for artifact in dispatching.values():
        for case in artifact.get("dispatched_cases") or ():
            row = (artifact.get("substitution") or {}).get(case)
            if row in EXACT_SUBSTITUTION_VERDICTS:
                drops.setdefault(case, set())
    for leg_dir in (run / leg for leg in dispatching):
        rows_path = leg_dir / "cases.jsonl"
        if not rows_path.is_file():
            continue
        for line in rows_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            substitution = row.get("substitution") or {}
            if substitution.get("verdict") not in EXACT_SUBSTITUTION_VERDICTS:
                continue
            drops.setdefault(row["case"], set()).add(
                (substitution["unfused"]["launches_per_step"],
                 substitution["fused"]["launches_per_step"]))
    measured = ", ".join(
        f"{case} {int(before)}->{int(after)}"
        for case in sorted(drops)
        for before, after in sorted(drops[case]))

    cases = sorted({case for cases in released_arms.values() for case in cases})
    previous = record["source_sha256"].copy()
    record["source_sha256"] = dict(ran)
    record["code_sha256"] = {name: code_digest_of_path(_resolve(arguments.backend, name))
                             for name in bound}
    # A FIRST CUT SUPERSEDES NOTHING, so it writes no log. The log records which
    # digests MOVED and what they moved from; on a block that did not exist, every
    # row reads ``from: null``, which is not history -- it is the cut itself said
    # twice. Keeping it put twenty superseded-shaped digests into a ledger that had
    # never argued for a historical exemption, and the CUDA weld contract refused
    # the tree by name for exactly that.
    #
    # THE KEY IS UNDERSCORE-PREFIXED, as the Triton ledger spells it: the leading
    # underscore is what marks an annotation the digest walker reads as narrative
    # rather than as a pin it must check.
    moved = [{"file": name, "from": previous.get(name), "to": ran[name]}
             for name in bound
             if previous.get(name) is not None and previous.get(name) != ran[name]]
    if moved:
        record.setdefault("_source_sha256_recut_log", []).append({
            "recut_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "cut_from": f"apps/api/parity/meep_gpu/results/{arguments.run}/",
            "files": moved,
            "backend": arguments.backend,
            "why": ("D1: this block, the artifact the release cites and the tree had "
                    "THREE different digests for fastpath.py at once (record 06cdbf37, "
                    "tree 87f740ee then d11454ba, and all four legs of the cited run "
                    "29ac1216), so the release authority had never executed the code "
                    "citing it. Every source edit of this round landed before the "
                    "campaign started, so the four legs recorded the shipping bytes "
                    "and these digests are READ from that recording by "
                    "parity/meep_gpu/recut_driver_dispatch_record.py, which refuses "
                    "unless leg, leg and tree all agree."),
            "not_a_device_claim": (
                "the *_device_gate welds that record fastpath.py are re-run on "
                "hardware, never re-typed; this tool does not touch one."),
        })

    # THE RUNNABLE SHAPE IS RE-CUT FROM THE LIVE CONSTANTS, not left where the last
    # campaign put it. ``fill_B``/``fill_D`` joined ``DRIVER_SLOTS`` on 2026-09-02
    # with the two driver fill consults, and a record that still described a
    # five-slot seam would be describing a driver that no longer exists — which is
    # the same defect the source digests above exist to prevent, one field over.
    # ``test_dispatch_contract`` compares all three of these against the tree.
    if arguments.backend == "metal":
        from meep_gpu.metal_kernels import launch as _launch  # noqa: PLC0415
    else:
        # THE CUDA TABLE RE-IMPORTS THIS SAME TUPLE (``cuda_kernels/arms.py``
        # imports ``STEP_ORDER`` from here), so there is one definition of the
        # driver's slot order and this record describes the same seam either way.
        from meep_gpu.triton_kernels import launch as _launch  # noqa: PLC0415
    record["table"] = arguments.backend
    record["runnable_slots"] = list(fastpath.DRIVER_SLOTS)
    unrunnable = sorted(set(_launch.STEP_ORDER) - set(fastpath.DRIVER_SLOTS))
    consequence = record.get("unrunnable_slots", {})
    record["unrunnable_slots"] = {
        name: (consequence.get(name) if isinstance(consequence, dict) else None)
              or "the composer can fill it and the driver has no consult site for "
                 "it, so a plan carrying it is refused whole at rung (6)"
        for name in unrunnable}
    if isinstance(consequence, dict) and "consequence" in consequence:
        record["unrunnable_slots"]["consequence"] = consequence["consequence"]
    record["far_fill_consults"] = dict(fastpath.FAR_FILL_PASSES)

    # THE ROUTE RUN'S OWN FACTS, collected rather than written: on the NVIDIA tables
    # they are filed under ``runs[<capability>]`` at the end of this function, so that
    # a second architecture adds a record instead of overwriting this one's.
    route_run: dict = {}

    if arguments.backend == "cuda":
        # THE ARBITRATION THE NUMBER IS CONDITIONED ON, written into the record so a
        # reader of the block never has to go looking for the condition. Both
        # controls are named, and the two dispatch numbers are kept apart by name
        # rather than by a footnote.
        alone = artifacts.get(CUDA_ALONE_LEG)
        leg_descriptions = {
            "cuda_shipped": f"{fastpath.BACKEND_PREFERENCE_SWITCH}=cuda",
            "cuda_default_precedence": "unset"}
        if alone is not None:
            leg_descriptions[CUDA_ALONE_LEG] = (
                "unset, with Triton made unimportable in the gate's own process "
                "(--without-triton), so the hand-CUDA table composes alone")
        route_run["arbitration"] = {
            "precedence": list(fastpath.BACKEND_PRECEDENCE),
            "preference_switch": fastpath.BACKEND_PREFERENCE_SWITCH,
            "yield_pending_primary_slots": table.YIELD_PENDING_PRIMARY_SLOTS,
            "legs": leg_descriptions,
            "measured": {case: sorted(seen) for case, seen in sorted(verdicts.items())},
            "read_this_as": (
                "every served_in_dispatch count this table publishes is a "
                "BY-PREFERENCE number: under the shipped precedence the "
                "release-gated Triton table composes first and holds every slot "
                "both admit, so the by-default count is what the "
                "cuda_default_precedence leg measured and is reported beside it"),
        }
        # THE POLICY SENTENCE NAMES THE LEGS THAT RAN, read off the run rather than
        # typed: a record cut with cuda_alone lists it among the keeping legs, and
        # one cut without it does not claim it.
        keeping = [leg for leg in ("cuda_shipped", "cuda_default_precedence",
                                   "cuda_harness_keep",
                                   "cuda_shipped_expansion_probe", CUDA_ALONE_LEG)
                   if leg in artifacts]
        route_run["subnormal_policy"] = (
            f"ieee_keep_ftz_stripped ({', '.join(keeping)} — dispatching"
            + ("; cuda_alone governs host and CuPy only, Triton being absent"
               if alone is not None else "")
            + ") + meep_x86_flush (cuda_flush — refused by name at rung 8b)")
        # ABSENT, PRESENT AND RELEASED, OR PRESENT AND NOT RELEASED, said in the
        # record. The third is the non-blocking case: the leg ran, it is not evidence
        # for anything above, and its own release reasons are carried so the reader
        # sees why.
        if alone is not None and CUDA_ALONE_LEG in unreleased_optional:
            route_run["cuda_alone_leg"] = {
                "status": "present, not released",
                "artifact": f"{CUDA_ALONE_LEG}/gate.json",
                "released": unreleased_optional[CUDA_ALONE_LEG].get("released"),
                "reasons": list(unreleased_optional[CUDA_ALONE_LEG].get("reasons")
                                or []),
                "triton_withheld": alone.get("triton_withheld"),
                "what_it_measures": (
                    "NOTHING this record relies on: the leg ran the composition a "
                    "host with no validated Triton gets and did not release, so it "
                    "witnesses no arm, no arbitration and no drop above; the five "
                    "standard legs were cut without it"),
            }
        elif alone is not None:
            route_run["cuda_alone_leg"] = {
                "status": "present, released",
                "artifact": f"{CUDA_ALONE_LEG}/gate.json",
                "released": (alone.get("release") or {}).get("released"),
                "triton_withheld": alone.get("triton_withheld"),
                "what_it_measures": (
                    "the composition a host with no validated Triton gets: the "
                    "preference unset, Triton unimportable in the gate's process, "
                    "rung 4 dropping the Triton table by name and the hand-CUDA "
                    "table composing alone with the array path filling its "
                    "refusals"),
            }
        else:
            route_run["cuda_alone_leg"] = {
                "status": "absent",
                "artifact": None,
                "released": None,
                "triton_withheld": None,
                "what_it_measures": (
                    "NOTHING in this record: the campaign ran no cuda_alone leg, so "
                    "the composition a host with no validated Triton gets — the "
                    "hand-CUDA table composing alone — is unmeasured by it, and "
                    "every leg above ran with Triton importable"),
            }
        record["kernel_module"] = [name for name in bound
                                   if name.startswith("meep_gpu/cuda_kernels/")]

    if arguments.backend in ("triton", "cuda"):
        # READ, NEVER TYPED, on BOTH NVIDIA records. The CUDA branch used to write a
        # literal False and the Triton branch wrote nothing at all, leaving whatever
        # the last hand edit said; either way a flipped constant produced a record
        # that contradicted it, which only another hand edit could fix.
        record["dispatch_by_default"] = fastpath.DISPATCH_BY_DEFAULT
    if arguments.backend == "triton":
        # THE SWITCH NAMES, from the constants. Both fields named the retired
        # ``*_FDTD_*`` spellings, which rung 2 now refuses by name — so the
        # record described a switch that turns a run into a refusal. The history
        # blocks that quote the old names (``driver_route_gate_2026-08-15``,
        # ``what_would_certify_this``) are measurements at their time and are left.
        record["enable"] = _enable_statement()
        if isinstance(record.get("kill_switch"), dict):
            record["kill_switch"]["variable"] = fastpath.FUSED_KILL_SWITCH
        # THE LEDGER'S OTHER STATEMENT OF THE DEFAULT, outside this block and cut
        # with it for the same reason: see :func:`_pending_dispatch_claim`.
        pending = document.get("pending_host_recut")
        if isinstance(pending, dict):
            pending["dispatch"] = _pending_dispatch_claim()

    released = record["exclusions"]["released_fused_arms"]
    released["gate"] = arguments.run
    released["arms"] = sorted(released_arms)
    released["driven_on_cases"] = cases
    released["per_arm_cases"] = {arm: list(value) for arm, value
                                 in released_arms.items()}
    released["envelope"] = {
        axis: {"required": sorted(value) if isinstance(value, frozenset) else value,
               "measured": why}
        for axis, value, why in envelope}
    # THE OTHER HALF OF THE PREDICATE. Since 2026-09-02 the release is decided by
    # the shared envelope AND by each arm's own axes, and a record carrying only
    # the first would UNDERSTATE what admits — a reader would conclude the folded
    # pairs run on every 2-D PML grid, which is the over-claim this whole block is
    # against. Both tables are read from the module, never transcribed.
    released["per_arm_axes"] = {
        arm: {axis: {"required": sorted(value)
                                 if isinstance(value, frozenset) else value,
                     "measured": why}
              for axis, value, why in axes}
        for arm, axes in per_arm_axes.items()}
    released["how_the_two_tables_combine"] = (
        "released_fused_arms(shape) admits an arm when the SHARED envelope holds "
        "AND that arm's own axes hold. 'required': null means the axis must be "
        "ABSENT from the run shape; 'required': 'required' means it must be "
        "PRESENT (how a folded product says it only exists on a fold); a list "
        "means any of those values.")
    released["artifact"] = (
        f"apps/api/parity/meep_gpu/results/{arguments.run}/ -- {len(names)} legs "
        f"({', '.join(names)}), all release.released=true; the "
        f"{len(cases)} in-envelope cases driven with MEEP_GPU_FUSE_ARMS UNSET, so "
        "RELEASED_FUSED_ARMS did the admitting, with substitution EXACT against a "
        "MEEP_GPU_FUSE_ARMS=0 baseline and the envelope shown declining "
        "pml_3d_diagonal in both directions. Supersedes "
        "dispatch_fused_route_2026-08-29_veto, which drove the same route but ran "
        "fastpath.py at 29ac1216 -- bytes that never shipped -- and "
        "dispatch_fused_route_2026-08-29_seamgate before it, which reached the "
        "arms through the MEEP_GPU_FUSE_ARMS opt-in.")
    released["what_was_measured"] = (
        "the arm's kernel LAUNCHED from FastPathPlan.dispatch at the leading "
        "consult inside driver.step(), on a driver lifted by lift_simulation from "
        "a real mp.Simulation; the absorbed consult answered True off the sentinel "
        "or the deposit repair; the whole run was byte-compared as uint32 against "
        "the same driver with MEEP_GPU_FUSED=0 at ten checkpoints and at the end, "
        "plus flux spectra, first_divergent_checkpoint null everywhere; and device "
        "launches per complete step dropped by exactly one per pair against a third "
        f"leg that dispatched the same slot set with fusion off ({measured}), "
        "counted twice over. synchronize_magnetic_fields -- the second consult "
        "site -- was exercised in the same runs. BOTH installed pair shapes are "
        "covered: magnetic_seam_2d carries a magnetic source, so its withheld "
        "control reports leading_repair_slots_left_running {\"step_B\": 12} where "
        "every earlier case reported {\"step_D\": 12} and the B-seam deposit repair "
        "went undriven.")

    if arguments.backend == "metal":
        # THE THREE FACTS THIS TABLE HAS AND THE OTHER ONE DOES NOT, each read off
        # the run or the module rather than typed: the residency seam it executed
        # over, the policy it certified under with the measured cliff behind that
        # choice, and the toolchain pair the ledger has actually seen.
        from meep_gpu import metal_dispatch as _table  # noqa: PLC0415

        residency = {leg: (artifact.get("residency") or {})
                     for leg, artifact in dispatching.items()}
        # THE MODE IS READ OFF THE LEGS, as metal_dispatch wrote it into each case's
        # dispatch record, never typed. The sentence this replaces said "held-mirror
        # modes are deferred" while every campaign since 2026-09-21 ran held, so the
        # record described a transport its own legs never executed.
        modes, invariants = {}, set()
        for leg_name, per_case in sorted(residency.items()):
            for case, block in sorted((per_case or {}).items()):
                if isinstance(block, dict) and block.get("mode"):
                    modes.setdefault(block["mode"], []).append(f"{leg_name}/{case}")
                    if block.get("invariant"):
                        invariants.add(block["invariant"])
        mode_of_record = (next(iter(modes)) if len(modes) == 1
                          else sorted(modes) if modes else None)
        # THE RESIDENCY AND THE LIFT ARE READ OFF THIS ARCHITECTURE'S LEGS, case by
        # case, so they are run facts: written into ``runs[<architecture>]`` below,
        # where a second Mac's cut adds its own beside them instead of overwriting.
        route_run["residency"] = {
            "invariant": (next(iter(invariants)) if len(invariants) == 1
                          else sorted(invariants) if invariants
                          else "unrecorded: no dispatching leg's case carried one"),
            "mode": mode_of_record,
            "mode_read_from": (
                "residency[case].mode on every dispatching leg's gate.json, as "
                "metal_dispatch wrote it into the case's dispatch record; a list "
                "means the legs executed more than one mode"),
            "cases_by_mode": {mode: len(where) for mode, where in sorted(modes.items())},
            "what_the_modes_are": {
                "shipped": ("a per-launch sync_in/sync_out bracket installed at plan "
                            "time by metal_kernels.launch.wrap_for_residency"),
                "held": ("mirrors held on the device between launches under "
                         "identity-keyed ownership, the host written through the "
                         "sparse host_writes doors and read back at the read "
                         "barrier; a mirror that cannot be held falls back to the "
                         "per-launch bracket and is listed in hold_refused"),
            },
            "per_case": residency,
            "verify_control": (
                "residency.names non-empty and residency.verify() == {} read "
                "IMMEDIATELY after each sync_out -- not at the step boundary, where "
                "a trailing deposit repair legitimately leaves the mirror behind "
                "the host (measured 2026-09-10 on bloch_2d: {'Ez': 2, 'Dz': 2, "
                "'f_w_Ez': 2}, which is the host-authoritative invariant working). "
                "Armed by dropping one sync_in per step and by moving a wrapper's "
                "bracket from its inner launch to the whole slot; both must diverge."),
        }
        record["subnormal"] = {
            "certification_policy": _table.certification_policy(),
            "governed_executors": list(_table.GOVERNED_EXECUTORS),
            "host_mechanism": "fenv:FE_DFL_DISABLE_DENORMS_ENV",
            "mps": "native (Metal fast-math denormal flushing); no lever exists",
            "overrode_run_resolution": True,
            "why": ("MPS flushes float32 denormals natively and exposes no lever -- "
                    "both denormal pragma spellings are compile errors on this "
                    "toolchain -- so flush is the only attainable device policy, and "
                    "this arm64 host resolves keep because MEEP's set_zero_subnormals "
                    "is the #if HAVE_IMMINTRIN_H no-op here. The run is therefore "
                    "ONE policy on both executors, reached by driving the host."),
            "measured_cliff": (
                "under a KEEPING host the released pairs are byte-identical to the "
                "array path at 12 complete steps and DIVERGE at 24, with every "
                "differing element inside a few ULP of the subnormal band; the "
                "harness_keep leg's cliff control records the bound it was confined "
                "to."),
        }
        # WHAT THE CITED WELDS CERTIFY, AS OF THIS CUT. Entry-level, so a later rebind
        # (a second Mac's runs joining) changes the ledger and leaves this copy behind
        # until the next cut; the admission never reads it, it reads the runs.
        recorded = _table.recorded_environments()
        record["environments"] = {
            "recorded": recorded,
            "read_from": ("the live per-architecture runs (runs[<architecture>]: its "
                          "key, torch and metal_frontend) of the welds "
                          "metal_dispatch.ARM_CERTIFICATION cites, in "
                          "metal_kernels/fingerprints.json, as of this cut; a later "
                          "rebind changes them until the next cut, and the dispatch "
                          "admission reads the ledger, never this copy"),
        }
        # THE TOOLCHAIN PAIRS THOSE RUNS RECORDED, rewritten on every cut for the same
        # reason. Before 2026-10-05 no tool in this tree wrote the block: it was carried
        # from cut to cut, still naming the one-run weld host strings it was read from.
        record["toolchain"] = {
            "validated": [list(pair) for pair in sorted(
                {(row["torch"], row["metal_frontend"])
                 for row in recorded if row["architecture"]}, key=repr)],
            "read_from": ("the torch and metal_frontend of the live per-architecture "
                          "runs of the welds metal_dispatch.ARM_CERTIFICATION cites, "
                          "as of this cut (environments.recorded)"),
        }
        record["pending_device_gate_arms"] = sorted(
            _table.METAL_PENDING_DEVICE_GATE_ARMS)
        # HOW THE LEGS WERE LIFTED, read off the legs. The route gate stamps
        # ``provenance.lift`` since 2026-09-27, when it began lifting Apple GPU
        # drivers (``prefer_gpu=True``); a leg without the stamp was lifted
        # ``prefer_gpu=False`` under the enable, which planned Metal before that date
        # and is the NumPy reference (no planner) after it.
        lifts = {leg: ((artifact.get("provenance") or {}).get("lift") or {})
                 for leg, artifact in dispatching.items()}
        spellings = sorted({
            ("lift_simulation(prefer_gpu=True) (an Apple GPU driver: NumPy host "
             "arrays, driver.gpu == 'metal' checked after every lift)")
            if lift.get("prefer_gpu") is True and lift.get("driver_gpu") == "metal"
            else "lift_simulation(prefer_gpu=False)"
            for lift in lifts.values()})
        lifted_by = " and ".join(spellings) if spellings else (
            "lift_simulation (no dispatching leg recorded how)")
        route_run["lift"] = {"per_leg": lifts, "read_from": (
            "provenance.lift on every dispatching leg's gate.json; an empty block is "
            "a leg cut before 2026-09-27, lifted prefer_gpu=False under the enable")}
        released["what_was_measured"] = (
            "the arm's kernel LAUNCHED from FastPathPlan.dispatch at the leading "
            "consult inside driver.step(), under the residency mode(s) the legs "
            f"recorded ({mode_of_record}; see residency.what_the_modes_are), on a "
            f"driver lifted by {lifted_by} from a real "
            "mp.Simulation; the absorbed consult answered True off the sentinel or "
            "the deposit repair; the whole run was byte-compared as uint32 against "
            "the same driver with MEEP_GPU_FUSED=0 at every rung of the ladder and "
            "at the end, first_divergent_checkpoint null everywhere; and compiled "
            "Metal function calls per complete step dropped by exactly one per pair "
            f"against a leg that dispatched the same slot set with fusion off "
            f"({measured}), counted twice over -- KernelPlan.launches and a "
            "CountingFunction wrapper on every compiled callable. "
            "programs_per_dispatch is None on this table BY CONSTRUCTION (a Metal "
            "KernelPlan spells no launch grid) and is explicitly not the witness.")
        released["artifact"] += (
            " The two complex cases dispatch ONLY on the shipped_expansion_probe "
            "leg; the shipped leg must show them refused by name without a licence. "
            "The band_witness leg installs keep, dispatches nothing, and is the "
            "vacuity floor: the oracle's own trajectory must enter the subnormal "
            "band, or the flush-equivalence this record rests on is about nothing.")

    staled: tuple = ()
    if arguments.backend == "metal":
        # THE ROUTE RUN, FILED UNDER ITS ARCHITECTURE, by the same rule as the NVIDIA
        # records below: the run's subkeys of ``released_fused_arms`` move with it, and
        # a cut on bytes that moved names every other architecture it would strand.
        route_run["released_fused_arms"] = {
            name: released.pop(name) for name in RELEASED_RUN_SUBKEYS
            if name in released}
        route_run["status"] = "PASS"
        route_run["verdict_read_from"] = "release.released"
        route_run["legs"] = [f"{name}/gate.json" for name in names]
        route_run["records"] = f"apps/api/parity/meep_gpu/results/{arguments.run}"
        route_run["recorded_utc"] = datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ")
        try:
            staled = metal_runs.bind_architecture(
                record, bound_before=bound_before, architecture=architecture,
                run=route_run, run_fields=metal_runs.DISPATCH_RUN_FIELDS,
                supersede=supersede)
        except metal_runs.ArchitectureStale as refused:
            print(f"REFUSING: architecture_runs_would_be_stranded: {refused}",
                  file=sys.stderr)
            return 1
        except metal_runs.ArchitectureRecordError as refused:
            print(f"REFUSING: architecture_record_refused: {refused}", file=sys.stderr)
            return 1
    if arguments.backend in ("triton", "cuda"):
        # WHICH CAMPAIGN DROVE THE ARMS IS A FACT ABOUT ONE RUN; which arms the
        # release admits, on which cases, under which envelope is a fact about the
        # shipped module. The first three move with the architecture and go into the
        # slot; the rest is the same statement on every device and stays beside the
        # digests.
        route_run["released_fused_arms"] = {
            name: released.pop(name) for name in RELEASED_RUN_SUBKEYS
            if name in released}
        route_run["status"] = "PASS"
        route_run["verdict_read_from"] = "release.released"
        route_run["legs"] = [f"{name}/gate.json" for name in names]
        route_run["records"] = f"apps/api/parity/meep_gpu/results/{arguments.run}"
        route_run["recorded_utc"] = datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ")
        if licence is not None:
            route_run["dispatch_by_default_licence"] = licence
        elif kept is not None:
            route_run["dispatch_by_default_licence"] = kept
        try:
            staled = fastpath.bind_capability(
                record, bound_before=bound_before, capability=capability,
                run=route_run, run_fields=fastpath.DISPATCH_RUN_FIELDS,
                supersede=supersede)
        except fastpath.CapabilityStale as refused:
            print(f"REFUSING: capability_records_would_be_stranded: {refused}",
                  file=sys.stderr)
            print(f"  --supersede {','.join(refused.names)} says so by name",
                  file=sys.stderr)
            return 1
        except fastpath.CapabilityRecordError as refused:
            print(f"REFUSING: capability_record_refused: {refused}", file=sys.stderr)
            return 1

    print(f"  run          {arguments.run}")
    print(f"  backend      {arguments.backend}  ({record_path})")
    print(f"  legs         {', '.join(names)} (all released)")
    for name in bound:
        marker = ("unchanged" if previous.get(name) == ran[name]
                  else f"was {previous.get(name)}")
        print(f"  {name:<48} {ran[name]}  [{marker}]")
    print(f"  cases        {cases}")
    print(f"  substitution {measured}")
    if arguments.backend in ("triton", "cuda"):
        slot = record[fastpath.RUNS][capability]
        print(f"  capability   {capability}  ({fastpath.RUNS}[{capability!r}], bound "
              f"{slot['bound_sha256'][:12]}; live "
              f"{list(fastpath.live_capabilities(record))})")
        if staled:
            print(f"  superseded   {list(staled)} (their records now read stale)")
        print(f"  default      dispatch_by_default={fastpath.DISPATCH_BY_DEFAULT}"
              f"; licensed on {primary or 'no admitting table'}")
        if licence is not None:
            print(f"  licence      {licence['run']} -- "
                  f"{len(licence['dispatched_cases'])} dispatched, "
                  f"{len(licence['fell_back_cases'])} fell back, of "
                  f"{len(licence['verdicts'])}; null control "
                  f"{(licence.get('null_control_run') or {}).get('run')}")
        elif kept is not None:
            print(f"  licence      kept: {kept.get('run')}")
        if licence_note:
            print(f"  {licence_note}")
    if arguments.backend == "triton" and isinstance(document.get("pending_host_recut"),
                                                    dict):
        print(f"  pending      pending_host_recut.dispatch = "
              f"{document['pending_host_recut']['dispatch'][:72]}...")
    if arguments.backend == "cuda":
        print(f"  cuda_alone   {route_run['cuda_alone_leg']['status']}")
    if arguments.backend == "metal":
        slot = record[metal_runs.RUNS][architecture]
        print(f"  architecture {architecture}  ({metal_runs.RUNS}[{architecture!r}], "
              f"bound {slot['bound_sha256'][:12]}; live "
              f"{list(metal_runs.live_architectures(record))})")
        if staled:
            print(f"  superseded   {list(staled)} (their runs now read stale)")
        print(f"  residency    {slot['residency']['mode']} "
              f"{slot['residency']['cases_by_mode']}")

    if arguments.write:
        record_path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n",
                               encoding="utf-8")
        print(f"  wrote {record_path}")
    else:
        print("  (report only -- pass --write to apply)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
