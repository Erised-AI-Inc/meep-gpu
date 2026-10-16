"""Byte gate for the Metal BFAST real-field PML curl kernel.

THE CLAIM, and the only one: **byte-identity to ``stepping.py`` on this host,
subject to a declared and CHECKED subnormal-free precondition.** Not a stated
tolerance. Leg 7 is the check, and it is required to FIRE on a scaled control — a
precondition never demonstrated to fire is decorative.

THE ORACLE IS IN PROCESS. ``stepping.step_B(fields, pml)`` and the Metal launch run
in ONE process against ONE seed, on real ``Grid``/``Fields``/``PML`` objects, on the
machine that is both target and oracle. The array path IS the reference: it steps
MEEP's per-chunk forms and runs BFAST as MEEP's own pass after each component's curl
(``_bfast_term`` and ``_apply_folded_increment``), which the kernel transcribes in
register (:mod:`meep_gpu.metal_kernels.folded_pass`). There is no in-gate stepping
transcription: every mutation row compares the mutant with ``stepping`` itself. The
one host reconstruction left (:func:`bfast_intermediates`) recomputes the pass's
un-storable intermediates for the subnormal census and steps nothing.

WHAT IS COMPARED: uint32 word equality on float32 storage, never ``allclose``. The
compared arrays are the sub-step's TARGETS, its PML AUXILIARIES (``fu_*``) **and its
THREE IIR STATES** (``f_bfast_*``). The states are the whole point of this family:
they are the only STATE the tail carries, the IIR is MARGINALLY STABLE by
construction (the homogeneous mode of ``F_n = -F_{n-1}`` is ``(-1)^n``, undamped
forever, stepping.py:868-874), so a kernel that is right for one launch and wrong
forever after diverges only in the multi-step leg — and a state error never decays
out. Omitting them is precisely how that defect hides.

THE LEGS:

  0b execution       proof the bytes came off the GPU: mirror residency, the entry
                     point's type, the launch counter, an unlaunched control, and
                     the MEASURED refusal of a host tensor at a Metal entry point
  1  reference       the Metal kernel vs ``stepping.py``, real engine objects, the
                     eight boundary triples in 3-D and the one-partner 2-D cases
                     that reach MEEP's plain one-partner branch, one of them with
                     a k on the invariant z axis
  2  identity        the HAS_BFAST=0 build vs the CERTIFIED ``pml_curl_step``,
                     with a HAS_BFAST=1 control on the same seeds and the same six
                     scalars proving they COULD have moved the states
  3  multi_step      B -> H -> D -> E cycles, the FULL stored inventory compared
                     after EVERY complete driver step, IIR states included
  4  pointer         the state mirrors write ``fields.f_bfast_*`` IN PLACE
  5  refusal         clause 11a: a k on a DECLARED-invariant axis is admitted,
                     the premise (it moves no word of the array path's step)
                     re-measured, and the admitted run byte-compared on the
                     device, both sub-steps
  6  signed_zero     +-0 lattice seeding, census-floored
  7  precondition    the subnormal window, the f_bfast reachability census, control
  8  mutations       armed, launch-counted, three-valued: the pass's own defects
                     (its two roundings, the plain branch, the mask, the state
                     store, the ladder), the previous package form folded into the
                     curl, the certified curl's per-chunk battery and the record
  9  composition     the engine route, and the two halves of the wiring rule: no
                     FOREIGN arm admits on step_B/step_D, and the shipped composer
                     SELECTS this family there — plus the constitutive pair it now
                     fills with the CERTIFIED entry point, checked by object
                     identity. Sweeping the WHOLE predicate table under both
                     absorber states is the composition PROBE's job, and its
                     measured answer is that the foreign exception list is NOT
                     empty: the null constitutive family admits on
                     update_H/update_E under an INACTIVE absorber, which this
                     family refuses by clause 3, so the two cannot collide

PREDICTED NULLS ARE RECORDED, NOT QUIETLY DROPPED. Three defects this gate CANNOT
hold because they are byte-invisible in float32 — the tail's POSITION between the
curl and the mask, the shifted-plus-centre SUM ORDER, and ``curl - adv`` spelled as
a subtraction rather than as an addition of a negation — are named in
``summary.predicted_nulls`` with the reason each is invisible and the SOURCE-TEXT
assertion in ``test_metal_bfast.py`` that pins it instead. That discipline matters
more here than on the Triton track, because there is NO PTX-EQUIVALENT AUDIT on this
executor: ``compile_shader`` exposes no disassembly, so the mutation legs and this
gate are the only arbiters.

CASE DISCIPLINE, inherited and non-negotiable: a non-power-of-two Courant FIRST —
and the marquee corpus row's own ``(1 - kx)/sqrt(3) ~ 0.105679`` is one of them;
every leg proves the step MOVED STATE; the signed-zero census carries a FLOOR
because a census of zero is VACUOUS and not a pass; armed mutations are
launch-counted — INCLUDING THE HOST MUTATIONS (m10, m11), which plant their defect
in the plan's arguments rather than in the source and were previously recorded with
a literal launch count, so a disarmed row would have reported itself CAUGHT (the
untouched input differs from the oracle just as a caught defect does); the top-level
verdict reads BOTH the compare COUNT and the OUTCOME, the outcome re-read from the
recorded rows rather than inferred from having reached the summary.

NaN AND OVERFLOW NEEDLES ARE EXCLUDED, on this track as on the Triton one: NaN sign
and payload are IEEE-unspecified and every Metal op carries NaN-class residuals, so
a needle there measures the toolchain's mood rather than the transcription.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
API_ROOT = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
if API_ROOT not in sys.path:
    sys.path.insert(0, API_ROOT)

# The MPS executor delivers `flush` natively and cannot deliver `keep`, and this
# arm64 host's DEFAULT resolves to keep (MEEP's set_zero_subnormals is a no-op
# under #if HAVE_IMMINTRIN_H, so match_meep measures "keep"). The gate requests
# flush EXPLICITLY, before anything resolves a policy, and stamps the resolution
# into the artifact — the claim is only as good as the precondition it was
# certified under.
os.environ.setdefault("MEEP_GPU_SUBNORMAL_POLICY", "flush")

from meep_gpu import stepping  # noqa: E402
from meep_gpu.fields import Fields  # noqa: E402
from meep_gpu.grid import Grid  # noqa: E402
from meep_gpu.metal_kernels import bfast_curl as bfast  # noqa: E402
from meep_gpu.metal_kernels import device, preconditions, shaders, subnormal  # noqa: E402
from meep_gpu.pml import PML  # noqa: E402

import metal_gate_kit as kit  # noqa: E402

import hashlib  # noqa: E402
import platform  # noqa: E402
from gate_chunk_records import layer_record  # noqa: E402

log, save, differing = kit.log, kit.save, kit.differing
needle = kit.needle

PERIODIC, METALLIC = shaders.PERIODIC, shaders.METALLIC


# ---------------------------------------------------------------------------
# Gate-local scaffolding
# ---------------------------------------------------------------------------
#
# ``metal_gate_kit`` carries the machinery every Metal gate shares, and this gate
# binds only the primitives that are stable across its revisions: ``log``,
# ``save``, ``words``/``differing``, ``Counter``, ``needle``, ``state_digest`` and
# the exit-75 constant. The four helpers below are spelled HERE rather than taken
# from the kit because they are small, because their signatures are still moving
# while the kit is written for seven families at once, and because a gate whose
# artifact shape can change underneath it is a gate whose green result means
# something different from one run to the next. They are the same helpers by
# behaviour; when the kit settles they become one-line forwards.


def cannot_certify(reason: str) -> int:
    """This HOST cannot arbitrate the claim. Exit 75, never 0 and never 1.

    "the kernel is wrong" and "this box cannot tell" are different facts, and a CI
    that collapses them will eventually read a missing GPU as a passing gate.
    """
    log(f"CANNOT CERTIFY HERE: {reason}")
    return 75


def assert_moved(moved: int, label: str) -> None:
    """A no-op agreeing with a no-op is trivially identical."""
    assert moved > 0, (
        f"VACUOUS: {label} moved no state, so 'identical' compares a no-op with a "
        f"no-op and certifies nothing")


def per_name_differing(got: Dict[str, Any], expected: Dict[str, Any],
                       names: Sequence[str]) -> Dict[str, int]:
    """Per-name differing word counts, empty when identical."""
    counts = {n: differing(got[n], expected[n]) for n in names}
    return {n: c for n, c in counts.items() if c}


def classify(missed: bool, ran: int, launches: int, caught: int) -> str:
    """The three-valued mutation verdict, in the order the failures must be checked.

    NEEDLE-MISSED first: nothing was planted, so the launch count is irrelevant.
    DISARMED next: something was planted but never ran, so the byte comparison was
    against the shipped kernel. Only then is caught/ran a real measurement.
    """
    if missed and ran == 0:
        return "NEEDLE-MISSED"
    if launches == 0:
        return "DISARMED"
    return f"CAUGHT {caught}/{ran}"


def record_mutation(rows: List[Dict[str, Any]], payload: Dict[str, Any], out: str,
                    label: str, verdict_text: str, launches: int, caught: int,
                    ran: int, must_catch: Optional[bool], why: str,
                    started: float) -> Dict[str, Any]:
    """Record one mutation row, print it, save, and enforce its contract."""
    row = {"mutation": label, "verdict": verdict_text, "launches": launches,
           "caught": caught, "ran": ran, "must_catch": must_catch, "why": why}
    rows.append(row)
    log(f"[leg8 mutation] {label:<40} launches={launches} {verdict_text} "
        f"({time.time() - started:.1f}s)")
    payload["legs"]["mutations"] = rows
    save(payload, out)
    assert verdict_text != "DISARMED", (
        f"{label} never launched: a mutation leg that certifies a defect it did "
        f"not run is a hollow pass")
    assert verdict_text != "NEEDLE-MISSED", (
        f"{label}: the transform matched nothing in the shipped source, so the "
        f"defect it names was never planted")
    if must_catch is True:
        assert caught == ran and ran > 0, f"{label} was NOT caught: {row}"
    elif must_catch is False:
        assert caught == 0, f"{label} WAS caught but is a recorded null: {row}"
    return row


def certified_from_rows(legs: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Re-read the recorded ARTIFACT and decide whether it says the bytes agreed.

    A SECOND, INDEPENDENT READ OF THE OUTCOME, and it is deliberately not the same
    evidence as the per-case assertions. The assertions live inside the legs; this
    reads the rows that were WRITTEN, so a leg that recorded a nonempty ``differing``
    and then failed to assert on it — the exact shape of the Triton track's round
    that stamped ``passed: true`` while every compare failed — is caught by the
    summary rather than by the code that produced it. Returns the verdict and the
    rows that carried it.
    """
    problems: List[str] = []
    for leg, rows in sorted(legs.items()):
        if not isinstance(rows, list):
            continue
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                continue
            if row.get("differing"):
                problems.append(f"{leg}[{index}]: differing={row['differing']}")
            verdict = row.get("verdict")
            if isinstance(verdict, str) and verdict in ("DISARMED", "NEEDLE-MISSED"):
                problems.append(f"{leg}[{index}]: {row.get('mutation')} {verdict}")
            if row.get("must_catch") is True and row.get("ran", 0) and \
                    row.get("caught") != row.get("ran"):
                problems.append(f"{leg}[{index}]: {row.get('mutation')} "
                                f"caught {row.get('caught')}/{row.get('ran')}")
    return (not problems), problems


def wiring_from_rows(rows: Any) -> str:
    """What the composer DID, read back off leg 9's rows.

    DERIVED, NEVER TYPED. This field was a hand-written sentence saying the family
    was unwired; the tree wired it and the sentence stayed true-looking in the
    artifact for as long as nobody re-read leg 9. Anything a summary asserts about
    the tree has to come out of a row this run recorded.
    """
    if not isinstance(rows, list) or not rows:
        return "NOT MEASURED: leg 9 did not run in this invocation"
    composed = next((row for row in reversed(rows)
                     if "shipped_plan_step_replaces" in row), None)
    if composed is None:
        return "NOT MEASURED: leg 9 recorded no composer row"
    filled = composed.get("shipped_plan_step_replaces") or []
    selected = composed.get("selected_arm_by_slot") or {}
    if not filled:
        return ("NOT WIRED: the shipped composer filled no sub-step on a BFAST run, "
                "so the array path steps all four")
    return (f"WIRED: the shipped composer fills {filled} on a BFAST run, selecting "
            f"{selected}; the curl slots carry this family's own kernel and the "
            f"constitutive pair carries the CERTIFIED kernel under this family's "
            f"restated predicate (leg 9). Residency across one complete step: "
            f"{'held' if composed.get('residency_verdict_held') else 'REFUSED'}")


def summarize(legs_run: Sequence[str], compared: int, claim: str, scope: str,
              elapsed: float, certified: bool = True,
              problems: Sequence[str] = (), wiring: str = "") -> Dict[str, Any]:
    """The summary block, reading BOTH the compare COUNT and the OUTCOME.

    THE BUG THIS EXISTS TO PREVENT IS A REAL ONE from the Triton track: a gate
    shipped a round where every compare FAILED and it still stamped
    ``passed: true`` and exited 0, because the summary read a count where it meant
    an outcome. ``compared`` is a COUNT — how many rows were recorded, and zero of
    them is VACUOUS, not a pass. ``certified`` is the OUTCOME, re-read from the
    recorded rows by :func:`certified_from_rows` rather than inferred from having
    reached this line: an earlier revision of this function computed
    ``passed = compared > 0`` and argued that the per-case assertions carried the
    outcome, which is the same argument the Triton gate's author could have made.
    ``passed`` is the conjunction, and both halves are recorded.
    """
    passed = compared > 0 and bool(certified)
    return {
        "status": ("passed" if passed else
                   "VACUOUS (nothing was compared)" if compared == 0 else
                   "FAILED (a recorded row did not agree)"),
        "passed": passed,
        "cases_compared": int(compared),
        "certified": bool(certified),
        "disagreeing_rows": list(problems),
        "vacuous": compared == 0,
        "legs_run": list(legs_run),
        "claim": claim,
        "scope": scope,
        "predicted_nulls": [{"defect": n, "why_not_measurable": w,
                             "pinned_by": p, "measured": False}
                            for n, w, p in PREDICTED_NULLS],
        # THE STEP BUDGET, STATED IN ONE PLACE. Scattered across leg rows it is
        # recoverable but not readable, and "how long did you actually run it for"
        # is the first question asked of any identity claim on a recurrence that
        # never decays.
        "step_budget": {
            "transcription_cycles_per_case": TRANSCRIPTION_CYCLES,
            "multi_step_complete_driver_steps_per_case": MULTI_STEP_CYCLES,
            "multi_step_cases": len(CONFIGS) * 2,
            "multi_step_full_inventory_compares":
                MULTI_STEP_CYCLES * len(CONFIGS) * 2,
            "reachability_driven_steps": REACHABILITY_STEPS,
            "reachability_census_every": REACHABILITY_EVERY,
            "source_free_control_steps": REACHABILITY_STEPS,
            "note": ("every number here is a COMPLETE driver step (B, update_H, D, "
                     "update_E) except the census interval"),
        },
        # WHICH SHIPPED SPECIALISATIONS WERE LAUNCHED, not merely fingerprinted. The
        # provenance block hashes all 32 emitted sources; hashing records that a
        # source did not change, which is a different claim from certifying it.
        "specialisation_coverage": {
            "boundary_triples_swept": [name for name, _, _ in CONFIGS],
            "boundary_triples_possible": 8,
            "shipped_sources_fingerprinted": len(bfast.enumerate_bfast_sources()),
            "has_bfast_on_sources_launched_against_the_oracle":
                len(CONFIGS) * len(SUB_STEPS),
            "reduced_dimension": ("dimensions=2 with a DECLARED-invariant z axis is "
                                  "byte-compared in leg 1 (one-partner cases, a k "
                                  "on z among them) and in leg 5 on both of its "
                                  "cases, the k on the invariant axis included, "
                                  "which clause 11a admits"),
        },
        # THE PRECONDITION IS NOT HYPOTHETICAL, AND SAYING SO IS A MEASUREMENT.
        # Leg 7 measures that a DRIVEN run does not reach the band. That is one
        # half of the question; the other half — can anything reach it — was
        # answered by `probe_metal_bfast_engine_route`, which drives the ENGINE
        # route over a long horizon and finds the band on an UNDRIVEN seeded
        # lattice. The numbers are carried here rather than left in a probe
        # artifact, because a claim made "subject to a checked precondition"
        # should say what happens when the check fails, and it should say whose
        # property that is.
        "precondition_reachability": {
            "driven": ("leg 7: 400 driven complete driver steps, zero band words; "
                       "the f_bfast floor bottoms at 1.79e-20 and the tail's "
                       "reconstructed intermediates at 5.40e-20"),
            "undriven_seeded": (
                "MEASURED TO FIRE at complete driver step 49 of an undriven seeded "
                "lattice — metallic_xy: 1 band word in fu_Dz at 5.19e-39; "
                "metallic_all: 3 words in fu_Dx/fu_Dy — and the bytes part in the "
                "same step, the device serving an exact 0.0 where NumPy serves the "
                "subnormal (results/metal_bfast_2026-08-16/engine_route.json)"),
            "attribution": (
                "NOT a BFAST property: the CERTIFIED tranche-1 curl "
                "(launch.plan_pml_curl, no tail, BFAST inactive) breaches at the "
                "SAME step, the SAME array and the SAME word on the same lattice. "
                "It is the split-field absorber recurrence descending into the band "
                "as an undriven run decays toward rest, and it applies to every "
                "family that shares that recurrence"),
            "bound_when_it_fires": (
                "every differing word stays within 1.18e-38 in absolute value "
                "(preconditions.MAX_SUBNORMAL), peaks at 238-318 of 32,400 stored "
                "words, and the two routes re-converge to byte equality by step 66"),
        },
        "stated_weakness": (
            "no PTX-equivalent audit exists on this executor: compile_shader "
            "exposes no disassembly, so the mutation legs and this gate are the "
            "only arbiters"),
        "wiring": wiring or "NOT MEASURED: leg 9 did not run in this invocation",
        "elapsed_s": round(elapsed, 1),
    }


def package_fingerprint_state() -> Dict[str, Any]:
    """Does the tree still match the package's CHECKED-IN fingerprints, and where not?

    RECORDED, NEVER RE-CUT HERE. ``launch.write_fingerprints`` is a deliberate act
    owned by whoever changed the module; a gate that silently re-cut it would launder
    an unrecorded change into a green artifact, which is the exact failure mode the
    fingerprint exists to prevent.

    IT BELONGS IN THIS ARTIFACT because it BOUNDS THIS GATE'S CLAIM. The composed
    route runs through ``launch.plan_step``, so a ``launch.py`` that has drifted from
    the last cut is a fact about what was certified here, not a housekeeping note
    about somebody else's file. The two questions are separated: a drifted KERNEL
    SOURCE would be a correctness event for this family and is called out as such; a
    drifted HOST MODULE is recorded with its name, and the reader can see whether
    this family's own module is among them.
    """
    from meep_gpu.metal_kernels import launch as metal_launch  # noqa: PLC0415

    stored = metal_launch.load_fingerprints()["metal_kernels"]
    computed = metal_launch.compute_fingerprints()["metal_kernels"]
    kernels_drifted = sorted(
        name for name, value in computed["kernel_source_sha256"].items()
        if stored["kernel_source_sha256"].get(name) != value)
    hosts_drifted = sorted(name for name, value in computed["host_sha256"].items()
                           if stored["host_sha256"].get(name) != value)
    return {
        "kernel_sources_match": not kernels_drifted,
        "kernel_sources_drifted": kernels_drifted,
        "host_modules_drifted": hosts_drifted,
        "this_familys_module_drifted": "bfast_curl.py" in hosts_drifted,
        "reading": (
            "the checked-in fingerprints are the package's, not this family's, and "
            "a drift outside this family's module is a fact about a concurrently "
            "edited tree that this gate RECORDS rather than re-cuts — re-cutting "
            "would launder an unrecorded change. It is kept here because the "
            "composed route runs through launch.py, so it bounds what was certified"),
    }


def environment_stamp() -> Dict[str, Any]:
    import torch  # noqa: PLC0415

    return {
        "torch": str(torch.__version__),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "mps_available": bool(torch.backends.mps.is_available()),
        "metal_frontend": device.metal_frontend_version(),
        "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def write_provenance(files: Dict[str, str], out_dir: str,
                     kernel_sources: Dict[str, str]) -> Dict[str, str]:
    """sha256 of every source whose bytes decide what this gate certified.

    The family is NOT WIRED, so ``fingerprints.json`` carries no entry for it and
    the byte gate binds its own provenance record inside its results directory —
    the convention the unwired Triton families follow.
    """
    record: Dict[str, str] = {}
    for label, path in files.items():
        with open(path, "rb") as handle:
            record[label] = hashlib.sha256(handle.read()).hexdigest()
    save({"sources": record, "kernel_sources": kernel_sources},
         os.path.join(out_dir, "provenance.json"))
    return record


# ---------------------------------------------------------------------------
# The case matrix
# ---------------------------------------------------------------------------

#: NON-POWER-OF-TWO FIRST, and the first entry is the marquee corpus row's own
#: Courant: MEEP's ``test_refl_angular`` builds ``(1 - kx)/sqrt(3)`` with
#: ``kx = 1.4*sin(35.7 deg)``, which is ~0.105679 — natively non-representable.
#: MEEP neither derives nor clamps Courant for BFAST (grid.py:719-724), so there is
#: no Courant gate to invent here; the sweep exists because the PML coefficients are
#: the same family of non-representable multiplicands.
COURANTS: Tuple[float, ...] = (0.105679, 0.35, 0.5)

#: THE STEP BUDGET, DECLARED ONCE AND CONSUMED AS THE DEFAULT ARGUMENT. Naming the
#: budget in a comment and passing a literal to the leg is how a recorded budget
#: drifts from the executed one; these constants ARE the legs' defaults and are what
#: ``summary.step_budget`` reports.
TRANSCRIPTION_CYCLES = 3
MULTI_STEP_CYCLES = 4

#: ALL EIGHT BOUNDARY TRIPLES, and the first three keep their positions because
#: several legs bind ``CONFIGS[1]`` (metallic_xy — masks and walls both live) by
#: index.
#:
#: THE GAP THIS CLOSES. ``enumerate_bfast_sources`` ships THIRTY-TWO specialised
#: sources (2 directions x 2 HAS_BFAST arms x 8 boundary triples) and the provenance
#: block fingerprints every one of them, but a three-triple matrix launched only six
#: of the sixteen HAS_BFAST=1 builds against ``stepping.py``. Fingerprinting a source
#: records that it did not change; it does not certify that it is right. The ghost
#: rule and the two ownership masks are emitted PER AXIS, so a defect in, say, the
#: metallic-z emitter reached the oracle on exactly one of the three old rows. Eight
#: triples cost about a second here and certify every shipped specialisation.
CONFIGS: Tuple[Tuple[str, Tuple[float, float, float], Any], ...] = (
    ("periodic_all", (1.2, 1.0, 0.9), "periodic"),
    ("metallic_xy", (1.2, 1.0, 0.9), ("metallic", "metallic", "periodic")),
    ("metallic_all", (1.1, 1.0, 0.9), "metallic"),
    ("metallic_z", (1.2, 1.0, 0.9), ("periodic", "periodic", "metallic")),
    ("metallic_y", (1.2, 1.0, 0.9), ("periodic", "metallic", "periodic")),
    ("metallic_yz", (1.2, 1.0, 0.9), ("periodic", "metallic", "metallic")),
    ("metallic_x", (1.2, 1.0, 0.9), ("metallic", "periodic", "periodic")),
    ("metallic_xz", (1.2, 1.0, 0.9), ("metallic", "periodic", "metallic")),
)

#: THE k VECTORS. The marquee is the one lifted corpus row (a k along x alone);
#: ``full_3d`` exercises all six (k1, k2) scalars simultaneously, which the marquee
#: does NOT — with k along x only, four of the six scalars are zero and a defect in
#: them would be invisible. ``signed`` carries a negative component so the D side's
#: host-f64 negation is exercised in both directions.
BFAST_KS: Tuple[Tuple[str, Tuple[float, float, float]], ...] = (
    ("marquee_x", (0.816958, 0.0, 0.0)),
    ("full_3d", (0.31, 0.17, 0.23)),
    ("signed", (-0.42, 0.19, -0.28)),
)

SUB_STEPS = bfast.SUB_STEPS
STATES = bfast.BFAST_STATE_NAMES

#: THE ONE-PARTNER CASES. Every 3-D row above gives every component both partners,
#: so MEEP's one-partner forms -- the plain branch's ``F = k1*(g1s + g1)`` with no
#: ``- F`` (step_generic.cpp:373-379), and the ``S - F`` of every other branch --
#: never run there. A z-invariant 2-D grid gives the x and y components one partner,
#: and the layer on x and y gives each of them plain, dsig and dsigu cells. The third
#: k carries a z component, on the invariant axis: clause 11a admits it, and every
#: product it scales multiplies a partner MEEP does not read (see leg 5).
ONE_PARTNER_CONFIGS: Tuple[Tuple[str, Tuple[float, float, float], Any], ...] = (
    ("one_partner_2d_metallic_x", (1.6, 1.2, 0.0), ("metallic", "periodic", "periodic")),
    ("one_partner_2d_periodic", (1.6, 1.2, 0.0), "periodic"),
)
ONE_PARTNER_KS: Tuple[Tuple[str, Tuple[float, float, float]], ...] = (
    ("inplane", (0.21, -0.13, 0.0)),
    ("marquee_x", (0.816958, 0.0, 0.0)),
    ("invariant_z", (0.21, -0.13, 0.17)),
)

#: Leg 1's case count, DERIVED from the matrices it sweeps.
REFERENCE_CASES = (len(CONFIGS) * len(BFAST_KS)
                   + len(ONE_PARTNER_CONFIGS) * len(ONE_PARTNER_KS)) * len(COURANTS) * len(
                       SUB_STEPS)

SIDES = {
    "H": {"targets": ("Hx", "Hy", "Hz"), "aux": ("f_w_Hx", "f_w_Hy", "f_w_Hz"),
          "sources": ("Bx", "By", "Bz"), "suffix": "", "step": "update_H"},
    "E": {"targets": ("Ex", "Ey", "Ez"), "aux": ("f_w_Ex", "f_w_Ey", "f_w_Ez"),
          "sources": ("Dx", "Dy", "Dz"), "suffix": "_h", "step": "update_E"},
}

STATE: Tuple[str, ...] = (
    "Bx", "By", "Bz", "Dx", "Dy", "Dz", "Ex", "Ey", "Ez", "Hx", "Hy", "Hz",
    "fu_Bx", "fu_By", "fu_Bz", "fu_Dx", "fu_Dy", "fu_Dz",
    "f_w_Hx", "f_w_Hy", "f_w_Hz", "f_w_Ex", "f_w_Ey", "f_w_Ez",
    "f_bfast_Bx", "f_bfast_By", "f_bfast_Bz",
    "f_bfast_Dx", "f_bfast_Dy", "f_bfast_Dz")


def build(cell, boundaries, courant: float, seed: int,
          bfast_k: Tuple[float, float, float] = (0.31, 0.17, 0.23),
          scale: float = 1.0, signed_zeros: bool = True,
          dimensions: int = 3) -> Tuple[Any, Any, Any]:
    """A real Grid/Fields/PML triple with BFAST active, seeded in the physical band.

    ``signed_zeros`` seeds a +-0 LATTICE into every volume. MEEP keeps float32
    subnormals on arm64 (``set_zero_subnormals`` is a no-op under
    ``#if HAVE_IMMINTRIN_H``), so the signed-zero class is constructible on the HOST
    side here — unlike on a flushing x86 host.

    THE IIR STATES ARE SEEDED TOO, and that is deliberate rather than incidental.
    With ``f_bfast`` left at zero the advance is ``total - 2*0``, so the ``-2*state``
    term — half the recurrence — would never be exercised, and the marginally
    stable pole would be invisible.
    """
    grid = Grid(resolution=10.0, cell_size=cell, boundaries=boundaries,
                dimensions=dimensions, courant=courant, k_point=(0.0, 0.0, 0.0),
                xp=np, bfast_scaled_k=bfast_k)
    fields = Fields(grid=grid, force_complex_fields=False)
    fields.enable_pml_storage()
    count = int(np.prod(grid.shape))
    index = np.arange(count, dtype=np.float32).reshape(grid.shape)
    epsilon = (1.45 + 0.30 * np.sin(index * np.float32(0.037))).astype(np.float32)
    fields.set_isotropic_epsilon_volume(
        epsilon, (np.float32(1.0) / epsilon).astype(np.float32))
    pml = PML(grid=grid,
              thickness=tuple((2, 2) if grid.shape[a] >= 6 else (0, 0)
                              for a in range(3)))
    rng = np.random.default_rng(seed)
    for name in STATE:
        array = getattr(fields, name, None)
        if array is None:
            continue
        host = (rng.standard_normal(grid.shape) * (0.37 * scale)).astype(np.float32)
        if signed_zeros:
            host.reshape(-1)[::17] = np.float32(-0.0)
            host.reshape(-1)[7::23] = np.float32(0.0)
        array[...] = host
    # A SEEDED STATE IS A MID-RUN ONE: H was started from B long ago, so
    # ``stepping.separate_h_from_b`` must not run on it again -- the array path would
    # copy B into H and f_w on the dsigw cells at the first update_H, and no kernel
    # this gate launches does (``launch.h_start_reasons``).
    fields.h_separated = True
    return grid, fields, pml


def snapshot(fields) -> Dict[str, Any]:
    return {name: np.array(getattr(fields, name), copy=True) for name in STATE
            if getattr(fields, name, None) is not None}


def restore(fields, state: Dict[str, Any]) -> None:
    for name, value in state.items():
        getattr(fields, name)[...] = value


def boundary_codes(grid, pml) -> Tuple[int, int, int]:
    kinds = stepping._boundary_kinds(grid, pml)
    return tuple(METALLIC if kind == "metallic" else PERIODIC for kind in kinds)


def curl_coefficients(pml, suffix: str) -> Dict[str, Any]:
    return {f"{stem}_{axis}": getattr(pml, f"{stem}_{axis}{suffix}")
            for axis in "xyz" for stem in ("kms", "sinv")}


def constitutive_coefficients(pml, suffix: str) -> Dict[str, Any]:
    return {f"{stem}_{axis}": getattr(pml, f"{stem}_{axis}{suffix}")
            for axis in "xyz" for stem in ("kps", "kms")}


def plan_ks(grid, sub_step: str) -> Tuple[float, ...]:
    invariant = tuple(grid.is_invariant(axis) for axis in range(3))
    return bfast.bfast_curl_coefficients(grid.bfast_scaled_k, invariant,
                                         magnetic=(sub_step == "step_B"))


# ---------------------------------------------------------------------------
# The pass's intermediates, reconstructed for the subnormal census
# ---------------------------------------------------------------------------

def _shifted(array: Any, axis: int, backward: bool, code: int) -> Any:
    """``stepping._shift_up`` / ``_shift_down`` for one axis, as a gather.

    ``np.roll`` is a pure gather and preserves every bit including signed zeros;
    the metallic wall is then overwritten with an exact ``+0.0``, which is what the
    kernel's ternary delivers.
    """
    out = np.roll(array, 1 if backward else -1, axis=axis)
    if code == METALLIC:
        index = [slice(None)] * 3
        index[axis] = 0 if backward else array.shape[axis] - 1
        out[tuple(index)] = np.float32(0.0)
    return out


def bfast_intermediates(sources: Sequence[Any], states: Sequence[Any],
                        codes: Sequence[int], backward: bool, ks: Sequence[float],
                        invariant: Sequence[bool]) -> Dict[str, Any]:
    """The BFAST pass's un-storable words, per target, as the kernel forms them.

    The two scaled partner sums, the state's two roundings (``F = S - F_prev``, and
    on a one-partner target the plain branch's ``S`` itself) and ``raw = F - F_prev``.
    No operand and no result census would see these, and ``raw`` is a cancellation
    that can sit decades below either operand. Steps nothing: the reference is
    ``stepping`` itself.
    """
    a, b, c = sources
    shifted = {"a_y": _shifted(a, 1, backward, codes[1]),
               "a_z": _shifted(a, 2, backward, codes[2]),
               "b_x": _shifted(b, 0, backward, codes[0]),
               "b_z": _shifted(b, 2, backward, codes[2]),
               "c_x": _shifted(c, 0, backward, codes[0]),
               "c_y": _shifted(c, 1, backward, codes[1])}
    centre = {"a": a, "b": b, "c": c}
    out: Dict[str, Any] = {}
    for target in range(3):
        g1, a1, g2, a2 = shaders.CURL_OPERANDS[target]
        first, second = "abc"[g1], "abc"[g2]
        k1, k2 = np.float32(ks[2 * target]), np.float32(ks[2 * target + 1])
        plus = k1 * (shifted[f"{first}_{'xyz'[a1]}"] + centre[first])
        minus = k2 * (shifted[f"{second}_{'xyz'[a2]}"] + centre[second])
        have_p, have_m = not invariant[a1], not invariant[a2]
        previous = states[target]
        if have_p and have_m:
            total = plus - minus
        elif have_p or have_m:
            total = plus if have_p else minus
        else:
            continue
        state = total - previous
        out[f"plus{target}"] = plus
        out[f"minus{target}"] = minus
        out[f"total{target}"] = total
        out[f"state{target}"] = state
        out[f"raw{target}"] = state - previous
    return out


def _intermediates_of(before: Dict[str, Any], pml, sub_step: str, codes,
                      ks: Sequence[float]) -> Dict[str, Any]:
    spec = SUB_STEPS[sub_step]
    grid = pml.grid
    return bfast_intermediates(
        [before[n] for n in spec["sources"]], [before[n] for n in STATES[sub_step]],
        codes, spec["backward"], ks,
        tuple(bool(grid.is_invariant(axis)) for axis in range(3)))


# ---------------------------------------------------------------------------
# Launching
# ---------------------------------------------------------------------------

def run_bfast_on_device(state: Dict[str, Any], pml, sub_step: str, codes,
                        dtdx: float, ks: Sequence[float],
                        source: Optional[str] = None,
                        contract: str = shaders.CONTRACT_OFF,
                        suffix: Optional[str] = None,
                        has_bfast: bool = True,
                        counter: Optional[List[kit.Counter]] = None,
                        record: Any = None,
                        ) -> Dict[str, Any]:
    """One BFAST curl launch through the SHIPPED plan object, returning host results.

    ``record`` replaces the one-rank chunk record of the case (a record mutation).

    Everything goes through ``plan_bfast_pml_curl_from_arrays`` ->
    ``BfastPmlCurlPlan.run`` so the bytes this gate certifies are the bytes the
    engine route would launch.

    ``counter`` LAUNCH-COUNTS THE SHIPPED KERNEL TOO, and that is not a
    convenience. The HOST mutations (m10, m11) plant their defect in the plan's own
    arguments rather than in the source, so they hand no ``source`` — and while this
    helper only wrapped a counter around a MUTATED source, those two rows could not
    be launch-counted at all. That is worse than a missing statistic: with the
    launch skipped, ``arrays`` still holds the untouched input, which differs from
    the oracle, so a DISARMED host mutation reports itself CAUGHT. Measured on this
    host by substituting a no-op function: ``caught = 2/2`` with nothing launched.
    Wrapping the shipped entry point costs one Python call and makes DISARMED
    reachable for every row. The wrapped function is compiled from the SAME source
    string the plan would build, so it is the same memoized library and the same
    entry point — the counter observes the shipped launch, it does not replace it.
    """
    spec = SUB_STEPS[sub_step]
    names = (tuple(spec["targets"]) + tuple(spec["sources"])
             + tuple("fu_" + n for n in spec["targets"]) + STATES[sub_step])
    arrays = {name: np.array(state[name], copy=True) for name in names}
    flat = {key: np.asarray(value).reshape(-1) for key, value in
            curl_coefficients(pml, spec["suffix"] if suffix is None
                              else suffix).items()}
    residency = device.Residency()
    functions = None
    if source is not None or contract != shaders.CONTRACT_OFF or counter is not None:
        text = source if source is not None else bfast.bfast_curl_source(
            codes, spec["backward"], contract, has_bfast)
        function = device.compile_source(text).bfast_pml_curl_step
        if counter is not None:
            function = kit.Counter(function)
            counter.append(function)
        functions = {shaders.CONTRACT_OFF: function}
    plan = bfast.plan_bfast_pml_curl_from_arrays(
        sub_step, arrays, flat, codes, dtdx, ks, residency,
        functions=functions, has_bfast=has_bfast,
        chunk_bounds=layer_record(pml.grid, pml) if record is None else record)
    plan.run()
    residency.sync_out()
    return arrays


def compared_names(sub_step: str) -> Tuple[str, ...]:
    """Targets, PML auxiliaries AND the three IIR states. The states are the point."""
    spec = SUB_STEPS[sub_step]
    return (tuple(spec["targets"]) + tuple("fu_" + n for n in spec["targets"])
            + STATES[sub_step])


# ---------------------------------------------------------------------------
# LEG 0b — PROOF THE METAL PATH EXECUTED
# ---------------------------------------------------------------------------

def leg_execution(payload: Dict[str, Any], out: str) -> None:
    """That the bytes this gate compares came off the GPU, measured four ways.

    THE FALSE-PASS THIS CLOSES. Every other device leg reads host arrays after the
    launch and compares them to ``stepping.py``. If the launch had silently not
    happened, or had happened on the host, those arrays would still be *some*
    numbers — and a gate that never asks "did the Metal kernel actually run" is
    certifying an unknown executor. The certified PML tranche established the launch
    counter for MUTATIONS, where a disarmed row is the failure mode; this leg asks
    the same question of the SHIPPED path, which is where a silent fallback would
    live. Four independent measurements, none of them an argument:

    1. RESIDENCY. Every mirror the engine-route plan binds is reported with its
       ``torch`` device string and every one must be ``mps``. The plan binds the
       engine's OWN volumes (``plan_bfast_pml_curl``, not the from-arrays harness
       route), so this is the residency the wired family would have.
    2. THE ENTRY POINT'S IDENTITY. The object the launch path calls is recorded by
       type. It must be ``torch._C._mps_MetalKernel`` — a compiled Metal handle from
       ``torch.mps.compile_shader``. No Python callable in this tree can satisfy
       that, so "a host function stood in for the kernel" is excluded by the type
       rather than by inspection of the call site.
    3. THE COUNTER. ``KernelPlan.launches`` is 0 before ``run`` and exactly 1 after,
       and 2 after a second ``run`` — so the counter counts launches rather than
       plans, and a leg that reported a launch it never made would have to lie about
       an integer this leg reads directly.
    4. THE NEGATIVE CONTROL, which is what makes 1-3 mean something. The host arrays
       are compared to the oracle BEFORE the launch: they differ (by exactly the
       words the launch is about to move), so "the plan was built and never run"
       CANNOT pass this gate's comparison. Recorded as ``unlaunched_differing``.

    AND THE FALLBACK ITSELF IS MEASURED, not argued away: the same shipped plan is
    built against a CPU-resident registry and launched. The platform raises
    ``RuntimeError: Passed CPU tensor to MPS op`` — so a host tensor reaching a
    Metal entry point is a LOUD failure on this executor, never a quiet one. The
    exact message is recorded, because that is a fact about this torch build and a
    future one could change it.
    """
    rows: List[Dict[str, Any]] = []
    started = time.time()
    case, cell, boundaries = CONFIGS[1]
    for sub_step in SUB_STEPS:
        grid, fields, pml = build(cell, boundaries, 0.105679, 20260815,
                                  bfast_k=BFAST_KS[1][1])
        grid_o, oracle_fields, pml_o = build(cell, boundaries, 0.105679, 20260815,
                                             bfast_k=BFAST_KS[1][1])
        getattr(stepping, sub_step)(oracle_fields, pml_o)
        after = snapshot(oracle_fields)
        compared = compared_names(sub_step)

        residency = device.Residency()
        plan = bfast.plan_bfast_pml_curl(fields, pml, sub_step, residency,
                                         chunk_bounds=layer_record(fields.grid, pml))
        assert plan is not None, "the engine route refused a covered configuration"

        devices = {mirror: str(residency.tensor(mirror).device)
                   for mirror in residency.names}
        host_resident = sorted(m for m, d in devices.items()
                               if not d.startswith("mps"))
        entry = plan._functions[shaders.CONTRACT_OFF]
        entry_type = f"{type(entry).__module__}.{type(entry).__name__}"

        # (4) THE NEGATIVE CONTROL — before any launch.
        assert plan.launches == 0, "a freshly built plan reports a launch"
        before_launch = snapshot(fields)
        unlaunched = sum(differing(before_launch[n], after[n]) for n in compared)

        plan.run()
        residency.sync_out()
        launched_once = plan.launches
        got = snapshot(fields)
        bad = per_name_differing(got, after, compared)
        # What the launch actually moved, measured against the pre-launch bytes
        # rather than inferred from the negative control's count.
        moved_words = sum(differing(got[n], before_launch[n]) for n in compared)

        # (3) the counter counts LAUNCHES, not plans: a second run must increment.
        plan.run()
        residency.sync_out()
        launched_twice = plan.launches

        # THE FALLBACK, MEASURED. A CPU-resident registry through the same shipped
        # plan builder and the same compiled entry point.
        cpu_error: Optional[str] = None
        cpu_residency = device.Residency(device="cpu")
        cpu_grid, cpu_fields, cpu_pml = build(cell, boundaries, 0.105679, 20260815,
                                              bfast_k=BFAST_KS[1][1])
        cpu_plan = bfast.plan_bfast_pml_curl(
            cpu_fields, cpu_pml, sub_step, cpu_residency,
            chunk_bounds=layer_record(cpu_fields.grid, cpu_pml))
        try:
            cpu_plan.run()
        except Exception as exc:  # noqa: BLE001
            cpu_error = f"{type(exc).__name__}: {exc}"

        row = {"case": case, "sub_step": sub_step,
               "mirror_devices": devices,
               "host_resident_mirrors": host_resident,
               "entry_point_type": entry_type,
               "launches_after_one_run": launched_once,
               "launches_after_two_runs": launched_twice,
               "unlaunched_differing": unlaunched,
               "moved": moved_words,
               "cpu_binding_error": cpu_error,
               "differing": bad}
        rows.append(row)
        log(f"[leg0b execution] {sub_step} mirrors={len(devices)} "
            f"all_mps={not host_resident} entry={entry_type} "
            f"launches={launched_once}->{launched_twice} "
            f"unlaunched_differing={unlaunched} "
            f"cpu_binding={'REFUSED' if cpu_error else 'ACCEPTED (!!)'} "
            f"{'IDENTICAL' if not bad else 'DIFFERS ' + str(bad)} "
            f"({time.time() - started:.1f}s)")
        payload["legs"]["execution"] = rows
        save(payload, out)

        assert devices, "VACUOUS: the plan bound no mirrors at all"
        assert not host_resident, (
            f"the plan bound HOST-resident mirrors {host_resident}: the bytes this "
            f"gate compares did not all come off the GPU")
        assert entry_type == "torch._C._mps_MetalKernel", (
            f"the launch path calls a {entry_type}, not a compiled Metal kernel "
            f"handle; the executor this gate certified is not Metal")
        assert launched_once == 1 and launched_twice == 2, (
            f"the launch counter reports {launched_once}/{launched_twice} for one "
            f"and two runs; it is not counting launches")
        assert unlaunched > 0, (
            f"VACUOUS negative control on {sub_step}: the host arrays already "
            f"matched the oracle BEFORE the launch, so this leg cannot tell a "
            f"kernel that ran from one that did not")
        assert_moved(moved_words, f"execution/{case}/{sub_step}")
        assert cpu_error is not None and "CPU tensor" in cpu_error, (
            f"a CPU-resident mirror bound to the Metal entry point did NOT raise "
            f"(got {cpu_error!r}): a silent host fallback is reachable on this "
            f"executor and every device leg's residency must then be asserted "
            f"per launch rather than per plan")
        assert not bad, row


# ---------------------------------------------------------------------------
# LEG 1 — the Metal kernel vs stepping.py, on real engine objects
# ---------------------------------------------------------------------------

def case_census(before: Dict[str, Any], after: Dict[str, Any], pml,
                sub_step: str, codes, dtdx: float,
                ks: Sequence[float]) -> Dict[str, int]:
    """The subnormal-free precondition, CHECKED ON THIS CASE rather than nearby.

    Leg 7 establishes the cliff and its control on one configuration. That leaves
    every OTHER certified case resting on an ASSUMED precondition — which is exactly
    the assumption fact (h) says not to make. This is the same window, run on the
    case being certified: operands, results, and the tail's reconstructed ``total``
    and ``advance`` intermediates, which no census of stored arrays would see.
    """
    spec = SUB_STEPS[sub_step]
    intermediates = _intermediates_of(before, pml, sub_step, codes, ks)
    window = preconditions.SubnormalWindow(0, 0, per_intermediate_words=1)
    compared = compared_names(sub_step)
    for name in tuple(spec["sources"]) + compared:
        window.observe(f"operand:{name}", before[name])
    for name in compared:
        window.observe(f"result:{name}", after[name])
    for name, value in intermediates.items():
        window.observe_intermediate(name, value)
    reasons = window.vacuity_reasons()
    assert not reasons, f"the per-case census is vacuous: {reasons}"
    return {"subnormal_words": int(window.subnormal_words),
            "censused_words": int(window.observed_words)}


def leg_reference(payload: Dict[str, Any], out: str) -> None:
    rows: List[Dict[str, Any]] = []
    started = time.time()
    total = 0
    matrix = ([(name, cell, boundaries, 3, BFAST_KS)
               for name, cell, boundaries in CONFIGS]
              + [(name, cell, boundaries, 2, ONE_PARTNER_KS)
                 for name, cell, boundaries in ONE_PARTNER_CONFIGS])
    for name, cell, boundaries, dimensions, ks_table in matrix:
        for courant in COURANTS:
            for k_label, bfast_k in ks_table:
                for sub_step in SUB_STEPS:
                    grid, fields, pml = build(cell, boundaries, courant, 20260815,
                                              bfast_k=bfast_k, dimensions=dimensions)
                    before = snapshot(fields)
                    getattr(stepping, sub_step)(fields, pml)
                    after = snapshot(fields)
                    codes = boundary_codes(grid, pml)
                    got = run_bfast_on_device(
                        before, pml, sub_step, codes, float(grid.dt / grid.dx),
                        plan_ks(grid, sub_step))
                    compared = compared_names(sub_step)
                    bad = per_name_differing(got, after, compared)
                    moved = sum(differing(after[n], before[n]) for n in compared)
                    state_moved = sum(differing(after[n], before[n])
                                      for n in STATES[sub_step])
                    census = case_census(before, after, pml, sub_step, codes,
                                         float(grid.dt / grid.dx),
                                         plan_ks(grid, sub_step))
                    total += 1
                    row = {"case": name, "courant": courant, "k": k_label,
                           "sub_step": sub_step,
                           "shape": [int(v) for v in grid.shape], "moved": moved,
                           "bfast_state_moved": state_moved,
                           "compared": list(compared), "differing": bad,
                           **census}
                    rows.append(row)
                    # DERIVED, never a literal: an edit to CONFIGS/COURANTS/BFAST_KS
                    # that left a hardcoded denominator behind would print a
                    # progress line that lies about how much work remains.
                    log(f"[leg1 reference] case {total}/{REFERENCE_CASES} "
                        f"{name:<13} C={courant} "
                        f"k={k_label:<9} {sub_step} moved={moved:<6} "
                        f"f_bfast={state_moved:<5} "
                        f"subnormal={census['subnormal_words']} "
                        f"{'IDENTICAL' if not bad else 'DIFFERS ' + str(bad)}"
                        f" ({time.time() - started:.1f}s)")
                    payload["legs"]["reference"] = rows
                    save(payload, out)
                    assert_moved(moved, f"{name}/{k_label}/{sub_step}")
                    assert state_moved > 0, (
                        f"VACUOUS: {name}/{k_label}/{sub_step} moved no f_bfast "
                        f"state")
                    # THE PRECONDITION IS CHECKED ON THE CASE BEING CERTIFIED.
                    assert census["subnormal_words"] == 0, (
                        f"PRECONDITION BREACH on {name}/C={courant}/{k_label}/"
                        f"{sub_step}: {census['subnormal_words']} of "
                        f"{census['censused_words']} censused words are subnormal, "
                        f"so byte-identity is NOT claimable on this case — the "
                        f"MPS flush has no lever and the reference does not flush")
                    assert not bad, row


# ---------------------------------------------------------------------------
# LEG 2 — the HAS_BFAST=0 identity arm
# ---------------------------------------------------------------------------

def leg_identity(payload: Dict[str, Any], out: str) -> None:
    """A HAS_BFAST=0 build must be byte-identical to the CERTIFIED plain curl.

    THIS IS WHAT LICENSES THE SPECIALISATION. The tail is substituted into a copy
    of ``shaders._CURL_TEMPLATE``'s body; if that copy had drifted — a reordered
    recurrence, a lost paren, a different ghost — this leg is where it shows,
    because the comparison is against the kernel whose sha256 is in
    ``fingerprints.json``.

    It also proves the extra bindings are inert when the tail is off: the three
    state pointers and the six scalars are bound and the results must still match
    a kernel that has neither.
    """
    from meep_gpu.metal_kernels import launch as metal_launch  # noqa: PLC0415

    rows: List[Dict[str, Any]] = []
    started = time.time()
    for name, cell, boundaries in CONFIGS:
        for sub_step, spec in SUB_STEPS.items():
            grid, fields, pml = build(cell, boundaries, 0.35, 20260815)
            before = snapshot(fields)
            codes = boundary_codes(grid, pml)
            dtdx = float(grid.dt / grid.dx)

            # The BFAST kernel with the tail COMPILED OUT, with deliberately
            # NONZERO scalars bound.
            scalars = (0.7, -0.3, 0.5, 0.2, -0.6, 0.4)
            got = run_bfast_on_device(before, pml, sub_step, codes, dtdx, scalars,
                                      has_bfast=False)

            # THE CONTROL THAT MAKES `state_touched == 0` A MEASUREMENT. Without it
            # the leg proves only that the states did not move, which is equally
            # true of a tail that IS present and whose six bound scalars are dead —
            # "if the tail were present these would move it" was an assertion about
            # the seeding, not an observation of it. The SAME seeds and the SAME six
            # scalars through a HAS_BFAST=1 build must move the states; the number is
            # recorded so a later edit that flattened the seeding shows up as a
            # shrinking control rather than as a silently easier leg.
            control = run_bfast_on_device(before, pml, sub_step, codes, dtdx,
                                          scalars, has_bfast=True)
            control_state_moved = sum(differing(control[n], before[n])
                                      for n in STATES[sub_step])

            # The CERTIFIED plain curl, through its own shipped plan.
            names = tuple(spec["targets"]) + tuple(spec["sources"])
            arrays = {n: np.array(before[n], copy=True) for n in names}
            arrays.update({"fu_" + n: np.array(before["fu_" + n], copy=True)
                           for n in spec["targets"]})
            flat = {k: np.asarray(v).reshape(-1)
                    for k, v in curl_coefficients(pml, spec["suffix"]).items()}
            residency = device.Residency()
            plain = metal_launch.plan_from_arrays(sub_step, arrays, flat, codes,
                                                  dtdx, residency,
                                                  chunk_bounds=layer_record(pml.grid, pml))
            plain.run()
            residency.sync_out()

            compared = (tuple(spec["targets"])
                        + tuple("fu_" + n for n in spec["targets"]))
            bad = per_name_differing(got, arrays, compared)
            moved = sum(differing(arrays[n], before[n]) for n in compared)
            # The states must be UNTOUCHED by a HAS_BFAST=0 build.
            state_touched = sum(differing(got[n], before[n]) for n in STATES[sub_step])
            row = {"case": name, "sub_step": sub_step, "moved": moved,
                   "bfast_state_touched": state_touched,
                   "control_state_moved": control_state_moved,
                   "differing": bad}
            rows.append(row)
            log(f"[leg2 identity] {name:<13} {sub_step} moved={moved:<6} "
                f"state_touched={state_touched} control_moved={control_state_moved} "
                f"{'IDENTICAL to certified' if not bad else 'DIFFERS ' + str(bad)} "
                f"({time.time() - started:.1f}s)")
            payload["legs"]["identity"] = rows
            save(payload, out)
            assert_moved(moved, f"identity/{name}/{sub_step}")
            assert not bad, (
                f"the HAS_BFAST=0 build is NOT the certified plain curl: {bad}. "
                f"The BFAST template's copy of the certified body has drifted")
            assert control_state_moved > 0, (
                f"VACUOUS identity arm on {name}/{sub_step}: the same seeds and the "
                f"same six scalars through a HAS_BFAST=1 build moved NO f_bfast "
                f"word, so 'the tail is compiled out' is indistinguishable from "
                f"'the tail is present and inert' on this case")
            assert state_touched == 0, (
                f"a HAS_BFAST=0 build wrote {state_touched} f_bfast words; the "
                f"tail is supposed to be compiled out entirely")


# ---------------------------------------------------------------------------
# LEG 3 — multi-step, where an IIR defect that never decays shows up
# ---------------------------------------------------------------------------

def first_divergence(got: Dict[str, Any], expected: Dict[str, Any],
                     names: Sequence[str]) -> Optional[Dict[str, Any]]:
    """The FIRST array and the FIRST word at which the device left the array path.

    A leg that reports only ``{name: count}`` says a step diverged; the failure
    protocol for this tranche says to report the first case, step AND array, which
    needs the word. Both values are reported as the uint32 word AND as the float it
    decodes to, because the two answer different questions: the word says how far
    apart the bits are (one ulp, a sign bit, a NaN class), the float says whether
    the number is physically anywhere near right.
    """
    for name in names:
        left, right = kit.words(got[name]), kit.words(expected[name])
        if left.shape != right.shape:
            return {"array": name, "reason": "shape mismatch",
                    "device_shape": list(np.shape(got[name])),
                    "oracle_shape": list(np.shape(expected[name]))}
        bad = np.flatnonzero(left != right)
        if bad.size:
            index = int(bad[0])
            flat_got = np.asarray(got[name]).reshape(-1)
            flat_expected = np.asarray(expected[name]).reshape(-1)
            return {"array": name, "flat_index": index,
                    "differing_words": int(bad.size),
                    "device_word": int(left[index]),
                    "oracle_word": int(right[index]),
                    "device_value": float(flat_got[index]),
                    "oracle_value": float(flat_expected[index])}
    return None


def leg_multi_step(payload: Dict[str, Any], out: str,
                   cycles: int = MULTI_STEP_CYCLES) -> None:
    """Full B -> H -> D -> E cycles, COMPARED AFTER EVERY COMPLETE DRIVER STEP.

    THE REASON THIS LEG IS LOAD-BEARING FOR THIS FAMILY IN PARTICULAR: the IIR is
    marginally stable, so an error in the state does not decay — it persists at
    full amplitude forever, alternating sign. A single-launch leg cannot see a
    defect that only enters through the state on the second launch.

    THE COMPARISON UNIT IS ONE COMPLETE DRIVER STEP, NOT THE WHOLE RUN, and that is
    a correction rather than a refinement. An earlier revision advanced both routes
    ``cycles`` times and compared once at the end. That detects a divergence — the
    IIR guarantees it never decays back — but it cannot ATTRIBUTE one: the artifact
    could say only "after four steps these thirty arrays disagree", and the failure
    protocol for this tranche requires the first case, the first STEP and the first
    ARRAY. Comparing the full stored inventory (all 30 volumes: targets, PML
    auxiliaries, both constitutive sides and the six IIR states) after each complete
    step costs six extra compares per case and makes the first divergent step the
    thing that is recorded.

    THE TWO ROUTES RUN ON SEPARATE ``Fields`` OBJECTS, in lockstep, rather than one
    object rewound between them: a rewind compares a route against itself if the
    restore is incomplete, and ``f_bfast`` — allocated lazily, carried across steps,
    and never decaying — is exactly the array a partial restore would miss.
    """
    rows: List[Dict[str, Any]] = []
    started = time.time()
    for name, cell, boundaries in CONFIGS:
        for k_label, bfast_k in BFAST_KS[:2]:
            grid, fields, pml = build(cell, boundaries, 0.105679, 20260815,
                                      bfast_k=bfast_k)
            grid_o, oracle_fields, pml_o = build(cell, boundaries, 0.105679,
                                                 20260815, bfast_k=bfast_k)
            initial = snapshot(fields)
            # The two builds must start from IDENTICAL bytes and identical absorber
            # coefficients, or "the routes agree" is a statement about two different
            # problems. Both are checked rather than trusted to the seed.
            drift = per_name_differing(snapshot(oracle_fields), initial,
                                       tuple(initial))
            assert not drift, (
                f"the two builds did not start from the same bytes: {drift}")
            coefficient_drift = {}
            for suffix in ("", "_h"):
                for label, left in (list(curl_coefficients(pml, suffix).items())
                                    + list(constitutive_coefficients(
                                        pml, suffix).items())):
                    right = (curl_coefficients(pml_o, suffix)
                             | constitutive_coefficients(pml_o, suffix))[label]
                    if differing(left, right):
                        coefficient_drift[f"{label}{suffix}"] = differing(left,
                                                                          right)
            assert not coefficient_drift, (
                f"the two PML objects carry different coefficients: "
                f"{coefficient_drift}")

            codes = boundary_codes(grid, pml)
            dtdx = float(grid.dt / grid.dx)
            ks = {s: plan_ks(grid, s) for s in SUB_STEPS}
            previous = initial

            for step in range(1, cycles + 1):
                # THE ORACLE: one COMPLETE driver step on the NumPy array path.
                stepping.step_B(oracle_fields, pml_o)
                stepping.update_H(oracle_fields, pml_o)
                stepping.step_D(oracle_fields, pml_o)
                stepping.update_E(oracle_fields, pml_o)
                oracle = snapshot(oracle_fields)

                # THE DEVICE ROUTE: the same complete step, with the BFAST curls on
                # Metal and the constitutive halves on the array path (this tranche
                # ports no constitutive kernel). The IIR state is carried across
                # steps through the engine's own host arrays.
                census = {"subnormal_words": 0, "censused_words": 0}
                for sub_step in ("step_B", "step_D"):
                    current = snapshot(fields)
                    got = run_bfast_on_device(current, pml, sub_step, codes, dtdx,
                                              ks[sub_step])
                    # The precondition, checked on THIS step rather than assumed to
                    # hold because it held on step 1 of another configuration.
                    sub_census = case_census(current, got, pml, sub_step, codes,
                                             dtdx, ks[sub_step])
                    for key in census:
                        census[key] += sub_census[key]
                    for volume, value in got.items():
                        getattr(fields, volume)[...] = value
                    if sub_step == "step_B":
                        stepping.update_H(fields, pml)
                    else:
                        stepping.update_E(fields, pml)
                got_state = snapshot(fields)

                compared = tuple(oracle)
                bad = per_name_differing(got_state, oracle, compared)
                moved = sum(differing(oracle[n], previous[n]) for n in compared)
                state_moved = sum(differing(oracle[n], previous[n])
                                  for n in STATES["step_B"] + STATES["step_D"])
                row = {"case": name, "k": k_label, "step": step,
                       "of_steps": cycles, "arrays_compared": len(compared),
                       "moved": moved, "bfast_state_moved": state_moved,
                       "oracle_digest": kit.state_digest(oracle),
                       "device_digest": kit.state_digest(got_state),
                       "differing": bad, **census,
                       "first_divergence": first_divergence(got_state, oracle,
                                                            compared)}
                rows.append(row)
                log(f"[leg3 multi_step] {name:<13} k={k_label:<9} "
                    f"step {step}/{cycles} arrays={len(compared)} "
                    f"moved={moved:<7} f_bfast={state_moved:<5} "
                    f"subnormal={census['subnormal_words']} "
                    f"{'IDENTICAL' if not bad else 'DIFFERS ' + str(bad)} "
                    f"({time.time() - started:.1f}s)")
                payload["legs"]["multi_step"] = rows
                save(payload, out)
                assert_moved(moved, f"multi_step/{name}/{k_label}/step{step}")
                assert state_moved > 0, (
                    f"VACUOUS: {name}/{k_label} step {step} moved no f_bfast "
                    f"state, so the IIR carried nothing into the next step")
                assert census["subnormal_words"] == 0, (
                    f"PRECONDITION BREACH at {name}/{k_label} step {step}: "
                    f"{census['subnormal_words']} of {census['censused_words']} "
                    f"censused words are subnormal, so the identity claim does "
                    f"not hold from this step onward")
                assert not bad, (
                    f"FIRST DIVERGENCE case={name} k={k_label} step={step}: "
                    f"{row['first_divergence']} (per-array counts {bad})")
                previous = oracle


# ---------------------------------------------------------------------------
# LEG 4 — the IIR states are mutated IN PLACE
# ---------------------------------------------------------------------------

def leg_pointer(payload: Dict[str, Any], out: str) -> None:
    """The driver-sync requirement, measured rather than commented.

    ``FdtdDriver`` backs up and RESTORES ``f_bfast_B*`` around the magnetic
    synchronization half-step (driver.py:4126-4135; MEEP energy_and_flux.cpp:113/130).
    If the kernel wrote a COPY instead of the engine's own array, that backup would
    restore a stale value — and because the IIR is marginally stable the error
    would never decay. On CuPy this is literal pointer identity; under the
    residency layer it is the mirror's ``host`` reference plus an in-place sync,
    which is the same requirement one level up and is what this leg asserts.
    """
    rows: List[Dict[str, Any]] = []
    started = time.time()
    for sub_step in SUB_STEPS:
        grid, fields, pml = build(CONFIGS[1][1], CONFIGS[1][2], 0.35, 20260815)
        codes = boundary_codes(grid, pml)
        residency = device.Residency()
        plan = bfast.plan_bfast_pml_curl(fields, pml, sub_step, residency,
                                         chunk_bounds=layer_record(fields.grid, pml))
        assert plan is not None, bfast.bfast_pml_curl_coverage(
            fields, pml, sub_step, residency).reasons

        identities = {name: getattr(fields, name) for name in STATES[sub_step]}
        before = {name: np.array(value, copy=True)
                  for name, value in identities.items()}
        plan.run()
        residency.sync_out()

        # The SAME python objects, and their contents moved.
        same_object = all(getattr(fields, name) is identities[name]
                          for name in STATES[sub_step])
        moved = sum(differing(getattr(fields, name), before[name])
                    for name in STATES[sub_step])
        row = {"sub_step": sub_step, "state_arrays_are_the_engine_objects":
               same_object, "state_words_moved": moved,
               "mirrors": [n for n in residency.names if "f_bfast" in n],
               "launches": plan.launches}
        rows.append(row)
        log(f"[leg4 pointer] {sub_step} same_object={same_object} "
            f"state_moved={moved} launches={plan.launches} "
            f"({time.time() - started:.1f}s)")
        payload["legs"]["pointer"] = rows
        save(payload, out)
        assert plan.launches == 1, row
        assert same_object, (
            f"{sub_step}: the plan did not bind fields.f_bfast_* themselves; the "
            f"driver's flux backup/restore would silently diverge, and because "
            f"the IIR is marginally stable it would never decay")
        assert moved > 0, (
            f"VACUOUS: {sub_step} wrote no f_bfast word in place, so 'in place' "
            f"was never exercised")
        assert len(row["mirrors"]) == 3, row


# ---------------------------------------------------------------------------
# LEG 5 — clause 11a: a k on a DECLARED-invariant axis, admitted and byte-compared
# ---------------------------------------------------------------------------

def leg_refusal(payload: Dict[str, Any], out: str) -> None:
    """Clause 11a: a nonzero k on a DECLARED-invariant axis is admitted and byte-compared.

    An earlier array path zeroed only the coefficient of an absent partner and kept the
    other product, which differed from MEEP's one-partner forms, and the predicate
    refused the class by name. The array path now takes MEEP's one-partner forms
    (``stepping._bfast_term``): an absent partner is not read and the surviving k is
    the one its presence zeroes, so a k on the invariant axis multiplies nothing MEEP
    reads -- the array path's separate-pass change measured that class (case ``bfz``)
    at 0 differing words against MEEP (CHANGELOG). The Triton predicate admits it since
    its fold lane, and this one with it (the two clause lists are pinned equal,
    ``test_metal_bfast``).

    This leg measures, on this host: the premise (on a z-invariant grid the array
    path's step at ``k = (0.4, 0, 0.3)`` stores EXACTLY its step at ``k = (0.4, 0, 0)``);
    that the predicate admits both cases with no invariant-axis reason; and the
    kernel's words against the array path's on the device, both sub-steps, for both
    cases.
    """
    rows: List[Dict[str, Any]] = []
    started = time.time()
    cell, boundaries = (1.2, 1.0, 0.0), ("metallic", "metallic", "periodic")

    def stepped(bfast_k, sub_step):
        grid, fields, pml = build(cell, boundaries, 0.4375, 20260815, bfast_k=bfast_k,
                                  dimensions=2)
        before = snapshot(fields)
        getattr(stepping, sub_step)(fields, pml)
        return grid, fields, pml, before, snapshot(fields)

    for label, bfast_k, expect_admitted in (
            ("k_on_live_axis_only", (0.4, 0.0, 0.0), True),
            ("k_on_invariant_axis", (0.4, 0.0, 0.3), True)):
        grid, fields, pml = build(cell, boundaries, 0.4375, 20260815, bfast_k=bfast_k,
                                  dimensions=2)
        invariant = [bool(grid.is_invariant(a)) for a in range(3)]
        residency = device.Residency()
        verdict = bfast.bfast_pml_curl_coverage(fields, pml, "step_B", residency)
        clause = [r for r in verdict.reasons if "invariant" in r]

        # The premise, re-measured: the invariant-axis k changes no stored word.
        dropped_k = (bfast_k[0], bfast_k[1], 0.0)
        premise: Dict[str, int] = {}
        for sub_step in SUB_STEPS:
            _g, _f, _p, _b, with_k = stepped(bfast_k, sub_step)
            _g, _f, _p, _b, without_k = stepped(dropped_k, sub_step)
            for name in compared_names(sub_step):
                premise[f"{sub_step}:{name}"] = differing(with_k[name], without_k[name])

        reduced: Optional[Dict[str, Any]] = None
        if verdict.covered:
            reduced = {}
            for sub_step in SUB_STEPS:
                sub_grid, _fields, sub_pml, sub_before, sub_after = stepped(bfast_k,
                                                                            sub_step)
                sub_codes = boundary_codes(sub_grid, sub_pml)
                got = run_bfast_on_device(
                    sub_before, sub_pml, sub_step, sub_codes,
                    float(sub_grid.dt / sub_grid.dx), plan_ks(sub_grid, sub_step))
                names = compared_names(sub_step)
                reduced[sub_step] = {
                    "shape": [int(v) for v in sub_grid.shape],
                    "moved": sum(differing(sub_after[n], sub_before[n])
                                 for n in names),
                    "bfast_state_moved": sum(
                        differing(sub_after[n], sub_before[n])
                        for n in STATES[sub_step]),
                    "differing": per_name_differing(got, sub_after, names)}

        row = {"case": label, "bfast_scaled_k": list(bfast_k),
               "invariant_axes": invariant,
               "admitted": bool(verdict.covered),
               "invariant_axis_reason": bool(clause),
               "words_the_invariant_axis_k_moves_on_the_array_path":
                   sum(premise.values()),
               "reduced_dimension_bytes": reduced,
               "differing": ({} if reduced is None else
                             {f"{s}:{n}": c for s, r in reduced.items()
                              for n, c in r["differing"].items()}),
               "reasons": list(verdict.reasons)[:6]}
        rows.append(row)
        log(f"[leg5 refusal] {label:<22} invariant={invariant} "
            f"admitted={verdict.covered} invariant_reason={bool(clause)} "
            f"k_z_moves={sum(premise.values())} "
            f"reduced_dim_bytes="
            f"{'n/a (refused)' if reduced is None else ('IDENTICAL' if not any(r['differing'] for r in reduced.values()) else 'DIFFERS')}"
            f" ({time.time() - started:.1f}s)")
        payload["legs"]["refusal"] = rows
        save(payload, out)

        if reduced is not None:
            for sub_step, record in reduced.items():
                assert_moved(record["moved"], f"refusal/{label}/{sub_step}")
                assert record["bfast_state_moved"] > 0, (
                    f"VACUOUS: {label}/{sub_step} on the reduced-dimension grid "
                    f"moved no f_bfast state")
                assert not record["differing"], (
                    f"{label}/{sub_step}: the predicate ADMITS a dimensions=2 BFAST "
                    f"run and the kernel is not byte-identical there: "
                    f"{record['differing']}")
        assert bool(verdict.covered) == expect_admitted, row
        assert not clause, row
        assert reduced is not None, row
        assert sum(premise.values()) == 0, (
            f"{label}: the invariant-axis k moved {sum(premise.values())} words on "
            f"the array path, which MEEP's one-partner forms never read: "
            f"{ {k: v for k, v in premise.items() if v} }")


# ---------------------------------------------------------------------------
# LEG 6 — zero init and the signed-zero census
# ---------------------------------------------------------------------------

def leg_signed_zero(payload: Dict[str, Any], out: str) -> None:
    """+-0 lattice seeding, with a census FLOOR.

    ZERO INIT IS NOT A FIXED POINT OF THIS SUB-STEP the way it is for the
    constitutive one — with zero state and zero fields the advance is
    ``0 - 2*0 = 0`` and the curl is zero, so nothing moves and the case would be
    vacuous. So the seeding is a ``+-0`` LATTICE with a live amplitude beside it,
    and the moved floor, the IIR-STATE floor and the census floor are all asserted.

    A CENSUS OF ZERO IS VACUOUS, NOT PASSED.

    THE FLOOR IS ASSERTED ON EVERY CONFIGURATION, and the exception that used to
    excuse the periodic one was FALSE. Earlier revisions carried a
    ``SIGNED_ZERO_CONFIGS`` allow-list and stamped ``periodic_all`` with the
    recorded reason "no metallic axis: the ownership mask writes no exact zeros
    through the recurrence, so the -0.0 OUTPUT class is not reachable on this
    configuration" — while the SAME ROW measured 674 negative-zero words on step_B
    and 538 on step_D. The class is reachable on every configuration because the
    ``+-0`` lattice is seeded into the INPUTS and the recurrence's multiplies carry a
    negative zero through; the ownership mask is one producer of exact zeros, not the
    only one. A predicted null that the artifact's own numbers contradict is worse
    than no null, so the allow-list is deleted and the floor applies everywhere.
    """
    rows: List[Dict[str, Any]] = []
    started = time.time()
    for name, cell, boundaries in CONFIGS:
        grid, fields, pml = build(cell, boundaries, 0.35, 20260815)
        for volume in STATE:
            array = getattr(fields, volume, None)
            if array is None:
                continue
            lattice = np.zeros(grid.shape, dtype=np.float32)
            lattice.reshape(-1)[::3] = np.float32(-0.0)
            lattice.reshape(-1)[1::5] = np.float32(0.25)
            array[...] = lattice
        seeded = snapshot(fields)
        codes = boundary_codes(grid, pml)
        dtdx = float(grid.dt / grid.dx)

        for sub_step in SUB_STEPS:
            restore(fields, seeded)
            getattr(stepping, sub_step)(fields, pml)
            after = snapshot(fields)
            got = run_bfast_on_device(seeded, pml, sub_step, codes, dtdx,
                                      plan_ks(grid, sub_step))
            compared = compared_names(sub_step)
            bad = per_name_differing(got, after, compared)
            moved = sum(differing(after[n], seeded[n]) for n in compared)
            state_moved = sum(differing(after[n], seeded[n])
                              for n in STATES[sub_step])
            census = sum(subnormal.signed_zero_census(after[n])["negative_zero"]
                         for n in compared)
            state_census = sum(
                subnormal.signed_zero_census(after[n])["negative_zero"]
                for n in STATES[sub_step])
            state_input_census = sum(
                subnormal.signed_zero_census(seeded[n])["negative_zero"]
                for n in STATES[sub_step])
            row = {"case": name, "sub_step": sub_step,
                   "seeding": "signed_zero_lattice", "moved": moved,
                   "bfast_state_moved": state_moved,
                   "negative_zero_words": census,
                   "negative_zero_words_into_f_bfast": state_input_census,
                   "negative_zero_words_in_f_bfast": state_census,
                   "f_bfast_negative_zero_output_is_a_measured_null": (
                       "UNREACHABLE BY CONSTRUCTION, measured 0 on every "
                       "configuration and both sub-steps while 264-288 words per "
                       "state go IN. The store is `state + advance`, and IEEE "
                       "round-to-nearest yields -0.0 only when BOTH operands are "
                       "-0.0; the ownership mask writes an exact +0.0f, and "
                       "`advance = total - 2*state` is -0.0 only when `total` is "
                       "-0.0 and `2*state` is +0.0 — in which case `state + "
                       "advance` is (+0) + (-0) = +0. So the -0.0 INPUT class is "
                       "fully constructed and the -0.0 OUTPUT class cannot exist "
                       "in f_bfast. This replaces an earlier recorded reason that "
                       "was FALSE: it claimed the -0.0 output class was "
                       "unreachable without a metallic axis, on rows whose own "
                       "aggregate census measured 674 and 538 negative-zero words"),
                   "differing": bad}
            rows.append(row)
            log(f"[leg6 signed_zero] {name:<13} {sub_step} moved={moved:<6} "
                f"f_bfast={state_moved:<5} neg_zero_words={census} "
                f"(into f_bfast {state_input_census}, out {state_census}) "
                f"{'IDENTICAL' if not bad else 'DIFFERS'} "
                f"({time.time() - started:.1f}s)")
            payload["legs"]["signed_zero"] = rows
            save(payload, out)
            assert not bad, row
            assert_moved(moved, f"signed_zero/{name}/{sub_step}")
            assert state_moved > 0, (
                f"VACUOUS: signed_zero/{name}/{sub_step} moved no f_bfast state, "
                f"so the IIR this family exists for saw no signed zero at all")
            assert census > 0, (
                f"VACUOUS signed-zero census on {name}/{sub_step}: a census of 0 "
                f"means the class was never constructed, not that it was handled")
            # THE INPUT FLOOR IS THE ONE THAT IS ASSERTABLE FOR THE TAIL. The -0.0
            # OUTPUT class in f_bfast is unreachable by construction (see the row's
            # own recorded reason), so asserting it would be asserting an
            # impossibility; asserting nothing would leave the tail's signed-zero
            # handling untested. What IS constructible, and what the tail must carry
            # correctly, is -0.0 GOING IN — every `st` read and every `2.0f * st`.
            assert state_input_census > 0, (
                f"VACUOUS signed-zero seeding on {name}/{sub_step}: no -0.0 word "
                f"reached the IIR states, so the tail never read one and its "
                f"`total - 2*state` was not exercised on the class")


# ---------------------------------------------------------------------------
# LEG 7 — the subnormal precondition, the f_bfast reachability census, the control
# ---------------------------------------------------------------------------

SUBNORMAL_SCALES = (("physical", 1.0, True), ("small_normal", 1e-20, True),
                    ("subnormal_band", 1e-38, False))

#: How long the f_bfast reachability census runs. The DEFERRAL'S ATTACHED
#: MEASUREMENT: this family's Triton counterpart was deferred pending a census of
#: ``f_bfast`` on its one lifted corpus row over a REALISTIC budget, with the claim
#: shape — byte-identity, stated tolerance, or refusal — to be decided by the
#: answer rather than assumed. This is that budget.
REACHABILITY_STEPS = 400
REACHABILITY_EVERY = 25


def leg_precondition(payload: Dict[str, Any], out: str) -> None:
    """The census that BOUNDS the claim, its control, and the f_bfast reachability run.

    THREE PARTS:

    1. on the physical band the census over every OPERAND, every RESULT **and every
       INTERMEDIATE** (the three ``total`` and three ``advance`` volumes, which
       exist only inside the kernel) is zero and the bytes are identical;
    2. on the 1e-38 control the census FIRES and the gate REFUSES the case. A
       precondition never demonstrated to fire is decorative;
    3. THE F_BFAST REACHABILITY RUN. The IIR is marginally stable — its
       homogeneous mode is undamped forever — so unlike every other auxiliary in
       this stepper its noise floor does NOT decay toward zero, and the question
       "can a real run drive f_bfast into the subnormal band" has a different
       answer here than for the certified curl. It is MEASURED over
       :data:`REACHABILITY_STEPS` driven steps rather than argued, and the claim
       shape recorded in the artifact follows the measurement.
    """
    rows: List[Dict[str, Any]] = []
    started = time.time()

    for label, scale, expect_clean in SUBNORMAL_SCALES:
        for sub_step, spec in SUB_STEPS.items():
            grid, fields, pml = build(CONFIGS[1][1], CONFIGS[1][2], 0.35, 20260815,
                                      scale=scale, signed_zeros=False)
            before = snapshot(fields)
            getattr(stepping, sub_step)(fields, pml)
            after = snapshot(fields)
            codes = boundary_codes(grid, pml)
            dtdx = float(grid.dt / grid.dx)
            ks = plan_ks(grid, sub_step)
            got = run_bfast_on_device(before, pml, sub_step, codes, dtdx, ks)
            compared = compared_names(sub_step)

            # Reconstruct the kernel's INTERMEDIATES on the host so they can be
            # censused: no operand and no result census would see them.
            intermediates = _intermediates_of(before, pml, sub_step, codes, ks)

            window = preconditions.SubnormalWindow(0, 0)
            for n in tuple(spec["sources"]) + compared:
                window.observe(f"operand:{n}", before[n])
            for n in compared:
                window.observe(f"result:{n}", after[n])
            for n, value in intermediates.items():
                window.observe_intermediate(n, value)

            bad = per_name_differing(got, after, compared)
            total_bad = sum(bad.values())
            fired = window.subnormal_words > 0
            row = {"scale": label, "factor": scale, "sub_step": sub_step,
                   "subnormal_words": window.subnormal_words,
                   "observed_words": window.observed_words,
                   "census_fired": fired,
                   "intermediates_censused": sorted(intermediates),
                   "gate_verdict": "REFUSED (precondition)" if fired
                                   else ("identical" if not total_bad else "DIFFERS"),
                   "differing_words": total_bad,
                   # `certified_from_rows` reads a top-level `differing`, and until
                   # this key existed EVERY leg-7 row was invisible to the summary's
                   # independent second read — the one mechanism whose stated job is
                   # to catch a leg that RECORDS a divergence and fails to assert on
                   # it. The control's divergence is EXPECTED (the census fired and
                   # the case is refused), so it is reported under its own key and
                   # `differing` stays empty there; a divergence on a band that must
                   # agree lands in `differing` and reaches the summary.
                   "differing": {} if not expect_clean else bad,
                   "expected_to_diverge": not expect_clean,
                   "control_differing": bad if not expect_clean else {}}
            rows.append(row)
            log(f"[leg7 precondition] {label:<14} {sub_step} "
                f"subnormal_words={window.subnormal_words} fired={fired} "
                f"differing={total_bad} ({time.time() - started:.1f}s)")
            payload["legs"]["precondition"] = rows
            save(payload, out)

            assert not window.vacuity_reasons(), window.vacuity_reasons()
            if expect_clean:
                assert not fired, (
                    f"{label}: the precondition fired on a band it must not — "
                    f"{window.subnormal_words} words")
                assert total_bad == 0, row
            else:
                assert fired, (
                    f"{label}: the census DID NOT FIRE on the scaled control. A "
                    f"precondition never demonstrated to fire is decorative")
                assert total_bad > 0, (
                    f"{label}: the control was expected to diverge; it did not, "
                    f"so the cliff this claim rests on was not reproduced")

    # --- part 3: is the band REACHABLE in a driven run? ---------------------
    reach = _reachability(payload, out, started)
    payload["legs"]["reachability"] = reach
    save(payload, out)


def _reachability(payload: Dict[str, Any], out: str, started: float) -> Dict[str, Any]:
    """Drive a real pulsed run and census the tail's STATE AND ITS INTERMEDIATES.

    THE MEASUREMENT THE DEFERRAL ATTACHED. A source is injected into Ez every step
    with a Gaussian envelope; after it dies the fields ring down. For the certified
    curl the smallest nonzero |f| was measured to bottom out at ~3e-19 and HOLD,
    because the tail decays to the float32 round-off floor and stops. The IIR here
    is marginally stable, so its own floor is measured separately rather than
    assumed to behave the same way.

    THE INTERMEDIATES ARE CENSUSED TOO, and leaving them out was a real hole rather
    than a tidiness point. ``preconditions`` requires the census to cover RESULTS and
    NAMED INTERMEDIATES because a kernel PRODUCES band values it was not given, and
    this tail's ``advance = total - 2*state`` is a CANCELLATION: ``total`` can
    approach ``2*state`` and leave an advance many decades below either operand,
    which no census of ``f_bfast`` alone would ever see. Neither is readable from the
    device, so they are reconstructed on the host (:func:`bfast_intermediates`) from a
    COPY of the live state, for both sub-steps, at every sample step -- the same
    expression and the same operand order the kernel evaluates. Reconstruction is weaker than reading the device's
    registers and the artifact says so; it is strictly stronger than censusing only
    what was stored.
    """
    grid, fields, pml = build((1.2, 1.0, 0.9), ("metallic", "metallic", "periodic"),
                              0.105679, 20260815, bfast_k=(0.816958, 0.0, 0.0),
                              signed_zeros=False)
    codes = boundary_codes(grid, pml)
    reach_ks = {s: plan_ks(grid, s) for s in SUB_STEPS}
    # Start from rest: a driven run, not a seeded one.
    for name in STATE:
        array = getattr(fields, name, None)
        if array is not None:
            array[...] = np.float32(0.0)
    centre = tuple(n // 2 for n in grid.shape)
    # The intermediate floor is DECLARED, so a window that censused no intermediate
    # reports itself vacuous instead of reporting clean.
    window = preconditions.SubnormalWindow(0, REACHABILITY_STEPS,
                                           per_intermediate_words=1)
    samples: List[Dict[str, Any]] = []
    smallest = float("inf")
    peak_subnormal = 0

    for step in range(REACHABILITY_STEPS + 1):
        # A Gaussian-envelope drive on Ez, injected as MEEP's step_source would.
        t = step * float(grid.dt)
        envelope = float(np.exp(-((t - 1.2) ** 2) / (2 * 0.25 ** 2)))
        fields.Ez[centre] += np.float32(envelope * np.sin(2 * np.pi * 3.0 * t))
        stepping.step_B(fields, pml)
        stepping.update_H(fields, pml)
        stepping.step_D(fields, pml)
        stepping.update_E(fields, pml)
        if step % REACHABILITY_EVERY:
            continue
        found = 0
        floor = float("inf")
        for name in STATES["step_B"] + STATES["step_D"]:
            array = getattr(fields, name)
            found += subnormal.census(array)
            window.observe(f"f_bfast@{step}:{name}", array, step)
            nonzero = np.abs(array[array != 0])
            if nonzero.size:
                floor = min(floor, float(nonzero.min()))
        # The tail's un-storable intermediates, reconstructed on a COPY so the live
        # run is untouched, for BOTH sub-steps.
        inner = 0
        inner_floor = float("inf")
        work = snapshot(fields)
        for sub_step, spec in SUB_STEPS.items():
            produced = _intermediates_of(work, pml, sub_step, codes,
                                         reach_ks[sub_step])
            for label, value in produced.items():
                inner += window.observe_intermediate(
                    f"{sub_step}:{label}@{step}", value, step)
                nonzero_inner = np.abs(value[value != 0])
                if nonzero_inner.size:
                    inner_floor = min(inner_floor, float(nonzero_inner.min()))
        found += inner
        peak_subnormal = max(peak_subnormal, found)
        smallest = min(smallest, floor)
        samples.append({"step": step, "subnormal_words": found,
                        "state_subnormal_words": found - inner,
                        "intermediate_subnormal_words": inner,
                        "smallest_nonzero_abs": None if floor == float("inf")
                        else floor,
                        "smallest_nonzero_intermediate_abs":
                            None if inner_floor == float("inf") else inner_floor})
        log(f"[leg7 reachability] step {step}/{REACHABILITY_STEPS} "
            f"f_bfast subnormal_words={found - inner} intermediate={inner} "
            f"smallest|f|="
            f"{'n/a' if floor == float('inf') else f'{floor:.3e}'} "
            f"smallest|tail|="
            f"{'n/a' if inner_floor == float('inf') else f'{inner_floor:.3e}'} "
            f"({time.time() - started:.1f}s)")
        payload["legs"]["reachability"] = {"samples": samples}
        save(payload, out)

    # THE CLAIM SHAPE FOLLOWS THE MEASUREMENT, and is not chosen in advance.
    if peak_subnormal == 0:
        shape = ("byte-identity under a CHECKED subnormal-free precondition: "
                 "neither f_bfast NOR the tail's reconstructed total/advance "
                 "intermediates entered the band in a driven run over "
                 f"{REACHABILITY_STEPS} steps")
    else:
        shape = ("REFUSAL or stated tolerance required: the tail reached the "
                 f"subnormal band ({peak_subnormal} words at peak, state + "
                 f"intermediates) in a driven run, so byte-identity cannot be "
                 f"claimed unconditionally for this family")
    smallest_inner = min(
        [row["smallest_nonzero_intermediate_abs"] for row in samples
         if row["smallest_nonzero_intermediate_abs"] is not None] or [float("inf")])

    # --- THE SOURCE-FREE CONTROL -------------------------------------------
    #
    # The reachability verdict is a statement about a DRIVEN run, and "driven" is
    # the half of it a census cannot see: every number above is consistent with a
    # run that was never driven at all and merely evolved seeded noise. The control
    # is the same configuration, the same rest initial condition and the same step
    # budget with the SOURCE LINE REMOVED. From rest and with no drive the stepper
    # is linear and homogeneous, so every volume must stay identically zero —
    # including f_bfast, whose marginally stable pole would otherwise keep any
    # injected noise alive forever. A nonzero word here would mean the driven run's
    # census measured something other than the source, and the reachability claim
    # would be about an unknown excitation.
    grid_c, control_fields, control_pml = build(
        (1.2, 1.0, 0.9), ("metallic", "metallic", "periodic"), 0.105679, 20260815,
        bfast_k=(0.816958, 0.0, 0.0), signed_zeros=False)
    for name in STATE:
        array = getattr(control_fields, name, None)
        if array is not None:
            array[...] = np.float32(0.0)
    for _ in range(REACHABILITY_STEPS + 1):
        stepping.step_B(control_fields, control_pml)
        stepping.update_H(control_fields, control_pml)
        stepping.step_D(control_fields, control_pml)
        stepping.update_E(control_fields, control_pml)
    control_nonzero = {
        name: int(np.count_nonzero(getattr(control_fields, name)))
        for name in STATE if getattr(control_fields, name, None) is not None}
    control_total = sum(control_nonzero.values())
    driven_state_nonzero = sum(
        int(np.count_nonzero(getattr(fields, name)))
        for name in STATES["step_B"] + STATES["step_D"])
    source_free = {
        "steps": REACHABILITY_STEPS,
        "nonzero_words": control_total,
        "nonzero_by_array": {n: c for n, c in control_nonzero.items() if c},
        "driven_f_bfast_nonzero_words": driven_state_nonzero,
        "reading": ("the drive is what excited the run: with the source line "
                    "removed and the same budget every volume stays identically "
                    "zero, while the driven run leaves "
                    f"{driven_state_nonzero} nonzero f_bfast words")}
    log(f"[leg7 source-free control] steps={REACHABILITY_STEPS} "
        f"control_nonzero_words={control_total} "
        f"driven_f_bfast_nonzero_words={driven_state_nonzero}")
    assert control_total == 0, (
        f"the SOURCE-FREE control did not stay at rest ({control_total} nonzero "
        f"words, {sorted(n for n, c in control_nonzero.items() if c)}): the driven "
        f"run's census cannot be attributed to the source")
    assert driven_state_nonzero > 0, (
        "VACUOUS drive: the driven run left f_bfast identically zero, so the "
        "source injected nothing this family could carry")

    record = {"samples": samples, "steps": REACHABILITY_STEPS,
              "source_free_control": source_free,
              "census_every": REACHABILITY_EVERY,
              "peak_subnormal_words": peak_subnormal,
              "smallest_nonzero_abs": None if smallest == float("inf") else smallest,
              "smallest_nonzero_intermediate_abs":
                  None if smallest_inner == float("inf") else smallest_inner,
              "intermediates_censused": True,
              "intermediate_reconstruction": (
                  "host-side, by bfast_intermediates (the kernel's own expressions "
                  "and operand order); the kernel's own registers are not readable, "
                  "and the artifact states that rather than implying otherwise"),
              "marginally_stable_pole": True,
              "claim_shape": shape,
              "window": window.report()}
    log(f"[leg7 reachability] VERDICT peak_subnormal_words={peak_subnormal} "
        f"smallest|f_bfast|={smallest:.3e} "
        f"smallest|tail intermediate|={smallest_inner:.3e} -> {shape}")
    assert samples, "VACUOUS: the reachability run censused nothing"
    assert any(row["subnormal_words"] is not None for row in samples)
    assert not window.vacuity_reasons(), window.vacuity_reasons()
    assert smallest < float("inf"), (
        "VACUOUS: f_bfast stayed identically zero for the whole driven run, so "
        "the reachability question was never actually posed")
    assert smallest_inner < float("inf"), (
        "VACUOUS: every reconstructed tail intermediate was identically zero, so "
        "the cancellation class this census exists for was never constructed")
    return record


# ---------------------------------------------------------------------------
# LEG 8 — mutations: armed, launch-counted, three-valued
# ---------------------------------------------------------------------------

def _drop_pass(source: str) -> str:
    """The whole BFAST pass removed (its two marker lines and everything between)."""
    from meep_gpu.metal_kernels import folded_pass  # noqa: PLC0415

    start = source.find(folded_pass.FOLDED_PASS_BEGIN)
    end = source.find(folded_pass.FOLDED_PASS_END)
    if start < 0 or end < 0:
        raise LookupError("the folded pass block")
    end += len(folded_pass.FOLDED_PASS_END) + 1
    return source[:start] + "    // BFAST pass dropped\n" + source[end:]


def _previous_package_form(source: str) -> str:
    """The PREVIOUS package form: the one-rounding advance ``total - 2*state`` folded
    into the curl before the mask and the forms, the state advanced in place."""
    lines = []
    for target in range(3):
        g1, a1, g2, a2 = shaders.CURL_OPERANDS[target]
        first, second = "abc"[g1], "abc"[g2]
        lines += [
            f"    float st{target} = s{target}[ii];",
            f"    float adv{target} = ((k1_{target} * ({first}_{'xyz'[a1]} + {first})) "
            f"- (k2_{target} * ({second}_{'xyz'[a2]} + {second}))) - (2.0f * st{target});",
            f"    s{target}[ii] = st{target} + adv{target};",
            f"    curl{target} = curl{target} - adv{target};"]
    anchor = "    // --- ownership mask"
    return needle(_drop_pass(source), anchor, "\n".join(lines) + "\n\n" + anchor)


MUTATIONS: Dict[str, Dict[str, Any]] = {
    "m1_k_assignment_swapped": {
        "must_catch": True,
        "why": "k1 and k2 exchanged on target 0 -- the CROSS-assignment (MEEP "
               "vec.hpp:445) stepping itself calls the single easiest mistake in the "
               "whole pass, and which is SILENT on a k along one axis",
        "apply": lambda s: needle(
            needle(s, "float sp0 = k1_0 * (c_y + c);", "float sp0 = k2_0 * (c_y + c);"),
            "float sm0 = k2_0 * (b_z + b);", "float sm0 = k1_0 * (b_z + b);"),
    },
    "m2_sum_becomes_difference": {
        "must_catch": True,
        "why": "the BFAST SUM written as the curl's DIFFERENCE",
        "apply": lambda s: needle(
            needle(s, "float sp1 = k1_1 * (a_z + a);", "float sp1 = k1_1 * (a_z - a);"),
            "float sm1 = k2_1 * (c_x + c);", "float sm1 = k2_1 * (c_x - c);"),
    },
    "m3_one_rounding_increment": {
        "must_catch": True,
        "why": "the increment formed as the previous package's ONE rounding of "
               "S - 2*F_prev instead of MEEP's raw = (S - F_prev) - F_prev (the "
               "array path's separate-pass change measured the second rounding "
               "necessary: 458,721 of 1,113,272 words)",
        "apply": lambda s: needle(
            s, "float fi0 = (has_y || has_z) ? (-(F0 - fp0)) : 0.0f;",
            "float fi0 = (has_y || has_z) ? (-((sp0 - sm0) - (2.0f * fp0))) : 0.0f;"),
    },
    "m4_state_stored_on_an_unowned_cell": {
        "must_catch": True,
        "why": "the state stored on every cell; MEEP writes F only inside its "
               "owned-cell loop, and the array path stores it on owned cells only",
        "apply": lambda s: needle(s, "if (own0 && (has_y || has_z)) { s0[ii] = F0; }",
                                  "s0[ii] = F0;"),
        "codes": (METALLIC, METALLIC, PERIODIC),
        "sub_steps": ("step_B",),
    },
    "m4b_increment_not_masked": {
        "must_catch": True,
        "why": "the increment kept on the cell the curl's mask drops",
        "apply": lambda s: needle(s, "    fi0 = at_x ? 0.0f : fi0;\n", ""),
        "codes": (METALLIC, METALLIC, PERIODIC),
        "sub_steps": ("step_B",),
    },
    "m5_increment_sign_flipped": {
        "must_catch": True,
        "why": "the increment not negated: the pass would subtract raw, inverting "
               "the sign of the whole feature",
        "apply": lambda s: needle(
            s, "float fi0 = (has_y || has_z) ? (-(F0 - fp0)) : 0.0f;",
            "float fi0 = (has_y || has_z) ? (F0 - fp0) : 0.0f;"),
    },
    "m6_dtdx_applied_to_the_pass": {
        "must_catch": True,
        "why": "the pass scaled by dtdx. MEEP passes dtdx to step_bfast and the body "
               "NEVER reads it -- the Tustin derivative already carries the dt",
        "apply": lambda s: needle(s, "float sp1 = k1_1 * (a_z + a);",
                                  "float sp1 = dtdx * (k1_1 * (a_z + a));"),
    },
    "m7_state_not_stored": {
        "must_catch": True,
        "why": "the state store dropped: the IIR loses its memory, and marginally "
               "stable means this never decays back",
        "apply": lambda s: needle(s, "if (own0 && (has_y || has_z)) { s0[ii] = F0; }",
                                  "// state store dropped"),
    },
    "m8_contract_on": {
        "must_catch": True,
        "why": "removes the one compile option. The pass's `k1*(sum) - k2*(sum)` and "
               "the ladder's `v - (si * df)` are multiply-subtracts",
        "apply": lambda s: needle(s, shaders.contraction_pragma("off"),
                                  shaders.contraction_pragma("fast")),
    },
    "m9_pass_reads_unshifted_pair": {
        "must_catch": True,
        "why": "the shifted operand replaced by its centre -- `(b + b)` instead of "
               "`(b_x + b)`; the pass reads the curl's own shifted/centre pairs",
        "apply": lambda s: needle(s, "float sp2 = k1_2 * (b_x + b);",
                                  "float sp2 = k1_2 * (b + b);"),
    },
    "m12_increment_mask_retargeted": {
        "must_catch": True,
        "why": "the increment's ownership mask applied to the WRONG target",
        "apply": lambda s: needle(s, "    fi0 = at_x ? 0.0f : fi0;",
                                  "    fi1 = at_x ? 0.0f : fi1;"),
        "codes": (METALLIC, METALLIC, PERIODIC),
        "sub_steps": ("step_B",),
    },
    "m13_state_pointers_swapped": {
        "must_catch": True,
        "why": "two of the three IIR state pointers exchanged; the IIR is marginally "
               "stable, so a permuted history never decays back",
        "apply": lambda s: needle(
            needle(s, "float fp0 = s0[ii];", "float fp0 = s1[ii];"),
            "float fp1 = s1[ii];", "float fp1 = s0[ii];"),
    },
    "m14_state_pointer_duplicated": {
        "must_catch": True,
        "why": "one state pointer read twice -- target 2 reading target 0's history",
        "apply": lambda s: needle(s, "float fp2 = s2[ii];", "float fp2 = s0[ii];"),
    },
    "m15_previous_package_form": {
        "must_catch": True,
        "why": "the PREVIOUS package form: the one-rounding advance folded into the "
               "curl before the mask and the per-chunk forms, instead of MEEP's pass "
               "after them with its own ladder (with the passes folded into the "
               "curl, the array path differed from MEEP on case bf in 120,035 of "
               "278,318 words; CHANGELOG)",
        "apply": _previous_package_form,
    },
    "m16_ladder_ignores_dsigu": {
        "must_catch": True,
        "why": "the dsigu chunks take the outside stage: fu not advanced and f not "
               "scaled by sinv_u",
        "apply": lambda s: needle(
            needle(s, "    n2 = du2 ? (n2 - fd2) : n2;", "    n2 = n2;"),
            "    v2 = du2 ? (v2 - (si_y * fd2)) : (v2 - fd2);", "    v2 = (v2 - fd2);"),
    },
    "m17_ladder_skips_the_dsig_coefficient": {
        "must_catch": True,
        "why": "df = raw instead of raw * siginv: wrong in a dsig chunk",
        "apply": lambda s: needle(s, "    float fd0 = (si_y * fi0);",
                                  "    float fd0 = fi0;"),
    },
}

#: The rows MEEP's one-partner forms decide, run on the one-partner 2-D case: on every
#: 3-D configuration each component has both partners and these lines never select.
ONE_PARTNER_MUTATIONS: Dict[str, Dict[str, Any]] = {
    "m18_plain_branch_keeps_the_state": {
        "must_catch": True,
        "why": "MEEP's plain one-partner line F = k1*(g1s + g1) (no `- F`, "
               "step_generic.cpp:373-379) spelled as the other branches' S - F",
        "apply": lambda s: needle(s, "(pl0 ? sp0 : (sp0 - fp0))", "(sp0 - fp0)"),
    },
    "m19_plain_branch_everywhere": {
        "must_catch": True,
        "why": "the plain branch taken on every cell, the layer chunks included",
        "apply": lambda s: needle(s, "(pl0 ? sp0 : (sp0 - fp0))", "sp0"),
    },
    "m20_absent_partner_read": {
        "must_catch": True,
        "why": "the one-partner target given both partners' terms: MEEP hands "
               "step_bfast NULL for the absent one and never reads it",
        "apply": lambda s: needle(s, "(pl0 ? sp0 : (sp0 - fp0))",
                                  "(pl0 ? (sp0 - sm0) : ((sp0 - sm0) - fp0))"),
    },
}


def leg_mutations(payload: Dict[str, Any], out: str) -> None:
    """Every mutation ARMED, LAUNCH-COUNTED and classified three ways."""
    rows: List[Dict[str, Any]] = []
    started = time.time()
    name, cell, boundaries = CONFIGS[1]  # metallic_xy: masks and walls both live
    grid, fields, pml = build(cell, boundaries, 0.35, 20260815,
                              bfast_k=BFAST_KS[1][1])
    codes = boundary_codes(grid, pml)
    dtdx = float(grid.dt / grid.dx)
    base = snapshot(fields)

    oracle: Dict[str, Dict[str, Any]] = {}
    for sub_step in SUB_STEPS:
        restore(fields, base)
        getattr(stepping, sub_step)(fields, pml)
        oracle[sub_step] = snapshot(fields)

    for label, entry in MUTATIONS.items():
        mutation_codes = entry.get("codes", codes)
        counters: List[kit.Counter] = []
        caught = ran = 0
        missed = False
        for sub_step in entry.get("sub_steps", tuple(SUB_STEPS)):
            spec = SUB_STEPS[sub_step]
            shipped = bfast.bfast_curl_source(mutation_codes, spec["backward"])
            try:
                mutated = entry["apply"](shipped)
            except LookupError:
                missed = True
                continue
            got = run_bfast_on_device(base, pml, sub_step, mutation_codes, dtdx,
                                      plan_ks(grid, sub_step), source=mutated,
                                      counter=counters)
            compared = compared_names(sub_step)
            ran += 1
            caught += int(any(differing(got[n], oracle[sub_step][n])
                              for n in compared))
        launches = sum(counter.launches for counter in counters)
        record_mutation(rows, payload, out, label,
                        classify(missed, ran, launches, caught), launches, caught,
                        ran, entry["must_catch"], entry["why"], started)

    # --- the one-partner rows, on a z-invariant 2-D case ------------------------
    one_name, one_cell, one_boundaries = ONE_PARTNER_CONFIGS[0]
    one_grid, one_fields, one_pml = build(one_cell, one_boundaries, 0.35, 20260815,
                                          bfast_k=ONE_PARTNER_KS[0][1], dimensions=2)
    one_codes = boundary_codes(one_grid, one_pml)
    one_dtdx = float(one_grid.dt / one_grid.dx)
    one_base = snapshot(one_fields)
    one_oracle: Dict[str, Dict[str, Any]] = {}
    for sub_step in SUB_STEPS:
        restore(one_fields, one_base)
        getattr(stepping, sub_step)(one_fields, one_pml)
        one_oracle[sub_step] = snapshot(one_fields)
    for label, entry in ONE_PARTNER_MUTATIONS.items():
        counters = []
        caught = ran = 0
        missed = False
        for sub_step in SUB_STEPS:
            spec = SUB_STEPS[sub_step]
            try:
                mutated = entry["apply"](bfast.bfast_curl_source(one_codes,
                                                                 spec["backward"]))
            except LookupError:
                missed = True
                continue
            got = run_bfast_on_device(one_base, one_pml, sub_step, one_codes, one_dtdx,
                                      plan_ks(one_grid, sub_step), source=mutated,
                                      counter=counters)
            ran += 1
            caught += int(any(differing(got[n], one_oracle[sub_step][n])
                              for n in compared_names(sub_step)))
        launches = sum(counter.launches for counter in counters)
        record_mutation(rows, payload, out, label,
                        classify(missed, ran, launches, caught), launches, caught,
                        ran, entry["must_catch"], entry["why"], started)

    # --- the certified curl's per-chunk form battery, on the derived source ----
    import metal_fused_battery as battery  # noqa: PLC0415

    sources_by_side = {sub_step: battery.source_rows(
        bfast.bfast_curl_source(codes, spec["backward"]), curl=True,
        constitutive=False) for sub_step, spec in SUB_STEPS.items()}
    for label in battery.CURL_ROWS:
        counters = []
        caught = ran = 0
        entry = sources_by_side["step_B"][label]
        for sub_step in SUB_STEPS:
            got = run_bfast_on_device(base, pml, sub_step, codes, dtdx,
                                      plan_ks(grid, sub_step),
                                      source=sources_by_side[sub_step][label]["source"],
                                      counter=counters)
            ran += 1
            caught += int(any(differing(got[n], oracle[sub_step][n])
                              for n in compared_names(sub_step)))
        launches = sum(counter.launches for counter in counters)
        record_mutation(rows, payload, out, label,
                        classify(False, ran, launches, caught), launches, caught,
                        ran, entry["must_catch"], entry["why"], started)

    # --- the record mutations: the kernel right, the record wrong --------------
    for label, (mutate, why) in battery.RECORD_ROWS.items():
        counters = []
        caught = ran = 0
        for sub_step in SUB_STEPS:
            got = run_bfast_on_device(base, pml, sub_step, codes, dtdx,
                                      plan_ks(grid, sub_step), counter=counters,
                                      record=mutate(layer_record(grid, pml)))
            ran += 1
            caught += int(any(differing(got[n], oracle[sub_step][n])
                              for n in compared_names(sub_step)))
        launches = sum(counter.launches for counter in counters)
        record_mutation(rows, payload, out, label,
                        classify(False, ran, launches, caught), launches, caught,
                        ran, True, why, started)

    # --- HOST mutations: the plan's own choices, not the kernel's source -----
    #
    # THESE TWO ARE LAUNCH-COUNTED LIKE THE SOURCE MUTATIONS, and the counting is
    # load-bearing rather than decorative. A host mutation plants its defect in the
    # plan's ARGUMENTS, so the shipped kernel runs unmodified — and if it did not
    # run at all, the returned arrays would still be the untouched input, which
    # differs from the oracle, so the row would report itself CAUGHT having measured
    # nothing. Measured on this host with a no-op substituted for the entry point:
    # `caught = 2/2`, launches zero. The counter is what turns that into DISARMED.
    #
    # m10: the six scalars computed for the WRONG SIDE. The D side negates both k
    # in host f64 (stepping.py:913-914); binding the B-side scalars to step_D is
    # exactly the sign error that turns -d/dt(k x H) back into +d/dt(k x E).
    caught = 0
    counters: List[kit.Counter] = []
    for sub_step in SUB_STEPS:
        wrong = plan_ks(grid, "step_D" if sub_step == "step_B" else "step_B")
        got = run_bfast_on_device(base, pml, sub_step, codes, dtdx, wrong,
                                  counter=counters)
        caught += int(any(differing(got[n], oracle[sub_step][n])
                          for n in compared_names(sub_step)))
    launches = sum(counter.launches for counter in counters)
    record_mutation(rows, payload, out, "m10_bfast_scalars_from_the_other_side",
                    classify(False, 2, launches, caught), launches, caught, 2, True,
                    "the D side's host-f64 negation of both k dropped, which "
                    "inverts the sign of the whole feature on that sub-step",
                    started)

    # m11: the Yee sub-lattice suffix swapped — a half-cell error in the absorber
    # profile, converged, smooth and wrong. Inherited from the certified gate
    # because this kernel makes the same host choice.
    caught = 0
    counters = []
    for sub_step, spec in SUB_STEPS.items():
        wrong_suffix = "" if spec["suffix"] == "_h" else "_h"
        got = run_bfast_on_device(base, pml, sub_step, codes, dtdx,
                                  plan_ks(grid, sub_step), suffix=wrong_suffix,
                                  counter=counters)
        caught += int(any(differing(got[n], oracle[sub_step][n])
                          for n in compared_names(sub_step)))
    launches = sum(counter.launches for counter in counters)
    record_mutation(rows, payload, out, "m11_half_integer_suffix_swap",
                    classify(False, 2, launches, caught), launches, caught, 2, True,
                    "the curl's PML coefficients taken from the other Yee "
                    "sub-lattice", started)


# ---------------------------------------------------------------------------
# LEG 9 — the engine route and composition disjointness
# ---------------------------------------------------------------------------

def leg_composition(payload: Dict[str, Any], out: str) -> None:
    """Plans from the engine's own objects, and what the SHIPPED COMPOSER now does.

    THIS LEG WAS INVERTED BY THE TREE, and the inversion is the tranche-2 fact worth
    stating plainly. It used to assert that ``plan_step`` leaves both curl slots
    UNSELECTED on a BFAST run — the safety argument for gating a family before
    dispatch was decided. ``bfast_curl.register_arms`` now registers four arms (two
    curls and the certified constitutive pair), so the composer fills all four slots,
    and the old assertion fails on a tree where nothing is wrong. Deleting it would
    have thrown away the disjointness check; so the rule is split, and BOTH halves
    are asserted here:

      * NO FOREIGN PRODUCT MAY ADMIT on ``step_B``/``step_D``. Unweakened, and now
        swept over the ARM TABLE for those slots rather than over one hand-named
        predicate: every registered arm whose family is not this one must refuse,
        and the shipped tranche-1 curl must refuse BY NAME (clause 11).
      * THE COMPOSER MUST SELECT THIS FAMILY on both curl slots, and the object it
        selected must be this family's plan class. An arm that is registered but
        never wins routes nothing, and the disjointness half alone would call that
        silence a pass.

    THE CONSTITUTIVE PAIR IS THE SECOND CLAIM AND IT IS A DIFFERENT ONE. This family
    also fills ``update_H``/``update_E``, with NO NEW ARITHMETIC: the plan is built by
    the CERTIFIED builder and only the ADMISSION is this module's. "Same kernel" is
    checked as an identity rather than asserted as a sentence — the plan's compiled
    entry point must BE the object ``launch.compile_constitutive(side, contract)``
    returns, which is memoized on the certified source string, and the source strings
    themselves are pinned unchanged by ``gate_metal_spine_byte_neutrality``. What the
    BYTES do on that composed route is not this leg's question; it is measured per
    complete driver step by ``probe_metal_bfast_engine_route``'s composed leg.

    THE RESIDENCY VERDICT IS RECORDED because the wiring changed it: with all four
    sub-steps on the device there is no array-path wall inside a step, so the mirror
    set may be held across one. That is a property of the COMPOSITION, not of any
    plan, and it is read here rather than inferred.
    """
    from meep_gpu.metal_kernels import arms as metal_arms  # noqa: PLC0415
    from meep_gpu.metal_kernels import coverage as metal_coverage  # noqa: PLC0415
    from meep_gpu.metal_kernels import launch as metal_launch  # noqa: PLC0415

    rows: List[Dict[str, Any]] = []
    started = time.time()
    grid, fields, pml = build(CONFIGS[1][1], CONFIGS[1][2], 0.35, 20260815)
    residency = device.Residency()

    for sub_step in SUB_STEPS:
        plan = bfast.plan_bfast_pml_curl(fields, pml, sub_step, residency,
                                         chunk_bounds=layer_record(fields.grid, pml))
        shipped = metal_coverage.pml_curl_coverage(fields, pml, sub_step, residency)
        bfast_clause = [r for r in shipped.reasons if "BFAST" in r]
        row = {"sub_step": sub_step, "bfast_plan_built": plan is not None,
               "shipped_curl_admits": bool(shipped.covered),
               "shipped_refuses_by_name": bool(bfast_clause),
               "arguments": len(plan._args) if plan is not None else None}
        rows.append(row)
        log(f"[leg9 composition] {sub_step} bfast_plan={plan is not None} "
            f"shipped_admits={shipped.covered} refused_by_name="
            f"{bool(bfast_clause)} ({time.time() - started:.1f}s)")
        payload["legs"]["composition"] = rows
        save(payload, out)
        assert plan is not None, "the engine route refused a configuration it covers"
        assert not shipped.covered, (
            f"the SHIPPED tranche-1 Metal curl admits a BFAST run on {sub_step}: "
            f"two products would admit the same slot and the composition would be "
            f"an ambiguity")
        assert bfast_clause, (
            f"the shipped curl refuses {sub_step} but NOT by naming BFAST; the "
            f"disjointness would then be incidental rather than declared")

    # ---- the FOREIGN half: every OTHER arm registered on a claimed slot refuses.
    metal_arms.ensure_registered()
    context = metal_arms.StepContext(fields, pml, residency,
                                     contract_variants=(shaders.CONTRACT_OFF,),
                                     sources=(), synced=())
    # THE SWEEP IS PARTITIONED BY ARM KIND, NOT BY ``wired``, and that distinction is
    # the whole correctness of this leg. A PEER replaces exactly the slot it sits on, so
    # two peers admitting one slot is the over-covering dispatch the composer fails
    # closed on, and that stays an error. A WELD (``spec.is_weld``) replaces a whole
    # seam -- its curl, the in-seam passes and the constitutive that closes it -- and
    # its predicate is a CONJUNCTION whose first conjunct is verbatim the wired curl
    # arm's, so it ALWAYS co-admits with the arm it welds. "At most one admits" over
    # welds asserts something no weld can ever satisfy.
    #
    # FILTERING ON ``not spec.wired`` WOULD BE WRONG WHILE LOOKING IDENTICAL: every
    # unwired arm on this tree is a weld (measured 16 of 16), so the two filters agree
    # today -- and the ``wired`` one would silently delete the case this sweep was built
    # for, a deferred SINGLE-SLOT peer registered ahead of its wiring.
    #
    # The vacuity floor counts PEERS, not the raw table: a guard measuring a different
    # population than the assertion it guards is not a guard.
    for sub_step in SUB_STEPS:
        table = metal_arms.registered(sub_step)
        foreign, welds_admitting, own_admitting = [], [], []
        for spec in table:
            try:
                admits = bool(spec.coverage(context, spec.slot).covered)
            except Exception:  # noqa: BLE001 - a raising predicate is a refusal
                admits = False
            if not admits:
                continue
            name = f"{spec.family}/{spec.label}"
            if spec.family == bfast.FAMILY:
                own_admitting.append(name)
            elif spec.is_weld:
                welds_admitting.append(name)
            else:
                foreign.append(name)
        peers_consulted = sum(1 for s in table
                              if s.family != bfast.FAMILY and not s.is_weld)
        row = {"sub_step": sub_step, "arms_registered_on_slot": len(table),
               "foreign_arms_consulted": peers_consulted,
               "foreign_arms_admitting": foreign,
               "welds_admitting": welds_admitting,
               "own_admitting": own_admitting,
               "welds_registered_on_slot": sum(1 for s in table if s.is_weld)}
        rows.append(row)
        log(f"[leg9 composition] {sub_step} arms={len(table)} "
            f"peers_consulted={peers_consulted} foreign_admitting={foreign} "
            f"welds_admitting={welds_admitting} own={own_admitting}")
        payload["legs"]["composition"] = rows
        save(payload, out)
        assert row["foreign_arms_consulted"] > 0, (
            f"VACUOUS disjointness on {sub_step}: no foreign PEER was registered "
            f"there, so 'nobody else admits' measures nothing")
        assert not foreign, (
            f"a FOREIGN PEER admits a BFAST run on {sub_step}: {foreign}")
        # THE WELD HALF IS A STRICTER CHECK, NOT A SKIP. A weld may co-admit -- that is
        # its construction -- but only ALONGSIDE a single-slot arm that admits the same
        # configuration. A weld admitting where NOTHING else does would mean it covers a
        # configuration no certified sub-step arm covers: that is the over-coverage this
        # leg exists to catch, wearing a weld's clothes. ``own_admitting`` is MEASURED
        # in the loop above rather than assumed, so this cannot pass by default.
        if welds_admitting:
            assert own_admitting or foreign, (
                f"on {sub_step} a WELD admits ({welds_admitting}) while NO single-slot "
                f"arm does -- neither this family ({own_admitting}) nor any peer "
                f"({foreign}); the weld covers a configuration no certified sub-step "
                f"arm covers")

    # ---- the OWN half: the composer selects this family, with this family's plan.
    composed = metal_launch.plan_step(fields, pml, residency=device.Residency(),
                                      sources=())
    selected = dict(composed.selected)
    filled = list(composed.replaces)
    classes = {slot: type(plan).__name__ for slot, plan in composed.plans.items()}
    curl_plan_classes = {slot: isinstance(composed.plans.get(slot),
                                          bfast.BfastPmlCurlPlan)
                         for slot in SUB_STEPS}
    # WHY THE ENTRY POINT IS CHECKED THROUGH THE LIBRARY CACHE AND NOT BY `is` ON
    # THE FUNCTION. Measured on this executor: `_mps_ShaderLibrary.<name>` mints a
    # FRESH `_mps_MetalKernel` on every attribute access, so
    # `compile_constitutive(side, mode) is compile_constitutive(side, mode)` is
    # False for the same compiled binary — an identity test there reports a defect
    # that is not one. What IS memoized, and memoized on the EXACT SOURCE STRING,
    # is `device.compile_source`'s library cache. So the checkable claim is the one
    # that matters anyway: the only constitutive kernel SOURCE this process ever
    # compiled is the certified text, whose bytes `gate_metal_spine_byte_neutrality`
    # separately pins unchanged.
    certified_sources = {side: shaders.constitutive_source(side,
                                                           shaders.CONTRACT_OFF)
                         for side in ("H", "E")}
    compiled_constitutive = {source for source in device._LIBRARY_CACHE
                             if "constitutive_step" in source}
    uncertified = compiled_constitutive - set(certified_sources.values())
    constitutive_ok = {}
    certified_entry_point = {}
    sub_lattice = {}
    for slot, side in (("update_H", "H"), ("update_E", "E")):
        plan = composed.plans.get(slot)
        constitutive_ok[slot] = isinstance(plan, metal_launch.ConstitutivePlan)
        certified_entry_point[slot] = bool(
            certified_sources[side] in device._LIBRARY_CACHE and not uncertified)
        # The half-cell trap, read off the mirrors the plan actually bound. The
        # composed plans SHARE one residency — which is the point of the composition
        # — so its name list carries every side's coefficients at once and cannot
        # attribute them. Rebuilding this one plan on a RESIDENCY OF ITS OWN is what
        # makes the question answerable: the E side must carry the `_h` sub-lattice
        # (stepping.py:1015) and the H side must not (stepping.py:948).
        alone = device.Residency()
        solo = bfast.plan_bfast_run_constitutive(fields, pml, side, alone,
                                                 chunk_bounds=layer_record(fields.grid, pml))
        sub_lattice[slot] = sorted(n for n in alone.names
                                   if n.startswith(("pml:kps_", "pml:kms_")))
        wrong = [n for n in sub_lattice[slot]
                 if n.endswith("_h") != (side == "E")]
        assert solo is not None, (
            f"the family's own constitutive builder refused {slot} on a "
            f"configuration its arm admits")
        assert sub_lattice[slot], (
            f"VACUOUS sub-lattice check on {slot}: the plan bound no PML "
            f"coefficient mirror at all")
        assert not wrong, (
            f"{slot} bound the WRONG Yee sub-lattice coefficients {wrong}: a swap "
            f"here is a half-cell error in the absorber profile — smooth, "
            f"converged and wrong")
    row = {"shipped_plan_step_replaces": filled,
           "selected_arm_by_slot": selected,
           "plan_class_by_slot": classes,
           "curl_slots_carry_this_familys_plan": curl_plan_classes,
           "constitutive_slots_carry_the_certified_plan_class": constitutive_ok,
           "constitutive_entry_point_is_the_certified_source":
               certified_entry_point,
           "constitutive_sources_compiled_in_this_process":
               len(compiled_constitutive),
           "uncertified_constitutive_sources_compiled": len(uncertified),
           "constitutive_coefficient_mirrors_by_slot": sub_lattice,
           "residency_verdict_held": bool(composed.residency
                                          and composed.residency.covered),
           "residency_reasons": list(getattr(composed.residency, "reasons", ()) or ())}
    rows.append(row)
    log(f"[leg9 composition] shipped plan_step replaces={filled} "
        f"selected={selected} residency_held={row['residency_verdict_held']}")
    payload["legs"]["composition"] = rows
    save(payload, out)
    assert filled == ["step_B", "update_H", "step_D", "update_E"], (
        f"the shipped composer did not fill every sub-step on a BFAST run: {row}")
    assert all(curl_plan_classes.values()), (
        f"a curl slot was filled by something other than this family's plan: {row}")
    assert all(constitutive_ok.values()), (
        f"a constitutive slot was filled by something other than the certified "
        f"ConstitutivePlan: {row}")
    assert all(certified_entry_point.values()), (
        f"a constitutive slot runs an entry point compiled from a source that is "
        f"NOT the certified text, so 'no new arithmetic' is not what shipped: "
        f"{row}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

LEGS = (
    ("execution", leg_execution),
    ("reference", leg_reference),
    ("identity", leg_identity),
    ("multi_step", leg_multi_step),
    ("pointer", leg_pointer),
    ("refusal", leg_refusal),
    ("signed_zero", leg_signed_zero),
    ("precondition", leg_precondition),
    ("mutations", leg_mutations),
    ("composition", leg_composition),
)

#: Defects this gate CANNOT hold because they are byte-invisible in float32. Named
#: with the reason and with the SOURCE-TEXT assertion that pins each instead, so
#: the record shows what was measured and what was not. There is no PTX-equivalent
#: audit on this executor, which is why the substitute pin is a text assertion and
#: why saying so matters.
PREDICTED_NULLS = (
    ("shifted_plus_centre_sum_order",
     "`(c_y + c)` versus `(c + c_y)` is the same float32 number for every input "
     "including signed zeros -- IEEE addition is commutative in bits -- so no "
     "operand can distinguish them",
     "test_metal_bfast.test_the_sums_put_the_shifted_operand_first"),
    ("d_side_negation_rounded_once",
     "negating in host float64 and then rounding to float32 gives the same word "
     "as rounding and then negating -- f64 negation and f32 rounding commute "
     "exactly -- so binding either cannot be told apart",
     "test_metal_bfast.test_the_d_side_negates_before_rounding"),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="artifact JSON path")
    parser.add_argument("--legs", default="", help="comma-separated leg subset")
    arguments = parser.parse_args()

    try:
        import torch  # noqa: PLC0415

        if not torch.backends.mps.is_available():
            return cannot_certify("no MPS device on this host")
    except Exception as exc:  # noqa: BLE001
        return cannot_certify(f"torch is not importable ({exc!r})")

    out_dir = os.path.dirname(os.path.abspath(arguments.out)) or "."
    os.makedirs(out_dir, exist_ok=True)
    started = time.time()
    payload: Dict[str, Any] = {
        "environment": environment_stamp(),
        # THE CLAIM IS ONLY AS GOOD AS THE PRECONDITION IT WAS CERTIFIED UNDER.
        "subnormal_policy": subnormal.mps_policy_report(),
        "provenance": write_provenance(
            {"stepping.py": os.path.join(API_ROOT, "meep_gpu", "stepping.py"),
             "metal_kernels/bfast_curl.py": os.path.join(
                 API_ROOT, "meep_gpu", "metal_kernels", "bfast_curl.py"),
             "metal_kernels/shaders.py": os.path.join(
                 API_ROOT, "meep_gpu", "metal_kernels", "shaders.py"),
             "metal_kernels/templates.py": os.path.join(
                 API_ROOT, "meep_gpu", "metal_kernels", "templates.py"),
             "metal_kernels/device.py": os.path.join(
                 API_ROOT, "meep_gpu", "metal_kernels", "device.py"),
             "metal_kernels/plans.py": os.path.join(
                 API_ROOT, "meep_gpu", "metal_kernels", "plans.py"),
             "metal_kernels/preconditions.py": os.path.join(
                 API_ROOT, "meep_gpu", "metal_kernels", "preconditions.py"),
             "gate_metal_bfast.py": os.path.abspath(__file__)},
            out_dir,
            {label: shaders.source_sha256(source) for label, source
             in bfast.enumerate_bfast_sources().items()}),
        "package_fingerprint_state": package_fingerprint_state(),
        "legs": {},
    }
    save(payload, arguments.out)
    drift = payload["package_fingerprint_state"]
    if drift["host_modules_drifted"]:
        log(f"[fingerprints] HOST DRIFT outside this family: "
            f"{drift['host_modules_drifted']} — recorded, not re-cut")

    wanted = tuple(n.strip() for n in arguments.legs.split(",") if n.strip())
    log(f"[env] torch={payload['environment']['torch']} "
        f"frontend={payload['environment']['metal_frontend']} "
        f"policy={payload['subnormal_policy']['resolved']}")
    if not payload["subnormal_policy"]["admitted"]:
        return cannot_certify(
            f"the MPS executor refused the resolved subnormal policy: "
            f"{payload['subnormal_policy']['reasons']}")

    executed: List[str] = []
    for leg_name, leg in LEGS:
        if wanted and leg_name not in wanted:
            log(f"[skip] leg {leg_name} (not in --legs)")
            continue
        log(f"=== LEG {leg_name} ===")
        leg(payload, arguments.out)
        executed.append(leg_name)

    compared = sum(len(rows) for rows in payload["legs"].values()
                   if isinstance(rows, list))
    certified, problems = certified_from_rows(payload["legs"])
    payload["summary"] = summarize(
        executed, compared,
        claim=("byte-identity to stepping.py on this host, under a CHECKED "
               "subnormal-free precondition (leg 7) — not a stated tolerance"),
        scope=("real-field split-field PML curl with the BFAST second additive "
               "pass and its three IIR states (step_B/step_D); periodic and "
               "metallic ghost rules; no fold, no cylindrical axis, no Bloch "
               "phase, no beta, no conductivity, no chi2/chi3, real float32 "
               "storage only; a nonzero k component on a DECLARED-invariant axis "
               "is admitted (clause 11a) and byte-compared in legs 1 and 5. THE "
               "FAMILY ALSO CLAIMS "
               "update_H/update_E as of tranche 2, with NO NEW ARITHMETIC — the "
               "CERTIFIED constitutive kernel under a restated predicate; leg 9 "
               "checks that the source compiled there is the certified text and "
               "that each side binds its own Yee sub-lattice, and "
               "probe_metal_bfast_engine_route's composed leg byte-compares the "
               "whole four-slot route per complete driver step"),
        elapsed=time.time() - started, certified=certified, problems=problems,
        wiring=wiring_from_rows(payload["legs"].get("composition")))
    save(payload, arguments.out)
    for problem in problems:
        log(f"[summary] DISAGREEING ROW {problem}")
    log(f"METAL BFAST GATE {payload['summary']['status']} "
        f"(compared={compared}, certified={certified}) in "
        f"{payload['summary']['elapsed_s']}s -> {arguments.out}")
    return 0 if payload["summary"]["passed"] else 1


if __name__ == "__main__":
    from metal_gate_runner import run_current_measurement

    raise SystemExit(run_current_measurement(__file__, sys.argv[1:]))
