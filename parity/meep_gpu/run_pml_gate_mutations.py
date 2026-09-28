"""Drive the real-field PML bit-identity gate and every mutation that pins it.

A gate no mutation exercises is indistinguishable from one that returns True.
This runs the gate once, then runs each mutation in its OWN PROCESS — the
source rewrites are monkeypatches on a loaded module, and a fresh interpreter
is the only isolation worth trusting between them — and records, per leg,
whether the gate's verdict flipped the way it must.

Three kinds of leg, and they are judged by different rules:

* KERNEL/CONFIG mutations (3-7, plus the auxiliary-store defect) break the
  kernel or lie to it. Each must be CAUGHT: the gate's verdict must go False.
* NULL mutations must be UNCAUGHT. A battery of only-must-be-caught legs scores
  identically whether the comparator works or has degenerated into failing
  everything: nine legs failing as required is the same picture either way. Each
  null is paired with a leg that edits the SAME expression and must be caught, so
  "inert" is a measurement rather than a leg nobody showed could fail.
* HARNESS mutations (1 and 8) weaken the gate itself. Each is paired with a
  real defect and must let that defect go UNCAUGHT — that is what proves the
  byte comparison and the auxiliary comparison are load-bearing rather than
  decorative.

Mutation 2 (drop the rounding guard) needs no leg of its own: the gate already
sweeps both guard sets, so the pair "identical with, non-identical without" is
read straight out of the gate artifact.

EVERY LEG WITH A SOURCE MUTATION MUST ACCOUNT FOR IT. The probe records how many
times the mutated bytes were HANDED TO THE COMPILER — one log entry per
``cp.RawKernel`` construction, which is not the same as one NVRTC compile, since
CuPy's disk cache can answer above the strip seam (measured: a cleared memo, a
fresh construction, and identical bytes back). For an ARMED mutation the
distinction does not bite, because mutated source is a miss in that source-keyed
disk cache; the counter is still named for what it counts. A leg that reports a
verdict with that count at zero is refused here rather than believed — a leg
reporting a pass for a mutation it never applied is indistinguishable, in a
summary, from a leg that applied it and found nothing, and the sibling track hit
exactly that three times.

EVERY LEG GETS A PRIVATE, POLICY-TOKEN-CARRYING ``CUPY_CACHE_DIR``. CuPy computes
its kernel cache key ABOVE the seam ``subnormal_policy`` strips ``-ftz=true`` at,
so a shared cache directory serves one policy's binaries under the other policy's
name and no amount of clearing an in-process memo helps. The directory is derived
from the RESOLVED policy (``match_meep`` is a measurement, not a policy an
executor can be driven to) and the leg name, both so the legs cannot poison each
other and so a keep-policy install (which REFUSES a directory without the
``ftz_stripped`` token) is satisfied by construction.

Run (the GPU host, one GPU)::

    CUDA_VISIBLE_DEVICES=0 python -u run_pml_gate_mutations.py \\
        --module ../../meep_gpu/cuda_kernels/step_curl_kernels.py \\
        --track hand --subnormal-policy keep \\
        --out-dir results/fused_pml_bit_identity_hand_<date>_keep

and again with ``--subnormal-policy flush`` into its own directory: the two
verdicts side by side are what closes §1.1, and neither alone says anything the
other does not.

THE TWO-POLICY RUN IS ONLY WORTH A SLOT IF THE LEG'S OPERANDS REACH THE BAND THE
POLICY GOVERNS, and that is a property of the fixture, not of this driver. Both
families now sweep a value class (``PML_VALUE_CLASSES`` and
``CONSTITUTIVE_VALUE_CLASSES``) and both publish a band count beside the
normal-number pass, so each leg's record says which class it measured. The
constitutive family had no such axis until 2026-08-15: its draws were
``uniform(-1, 1)`` for the fields, ``uniform(0.2, 0.9)`` for inverse epsilon and
``uniform(0.5, 1.0)`` for the coefficients, which over the four swept shapes and
both sides produced 1,907,550 float32 operands and intermediates containing zero
subnormals, zero signed zeros and zero zeros. Running THAT twice would have cost
two device slots to establish nothing, and the summary would have been silent
rather than wrong — ``single_launch_subnormal_band`` was recorded as ``None`` by
name. ``single_launch_band_is_non_vacuous`` is carried per leg so the same
failure cannot recur unnoticed.

One flushed line per leg as it lands, and the summary is rewritten after each
(the progress-reporting rule) so an interrupted run keeps everything up to the failure.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

HERE = os.path.dirname(os.path.abspath(__file__))
PROBE = os.path.join(HERE, "probe_fused_kernel_bit_identity.py")
REPO_API = os.path.abspath(os.path.join(HERE, "..", ".."))


def resolved_policy(policy: Optional[str]) -> Optional[str]:
    """The policy that will actually be INSTALLED in a leg, not the one requested.

    ``match_meep`` — the shipped default, and so the likely thing to type — is
    not a policy an executor can be driven to. It is a MEASUREMENT: MEEP's own
    build flushes on x86 and keeps on arm64, and ``subnormal_policy`` resolves it
    to ``flush`` or ``keep`` accordingly. Everything downstream that branches on
    the policy has to branch on the ANSWER:

    * :func:`leg_cache_dir` must emit an ``ftz_stripped`` directory whenever the
      resolution is ``keep``, because ``subnormal_policy.cupy_cache_reasons``
      REFUSES a keep install into a directory without that token — measured:
      ``policy_match_meep`` was refused with "does not carry the policy token
      'ftz_stripped'``", so every leg died at the install;
    * ``CUPY_ACCELERATORS`` must be emptied whenever the resolution is ``flush``,
      because CuPy's CUB reductions are governed by that variable at import time
      and ``install_cupy_policy`` refuses ``flush`` by name without it.

    The resolution is taken HERE, in the parent, and it is valid for the legs
    because they are subprocesses on this same machine — the measurement is of
    the host. Returns ``None`` unchanged: no policy requested means no install.
    """
    if policy is None:
        return None
    if REPO_API not in sys.path:
        sys.path.insert(0, REPO_API)
    from meep_gpu import subnormal_policy  # noqa: PLC0415

    return subnormal_policy.resolve_policy(policy)[0]

# WHICH SUB-STEP FAMILY A LEG MEASURES. A leg names one; the family decides the
# probe's ``--experiments``, which artifact sections carry its counts, and which
# verdict key holds its fraction. Before the constitutive pair existed every leg
# was the curl and all four were hard-coded — and a constitutive leg run under
# the curl's wiring would have read ``curl_single_launch`` (absent, so 0/0),
# reported ``single_ran == 0``, and failed for a reason that says nothing about
# the kernel. Keeping the mapping in one table is what stops that.
FAMILIES: Dict[str, Dict[str, Any]] = {
    "curl": {
        "experiments": "pml,multistep",
        "single_section": "pml_bit_identity",
        "multi_section": "pml_multi_step",
        "single_key": "curl_single_launch",
        "band_key": "curl_single_launch_subnormal_band",
        "multi_key": "curl_multi_step_all_identical",
    },
    "constitutive": {
        "experiments": "constitutive,constitutive_multistep",
        "single_section": "constitutive_bit_identity",
        "multi_section": "constitutive_multi_step",
        "single_key": "constitutive_single_launch",
        # The constitutive leg gained its value-class axis on 2026-08-15. Before
        # that this key had no counterpart and the runner recorded None here by
        # name, which was honest about the artifact and silent about the reason:
        # the leg's operands provably could not reach the band, so the two-policy
        # run this driver's docstring prescribes agreed by construction.
        "band_key": "constitutive_single_launch_subnormal_band",
        "multi_key": "constitutive_multi_step_all_identical",
    },
}

DEFAULT_FAMILY = "curl"


def family_of(leg: Dict[str, Any]) -> Dict[str, Any]:
    """The family wiring for one leg; ``curl`` unless the leg says otherwise."""
    return FAMILIES[leg.get("family", DEFAULT_FAMILY)]


# (leg, expectation, extra probe arguments, why this leg exists)
#   expectation "caught"   -> the gate verdict must be False
#   expectation "uncaught" -> the gate verdict must be True, and that is the point
#   family                 -> which sub-step's wiring; "curl" when omitted
LEGS: Sequence[Dict[str, Any]] = (
    {
        "leg": "gate",
        "expect": "pass",
        "args": ["--coefficients", "synthetic,real"],
        "why": "the gate itself: the full product, both coefficient sources, "
               "both guard sets, plus the multi-step leg",
    },
    {
        "leg": "m3_regroup_stencil",
        "expect": "caught",
        "args": ["--source-mutation", "regroup_stencil", "--guard", "fmad_false",
                 "--coefficients", "synthetic"],
        "why": "mutation 3: ((a-b)+(c-d)) rewritten as C's ((a-b)+c)-d. If someone "
               "later 'simplifies' the parentheses, the gate must fail.",
    },
    {
        "leg": "m4_drop_metallic_mask",
        "expect": "caught",
        "args": ["--source-mutation", "drop_metallic_mask", "--guard", "fmad_false",
                 "--coefficients", "synthetic"],
        "why": "mutation 4: the wall cells stop being masked, so an unowned cell "
               "integrates a curl assembled from a ghost it does not own",
    },
    {
        "leg": "m5_swap_dsig_dsigu",
        "expect": "caught",
        "args": ["--source-mutation", "swap_dsig_dsigu", "--guard", "fmad_false",
                 "--coefficients", "synthetic"],
        "why": "mutation 5: the auxiliary's PML axis swapped with the field's",
    },
    {
        "leg": "m6_integer_coefficients_on_B",
        "expect": "caught",
        "args": ["--host-mutation", "integer_coefficients_on_B", "--guard",
                 "fmad_false", "--coefficients", "synthetic"],
        "why": "mutation 6: the D-side integer table fed to the B side — a "
               "half-cell error, not a crash",
    },
    {
        "leg": "m7_metallic_as_periodic",
        "expect": "caught",
        "args": ["--host-mutation", "metallic_as_periodic", "--guard", "fmad_false",
                 "--coefficients", "synthetic"],
        "why": "mutation 7: a wall told to the kernel as a wrap — what porting the "
               "complex kernels' hard-coded layout produces",
    },
    {
        "leg": "m8x_drop_fu_store",
        "expect": "caught",
        "args": ["--source-mutation", "drop_fu_store", "--guard", "fmad_false",
                 "--coefficients", "synthetic"],
        "why": "the auxiliary-only defect: fu_new is computed and used for f but "
               "never stored. f is right on every single launch; fu is wrong from "
               "the first. Only the auxiliary comparison can see it.",
    },
    {
        "leg": "m9_read_fprev_after_store",
        "expect": "caught",
        "args": ["--source-mutation", "read_fprev_after_store", "--guard",
                 "fmad_false", "--coefficients", "synthetic"],
        "why": "the aliasing trap: fprev read back out of fu AFTER the new value "
               "lands, so the recurrence loses its history term. It is also the "
               "DISCRIMINATOR for n1 below — same two lines, one lethal and one "
               "inert — without which 'the reload is inert' would be a claim "
               "about a leg nobody showed could fail.",
    },
    {
        "leg": "n1_reload_fu_from_memory_must_be_uncaught",
        "expect": "uncaught",
        "args": ["--source-mutation", "reload_fu_from_memory", "--guard",
                 "fmad_false", "--coefficients", "synthetic"],
        "why": "NULL: use the value just stored to fu instead of the register. A "
               "float32 store-then-load cannot re-round, so this must come back "
               "N/N identical AND multi-step identical. A leg reporting it CAUGHT "
               "is reporting a defect in the kernel or the harness.",
    },
    {
        "leg": "n2_commute_dtdx_scale_must_be_uncaught",
        "expect": "uncaught",
        "args": ["--source-mutation", "commute_dtdx_scale", "--guard",
                 "fmad_false", "--coefficients", "synthetic"],
        "why": "NULL: dtdx * curl written curl * dtdx. IEEE multiply commutes. Its "
               "discriminator is m3, which edits the same expression's ASSOCIATION "
               "and is measured caught — so the pair says the comparator is "
               "sensitive to the grouping and insensitive to the operand order.",
    },
    {
        "leg": "m10_fortran_order_index_decomposition",
        "expect": "caught",
        "args": ["--source-mutation", "fortran_order_index_decomposition",
                 "--guard", "fmad_false", "--coefficients", "synthetic"],
        "why": "THE LINEAR-INDEX DECOMPOSITION, on the CERTIFIED pair. It is the "
               "same three lines in all four gated kernels, and it was armed in "
               "none of them: every curl mutation on this list edits the stencil, "
               "the mask or the recurrence, and none edits which cell a thread "
               "believes it is. Found while arming the constitutive pair's "
               "version (c8) — the needle resolved on the curl kernels too, which "
               "is how a hole in the certified pair's own battery surfaced. In "
               "bounds on every swept shape by construction.",
    },
    {
        "leg": "m1_allclose_hides_regroup",
        "expect": "some_uncaught",
        "args": ["--allclose", "--source-mutation", "regroup_stencil", "--guard",
                 "fmad_false", "--coefficients", "synthetic"],
        "why": "harness mutation 1: relax the byte comparison to np.allclose and "
               "count how many cases of a REAL regrouping defect walk through. The "
               "byte gate catches all of them; any that np.allclose lets past is a "
               "case where a magnitude comparison would have shipped the defect.",
    },
    {
        "leg": "m8_no_fu_vs_drop_mask",
        "expect": "measure",
        "args": ["--no-fu-compare", "--source-mutation", "drop_metallic_mask",
                 "--guard", "fmad_false", "--coefficients", "synthetic"],
        "why": "harness mutation 8 as specified: drop the auxiliary comparison and "
               "ask whether mutation 4 still lands. MEASURED, not assumed — the "
               "masked curl feeds the recurrence, so the target field may well "
               "carry the defect on its own.",
    },
    {
        "leg": "m8b_no_fu_hides_drop_fu_store",
        "expect": "single_all_uncaught",
        "args": ["--no-fu-compare", "--source-mutation", "drop_fu_store", "--guard",
                 "fmad_false", "--coefficients", "synthetic"],
        "why": "harness mutation 8, against the defect that really is "
               "auxiliary-only. A single-launch, target-only gate must be TOTALLY "
               "blind to it — that is R2 measured rather than argued — while the "
               "gate with fu (m8x) catches it on launch one and the multi-step leg "
               "catches it from step two.",
    },
    # ----------------------------------------------------------------------
    # THE REAL-STORAGE CONSTITUTIVE PAIR (update_H + update_E).
    #
    # Authored, predicated, NOT YET CERTIFIED: these legs are what would move
    # ``constitutive_kernels.CERTIFIED_KERNELS`` off empty, and they need an
    # A6000. They are written here rather than after the device run so that the
    # gate's shape is reviewable, and its mutation set arguable, before a slot
    # is spent — and so a slot is spent once.
    #
    # Six mutations: the four the gate requires of any constitutive kernel
    # (``CONSTITUTIVE_SOURCE_MUTATIONS``), two host mutations that only exist
    # once a sub-lattice and a coefficient pair are bound on the host, and two
    # NULLS that must come back UNCAUGHT, each paired with a defect on the same
    # expression that must be caught.
    # ----------------------------------------------------------------------
    {
        "leg": "cgate",
        "family": "constitutive",
        "expect": "pass",
        "args": ["--coefficients", "synthetic,real"],
        "why": "the constitutive gate itself: 4 shapes x 4 boundary sets x 2 "
               "Courants x 2 sides x both coefficient sources, both guard sets, "
               "plus the multi-step leg. The auxiliary f_w is compared "
               "throughout — it is this sub-step's fu.",
    },
    {
        "leg": "c1_regroup_constitutive",
        "family": "constitutive",
        "expect": "caught",
        "args": ["--source-mutation", "regroup_constitutive", "--guard",
                 "fmad_false", "--coefficients", "synthetic"],
        "why": "((f + kps*src) - kms*prev) flattened to f + (kps*src - kms*prev). "
               "The array path accumulates twice, left to right; the flattened "
               "form is what BOTH complex constitutive kernels in "
               "step_curl_kernels.py are written as, so this is the defect the "
               "template would have introduced.",
    },
    {
        "leg": "c2_drop_fw_store",
        "family": "constitutive",
        "expect": "caught",
        "args": ["--source-mutation", "drop_fw_store", "--guard", "fmad_false",
                 "--coefficients", "synthetic"],
        "why": "the auxiliary-only defect: f_w is never written. The target is "
               "bit-identical on launch one and wrong from launch two, so only "
               "the auxiliary comparison and the multi-step leg can see it.",
    },
    {
        "leg": "c3_store_fw_before_reading_prev",
        "family": "constitutive",
        "expect": "caught",
        "args": ["--source-mutation", "store_fw_before_reading_prev", "--guard",
                 "fmad_false", "--coefficients", "synthetic"],
        "why": "the aliasing trap the array path's fw.copy() exists to prevent: "
               "prev becomes the value just stored and the recurrence loses its "
               "history term. Wrong only where kms != 0, i.e. inside the layer.",
    },
    {
        "leg": "c4_drop_inverse_epsilon",
        "family": "constitutive",
        "expect": "caught",
        "args": ["--source-mutation", "drop_inverse_epsilon", "--guard",
                 "fmad_false", "--coefficients", "synthetic"],
        "why": "E side: the source becomes D instead of D*inv_eps. It is also the "
               "DISCRIMINATOR for cn1 — same three multiplications, one lethal "
               "and one inert.",
    },
    {
        "leg": "c5_swap_constitutive_sublattice",
        "family": "constitutive",
        "expect": "caught",
        "args": ["--host-mutation", "swap_constitutive_sublattice", "--guard",
                 "fmad_false", "--coefficients", "synthetic"],
        "why": "the half-cell error: integer coefficients on E, half-integer on "
               "H. A HOST mutation because the sub-lattice is chosen by the "
               "wrapper (stepping.py:948 vs :1015), not by the kernel — the "
               "kernel never chooses and cannot choose wrong, but its caller can.",
    },
    {
        "leg": "c6_swap_kps_kms",
        "family": "constitutive",
        "expect": "caught",
        "args": ["--host-mutation", "swap_kps_kms", "--guard", "fmad_false",
                 "--coefficients", "synthetic"],
        "why": "(kap+sig) and (kap-sig) exchanged: the sign of the absorption "
               "reversed on the history term. Outside the layer kps == kms == 1 "
               "and it is exactly bit-identical, so it can ONLY be caught inside "
               "the absorber — which is what makes it a test of whether the "
               "swept shapes reach the layer at all.",
    },
    {
        "leg": "c7_own_axis_to_x_for_all_three",
        "family": "constitutive",
        "expect": "caught",
        "args": ["--source-mutation", "own_axis_to_x_for_all_three", "--guard",
                 "fmad_false", "--coefficients", "synthetic"],
        "why": "THE AXIS MAPPING, which the kernel's own note 2 calls the "
               "highest-consequence confusion in the file — MEEP's dsigw is the "
               "absorption a component accumulates along the direction it points "
               "in, so component 0 reads the x table, 1 the y, 2 the z. Nothing "
               "armed it until 2026-08-15. Every other constitutive mutation "
               "edits the ARITHMETIC; this one edits WHICH COEFFICIENT A CELL "
               "READS, and the result is a converged, smooth, entirely wrong "
               "absorber rather than a visibly broken field. In bounds on every "
               "shape by construction, so it must be caught as a wrong answer "
               "and not as a launch failure.",
    },
    {
        "leg": "c8_fortran_order_index_decomposition",
        "family": "constitutive",
        "expect": "caught",
        "args": ["--source-mutation", "fortran_order_index_decomposition",
                 "--guard", "fmad_false", "--coefficients", "synthetic"],
        "why": "the other half of the indexing question: the linear index "
               "decomposed for column-major storage, so i and k swap and each "
               "component reads another axis's profile. Also in bounds on every "
               "shape by construction — a decomposition that read PAST a "
               "coefficient vector would be caught by a fault on some shapes and "
               "by garbage on others, and neither is the defect being measured.",
    },
    {
        "leg": "c9_bind_Ez_inv_eps_for_all_three",
        "family": "constitutive",
        "expect": "caught",
        "args": ["--source-mutation", "bind_Ez_inv_eps_for_all_three", "--guard",
                 "fmad_false", "--coefficients", "synthetic"],
        "why": "defect 2 of the complex template, armed: update_E_pml_complex "
               "passes fields.inv_eps — the Ez view (fields.py:1259-1260) — for "
               "all three components, so a diagonal anisotropic epsilon updates "
               "Ex and Ey with Ez's material. c4 removes the multiply entirely "
               "and could never see it. It is EXACTLY bit-identical under an "
               "isotropic epsilon, so this leg is also the measurement that the "
               "fixture's three independent inv_eps volumes are load-bearing: a "
               "verdict of UNCAUGHT here reports a collapsed fixture, not a "
               "correct kernel.",
    },
    {
        "leg": "cn1_inv_eps_left_must_be_uncaught",
        "family": "constitutive",
        "expect": "uncaught",
        "args": ["--source-mutation", "inv_eps_left", "--guard", "fmad_false",
                 "--coefficients", "synthetic"],
        "why": "NULL: inv_eps * D in place of D * inv_eps. IEEE multiply "
               "commutes; the sibling track measured this exact null at 30/30 "
               "identical. Its discriminator is c4, which edits the same three "
               "multiplications and must be caught.",
    },
    {
        "leg": "cn2_commute_constitutive_scale_must_be_uncaught",
        "family": "constitutive",
        "expect": "uncaught",
        "args": ["--source-mutation", "commute_constitutive_scale", "--guard",
                 "fmad_false", "--coefficients", "synthetic"],
        "why": "NULL: src * kps in place of kps * src on the accumulation "
               "itself. Its discriminator is c1, which edits the ASSOCIATION of "
               "the same two lines and must be caught — so the pair says the "
               "comparator is sensitive to the grouping and insensitive to the "
               "operand order, which is what IEEE-754 says.",
    },
    {
        "leg": "cm8_no_fw_hides_drop_fw_store",
        "family": "constitutive",
        "expect": "single_all_uncaught",
        "args": ["--no-fu-compare", "--source-mutation", "drop_fw_store",
                 "--guard", "fmad_false", "--coefficients", "synthetic"],
        "why": "harness mutation 8 on this sub-step: drop the auxiliary "
               "comparison and a single-launch gate must be TOTALLY blind to the "
               "auxiliary-only defect, while c2 catches it with f_w compared. "
               "That pair is what makes 'the f_w comparison is load-bearing' a "
               "measurement rather than an assumption.",
    },
)


def log(message: str) -> None:
    print(message, flush=True)


def leg_cache_dir(out_dir: str, policy: Optional[str], leg_name: str) -> str:
    """A PRIVATE CuPy kernel cache for this leg, named after the policy INSTALLED.

    Two independent reasons, and either alone would be enough:

    * CuPy's cache key is computed above the ``compile_using_nvrtc`` seam where
      the ``"keep"`` policy strips ``-ftz=true``, so the same source under two
      policies has ONE key and two answers. A shared directory therefore either
      gets poisoned with stripped binaries or silently serves flushed ones —
      which is why ``subnormal_policy.cupy_cache_reasons`` REFUSES a keep install
      whose directory does not carry ``ftz_stripped``, and refuses a flush install
      whose directory does.
    * A per-leg directory also makes each mutation leg compile cold, so a leg
      cannot be served the previous leg's unmutated binary off disk.

    ``policy`` is the RESOLVED policy (:func:`resolved_policy`), never the raw
    request: keying on the literal string sent ``match_meep`` to a
    ``policy_match_meep`` directory that the keep install then refused, so every
    leg of that run died before it launched anything.
    """
    token = "ftz_stripped" if policy == "keep" else f"policy_{policy or 'default'}"
    return os.path.join(os.path.abspath(out_dir), "cupy_cache", token, leg_name)


def run_leg(leg: Dict[str, Any], module: str, track: str, out_dir: str,
            repo_root: str, python: str,
            policy: Optional[str] = None,
            resolved: Optional[str] = None) -> Dict[str, Any]:
    """One leg in its own process. ``policy`` is REQUESTED, ``resolved`` INSTALLED.

    The probe is handed the request (so its own stamp records what was asked for
    and the measurement behind it); everything this function decides — the cache
    directory and ``CUPY_ACCELERATORS`` — is decided by the RESOLUTION, because
    those two are refusals keyed on what actually gets installed.
    """
    if resolved is None:
        resolved = resolved_policy(policy)
    family = family_of(leg)
    artifact = os.path.join(out_dir, f"{leg['leg']}.json")
    command = [
        python, "-u", PROBE,
        "--module", module,
        "--track", track,
        "--out", artifact,
        "--experiments", family["experiments"],
        "--repo-root", repo_root,
        "--label", leg["leg"],
    ] + (["--subnormal-policy", policy] if policy else []) + (
        # Keyed on the RESOLUTION, like the two environment levers below and for
        # the same reason: on x86 'match_meep' resolves to flush, and a flush
        # install is REFUSED in a process that has not imported MEEP — the host
        # FTZ/DAZ bits are only reachable through mp.set_zero_subnormals, and
        # meep_gpu may not import MEEP itself. Measured: without this the whole
        # flush leg raises SubnormalPolicyUnattainable before the first compile.
        ["--import-meep-for-host-policy"] if resolved == "flush" else []
    ) + list(leg["args"])

    cache_dir = leg_cache_dir(out_dir, resolved, leg["leg"])
    os.makedirs(cache_dir, exist_ok=True)
    # MEASURED BEFORE THE RUN, or it measures nothing: after the leg the directory
    # holds whatever the leg compiled. A leg that found a warm cache may have been
    # served a binary it did not compile, which is the whole reason the directory
    # is per leg and per policy.
    cache_was_cold = not os.listdir(cache_dir)
    environment = dict(os.environ)
    environment["CUPY_CACHE_DIR"] = cache_dir
    if resolved == "flush":
        # CuPy's CUB reduction accelerator is a binary built with CuPy, so the
        # only lever is the environment BEFORE the import; install_cupy_policy
        # refuses 'flush' without it, by name. Keyed on the RESOLUTION: on x86
        # 'match_meep' resolves to flush, and keying on the literal request left
        # this unset and the install refused.
        environment["CUPY_ACCELERATORS"] = ""

    started = time.time()
    completed = subprocess.run(command, capture_output=True, text=True,
                               env=environment)
    elapsed = round(time.time() - started, 2)
    record: Dict[str, Any] = {
        "leg": leg["leg"], "expect": leg["expect"], "why": leg["why"],
        "command": command, "returncode": completed.returncode,
        "seconds": elapsed, "artifact": artifact,
        "subnormal_policy_requested": policy,
        "subnormal_policy_resolved": resolved,
        "CUPY_ACCELERATORS": environment.get("CUPY_ACCELERATORS", "<inherited>"),
        "CUPY_CACHE_DIR": cache_dir,
        "cupy_cache_was_cold": cache_was_cold,
    }
    tail = "\n".join(completed.stdout.strip().splitlines()[-25:])
    record["stdout_tail"] = tail
    record["stderr_tail"] = "\n".join(completed.stderr.strip().splitlines()[-25:])
    if os.path.exists(artifact):
        with open(artifact) as handle:
            payload = json.load(handle)
        record["verdict"] = payload.get("verdict")
        record["mutation"] = payload.get("mutation")
        record["guard_sets"] = payload.get("guard_sets")
        record["subnormal_policy"] = payload.get("subnormal_policy")
        single = payload.get(family["single_section"], {}).get(leg["leg"], {})
        record["per_guard"] = single.get("per_guard")
        record["per_value_class"] = single.get("per_value_class")
        multi = payload.get(family["multi_section"], {}).get(leg["leg"], {})
        record["multi_step_budget"] = multi.get("steps_budget")
        record["multi_step_all_identical"] = multi.get("all_steps_identical")
        record["multi_step_first_divergences"] = [
            {"shape": r["shape"], "sub_step": r["sub_step"],
             "identical_steps": r["identical_steps"],
             "first_divergence": r["first_divergence"]}
            for r in multi.get("runs", [])]
    record["family"] = leg.get("family", DEFAULT_FAMILY)
    verdict = record.get("verdict") or {}
    gate_passed = bool(verdict.get("pass"))
    record["gate_passed"] = gate_passed
    identical, ran = _split_fraction(verdict.get(family["single_key"]))
    record["single_identical"] = identical
    record["single_ran"] = ran
    # THE BAND COUNT COMES FROM THE LEG'S OWN FAMILY. Both families now sweep the
    # value class, so reading the curl's key on a constitutive leg would report
    # None for a sweep that ran — which is how this line read until 2026-08-15,
    # when the constitutive leg had no band axis to report.
    record["single_launch_subnormal_band"] = verdict.get(family["band_key"])
    # Read off the leg's own summarizer block rather than the verdict, because
    # ``summarize_pml`` computes it for every family and the verdict only
    # republishes some of them. A band sweep whose operands held no subnormal
    # exercised the policy exactly as much as the uniform class did.
    record["single_launch_band_is_non_vacuous"] = single.get(
        "subnormal_band_is_non_vacuous")
    record["single_launch_band_operands"] = single.get("subnormal_band_operands")

    # THE ARMED-MUTATION ACCOUNTING. A leg carrying a source mutation whose bytes
    # never reached NVRTC has measured nothing, whatever its verdict says.
    accounting = (record.get("mutation") or {}).get("armed_accounting") or {}
    record["armed_accounting"] = accounting
    armed_ok = True
    if accounting.get("applies"):
        if not accounting.get("measurable"):
            armed_ok = False
            record["armed_accounting_problem"] = (
                "a source mutation was armed and this track's module keeps no "
                "compile log, so nothing can say the mutated bytes were compiled")
        elif not accounting.get("mutated_source_reached_the_compiler"):
            armed_ok = False
            record["armed_accounting_problem"] = (
                f"a source mutation was armed and the mutated bytes were handed "
                f"to the compiler "
                f"{accounting.get('compiles_from_mutated_source')} times out of "
                f"{accounting.get('compiles')} kernel constructions; this leg's "
                f"verdict describes the UNMUTATED kernel")
    record["armed_as_required"] = armed_ok

    ok = completed.returncode == 0 and ran > 0 and armed_ok
    if leg["expect"] == "pass":
        record["as_required"] = ok and gate_passed
    elif leg["expect"] == "caught":
        record["as_required"] = ok and not gate_passed
    elif leg["expect"] == "uncaught":
        # A NULL mutation: the gate must still pass, on every single-launch case
        # AND through the multi-step leg. Anything less and the transformation
        # claimed inert is not inert.
        record["as_required"] = bool(
            ok and gate_passed and identical == ran
            and record.get("multi_step_all_identical"))
    elif leg["expect"] == "some_uncaught":
        # The weakened comparator must let at least one real defect through.
        record["as_required"] = ok and identical > 0
        record["cases_the_weakened_comparator_let_through"] = identical
    elif leg["expect"] == "single_all_uncaught":
        # A single-launch, target-only gate must be blind to it in every case.
        record["as_required"] = ok and identical == ran
    else:  # "measure" — recorded, never asserted
        record["as_required"] = ok
    return record


def _split_fraction(text: Optional[str]) -> tuple:
    """Parse the verdict's ``"identical/ran"`` string; (0, 0) when absent.

    (0, 0) makes ``ran > 0`` false and so fails the leg, which is the correct
    answer for a verdict this runner could not read — and is the failure mode
    this function was silently in until 2026-08-15, when it was still asking for
    a ``single_launch`` key the probe had renamed to ``curl_single_launch``.
    """
    if not text or "/" not in text:
        return 0, 0
    left, _, right = text.partition("/")
    try:
        return int(left), int(right)
    except ValueError:
        return 0, 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", required=True)
    parser.add_argument("--track", default="hand")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--repo-root", default=os.path.abspath(
        os.path.join(HERE, "..", "..")))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--only", default=None,
                        help="comma-separated leg names, for re-running one")
    parser.add_argument("--subnormal-policy", default=None,
                        choices=("flush", "keep", "match_meep"),
                        help="install this float32 subnormal policy in every leg "
                             "and stamp every artifact with it. The 2026-08-09/10 "
                             "record names none, so it cannot be filtered by one; "
                             "run this twice, once per policy, into two output "
                             "directories. match_meep is resolved by measurement "
                             "(flush on x86, keep on arm64) before the cache "
                             "directory and CUPY_ACCELERATORS are chosen, and both "
                             "the request and the resolution go into the summary.")
    args = parser.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    wanted = ({name.strip() for name in args.only.split(",")}
              if args.only else None)
    legs = [leg for leg in LEGS if wanted is None or leg["leg"] in wanted]

    # Resolved ONCE, in the parent, and passed down: 'match_meep' is a
    # measurement of this host and the legs are subprocesses on it, so one
    # resolution governs every leg and appears in the summary beside the request.
    resolved = resolved_policy(args.subnormal_policy)
    if args.subnormal_policy:
        log(f"[policy] requested {args.subnormal_policy!r} -> resolved "
            f"{resolved!r}; leg caches under "
            f"{leg_cache_dir(args.out_dir, resolved, '<leg>')}")

    summary: Dict[str, Any] = {
        "driver": "run_pml_gate_mutations",
        "track": args.track,
        "module": os.path.abspath(args.module),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
        "subnormal_policy_requested": args.subnormal_policy,
        "subnormal_policy_resolved": resolved,
        "_subnormal_policy_note":
            "null means NO policy was installed, which is what CuPy's default "
            "does: it appends -ftz=true to every NVRTC compile, so the bytes are "
            "FLUSHED. The 2026-08-09/10 record was cut in exactly this state and "
            "does not say so anywhere. 'requested' may be match_meep, which is "
            "not a policy an executor can be driven to; 'resolved' is the "
            "measured answer for this host and is what the cache directory and "
            "CUPY_ACCELERATORS were keyed on.",
        "cupy_cache_root": os.path.join(os.path.abspath(args.out_dir), "cupy_cache"),
        "legs": [],
    }
    summary_path = os.path.join(args.out_dir, "MUTATION_SUMMARY.json")
    started = time.time()
    for index, leg in enumerate(legs, 1):
        log(f"[leg] {index}/{len(legs)} {leg['leg']} (expect {leg['expect']}) ...")
        record = run_leg(leg, args.module, args.track, args.out_dir,
                         args.repo_root, args.python, args.subnormal_policy,
                         resolved)
        summary["legs"].append(record)
        accounting = record.get("armed_accounting") or {}
        # The single-launch fraction is read through the leg's OWN family key. It
        # was hard-coded to the curl's, so every constitutive leg printed
        # ``single=None`` while its record carried the real fraction.
        log(f"[leg] {index}/{len(legs)} {leg['leg']}: gate_passed="
            f"{record['gate_passed']} expect={leg['expect']} "
            f"single={record['single_identical']}/{record['single_ran']} "
            f"as_required={record['as_required']} "
            f"band={record.get('single_launch_subnormal_band')} "
            f"band_non_vacuous={record.get('single_launch_band_is_non_vacuous')} "
            f"multi_identical={record.get('multi_step_all_identical')}"
            f"@{record.get('multi_step_budget')} "
            f"policy={((record.get('subnormal_policy') or {}).get('policy'))} "
            f"armed_compiles={accounting.get('compiles_from_mutated_source')}"
            f"/{accounting.get('compiles')} "
            f"({record['seconds']} s)")
        if record.get("armed_accounting_problem"):
            log(f"[leg] {leg['leg']} ARMED-MUTATION PROBLEM: "
                f"{record['armed_accounting_problem']}")
        if not record["as_required"]:
            log(f"[leg] {leg['leg']} STDERR TAIL:\n{record['stderr_tail']}")
        with open(summary_path + ".tmp", "w") as handle:
            json.dump(summary, handle, indent=2)
        os.replace(summary_path + ".tmp", summary_path)

    required = [r for r in summary["legs"] if r["expect"] != "measure"]
    summary["all_legs_as_required"] = all(r["as_required"] for r in required)
    summary["legs_as_required"] = (
        f"{sum(r['as_required'] for r in required)}/{len(required)}")
    armed = [r for r in summary["legs"]
             if (r.get("armed_accounting") or {}).get("applies")]
    summary["armed_mutation_accounting"] = {
        "legs_with_a_source_mutation": len(armed),
        "legs_whose_mutated_bytes_reached_the_compiler": sum(
            1 for r in armed
            if (r["armed_accounting"] or {}).get("mutated_source_reached_the_compiler")),
        "legs_reporting_a_verdict_for_a_mutation_never_compiled": [
            r["leg"] for r in armed if not r.get("armed_as_required")],
    }
    summary["null_mutations_that_must_be_uncaught"] = [
        {"leg": r["leg"], "gate_passed": r["gate_passed"],
         "single": f"{r['single_identical']}/{r['single_ran']}",
         "multi_step_all_identical": r.get("multi_step_all_identical"),
         "as_required": r["as_required"]}
        for r in summary["legs"] if r["expect"] == "uncaught"]
    summary["elapsed_seconds"] = round(time.time() - started, 2)
    with open(summary_path + ".tmp", "w") as handle:
        json.dump(summary, handle, indent=2)
    os.replace(summary_path + ".tmp", summary_path)
    log(f"[done] {summary['legs_as_required']} legs behaved as required "
        f"({summary['elapsed_seconds']} s) -> {summary_path}")
    return 0 if summary["all_legs_as_required"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
