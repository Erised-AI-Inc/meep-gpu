"""Install, measure and stamp the campaign's subnormal policy in a composition gate.

WHY THIS EXISTS. The five Triton composition gates (``probe_triton_cylindrical_
composition``, ``probe_triton_symmetry_composition``, ``probe_triton_conductivity_
composition``, ``probe_triton_engine_route`` and ``gate_triton_no_pml``) certify the
routing of 11 dispatchable arms (``fastpath.ARM_CERTIFICATION``: cylindrical 2,
folded 3, conductive PML 1, dispersive 4, no-PML 1). Until owner ruling R16 they
installed no float32 subnormal policy and stated none, so the CuPy array path they
compare against flushed (CuPy's default ``-ftz=true``) while Triton kept natively, and
``recut_composition_records.py --flush-run`` could never record ``flush`` for them: a
flush campaign of these gates released with no stamp (measured on the GPU host,
2026-10-09, ``cylindrical_composition``). The flush dispatch licence admits a
caller-installed ``flush`` per arm only on that evidence, so their 11 arms were
refused on every flush leg whatever the gates measured.

WHAT EACH GATE NOW DOES, through :func:`install_campaign_policy`, before it builds a
driver or compiles a kernel:

* reads the campaign's policy from ``MEEP_GPU_SUBNORMAL_POLICY`` (the variable the
  fleet driver exports per gate; ``keep``, the Triton table's install default, when it
  is unset). Only ``keep`` and ``flush`` are accepted: ``match_meep`` asks a question
  this host answers by measurement, and a certification run names its policy;
* drives every executor to it with ``install_subnormal_policy(policy, strict=True)``,
  the install every flag-driven Triton gate uses;
* MEASURES that it is attained (installed, resolved to the request, no executor in
  ``unattained``) and refuses by name otherwise;
* stamps ``payload["subnormal_policy"] = policy_stamp()`` -- the top-level block
  ``seed_triton_welds.policy_line`` reads first and the field most Triton welds were
  cut from -- so the re-cut and rebind tools read the attained policy without a new
  spelling.

:func:`stamp_campaign_policy` and :func:`confirm_campaign_policy` are the two calls a
gate makes (after its imports, and before it writes a passing verdict).

:func:`end_of_run_reasons` is the same measurement at the END of the run, and
re-stamps the artifact with the counters the run accumulated: every executor must
still attain the policy and the host FPU must still obey it. MEEP's import flushes the host (``mp.setup()``), so a
gate that imported MEEP after a ``keep`` install would end flushing under a ``keep``
stamp; gates that lift MEEP simulations therefore import MEEP before installing
(``import_meep=True``), and this check refuses if anything moved the host since.

A REFUSAL IS A VERDICT, NOT A CRASH. :class:`CampaignPolicyRefused` carries the
reason and the stamp as it stood, and :func:`refusal` writes the block the gate's own
verdict spelling reads (``summary.status`` or ``validation.status`` = ``refused``), so
``recut_composition_records.released`` and ``gate_provenance.read_verdict`` both read
the run as not released.

Under ``flush`` on an x86 host the host half needs MEEP imported first (its
``set_zero_subnormals`` is the only lever this package may use there). The round's
flush fleet starts every gate through its MEEP-importing wrapper
(``rounds/1.0/fleet_launch.py``); a direct flush run of a gate that does not lift MEEP
simulations refuses by name, naming that remedy.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

#: The policies a composition gate may be asked to certify.
CERTIFIABLE = ("keep", "flush")

#: What a gate installs when the campaign names no policy: the Triton table's install
#: default (``fastpath.TABLE_SUBNORMAL_POLICY["triton"]``), so a direct run with no
#: campaign around it certifies the policy dispatch installs.
DEFAULT_POLICY = "keep"

#: The artifact field the stamp is written to (``seed_triton_welds.POLICY_STAMP_BLOCKS[0]``).
STAMP_FIELD = "subnormal_policy"

#: The executors a composition gate drives: the host (the driver's NumPy setup), CuPy
#: (the array path it compares against) and Triton (the plans it certifies).
EXECUTORS = ("host", "cupy", "triton")


class CampaignPolicyRefused(RuntimeError):
    """The campaign's policy could not be installed, or did not hold. Carries the stamp."""

    def __init__(self, reason: str, stamp: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(reason)
        self.stamp = stamp


def requested_policy(environ: Optional[Dict[str, str]] = None) -> str:
    """The policy this run is asked to certify: ``MEEP_GPU_SUBNORMAL_POLICY`` or keep."""
    environ = os.environ if environ is None else environ
    value = str(environ.get("MEEP_GPU_SUBNORMAL_POLICY", "") or "").strip().lower()
    if not value:
        return DEFAULT_POLICY
    if value not in CERTIFIABLE:
        raise CampaignPolicyRefused(
            f"MEEP_GPU_SUBNORMAL_POLICY={value!r}: a composition gate certifies one of "
            f"{list(CERTIFIABLE)} and installs it on every executor; {value!r} names no "
            f"policy a record can claim")
    return value


def attainment_reasons(stamp: Dict[str, Any], policy: str) -> List[str]:
    """Why ``stamp`` does not show ``policy`` attained on every executor. Empty when it does."""
    reasons: List[str] = []
    if not stamp.get("installed"):
        reasons.append(f"no subnormal policy is installed (stamp policy "
                       f"{stamp.get('policy')!r})")
    if stamp.get("resolved") != policy:
        reasons.append(f"the installed policy resolved to {stamp.get('resolved')!r}, "
                       f"not the requested {policy!r}")
    executors = stamp.get("executors") or {}
    for name in EXECUTORS:
        report = executors.get(name)
        if not isinstance(report, dict):
            reasons.append(f"the {name!r} executor was never driven to {policy!r}")
        elif not report.get("attained"):
            reasons.append(f"the {name!r} executor did not attain {policy!r}: "
                           + "; ".join(report.get("reasons") or ["(no reason recorded)"]))
    for name in stamp.get("unattained") or ():
        if name not in EXECUTORS:
            reasons.append(f"the {name!r} executor reports the policy unattained")
    return reasons


def install_campaign_policy(cupy: Any, *, import_meep: bool = False,
                            environ: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Install the campaign's policy on every executor, measure it, and return the stamp.

    ``import_meep`` imports MEEP FIRST, for a gate that lifts MEEP simulations later:
    importing MEEP flushes the host, so it must happen before a ``keep`` install rather
    than after it. Raises :class:`CampaignPolicyRefused` with the reason by name.
    """
    policy = requested_policy(environ)
    from meep_gpu import subnormal_policy  # noqa: PLC0415 - before any device compile

    if import_meep:
        try:
            import meep  # noqa: F401, PLC0415 - see the docstring
        except Exception as exc:  # noqa: BLE001 - the reason is the refusal
            raise CampaignPolicyRefused(
                f"this gate lifts MEEP simulations, so MEEP is imported before the "
                f"policy is installed, and the import failed: "
                f"{type(exc).__name__}: {exc}"[:600],
                subnormal_policy.policy_stamp()) from None
    try:
        subnormal_policy.install_subnormal_policy(policy, cupy=cupy, strict=True,
                                                  executors=EXECUTORS)
    except (subnormal_policy.SubnormalPolicyUnattainable,
            subnormal_policy.SubnormalPolicyLocked, ValueError) as exc:
        raise CampaignPolicyRefused(
            f"the {policy!r} subnormal policy was not installed: {exc}"[:1500],
            subnormal_policy.policy_stamp()) from None
    stamp = subnormal_policy.policy_stamp()
    reasons = attainment_reasons(stamp, policy)
    if reasons:
        raise CampaignPolicyRefused("; ".join(reasons), stamp)
    return stamp


def policy_held_reasons(stamp: Dict[str, Any]) -> List[str]:
    """Why the policy ``stamp`` attained no longer holds at the end of the run. Empty if it does."""
    from meep_gpu import backends, subnormal_policy  # noqa: PLC0415

    policy = stamp.get("resolved")
    reasons: List[str] = []
    if subnormal_policy.get_subnormal_policy() != policy:
        reasons.append(f"the installed policy is now "
                       f"{subnormal_policy.get_subnormal_policy()!r}, not {policy!r}")
    flushing = backends.subnormals_flushed()
    if flushing != (policy == "flush"):
        reasons.append(f"the host FPU is {'flushing' if flushing else 'keeping'} "
                       f"subnormals at the end of the run under the {policy!r} policy: "
                       f"something moved it after the install (importing MEEP flushes "
                       f"the host)")
    return reasons


def refusal(reason: str) -> Dict[str, Any]:
    """The verdict block a refused run writes: ``status`` ``refused`` and the reason."""
    return {"status": "refused", "reasons": [reason]}


def stamp_campaign_policy(payload: Dict[str, Any], cupy: Any, *, verdict: str = "summary",
                          import_meep: bool = False) -> Optional[str]:
    """Install the campaign's policy into ``payload``; the refusal reason, or ``None``.

    On success ``payload[STAMP_FIELD]`` is the attained stamp. On a refusal it is the
    stamp as it stood (no policy installed, or the executor that failed), and
    ``payload[verdict]`` is :func:`refusal`, so the artifact reads as not released.
    """
    try:
        payload[STAMP_FIELD] = install_campaign_policy(cupy, import_meep=import_meep)
    except CampaignPolicyRefused as exc:
        payload[STAMP_FIELD] = exc.stamp
        payload[verdict] = refusal(str(exc))
        return str(exc)
    return None


def end_of_run_reasons(payload: Dict[str, Any]) -> List[str]:
    """At the end of the run: re-stamp ``payload``, and say why the policy no longer holds.

    The stamp taken at the install carries zero compile counters (nothing has compiled
    yet), so it is REPLACED by the stamp as it stands now: the same policy, plus the
    NVRTC strip and Triton injection and audit counters the run accumulated. Empty when
    the policy still holds on every executor and on the host FPU.
    """
    from meep_gpu import subnormal_policy  # noqa: PLC0415

    installed = payload.get(STAMP_FIELD) or {}
    policy = installed.get("resolved")
    reasons = policy_held_reasons(installed)
    fresh = subnormal_policy.policy_stamp()
    reasons += [reason for reason in attainment_reasons(fresh, policy)
                if reason not in reasons]
    payload[STAMP_FIELD] = fresh
    return reasons


def confirm_campaign_policy(payload: Dict[str, Any], *,
                            verdict: str = "summary") -> Optional[str]:
    """At the end of the run: the reason the stamped policy no longer holds, or ``None``.

    Re-stamps ``payload`` (:func:`end_of_run_reasons`); a reason also writes
    ``payload[verdict]`` = :func:`refusal`.
    """
    reasons = end_of_run_reasons(payload)
    if not reasons:
        return None
    reason = "; ".join(reasons)
    payload[verdict] = refusal(reason)
    return reason
