"""Call the SHIPPED hand-CUDA coverage predicates on one lifted (fields, pml, grid) triple.

PROMOTED 2026-08-28 out of ``results/cuda_predicate_coverage_2026-08-20_closed/`` --
the tranche whose ``coverage_report.txt`` records 759 / 759 slots on 186 / 186 rows
across 22 families with zero overlaps -- and made SELF-CONTAINED. A battery that only
exists inside the tranche it cut cannot be re-cut without copying a directory, and the
copy is where the drift starts: five tranches on 2026-08-20 alone carry five different
batteries, and ``build_cuda_fusion_matrix.py`` reads a SIXTH (``_all_families``, twelve
families short of this one). This is the standing one.

WHAT THE PROMOTION CHANGED, and nothing else changed:

* The package root is resolved BY NAME (:func:`_find_api_root`). The depth-anchored
  ``parents[4]`` was correct at ``results/<round>/`` and is wrong here, and its failure
  mode is silent -- see that function's comment.
* ``configuration`` now records ``source_field_types``, and the row records
  ``plan_step.live``. Both are ENGINE facts -- properties of the lifted script, not of
  a backend -- and ``build_cuda_fusion_matrix.py`` had to JOIN them out of the METAL
  census (``metal_coverage_tranche6_2026-08-19``) because this census never recorded
  them, raising unless that join came out total. A CUDA census that cannot answer what
  a row's sources are is not self-contained; recording them here retires the join.
* :func:`_cuda_fused_magnetic_pair_verdict` asks ``covers_fused_magnetic_pair``, which
  did not exist when the closed tranche ran. It is recorded OUTSIDE the union census
  and the docstring there says why.

THE COMPOSER COLUMN, ADDED 2026-08-28 AND ADDING NOTHING TO ANY DENOMINATOR.
``meep_gpu/cuda_kernels/arms.py`` now ships a composer, so "which arm fills this slot"
is a question with an answer instead of something a board had to reconstruct from
which predicate admits. :func:`plan_step_selected` asks it and records the selection
per slot, alongside a NAMED refusal for every slot the composer did not fill.

IT IS A COLUMN AND NOT A VERDICT. Every leg above is untouched, so the 22 families,
the 186 rows and the 759 slots are the same measurement they were; ADMISSION is what
they count and SELECTION is what the new column carries, and the two differ exactly
where a slot is co-admitted, a builder refuses, or no arm is registered at all. Where
they differ the composer's answer is the one an engine would run.

RE-CUT AGAIN 2026-08-15, AFTER THE CURL PREDICATE BECAME SUB-STEP-AWARE. The
``_constitutive`` run beside this one measured seven sub-steps on which
``covers_real_pml_curl`` refused and Triton's ``pml_curl_coverage`` admitted, and zero
the other way — six of them a dispersion clause that belongs to ``update_E`` and one a
D conductivity charged to ``step_B``. Both were artefacts of one predicate answering
for two kernels: it took no sub-step name, so it could only return the intersection of
the two sub-steps' clause sets. ``covers_real_pml_curl(fields, pml, grid, sub_step)``
now takes the name and this battery calls it ONCE PER SUB-STEP, which is the only
change to the subject side. Everything else — legs, lift machinery, denominators,
factoring — is the previous round's, so the two are commensurable to the slot.

RE-CUT 2026-08-15, AFTER THE CONSTITUTIVE PAIR LANDED. The 2026-08-15 run beside this
one had exactly ONE hand-CUDA predicate to call — ``covers_real_pml_curl`` — so its
``update_H``/``update_E`` numbers were the intersection of the CUDA curl predicate with
TRITON's constitutive predicate: a statement about what a kernel built to that clause
set WOULD reach, explicitly not a measurement of a kernel. ``cuda_kernels/coverage.py``
now ships ``covers_real_pml_constitutive`` and ``constitutive_kernels.py`` ships the two
kernels it gates, so those two sub-steps can be MEASURED on the shipped predicate.

BOTH ARE RECORDED SIDE BY SIDE AND NEITHER REPLACES THE OTHER. ``ladder`` keeps the
intersection exactly as the earlier run computed it, and ``cuda_constitutive`` adds the
shipped predicate's own verdict per side. The interesting number is where the two
DISAGREE: the projection assumed a kernel written to Triton's clause set, and the
shipped predicate deliberately differs from it on one axis (it admits a conductivity —
``fields.condfac_for`` is read in ``stepping._apply_curl`` at stepping.py:508 and
nowhere else), so a disagreement is a finding rather than a defect. Every row records
both, so which it is can be read off rather than argued.

This file is the CUDA counterpart of
``results/predicate_coverage_2026-08-14_allnine/predicate_battery.py`` and is dropped
into a byte-identical copy of that round's ``measure_predicate_coverage.py`` — same
denominators, same lift machinery, same three legs, same ``match_param_rows.py``. Only
the subject changed, so the two rounds' numbers are commensurable by construction.

WHY THIS CAN EXIST AT ALL. ``cuda_kernels/coverage.py`` is stdlib-only as of the
2026-08-15 patch round. Before it, ``covers_real_pml_curl`` lived beside
``cp.RawKernel`` objects in a module that imports ``cupy`` at scope, so it could not be
imported — let alone exercised — off-device, and every CUDA coverage number in
``the design notes (cuda-kernel-track-disposition)`` §2.1 is DERIVED from the Triton
battery under a clause-subset relation rather than measured. §5 of that document says
so in as many words and says the ladder "will need re-measuring once the predicate is
extractable". This is that measurement.

THE ONE CLAUSE FACTORED, and how it differs from the Triton round's factoring. Every
Triton predicate ACCUMULATES its refusals and returns them all, so that round could
strip the single string ``array module is 'numpy', not cupy`` out of a returned tuple
and leave every other clause exactly as written. ``covers_real_pml_curl``
SHORT-CIRCUITS: it returns ``(False, "backend is not CuPy")`` on its first line and
never evaluates clause 2. Post-hoc string filtering is therefore not available, and the
factoring has to happen at the input instead:

* :class:`_CupyNamed` wraps the lifted grid's real array module and reports
  ``__name__ == "cupy"``. EVERY other attribute is forwarded to the wrapped module
  unchanged, and :func:`shim_soundness` measures that — ``xp.float32 is numpy.float32``
  is checked and recorded per row, because the predicate's dtype clauses compare
  against ``xp.float32`` and a shim that changed it would silently widen them.
* :class:`_GridWithCupyBackend` forwards every attribute and every method to the real
  grid and overrides ``xp`` alone. :func:`shim_soundness` probes the eleven grid
  attributes and methods the predicate reads and records, per row, that the proxy and
  the real grid return equal answers.

That is the whole factoring. ``fields`` and ``pml`` are passed through untouched.

BOTH VERDICTS ARE RECORDED, never mixed: ``covered`` is what the shipped predicate
returns on this NumPy host (False on every row, always for the backend clause) and
``covered_modulo_backend`` is the verdict through the shim. The second is the coverage
number, exactly as in the Triton round.

THE LADDER LEGS ARE MEASURED INTERSECTIONS OF TWO SHIPPED PREDICATES, not a model of a
kernel that does not exist. ``update_H``, ``update_E`` and ``update_P`` have no CUDA
kernel and no CUDA predicate — that is the finding, not an omission — so "what building
the real-storage constitutive pair would buy" is reported as the set of rows where the
shipped CUDA curl predicate admits AND the shipped Triton predicate for that sub-step
admits. Each leg names the two predicates it intersected. Nothing here re-derives a
clause, restates a refusal or models a kernel.

THE CROSS-CHECK LEG. Triton's ``pml_curl_coverage`` is called on the same objects at the
same two sub-steps, so the clause-subset relation §2.1 assumed can be tested directly
rather than assumed: a row the CUDA predicate admits and the Triton one refuses (or the
reverse) is a measured disagreement with the derivation.

THE SPECIALIZATION CENSUS. §2.3 of the disposition argues the CUDA cost driver is
Triton's ``tl.constexpr`` expansion and cites the static counts (12 of 39 bodies carry
``EXPANSION``; ``pml_curl_step`` is 16 variants from one source) — but nobody has
measured how many of those variants a real corpus actually *drives*. A specialization
axis whose domain is a singleton over 186 rows costs a hand author one kernel, not two.
So :func:`specialization_tuple` records, per row, the exact constexpr tuple three
families' planners would request. The values are not modelled: two come from calling
the shipped helper (``offdiag_update_e.row_volumes_for``, ``.wall_mask_axes``) and the
rest are the planner's own one-line expressions, transcribed with the line they are
transcribed from. The distinct-tuple count over the corpus is then a measurement.
"""

from __future__ import annotations

import pathlib
import re

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


#: ``the repository root``, resolved BY NAME and never by depth. Anchored on ``__file__``
#: rather than on the cwd because every row is measured in a CHILD process this
#: file does not choose the working directory of -- and anchored by name because
#: this file has now MOVED: it was written at ``results/<round>/`` where the root
#: sat four parents up, and it lives two parents up from here. A hard-coded
#: ``parents[4]`` followed it silently and resolved to an ancestor of the repo, so
#: every probe record read below missed, every child died on the resulting raise,
#: and the parent still wrote a full-length jsonl of ``measured: false`` rows that
#: reads like a census. Refuse to run rather than resolve to the wrong root. This
#: is the same helper, and the same reasoning, as
#: ``measure_predicate_coverage.py:50``.
def _find_api_root(start: pathlib.Path) -> pathlib.Path:
    for candidate in (start, *start.parents):
        if (candidate / "parity" / "meep_gpu").is_dir() and (candidate / "meep_gpu").is_dir():
            return candidate
    raise SystemExit(
        f"cannot locate the repository root above {start}: no ancestor holds "
        "both parity/meep_gpu and meep_gpu. Refusing to guess -- a wrong root makes "
        "every child die on import while still producing a plausible-looking census.")


_API_ROOT = _find_api_root(pathlib.Path(__file__).resolve().parent)

#: The kernels package these predicates measure. ``measure_predicate_coverage`` digests
#: it (plus the shared engine files) into every census row's ``subject_manifest_sha256``.
#: Declared here rather than inferred from imports: this battery imports all three
#: kernels packages because it compares against the other two, and measures one.
SUBJECT_PACKAGE = "cuda_kernels"

#: The one clause a NumPy laptop cannot satisfy, in the Triton predicates' spelling.
#: Used only on the Triton legs, where refusals arrive as an accumulated tuple.
BACKEND_CLAUSE_HEAD = "array module is "
BACKEND_CLAUSE_TAIL = ", not cupy"

#: The hand-CUDA predicate's spelling of the same clause, read from
#: ``cuda_kernels/coverage.py:82``. Recorded so a row can assert the unfactored verdict
#: really was refused for the backend and nothing else.
CUDA_BACKEND_CLAUSE = "backend is not CuPy"

#: The in-seam predicates' spelling of the same clause, read from
#: ``cuda_kernels/in_seam_coverage.py`` (``_shared_reasons``): ``backend is 'numpy',
#: not cupy; these are CUDA kernels``. It reaches a slot through the mirror-fill arm
#: (``arms.covers_mirror_fill``), which conjoins two of those predicates. It carries a
#: "; " of its own and the in-seam predicates ACCUMULATE their clauses with "; ", so
#: it is matched WHOLE -- the refusal is the backend clause alone only when nothing
#: else is joined to it -- rather than by suffix.
IN_SEAM_BACKEND_CLAUSE = re.compile(
    r"backend is '[^']*', not cupy; these are CUDA kernels")


def is_backend_clause_alone(reason: str) -> bool:
    """Is this arm refusal the backend clause and nothing else, in either spelling?

    ``reason`` is a composer refusal as ``plan.reasons`` carries it, prefixed with the
    refusing arm's ``"<label>: "``. The hand-CUDA predicates return their FIRST
    refusal, so a suffix match is exact for them; the in-seam spelling is matched
    whole against what follows the prefix.
    """
    text = str(reason)
    if text.endswith(CUDA_BACKEND_CLAUSE):
        return True
    _prefix, colon, body = text.partition(": ")
    return bool(IN_SEAM_BACKEND_CLAUSE.fullmatch(body if colon else text))

#: The two sub-steps the certified CUDA pair serves — and, since 2026-08-15, the
#: argument ``covers_real_pml_curl`` takes. The kernels differ AND the admission
#: clause set differs: ``_apply_curl`` reads ``fields.condfac_for(term.target)`` per
#: term (stepping.py:508), so a sigma reaches one curl and not the other.
CUDA_CURL_SUB_STEPS: Tuple[str, ...] = ("step_B", "step_D")

#: Which curl each ladder rung's sub-step is paired with, for the one leg that is
#: still a projection. MEEP closes B with H and D with E (``step.cpp:67-72``,
#: :95-103), so ``update_H`` pairs with ``step_B`` and ``update_E`` with ``step_D``.
#: The earlier rounds used ONE curl verdict for every rung because there was only
#: one; both curls' verdicts are recorded per rung as well, so the effect of the
#: choice can be read off rather than assumed away.
LADDER_CURL_PAIRING = {"update_H": "step_B", "update_E": "step_D"}

#: Every attribute and method of ``grid`` that ``covers_real_pml_curl`` and
#: ``real_pml_boundary_kinds`` read, as read out of the shipped source. The proxy is
#: probed on exactly these per row.
GRID_READS: Tuple[Tuple[str, Tuple[Any, ...]], ...] = (
    ("has_symmetry", ()), ("cylindrical", None), ("shape", None),
    ("has_bloch", None), ("bfast_active", None), ("beta", None),
    ("is_mirrored", (0,)), ("is_mirrored", (1,)), ("is_mirrored", (2,)),
    ("is_axis", (0,)), ("is_axis", (1,)), ("is_axis", (2,)),
    ("is_metallic", (0,)), ("is_metallic", (1,)), ("is_metallic", (2,)),
)


class _CupyNamed:
    """The lifted grid's real array module, reporting ``cupy`` as its name.

    Nothing else is overridden. ``float32`` in particular is the wrapped module's own
    object, which is what keeps the predicate's dtype clauses meaning what they mean.
    """

    def __init__(self, real: Any) -> None:
        object.__setattr__(self, "_real", real)

    __name__ = "cupy"

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_real"), name)


class _GridWithCupyBackend:
    """The lifted grid with ``xp`` replaced by :class:`_CupyNamed` and nothing else."""

    def __init__(self, grid: Any) -> None:
        object.__setattr__(self, "_grid", grid)
        object.__setattr__(self, "xp", _CupyNamed(getattr(grid, "xp", None)))

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_grid"), name)


def shim_soundness(grid: Any, proxy: Any) -> Dict[str, Any]:
    """Measure that the proxy changed the backend NAME and nothing else.

    Recorded per row rather than asserted once, because a proxy that silently answered
    a grid question differently would move the coverage number without any refusal
    saying so.
    """
    real_xp = getattr(grid, "xp", None)
    record: Dict[str, Any] = {
        "real_xp_name": getattr(real_xp, "__name__", None),
        "proxy_xp_name": getattr(proxy.xp, "__name__", None),
        "float32_is_identical": proxy.xp.float32 is getattr(real_xp, "float32", object()),
        "disagreements": [],
    }
    for name, arguments in GRID_READS:
        def read(target: Any) -> Any:
            attribute = getattr(target, name, "<<absent>>")
            if arguments is None or not callable(attribute):
                return attribute
            return attribute(*arguments)
        try:
            mine, theirs = read(proxy), read(grid)
            agree = bool(mine == theirs)
        except BaseException as exc:  # noqa: BLE001
            mine, theirs, agree = f"RAISED {type(exc).__name__}", None, False
        if not agree:
            record["disagreements"].append(
                {"read": f"{name}{arguments if arguments else ''}",
                 "proxy": repr(mine)[:120], "grid": repr(theirs)[:120]})
    record["sound"] = (record["proxy_xp_name"] == "cupy"
                       and record["real_xp_name"] != "cupy"
                       and record["float32_is_identical"]
                       and not record["disagreements"])
    return record


def is_backend_clause(reason: str) -> bool:
    return BACKEND_CLAUSE_HEAD in reason and reason.endswith(BACKEND_CLAUSE_TAIL)


def strip_backend(reasons: Tuple[str, ...]) -> List[str]:
    """A Triton refusal list with ONLY the CuPy-backend clause removed."""
    return [reason for reason in reasons if not is_backend_clause(reason)]


def _triton_verdict(call: Callable[[], Any]) -> Dict[str, Any]:
    """One Triton predicate; a raise is a refusal WITH ITS REASON, never a crash."""
    try:
        coverage = call()
    except BaseException as exc:  # noqa: BLE001
        return {"covered": False, "covered_modulo_backend": False,
                "reasons": [f"PREDICATE RAISED {type(exc).__name__}: {exc}"[:400]],
                "residual_reasons": [f"PREDICATE RAISED {type(exc).__name__}"],
                "raised": True}
    reasons = tuple(getattr(coverage, "reasons", ()) or ())
    residual = strip_backend(reasons)
    return {"covered": bool(getattr(coverage, "covered", False)),
            "covered_modulo_backend": not residual,
            "reasons": [str(reason)[:400] for reason in reasons],
            "residual_reasons": [str(reason)[:400] for reason in residual],
            "raised": False}


def _cuda_verdict(fields: Any, pml: Any, grid: Any, sub_step: str) -> Dict[str, Any]:
    """The shipped hand-CUDA curl predicate for ONE sub-step, both verdicts recorded.

    Called once per sub-step since 2026-08-15. The previous rounds called it once and
    copied the answer to both, which is what the predicate itself was doing
    internally; the seven-sub-step gap against Triton was the cost of that.
    """
    from meep_gpu.cuda_kernels.coverage import covers_real_pml_curl  # noqa: PLC0415

    def once(target_grid: Any) -> Dict[str, Any]:
        try:
            covered, reason = covers_real_pml_curl(fields, pml, target_grid, sub_step)
        except BaseException as exc:  # noqa: BLE001
            return {"covered": False,
                    "reason": f"PREDICATE RAISED {type(exc).__name__}: {exc}"[:400],
                    "raised": True}
        return {"covered": bool(covered), "reason": str(reason)[:400], "raised": False}

    proxy = _GridWithCupyBackend(grid)
    raw = once(grid)
    factored = once(proxy)
    return {
        "covered": raw["covered"],
        "raw_reason": raw["reason"],
        "raw_refusal_is_backend_clause": raw["reason"] == CUDA_BACKEND_CLAUSE,
        "covered_modulo_backend": factored["covered"],
        # The SHORT-CIRCUIT answer: the first clause the predicate refused on with the
        # backend clause satisfied. Not a residual set — this predicate does not build
        # one — so the ranking downstream counts dominant blockers, and says so.
        "first_refusal": None if factored["covered"] else factored["reason"],
        "raised": raw["raised"] or factored["raised"],
        "shim": shim_soundness(grid, proxy),
    }


def _cuda_constitutive_verdict(fields: Any, pml: Any, grid: Any,
                               side: str) -> Dict[str, Any]:
    """The SHIPPED hand-CUDA constitutive predicate for one side, both verdicts.

    Same input factoring as :func:`_cuda_verdict` and for the same reason: this
    predicate short-circuits on the backend clause too, so the CuPy name has to be
    supplied at the input rather than filtered out of an accumulated refusal list.
    The proxy's soundness is re-measured here rather than borrowed, because this
    predicate reads a slightly different set of grid questions.
    """
    from meep_gpu.cuda_kernels.coverage import (  # noqa: PLC0415
        covers_real_pml_constitutive,
    )

    def once(target_grid: Any) -> Dict[str, Any]:
        try:
            covered, reason = covers_real_pml_constitutive(
                fields, pml, target_grid, side)
        except BaseException as exc:  # noqa: BLE001
            return {"covered": False,
                    "reason": f"PREDICATE RAISED {type(exc).__name__}: {exc}"[:400],
                    "raised": True}
        return {"covered": bool(covered), "reason": str(reason)[:400], "raised": False}

    proxy = _GridWithCupyBackend(grid)
    raw = once(grid)
    factored = once(proxy)
    return {
        "side": side,
        "kernel": f"fused_update_{side}_pml_real",
        "predicate": "cuda_kernels.coverage.covers_real_pml_constitutive",
        "covered": raw["covered"],
        "raw_reason": raw["reason"],
        "raw_refusal_is_backend_clause": raw["reason"] == CUDA_BACKEND_CLAUSE,
        "covered_modulo_backend": factored["covered"],
        "first_refusal": None if factored["covered"] else factored["reason"],
        "raised": raw["raised"] or factored["raised"],
        "shim": shim_soundness(grid, proxy),
    }


# ---------------------------------------------------------------------------
# THE FIVE FAMILIES THE 2026-08-20 RE-CENSUS DID NOT ASK ABOUT
#
# MEASURED 2026-08-20: the battery asked three predicates -- the real curl, the
# real constitutive pair and the real off-diagonal update_E -- while
# cuda_kernels/ had by then grown complex_pml_kernels.py, cylindrical_kernels.py,
# no_pml_constitutive.py and ade_kernels.py, each with its own shipped predicate.
# Every row those families serve was therefore counted as REFUSED, and "complex64
# storage" ranked as the dominant blocker on 47 rows that a shipped CUDA predicate
# may well admit. A census that does not ask a predicate cannot report what it
# covers; this closes that, family by family, with no change to the three legs
# above so the two rounds stay commensurable to the slot.
# ---------------------------------------------------------------------------

#: The complex arms are compiled at the expansion seam, so their predicates take a
#: LICENSE and a POLICY -- and neither is a property of the census host. The
#: question this census asks is "would the SHIPPED predicate admit this row on the
#: target the complex family was certified on", so both are supplied from that
#: target's own record: the expansion probe cut on the GPU host's RTX A6000 under the
#: 'keep' policy, which classifies the arm FMA_V1 on a measured basis with no
#: refusals. Passing a placeholder instead was measured on 2026-08-20 to refuse
#: every row with "the expansion licence is str, not the verdict dict" -- a
#: refusal about the census, reported as though it were about the corpus.
COMPLEX_PROBE_RECORD = ("parity/meep_gpu/results/expansion_probe_2026-08-17/"
                        "expansion_probe_keep.json")
COMPLEX_POLICY = "keep"


def _complex_license() -> Any:
    """The expansion verdict the complex predicates gate on, or the failure reason.

    Loaded ONCE per child process. A record that does not license an arm is not
    silently downgraded to a placeholder: the reason travels into every row, so a
    census run against a bad record reads as a census failure rather than as 186
    rows of coverage gap.
    """
    import json as _json  # noqa: PLC0415
    from meep_gpu.triton_kernels import complex_fields  # noqa: PLC0415

    path = _API_ROOT / COMPLEX_PROBE_RECORD
    record = _json.loads(path.read_text(encoding="utf-8"))
    verdict = complex_fields.expansion_license(record)
    policy_reasons = list(
        complex_fields.expansion_policy_reasons(record, COMPLEX_POLICY))
    if not verdict.get("arm") or verdict.get("refusals") or policy_reasons:
        raise RuntimeError(
            f"{COMPLEX_PROBE_RECORD} licenses no arm under {COMPLEX_POLICY!r}: "
            f"arm={verdict.get('arm')!r} refusals={verdict.get('refusals')} "
            f"policy_reasons={policy_reasons[:1]}")
    return verdict


def _one(call: Callable[[], Any]) -> Dict[str, Any]:
    """One ``(covered, reason)`` predicate; a raise is a refusal WITH its reason."""
    try:
        covered, reason = call()
    except BaseException as exc:  # noqa: BLE001
        return {"covered": False,
                "reason": f"PREDICATE RAISED {type(exc).__name__}: {exc}"[:400],
                "raised": True}
    return {"covered": bool(covered), "reason": str(reason)[:400], "raised": False}


def _factored(fields: Any, pml: Any, grid: Any,
              call: Callable[[Any], Any]) -> Dict[str, Any]:
    """The two-verdict shape every CUDA leg in this file records.

    ``call`` takes the grid to ask with, so the CuPy-backend clause is satisfied at
    the INPUT rather than filtered out of a refusal string -- these predicates
    short-circuit, so there is no accumulated list to filter.
    """
    proxy = _GridWithCupyBackend(grid)
    raw = _one(lambda: call(grid))
    factored = _one(lambda: call(proxy))
    return {"covered": raw["covered"],
            "raw_reason": raw["reason"],
            "raw_refusal_is_backend_clause": raw["reason"] == CUDA_BACKEND_CLAUSE,
            "covered_modulo_backend": factored["covered"],
            "first_refusal": None if factored["covered"] else factored["reason"],
            "raised": raw["raised"] or factored["raised"],
            "shim": shim_soundness(grid, proxy)}


def _cuda_complex_verdicts(fields: Any, pml: Any, grid: Any) -> Dict[str, Any]:
    """``covers_real_pml_complex_curl`` per sub-step and ``..._constitutive`` per side."""
    from meep_gpu.cuda_kernels.coverage import (  # noqa: PLC0415
        covers_real_pml_complex_constitutive,
        covers_real_pml_complex_curl,
    )
    licence = _complex_license()
    out: Dict[str, Any] = {}
    for name in CUDA_CURL_SUB_STEPS:
        out[name] = dict(_factored(
            fields, pml, grid,
            lambda g, n=name: covers_real_pml_complex_curl(
                fields, pml, g, n, license=licence, subnormal_policy=COMPLEX_POLICY)),
            sub_step=name, kernel=f"fused_{name}_pml_complex_bloch",
            predicate="cuda_kernels.coverage.covers_real_pml_complex_curl",
            family="cuda_complex")
    for side in ("H", "E"):
        out[f"update_{side}"] = dict(_factored(
            fields, pml, grid,
            lambda g, s=side: covers_real_pml_complex_constitutive(
                fields, pml, g, s, license=licence, subnormal_policy=COMPLEX_POLICY)),
            sub_step=f"update_{side}", kernel=f"fused_update_{side}_pml_complex_bloch",
            predicate="cuda_kernels.coverage.covers_real_pml_complex_constitutive",
            family="cuda_complex")
    return out


def _cuda_cylindrical_verdicts(fields: Any, pml: Any, grid: Any) -> Dict[str, Any]:
    """``covers_real_pml_cylindrical_curl`` per sub-step."""
    from meep_gpu.cuda_kernels.cylindrical_coverage import (  # noqa: PLC0415
        covers_real_pml_cylindrical_curl,
    )
    return {name: dict(_factored(
        fields, pml, grid,
        lambda g, n=name: covers_real_pml_cylindrical_curl(fields, pml, g, n)),
        sub_step=name, kernel=f"fused_{name}_pml_cylindrical",
        predicate="cuda_kernels.cylindrical_coverage.covers_real_pml_cylindrical_curl",
        family="cuda_cylindrical")
        for name in CUDA_CURL_SUB_STEPS}


def _cuda_no_pml_verdicts(fields: Any, pml: Any, grid: Any) -> Dict[str, Any]:
    """``covers_no_pml_null_constitutive`` per side -- the no-absorber null pair."""
    from meep_gpu.cuda_kernels.no_pml_constitutive import (  # noqa: PLC0415
        covers_no_pml_null_constitutive,
    )
    return {f"update_{side}": dict(_factored(
        fields, pml, grid,
        lambda g, s=side: covers_no_pml_null_constitutive(fields, pml, g, s)),
        sub_step=f"update_{side}", kernel=f"fused_update_{side}_no_pml_null",
        predicate="cuda_kernels.no_pml_constitutive.covers_no_pml_null_constitutive",
        family="cuda_no_pml")
        for side in ("H", "E")}


def _cuda_ade_verdicts(fields: Any, pml: Any, grid: Any) -> Dict[str, Any]:
    """``covers_real_pml_ade_update_p`` -- the whole-sub-step ADE verdict.

    One verdict for the slot, not one per driven component: update_P is ONE slot in
    the 759-slot denominator, and a kernel that serves the sub-step has to serve
    every component the row drives.
    """
    from meep_gpu.cuda_kernels.coverage import (  # noqa: PLC0415
        covers_real_pml_ade_update_p,
    )
    return {"update_P": dict(_factored(
        fields, pml, grid,
        lambda g: covers_real_pml_ade_update_p(fields, pml, g)),
        sub_step="update_P", kernel="fused_update_P_pml_real",
        predicate="cuda_kernels.coverage.covers_real_pml_ade_update_p",
        family="cuda_ade")}


# ---------------------------------------------------------------------------
# THE FAMILY THE CLOSEOUT ROUND HAD NO PREDICATE FOR: COMPLEX STORAGE WITH NO
# ABSORBER.
#
# The closeout census left 22 unserved slots on six rows whose union refusal was
# "no active PML layer" (cuda_complex) and "Bx/Dx is complex64, not float32"
# (cuda_no_pml_curl). ``cuda_kernels/complex_no_pml_kernels.py`` now ships six
# kernels and three predicates for twenty of those slots; the other two are the
# off-diagonal complex update_E, which that file refuses BY NAME rather than
# leaving unclaimed.
#
# THIS LEG BINDS A DIFFERENT PROBE RECORD FROM THE OTHER COMPLEX LEGS, and that
# is a property of the sub-step rather than a convenience. ``update_P``'s first
# term is ``xp.multiply(P, c_now)`` -- a complex64 ARRAY times a PYTHON FLOAT --
# and neither ``c8_mul_f4_field_left`` (array coefficient) nor
# ``python_float_left`` (scalar on the left) is that orientation. The
# four-pattern ``expansion_probe_2026-08-17`` record the Cartesian complex legs
# bind does not classify it; the SEVEN-pattern keep-policy record cut by
# ``gate_triton_unified_expansion`` on 2026-08-19 does. Binding the four-pattern
# record here would measure a refusal about the ARTIFACT and report it as a
# refusal about the corpus -- the exact failure this file's own
# COMPLEX_PROBE_RECORD comment records from 2026-08-20.
# ---------------------------------------------------------------------------

COMPLEX_NO_PML_PROBE_RECORD = ("parity/meep_gpu/results/triton_welds_2026-08-19/"
                               "unified_expansion/gate.json")


def _complex_no_pml_license() -> Any:
    """The seven-pattern verdict this family's three predicates gate on.

    ``expansion_license`` computes the ARM from the record; the PATTERN TABLE it
    was computed from is carried alongside, because ``update_P``'s probe clause
    asks a question about the record and not only about the verdict. A record
    that does not license an arm is not silently downgraded: the reason travels
    into every row, so a census run against a bad record reads as a census
    failure rather than as 186 rows of coverage gap.
    """
    import json as _json  # noqa: PLC0415
    from meep_gpu.triton_kernels import complex_fields  # noqa: PLC0415

    path = _API_ROOT / COMPLEX_NO_PML_PROBE_RECORD
    record = _json.loads(path.read_text(encoding="utf-8"))
    verdict = complex_fields.expansion_license(record)
    verdict.setdefault("patterns", record.get("patterns"))
    policy_reasons = list(
        complex_fields.expansion_policy_reasons(record, COMPLEX_POLICY))
    if not verdict.get("arm") or verdict.get("refusals") or policy_reasons:
        raise RuntimeError(
            f"{COMPLEX_NO_PML_PROBE_RECORD} licenses no arm under "
            f"{COMPLEX_POLICY!r}: arm={verdict.get('arm')!r} "
            f"refusals={verdict.get('refusals')} "
            f"policy_reasons={policy_reasons[:1]}")
    return verdict


def _cuda_complex_no_pml_verdicts(fields: Any, pml: Any,
                                  grid: Any) -> Dict[str, Any]:
    """The three complex/no-absorber predicates, one entry per sub-step slot.

    ``update_H`` IS DELIBERATELY ABSENT. Under an inert layer ``stepping.update_H``
    returns at :916-917 before reading an array whatever the storage width is, so
    that slot is ``no_pml_constitutive``'s NULL arm and is already served -- the
    closeout census attributes it to ``cuda_no_pml`` on exactly these six rows. A
    second admitter there would be a widening, not extra coverage.
    """
    from meep_gpu.cuda_kernels import complex_no_pml_kernels as arm  # noqa: PLC0415

    licence = _complex_no_pml_license()
    out: Dict[str, Any] = {}
    for name in CUDA_CURL_SUB_STEPS:
        out[name] = dict(_factored(
            fields, pml, grid,
            lambda g, n=name: arm.covers_complex_no_pml_curl(
                fields, pml, g, n, license=licence,
                subnormal_policy=COMPLEX_POLICY)),
            sub_step=name, kernel=f"fused_{name}_no_pml_complex[_conductive]",
            predicate=("cuda_kernels.complex_no_pml_kernels."
                       "covers_complex_no_pml_curl"),
            family="cuda_complex_no_pml")
        try:
            out[name]["arm"] = arm.complex_no_pml_curl_arm(fields, name)
        except BaseException as exc:  # noqa: BLE001
            out[name]["arm"] = f"UNCLASSIFIED {type(exc).__name__}"[:120]
    out["update_E"] = dict(_factored(
        fields, pml, grid,
        lambda g: arm.covers_complex_no_pml_stored_e(
            fields, pml, g, license=licence, subnormal_policy=COMPLEX_POLICY)),
        sub_step="update_E", kernel="fused_update_E_no_pml_complex_stored",
        predicate=("cuda_kernels.complex_no_pml_kernels."
                   "covers_complex_no_pml_stored_e"),
        family="cuda_complex_no_pml")
    out["update_P"] = dict(_factored(
        fields, pml, grid,
        lambda g: arm.covers_complex_no_pml_ade_update_p(
            fields, pml, g, license=licence, subnormal_policy=COMPLEX_POLICY)),
        sub_step="update_P", kernel="fused_update_P_no_pml_complex[_uniform]",
        predicate=("cuda_kernels.complex_no_pml_kernels."
                   "covers_complex_no_pml_ade_update_p"),
        family="cuda_complex_no_pml")
    return out


def _cuda_dispersive_verdicts(fields: Any, pml: Any, grid: Any) -> Dict[str, Any]:
    """``covers_real_pml_dispersive_constitutive`` -- arm 1, update_E under a layer.

    NEW THIS ROUND, and the slot it answers for was UNSERVED rather than refused
    by a family that had been asked: the closeout census attributes 8 update_E
    slots to ``cuda_constitutive``'s "dispersion: update_E's source is
    (D - sum P), not D" and no other arm had a clause to offer. A census that
    does not ASK a predicate cannot report what it covers, which is the failure
    that cost this project two rounds -- so the leg is wired in the same change
    as the predicate.
    """
    from meep_gpu.cuda_kernels.dispersive_kernels import (  # noqa: PLC0415
        covers_real_pml_dispersive_constitutive,
    )
    return {"update_E": dict(_factored(
        fields, pml, grid,
        lambda g: covers_real_pml_dispersive_constitutive(fields, pml, g)),
        sub_step="update_E", kernel="fused_update_E_pml_real_dispersive",
        predicate=("cuda_kernels.dispersive_kernels."
                   "covers_real_pml_dispersive_constitutive"),
        family="cuda_dispersive")}


def _cuda_no_pml_dispersive_verdicts(fields: Any, pml: Any,
                                     grid: Any) -> Dict[str, Any]:
    """``covers_no_pml_dispersive_constitutive`` -- arm 2, the STORE at stepping.py:1022.

    A DIFFERENT SUB-STEP from arm 1, not a special case of it: no auxiliary, no
    coefficient vector, no history. The closeout census attributes its three
    slots to ``cuda_no_pml``'s "fields.stores_E is True: update_E writes
    E[...] = (D - sum P) * inv_eps (stepping.py:1022) ... that is the STORE arm,
    which this package has not built". This is that arm.
    """
    from meep_gpu.cuda_kernels.dispersive_kernels import (  # noqa: PLC0415
        covers_no_pml_dispersive_constitutive,
    )
    return {"update_E": dict(_factored(
        fields, pml, grid,
        lambda g: covers_no_pml_dispersive_constitutive(fields, pml, g)),
        sub_step="update_E", kernel="update_E_no_pml_real_dispersive",
        predicate=("cuda_kernels.dispersive_kernels."
                   "covers_no_pml_dispersive_constitutive"),
        family="cuda_no_pml_dispersive")}


def _cuda_no_pml_ade_verdicts(fields: Any, pml: Any, grid: Any) -> Dict[str, Any]:
    """``covers_no_pml_ade_update_p`` -- arm 3, and NO NEW KERNEL behind it.

    The kernel named here is the SHIPPED ``fused_update_P_pml_real``: the ADE
    device text, the coefficient triple, the rotation and the sigma
    specialization are all unchanged, and the only thing an absorber decides is
    which array ``Fields.drive_field`` hands the launcher (fields.py:1140-1163).
    One verdict for the slot, not one per driven component -- update_P is ONE
    slot in the 759-slot denominator.
    """
    from meep_gpu.cuda_kernels.dispersive_kernels import (  # noqa: PLC0415
        covers_no_pml_ade_update_p,
    )
    return {"update_P": dict(_factored(
        fields, pml, grid,
        lambda g: covers_no_pml_ade_update_p(fields, pml, g)),
        sub_step="update_P", kernel="fused_update_P_pml_real",
        predicate=("cuda_kernels.dispersive_kernels."
                   "covers_no_pml_ade_update_p"),
        family="cuda_no_pml_ade")}


def _cuda_offdiag_verdict(fields: Any, pml: Any, grid: Any) -> Dict[str, Any]:
    """The SHIPPED hand-CUDA off-diagonal ``update_E`` predicate, both verdicts.

    NEW THIS ROUND. The previous run reported this family as a LADDER LEG — the
    intersection of the CUDA curl predicate with TRITON's
    ``offdiag_constitutive_coverage`` — because there was no CUDA kernel and no
    CUDA predicate to ask. ``cuda_kernels/coverage.py`` now ships
    ``covers_real_pml_offdiag_constitutive`` and ``offdiag_emitter.py`` emits the
    kernel it gates, so this sub-step can be MEASURED on the shipped predicate.

    Both are recorded side by side and neither replaces the other: ``ladder`` keeps
    the intersection exactly as the earlier rounds computed it, and this adds the
    shipped predicate's own verdict. Where the two disagree is the interesting
    number, and it is readable rather than argued.

    Same input factoring as the two verdict functions above, and for the same
    reason: this predicate short-circuits on the backend clause too.
    """
    from meep_gpu.cuda_kernels.coverage import (  # noqa: PLC0415
        covers_real_pml_offdiag_constitutive,
    )

    def once(target_grid: Any) -> Dict[str, Any]:
        try:
            covered, reason = covers_real_pml_offdiag_constitutive(
                fields, pml, target_grid)
        except BaseException as exc:  # noqa: BLE001
            return {"covered": False,
                    "reason": f"PREDICATE RAISED {type(exc).__name__}: {exc}"[:400],
                    "raised": True}
        return {"covered": bool(covered), "reason": str(reason)[:400], "raised": False}

    proxy = _GridWithCupyBackend(grid)
    raw = once(grid)
    factored = once(proxy)
    return {
        "sub_step": "update_E",
        "kernel": "fused_update_E_pml_real_offdiag",
        "predicate": "cuda_kernels.coverage.covers_real_pml_offdiag_constitutive",
        "covered": raw["covered"],
        "raw_reason": raw["reason"],
        "raw_refusal_is_backend_clause": raw["reason"] == CUDA_BACKEND_CLAUSE,
        "covered_modulo_backend": factored["covered"],
        "first_refusal": None if factored["covered"] else factored["reason"],
        "raised": raw["raised"] or factored["raised"],
        "shim": shim_soundness(grid, proxy),
    }


# ---------------------------------------------------------------------------
# THE SIX PREDICATES THE 2026-08-20 all_families ROUND DID NOT ASK ABOUT
#
# MEASURED 2026-08-20 (closeout): ``grep '^def covers_' meep_gpu/cuda_kernels/*.py``
# lists sixteen shipped predicates. The all_families battery called eight of them.
# The eight it did not call are (a) ``covers_real_pml_ade_component``, a per-
# component helper the whole-sub-step ``covers_real_pml_ade_update_p`` already
# folds in -- asking it separately would double-count a slot -- and (b) SIX
# family entry points with no caller in the census at all:
#
#   no_pml_curl.covers_no_pml_curl                        step_B / step_D
#   cylindrical_coverage.covers_pml_cylindrical_complex_curl  step_B / step_D
#   nonlinear_constitutive.covers_real_pml_nonlinear_constitutive  update_H / update_E
#   special_kz_curl.covers_special_kz_curl                step_B / step_D
#   special_kz_curl.covers_special_kz_constitutive        update_H / update_E
#   bfast_curl.covers_bfast_curl                          step_B / step_D
#   bfast_curl.covers_bfast_constitutive                  update_H / update_E
#
# (that is six FILES-worth of entry points, seven functions). Every row those
# serve was being counted as refused. Same two-verdict factoring as every leg
# above -- these predicates short-circuit on the backend clause too, so the CuPy
# name is supplied at the INPUT rather than filtered out of the refusal string.
# ---------------------------------------------------------------------------


def _cuda_no_pml_curl_verdicts(fields, pml, grid):
    """``covers_no_pml_curl`` per sub-step -- the no-absorber curl arms."""
    from meep_gpu.cuda_kernels.no_pml_curl import (  # noqa: PLC0415
        covers_no_pml_curl,
    )
    return {name: dict(_factored(
        fields, pml, grid,
        lambda g, n=name: covers_no_pml_curl(fields, pml, g, n)),
        sub_step=name, kernel=f"fused_{name}_no_pml_real",
        predicate="cuda_kernels.no_pml_curl.covers_no_pml_curl",
        family="cuda_no_pml_curl")
        for name in CUDA_CURL_SUB_STEPS}


def _cuda_cyl_complex_verdicts(fields, pml, grid):
    """``covers_pml_cylindrical_complex_curl`` per sub-step.

    Takes the same expansion licence and policy as the Cartesian complex arms and
    for the same reason: the arm is compiled in at this seam.
    """
    from meep_gpu.cuda_kernels.cylindrical_coverage import (  # noqa: PLC0415
        covers_pml_cylindrical_complex_curl,
    )
    licence = _complex_license()
    return {name: dict(_factored(
        fields, pml, grid,
        lambda g, n=name: covers_pml_cylindrical_complex_curl(
            fields, pml, g, n, license=licence, subnormal_policy=COMPLEX_POLICY)),
        sub_step=name, kernel=f"fused_cyl_{name}_pml_complex",
        predicate=("cuda_kernels.cylindrical_coverage."
                   "covers_pml_cylindrical_complex_curl"),
        family="cuda_cyl_complex")
        for name in CUDA_CURL_SUB_STEPS}


def _cuda_nonlinear_verdicts(fields, pml, grid):
    """``covers_real_pml_nonlinear_constitutive`` per side."""
    from meep_gpu.cuda_kernels.nonlinear_constitutive import (  # noqa: PLC0415
        covers_real_pml_nonlinear_constitutive,
    )
    return {f"update_{side}": dict(_factored(
        fields, pml, grid,
        lambda g, s=side: covers_real_pml_nonlinear_constitutive(fields, pml, g, s)),
        sub_step=f"update_{side}",
        kernel=("fused_update_E_pml_real_nonlinear" if side == "E"
                else "fused_update_H_pml_real"),
        predicate=("cuda_kernels.nonlinear_constitutive."
                   "covers_real_pml_nonlinear_constitutive"),
        family="cuda_nonlinear")
        for side in ("H", "E")}


def _cuda_special_kz_verdicts(fields, pml, grid):
    """``covers_special_kz_curl`` per sub-step and ``..._constitutive`` per side."""
    from meep_gpu.cuda_kernels.special_kz_curl import (  # noqa: PLC0415
        covers_special_kz_constitutive,
        covers_special_kz_curl,
    )
    out = {}
    for name in CUDA_CURL_SUB_STEPS:
        out[name] = dict(_factored(
            fields, pml, grid,
            lambda g, n=name: covers_special_kz_curl(fields, pml, g, n)),
            sub_step=name, kernel=f"fused_{name}_special_kz_real",
            predicate="cuda_kernels.special_kz_curl.covers_special_kz_curl",
            family="cuda_special_kz")
    for side in ("H", "E"):
        out[f"update_{side}"] = dict(_factored(
            fields, pml, grid,
            lambda g, s=side: covers_special_kz_constitutive(fields, pml, g, s)),
            sub_step=f"update_{side}", kernel=f"fused_update_{side}_pml_real",
            predicate=("cuda_kernels.special_kz_curl."
                       "covers_special_kz_constitutive"),
            family="cuda_special_kz")
    return out


def _cuda_bfast_verdicts(fields, pml, grid):
    """``covers_bfast_curl`` per sub-step and ``covers_bfast_constitutive`` per side."""
    from meep_gpu.cuda_kernels.bfast_curl import (  # noqa: PLC0415
        covers_bfast_constitutive,
        covers_bfast_curl,
    )
    out = {}
    for name in CUDA_CURL_SUB_STEPS:
        out[name] = dict(_factored(
            fields, pml, grid,
            lambda g, n=name: covers_bfast_curl(fields, pml, g, n)),
            sub_step=name, kernel=f"fused_{name}_bfast_real",
            predicate="cuda_kernels.bfast_curl.covers_bfast_curl",
            family="cuda_bfast")
    for side in ("H", "E"):
        out[f"update_{side}"] = dict(_factored(
            fields, pml, grid,
            lambda g, s=side: covers_bfast_constitutive(fields, pml, g, s)),
            sub_step=f"update_{side}", kernel=f"fused_update_{side}_pml_real",
            predicate="cuda_kernels.bfast_curl.covers_bfast_constitutive",
            family="cuda_bfast")
    return out

# ---------------------------------------------------------------------------
# THE TWO FAMILIES THE 2026-08-20 CLOSEOUT ROUND DID NOT ASK ABOUT
#
# ``complex_folded_kernels`` and ``complex_beta_kernels`` are new on 2026-08-20 and
# have no caller in the closeout battery, so every slot they serve was being
# counted as refused -- with "mirror symmetry: different ghost rule, a parity mask
# and two fill passes" ranking as the dominant blocker on 32 slots that a shipped
# CUDA predicate now admits, and "special_kz (grid.beta != 0)" on 16 more. A census
# that does not ask a predicate cannot report what it covers.
#
# BOTH TAKE THE EXPANSION LICENCE AND THE POLICY, and for the same reason the
# Cartesian complex arms do: they delegate into ``covers_real_pml_complex_curl`` /
# ``..._constitutive``, where the arm is compiled into the binary. Passing a
# placeholder instead refuses every row with a message about the census rather than
# about the corpus.
#
# THE TWO ARE DISJOINT BY CONSTRUCTION and the union census CHECKS it rather than
# assuming: the folded family inherits the certified predicate's beta refusal, and
# the beta family requires a nonzero beta. Twelve of the beta family's sixteen
# census slots are also folded, so a leak either way would show as an overlap.
# ---------------------------------------------------------------------------


def _cuda_complex_folded_verdicts(fields, pml, grid):
    """``covers_complex_folded_curl`` per sub-step and ``..._constitutive`` per side."""
    from meep_gpu.cuda_kernels.complex_folded_kernels import (  # noqa: PLC0415
        covers_complex_folded_constitutive,
        covers_complex_folded_curl,
    )
    licence = _complex_license()
    out = {}
    for name in CUDA_CURL_SUB_STEPS:
        out[name] = dict(_factored(
            fields, pml, grid,
            lambda g, n=name: covers_complex_folded_curl(
                fields, pml, g, n, license=licence,
                subnormal_policy=COMPLEX_POLICY)),
            sub_step=name, kernel=f"fused_{name}_pml_complex_folded",
            predicate=("cuda_kernels.complex_folded_kernels."
                       "covers_complex_folded_curl"),
            family="cuda_complex_folded")
    for side in ("H", "E"):
        out[f"update_{side}"] = dict(_factored(
            fields, pml, grid,
            lambda g, s=side: covers_complex_folded_constitutive(
                fields, pml, g, s, license=licence,
                subnormal_policy=COMPLEX_POLICY)),
            sub_step=f"update_{side}",
            kernel=f"fused_update_{side}_pml_complex_bloch",
            predicate=("cuda_kernels.complex_folded_kernels."
                       "covers_complex_folded_constitutive"),
            family="cuda_complex_folded")
    return out


def _cuda_complex_beta_verdicts(fields, pml, grid):
    """``covers_complex_beta_curl`` per sub-step and ``..._constitutive`` per side."""
    from meep_gpu.cuda_kernels.complex_beta_kernels import (  # noqa: PLC0415
        covers_complex_beta_constitutive,
        covers_complex_beta_curl,
    )
    licence = _complex_license()
    out = {}
    for name in CUDA_CURL_SUB_STEPS:
        out[name] = dict(_factored(
            fields, pml, grid,
            lambda g, n=name: covers_complex_beta_curl(
                fields, pml, g, n, license=licence,
                subnormal_policy=COMPLEX_POLICY)),
            sub_step=name, kernel=f"fused_{name}_pml_complex_beta",
            predicate=("cuda_kernels.complex_beta_kernels."
                       "covers_complex_beta_curl"),
            family="cuda_complex_beta")
    for side in ("H", "E"):
        out[f"update_{side}"] = dict(_factored(
            fields, pml, grid,
            lambda g, s=side: covers_complex_beta_constitutive(
                fields, pml, g, s, license=licence,
                subnormal_policy=COMPLEX_POLICY)),
            sub_step=f"update_{side}",
            kernel=f"fused_update_{side}_pml_complex_bloch",
            predicate=("cuda_kernels.complex_beta_kernels."
                       "covers_complex_beta_constitutive"),
            family="cuda_complex_beta")
    return out



# ---------------------------------------------------------------------------
# THE FAMILY THE 2026-08-20 CLOSEOUT ROUND COULD NOT ASK ABOUT
#
# ``cuda_kernels/conductive_kernels.py`` did not exist when that round ran. Its
# predicate is the exact per-sub-step INVERSE of the conductivity clause the
# certified curl predicate (coverage.py:972-980) and ``covers_no_pml_curl`` (which
# delegates to it) both refuse on, so the seven slots the closeout report ranks
# under "Bx/Dx carries a conductivity: routes to the three-history conductive-PML
# recurrence" are exactly what this leg asks about. Same two-verdict factoring as
# every CUDA leg above -- the predicate short-circuits on the backend clause too.
# ---------------------------------------------------------------------------


def _cuda_conductive_verdicts(fields, pml, grid):
    """``covers_conductive_curl`` per sub-step -- a sigma on a curl target."""
    from meep_gpu.cuda_kernels.conductive_kernels import (  # noqa: PLC0415
        covers_conductive_curl,
    )
    return {name: dict(_factored(
        fields, pml, grid,
        lambda g, n=name: covers_conductive_curl(fields, pml, g, n)),
        sub_step=name, kernel=f"fused_{name}_(no_)pml_conductive",
        predicate="cuda_kernels.conductive_kernels.covers_conductive_curl",
        family="cuda_conductive")
        for name in CUDA_CURL_SUB_STEPS}


# ---------------------------------------------------------------------------
# THE FAMILY THIS ROUND ADDS: the FOLDED off-diagonal update_E kernel
#
# MEASURED 2026-08-20 (closeout): update_E carried 70 slots no family served, 26
# of them refused FIRST by ``covers_real_pml_offdiag_constitutive``'s mirror
# clause, and 20 of those needed "a MIRROR branch in the emitted coord_dn ... a
# new kernel variant with its own byte gate"
# (``coverage.FOLDED_OFFDIAG_ROW_MASK_ADMISSION["what_would_move_the_twenty"]``).
# ``cuda_kernels/folded_offdiag_kernels.py`` now emits that variant and ships its
# own predicate, so those slots can be MEASURED rather than counted as refused.
#
# THE CENSUS ASKS THE **COMPOSITION** VERDICT, never the standalone one. The
# standalone predicate deliberately admits an UNFOLDED grid so its gate can
# measure reduction to the certified family; asking it here would claim the
# certified family's 16 slots a second time and the analyzer would report an
# overlap that is an artifact of the question. The composition verdict requires a
# real fold, which is exactly the complement of the certified predicate's two
# fold refusals -- so the two partition the corpus by construction and the
# analyzer's disjointness check measures that rather than assuming it.
# ---------------------------------------------------------------------------


def _cuda_folded_offdiag_verdict(fields: Any, pml: Any, grid: Any) -> Dict[str, Any]:
    """``covers_folded_offdiag_composition``, both verdicts, same factoring.

    Same input factoring as every CUDA leg in this file, and for the same reason:
    this predicate short-circuits on the backend clause too, so the CuPy name is
    supplied at the INPUT rather than filtered out of the refusal string.
    """
    from meep_gpu.cuda_kernels.folded_offdiag_kernels import (  # noqa: PLC0415
        covers_folded_offdiag_composition,
        covers_folded_offdiag_constitutive,
    )
    record = dict(_factored(
        fields, pml, grid,
        lambda g: covers_folded_offdiag_composition(fields, pml, g)),
        sub_step="update_E",
        kernel="fused_update_E_pml_real_folded_offdiag",
        predicate=("cuda_kernels.folded_offdiag_kernels."
                   "covers_folded_offdiag_composition"),
        family="cuda_folded_offdiag")
    # The STANDALONE verdict is recorded beside it and never counted: it is what
    # the gate certifies against, and a record that carried only the composition
    # answer could not show that the narrowing is the only difference.
    standalone = _factored(
        fields, pml, grid,
        lambda g: covers_folded_offdiag_constitutive(fields, pml, g))
    record["standalone_covered_modulo_backend"] = standalone["covered_modulo_backend"]
    record["standalone_first_refusal"] = standalone["first_refusal"]
    return record


# ---------------------------------------------------------------------------
# THE FAMILY THIS ROUND ADDS: complex64 storage x the OFF-DIAGONAL row at update_E
#
# MEASURED 2026-08-20 (final census): 6 of 759 slots unserved, all at update_E, and
# FIVE of them are one intersection -- MEEP's tensor row product under complex64
# storage. Every family that was asked refused on KERNEL SHAPE: ``cuda_offdiag``
# and ``cuda_folded_offdiag`` for "complex64 storage", ``cuda_complex``,
# ``cuda_complex_folded`` and ``cuda_complex_no_pml`` for "off-diagonal chi1inv".
# ``complex_no_pml_kernels.WHAT_IS_NOT_BUILT`` names this transcription as the
# obvious next target and ``meep_gpu/cuda_kernels/complex_offdiag_update_e.py``
# now emits it, so those slots can be MEASURED rather than counted as refused.
#
# TWO PREDICATES, ONE PER TAIL, and both are asked at the SAME slot. The corpus
# splits 3 / 2 on the absorber and the two arms are disjoint by the same
# ``_pml_is_active`` switch the array path branches on (stepping.py:1015 against
# :1022), so at most one can ever admit a row -- which the analyzer's disjointness
# check measures rather than assumes. The entry recorded under the slot is the one
# that admitted, or the PML arm's refusal when neither did.
#
# THIS LEG BINDS THE **PARITY** LICENCE, not the base four-pattern one, and that is
# a property of the arithmetic rather than a convenience. The mirror ghost is
# ``_symmetry_phase(...) * _mirror_source(...)`` (stepping.py:1873-1875) -- a python
# int on the LEFT of a complex64 plane, which is
# ``folded_complex.PARITY_PROBE_PATTERN`` and none of the base four orientations.
# The shipped ``expansion_probe_2026-08-17`` record DOES carry it, so this is not a
# refusal about the artifact; binding ``complex_fields.expansion_license`` here
# would produce one, and that is exactly the failure this file's own
# COMPLEX_NO_PML_PROBE_RECORD comment records from 2026-08-20.
# ---------------------------------------------------------------------------


def _complex_offdiag_license() -> Any:
    """The PARITY-pattern verdict this family's two predicates gate on.

    ``folded_complex.parity_expansion_license`` over the same record the other
    Cartesian complex legs bind, with the fifth orientation included. The verdict
    carries ``probe_patterns``, which is the licence's own record of what its
    arbiter looked at, and the predicates ask THAT rather than an attached table.

    A record that does not licence an arm is not silently downgraded: the reason
    travels into the raise, so a census run against a bad record reads as a census
    failure rather than as 186 rows of coverage gap.
    """
    import json as _json  # noqa: PLC0415
    from meep_gpu.triton_kernels import complex_fields  # noqa: PLC0415
    from meep_gpu.triton_kernels import folded_complex  # noqa: PLC0415

    path = _API_ROOT / COMPLEX_PROBE_RECORD
    record = _json.loads(path.read_text(encoding="utf-8"))
    verdict = folded_complex.parity_expansion_license(record)
    policy_reasons = list(
        complex_fields.expansion_policy_reasons(record, COMPLEX_POLICY))
    if not verdict.get("arm") or verdict.get("refusals") or policy_reasons:
        raise RuntimeError(
            f"{COMPLEX_PROBE_RECORD} licenses no PARITY arm under "
            f"{COMPLEX_POLICY!r}: arm={verdict.get('arm')!r} "
            f"refusals={verdict.get('refusals')} "
            f"policy_reasons={policy_reasons[:1]}")
    return verdict


def _cuda_complex_offdiag_verdicts(fields: Any, pml: Any,
                                   grid: Any) -> Dict[str, Any]:
    """The two complex tensor-row predicates, folded into ONE update_E slot entry.

    BOTH ANSWERS ARE RECORDED and only one is counted. The union census reads
    ``covered_modulo_backend`` off the slot entry, so folding the two arms into one
    entry is what keeps this family from bidding twice at a slot it can only serve
    once; the per-arm verdicts ride alongside so a reader can see WHICH arm took a
    row and what the other one said about it.
    """
    from meep_gpu.cuda_kernels import (  # noqa: PLC0415
        complex_offdiag_update_e as arm)

    licence = _complex_offdiag_license()
    entries = {}
    for name, predicate in (
            ("pml", arm.covers_complex_offdiag_pml_update_e),
            ("no_pml", arm.covers_complex_no_pml_offdiag_update_e)):
        entries[name] = _factored(
            fields, pml, grid,
            lambda g, p=predicate: p(fields, pml, g, license=licence,
                                     subnormal_policy=COMPLEX_POLICY))
    winner = ("pml" if entries["pml"]["covered_modulo_backend"]
              else "no_pml" if entries["no_pml"]["covered_modulo_backend"]
              else "pml")
    record = dict(entries[winner],
                  sub_step="update_E",
                  kernel=arm.KERNEL_NAMES[winner],
                  tail=winner,
                  predicate=("cuda_kernels.complex_offdiag_update_e."
                             + ("covers_complex_offdiag_pml_update_e"
                                if winner == "pml"
                                else "covers_complex_no_pml_offdiag_update_e")),
                  family="cuda_complex_offdiag")
    record["arms"] = {
        name: {"covered_modulo_backend": entry["covered_modulo_backend"],
               "first_refusal": entry["first_refusal"]}
        for name, entry in entries.items()}
    record["arms_disjoint"] = not (entries["pml"]["covered_modulo_backend"]
                                   and entries["no_pml"]["covered_modulo_backend"])
    return {"update_E": record}


# ---------------------------------------------------------------------------
# THE DISPERSIVE OFF-DIAGONAL update_E FAMILY
# ---------------------------------------------------------------------------
#
# The last non-complex slot in the 759. The 2026-08-20 final census left SIX slots
# unserved; five are complex64 and the sixth is
# ``examples/absorbed_power_density.py`` at ``update_E``, refused by FOUR shipped
# families for four different true reasons:
#
#     cuda_constitutive     "dispersion: update_E's source is (D - sum P), not D"
#     cuda_folded_offdiag   the same clause, same words
#     cuda_dispersive       "off-diagonal chi1inv: the row product reads the other
#                            components' (D - sum P) volumes at neighbouring cells"
#     cuda_offdiag          "mirror symmetry: the fold changes the stored extent"
#
# ``cuda_kernels/dispersive_offdiag_update_e.py`` emits the kernel those four
# refusals specify, so that slot can be MEASURED rather than counted as refused.
#
# THERE IS ONE PREDICATE, NOT TWO. The folded off-diagonal family needs a
# composition verdict because its standalone predicate deliberately admits an
# UNFOLDED grid that the CERTIFIED off-diagonal family already claims. This family
# has no such overlap: an off-diagonal DISPERSIVE update_E is refused by
# ``cuda_offdiag`` for its dispersion whether or not a fold is present, and by
# ``cuda_dispersive`` for its off-diagonal row. So the census asks the one
# predicate and the analyzer's disjointness check measures the partition rather
# than assuming it.
# ---------------------------------------------------------------------------


def _cuda_dispersive_offdiag_verdict(fields: Any, pml: Any,
                                     grid: Any) -> Dict[str, Any]:
    """``covers_real_pml_dispersive_offdiag_constitutive``, same factoring.

    Same input factoring as every CUDA leg in this file, and for the same reason:
    this predicate short-circuits on the backend clause too, so the CuPy name is
    supplied at the INPUT rather than filtered out of the refusal string.
    """
    from meep_gpu.cuda_kernels.dispersive_offdiag_update_e import (  # noqa: PLC0415
        covers_real_pml_dispersive_offdiag_constitutive,
    )
    return dict(_factored(
        fields, pml, grid,
        lambda g: covers_real_pml_dispersive_offdiag_constitutive(
            fields, pml, g)),
        sub_step="update_E",
        kernel="fused_update_E_pml_real_folded_offdiag_dispersive",
        predicate=("cuda_kernels.dispersive_offdiag_update_e."
                   "covers_real_pml_dispersive_offdiag_constitutive"),
        family="cuda_dispersive_offdiag")


# ---------------------------------------------------------------------------
# THE FAMILY ADDED AFTER THE CLOSED TRANCHE: the FUSED MAGNETIC PAIR
#
# ``cuda_kernels/fused_magnetic_pair.py`` did not exist on 2026-08-20, so the closed
# census had no caller for ``covers_fused_magnetic_pair`` and the tranche's report
# says nothing about it. Every other family added since that round -- cylindrical
# real (``cylindrical_coverage.covers_real_pml_cylindrical_curl``) and cylindrical
# complex (``..covers_pml_cylindrical_complex_curl``) -- was ALREADY asked by this
# battery under ``cuda_cylindrical`` and ``cuda_cyl_complex``; only the emitters
# beneath them moved, and their signatures did not. Checked before adding, because a
# family asked twice bids twice and the analyzer would report an overlap that is an
# artifact of the question.
#
# IT IS RECORDED OUTSIDE THE UNION CENSUS AND THAT IS THE POINT OF THE FAMILY. The
# 759-slot denominator counts SUB-STEP SLOTS; this predicate answers for a SEAM --
# one launch spanning step_B -> zero_metal_B -> update_H -- and both of the slots it
# spans are already served, by ``cuda_curl`` and ``cuda_constitutive`` respectively.
# Listing it in ``UNION_FAMILIES`` would make the disjointness check fire on every
# row it admits, and the union number would not move by one slot, because a fused
# product does not COVER anything a union census can count: it welds two admissions
# that are already there. The board that asks the seam question is
# ``build_cuda_fusion_matrix.py``; this leg gives it the shipped verdict instead of
# a reconstruction.
#
# IT IS THE ONLY LEG IN THIS FILE THAT TAKES ``sources``, and that is not an
# ergonomic choice. A MAGNETIC source is injected BETWEEN the two halves
# (driver.py:3293), so the predicate refuses one; ``Fields`` does not hold the source
# list, so passing nothing makes it refuse with "the source set was not declared"
# (deposit_repair.py:228-229) -- a refusal about the CENSUS reported as one about the
# corpus, which is the failure this file's COMPLEX_PROBE_RECORD comment records from
# 2026-08-20. ``driver._sources`` is what the driver itself steps with.
#
# MEASURED 2026-08-28, AND IT IS WHY THIS LEG HAS A THIRD OUTCOME: ON A NUMPY HOST
# THE PREDICATE COULD NOT BE ASKED AT ALL. ``fused_magnetic_pair.py`` opened with a
# bare ``import cupy as cp`` AT MODULE SCOPE, so importing the module to reach
# ``covers_fused_magnetic_pair`` raised ``ModuleNotFoundError`` off-device, and the
# 2026-08-28b census recorded ``askable: false`` on all 186 rows. Every other family
# this battery asks lives in a module that keeps that import reachable off-device --
# the idiom ``import cupy as cp  # noqa: PLC0415 - device-only, in a laptop-importable
# module`` at ``folded_offdiag_kernels.py:1187``, ``conductive_kernels.py:536`` and
# six more -- and ``coverage.py`` was patched to be stdlib-only on 2026-08-15 for
# exactly this reason (this file's own preamble, "WHY THIS CAN EXIST AT ALL").
#
# FIXED LATER THE SAME DAY, in the edit that landed the deposit carry: that module's
# ``cupy`` import (and the three certified-half imports it splices from) became
# defensive, and ``_get_kernel`` refuses BY NAME when ``cp is None``. The predicate
# touches neither, so this leg now records a real verdict. THE UNASKABLE BRANCH BELOW
# STAYS: it is what keeps one unimportable predicate from costing the census all 186
# rows, and a census that could not tell "not measured" from "refused" is the failure
# this file exists to avoid.
#
# THE LEG RECORDS THAT RATHER THAN RAISING, AND IT IS NOT RECORDED AS A REFUSAL. An
# ImportError escaping here would make ``evaluate`` raise, which
# ``measure_predicate_coverage`` catches into ``battery_error`` and ``measured:
# false`` -- one unaskable predicate would cost the census all 186 rows. Folding it
# into ``covered_modulo_backend: False`` would be worse: that column means "the
# shipped predicate refused this row", and a census that reported a MISSING
# MEASUREMENT as a refusal about the corpus is the failure this file's
# COMPLEX_PROBE_RECORD comment already records twice. So the row carries
# ``askable: false`` with the import error, and every verdict field is None.
# ---------------------------------------------------------------------------


def _cuda_fused_magnetic_pair_verdict(fields: Any, pml: Any, grid: Any,
                                      sources: Any) -> Dict[str, Any]:
    """``covers_fused_magnetic_pair``, same two-verdict factoring as every CUDA leg.

    The predicate short-circuits on the backend clause like its siblings -- it opens
    by delegating into ``covers_real_pml_curl``, whose first line is that clause -- so
    the CuPy name is supplied at the INPUT rather than filtered out of the returned
    string.

    THREE FIELDS EXIST TO KEEP THREE DIFFERENT ANSWERS APART, because two of them are
    about the census and only one is about the row:

    * ``askable`` -- False when the module cannot be imported on this host. The
      verdict fields are then None, never False. See the preamble.
    * ``sources_declared`` -- False when the caller passed no source list, which this
      predicate refuses by name (deposit_repair.py:228-229). A real refusal, but about
      what the census was told rather than about what the row is.
    * ``covered_modulo_backend`` -- the shipped predicate's verdict on the row.
    """
    record: Dict[str, Any] = {
        "seam": "step_B -> zero_metal_B -> update_H",
        "spans": ["step_B", "update_H"],
        "predicate": ("cuda_kernels.fused_magnetic_pair."
                      "covers_fused_magnetic_pair"),
        "family": "cuda_fused_magnetic_pair",
        "counted_in_union_census": False,
        "sources_declared": sources is not None,
        "askable": True,
        "unaskable_reason": None,
    }
    try:
        from meep_gpu.cuda_kernels.fused_magnetic_pair import (  # noqa: PLC0415
            KERNEL_NAME,
            covers_fused_magnetic_pair,
        )
    except BaseException as exc:  # noqa: BLE001 - an unaskable predicate is not a refusal
        record.update(
            askable=False,
            unaskable_reason=(
                f"{type(exc).__name__}: {exc} -- the module could not be imported on "
                f"this host, so this predicate could not be reached. Its own cupy "
                f"import is defensive; something it depends on is not"[:400]),
            kernel=None, covered=None, covered_modulo_backend=None,
            raw_reason=None, raw_refusal_is_backend_clause=None,
            first_refusal=None, raised=None, shim=None)
        return record
    record.update(_factored(
        fields, pml, grid,
        lambda g: covers_fused_magnetic_pair(fields, pml, g, sources)),
        kernel=KERNEL_NAME)
    return record


def _cuda_fused_electric_pair_verdict(fields: Any, pml: Any, grid: Any,
                                      sources: Any) -> Dict[str, Any]:
    """``covers_fused_electric_pair``, in the shape the board's column reads.

    THE SAME THREE-FIELD FACTORING as the real magnetic pair's, and the same reason
    for taking ``sources``: an ELECTRIC source is injected BETWEEN the two halves
    (driver.py:3305/:3308), so the predicate refuses one it cannot repair, and
    ``Fields`` does not hold the source list -- passing nothing makes it refuse with
    "the source set was not declared", a refusal about the CENSUS reported as one
    about the corpus.

    IT IS THE FIRST COLUMN ON THIS CENSUS THAT ANSWERS FOR THE ELECTRIC SEAM, which is
    also why the board reads ``spans`` off the block rather than assuming ``step_B``:
    the three columns beside it span ``["step_B", "update_H"]`` and this one spans
    ``["step_D", "update_E"]``, and a board that keyed the seam by position would
    price this cell against the wrong injection.
    """
    record: Dict[str, Any] = {
        "seam": "step_D -> zero_metal_D -> update_E",
        "spans": ["step_D", "update_E"],
        "predicate": ("cuda_kernels.fused_electric_pair."
                      "covers_fused_electric_pair"),
        "family": "cuda_fused_electric_pair",
        "counted_in_union_census": False,
        "sources_declared": sources is not None,
        "askable": True,
        "unaskable_reason": None,
    }
    try:
        from meep_gpu.cuda_kernels.fused_electric_pair import (  # noqa: PLC0415
            KERNEL_NAME,
            covers_fused_electric_pair,
        )
    except BaseException as exc:  # noqa: BLE001 - an unaskable predicate is not a refusal
        record.update(
            askable=False,
            unaskable_reason=(
                f"{type(exc).__name__}: {exc} -- the module could not be imported on "
                f"this host, so this predicate could not be reached. Its own cupy "
                f"import is defensive; something it depends on is not"[:400]),
            kernel=None, covered=None, covered_modulo_backend=None,
            raw_reason=None, raw_refusal_is_backend_clause=None,
            first_refusal=None, raised=None, shim=None)
        return record
    record.update(_factored(
        fields, pml, grid,
        lambda g: covers_fused_electric_pair(fields, pml, g, sources)),
        kernel=KERNEL_NAME)
    return record


def _cuda_fused_polarization_pair_verdict(function: str, family: str,
                                          fields: Any, pml: Any, grid: Any,
                                          sources: Any) -> Dict[str, Any]:
    """One of the two E->P pair predicates, in the shape the board's column reads.

    THE FIRST COLUMNS ON THIS CENSUS THAT ANSWER FOR THE E->P SEAM, whose two
    consults are ADJACENT driver statements (driver.py:3313/:3315): the seam
    string carries no in-between pass because the driver puts none there, and the
    board's ``fused_seam_of`` reads the ``spans`` off this block as it does for
    every other column.

    ``sources`` IS STILL PASSED AND STILL RECORDED, and the predicate treats it
    as INERT rather than refusing an undeclared list -- a driver fact, not a
    leniency: no source list can put a deposit between two adjacent statements.
    ``sources_declared`` stays in the record so this column's rows stay
    field-commensurable with the seven beside it.
    """
    record: Dict[str, Any] = {
        "seam": "update_E -> update_P",
        "spans": ["update_E", "update_P"],
        "predicate": f"cuda_kernels.fused_polarization_pair.{function}",
        "family": family,
        "counted_in_union_census": False,
        "sources_declared": sources is not None,
        "askable": True,
        "unaskable_reason": None,
    }
    try:
        from meep_gpu.cuda_kernels import fused_polarization_pair  # noqa: PLC0415
        predicate = getattr(fused_polarization_pair, function)
        kernel_label = fused_polarization_pair.KERNEL_NAME
    except BaseException as exc:  # noqa: BLE001 - an unaskable predicate is not a refusal
        record.update(
            askable=False,
            unaskable_reason=(
                f"{type(exc).__name__}: {exc} -- the module could not be imported on "
                f"this host, so this predicate could not be reached. Its own cupy "
                f"import is defensive; something it depends on is not"[:400]),
            kernel=None, covered=None, covered_modulo_backend=None,
            raw_reason=None, raw_refusal_is_backend_clause=None,
            first_refusal=None, raised=None, shim=None)
        return record
    record.update(_factored(
        fields, pml, grid,
        lambda g: predicate(fields, pml, g, sources)),
        kernel=kernel_label)
    return record


#: Which seam each COMPLEX fused product spans, keyed by its module. READ HERE rather
#: than assumed from the family being complex: the board keys a column's seam off the
#: ``spans`` this block records, and until 2026-08-30 every complex product on this
#: census spanned ``step_B -> update_H`` -- so a hard-coded seam was right by accident
#: and would have priced the two ELECTRIC products against the magnetic injection, which
#: is the wrong clause and a silently wrong ceiling.
_COMPLEX_PAIR_SEAM: Dict[str, Dict[str, Any]] = {
    "complex_fused_magnetic_pair": {
        "seam": "step_B -> zero_metal_B -> update_H",
        "spans": ["step_B", "update_H"]},
    "cylindrical_fused_magnetic_pair": {
        "seam": "step_B -> zero_metal_B -> update_H",
        "spans": ["step_B", "update_H"]},
    "complex_fused_electric_pair": {
        "seam": "step_D -> zero_metal_D -> update_E",
        "spans": ["step_D", "update_E"]},
    "cylindrical_fused_electric_pair": {
        "seam": "step_D -> zero_metal_D -> update_E",
        "spans": ["step_D", "update_E"]},
    # THE SHORTEST SEAM ON THIS TABLE, and it is shorter rather than differently
    # filled: the complex no-absorber product refuses every walled run, so
    # ``zero_metal_D`` writes nothing on any row it admits and its two consults are
    # CONTIGUOUS in driver order. The string is read by the board, which is why it is
    # this product's own ``REPLACES`` rather than the seam's shape.
    "no_pml_complex_fused_electric_pair": {
        "seam": "step_D -> update_E",
        "spans": ["step_D", "update_E"]},
    # THE COMPLEX CYLINDRICAL H->D PRODUCT, 2026-09-07: the fourth seam's only
    # complex row on this table. Its two consults are contiguous in driver order
    # (nothing is INJECTED between update_H and step_D; what sits there is the
    # electric integrated-source WITHDRAW, which the predicate refuses by name), so
    # the string is the product's own ``REPLACES`` and the board resolves it to
    # ``h_to_d_seam.SEAM`` exactly as it does the real ``cuda_fused_hd_pair`` column.
    "cylindrical_fused_hd_pair": {
        "seam": "update_H -> step_D",
        "spans": ["update_H", "step_D"]},
    # THE CARTESIAN COMPLEX H->D PRODUCT, 2026-09-07: the same fourth seam one grid
    # over. Its two consults are contiguous in driver order for the same reason --
    # nothing is INJECTED between ``update_H`` and ``step_D``; the electric
    # integrated-source WITHDRAW sits there and the predicate refuses a standing one
    # by name -- so the string is the product's own ``REPLACES`` and the board
    # resolves it to ``h_to_d_seam.SEAM``. ONE ROW FOR BOTH VARIANTS: the predicate
    # reads the fold off the grid and asks the curl predicate its variant selects, so
    # the plain and folded board cells are separated by the composer's selection at
    # ``step_D`` rather than by a second seam row here.
    "complex_fused_hd_pair": {
        "seam": "update_H -> step_D",
        "spans": ["update_H", "step_D"]},
    # THE COMPLEX BETA H->D WELD, 2026-09-08. Same seam and same spans as the plain
    # complex product above -- the beta insert changes the CURL's arithmetic, not
    # which two consults the launch spans.
    "complex_beta_fused_hd_pair": {
        "seam": "update_H -> step_D",
        "spans": ["update_H", "step_D"]},
    # THE TWO FILL-CARRYING RESIDUAL WELDS, 2026-09-02: the LONGEST seams on
    # this table -- all five driver passes, the two mirror fills carried by the
    # ownership inversion in cf arithmetic. The strings are each product's own
    # ``REPLACES``, which is what the board reads.
    "complex_folded_fused_magnetic_pair": {
        "seam": ("step_B -> fill_symmetry_bc_B -> zero_metal_B -> "
                 "fill_folded_far_ghosts_B -> update_H"),
        "spans": ["step_B", "update_H"]},
    "complex_beta_fused_magnetic_pair": {
        "seam": ("step_B -> fill_symmetry_bc_B -> zero_metal_B -> "
                 "fill_folded_far_ghosts_B -> update_H"),
        "spans": ["step_B", "update_H"]},
    # THE TWO FILL-CARRYING COMPLEX ELECTRIC TWINS, 2026-09-02 (residue round):
    # the D-side mirror of the two rows above, all five driver passes carried
    # through the shared D-side complex carry. Each string is the product's own
    # ``REPLACES``, which is what the board reads.
    "complex_folded_fused_electric_pair": {
        "seam": ("step_D -> fill_symmetry_bc_D -> zero_metal_D -> "
                 "fill_folded_far_ghosts_D -> update_E"),
        "spans": ["step_D", "update_E"]},
    # THE TWO COMPLEX STENCIL WELDS, 2026-09-02 -- the LAST two cells this board
    # recorded UNBUILDABLE. Each string is the product's own ``REPLACES``, which is
    # what the board reads. The folded one carries all five passes; the no-absorber
    # one carries THREE, and that is its module's own declaration rather than a
    # shortened copy: its curl half refuses a mirror fold BY NAME, so the two fill
    # passes are inert on every row it admits and naming them would claim work the
    # launch never has.
    "folded_complex_offdiag_fused_electric_pair": {
        "seam": ("step_D -> fill_symmetry_bc_D -> zero_metal_D -> "
                 "fill_folded_far_ghosts_D -> update_E"),
        "spans": ["step_D", "update_E"]},
    "complex_no_pml_offdiag_fused_electric_pair": {
        "seam": "step_D -> zero_metal_D -> update_E",
        "spans": ["step_D", "update_E"]},
    "complex_beta_fused_electric_pair": {
        "seam": ("step_D -> fill_symmetry_bc_D -> zero_metal_D -> "
                 "fill_folded_far_ghosts_D -> update_E"),
        "spans": ["step_D", "update_E"]},
    # THE THIRD E->P PRODUCT, 2026-09-02: the complex no-absorber polarization
    # pair. Its two consults are ADJACENT driver statements, so the seam string
    # carries no in-between pass -- the driver's own answer, same as the two
    # real E->P columns'.
    "complex_no_pml_fused_polarization_pair": {
        "seam": "update_E -> update_P",
        "spans": ["update_E", "update_P"]},
    # THE COMPLEX NO-ABSORBER THREE-SLOT WELD, 2026-09-04: the first COMPLEX column
    # whose ``spans`` names three slots. The seam string is the product's own
    # ``REPLACES`` -- no fill and no wall clear, because its D->E half refuses a
    # fold and a wall by name -- and the spans are its ``SLOTS``, spelled here so the
    # board credits it at BOTH ``D_to_E`` and ``E_to_P`` rather than refusing a
    # first-and-last reading of ``["step_D", "update_P"]``.
    "complex_no_pml_three_slot_dispersive_weld": {
        "seam": "step_D -> update_E -> update_P",
        "spans": ["step_D", "update_E", "update_P"]},
}


#: Which expansion record each complex fused product's predicate is asked with. The
#: default is :data:`COMPLEX_PROBE_RECORD`, the four-pattern one every ``cuda_complex``
#: arm binds through. THE NO-ABSORBER PRODUCTS NEED THE OTHER ONE and it is a refusal
#: rather than a preference: their halves are the ``cuda_complex_no_pml`` family's,
#: that family's arms are bound through :data:`COMPLEX_NO_PML_PROBE_RECORD` (the
#: seven-pattern record, the only one that classifies ``update_P``'s fifth operand
#: orientation), and asking a predicate with a verdict cut for a different family is
#: asking about a different arm.
_FUSED_PAIR_LICENCE: Dict[str, Callable[[], Any]] = {
    "no_pml_complex_fused_electric_pair": lambda: _complex_no_pml_license(),
    "complex_no_pml_fused_polarization_pair": lambda: _complex_no_pml_license(),
    "complex_no_pml_three_slot_dispersive_weld": lambda: _complex_no_pml_license(),
}


def _cuda_complex_fused_pair_verdict(module: str, function: str, family: str,
                                     fields: Any, pml: Any, grid: Any,
                                     sources: Any) -> Dict[str, Any]:
    """One COMPLEX fused-pair predicate, in the shape the board's column reads.

    THE SAME THREE-FIELD FACTORING as the real pair's, and one addition: these
    predicates take the EXPANSION LICENCE. It is supplied from
    :data:`COMPLEX_PROBE_RECORD` under :data:`COMPLEX_POLICY`, exactly as the
    ``cuda_complex`` slot arms above are asked, so the seam and the cells it is
    priced against are read through one licence rather than two. Asking with
    ``license=None`` would return that family's own named refusal on EVERY row --
    a statement about this census, recorded as if it were about the corpus.

    THE SEAM COMES FROM :data:`_COMPLEX_PAIR_SEAM`, not from a constant in this
    function: two of the four complex products span the ELECTRIC seam, and the
    ``sources`` argument is the same list for both -- it is the PREDICATE that selects
    which of them are in ITS seam, through ``deposit_repair.seam_source_reasons``.
    """
    if module not in _COMPLEX_PAIR_SEAM:
        raise KeyError(
            f"{module} has no seam row in _COMPLEX_PAIR_SEAM; the board reads the "
            f"seam off this block and may not guess it")
    record: Dict[str, Any] = {
        **_COMPLEX_PAIR_SEAM[module],
        "predicate": f"cuda_kernels.{module}.{function}",
        "family": family,
        "counted_in_union_census": False,
        "sources_declared": sources is not None,
        "askable": True,
        "unaskable_reason": None,
        "expansion_probe": (COMPLEX_NO_PML_PROBE_RECORD
                            if module in _FUSED_PAIR_LICENCE
                            else COMPLEX_PROBE_RECORD),
        "expansion_policy": COMPLEX_POLICY,
    }
    try:
        from importlib import import_module  # noqa: PLC0415

        loaded = import_module(f"meep_gpu.cuda_kernels.{module}")
        predicate = getattr(loaded, function)
        kernel_name = loaded.KERNEL_NAME
        licence = _FUSED_PAIR_LICENCE.get(module, _complex_license)()
    except BaseException as exc:  # noqa: BLE001 - an unaskable predicate is not a refusal
        record.update(
            askable=False,
            unaskable_reason=(
                f"{type(exc).__name__}: {exc} -- the module could not be imported or "
                f"the expansion licence could not be read on this host, so this "
                f"predicate could not be reached"[:400]),
            kernel=None, covered=None, covered_modulo_backend=None,
            raw_reason=None, raw_refusal_is_backend_clause=None,
            first_refusal=None, raised=None, shim=None)
        return record
    record.update(_factored(
        fields, pml, grid,
        lambda g: predicate(fields, pml, g, sources, licence, COMPLEX_POLICY)),
        kernel=kernel_name)
    return record


def _cuda_complex_offdiag_fused_pair_verdict(module: str, function: str,
                                            family: str, curl_licence: Any,
                                            curl_record: str, fields: Any, pml: Any,
                                            grid: Any, sources: Any) -> Dict[str, Any]:
    """One COMPLEX OFF-DIAGONAL fused-pair predicate, in the board's column shape.

    THE ONLY COLUMN ON THIS BATTERY THAT BINDS TWO LICENCES, and it needs two
    because its product's two halves sit in different licence families: the curl
    binds ``LICENSE_COMPLEX`` (folded) or ``LICENSE_COMPLEX_NO_PML``, and the
    off-diagonal ``update_E`` binds ``LICENSE_COMPLEX_OFFDIAG``, whose arbiter is
    ``folded_complex.parity_expansion_license`` and classifies a FIFTH multiply
    orientation -- the mirror parity as a coefficient on the left -- that none of
    the base-four records carries.

    Asking BOTH halves with one verdict would record that one record's pattern set
    answered for the other's, which is exactly the failure this file's
    ``COMPLEX_NO_PML_PROBE_RECORD`` comment records from 2026-08-20. Both records
    are named in the row, so a reader can see which artifact licensed which half.
    """
    if module not in _COMPLEX_PAIR_SEAM:
        raise KeyError(
            f"{module} has no seam row in _COMPLEX_PAIR_SEAM; the board reads the "
            f"seam off this block and may not guess it")
    record: Dict[str, Any] = {
        **_COMPLEX_PAIR_SEAM[module],
        "predicate": f"cuda_kernels.{module}.{function}",
        "family": family,
        "counted_in_union_census": False,
        "sources_declared": sources is not None,
        "askable": True,
        "unaskable_reason": None,
        "expansion_probe": COMPLEX_PROBE_RECORD,
        "curl_expansion_probe": curl_record,
        "expansion_policy": COMPLEX_POLICY,
    }
    try:
        from importlib import import_module  # noqa: PLC0415

        loaded = import_module(f"meep_gpu.cuda_kernels.{module}")
        predicate = getattr(loaded, function)
        kernel_name = loaded.KERNEL_NAME
        constitutive = _complex_offdiag_license()
        curl = curl_licence()
    except BaseException as exc:  # noqa: BLE001 - unaskable is not a refusal
        record.update(
            askable=False,
            unaskable_reason=(
                f"{type(exc).__name__}: {exc} -- the module could not be imported "
                f"or one of the two expansion licences could not be read on this "
                f"host, so this predicate could not be reached"[:400]),
            kernel=None, covered=None, covered_modulo_backend=None,
            raw_reason=None, raw_refusal_is_backend_clause=None,
            first_refusal=None, raised=None, shim=None)
        return record
    record.update(_factored(
        fields, pml, grid,
        lambda g: predicate(fields, pml, g, sources, constitutive, COMPLEX_POLICY,
                            curl_license=curl)),
        kernel=kernel_name)
    return record


def _cuda_real_fused_pair_verdict(module: str, function: str, family: str,
                                  seam: str, fields: Any, pml: Any, grid: Any,
                                  sources: Any,
                                  spans: Optional[Sequence[str]] = None
                                  ) -> Dict[str, Any]:
    """One REAL-storage residual fused-pair predicate, in the board's column shape.

    The same three-field factoring as every fused column; NO LICENCE, because
    these products' halves are real-storage families that bind no expansion arm
    -- asking with one would be asking a different question than the predicate
    answers, and the record says so by carrying no ``expansion_probe`` key.

    THE SPANS ARE READ OFF THE SEAM STRING rather than hard-coded: the
    2026-09-02 electric twins put this shape on the D seam too, and a constant
    ``["step_B", "update_H"]`` would have priced them against the MAGNETIC
    injection -- the wrong clause and a silently wrong ceiling, the exact
    defect ``_COMPLEX_PAIR_SEAM``'s own preamble records for the complex rows.

    ``spans`` OVERRIDES THAT DERIVATION AND EXISTS FOR ONE SHAPE, 2026-09-02: a
    THREE-SLOT weld. First-and-last is the right reading for every product that
    spans ONE seam, and exactly the wrong one for a product that spans two -- it
    would record ``["step_D", "update_P"]``, which is no seam this board knows, and
    ``build_cuda_fusion_matrix.fused_seams_of`` would refuse the column by name
    rather than credit it at BOTH ``D_to_E`` and ``E_to_P``. The override is the
    product's own ``SLOTS``, spelled at the call site with the rest of its row.
    """
    passes = [name.strip() for name in seam.split("->")]
    record: Dict[str, Any] = {
        "seam": seam,
        "spans": ([str(name) for name in spans] if spans
                  else [passes[0], passes[-1]]),
        "predicate": f"cuda_kernels.{module}.{function}",
        "family": family,
        "counted_in_union_census": False,
        "sources_declared": sources is not None,
        "askable": True,
        "unaskable_reason": None,
    }
    try:
        from importlib import import_module  # noqa: PLC0415

        loaded = import_module(f"meep_gpu.cuda_kernels.{module}")
        predicate = getattr(loaded, function)
        kernel_name = loaded.KERNEL_NAME
    except BaseException as exc:  # noqa: BLE001 - an unaskable predicate is not a refusal
        record.update(
            askable=False,
            unaskable_reason=(
                f"{type(exc).__name__}: {exc} -- the module could not be imported "
                f"on this host, so this predicate could not be reached"[:400]),
            kernel=None, covered=None, covered_modulo_backend=None,
            raw_reason=None, raw_refusal_is_backend_clause=None,
            first_refusal=None, raised=None, shim=None)
        return record
    record.update(_factored(
        fields, pml, grid,
        lambda g: predicate(fields, pml, g, sources)),
        kernel=kernel_name)
    return record


def cuda_offdiag_row_mask(fields: Any) -> Dict[str, Any]:
    """The row mask the hand-CUDA emitter would specialize on, from the SHIPPED helper.

    The census below asks the SIBLING TRACK's ``row_volumes_for``; this asks this
    track's own ``offdiag_row_mask``, and both are recorded, so a drift between the
    two shipped helpers shows up as a per-row disagreement rather than as a silently
    different variant count.
    """
    from meep_gpu.cuda_kernels.coverage import offdiag_row_mask  # noqa: PLC0415

    try:
        return {"mask": list(offdiag_row_mask(fields)), "error": None}
    except BaseException as exc:  # noqa: BLE001
        return {"mask": None, "error": f"{type(exc).__name__}: {exc}"[:200]}


def specialization_tuple(fields: Any, pml: Any, grid: Any) -> Dict[str, Any]:
    """The constexpr tuple each of three families' planners would request on this row.

    Every value is the SHIPPED derivation, cited. Nothing is modelled:

    * ``pml_curl_step`` — ``BACKWARD`` is ``launch.SUB_STEPS[sub_step]['backward']``
      (launch.py:933) and ``BC*`` is ``[1 if kind == 'metallic' else 0 for kind in
      _boundary_kinds(grid, pml)]`` (launch.py:995-996). ``BLOCK`` is
      ``kernels.DEFAULT_BLOCK`` on every production plan (launch.py:993), so it is
      read from the module rather than assumed constant.
    * ``constitutive_step`` — ``SCALE`` is ``1 if side == 'E' else 0``
      (launch.py:1054).
    * ``offdiag_constitutive_step`` — ``R01..R22`` is
      ``tuple(int(v is not None) for v in row_volumes_for(fields))``
      (offdiag_update_e.py:765 over the helper at :539), ``BC*`` is
      ``tuple(BOUNDARY_CODES[kind] for kind in kinds)`` (:876), and ``WM_*`` is the
      shipped ``wall_mask_axes(grid)`` (:889 over the helper at :521). The last two are
      called, not transcribed.

    A row whose configuration the family refuses still yields a tuple here, and that is
    deliberate: the census asks how wide the axis is across the corpus, which is a
    question about the configurations, not about admission. The admitted-only count is
    computed downstream from the verdicts, so both are available.
    """
    record: Dict[str, Any] = {"errors": []}

    def guard(label: str, call: Callable[[], Any]) -> Any:
        try:
            return call()
        except BaseException as exc:  # noqa: BLE001
            record["errors"].append(f"{label}: {type(exc).__name__}: {exc}"[:200])
            return None

    kinds = guard("boundary_kinds", lambda: _resolved_boundary_kinds(grid, pml))
    bc_metallic = (None if kinds is None
                   else [1 if kind == "metallic" else 0 for kind in kinds])

    block = guard("DEFAULT_BLOCK", _default_block)
    record["BLOCK"] = block

    # pml_curl_step: one tuple per sub-step, because BACKWARD differs by sub-step and
    # both sub-steps run on every row.
    sub_step_backward = guard("SUB_STEPS", _sub_step_backward) or {}
    record["pml_curl_step"] = {
        name: (None if bc_metallic is None else
               {"BACKWARD": backward, "BCX": bc_metallic[0], "BCY": bc_metallic[1],
                "BCZ": bc_metallic[2], "BLOCK": block})
        for name, backward in sub_step_backward.items()}

    # constitutive_step
    record["constitutive_step"] = {
        "update_H": {"SCALE": 0, "BLOCK": block},
        "update_E": {"SCALE": 1, "BLOCK": block}}

    # offdiag_constitutive_step
    def offdiag() -> Dict[str, Any]:
        from meep_gpu.triton_kernels.offdiag_update_e import (  # noqa: PLC0415
            BOUNDARY_CODES,
            row_volumes_for,
            wall_mask_axes,
        )
        rows = tuple(int(value is not None) for value in row_volumes_for(fields))
        codes = (None if kinds is None
                 else tuple(int(BOUNDARY_CODES[kind]) for kind in kinds))
        walls = tuple(int(flag) for flag in wall_mask_axes(grid))
        return {"R01": rows[0], "R02": rows[1], "R11": rows[2], "R12": rows[3],
                "R21": rows[4], "R22": rows[5],
                "BCX": None if codes is None else codes[0],
                "BCY": None if codes is None else codes[1],
                "BCZ": None if codes is None else codes[2],
                "WM_X": walls[0], "WM_Y": walls[1], "WM_Z": walls[2],
                "BLOCK": block}

    record["offdiag_constitutive_step"] = guard("offdiag", offdiag)
    return record


def _resolved_boundary_kinds(grid: Any, pml: Any) -> Any:
    from meep_gpu.stepping import _boundary_kinds  # noqa: PLC0415
    active = pml is not None and getattr(pml, "is_active", False)
    return list(_boundary_kinds(grid, pml if active else None) or ())


def _default_block() -> int:
    """``kernels.DEFAULT_BLOCK``, read from the source because the module needs Triton.

    ``triton_kernels/kernels.py`` imports ``triton`` at module scope, so the constant
    the planners bind (launch.py:993, :1105) cannot be imported on this host. It is a
    module-level literal, so it is read with ``ast`` from the shipped file rather than
    restated here — a restated 256 would silently survive the constant changing.
    """
    import ast  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    import meep_gpu  # noqa: PLC0415

    source = Path(meep_gpu.__file__).parent / "triton_kernels" / "kernels.py"
    for node in ast.parse(source.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "DEFAULT_BLOCK"
                for target in node.targets):
            return int(ast.literal_eval(node.value))
    raise LookupError("DEFAULT_BLOCK is not a module-level literal in kernels.py")


def _sub_step_backward() -> Dict[str, int]:
    from meep_gpu.triton_kernels.launch import SUB_STEPS  # noqa: PLC0415
    return {name: int(spec["backward"]) for name, spec in SUB_STEPS.items()}


def plan_step_live(fields: Any, pml: Any, sources: Any) -> Dict[str, Any]:
    """Which in-seam driver passes actually run on this row, under key ``plan_step``.

    THE VALUE IS AN ENGINE FACT AND THE KEY IS THE METAL CENSUS'S. ``live_sub_steps``
    is the function ``metal_kernels.launch.plan_step`` itself calls at launch.py:977
    to fill ``MetalStepPlan.live``, and it reads only the driver's source list, the
    grid's metallic and mirror declarations and the registered polarizations -- no
    device, no shader, no arm. Calling it directly rather than through ``plan_step``
    is deliberate: ``plan_step`` also returns ``selected``, which is METAL ARM ROUTING
    -- a shader on a track with its own residency and its own binding ceiling -- and a
    census that carried it would invite someone to read it as a CUDA cell. THE CUDA
    CELL IS :func:`plan_step_selected`'s, taken from the hand-CUDA composer, and the
    two ride in this same record under different keys precisely so neither can be
    mistaken for the other.

    Recorded under ``plan_step.live`` because that is where the consumer looks
    (``build_cuda_fusion_matrix.py:167``), and it looks there because it was JOINING
    this fact out of the Metal census. Deriving it from the boundary triple instead
    was MEASURED wrong on 85 rows, so it is asked rather than reconstructed.

    ``live_resolved`` is the honest half. ``live_sub_steps`` returns ``None`` when the
    question cannot be answered -- an undeclared source list or unreadable engine
    state -- and ``plan_step`` folds that to ``()``. An empty pass list and an
    unanswerable one price a fused seam very differently (pass-free is the OPTIMISTIC
    direction), so the flag rides alongside instead of being flattened away.
    """
    record: Dict[str, Any] = {"live": [], "live_resolved": False, "error": None}
    try:
        from meep_gpu.metal_kernels.launch import live_sub_steps  # noqa: PLC0415

        live = live_sub_steps(fields, pml, sources)
    except BaseException as exc:  # noqa: BLE001 - an unreadable seam is not an empty one
        record["error"] = f"{type(exc).__name__}: {exc}"[:400]
        return record
    record["live_resolved"] = live is not None
    record["live"] = [str(name) for name in (live or ())]
    return record


# ---------------------------------------------------------------------------
# WHICH ARM THE COMPOSER SELECTS -- the CUDA counterpart of the Metal census's
# ``plan_step.selected`` (``predicate_battery.py:759-800``).
#
# WHY THE COLUMN EXISTS AT ALL. Every leg above records what a PREDICATE says: 22
# families, 759 of 759 slots admitted. Admission is not selection. Until
# ``cuda_kernels/arms.py`` landed there was nothing to ask -- ``build_cuda_fusion_
# matrix.py`` derived a cell from WHICH PREDICATE ADMITS, which is only a cell while
# exactly one does, and it now refuses to walk for exactly that reason. This leg asks
# the shipped composer instead, so the board reads a DECISION rather than a
# reconstruction of one.
#
# ADMITTED != SELECTED, and the gap is the finding this column can produce. A slot two
# families admit is UNSELECTED under ``_select_slot`` (launch.py:2004-2050) naming
# both; a slot one family admits but whose builder refuses is unselected too, with a
# different reason; and a slot no arm is registered on is a fourth kind (``fill_B``/
# ``fill_D`` were that slot until 2026-09-27, when the mirror-fill arm was
# registered on both -- on an UNFOLDED row that arm refuses by name, a fifth kind,
# ``not_folded``, because the slot has nothing to serve there). Every kind is
# recorded as a refusal, because a blank cell and a refused one are different engine
# facts and only some of them are coverage gaps.
# ---------------------------------------------------------------------------


def _composer_licenses() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """The three expansion licences :class:`arms.StepContext` binds, and their record.

    THREE AND NOT ONE, for the reason the legs above already measure: the Cartesian /
    cylindrical / folded / beta complex arms take the four-pattern record, the
    complex no-absorber arms need the seven-pattern one, and the complex off-diagonal
    arms need the PARITY licence. The keys are :mod:`registry`'s own constants, not
    strings spelled here -- a key spelled here that drifted from the table's would
    hand every complex family ``license=None``, which is a NAMED refusal per family
    and would read off the census as a coverage gap rather than as a harness defect.

    A licence that will not load is recorded and the composition still runs. That is
    the fail-closed direction: the family gets ``None``, refuses by its own name, and
    the reason travels into ``refusals`` where it can be read -- as against raising
    and losing the other 30-odd arms' answers for this row.
    """
    from meep_gpu.cuda_kernels import registry  # noqa: PLC0415

    loaders = (
        (registry.LICENSE_COMPLEX, COMPLEX_PROBE_RECORD, _complex_license),
        (registry.LICENSE_COMPLEX_NO_PML, COMPLEX_NO_PML_PROBE_RECORD,
         _complex_no_pml_license),
        (registry.LICENSE_COMPLEX_OFFDIAG, COMPLEX_PROBE_RECORD,
         _complex_offdiag_license),
    )
    licenses: Dict[str, Any] = {}
    report: Dict[str, Any] = {}
    for key, record_path, load in loaders:
        try:
            verdict = load()
        except BaseException as exc:  # noqa: BLE001 - a missing licence is a refusal
            report[key] = {"bound": False, "record": record_path, "arm": None,
                           "error": f"{type(exc).__name__}: {exc}"[:400]}
            continue
        licenses[key] = verdict
        report[key] = {"bound": True, "record": record_path,
                       "arm": verdict.get("arm") if isinstance(verdict, dict)
                       else None,
                       "error": None}
    return licenses, report


def _refusal_kind(slot: str, reasons: Tuple[str, ...]) -> Dict[str, Any]:
    """Classify ONE unselected slot's refusal, against the shipped strings themselves.

    NOTHING HERE TRANSCRIBES A MESSAGE. The no-arm sentence is compared against
    ``arms.NO_ARM_REASONS`` and the ambiguity sentence against a probe built by
    calling ``arms.ambiguity(slot)`` with two placeholder labels, so a reworded
    composer message reclassifies the row instead of silently falling through to the
    catch-all. The admitters are recovered from between the probe's own head and
    tail, which is the only place ``CudaStepPlan`` carries them: ``_select_slot``
    RETURNS the admitted labels and ``arms.plan_step`` does not keep them
    (``arms.py:670``), so on an ambiguous slot the reason string is the record.
    """
    from meep_gpu.cuda_kernels import arms  # noqa: PLC0415

    if not reasons:
        return {"kind": "unreported", "admitters": [],
                "note": "the slot is unselected and the composer recorded no reason"}
    if len(reasons) == 1:
        reason = reasons[0]
        if reason in (arms.NO_ARM_REASONS.get(slot),
                      f"no CUDA arm is registered on {slot}"):
            return {"kind": "no_arm_registered", "admitters": []}
        # THE FILL SLOT OF AN UNFOLDED ROW (since 2026-09-27). The mirror-fill arm is
        # registered on both fill slots and refuses an unfolded grid BY NAME, so the
        # slot is unselected there -- and has nothing to serve, which is not the
        # coverage gap ``no_admitter`` names. A fold the arm refuses (complex
        # storage, a cylindrical radial axis) keeps ``no_admitter``: that one is.
        _prefix, _colon, body = str(reason).partition(": ")
        if body.startswith(arms.MIRROR_FILL_UNFOLDED_REFUSAL):
            return {"kind": "not_folded", "admitters": []}
        probe = arms.ambiguity(slot)(("<<A>>", "<<B>>"))
        head, marker, tail = probe.partition("<<A>>, <<B>>")
        if marker and reason.startswith(head) and reason.endswith(tail):
            inner = reason[len(head):len(reason) - len(tail)] if tail else reason[len(head):]
            return {"kind": "ambiguous",
                    "admitters": [part.strip() for part in inner.split(",")]}
        if f" coverage admitted {slot} but its builder " in reason:
            return {"kind": "builder_refused", "admitters": []}
        if reason.startswith(f"no consulted product admitted {slot}"):
            return {"kind": "no_admitter_no_reason", "admitters": []}
    return {"kind": "no_admitter", "admitters": []}


def plan_step_selected(fields: Any, pml: Any, grid: Any) -> Dict[str, Any]:
    """Run the SHIPPED hand-CUDA composer and record which arm won each slot.

    THE GRID IS THE SHIMMED ONE, and that is what makes this column commensurable
    with every other number in this file. Each shipped CUDA predicate SHORT-CIRCUITS
    on ``backend is not CuPy`` (coverage.py:82), so on this NumPy host the composer
    asked with the real grid selects NOTHING on every slot of every row -- a true
    statement about the host and a useless one about the corpus. The census's
    counting verdict is ``covered_modulo_backend``, measured through
    :class:`_GridWithCupyBackend`, so the SELECTION is measured through the same
    proxy and against the same soundness probe (:func:`shim_soundness`, recorded
    here per row exactly as the other legs record it). ``unfactored`` carries the
    real-grid composition beside it, so the shim's effect is readable rather than
    assumed: ``winner_without_shim`` says, per slot, what the arm the shimmed
    composition chose did WITHOUT the proxy -- refused for the backend clause, or
    refused for something else (a finding about the shim), or admitted anyway, which
    is the ``no-PML null`` arm's measured state because a null kernel has no array to
    be on the wrong backend.

    AN EMPTY SELECTION IS A FINDING, NOT A BLANK. Every slot of ``STEP_ORDER`` comes
    back either in ``selected`` or in ``refusals`` with its kind and its reasons --
    ambiguity (two families admit, and the composer refuses rather than picking by
    table order), a builder that refused, no arm registered on the slot at all
    (none since 2026-09-27; the fill slots carry the mirror-fill arm, which refuses an
    unfolded row as ``not_folded``), or no admitter with every consulted arm's own named
    reason. ``unselected`` lists them so a board can count them without re-deriving
    the partition.

    ``selected_kernel`` AND ``selected_family`` ARE NOT DECORATION. ``selected``
    carries the arm LABEL, which is what ``_select_slot`` records, and two different
    families ship the label a reader would expect to disambiguate them -- the
    census's own ``kernel=`` string is the plan's, so it is read off
    ``CudaSlotPlan`` rather than looked up from the label. ``launchable`` is the
    subset whose plan carries an established launch-argument resolution; it is
    SMALLER than ``replaces`` on purpose and says so rather than being read as a
    dispatch claim.

    NOTHING IS DISPATCHED AND NOTHING IS PATCHED. ``arms.plan_step`` builds plans and
    never calls their launch-argument resolvers, ``meep_gpu.fastpath.plan_fast_path``
    still returns ``None`` on every branch, and no gate, builder or predicate is
    monkeypatched here. The only input factoring is the grid proxy the whole file
    already uses.
    """
    from meep_gpu.cuda_kernels import arms  # noqa: PLC0415

    proxy = _GridWithCupyBackend(grid)
    licenses, license_report = _composer_licenses()
    record: Dict[str, Any] = {
        "selected": {}, "selected_kernel": {}, "selected_family": {},
        "replaces": [], "launchable": [], "reasons": {}, "refusals": {},
        "unselected": [], "licenses": license_report,
        "subnormal_policy": COMPLEX_POLICY,
        "registration_state": None,
        "grid_factoring": "xp.__name__ reported as cupy; every other read forwarded",
        "shim": shim_soundness(grid, proxy),
        "unfactored": {"selected": {}, "replaces": [],
                       "winner_without_shim": {},
                       "refusal_clauses": {}, "error": None},
        "composer_error": None,
    }

    def compose(target_grid: Any) -> Any:
        # UNCHANGED BY THE DISPATCH WIRING, DELIBERATELY. ``plan_step`` grew a
        # ``fuse_labels`` argument when the seam started composing this table, and
        # this call does not pass it: ``None`` means NO RESTRICTION, which composes
        # byte-for-byte what the census has always measured. A census that started
        # offering a subset would be measuring the DISPATCHER's admission rather
        # than the table's coverage, which is the one thing this battery is for.
        # ``fuse`` is likewise still absent, so the fusion block does not run here
        # at all.
        return arms.plan_step(fields, pml, target_grid, licenses=licenses,
                              subnormal_policy=COMPLEX_POLICY)

    try:
        plan = compose(proxy)
    except BaseException as exc:  # noqa: BLE001 - plan_step must never raise
        record["composer_error"] = f"{type(exc).__name__}: {exc}"[:400]
        return record

    record["registration_state"] = arms.registration_state()
    record["selected"] = dict(plan.selected)
    record["replaces"] = list(plan.replaces)
    record["launchable"] = list(plan.launchable)
    for slot, slot_plan in plan.plans.items():
        record["selected_kernel"][slot] = str(
            getattr(slot_plan, "kernel_label", ""))
        record["selected_family"][slot] = str(getattr(slot_plan, "family", ""))
    record["reasons"] = {slot: [str(reason)[:400] for reason in reasons][:3]
                         for slot, reasons in plan.reasons.items()}
    for slot in arms.STEP_ORDER:
        if slot in plan.selected:
            continue
        reasons = tuple(plan.reasons.get(slot, ()))
        record["unselected"].append(slot)
        record["refusals"][slot] = dict(
            _refusal_kind(slot, reasons),
            reason_count=len(reasons),
            reasons=[str(reason)[:400] for reason in reasons][:6])

    try:
        raw = compose(grid)
    except BaseException as exc:  # noqa: BLE001
        record["unfactored"]["error"] = f"{type(exc).__name__}: {exc}"[:400]
        return record
    record["unfactored"]["selected"] = dict(raw.selected)
    record["unfactored"]["replaces"] = list(raw.replaces)
    raw_reasons = [str(reason) for slot in arms.STEP_ORDER
                   if slot not in arms.NO_ARM_REASONS
                   for reason in raw.reasons.get(slot, ())]
    backend = [reason for reason in raw_reasons if is_backend_clause_alone(reason)]
    record["unfactored"]["refusal_clauses"] = {
        "backend": len(backend), "other": len(raw_reasons) - len(backend)}
    # WHAT THE SHIM DID TO THE WINNER, PER SLOT AND NOT IN AGGREGATE. "every
    # unshimmed refusal is the backend clause" is FALSE on this corpus and is not the
    # claim: several families reach a configuration clause before the backend one, so
    # a beta = 0 row is refused by ``real beta`` for its beta whatever the array
    # module is. What matters is narrower -- what the arm the SHIMMED composition
    # selected did without the shim -- and it is recorded as one of three states
    # rather than a boolean, because the third is real and MEASURED:
    #
    #   ``backend``  refused, and for the backend clause alone. The expected state.
    #   ``selected`` ADMITTED WITHOUT THE SHIM and won the slot unshimmed too. Not
    #                a defect and not an accident: ``covers_no_pml_null_constitutive``
    #                gates a NULL kernel, so it has no array to be on the wrong
    #                backend, and it is the winner on the no-absorber rows.
    #   ``other``    refused for something that is not the backend clause. A finding
    #                about the shim -- it would mean the proxy changed the verdict for
    #                a reason about the row -- so it gets its own name instead of
    #                sharing ``False`` with the state above.
    #
    # An arm's refusals carry its own ``prefix`` (``arms.ArmSpec``), which is what
    # makes the winner's line findable among every consulted arm's.
    for slot, label in plan.selected.items():
        if raw.selected.get(slot) == label:
            state = "selected"
        else:
            mine = [reason for reason in map(str, raw.reasons.get(slot, ()))
                    if reason.startswith(f"{label}: ")]
            state = ("backend" if mine and all(
                is_backend_clause_alone(reason) for reason in mine)
                else "other")
        record["unfactored"]["winner_without_shim"][slot] = state
    return record


def _merge_plan_step(live: Dict[str, Any],
                     composed: Dict[str, Any]) -> Dict[str, Any]:
    """One ``plan_step`` record from the seam fact and the composition, keys disjoint.

    REFUSES A COLLISION rather than letting one half win. The live record's ``error``
    is about ``live_sub_steps`` and the composition's is about ``plan_step``; a merge
    that silently overwrote either would put one function's failure under the other
    function's name, which is why the composition spells ``composer_error``. A future
    key added to one side that shadows the other is caught here, at the row that
    would have carried it, instead of in a board reading a number that moved.
    """
    clash = sorted(set(live) & set(composed))
    if clash:
        raise KeyError(
            f"plan_step keys {clash} are recorded by BOTH the live-seam leg and the "
            f"composer leg; one would silently overwrite the other")
    return dict(live, **composed)


def evaluate(driver: Any, probe: Any) -> Dict[str, Any]:
    """The CUDA predicate, the Triton cross-check, and the ladder legs. JSON-safe."""
    from meep_gpu.triton_kernels import (  # noqa: PLC0415
        ade_update_p_coverage,
        constitutive_coverage,
        pml_curl_coverage,
    )
    from meep_gpu.triton_kernels.offdiag_update_e import (  # noqa: PLC0415
        offdiag_constitutive_coverage,
    )

    fields = driver.fields
    pml = driver.pml
    grid = driver.grid
    sources = getattr(driver, "_sources", None)
    states = tuple(getattr(fields, "polarizations", ()) or ())

    # --- THE SUBJECT: the shipped hand-CUDA curl predicate, ONE CALL PER SUB-STEP --
    cuda_curl = {name: dict(_cuda_verdict(fields, pml, grid, name),
                            kernel="covers_real_pml_curl", sub_step=name,
                            family="cuda_certified")
                 for name in CUDA_CURL_SUB_STEPS}

    # --- THE CROSS-CHECK: Triton's predicate for the SAME sub-step -----------------
    triton_curl = {name: dict(_triton_verdict(
        lambda n=name: pml_curl_coverage(fields, pml, n)),
        kernel="pml_curl", sub_step=name, family="triton_shipped")
        for name in CUDA_CURL_SUB_STEPS}

    # --- THE LADDER: sub-steps with NO CUDA kernel and NO CUDA predicate -----------
    # Each leg is an intersection of two SHIPPED predicates' verdicts. The CUDA half
    # is now the PAIRED curl's verdict rather than one answer copied to every leg —
    # MEEP closes B with H and D with E — and both curls' verdicts ride along so the
    # effect of that pairing is readable instead of baked in. The Triton half is that
    # sub-step's own shipped predicate, unchanged.
    ladder: Dict[str, Any] = {}
    for side, sub_step in (("H", "update_H"), ("E", "update_E")):
        ladder[f"constitutive@{sub_step}"] = {
            "sub_step": sub_step,
            "intersects": ["cuda_kernels.coverage.covers_real_pml_curl"
                           f"(sub_step={LADDER_CURL_PAIRING[sub_step]!r})",
                           f"triton_kernels.constitutive_coverage(side={side!r})"],
            "triton": _triton_verdict(lambda s=side: constitutive_coverage(fields, pml, s)),
        }
    ladder["offdiag_constitutive@update_E"] = {
        "sub_step": "update_E",
        "intersects": ["cuda_kernels.coverage.covers_real_pml_curl(sub_step='step_D')",
                       "triton_kernels.offdiag_update_e.offdiag_constitutive_coverage"],
        "triton": _triton_verdict(lambda: offdiag_constitutive_coverage(fields, pml)),
    }
    for key, entry in ladder.items():
        paired = LADDER_CURL_PAIRING[entry["sub_step"]]
        admits = {name: bool(cuda_curl[name]["covered_modulo_backend"])
                  for name in CUDA_CURL_SUB_STEPS}
        entry["cuda_curl_admits_by_sub_step"] = admits
        entry["cuda_curl_paired_with"] = paired
        entry["cuda_curl_admits"] = admits[paired]
        entry["cuda_curl_admits_both"] = all(admits.values())
        entry["both_admit"] = bool(admits[paired]
                                   and entry["triton"]["covered_modulo_backend"])

    # --- THE MEASUREMENT the earlier run could not take: the SHIPPED CUDA
    # constitutive predicate, one call per side. This is no longer an
    # intersection of two predicates standing in for a kernel — the kernels exist
    # (constitutive_kernels.fused_update_{H,E}_pml_real) and this is their gate.
    cuda_constitutive = {
        f"update_{side}": dict(
            _cuda_constitutive_verdict(fields, pml, grid, side),
            sub_step=f"update_{side}", family="cuda_authored_ungated")
        for side in ("H", "E")}
    # Where the shipped predicate and the projection disagree, per sub-step. The
    # projection assumed a kernel written to Triton's clause set; the shipped one
    # deliberately admits a conductivity that Triton's shared _grid_reasons does
    # not check either — so a disagreement is a finding to read, not a defect.
    for sub_step, entry in cuda_constitutive.items():
        projected = ladder.get(f"constitutive@{sub_step}", {})
        entry["projection_both_admit"] = projected.get("both_admit")
        entry["agrees_with_projection"] = (
            entry["covered_modulo_backend"] == projected.get("both_admit"))

    # --- update_P, per (state, driven component) -----------------------------------
    polarization: List[Dict[str, Any]] = []
    for index, state in enumerate(states):
        driven: Tuple[str, ...] = ()
        reader = getattr(state, "driven", None)
        if callable(reader):
            try:
                driven = tuple(reader())
            except BaseException:  # noqa: BLE001
                driven = ()
        row = {"index": index,
               "kind": str(getattr(getattr(state, "susceptibility", None), "kind", None)),
               "driven": list(driven),
               "components": {}}
        for component in driven:
            row["components"][component] = _triton_verdict(
                lambda s=state, c=component: ade_update_p_coverage(fields, s, c))
        polarization.append(row)

    # --- what the configuration IS, so a refusal can be attributed ------------------
    def ask(name: str, *args: Any) -> Any:
        attribute = getattr(grid, name, None)
        if attribute is None:
            return None
        if callable(attribute):
            try:
                return attribute(*args)
            except BaseException:  # noqa: BLE001
                return None
        return attribute

    configuration = {
        "shape": [int(v) for v in getattr(grid, "shape", ()) or ()],
        "xp": getattr(getattr(grid, "xp", None), "__name__", None),
        "force_complex_fields": bool(getattr(fields, "force_complex_fields", False)),
        "pml_active": bool(pml is not None and getattr(pml, "is_active", False)),
        "has_bloch": bool(ask("has_bloch") or False),
        "k_point": [float(v) for v in (getattr(grid, "k_point", None) or (0.0, 0.0, 0.0))],
        "beta": float(getattr(grid, "beta", 0.0) or 0.0),
        "bfast_active": bool(getattr(grid, "bfast_active", False)),
        "cylindrical": bool(getattr(grid, "cylindrical", False)),
        "has_symmetry": bool(ask("has_symmetry") or False),
        "mirrored": [bool(ask("is_mirrored", axis)) for axis in range(3)],
        "is_axis": [bool(ask("is_axis", axis)) for axis in range(3)],
        "has_metallic": bool(ask("has_metallic") or False),
        "metallic": [bool(ask("is_metallic", axis)) for axis in range(3)],
        "has_nonlinearity": bool(getattr(fields, "has_nonlinearity", False)),
        "has_offdiagonal_epsilon": bool(getattr(fields, "has_offdiagonal_epsilon", False)),
        "has_conductivity": bool(getattr(fields, "has_conductivity", False)),
        "has_magnetic_conductivity": bool(
            getattr(fields, "has_magnetic_conductivity", False)),
        "stores_E": bool(getattr(fields, "stores_E", False)),
        "n_polarizations": len(states),
        "n_sources": (len(tuple(sources)) if sources is not None else None),
        # Whether a row's sources are magnetic or electric, which is what decides the
        # SOURCE-SEAM CEILING on a fused pair. A property of the LIFTED SCRIPT, not of
        # a backend, so it belongs in every census rather than in one of them.
        "source_field_types": (
            [str(getattr(s, "field_type", "")) for s in tuple(sources)]
            if sources is not None else None),
    }
    try:
        from meep_gpu.cuda_kernels.coverage import (  # noqa: PLC0415
            real_pml_boundary_kinds,
        )
        configuration["cuda_boundary_kinds"] = list(real_pml_boundary_kinds(grid))
    except BaseException as exc:  # noqa: BLE001
        configuration["cuda_boundary_kinds"] = f"UNRESOLVED {type(exc).__name__}"[:200]
    try:
        from meep_gpu.stepping import _boundary_kinds as resolve  # noqa: PLC0415
        configuration["boundary_kinds"] = list(
            resolve(grid, pml if configuration["pml_active"] else None) or ())
    except BaseException as exc:  # noqa: BLE001
        configuration["boundary_kinds"] = f"UNRESOLVED {type(exc).__name__}: {exc}"[:200]

    # --- THE MEASUREMENT THIS ROUND ADDS: the SHIPPED CUDA off-diagonal predicate.
    # Its ladder leg (offdiag_constitutive@update_E, an intersection standing in
    # for a kernel that did not exist) stays above, unchanged, so the projection
    # and the measurement can be read side by side.
    cuda_offdiag = dict(_cuda_offdiag_verdict(fields, pml, grid),
                        family="cuda_authored_ungated")
    projected = ladder.get("offdiag_constitutive@update_E", {})
    cuda_offdiag["projection_both_admit"] = projected.get("both_admit")
    cuda_offdiag["agrees_with_projection"] = (
        cuda_offdiag["covered_modulo_backend"] == projected.get("both_admit"))
    cuda_offdiag["row_mask"] = cuda_offdiag_row_mask(fields)

    return {"cuda_curl": cuda_curl, "triton_curl": triton_curl,
            "cuda_constitutive": cuda_constitutive,
            "cuda_offdiag": cuda_offdiag, "ladder": ladder,
            "cuda_complex": _cuda_complex_verdicts(fields, pml, grid),
            "cuda_cylindrical": _cuda_cylindrical_verdicts(fields, pml, grid),
            "cuda_no_pml": _cuda_no_pml_verdicts(fields, pml, grid),
            "cuda_ade": _cuda_ade_verdicts(fields, pml, grid),
            "cuda_no_pml_curl": _cuda_no_pml_curl_verdicts(fields, pml, grid),
            "cuda_cyl_complex": _cuda_cyl_complex_verdicts(fields, pml, grid),
            "cuda_nonlinear": _cuda_nonlinear_verdicts(fields, pml, grid),
            "cuda_special_kz": _cuda_special_kz_verdicts(fields, pml, grid),
            "cuda_bfast": _cuda_bfast_verdicts(fields, pml, grid),
            "cuda_complex_no_pml": _cuda_complex_no_pml_verdicts(
                fields, pml, grid),
            "cuda_complex_folded": _cuda_complex_folded_verdicts(fields, pml, grid),
            "cuda_complex_beta": _cuda_complex_beta_verdicts(fields, pml, grid),
            "cuda_conductive": _cuda_conductive_verdicts(fields, pml, grid),
            "cuda_dispersive": _cuda_dispersive_verdicts(fields, pml, grid),
            "cuda_no_pml_dispersive": _cuda_no_pml_dispersive_verdicts(
                fields, pml, grid),
            "cuda_no_pml_ade": _cuda_no_pml_ade_verdicts(fields, pml, grid),
            "cuda_folded_offdiag": {
                "update_E": _cuda_folded_offdiag_verdict(fields, pml, grid)},
            "cuda_complex_offdiag": _cuda_complex_offdiag_verdicts(
                fields, pml, grid),
            "cuda_dispersive_offdiag": {
                "update_E": _cuda_dispersive_offdiag_verdict(fields, pml, grid)},
            # A SEAM verdict, not a slot family: never listed in UNION_FAMILIES.
            # See _cuda_fused_magnetic_pair_verdict's preamble for why.
            "cuda_fused_magnetic_pair": _cuda_fused_magnetic_pair_verdict(
                fields, pml, grid, sources),
            # THE TWO COMPLEX FUSED PRODUCTS, ADDED 2026-08-30 with the products.
            # Seam verdicts like the one above: never listed in UNION_FAMILIES,
            # because a fused pair bids at no slot and would be double-counted
            # against the arms it absorbs.
            "cuda_complex_fused_magnetic_pair": _cuda_complex_fused_pair_verdict(
                "complex_fused_magnetic_pair",
                "covers_complex_fused_magnetic_pair",
                "cuda_complex_fused_magnetic_pair", fields, pml, grid, sources),
            "cuda_cylindrical_fused_magnetic_pair":
                _cuda_complex_fused_pair_verdict(
                    "cylindrical_fused_magnetic_pair",
                    "covers_cylindrical_fused_magnetic_pair",
                    "cuda_cylindrical_fused_magnetic_pair", fields, pml, grid,
                    sources),
            # THE FIRST ELECTRIC-SEAM PRODUCT, ADDED 2026-08-30 with the product.
            # Also a seam verdict and also outside UNION_FAMILIES; what is new is
            # the SEAM it spans, which the board reads off this block's own
            # ``spans`` rather than assuming.
            "cuda_fused_electric_pair": _cuda_fused_electric_pair_verdict(
                fields, pml, grid, sources),
            # THE TWO COMPLEX ELECTRIC-SEAM PRODUCTS, ADDED 2026-08-30 with them.
            # Same shape as the two complex magnetic columns above; what differs is
            # the seam, which _COMPLEX_PAIR_SEAM supplies per module.
            "cuda_complex_fused_electric_pair": _cuda_complex_fused_pair_verdict(
                "complex_fused_electric_pair",
                "covers_complex_fused_electric_pair",
                "cuda_complex_fused_electric_pair", fields, pml, grid, sources),
            "cuda_cylindrical_fused_electric_pair":
                _cuda_complex_fused_pair_verdict(
                    "cylindrical_fused_electric_pair",
                    "covers_cylindrical_fused_electric_pair",
                    "cuda_cylindrical_fused_electric_pair", fields, pml, grid,
                    sources),
            # THE NO-ABSORBER ELECTRIC PRODUCT, ADDED 2026-08-31 with it. Same shape
            # as the three complex columns above and one difference that is not
            # cosmetic: it is asked with the SEVEN-PATTERN no-absorber record, because
            # its halves are the cuda_complex_no_pml family's and that family's arms
            # bind through that record. _FUSED_PAIR_LICENCE is where that is decided,
            # once, so this row and the three slot rows of the same family cannot be
            # asked two different questions about one arm.
            "cuda_no_pml_complex_fused_electric_pair":
                _cuda_complex_fused_pair_verdict(
                    "no_pml_complex_fused_electric_pair",
                    "covers_no_pml_complex_fused_electric_pair",
                    "cuda_no_pml_complex_fused_electric_pair", fields, pml, grid,
                    sources),
            # THE TWO E->P PRODUCTS, ADDED 2026-09-01 with them. Seam verdicts
            # like the seven above -- never in UNION_FAMILIES -- and the first
            # on a seam with NOTHING between its two consults; the board reads
            # the seam off each block's own ``spans``, which is what lets a new
            # seam arrive as data rather than as a board edit.
            "cuda_fused_polarization_pair": _cuda_fused_polarization_pair_verdict(
                "covers_fused_polarization_pair",
                "cuda_fused_polarization_pair", fields, pml, grid, sources),
            "cuda_no_pml_fused_polarization_pair":
                _cuda_fused_polarization_pair_verdict(
                    "covers_no_pml_fused_polarization_pair",
                    "cuda_no_pml_fused_polarization_pair", fields, pml, grid,
                    sources),
            # THE FIVE RESIDUAL MAGNETIC-SEAM PRODUCTS, ADDED 2026-09-02 with
            # them -- the board's last buildable cells. Seam verdicts like every
            # fused column: never in UNION_FAMILIES. The two complex ones are
            # asked with the licence their families bind through; the three
            # real ones bind no arm and are asked without one.
            "cuda_complex_folded_fused_magnetic_pair":
                _cuda_complex_fused_pair_verdict(
                    "complex_folded_fused_magnetic_pair",
                    "covers_complex_folded_fused_magnetic_pair",
                    "cuda_complex_folded_fused_magnetic_pair", fields, pml,
                    grid, sources),
            "cuda_complex_beta_fused_magnetic_pair":
                _cuda_complex_fused_pair_verdict(
                    "complex_beta_fused_magnetic_pair",
                    "covers_complex_beta_fused_magnetic_pair",
                    "cuda_complex_beta_fused_magnetic_pair", fields, pml,
                    grid, sources),
            "cuda_cylindrical_real_fused_magnetic_pair":
                _cuda_real_fused_pair_verdict(
                    "cylindrical_real_fused_magnetic_pair",
                    "covers_cylindrical_real_fused_magnetic_pair",
                    "cuda_cylindrical_real_fused_magnetic_pair",
                    "step_B -> zero_metal_B -> update_H",
                    fields, pml, grid, sources),
            "cuda_special_kz_fused_magnetic_pair":
                _cuda_real_fused_pair_verdict(
                    "special_kz_fused_magnetic_pair",
                    "covers_special_kz_fused_magnetic_pair",
                    "cuda_special_kz_fused_magnetic_pair",
                    # Five passes since 2026-09-02: the residue round ported
                    # the real pair's fill carry into this weld (its blocked
                    # row is folded), so the string is the module's own new
                    # REPLACES.
                    "step_B -> fill_symmetry_bc_B -> zero_metal_B -> "
                    "fill_folded_far_ghosts_B -> update_H",
                    fields, pml, grid, sources),
            "cuda_bfast_fused_magnetic_pair":
                _cuda_real_fused_pair_verdict(
                    "bfast_fused_magnetic_pair",
                    "covers_bfast_fused_magnetic_pair",
                    "cuda_bfast_fused_magnetic_pair",
                    "step_B -> zero_metal_B -> update_H",
                    fields, pml, grid, sources),
            # THE FIVE ELECTRIC TWINS AND THE THIRD E->P PRODUCT, 2026-09-02
            # (the residue round) -- seam verdicts like every fused column,
            # never in UNION_FAMILIES. The two complex twins and the E->P
            # product are asked with the licence their families bind through
            # (_FUSED_PAIR_LICENCE routes the no-absorber one to the
            # seven-pattern record); the three real twins bind no arm and are
            # asked without one, with their spans read off their own seam
            # strings -- two of which carry the D-side fills.
            "cuda_complex_folded_fused_electric_pair":
                _cuda_complex_fused_pair_verdict(
                    "complex_folded_fused_electric_pair",
                    "covers_complex_folded_fused_electric_pair",
                    "cuda_complex_folded_fused_electric_pair", fields, pml,
                    grid, sources),
            "cuda_complex_beta_fused_electric_pair":
                _cuda_complex_fused_pair_verdict(
                    "complex_beta_fused_electric_pair",
                    "covers_complex_beta_fused_electric_pair",
                    "cuda_complex_beta_fused_electric_pair", fields, pml,
                    grid, sources),
            "cuda_cylindrical_real_fused_electric_pair":
                _cuda_real_fused_pair_verdict(
                    "cylindrical_real_fused_electric_pair",
                    "covers_cylindrical_real_fused_electric_pair",
                    "cuda_cylindrical_real_fused_electric_pair",
                    "step_D -> zero_metal_D -> update_E",
                    fields, pml, grid, sources),
            "cuda_special_kz_fused_electric_pair":
                _cuda_real_fused_pair_verdict(
                    "special_kz_fused_electric_pair",
                    "covers_special_kz_fused_electric_pair",
                    "cuda_special_kz_fused_electric_pair",
                    "step_D -> fill_symmetry_bc_D -> zero_metal_D -> "
                    "fill_folded_far_ghosts_D -> update_E",
                    fields, pml, grid, sources),
            "cuda_bfast_fused_electric_pair":
                _cuda_real_fused_pair_verdict(
                    "bfast_fused_electric_pair",
                    "covers_bfast_fused_electric_pair",
                    "cuda_bfast_fused_electric_pair",
                    "step_D -> zero_metal_D -> update_E",
                    fields, pml, grid, sources),
            # THE H->D WELD, 2026-09-06 -- THE FOURTH SEAM, and the first column on
            # this battery whose seam is neither curl->constitutive nor
            # constitutive->polarization but CONSTITUTIVE->CURL. Its ``seam`` string
            # is therefore ``update_H -> step_D`` and the derived span is exactly
            # those two, which is what ``build_cuda_fusion_matrix.fused_seams_of``
            # resolves to ``h_to_d_seam.SEAM`` and what makes the board file a
            # ``served_by`` verdict per row on that seam.
            #
            # ASKED WITH NO LICENCE, like every real-storage residual pair above: both
            # halves are real-storage families that bind no expansion arm, and asking
            # with one would be asking a different question than the predicate answers.
            #
            # THE COLUMN IS NOT A CLAIM THAT THE PRODUCT RUNS. Its module declares
            # INSTALLABLE = False and the composer refuses it on every configuration;
            # what this column carries is the PREDICATE's answer, which is what the
            # board's served bucket is defined over.
            "cuda_fused_hd_pair":
                _cuda_real_fused_pair_verdict(
                    "fused_hd_pair", "covers_fused_hd_pair",
                    "cuda_fused_hd_pair", "update_H -> step_D",
                    fields, pml, grid, sources),
            # THE TWO CYLINDRICAL H->D PRODUCTS, 2026-09-07 -- the second and third
            # columns on the fourth seam, one per storage. The real m = 0 one is asked
            # like ``cuda_fused_hd_pair`` (no licence: both halves are real-storage
            # families); the complex one is asked with the expansion licence its
            # halves bind through, like ``cuda_cylindrical_fused_magnetic_pair``,
            # and reads its seam off ``_COMPLEX_PAIR_SEAM``. Both modules declare
            # INSTALLABLE = False; each column carries the PREDICATE's answer, which is
            # what the board's served bucket is defined over, and the board records
            # the declaration beside the count.
            "cuda_cylindrical_real_fused_hd_pair":
                _cuda_real_fused_pair_verdict(
                    "cylindrical_real_fused_hd_pair",
                    "covers_cylindrical_real_fused_hd_pair",
                    "cuda_cylindrical_real_fused_hd_pair", "update_H -> step_D",
                    fields, pml, grid, sources),
            "cuda_cylindrical_fused_hd_pair":
                _cuda_complex_fused_pair_verdict(
                    "cylindrical_fused_hd_pair",
                    "covers_cylindrical_fused_hd_pair",
                    "cuda_cylindrical_fused_hd_pair", fields, pml, grid, sources),
            # THE CARTESIAN COMPLEX H->D PRODUCT, 2026-09-07 -- the fourth column on
            # the fourth seam. Asked with the expansion licence its two halves bind
            # through (the four-pattern ``cuda_complex`` record, like
            # ``cuda_complex_fused_magnetic_pair``), and its seam comes from
            # ``_COMPLEX_PAIR_SEAM``. ONE COLUMN, TWO BOARD CELLS: the predicate reads
            # the fold off the grid and asks the certified curl predicate the variant
            # selects, so the plain and folded cells separate through the composer's
            # own ``step_D`` selection rather than through a second column here. The
            # module declares INSTALLABLE = False; this column carries the PREDICATE's
            # answer, which is what the board's served bucket is defined over.
            "cuda_complex_fused_hd_pair":
                _cuda_complex_fused_pair_verdict(
                    "complex_fused_hd_pair",
                    "covers_complex_fused_hd_pair",
                    "cuda_complex_fused_hd_pair", fields, pml, grid, sources),
            # THE TWO BETA COLUMNS ON THE FOURTH SEAM, 2026-09-08. Both products were
            # built and RELEASED on 2026-09-07 (complex_beta lift 4 of 4 driven
            # bit-identical, special_kz 2 of 2, each on BOTH subnormal policies) and
            # then went uncounted for a day because neither this battery nor
            # ``FUSED_PRODUCT_COLUMNS`` named them -- so the six seam-instances they
            # cover read as ``buildable_not_built`` while the welds sat earned. As
            # with the row below, THE COLUMN IS NOT A CLAIM THAT THE PRODUCT RUNS:
            # both modules declare ``INSTALLABLE = False``; what this carries is the
            # PREDICATE's answer, which is what the board's served bucket is over.
            "cuda_complex_beta_fused_hd_pair":
                _cuda_complex_fused_pair_verdict(
                    "complex_beta_fused_hd_pair",
                    "covers_complex_beta_fused_hd_pair",
                    "cuda_complex_beta_fused_hd_pair", fields, pml, grid, sources),
            "cuda_special_kz_fused_hd_pair":
                _cuda_real_fused_pair_verdict(
                    "special_kz_fused_hd_pair",
                    "covers_special_kz_fused_hd_pair",
                    "cuda_special_kz_fused_hd_pair", "update_H -> step_D",
                    fields, pml, grid, sources),
            # THE FIFTH COLUMN ON THE FOURTH SEAM, 2026-09-08 -- the conductive and
            # BFAST curl tails, ONE product over two board cells. It is added here
            # because ``build_cuda_fusion_matrix.FUSED_PRODUCT_COLUMNS`` declared it
            # with the wiring and this battery did not, so ``_fused_verdicts`` refused
            # every argument-free board run BY NAME: 36 columns declared, 35 answered.
            # The board's own note says the only way out is the one taken here -- the
            # battery gains the column and a census is cut after it -- because a board
            # that inferred the missing answer would be reporting a number nobody
            # measured.
            #
            # ASKED WITHOUT A LICENCE, like ``cuda_fused_hd_pair`` above and for the
            # same reason: both variants this product resolves between are
            # real-storage families that bind no expansion arm, so asking with one
            # would be asking a different question than the predicate answers. Its
            # seam string is the module's own ``update_H -> step_D`` (``SEAM`` reads
            # ``H_to_D`` on this family exactly as it does on ``fused_hd_pair``), so
            # the derived span is those two passes and
            # ``build_cuda_fusion_matrix.fused_seams_of`` resolves it to
            # ``h_to_d_seam.SEAM``.
            #
            # ONE COLUMN, TWO CELLS, resolved inside the predicate rather than here:
            # ``covers_conductive_bfast_fused_hd_pair`` reads the variant off the run
            # (``grid.bfast_active`` and ``fields.condfac_for``, the two readers the
            # array path itself uses) and then asks the pair of certified predicates
            # that variant selects, so the conductive and BFAST cells separate through
            # the composer's own ``step_D`` selection and never through a second
            # column. THE COLUMN IS NOT A CLAIM THAT THE PRODUCT RUNS: the module
            # declares ``INSTALLABLE = False``; what this carries is the PREDICATE's
            # answer, which is what the board's served bucket is defined over.
            "cuda_conductive_bfast_fused_hd_pair":
                _cuda_real_fused_pair_verdict(
                    "conductive_bfast_fused_hd_pair",
                    "covers_conductive_bfast_fused_hd_pair",
                    "cuda_conductive_bfast_fused_hd_pair", "update_H -> step_D",
                    fields, pml, grid, sources),
            "cuda_complex_no_pml_fused_polarization_pair":
                _cuda_complex_fused_pair_verdict(
                    "complex_no_pml_fused_polarization_pair",
                    "covers_complex_no_pml_fused_polarization_pair",
                    "cuda_complex_no_pml_fused_polarization_pair", fields, pml,
                    grid, sources),
            # THE COMPLEX NO-ABSORBER THREE-SLOT WELD, 2026-09-04: the third
            # three-slot column and the first complex one. It occupies BOTH sides
            # of the shared ``update_E`` slot on the four TestLoadDump 3-D rows
            # where ``cuda_no_pml_complex_fused_electric_pair`` and
            # ``cuda_complex_no_pml_fused_polarization_pair`` traded that slot one
            # for one. Asked with the seven-pattern no-absorber licence like its
            # two halves; its spans come from _COMPLEX_PAIR_SEAM's row.
            "cuda_three_slot_complex_no_pml_dispersive_weld":
                _cuda_complex_fused_pair_verdict(
                    "complex_no_pml_three_slot_dispersive_weld",
                    "covers_three_slot_complex_no_pml_dispersive_weld",
                    "cuda_three_slot_complex_no_pml_dispersive_weld", fields, pml,
                    grid, sources),
            # THE TWO STENCIL WELDS, 2026-09-02 -- the first fused columns on this
            # battery whose board cells were recorded UNBUILDABLE rather than
            # unbuilt. Seam verdicts like every fused column: never in
            # UNION_FAMILIES, because a fused pair bids at no slot and would be
            # double-counted against the arms it absorbs. Both bind no expansion
            # arm (their halves are real-storage families), so both are asked
            # WITHOUT a licence, and their spans are read off their own seam
            # strings -- the folded one carries all three in-seam passes.
            "cuda_offdiag_fused_electric_pair":
                _cuda_real_fused_pair_verdict(
                    "offdiag_fused_electric_pair",
                    "covers_offdiag_fused_electric_pair",
                    "cuda_offdiag_fused_electric_pair",
                    "step_D -> zero_metal_D -> update_E",
                    fields, pml, grid, sources),
            "cuda_folded_offdiag_fused_electric_pair":
                _cuda_real_fused_pair_verdict(
                    "folded_offdiag_fused_electric_pair",
                    "covers_folded_offdiag_fused_electric_pair",
                    "cuda_folded_offdiag_fused_electric_pair",
                    "step_D -> fill_symmetry_bc_D -> zero_metal_D -> "
                    "fill_folded_far_ghosts_D -> update_E",
                    fields, pml, grid, sources),
            # THE LAST UNSERVED E->P CELL, 2026-09-02. A seam verdict like every
            # fused column -- never in UNION_FAMILIES -- and asked WITHOUT a
            # licence, because both halves are real-storage families that bind no
            # expansion arm. Its span is read off its own seam string, which is
            # the driver's two adjacent consults with nothing between them.
            "cuda_dispersive_offdiag_fused_polarization_pair":
                _cuda_real_fused_pair_verdict(
                    "dispersive_offdiag_fused_polarization_pair",
                    "covers_dispersive_offdiag_fused_polarization_pair",
                    "cuda_dispersive_offdiag_fused_polarization_pair",
                    "update_E -> update_P",
                    fields, pml, grid, sources),
            # THE DISPERSIVE ELECTRIC PAIR, 2026-09-02, and the column that prices
            # the board's largest cell that no product occupied: D_to_E
            # (cuda_curl/PML, cuda_dispersive/dispersive), 7 corpus rows. A seam
            # verdict like every fused column -- never in UNION_FAMILIES, because a
            # fused pair bids at no slot and would be double-counted against the arms
            # it absorbs. Asked WITHOUT a licence: both halves are real-storage
            # families that bind no expansion arm. Its span is its own REPLACES, all
            # five passes, because four of the seven rows it exists for are FOLDED and
            # the two mirror fills are carried rather than refused -- which is what
            # separates this product from BOTH sibling backends' dispersive pairs.
            "cuda_dispersive_fused_electric_pair":
                _cuda_real_fused_pair_verdict(
                    "dispersive_fused_electric_pair",
                    "covers_dispersive_fused_electric_pair",
                    "cuda_dispersive_fused_electric_pair",
                    "step_D -> fill_symmetry_bc_D -> zero_metal_D -> "
                    "fill_folded_far_ghosts_D -> update_E",
                    fields, pml, grid, sources),
            # THE TWO CONDUCTIVE-CURL ELECTRIC PAIRS, 2026-09-02, which price the
            # last POINTWISE-BUILDABLE D_to_E cells this board had:
            #
            #   (cuda_conductive/conductive,  cuda_no_pml_dispersive)  2 rows
            #   (cuda_no_pml_curl/no-PML curl, cuda_no_pml_dispersive) 1 row
            #   (cuda_conductive/conductive,  cuda_constitutive/ordinary) 1 row
            #
            # Seam verdicts like every fused column -- never in UNION_FAMILIES,
            # because a fused pair bids at no slot and would be double-counted
            # against the arms it absorbs. Both are asked WITHOUT a licence: every
            # half involved is a real-storage family that binds no expansion arm.
            #
            # THEIR SPANS ARE THREE PASSES, NOT FIVE, and that is each module's own
            # REPLACES rather than a shortened copy of the row above's: both REFUSE a
            # mirror plane by name, so ``fill_symmetry_bc_D`` and
            # ``fill_folded_far_ghosts_D`` are INERT on every row they admit (they
            # return at their first line without a fold) and naming them here would
            # claim two passes the launch does not perform.
            "cuda_no_pml_dispersive_fused_electric_pair":
                _cuda_real_fused_pair_verdict(
                    "no_pml_dispersive_fused_electric_pair",
                    "covers_no_pml_dispersive_fused_electric_pair",
                    "cuda_no_pml_dispersive_fused_electric_pair",
                    "step_D -> zero_metal_D -> update_E",
                    fields, pml, grid, sources),
            "cuda_conductive_fused_electric_pair":
                _cuda_real_fused_pair_verdict(
                    "conductive_fused_electric_pair",
                    "covers_conductive_fused_electric_pair",
                    "cuda_conductive_fused_electric_pair",
                    "step_D -> zero_metal_D -> update_E",
                    fields, pml, grid, sources),
            # THE TWO THREE-SLOT WELDS, 2026-09-02, and the FIRST columns on this
            # census whose ``spans`` names three slots rather than two. They are the
            # only products that occupy BOTH sides of the shared ``update_E`` slot,
            # which is the whole reason they exist: on the ten rows below, a D->E
            # product and an E->P product trade that slot one for one and the board's
            # ``_loses_the_shared_slot`` withholds the D side.
            #
            #   cuda_three_slot_dispersive_weld           D_to_E + E_to_P, 7 rows
            #   cuda_three_slot_no_pml_dispersive_weld    D_to_E + E_to_P, 3 rows
            #
            # THE SPANS ARE PASSED EXPLICITLY, and that is not a convenience: the
            # first-and-last derivation would record ``["step_D", "update_P"]``,
            # which is no seam, and the board would refuse the column rather than
            # credit it at both. Seam verdicts like every fused column -- never in
            # UNION_FAMILIES -- and asked WITHOUT a licence, because all four halves
            # are real-storage families that bind no expansion arm.
            "cuda_three_slot_dispersive_weld":
                _cuda_real_fused_pair_verdict(
                    "three_slot_dispersive_weld",
                    "covers_three_slot_dispersive_weld",
                    "cuda_three_slot_dispersive_weld",
                    "step_D -> fill_symmetry_bc_D -> zero_metal_D -> "
                    "fill_folded_far_ghosts_D -> update_E -> update_P",
                    fields, pml, grid, sources,
                    spans=("step_D", "update_E", "update_P")),
            "cuda_three_slot_no_pml_dispersive_weld":
                _cuda_real_fused_pair_verdict(
                    "no_pml_three_slot_dispersive_weld",
                    "covers_no_pml_three_slot_dispersive_weld",
                    "cuda_three_slot_no_pml_dispersive_weld",
                    "step_D -> zero_metal_D -> update_E -> update_P",
                    fields, pml, grid, sources,
                    spans=("step_D", "update_E", "update_P")),
            # THE TWO COMPLEX STENCIL WELDS, 2026-09-02, and the columns that price
            # the LAST two cells this board recorded UNBUILDABLE:
            #
            #   (cuda_complex_folded/folded complex,
            #    cuda_complex_offdiag/complex off-diagonal PML)     3 rows
            #   (cuda_complex_no_pml/complex no-PML curl,
            #    cuda_complex_offdiag/complex off-diagonal no-PML)  2 rows
            #
            # Seam verdicts like every fused column -- never in UNION_FAMILIES,
            # because a fused pair bids at no slot and would be double-counted
            # against the arms it absorbs. THEY ARE THE ONLY COLUMNS ASKED WITH TWO
            # LICENCES: each product's halves sit in different licence families, and
            # the row names both records so a reader can see which artifact licensed
            # which half.
            "cuda_folded_complex_offdiag_fused_electric_pair":
                _cuda_complex_offdiag_fused_pair_verdict(
                    "folded_complex_offdiag_fused_electric_pair",
                    "covers_folded_complex_offdiag_fused_electric_pair",
                    "cuda_folded_complex_offdiag_fused_electric_pair",
                    _complex_license, COMPLEX_PROBE_RECORD,
                    fields, pml, grid, sources),
            "cuda_complex_no_pml_offdiag_fused_electric_pair":
                _cuda_complex_offdiag_fused_pair_verdict(
                    "complex_no_pml_offdiag_fused_electric_pair",
                    "covers_complex_no_pml_offdiag_fused_electric_pair",
                    "cuda_complex_no_pml_offdiag_fused_electric_pair",
                    _complex_no_pml_license, COMPLEX_NO_PML_PROBE_RECORD,
                    fields, pml, grid, sources),
            "polarization": polarization, "configuration": configuration,
            # Two things under one key. ``live`` is the ENGINE fact
            # build_cuda_fusion_matrix.py used to JOIN out of the Metal census --
            # same key, same shape, so the consumer reads it here. The rest is the
            # hand-CUDA COMPOSER's answer: which arm won each slot, and the named
            # refusal on each slot it did not fill. Merged rather than nested so
            # ``plan_step.selected`` means on this census what it means on the Metal
            # one; the key sets are disjoint and asserted so below.
            "plan_step": _merge_plan_step(
                plan_step_live(fields, pml, sources),
                plan_step_selected(fields, pml, grid)),
            "specialization": specialization_tuple(fields, pml, grid),
            "probe_unused": True}
