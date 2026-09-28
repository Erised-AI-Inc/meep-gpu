"""Call every LIVE Triton coverage predicate on one lifted (fields, pml) pair.

This is a MEASUREMENT, not an argument. Nothing here re-derives a clause, restates a
refusal, or infers a verdict: each entry calls the shipped function and records what it
returned, verbatim.

Two clauses cannot be satisfied on a laptop and are FACTORED OUT rather than dropped:

* ``array module is 'numpy', not cupy`` — every predicate's clause 1. Emitted by
  ``coverage.py:141``/``:528`` and the identical line in nine sibling modules; the
  string is byte-identical in all of them, which is what makes the filter exact.
* the complex-multiply EXPANSION probe (``complex_fields._expansion_reasons``) — a
  MEASURED device fact. Not guessed here either: the certified artifact from the
  recut jobs is loaded off disk and passed in as ``probe=``, so the clause is
  satisfied by the same record the gate ran under, and the row says which file.

Every OTHER refusal is reported as returned.

2026-08-14 ROUND — ALL NINE FAMILIES ARE WIRED. Two things changed here and nothing
else did:

* the four families that were absent from the 2026-08-13 battery are now entries
  like the other five (``folded_complex``, ``folded_offdiag``, ``cylindrical_complex``,
  ``no_pml_null``). The nine previous entry blocks are UNCHANGED, byte for byte, so
  the per-sub-step census stays commensurable with 482/759 and 576/759.
* :func:`plan_step_admission` runs the SHIPPED :func:`launch.plan_step` on the same
  lifted objects and records, per slot, every arm it consulted and that arm's verdict.
  ``plan_step`` is not re-derived here: its own :func:`launch._select_slot` is wrapped
  to report the admitters it already returns, and the ONE CuPy clause is factored at
  the forwarder layer with exactly the filter :func:`strip_backend` uses.

:data:`RESERVED_FAMILIES` is empty because no family is held back any more.

2026-08-16 ROUND — THE RESIDUAL-GROUP ADMISSIONS ARE WIRED. One family entry is
added, ``residual_wired``, and nothing else here changed:

* four predicate families, six entry points, NO new kernel. Each restates a
  shipped clause set with the one over-broad clause inverted so a CERTIFIED body
  may step a configuration it was refused for
  (results/residual_closure_2026-08-15/device/newpred/new_predicates.json).
* ``update_P`` gains a SECOND family per (state, component):
  ``no_pml_ade_update_p_coverage``. The roll-up's ``update_P`` rule is therefore
  "SOME family admits every driven component of every state", which is
  ``launch.plan_step``'s own rule for that slot.

The thirteen entry blocks above it are UNCHANGED, byte for byte, so the per-sub-step
census stays commensurable with 482/759, 576/759 and 702/759.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

#: The kernels package these predicates measure. ``measure_predicate_coverage`` digests
#: it (plus the shared engine files) into every census row's ``subject_manifest_sha256``.
#: Declared here rather than inferred from imports: a battery may import a sibling
#: package to compare against it without measuring it.
SUBJECT_PACKAGE = "triton_kernels"

#: The one clause a NumPy laptop cannot satisfy, byte-identical across all ten
#: modules that emit it. ``fused_pair_coverage`` prefixes its halves' reasons
#: ("curl half: ...") so the test is a substring test, anchored on both ends.
BACKEND_CLAUSE_HEAD = "array module is "
BACKEND_CLAUSE_TAIL = ", not cupy"

#: Families whose predicates exist but are NOT integrated this round. Empty as of
#: 2026-08-14: all nine are wired into ``launch.plan_step``.
RESERVED_FAMILIES: Tuple[str, ...] = ()

#: The families measured, in the order they were certified. The first five are the
#: ones the 576/759 number was taken over; the last four are this round's.
FIVE_WIRED: Tuple[str, ...] = ("complex", "special_kz", "nonlinear", "offdiag", "bfast")
FOUR_NEW: Tuple[str, ...] = ("folded_complex", "folded_offdiag", "cylindrical_complex",
                             "no_pml_null")
#: The residual-group ADMISSIONS wired on 2026-08-16 — the delta 702/759 is
#: measured against. One label, four predicate families, no new kernel.
RESIDUAL_WIRED: Tuple[str, ...] = ("residual_wired",)

#: The four sub-steps a whole timestep is composed of, plus the polarization pass.
SUB_STEPS: Tuple[str, ...] = ("step_B", "step_D", "update_H", "update_E")

#: The UNIFIED expansion record, repository-relative. ``complex_fused_ade_chain``'s
#: ADE half needs ``c8_mul_python_float_field_left`` on top of the four base
#: patterns, and NO per-family probe carries it — see the seam block below for the
#: measurement, and drive_triton_weld_gates.py:255-265 for the same fact stated by
#: the family's own gate. Named here rather than passed in: this is not a choice the
#: caller gets to make, it is the artifact that family's released gate ran under.
UNIFIED_EXPANSION_RECORD = (
    "parity/meep_gpu/results/unified_expansion_2026-08-27/keep/gate.json")


def _load_unified_expansion_record() -> Optional[Dict[str, Any]]:
    """The unified record off disk, or ``None`` — which is itself a refusal.

    A MISSING ARTIFACT IS NOT A PASS. ``complex_fused_ade_chain_coverage(probe=None)``
    falls through to ``load_expansion_probe()``, whose answer depends on whatever the
    process happens to have installed — so handing it ``None`` silently would make the
    verdict a property of the host rather than of the artifact. The row records
    ``expansion_probe_loaded`` so a reader can tell an artifact that refused from an
    artifact that was never there.
    """
    import json  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    path = Path(__file__).resolve().parents[2] / UNIFIED_EXPANSION_RECORD
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except BaseException:  # noqa: BLE001 - an unreadable artifact is not a licence
        return None


def is_backend_clause(reason: str) -> bool:
    return BACKEND_CLAUSE_HEAD in reason and reason.endswith(BACKEND_CLAUSE_TAIL)


def strip_backend(reasons: Tuple[str, ...]) -> List[str]:
    """The refusal list with ONLY the CuPy-backend clause removed."""
    return [reason for reason in reasons if not is_backend_clause(reason)]


def _verdict(call: Callable[[], Any]) -> Dict[str, Any]:
    """Call one predicate; a raise is a refusal WITH ITS REASON, never a crash."""
    try:
        coverage = call()
    except BaseException as exc:  # noqa: BLE001 - an unanswerable predicate is a refusal
        return {"covered": False, "covered_modulo_backend": False,
                "reasons": [f"PREDICATE RAISED {type(exc).__name__}: {exc}"[:400]],
                "raised": True}
    reasons = tuple(getattr(coverage, "reasons", ()) or ())
    residual = strip_backend(reasons)
    return {"covered": bool(getattr(coverage, "covered", False)),
            "covered_modulo_backend": not residual,
            "reasons": [str(reason)[:400] for reason in reasons],
            "residual_reasons": [str(reason)[:400] for reason in residual],
            "raised": False}


def evaluate(driver: Any, probe: Any) -> Dict[str, Any]:
    """Every predicate, on the real lifted objects. Returns a JSON-safe record."""
    from meep_gpu.triton_kernels import (  # noqa: PLC0415
        ade_update_p_coverage,
        conductive_pml_curl_coverage,
        constitutive_coverage,
        cylindrical_constitutive_coverage,
        cylindrical_curl_coverage,
        dispersive_constitutive_coverage,
        dispersive_fused_pair_coverage,
        folded_constitutive_coverage,
        fused_ade_state_coverage,
        fused_pair_coverage,
        mirror_ghost_fill_coverage,
        plain_curl_coverage,
        pml_curl_coverage,
    )
    from meep_gpu.triton_kernels.bfast_curl import (  # noqa: PLC0415
        bfast_pml_curl_coverage,
        bfast_run_constitutive_coverage,
    )
    from meep_gpu.triton_kernels.complex_fields import (  # noqa: PLC0415
        complex_constitutive_coverage,
        complex_pml_curl_coverage,
    )
    from meep_gpu.triton_kernels.nonlinear_update_e import (  # noqa: PLC0415
        nonlinear_constitutive_coverage,
    )
    from meep_gpu.triton_kernels.offdiag_update_e import (  # noqa: PLC0415
        offdiag_constitutive_coverage,
    )
    from meep_gpu.triton_kernels.special_kz import (  # noqa: PLC0415
        beta_bloch_pml_curl_coverage,
        beta_pml_curl_coverage,
        beta_run_complex_constitutive_coverage,
        beta_run_constitutive_coverage,
    )
    from meep_gpu.triton_kernels.symmetry import (  # noqa: PLC0415
        folded_composition_curl_coverage,
    )

    fields = driver.fields
    pml = driver.pml
    grid = driver.grid
    sources = getattr(driver, "_sources", None)
    states = tuple(getattr(fields, "polarizations", ()) or ())

    entries: List[Tuple[str, str, str, Callable[[], Any]]] = []

    def add(family: str, label: str, sub_step: str, call: Callable[[], Any]) -> None:
        entries.append((family, label, sub_step, call))

    # --- SHIPPED: the set plan_step actually consults ------------------------------
    for name in ("step_B", "step_D"):
        add("shipped", "pml_curl", name,
            lambda n=name: pml_curl_coverage(fields, pml, n))
        add("shipped", "conductive_pml_curl", name,
            lambda n=name: conductive_pml_curl_coverage(fields, pml, n))
        add("shipped", "plain_curl", name,
            lambda n=name: plain_curl_coverage(fields, pml, n))
        add("shipped", "folded_composition_curl", name,
            lambda n=name: folded_composition_curl_coverage(fields, pml, n))
        add("shipped", "cylindrical_curl", name,
            lambda: cylindrical_curl_coverage(fields, pml))
    for side, sub_step in (("H", "update_H"), ("E", "update_E")):
        add("shipped", "constitutive", sub_step,
            lambda s=side: constitutive_coverage(fields, pml, s))
        add("shipped", "folded_constitutive", sub_step,
            lambda s=side: folded_constitutive_coverage(fields, pml, s))
        add("shipped", "cylindrical_constitutive", sub_step,
            lambda s=side: cylindrical_constitutive_coverage(fields, pml, s))
    add("shipped", "dispersive_constitutive", "update_E",
        lambda: dispersive_constitutive_coverage(fields, pml))

    # --- THE FIVE CERTIFIED FAMILIES: standalone, unreachable from plan_step -------
    for name in ("step_B", "step_D"):
        add("complex", "complex_pml_curl", name,
            lambda n=name: complex_pml_curl_coverage(fields, pml, n, probe))
        add("special_kz", "beta_pml_curl", name,
            lambda n=name: beta_pml_curl_coverage(fields, pml, n))
        add("special_kz", "beta_bloch_pml_curl", name,
            lambda n=name: beta_bloch_pml_curl_coverage(fields, pml, n, probe))
        add("bfast", "bfast_pml_curl", name,
            lambda n=name: bfast_pml_curl_coverage(fields, pml, n))
    for side, sub_step in (("H", "update_H"), ("E", "update_E")):
        add("complex", "complex_constitutive", sub_step,
            lambda s=side: complex_constitutive_coverage(fields, pml, s, probe))
        add("special_kz", "beta_run_constitutive", sub_step,
            lambda s=side: beta_run_constitutive_coverage(fields, pml, s))
        add("special_kz", "beta_run_complex_constitutive", sub_step,
            lambda s=side: beta_run_complex_constitutive_coverage(fields, pml, s, probe))
        add("bfast", "bfast_run_constitutive", sub_step,
            lambda s=side: bfast_run_constitutive_coverage(fields, pml, s))
    add("nonlinear", "nonlinear_constitutive", "update_E",
        lambda: nonlinear_constitutive_coverage(fields, pml))
    add("offdiag", "offdiag_constitutive", "update_E",
        lambda: offdiag_constitutive_coverage(fields, pml))

    # --- THE FOUR FAMILIES WIRED ON 2026-08-14 -------------------------------------
    # Called through ``launch``'s forwarders — the same functions ``plan_step``'s arms
    # call — so an entry here and its arm ask one function, not two spellings of one.
    # The COMPOSITION verdict is taken where ``plan_step`` takes it (K1's curl, the
    # folded off-diagonal update_E): the standalone verdict admits an unfolded grid to
    # prove reduction, which is an equivalence claim, not a routing one.
    from meep_gpu.triton_kernels import launch  # noqa: PLC0415

    for name in ("step_B", "step_D"):
        add("folded_complex", "folded_complex_composition_curl", name,
            lambda n=name: launch.folded_complex_composition_curl_coverage(
                fields, pml, n, probe))
        add("folded_complex", "folded_beta_pml_curl", name,
            lambda n=name: launch.folded_beta_pml_curl_coverage(fields, pml, n))
        add("folded_complex", "folded_beta_bloch_pml_curl", name,
            lambda n=name: launch.folded_beta_bloch_pml_curl_coverage(
                fields, pml, n, probe))
        add("cylindrical_complex", "cylindrical_complex_curl", name,
            lambda n=name: launch.cylindrical_complex_curl_coverage(
                fields, pml, n, probe))
    for side, sub_step in (("H", "update_H"), ("E", "update_E")):
        add("folded_complex", "folded_complex_constitutive", sub_step,
            lambda s=side: launch.folded_complex_constitutive_coverage(
                fields, pml, s, probe))
        add("folded_complex", "folded_beta_run_constitutive", sub_step,
            lambda s=side: launch.folded_beta_run_constitutive_coverage(
                fields, pml, s))
        add("cylindrical_complex", "cylindrical_complex_constitutive", sub_step,
            lambda s=side: launch.cylindrical_complex_constitutive_coverage(
                fields, pml, s, probe))
        add("no_pml_null", "null_constitutive", sub_step,
            lambda s=side: launch.null_constitutive_coverage(fields, pml, s))
    add("folded_offdiag", "folded_offdiag_composition", "update_E",
        lambda: launch.folded_offdiag_composition_coverage(fields, pml))

    # --- THE RESIDUAL-GROUP ADMISSIONS, WIRED 2026-08-16 --------------------------
    # Called through ``launch``'s forwarders — the same functions ``plan_step``'s
    # arms call — and asked ONLY at the sub-steps their arms serve. The two
    # constitutive predicates take a ``side`` and refuse ``side='E'`` BY NAME
    # (the Pade factor's slot; the row product's slot), so neither is asked at
    # update_E: an arm that is not consulted there must not be counted there.
    for name in ("step_B", "step_D"):
        add("residual_wired", "nonlinear_run_pml_curl", name,
            lambda n=name: launch.nonlinear_run_pml_curl_coverage(fields, pml, n))
        add("residual_wired", "folded_complex_offdiag_pml_curl", name,
            lambda n=name: launch.folded_complex_offdiag_pml_curl_coverage(
                fields, pml, n, probe))
    add("residual_wired", "nonlinear_run_constitutive", "update_H",
        lambda: launch.nonlinear_run_constitutive_coverage(fields, pml, "H"))
    add("residual_wired", "folded_complex_offdiag_constitutive", "update_H",
        lambda: launch.folded_complex_offdiag_constitutive_coverage(
            fields, pml, "H", probe))
    add("residual_wired", "folded_dispersive_constitutive", "update_E",
        lambda: launch.folded_dispersive_constitutive_coverage(fields, pml))

    # --- seam and cross-sub-step predicates, recorded but NOT rolled into a sub-step
    seam: Dict[str, Any] = {}
    for family_name in ("B", "D"):
        seam[f"mirror_ghost_fill_{family_name}"] = _verdict(
            lambda f=family_name: mirror_ghost_fill_coverage(fields, f))
        seam[f"fused_pair_{family_name}"] = _verdict(
            lambda p=family_name: fused_pair_coverage(fields, pml, p, sources))
        seam[f"folded_complex_fill_{family_name}"] = _verdict(
            lambda f=family_name: launch.folded_mirror_ghost_fill_complex_coverage(
                fields, f, probe))
    seam["dispersive_fused_pair_D"] = _verdict(
        lambda: dispersive_fused_pair_coverage(fields, pml, sources))

    # --- 2026-08-30: THE SEVEN FUSED PRODUCTS THE BOARD USED TO TRANSCRIBE --------
    #
    # WHY THESE ARE HERE, AND WHY IT IS A CORRECTION RATHER THAN AN ADDITION. The
    # fusion board scored these products from hand-written CLAUSE LADDERS that
    # restated each predicate's conjunction from the ``configuration`` block. A
    # transcription cannot notice when the shipped predicate changes, and one of
    # them did not: ``folded_fused_pair_coverage`` retired its blanket
    # ``no_folded_periodic_axis`` refusal on 2026-08-21 when the far carry landed
    # (folded_fused_pair.py:1046-1110, gate run_farcarryD5), and the board's mirror
    # went on refusing rows the product admits. The magnetic twin's mirror WAS
    # updated when its own carry landed; the electric one never was.
    #
    # Recording the shipped verdict is the structural fix: a board that reads THIS
    # cannot drift from the product, because it is not holding a second copy of the
    # answer. Every entry calls the function the launcher's own installer calls,
    # with the run's real sources, exactly as the four entries above already do.
    #
    # The two ADE CHAINS are the same move for the E->P seam, where the board
    # priced two BUILT, GATED products at zero for want of an admission census —
    # ``fused_ade_chain`` reads live polarization objects (``state.driven()``, the
    # pole partition, ``fields.drive_field``) that no configuration block carries,
    # so it could never have been transcribed at all. It is asked once PER ARM
    # because ``ARMS`` is a predicate-level split (one kernel, three admissions),
    # and the arm is part of what the answer is about.
    from meep_gpu.triton_kernels.folded_fused_pair import (  # noqa: PLC0415
        folded_fused_pair_coverage)
    from meep_gpu.triton_kernels.folded_fused_magnetic_pair import (  # noqa: PLC0415
        folded_fused_magnetic_pair_coverage)
    from meep_gpu.triton_kernels.complex_fused_magnetic_pair import (  # noqa: PLC0415
        complex_fused_magnetic_pair_coverage)
    from meep_gpu.triton_kernels.complex_fused_electric_pair import (  # noqa: PLC0415
        complex_fused_electric_pair_coverage)
    from meep_gpu.triton_kernels.cylindrical_fused_electric_pair import (  # noqa: PLC0415
        cylindrical_fused_electric_pair_coverage)
    from meep_gpu.triton_kernels.complex_conductive_fused_pair import (  # noqa: PLC0415
        complex_conductive_fused_pair_coverage)
    from meep_gpu.triton_kernels.folded_dispersive_fused_pair import (  # noqa: PLC0415
        folded_dispersive_fused_pair_coverage)
    from meep_gpu.triton_kernels.cylindrical_real_fused_electric_pair import (  # noqa: PLC0415
        cylindrical_real_fused_electric_pair_coverage)
    from meep_gpu.triton_kernels.folded_complex_fused_magnetic_pair import (  # noqa: PLC0415
        folded_complex_fused_magnetic_pair_coverage)
    from meep_gpu.triton_kernels.folded_complex_fused_pair import (  # noqa: PLC0415
        folded_complex_fused_pair_coverage)
    from meep_gpu.triton_kernels.beta_fused_electric_pair import (  # noqa: PLC0415
        beta_fused_electric_pair_coverage)
    from meep_gpu.triton_kernels.bfast_fused_electric_pair import (  # noqa: PLC0415
        bfast_fused_electric_pair_coverage)
    from meep_gpu.triton_kernels.conductive_fused_electric_pair import (  # noqa: PLC0415, E501
        conductive_fused_electric_pair_coverage)
    from meep_gpu.triton_kernels.no_pml_fused_electric_pair import (  # noqa: PLC0415, E501
        no_pml_fused_electric_pair_coverage)
    from meep_gpu.triton_kernels.offdiag_fused_electric_pair import (  # noqa: PLC0415, E501
        offdiag_fused_electric_pair_coverage)
    from meep_gpu.triton_kernels.folded_offdiag_fused_electric_pair import (  # noqa: PLC0415, E501
        folded_offdiag_fused_electric_pair_coverage)
    from meep_gpu.triton_kernels.folded_beta_fused_electric_pair import (  # noqa: PLC0415, E501
        folded_beta_fused_electric_pair_coverage)
    from meep_gpu.triton_kernels.folded_beta_fused_magnetic_pair import (  # noqa: PLC0415, E501
        folded_beta_fused_magnetic_pair_coverage)
    from meep_gpu.triton_kernels.complex_beta_fused_electric_pair import (  # noqa: PLC0415, E501
        complex_beta_fused_electric_pair_coverage)
    from meep_gpu.triton_kernels.complex_beta_fused_magnetic_pair import (  # noqa: PLC0415, E501
        complex_beta_fused_magnetic_pair_coverage)
    from meep_gpu.triton_kernels.folded_beta_complex_fused_pair import (  # noqa: PLC0415, E501
        folded_beta_complex_fused_pair_coverage)
    from meep_gpu.triton_kernels.folded_beta_complex_fused_magnetic_pair import (  # noqa: PLC0415, E501
        folded_beta_complex_fused_magnetic_pair_coverage)
    from meep_gpu.triton_kernels.fused_ade_chain import (  # noqa: PLC0415
        ARMS as FUSED_ADE_CHAIN_ARMS, fused_ade_chain_coverage)
    from meep_gpu.triton_kernels.complex_fused_ade_chain import (  # noqa: PLC0415
        complex_fused_ade_chain_coverage)
    # THE MODULE, NOT THE PACKAGE. ``meep_gpu.triton_kernels.__init__`` does not
    # export this family (it is deferred — launch.py never imports it), and a
    # package-level import would be an ImportError on a tree where the deferral is
    # working exactly as designed.
    from meep_gpu.triton_kernels.folded_offdiag_fused_ade_chain import (  # noqa: PLC0415, E501
        folded_offdiag_fused_ade_chain_coverage)

    unified = _load_unified_expansion_record()

    seam["folded_fused_pair_D"] = _verdict(
        lambda: folded_fused_pair_coverage(fields, pml, sources))
    seam["folded_fused_magnetic_pair_B"] = _verdict(
        lambda: folded_fused_magnetic_pair_coverage(fields, pml, sources))
    seam["complex_fused_magnetic_pair_B"] = _verdict(
        lambda: complex_fused_magnetic_pair_coverage(fields, pml, sources, probe))
    # THE ELECTRIC TWIN, added 2026-08-30 in the same round that built it. Asked
    # ONCE, under the census probe, for the reason the magnetic twin is: its
    # PRODUCT_PROBE_PATTERNS is the BASE four -- the inv_eps multiply its SCALE=1 arm
    # adds is `_mul_field_left`, an orientation the curl half already launches -- so
    # unlike `folded_complex_fused_magnetic_pair` it needs no unified-record swap.
    # Its verdict is what decides 16 D->E seam-instances, and it is READ rather than
    # transcribed: the source clause it turns on consults deposit_repair.repairable
    # per run, which no configuration block carries.
    seam["complex_fused_electric_pair_D"] = _verdict(
        lambda: complex_fused_electric_pair_coverage(fields, pml, sources, probe))
    # THE Dcyl ELECTRIC TWIN, added 2026-08-30 in the same round that gated it.
    # Asked ONCE, under the census probe, for the reason the Cartesian twin is: its
    # PRODUCT_PROBE_PATTERNS is the BASE four — the cylindrical-only orientations
    # are arm-degenerate (their coefficient's real word is a zero) and the E-side
    # `inv_eps` multiply is `_mul_field_left`, which the recurrence above it already
    # launches six times — so it needs no unified-record swap.
    #
    # THERE IS NO LADDER FOR IT AND THERE NEVER WILL BE. The clause that decides
    # ALL SIXTEEN of its rows consults `deposit_repair.repairable` on the live run,
    # which no configuration block carries: every Dcyl row declares an electric
    # source, so a transcription could only have guessed, and guessing False would
    # have priced the largest cell on the board at zero.
    seam["cylindrical_fused_electric_pair_D"] = _verdict(
        lambda: cylindrical_fused_electric_pair_coverage(fields, pml, sources,
                                                         probe))
    # THE COMPLEX CONDUCTIVE no-PML D->E PAIR, added in the same round. Its cell is
    # the four `TestLoadDump.test_load_dump_*_3d` rows, and unlike the two pairs
    # above it carries NO deposit repair — every one of those rows declares a
    # MAGNETIC source, injected in the other seam — so its source clause costs it
    # nothing. Recorded rather than transcribed for the same structural reason: a
    # board that holds a second copy of a predicate's answer is a board that can
    # drift from it.
    seam["complex_conductive_fused_pair_D"] = _verdict(
        lambda: complex_conductive_fused_pair_coverage(fields, pml, sources,
                                                       probe))
    # THE FOLDED DISPERSIVE D->E PAIR, added 2026-08-31 in the same round that built
    # it. Its cell is the four `TestLoadDump.test_load_dump_*_2d` rows — the largest
    # remaining unbuilt cell on this board — and like the two D->E pairs above it
    # carries the deposit repair, because every one of those rows declares an
    # ELECTRIC source. THERE IS NO LADDER FOR IT AND THERE NEVER WILL BE: the clause
    # that decides all four consults `deposit_repair.repairable` on the live run,
    # which no configuration block carries, and its E half reads live polarization
    # objects the way `fused_ade_chain` does. A transcription could only have
    # guessed, and guessing False would price the cell at zero.
    #
    # NO PROBE ARGUMENT: this family is REAL-STORAGE throughout — its curl half is
    # `symmetry.pml_curl_step_folded` and its constitutive half the real dispersive
    # `update_E` — so it launches no complex multiply and needs no expansion licence.
    seam["folded_dispersive_fused_pair_D"] = _verdict(
        lambda: folded_dispersive_fused_pair_coverage(fields, pml, sources))
    # THE Dcyl m = 0 D->E PAIR, added 2026-08-31 in the same round that built it. Its
    # cell is the three rows TestAdjointSolver.test_adjoint_solver_cyl_n2f_fields_
    # {0_0,1_0} and TestPMLCylindrical.test_pml_cyl_0_0_0 — the same three its
    # MAGNETIC twin serves, on the other seam. Like every D->E pair on this board it
    # carries the deposit repair, and here too the flag IS the product: all three
    # rows declare an electric source, so the same module with the flag at False
    # would serve nothing at all. Recorded rather than transcribed for that reason —
    # the clause consults `deposit_repair.repairable` on the live run.
    #
    # NO PROBE ARGUMENT: this family is REAL storage at m = 0 throughout.
    seam["cylindrical_real_fused_electric_pair_D"] = _verdict(
        lambda: cylindrical_real_fused_electric_pair_coverage(fields, pml, sources))
    # ASKED TWICE, UNDER TWO NAMED ARTIFACTS, for the reason the complex ADE chain
    # below is. MEASURED 2026-08-30 on a folded complex grid: this product's
    # PRODUCT_PROBE_PATTERNS adds `c8_mul_c8_parity_coefficient_left` on top of the
    # four base ones, and no per-family probe carries it -- under this census's probe
    # the fill half refuses ("does not license an EXPANSION over the EXTENDED
    # patterns") and under results/unified_expansion_2026-08-27/keep/gate.json it
    # admits. Its own released gate is the one that named the requirement first: run
    # with the complex family's probe it refuses before the first device leg.
    #
    # The SIBLING complex_fused_magnetic_pair needs NO such swap -- measured the same
    # way, it admits under both -- which is why only this one is doubled. Recording
    # one verdict per artifact rather than picking the permissive one is the whole
    # point: under the census probe the refusal is TRUE, and a launch under that
    # licence would be unlicensed arithmetic.
    seam["folded_complex_fused_magnetic_pair_B"] = dict(
        _verdict(lambda: folded_complex_fused_magnetic_pair_coverage(
            fields, pml, sources, probe)),
        expansion_probe="the census probe (--probe-artifact)")
    seam["folded_complex_fused_magnetic_pair_B@unified_expansion"] = dict(
        _verdict(lambda: folded_complex_fused_magnetic_pair_coverage(
            fields, pml, sources, unified)),
        expansion_probe=UNIFIED_EXPANSION_RECORD,
        expansion_probe_loaded=unified is not None)
    # THE ELECTRIC TWIN OF THE PAIR ABOVE, added 2026-08-31 in the round that built
    # it. ASKED TWICE UNDER THE SAME TWO ARTIFACTS AND FOR THE SAME MEASURED REASON:
    # its PRODUCT_PROBE_PATTERNS is `folded_complex.PARITY_PROBE_PATTERNS`, the same
    # extended set — the two carries call the same
    # `special_kz._mul_imag_coefficient_left` — so no per-family probe classifies it
    # either. Recording one verdict per artifact rather than picking the permissive
    # one is the whole point: under the census probe the refusal is TRUE.
    #
    # THERE IS NO LADDER FOR IT AND THERE NEVER WILL BE. The clause that decides BOTH
    # of its rows consults `deposit_repair.repairable` on the live run, which no
    # configuration block carries: both rows declare an electric source, so a
    # transcription could only have guessed, and guessing False would price the cell
    # at zero. Its MAGNETIC twin is the mirror image — on the B seam those same two
    # rows are electric-only, so no repair is consulted at all.
    seam["folded_complex_fused_pair_D"] = dict(
        _verdict(lambda: folded_complex_fused_pair_coverage(
            fields, pml, sources, probe)),
        expansion_probe="the census probe (--probe-artifact)")
    seam["folded_complex_fused_pair_D@unified_expansion"] = dict(
        _verdict(lambda: folded_complex_fused_pair_coverage(
            fields, pml, sources, unified)),
        expansion_probe=UNIFIED_EXPANSION_RECORD,
        expansion_probe_loaded=unified is not None)
    # THE REAL-BETA ELECTRIC WELD, added 2026-08-31 in the round that built it.
    # ASKED ONCE, and under NO probe at all: this family is REAL storage throughout,
    # so it launches no complex multiply and needs no expansion licence.
    #
    # THERE IS NO LADDER FOR IT AND THERE NEVER WILL BE, while its MAGNETIC twin is
    # still scored from one. The clause that decides its single row consults
    # `deposit_repair.repairable` on the live run, which no configuration block
    # carries: the row declares an electric source, so a transcription could only
    # have guessed, and guessing False would price the cell at zero — which is
    # exactly what the twin's own docstring did before the repair existed.
    seam["beta_fused_electric_pair_D"] = _verdict(
        lambda: beta_fused_electric_pair_coverage(fields, pml, sources))
    # THE BFAST ELECTRIC WELD, added 2026-08-31 in the round that built it. ASKED
    # ONCE and under NO probe: real storage throughout. NO LADDER, for the reason
    # the real-beta twin above has none — the clause that decides its single row
    # consults `deposit_repair.repairable` on the live run, and the row declares an
    # electric source, so a transcription could only have guessed.
    seam["bfast_fused_electric_pair_D"] = _verdict(
        lambda: bfast_fused_electric_pair_coverage(fields, pml, sources))
    # THE CONDUCTIVE PML ELECTRIC WELD, added 2026-08-31 in the round that built it.
    # ASKED ONCE and under NO probe: real storage throughout. NO LADDER, for the two
    # reasons the siblings above have none and one of its own — its curl half's
    # `conductive_targets` reads `Fields.condfac_for` PER COMPONENT, so which of the
    # three COND flags is live is a fact about the live object and not about the row.
    seam["conductive_fused_electric_pair_D"] = _verdict(
        lambda: conductive_fused_electric_pair_coverage(fields, pml, sources))
    # THE NO-ABSORBER STORED-E ELECTRIC WELD, added 2026-08-31 in the round that built
    # it. ASKED ONCE and under NO probe: real storage throughout. NO LADDER, for the
    # reasons its siblings have none and one of its own — its curl half is a
    # DISJUNCTION over the two certified no-absorber curl arms, so a ladder would have
    # to name which arm won on each row, which is a fact about the live object.
    seam["no_pml_fused_electric_pair_D"] = _verdict(
        lambda: no_pml_fused_electric_pair_coverage(fields, pml, sources))
    # THE TWO SCRATCH-OUTPUT OFF-DIAGONAL WELDS, 2026-09-02. NO PROBE ARGUMENT:
    # real storage throughout. Asked once each, and there is NO LADDER for either
    # — the clause that decides half their rows consults
    # `deposit_repair.seam_source_reasons` on the LIVE source list, which no
    # configuration block carries, and the folded one's far-ghost clause reads
    # `stepping._stored_past_owned` off the live grid.
    seam["offdiag_fused_electric_pair_D"] = _verdict(
        lambda: offdiag_fused_electric_pair_coverage(fields, pml, sources))
    seam["folded_offdiag_fused_electric_pair_D"] = _verdict(
        lambda: folded_offdiag_fused_electric_pair_coverage(fields, pml, sources))
    # THE TWO FOLDED-BETA WELDS, added 2026-08-31 in the round that built them.
    # ASKED ONCE EACH and under NO probe: real storage throughout. NO LADDER, for
    # the reasons every sibling above has none plus one of this family's own — the
    # clause that decides its single row consults `deposit_repair.repairable` on
    # the live run, whose FOLD clauses read the grid's own fill map
    # (`_folded_seam_reasons`), and the row declares in-seam sources on BOTH
    # seams, so a transcription guessing either way would misprice both cells.
    seam["folded_beta_fused_electric_pair_D"] = _verdict(
        lambda: folded_beta_fused_electric_pair_coverage(fields, pml, sources))
    seam["folded_beta_fused_magnetic_pair_B"] = _verdict(
        lambda: folded_beta_fused_magnetic_pair_coverage(fields, pml, sources))
    # THE TWO COMPLEX-BETA WELDS, added 2026-09-01 in the round that built them.
    # ASKED ONCE EACH, UNDER THE CENSUS PROBE — and that is a statement about the
    # census probe, not a default: their PRODUCT_PROBE_PATTERNS is
    # `special_kz.BETA_PROBE_PATTERNS` (the base four plus
    # `c8_mul_c8_imaginary_coefficient_left`), which the census probe carries as
    # of the 2026-09-01 extension cut
    # (results/complex_expansion_beta_extension_2026-09-01/probe_keep — the same
    # keep artifact whose base-four table is character-identical to the
    # 2026-08-16 standing one). Under a base-four artifact both refuse BY NAME on
    # the extended-pattern clause, which is the recorded state of every board cut
    # before this round. NO LADDER, for the reason every recent weld has none:
    # the electric twin's deciding clause consults `deposit_repair.repairable` on
    # the live run, and the magnetic twin's source clause reads the same live
    # objects.
    seam["complex_beta_fused_electric_pair_D"] = _verdict(
        lambda: complex_beta_fused_electric_pair_coverage(fields, pml, sources,
                                                          probe))
    seam["complex_beta_fused_magnetic_pair_B"] = _verdict(
        lambda: complex_beta_fused_magnetic_pair_coverage(fields, pml, sources,
                                                          probe))
    # THE TWO FOLDED COMPLEX-BETA WELDS, added 2026-09-02 with the modules (the
    # folded half of the same no-admitting-arm residue). ASKED TWICE EACH, under
    # the census probe and under the unified record, exactly as the beta-less
    # folded complex twins are and for the same measured reason extended by one
    # pattern: their PRODUCT_PROBE_PATTERNS is the SIX-pattern union (base four
    # + parity + the beta imaginary-coefficient-left). The 2026-09-01 census
    # probe classifies five — everything but the PARITY pattern — so under it
    # the fill half refuses; the unified record classifies all six and both
    # extended licences answer FMA_V1 on it. Recording one verdict per artifact
    # rather than picking the permissive one is the whole point. NO LADDER for
    # either: the deciding clauses consult `deposit_repair.repairable` on the
    # live run (all three of the electric twin's rows and one of the magnetic
    # twin's declare in-seam deposits), which no configuration block carries.
    seam["folded_beta_complex_fused_pair_D"] = dict(
        _verdict(lambda: folded_beta_complex_fused_pair_coverage(
            fields, pml, sources, probe)),
        expansion_probe="the census probe (--probe-artifact)")
    seam["folded_beta_complex_fused_pair_D@unified_expansion"] = dict(
        _verdict(lambda: folded_beta_complex_fused_pair_coverage(
            fields, pml, sources, unified)),
        expansion_probe=UNIFIED_EXPANSION_RECORD,
        expansion_probe_loaded=unified is not None)
    seam["folded_beta_complex_fused_magnetic_pair_B"] = dict(
        _verdict(lambda: folded_beta_complex_fused_magnetic_pair_coverage(
            fields, pml, sources, probe)),
        expansion_probe="the census probe (--probe-artifact)")
    seam["folded_beta_complex_fused_magnetic_pair_B@unified_expansion"] = dict(
        _verdict(lambda: folded_beta_complex_fused_magnetic_pair_coverage(
            fields, pml, sources, unified)),
        expansion_probe=UNIFIED_EXPANSION_RECORD,
        expansion_probe_loaded=unified is not None)
    for chain_arm in FUSED_ADE_CHAIN_ARMS:
        seam[f"fused_ade_chain@{chain_arm}"] = _verdict(
            lambda a=chain_arm: fused_ade_chain_coverage(fields, pml, a))

    # THE COMPLEX CHAIN IS ASKED TWICE, UNDER TWO NAMED ARTIFACTS, BECAUSE ITS
    # EXPANSION LICENCE IS NOT THIS CENSUS'S. MEASURED on 2026-08-30, both ways:
    #
    #   this census's probe (complex_expansion_convention_2026-08-16/probe_keep)
    #       expansion_license(..., COMPLEX_ADE_PROBE_PATTERNS) -> None
    #       "pattern 'c8_mul_python_float_field_left' is not classified"
    #   unified_expansion_2026-08-27/keep/gate.json
    #       expansion_license(..., COMPLEX_ADE_PROBE_PATTERNS) -> 1, no refusals
    #
    # This is the same fact the family's own gate entry records
    # (drive_triton_weld_gates.py:255-265): "REQUIRES the UNIFIED EXPANSION RECORD,
    # not a per-family probe... the ADE half reads complex_ade. With the complex
    # gate's probe (or with none) the gate reports 'expansion: E half=1 ADE
    # half=None shared=None' and refuses at one_arm_for_both_halves."
    #
    # BOTH verdicts are recorded and NEITHER is discarded. Under the census probe
    # the product REFUSES BY NAME and that refusal is true — a launch under that
    # licence would be unlicensed arithmetic. The unified verdict is the one its
    # RELEASED gate ran under, and a board may credit the family only against the
    # artifact the gate names, which is why the artifact is carried in the row.
    # Reporting one number without the other would either lose a real admission or
    # credit a licence the run did not hold.
    seam["complex_fused_ade_chain"] = dict(
        _verdict(lambda: complex_fused_ade_chain_coverage(fields, pml, probe)),
        expansion_probe="the census probe (--probe-artifact)")
    seam["complex_fused_ade_chain@unified_expansion"] = dict(
        _verdict(lambda: complex_fused_ade_chain_coverage(fields, pml, unified)),
        expansion_probe=UNIFIED_EXPANSION_RECORD,
        expansion_probe_loaded=unified is not None)

    # THE FOLDED OFF-DIAGONAL E->P CHAIN, added 2026-09-10 with the module. ASKED
    # ONCE, UNDER ONE KEY, and both facts are the product's own rather than this
    # file's convention: its predicate takes no ``arm`` (there is one body and one
    # admission, unlike ``fused_ade_chain``'s three) and it consumes NO expansion
    # record at all — the E half it composes is real float32 storage, so neither
    # the census probe nor the unified record enters its verdict and a second key
    # would distinguish nothing.
    #
    # RECORDED ON EVERY ROW, like every other entry in this block. A verdict
    # present on some rows and absent on others makes the census non-uniform, and
    # the board refuses a non-uniform census rather than counting a cell at zero
    # because nobody asked (build_triton_fusion_matrix's seam-block floor).
    seam["folded_offdiag_fused_ade_chain"] = _verdict(
        lambda: folded_offdiag_fused_ade_chain_coverage(fields, pml))

    results: Dict[str, Any] = {}
    for family, label, sub_step, call in entries:
        results[f"{label}@{sub_step}"] = dict(
            _verdict(call), family=family, kernel=label, sub_step=sub_step)

    # --- update_P: per (state, component), plus the fused all-component form -------
    polarization: List[Dict[str, Any]] = []
    for index, state in enumerate(states):
        driven = ()
        reader = getattr(state, "driven", None)
        if callable(reader):
            try:
                driven = tuple(reader())
            except BaseException:  # noqa: BLE001
                driven = ()
        row = {"index": index,
               "kind": str(getattr(getattr(state, "susceptibility", None), "kind", None)),
               "driven": list(driven),
               "fused": dict(_verdict(lambda s=state: fused_ade_state_coverage(fields, s)),
                             family="shipped", kernel="fused_ade_state",
                             sub_step="update_P"),
               "components": {}}
        for component in driven:
            row["components"][component] = dict(
                _verdict(lambda s=state, c=component: ade_update_p_coverage(fields, s, c)),
                family="shipped", kernel="ade_update_p", sub_step="update_P")
        # THE SECOND update_P FAMILY, wired 2026-08-16. The incumbent pins the
        # EXISTENCE of ``f_w_*``; this one inverts that clause to "the layer is
        # inert and ``drive_field`` hands back the stored E". Recorded per
        # component beside the incumbent, never merged into it, so the roll-up
        # can apply ``plan_step``'s own rule — SOME family admits every driven
        # component of every state — instead of a rule of its own.
        row["components_no_pml_ade"] = {
            component: dict(
                _verdict(lambda s=state, c=component:
                         launch.no_pml_ade_update_p_coverage(fields, pml, s, c)),
                family="residual_wired", kernel="no_pml_ade_update_p",
                sub_step="update_P")
            for component in driven}
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
        "source_field_types": ([str(getattr(s, "field_type", "")) for s in tuple(sources)]
                               if sources is not None else None),
    }
    try:
        from meep_gpu.stepping import _boundary_kinds as resolve  # noqa: PLC0415
        configuration["boundary_kinds"] = list(
            resolve(grid, pml if configuration["pml_active"] else None) or ())
    except BaseException as exc:  # noqa: BLE001
        configuration["boundary_kinds"] = f"UNRESOLVED {type(exc).__name__}: {exc}"[:200]

    return {"predicates": results, "seam": seam, "polarization": polarization,
            "configuration": configuration,
            "plan_step": plan_step_admission(fields, pml, sources, probe)}


# --- what plan_step ITSELF admits -----------------------------------------------------


def plan_step_admission(fields: Any, pml: Any, sources: Any,
                        probe: Any) -> Dict[str, Any]:
    """Run the SHIPPED ``launch.plan_step`` and record what each slot's arms said.

    NOTHING about the composition is re-implemented here. Two seams are borrowed:

    * every ``*_coverage`` forwarder in ``launch`` is wrapped so its verdict comes
      back with the ONE CuPy-backend clause removed — the same filter, character for
      character, that :func:`strip_backend` applies to the battery entries. A verdict
      whose only refusal was that clause becomes ``covered=True``; every other clause
      is left exactly as the predicate wrote it. Nothing else is patched: no gate, no
      builder, no attribute of the lifted objects.
    * ``launch._select_slot`` already RETURNS the admitters it selected from, and
      already receives the arm table. It is wrapped to record that, then delegates to
      the original and returns whatever the original returned. The selection rule
      that decides the slot is the shipped one.

    THE BUILDERS ARE NOT MEASURABLE ON THIS HOST and are not counted. Every kernel
    builder imports ``triton``, which is absent here, so ``_select_slot`` catches an
    import error and leaves the slot unselected with a named reason. That is why the
    number reported is ADMISSION (what the arm table's predicates said), which is
    exactly the layer 482/759 and 576/759 were counted at. ``selected`` and
    ``replaces`` are recorded too, so the builder-blocked slots are visible rather
    than inferred.
    """
    from meep_gpu.triton_kernels import launch  # noqa: PLC0415
    from meep_gpu.triton_kernels.coverage import Coverage  # noqa: PLC0415

    def factored(function: Callable[..., Any]) -> Callable[..., Any]:
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            verdict = function(*args, **kwargs)
            if getattr(verdict, "covered", False):
                return verdict
            reasons = tuple(str(reason) for reason in
                            (getattr(verdict, "reasons", ()) or ()))
            residual = tuple(reason for reason in reasons
                             if not is_backend_clause(reason))
            return Coverage(not residual, residual)
        wrapped.__name__ = getattr(function, "__name__", "wrapped")
        return wrapped

    record: Dict[str, Any] = {"slots": {}, "patched": [], "error": None}
    original: Dict[str, Any] = {}
    for name in dir(launch):
        if not name.endswith("_coverage"):
            continue
        attribute = getattr(launch, name)
        if callable(attribute):
            original[name] = attribute
            setattr(launch, name, factored(attribute))
    record["patched"] = sorted(original)
    original_select = launch._select_slot

    def recording_select(slot: str, arms: Any, ambiguity: Any, plans: Any,
                         reasons: Any, selected: Any) -> Any:
        arms = tuple(arms)
        consulted = []
        for arm in arms:
            if not arm.gate:
                consulted.append({"arm": arm.label, "gate": False})
                continue
            verdict = launch._arm_verdict(arm)
            consulted.append({
                "arm": arm.label, "gate": True,
                "covered": bool(verdict.covered),
                "reasons": [str(reason)[:400] for reason in verdict.reasons]})
        admitted = original_select(slot, arms, ambiguity, plans, reasons, selected)
        record["slots"][slot] = {
            "arms": consulted,
            "admitted": [str(label) for label in admitted],
            "n_gated": sum(1 for entry in consulted if entry["gate"]),
        }
        return admitted

    launch._select_slot = recording_select
    try:
        plan = launch.plan_step(fields, pml, sources=sources, probe=probe)
        record["selected"] = {str(k): str(v) for k, v in plan.selected.items()}
        record["replaces"] = [str(name) for name in plan.replaces]
        record["reasons"] = {str(k): [str(r)[:400] for r in v]
                             for k, v in plan.reasons.items()}
        record["n_polarization_plans"] = len(plan.polarization_plans)
    except BaseException as exc:  # noqa: BLE001 - plan_step must never raise; record it
        record["error"] = f"PLAN_STEP RAISED {type(exc).__name__}: {exc}"[:400]
    finally:
        launch._select_slot = original_select
        for name, attribute in original.items():
            setattr(launch, name, attribute)
    return record


# ==========================================================================
# PROMOTION NOTE -- appended 2026-08-28, and appended rather than woven in.
#
# Everything above this line is a BYTE-IDENTICAL copy of
# ``results/predicate_coverage_2026-08-16_wired_convention/predicate_battery.py``
# (sha256 recorded in the census tranche's PROVENANCE.md, and re-checkable with
# ``head -c <n>``). That is deliberate: the 716/759 headline and every earlier
# Triton census were taken over these exact bytes, and a battery whose entry list
# has been "tidied" during promotion is no longer commensurable with them -- the
# one property the whole campaign's ladder rests on.
#
# WHY IT WAS PROMOTED. It lived inside a results tranche, so the only way to cut a
# fresh Triton census was to copy the tranche. Nineteen divergent copies of
# ``fusion_matrix.py`` under ``results/`` is what that habit produced on the board
# side (build_fusion_matrix.py:110-116), and a census battery is the same hazard one
# layer down. ``measure_predicate_coverage.py --battery triton_predicate_battery``
# now names it by module, beside ``predicate_battery`` (Metal), so neither backend's
# census depends on which directory it was launched from.
#
# WHAT IS *NOT* CLAIMED. The entry list is the 2026-08-16 one. Triton families that
# landed after that date are NOT entered here, and this file must not be read as a
# current inventory of the tree's predicates -- only ``plan_step``'s own per-slot
# ``admitted`` record (which is read live off ``launch``'s arm table, and so does
# pick up everything registered today) carries that. Every import above was
# re-checked against this tree on 2026-08-28 and all resolve; the 49 ``*_coverage``
# forwarders ``plan_step_admission`` wraps are found by ``dir(launch)`` at run time
# and need no list here.
# ==========================================================================
