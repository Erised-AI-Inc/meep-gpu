"""The one module that puts every hand-CUDA family in the arm table.

WHAT IS IN THE TABLE. Every family the closed coverage census asked about, with the
predicate it asked and the slots it asked on — read off
``parity/meep_gpu/results/cuda_predicate_coverage_2026-08-20_closed/
predicate_battery.py:1257-1447``, which is the artifact behind "759 of 759 slots on
186 of 186 rows, 22 families, zero overlaps"
(``.../coverage_report.txt:270-285``). Nothing here re-derives what a family covers;
it binds the SAME entry point the census scored, at the SAME slots.

REGISTRATION IS A CALL HERE, NOT AN IMPORT SIDE EFFECT IN EACH FAMILY, AND THAT IS
THE ONE DELIBERATE DIVERGENCE FROM THE METAL TABLE. On that track a family module
registers its own arms at module scope and :mod:`.registry` merely imports it. The
hand-CUDA family modules ship a predicate and a kernel and nothing else — and most
of them ``import cupy`` at module scope, so registering from inside one would make
the arm table unbuildable on a host with no CuPy. The census that establishes what
this table covers is BACKEND-FREE (``measure_predicate_coverage.py:127`` and :201
both lift with ``prefer_gpu=False``), so a table that could not be built on that
host could not be checked against the measurement that justifies it. The predicate
modules imported below are the laptop-evaluable ones by construction; the kernel
modules are reached only from a launch-argument resolver, which nothing calls at
plan time.

WHAT IS DELIBERATELY NOT REGISTERED, and why each absence is a decision:

* ``coverage.covers_real_pml_ade_component`` and
  ``dispersive_kernels.covers_no_pml_ade_component`` — per-component helpers that
  the whole-sub-step ``..._ade_update_p`` predicates already fold in. ``update_P``
  is ONE slot in the 759-slot denominator; registering the component helpers beside
  the sub-step verdict would bid twice for a slot a family can serve once
  (``predicate_battery.py:447-460`` and the closeout note at :880-897).
* ``folded_offdiag_kernels.covers_folded_offdiag_constitutive`` — the STANDALONE
  verdict, which deliberately admits an UNFOLDED grid so its own gate can measure
  reduction to the certified family. Its composition sibling
  ``covers_folded_offdiag_composition`` requires a real fold, which is exactly the
  complement of the certified off-diagonal family's two fold refusals, so the two
  partition the corpus. Registering the standalone one would make ``update_E``
  permanently ambiguous with ``cuda_offdiag`` on every unfolded off-diagonal row
  (``predicate_battery.py:928-945``).
* ``in_seam_coverage``'s three pass predicates, and ``in_seam_passes.covers_pass``
  which dispatches over them, ON THEIR OWN. The fill slot runs TWO of those passes
  (the near and far ghost fills) with the third (the wall wipe) between them on the
  array path, so the arm that serves it is ``arms.covers_mirror_fill`` -- the two
  fill predicates conjoined, in the seventh positional shape ``fill_by_family`` --
  registered on ``fill_B``/``fill_D`` since 2026-09-27. Registering a pass
  predicate by itself would bid for a slot with half of what the slot runs.
* ``fused_magnetic_pair.covers_fused_magnetic_pair`` — a FUSED PRODUCT, not a slot
  arm. One launch spans ``step_B -> zero_metal_B -> update_H``, so it has no single
  slot to bid at, and both slots it spans are already served (by ``cuda_curl`` and
  ``cuda_constitutive``). Admitting it into a slot table would fire the composer's
  ambiguity refusal on every row it covers while moving the union census by zero,
  because a fused product welds two admissions that are already there rather than
  covering anything new. The same reasoning is recorded beside its census leg,
  which carries ``counted_in_union_census: False``
  (``parity/meep_gpu/cuda_predicate_battery.py``). Composing fused products is a
  separate question from filling slots and this table does not answer it --
  :mod:`.fused_pairs` does, in an opt-in block that runs AFTER the table has selected
  and is bounded by its own ``FUSED_PAIR_ARMS`` declaration, so a fused product may
  absorb a slot only where the table already gave that slot to the arm the fused
  kernel implements. That block, not this table, is what makes the CUDA product
  reachable, and it is also the sole constructor of the two deposit-repair plans on
  this track.

THAT LIST IS ENFORCED, NOT DECORATIVE. :data:`NOT_REGISTERED` names each absence and
``test_arms.test_every_shipped_predicate_is_registered_or_documented`` walks the
package for ``covers_*`` and requires every one to be in the table or in that map. A
family that lands with no arm would otherwise never be selected and nothing would go
red — the silent-divergence class this track spends its gates on.

THE ONE FAMILY THAT REGISTERS TWO ARMS ON ONE SLOT is ``cuda_complex_offdiag``: a
PML tail and a no-PML tail, both at ``update_E``. The census folds them into a
single entry precisely so that family cannot bid twice at a slot it serves once,
and records ``arms_disjoint`` beside it — the two are separated by the same
``_pml_is_active`` switch the array path branches on (``stepping.py:1015`` against
:993), so at most one can ever admit a row. HERE THEY ARE REGISTERED SEPARATELY AND
THE COMPOSER ENFORCES IT: if both ever admit, ``_select_slot`` leaves ``update_E``
UNSELECTED naming both, which is the fail-closed answer. Folding them into one arm
would have hidden the very disagreement the census measures.

DISJOINTNESS IS NOT ASSERTED HERE. The census measured zero overlaps across these
families on the 186 rows; this module binds them and the composer refuses if that
ever stops being true on a configuration nobody measured. A table that ASSUMED
disjointness would silently pick by registration order, which is the defect the
shared ``_select_slot`` exists to make impossible.

NOTHING HERE IS DISPATCH. ``meep_gpu.fastpath.plan_fast_path`` still returns ``None``
on every branch; this module decides only what :func:`.arms.plan_step` may compose.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

from . import arms

#: The three expansion licences the complex families bind, as
#: :class:`.arms.StepContext` keys. THREE AND NOT ONE — the reason is in that
#: class's docstring and the measurement is at ``predicate_battery.py:475-495``
#: and :985-1005.
LICENSE_COMPLEX = "complex"
LICENSE_COMPLEX_NO_PML = "complex_no_pml"
LICENSE_COMPLEX_OFFDIAG = "complex_offdiag"

_CURL_SLOTS: Tuple[str, ...] = ("step_B", "step_D")
_CONSTITUTIVE_SLOTS: Tuple[str, ...] = ("update_H", "update_E")


def _predicate(module: str, name: str) -> Any:
    """One shipped predicate, imported from the family module that owns it.

    Imported in a function body rather than at module scope so a family whose
    predicate module grows a device import does not take the whole arm table down
    with it — the failure would be one arm missing, which changes the SELECTION
    silently, so it is raised here instead.
    """
    from importlib import import_module  # noqa: PLC0415

    return getattr(import_module(f".{module}", __package__), name)


#: THE TABLE. One row per (family, slots, predicate, positional shape, kernel label),
#: in the order the census reports them.
#:
#: ``kernel`` IS THE CENSUS RECORD'S LABEL, QUOTED, and it is neither a device entry
#: point nor a certification. Several of the record's strings are descriptions rather
#: than identifiers — it calls the certified curl pair ``fused_step_{B,D}_pml_real``
#: after the python wrapper, while the declared entry points are
#: ``step_{B,D}_pml_real`` — and the complex families emit their names through
#: ``complex_emitter``'s ``__NAME__`` substitution, so no literal exists to quote.
#: See :class:`.arms.CudaSlotPlan`. A kernel is CERTIFIED only where its own released
#: gate artifact says so; nothing in this table claims that for any of them.
#:
#: A row whose label differs per slot spells a DICT keyed by slot. ``cuda_nonlinear``
#: is the one that needs it and the asymmetry is real rather than cosmetic: the
#: engine reads chi2/chi3 in ``update_E`` only, so the E side has a dedicated body
#: while the H side reuses the ordinary PML constitutive under a nonlinear-only spine
#: arm (``predicate_battery.py:744-757``). One format string would have named the
#: dedicated body on both.
#:
#: ``resolve`` is the launch-argument resolver, present only where one has been
#: established: the certified real-field PML curl pair and the certified real-field
#: PML constitutive pair, both lifted verbatim from the harnesses that already drive
#: them (2026-09-17); and, since 2026-09-27, the three off-diagonal ``update_E``
#: families and the mirror fill, each resolved through the derivations its own
#: wrapper performs when handed the layer. Every other family plans to a
#: :class:`.arms.CudaSlotPlan` that names its kernel and says ``launchable=False``,
#: because its argument resolution lives in its own gate and guessing it is a
#: half-cell error away from converged, smooth and wrong.
#:
#: ``plan_class`` names a plan class other than :class:`.arms.CudaSlotPlan`; only
#: the fill carries one (:data:`_PLAN_CLASSES`).
_TABLE: Tuple[Dict[str, Any], ...] = (
    # --- the two certified real-field families whose resolvers came first ----------
    # (2026-09-17, lifted from the harnesses that drive them). Four more rows carry a
    # ``resolve`` token since 2026-09-27: the three off-diagonal update_E families
    # below and the mirror fill at the end of the table.
    {"family": "cuda_curl", "label": "PML", "noun": "real PML curl",
     "slots": _CURL_SLOTS, "module": "coverage", "name": "covers_real_pml_curl",
     "shape": "curl_by_sub_step", "kernel": "fused_{slot}_pml_real",
     "resolve": "curl"},
    {"family": "cuda_constitutive", "label": "ordinary",
     "noun": "real PML constitutive", "slots": _CONSTITUTIVE_SLOTS,
     "module": "coverage", "name": "covers_real_pml_constitutive",
     "shape": "constitutive_by_side", "kernel": "fused_{slot}_pml_real",
     "resolve": "constitutive"},

    # --- the remaining real-field families ---------------------------------------
    {"family": "cuda_offdiag", "label": "off-diagonal",
     "noun": "off-diagonal update_E", "slots": ("update_E",), "module": "coverage",
     "name": "covers_real_pml_offdiag_constitutive", "shape": "whole_slot",
     "kernel": "fused_update_E_pml_real_offdiag", "resolve": "offdiag"},
    {"family": "cuda_ade", "label": "ADE", "noun": "PML ADE update_P",
     "slots": ("update_P",), "module": "coverage",
     "name": "covers_real_pml_ade_update_p", "shape": "whole_slot",
     "kernel": "fused_update_P_pml_real"},
    {"family": "cuda_cylindrical", "label": "cylindrical",
     "noun": "cylindrical PML curl", "slots": _CURL_SLOTS,
     "module": "cylindrical_coverage", "name": "covers_real_pml_cylindrical_curl",
     # ``fused_cyl_{slot}_pml_real``, matching the complex sibling's spelling two
     # rows down, NOT the ``fused_{slot}_pml_cylindrical`` this row carried until
     # 2026-09-01: the census label is a description, but the fusion board
     # resolves each cell's halves to entry points by stripping ``fused_`` --
     # and ``step_B_pml_cylindrical`` names nothing in this tree, so the board
     # reported the CERTIFIED ``cyl_step_B_pml_real``
     # (``cuda_cylindrical_real_2026-08-27``) as NOT AN ENTRY POINT on the
     # ``B_to_H (cuda_cylindrical, cuda_constitutive)`` cell.
     "shape": "curl_by_sub_step", "kernel": "fused_cyl_{slot}_pml_real"},
    {"family": "cuda_no_pml", "label": "no-PML null",
     "noun": "no-absorber null constitutive", "slots": _CONSTITUTIVE_SLOTS,
     "module": "no_pml_constitutive", "name": "covers_no_pml_null_constitutive",
     "shape": "constitutive_by_side", "kernel": "fused_{slot}_no_pml_null"},
    {"family": "cuda_no_pml_curl", "label": "no-PML curl",
     "noun": "no-absorber curl", "slots": _CURL_SLOTS, "module": "no_pml_curl",
     "name": "covers_no_pml_curl", "shape": "curl_by_sub_step",
     "kernel": "fused_{slot}_no_pml_real"},
    {"family": "cuda_conductive", "label": "conductive",
     "noun": "conductive curl", "slots": _CURL_SLOTS,
     "module": "conductive_kernels", "name": "covers_conductive_curl",
     "shape": "curl_by_sub_step", "kernel": "fused_{slot}_(no_)pml_conductive"},
    {"family": "cuda_special_kz", "label": "real beta", "noun": "special_kz curl",
     "slots": _CURL_SLOTS, "module": "special_kz_curl",
     "name": "covers_special_kz_curl", "shape": "curl_by_sub_step",
     "kernel": "fused_{slot}_special_kz_real"},
    {"family": "cuda_special_kz", "label": "real beta",
     "noun": "special_kz constitutive", "slots": _CONSTITUTIVE_SLOTS,
     "module": "special_kz_curl", "name": "covers_special_kz_constitutive",
     "shape": "constitutive_by_side", "kernel": "fused_{slot}_pml_real"},
    {"family": "cuda_bfast", "label": "BFAST", "noun": "BFAST curl",
     "slots": _CURL_SLOTS, "module": "bfast_curl", "name": "covers_bfast_curl",
     "shape": "curl_by_sub_step", "kernel": "fused_{slot}_bfast_real"},
    {"family": "cuda_bfast", "label": "BFAST", "noun": "BFAST constitutive",
     "slots": _CONSTITUTIVE_SLOTS, "module": "bfast_curl",
     "name": "covers_bfast_constitutive", "shape": "constitutive_by_side",
     "kernel": "fused_{slot}_pml_real"},
    {"family": "cuda_nonlinear", "label": "nonlinear",
     "noun": "nonlinear constitutive", "slots": _CONSTITUTIVE_SLOTS,
     "module": "nonlinear_constitutive",
     "name": "covers_real_pml_nonlinear_constitutive",
     "shape": "constitutive_by_side",
     "kernel": {"update_H": "fused_update_H_pml_real",
                "update_E": "fused_update_E_pml_real_nonlinear"}},
    {"family": "cuda_dispersive", "label": "dispersive",
     "noun": "PML dispersive update_E", "slots": ("update_E",),
     "module": "dispersive_kernels",
     "name": "covers_real_pml_dispersive_constitutive", "shape": "whole_slot",
     "kernel": "fused_update_E_pml_real_dispersive"},
    {"family": "cuda_no_pml_dispersive", "label": "no-PML dispersive store",
     "noun": "no-absorber dispersive update_E", "slots": ("update_E",),
     "module": "dispersive_kernels",
     "name": "covers_no_pml_dispersive_constitutive", "shape": "whole_slot",
     "kernel": "update_E_no_pml_real_dispersive"},
    {"family": "cuda_no_pml_ade", "label": "no-PML ADE",
     "noun": "no-absorber ADE update_P", "slots": ("update_P",),
     "module": "dispersive_kernels", "name": "covers_no_pml_ade_update_p",
     "shape": "whole_slot", "kernel": "fused_update_P_pml_real"},
    {"family": "cuda_folded_offdiag", "label": "folded off-diagonal",
     "noun": "folded off-diagonal update_E", "slots": ("update_E",),
     "module": "folded_offdiag_kernels",
     "name": "covers_folded_offdiag_composition", "shape": "whole_slot",
     "kernel": "fused_update_E_pml_real_folded_offdiag",
     "resolve": "folded_offdiag"},
    {"family": "cuda_dispersive_offdiag", "label": "dispersive off-diagonal",
     "noun": "dispersive off-diagonal update_E", "slots": ("update_E",),
     "module": "dispersive_offdiag_update_e",
     "name": "covers_real_pml_dispersive_offdiag_constitutive",
     "shape": "whole_slot",
     "kernel": "fused_update_E_pml_real_folded_offdiag_dispersive",
     "resolve": "dispersive_offdiag"},

    # --- the complex families, each with the licence its arithmetic needs ---------
    {"family": "cuda_complex", "label": "complex", "noun": "complex PML curl",
     "slots": _CURL_SLOTS, "module": "coverage",
     "name": "covers_real_pml_complex_curl", "shape": "licensed_curl_by_sub_step",
     "license": LICENSE_COMPLEX, "kernel": "fused_{slot}_pml_complex_bloch"},
    {"family": "cuda_complex", "label": "complex",
     "noun": "complex PML constitutive", "slots": _CONSTITUTIVE_SLOTS,
     "module": "coverage", "name": "covers_real_pml_complex_constitutive",
     "shape": "licensed_constitutive_by_side", "license": LICENSE_COMPLEX,
     "kernel": "fused_{slot}_pml_complex_bloch"},
    {"family": "cuda_cyl_complex", "label": "cylindrical complex",
     "noun": "cylindrical complex curl", "slots": _CURL_SLOTS,
     "module": "cylindrical_coverage",
     "name": "covers_pml_cylindrical_complex_curl",
     "shape": "licensed_curl_by_sub_step", "license": LICENSE_COMPLEX,
     "kernel": "fused_cyl_{slot}_pml_complex"},
    {"family": "cuda_complex_folded", "label": "folded complex",
     "noun": "folded complex curl", "slots": _CURL_SLOTS,
     "module": "complex_folded_kernels", "name": "covers_complex_folded_curl",
     "shape": "licensed_curl_by_sub_step", "license": LICENSE_COMPLEX,
     "kernel": "fused_{slot}_pml_complex_folded"},
    {"family": "cuda_complex_folded", "label": "folded complex",
     "noun": "folded complex constitutive", "slots": _CONSTITUTIVE_SLOTS,
     "module": "complex_folded_kernels",
     "name": "covers_complex_folded_constitutive",
     "shape": "licensed_constitutive_by_side", "license": LICENSE_COMPLEX,
     "kernel": "fused_{slot}_pml_complex_bloch"},
    {"family": "cuda_complex_beta", "label": "complex beta",
     "noun": "complex beta curl", "slots": _CURL_SLOTS,
     "module": "complex_beta_kernels", "name": "covers_complex_beta_curl",
     "shape": "licensed_curl_by_sub_step", "license": LICENSE_COMPLEX,
     "kernel": "fused_{slot}_pml_complex_beta"},
    {"family": "cuda_complex_beta", "label": "complex beta",
     "noun": "complex beta constitutive", "slots": _CONSTITUTIVE_SLOTS,
     "module": "complex_beta_kernels",
     "name": "covers_complex_beta_constitutive",
     "shape": "licensed_constitutive_by_side", "license": LICENSE_COMPLEX,
     "kernel": "fused_{slot}_pml_complex_bloch"},

    # --- complex with NO absorber: the seven-pattern licence ----------------------
    #
    # ``update_H`` IS DELIBERATELY ABSENT from this family. Under an inert layer
    # ``stepping.update_H`` returns at :916-917 before reading an array whatever the
    # storage width is, so that slot is ``cuda_no_pml``'s NULL arm and is already
    # served on exactly these rows. A second admitter there would be a widening, not
    # extra coverage — and under this composer it would be an AMBIGUITY that leaves
    # the slot unselected (``predicate_battery.py:521-530``).
    {"family": "cuda_complex_no_pml", "label": "complex no-PML curl",
     "noun": "complex no-absorber curl", "slots": _CURL_SLOTS,
     "module": "complex_no_pml_kernels", "name": "covers_complex_no_pml_curl",
     "shape": "licensed_curl_by_sub_step", "license": LICENSE_COMPLEX_NO_PML,
     "kernel": "fused_{slot}_no_pml_complex[_conductive]"},
    {"family": "cuda_complex_no_pml", "label": "complex no-PML stored E",
     "noun": "complex no-absorber stored update_E", "slots": ("update_E",),
     "module": "complex_no_pml_kernels", "name": "covers_complex_no_pml_stored_e",
     "shape": "licensed_whole_slot", "license": LICENSE_COMPLEX_NO_PML,
     "kernel": "fused_update_E_no_pml_complex_stored"},
    {"family": "cuda_complex_no_pml", "label": "complex no-PML ADE",
     "noun": "complex no-absorber update_P", "slots": ("update_P",),
     "module": "complex_no_pml_kernels",
     "name": "covers_complex_no_pml_ade_update_p", "shape": "licensed_whole_slot",
     "license": LICENSE_COMPLEX_NO_PML,
     "kernel": "fused_update_P_no_pml_complex[_uniform]"},

    # --- the complex tensor row: TWO arms on one slot, separated by the absorber ---
    {"family": "cuda_complex_offdiag", "label": "complex off-diagonal PML",
     "noun": "complex off-diagonal PML update_E", "slots": ("update_E",),
     "module": "complex_offdiag_update_e",
     "name": "covers_complex_offdiag_pml_update_e", "shape": "licensed_whole_slot",
     "license": LICENSE_COMPLEX_OFFDIAG,
     # This family is the one whose labels the census reads off the module itself
     # (``arm.KERNEL_NAMES[winner]``, ``predicate_battery.py:1063``) rather than
     # spelling, so these two are the module's own names and carry no ``fused_``.
     "kernel": "update_E_pml_complex_offdiag"},
    {"family": "cuda_complex_offdiag", "label": "complex off-diagonal no-PML",
     "noun": "complex off-diagonal no-absorber update_E", "slots": ("update_E",),
     "module": "complex_offdiag_update_e",
     "name": "covers_complex_no_pml_offdiag_update_e",
     "shape": "licensed_whole_slot", "license": LICENSE_COMPLEX_OFFDIAG,
     "kernel": "update_E_no_pml_complex_offdiag"},

    # --- the fill slots (2026-09-27): the certified real-storage in-seam kernels ---
    #
    # OUTSIDE THE 759-SLOT CENSUS DENOMINATOR, which counts the four curl and
    # constitutive slots plus update_P; the fill slots are what rung 6b of the
    # dispatch seam requires on a folded run, and without an arm here every folded
    # row was refused WHOLE whenever this table composed without a second one. The
    # predicate is ``arms.covers_mirror_fill`` (both in-seam pass predicates
    # conjoined, an unfolded grid refused by name) and the plan is split at the
    # wall wipe the driver runs between the two passes (``arms.CudaMirrorFillPlan``).
    # Real float32 storage only: the in-seam kernels index float32, and a complex
    # fold has no standalone CUDA fill kernel, so it still refuses by name here.
    {"family": "cuda_mirror_fill", "label": "mirror fill",
     "noun": "real mirror-ghost fill", "slots": ("fill_B", "fill_D"),
     "module": "arms", "name": "covers_mirror_fill", "shape": "fill_by_family",
     "kernel": {"fill_B": "fill_symmetry_B+fill_folded_far_B",
                "fill_D": "fill_symmetry_D+fill_folded_far_D"},
     "resolve": "fill", "plan_class": "fill"},
)

#: The launch-argument resolvers, by the key a table row names. The first two are
#: the ones the harnesses established (2026-09-17); the three off-diagonal ones and
#: the fill (2026-09-27) each call the derivations the family's own wrapper calls
#: when handed the layer, and nothing else. See :class:`.arms.CudaSlotPlan` for why
#: a family without one plans to ``launchable=False`` rather than to a guess.
_RESOLVERS = {
    "curl": arms.resolve_curl_launch_args,
    "constitutive": arms.resolve_constitutive_launch_args,
    "offdiag": arms.resolve_offdiag_launch_args,
    "folded_offdiag": arms.resolve_folded_offdiag_launch_args,
    "dispersive_offdiag": arms.resolve_dispersive_offdiag_launch_args,
    "fill": arms.resolve_fill_launch_args,
}

#: The launchers, keyed by the SAME token: a resolution and a launcher are two ends of
#: one signature, and a row that named one without the other would plan to a
#: ``launchable`` False rather than to half a launch.
_LAUNCHERS = {
    "curl": arms.launch_real_pml_curl,
    "constitutive": arms.launch_real_pml_constitutive,
    "offdiag": arms.launch_offdiag,
    "folded_offdiag": arms.launch_folded_offdiag,
    "dispersive_offdiag": arms.launch_dispersive_offdiag,
    "fill": arms.launch_mirror_fill,
}

#: The plan class a row builds, by the key it names; a row that names none builds a
#: :class:`.arms.CudaSlotPlan`. Explicit rather than read off the shape, because
#: what makes the fill different is that its slot is TWO driver passes, which is a
#: fact about the plan and not about how its predicate is called.
_PLAN_CLASSES = {
    "fill": arms.CudaMirrorFillPlan,
}

#: Every family with a row in :data:`_TABLE`. Named so a test can assert the package
#: holds no OTHER shipped predicate that should have a row — a family in the tree but
#: not in the table is invisible to ``plan_step``, which is a silent coverage loss
#: rather than an error.
FAMILIES: Tuple[str, ...] = tuple(dict.fromkeys(row["family"] for row in _TABLE))

#: Every shipped ``covers_*`` predicate that deliberately has NO row, keyed
#: ``module.name``, with the one-line reason. The prose above argues each one; this
#: is the machine-readable half, so the completeness test can demand that a predicate
#: be registered OR be here, and an undecided one fails rather than passing silently.
#: Adding a name here is a DECISION and must come with the argument above it.
NOT_REGISTERED: Dict[str, str] = {
    "coverage.covers_real_pml_ade_component":
        "per-component helper; the whole-sub-step ade_update_p verdict folds it in, "
        "and update_P is one slot",
    "dispersive_kernels.covers_no_pml_ade_component":
        "per-component helper; same reason as its PML sibling",
    "folded_offdiag_kernels.covers_folded_offdiag_constitutive":
        "the STANDALONE verdict, which admits an unfolded grid on purpose; its "
        "composition sibling is the one that partitions against cuda_offdiag",
    "in_seam_coverage.covers_zero_metal":
        "a seam pass, not a sub-step slot: the driver runs the wall wipe "
        "unconditionally BETWEEN the two fill passes, behind no consult, and "
        "deposit_repair relies on it running there, so no arm takes it",
    "in_seam_coverage.covers_fill_symmetry":
        "not registered on its own: the fill slot runs BOTH fill passes, so its arm "
        "is arms.covers_mirror_fill, which conjoins this with covers_fill_folded_far",
    "in_seam_coverage.covers_fill_folded_far":
        "not registered on its own: the far half of the fill slot, composed through "
        "arms.covers_mirror_fill with covers_fill_symmetry",
    "in_seam_passes.covers_pass":
        "the dispatcher over the three seam passes; the fill arm asks the two fill "
        "predicates directly (arms.covers_mirror_fill) and the wall wipe has no arm",
    "fused_magnetic_pair.covers_fused_magnetic_pair":
        "a FUSED PRODUCT spanning step_B -> zero_metal_B -> update_H; it has no "
        "single slot to bid at and both slots it spans are already served. Reached "
        "instead by arms.plan_step(fuse=True) through fused_pairs.FUSED_PRODUCTS, "
        "which runs after the table has selected and may absorb only what the table "
        "already gave to the arms this kernel implements",
    "special_kz_fused_hd_pair.covers_special_kz_fused_hd_pair":
        "a FUSED PRODUCT spanning update_H -> the electric withdraw -> step_D on a "
        "REAL float32 special_kz (grid.beta != 0) run; the same shape and the same "
        "exclusion as fused_hd_pair's beta-free sibling. It has no single slot to bid "
        "at, and both slots it spans are already served by the cuda_special_kz "
        "family's own two arms -- which is precisely what fused_pairs.FUSED_PAIR_ARMS "
        "requires before it may absorb them. Reached instead by "
        "arms.plan_step(fuse=True) through fused_pairs.FUSED_PRODUCTS, which refuses "
        "it anyway on its own INSTALLABLE = False",
    "complex_beta_fused_hd_pair.covers_complex_beta_fused_hd_pair":
        "a FUSED PRODUCT spanning update_H -> the electric withdraw -> step_D under "
        "complex64 storage with grid.beta live; same shape and same exclusion as the "
        "real-storage sibling above and as complex_fused_hd_pair. Its two slots are "
        "served by the cuda_complex_beta family's own two arms, and it may absorb "
        "them only where the table gave them to exactly those -- and the composer "
        "refuses it on its own INSTALLABLE = False in any case",
    "complex_fused_hd_pair.covers_complex_fused_hd_pair":
        "a FUSED PRODUCT spanning update_H -> the electric withdraw -> step_D under "
        "complex64 storage; the same shape and the same exclusion as fused_hd_pair's "
        "Cartesian sibling. It has no single slot to bid at, and both slots it spans "
        "are already served by the cuda_complex family's own arms. Reached instead by "
        "arms.plan_step(fuse=True) through fused_pairs.FUSED_PRODUCTS, which runs "
        "after the table has selected and may absorb only what the table already gave "
        "to the arms this kernel implements -- and which refuses it anyway on its own "
        "INSTALLABLE = False",
    "complex_fused_magnetic_pair.covers_complex_fused_magnetic_pair":
        "a FUSED PRODUCT spanning step_B -> zero_metal_B -> update_H under complex64 "
        "storage; same shape and same exclusion as its real sibling above. It has no "
        "single slot to bid at, and the two slots it spans are already served by the "
        "cuda_complex family's own arms -- which is precisely what "
        "fused_pairs.FUSED_PAIR_ARMS requires before it may absorb them",
    "cylindrical_fused_magnetic_pair.covers_cylindrical_fused_magnetic_pair":
        "a FUSED PRODUCT spanning step_B -> zero_metal_B -> update_H on a Dcyl grid "
        "at |m| >= 1; same shape and same exclusion. Its two slots are served by the "
        "cuda_cyl_complex curl arm and the cuda_complex constitutive arm, and it may "
        "absorb them only where the table gave them to exactly those",
    "complex_folded_fused_magnetic_pair.covers_complex_folded_fused_magnetic_pair":
        "a FUSED PRODUCT spanning step_B -> fill_symmetry_bc_B -> zero_metal_B -> "
        "fill_folded_far_ghosts_B -> update_H under complex64 storage on a FOLDED "
        "grid; same shape and same exclusion as its unfolded sibling above, plus "
        "the two fills carried by the ownership inversion. Its two slots are served "
        "by the cuda_complex_folded curl arm and the cuda_complex constitutive arm, "
        "and fused_pairs.FUSED_PAIR_ARMS requires the table to have given them to "
        "exactly those before it may absorb",
    "complex_beta_fused_magnetic_pair.covers_complex_beta_fused_magnetic_pair":
        "a FUSED PRODUCT spanning the same five passes as the folded sibling above, "
        "under complex64 storage with grid.beta live; same shape and same "
        "exclusion. Its two slots are served by the cuda_complex_beta family's own "
        "two arms, and it may absorb them only where the table gave them to "
        "exactly those",
    "cylindrical_real_fused_magnetic_pair."
    "covers_cylindrical_real_fused_magnetic_pair":
        "a FUSED PRODUCT spanning step_B -> zero_metal_B -> update_H on a REAL "
        "m = 0 Dcyl grid; same shape and same exclusion. Its two slots are served "
        "by the cuda_cylindrical curl arm and the cuda_constitutive ordinary arm, "
        "and it may absorb them only where the table gave them to exactly those",
    "special_kz_fused_magnetic_pair.covers_special_kz_fused_magnetic_pair":
        "a FUSED PRODUCT spanning step_B -> fill_symmetry_bc_B -> zero_metal_B "
        "-> fill_folded_far_ghosts_B -> update_H on a REAL beta run (the two "
        "fills joined 2026-09-02 with the real pair's ported carry); same shape "
        "and same exclusion. Its two slots are served by the cuda_special_kz "
        "family's own curl and constitutive arms, and it may absorb them only "
        "where the table gave them to exactly those",
    "complex_folded_fused_electric_pair."
    "covers_complex_folded_fused_electric_pair":
        "a FUSED PRODUCT spanning the five passes of the folded ELECTRIC seam "
        "under complex64 storage (the D-side mirror of the folded magnetic "
        "weld, through the shared complex_electric_fill_carry); same shape and "
        "same exclusion. Its two slots are served by the cuda_complex_folded "
        "curl arm and the cuda_complex constitutive arm",
    "complex_beta_fused_electric_pair.covers_complex_beta_fused_electric_pair":
        "a FUSED PRODUCT spanning the same five electric passes with grid.beta "
        "live; same shape and same exclusion. Its two slots are served by the "
        "cuda_complex_beta family's own two arms",
    "cylindrical_real_fused_electric_pair."
    "covers_cylindrical_real_fused_electric_pair":
        "a FUSED PRODUCT spanning step_D -> zero_metal_D -> update_E on a REAL "
        "m = 0 Dcyl grid; same shape and same exclusion. Its two slots are "
        "served by the cuda_cylindrical curl arm and the cuda_constitutive "
        "ordinary arm",
    "special_kz_fused_electric_pair.covers_special_kz_fused_electric_pair":
        "a FUSED PRODUCT spanning the five passes of the folded electric seam "
        "on a REAL beta run (the real electric pair's carry over the beta "
        "curl); same shape and same exclusion. Its two slots are served by the "
        "cuda_special_kz family's own curl and constitutive arms",
    "bfast_fused_electric_pair.covers_bfast_fused_electric_pair":
        "a FUSED PRODUCT spanning step_D -> zero_metal_D -> update_E on a "
        "BFAST run; same shape and same exclusion. Its two slots are served by "
        "the cuda_bfast family's own curl and constitutive arms",
    "offdiag_fused_electric_pair.covers_offdiag_fused_electric_pair":
        "a FUSED PRODUCT spanning step_D -> zero_metal_D -> update_E on an "
        "OFF-DIAGONAL chi1inv run -- the first product this track has put on a "
        "cell the board recorded UNBUILDABLE, through the SCRATCH-OUTPUT weld "
        "(the curl half writes a launch-local allocation the launcher rotates "
        "in afterwards, and every foreign sample the stencil reads is "
        "recomputed from pre-launch state rather than loaded from another "
        "block's output). Same shape and same exclusion as every fused product "
        "on this list: it bids at no slot, and its two are served by the "
        "cuda_curl PML arm and the cuda_offdiag off-diagonal arm, which "
        "fused_pairs.FUSED_PAIR_ARMS requires the table to have given it before "
        "it may absorb them",
    "dispersive_offdiag_fused_polarization_pair."
    "covers_dispersive_offdiag_fused_polarization_pair":
        "a FUSED PRODUCT spanning update_E -> update_P on an OFF-DIAGONAL "
        "DISPERSIVE run -- the shape the shipped E->P pair's own preamble "
        "refuses by name for the per-component split, built here as the "
        "MONOLITHIC update_E the refusal points at. Same shape and same "
        "exclusion as every fused product on this list: it bids at no slot, "
        "and its two are served by the cuda_dispersive_offdiag family's "
        "'dispersive off-diagonal' arm and the cuda_ade family's 'ADE' arm",
    "folded_offdiag_fused_electric_pair."
    "covers_folded_offdiag_fused_electric_pair":
        "the FOLDED sibling of the row above, spanning all five passes of the "
        "folded electric seam on an off-diagonal chi1inv run; same shape and "
        "same exclusion. Its two slots are served by the cuda_curl PML arm and "
        "the cuda_folded_offdiag folded off-diagonal arm -- and the two "
        "products partition on the fold, which one refuses twice and the other "
        "requires",
    "complex_no_pml_fused_polarization_pair."
    "covers_complex_no_pml_fused_polarization_pair":
        "a FUSED PRODUCT spanning update_E -> update_P under complex64 storage "
        "with no absorber, per component with host rotation between -- the "
        "complex sibling of the two E->P products above it on this list; same "
        "shape and same exclusion. Its two slots are served by the "
        "cuda_complex_no_pml family's own stored-E and ADE arms",
    "bfast_fused_magnetic_pair.covers_bfast_fused_magnetic_pair":
        "a FUSED PRODUCT spanning step_B -> zero_metal_B -> update_H on a BFAST "
        "run; same shape and same exclusion. Its two slots are served by the "
        "cuda_bfast family's own curl and constitutive arms, and it may absorb "
        "them only where the table gave them to exactly those",
    "fused_electric_pair.covers_fused_electric_pair":
        "a FUSED PRODUCT spanning step_D -> zero_metal_D -> update_E, the first on "
        "the ELECTRIC seam; same shape and same exclusion as the three B/H products "
        "above. Its two slots are already served by the cuda_curl PML arm and the "
        "cuda_constitutive ordinary arm -- the same two labels the real magnetic pair "
        "implements one seam earlier -- and fused_pairs.FUSED_PAIR_ARMS is what "
        "requires the table to have given them to exactly those before it may absorb",
    "fused_hd_pair.covers_fused_hd_pair":
        "a FUSED PRODUCT spanning update_H -> step_D, the FOURTH seam and the first "
        "whose two halves sit in the order constitutive-then-curl; same shape and "
        "same exclusion as every fused product above. Its two slots are already "
        "served by the cuda_constitutive ordinary arm and the cuda_curl PML arm, and "
        "fused_pairs.FUSED_PAIR_ARMS is what requires the table to have given them to "
        "exactly those before it may absorb. IT IS ALSO REFUSED BY THE COMPOSER ON "
        "EVERY CONFIGURATION -- fused_hd_pair.INSTALLABLE is False, a measured verdict "
        "about the slot arbitration rather than about the arithmetic -- so this "
        "exclusion is doubled: it has no single slot to bid at here, and it does not "
        "take the two it spans there either",
    "cylindrical_real_fused_hd_pair.covers_cylindrical_real_fused_hd_pair":
        "a FUSED PRODUCT spanning update_H -> step_D on a REAL m = 0 Dcyl grid in "
        "TWO launches (the certified update_H recompute plus the fused pre-cumsum "
        "increment; xp.cumsum untouched on the array path; the certified "
        "cyl_step_D_pml_real); same shape and same exclusion as fused_hd_pair. Its "
        "two slots are served by the cuda_constitutive ordinary arm and the "
        "cuda_cylindrical curl arm, and it declares INSTALLABLE = False: the "
        "cylindrical launch algebra favours it and the composer's 4 - pairs rule "
        "does not know it, so the flip is a measured composition gate",
    "cylindrical_fused_hd_pair.covers_cylindrical_fused_hd_pair":
        "the COMPLEX Dcyl twin of the row above, any m, in TWO launches with the "
        "increment's multiply and divide spelled as CuPy's device path spells them "
        "(FMA_V1 mul_field_left; the scaled complex division); same shape and same "
        "exclusion. Its two slots are served by the cuda_complex constitutive arm "
        "and the cuda_cyl_complex curl arm, and it declares INSTALLABLE = False",
    "conductive_bfast_fused_hd_pair.covers_conductive_bfast_fused_hd_pair":
        "a FUSED PRODUCT spanning update_H -> step_D on the two curl tails the "
        "Cartesian H->D product does not reach, in ONE launch and TWO variants -- the "
        "conductive four-case PML recurrence and the BFAST insert, selected from the "
        "run rather than from a caller. Same shape and same exclusion as "
        "fused_hd_pair: it has no single slot to bid at, and both slots it spans are "
        "already served -- the conductive cell's by the cuda_constitutive ordinary arm "
        "and the cuda_conductive conductive arm, the BFAST cell's by the two arms the "
        "cuda_bfast family registers (whose constitutive arm's kernel IS the shipped "
        "fused_update_H_pml_real, so both cells' update_H halves are the same "
        "certified body). It also declares INSTALLABLE = False on the same measured "
        "arbitration verdict, so this exclusion is doubled",
    "dispersive_fused_electric_pair.covers_dispersive_fused_electric_pair":
        "the DISPERSIVE twin of the row above, spanning all five passes of the "
        "electric seam on a run carrying a susceptibility -- the board's largest "
        "unoccupied cell when it landed (D_to_E cuda_curl/PML x "
        "cuda_dispersive/dispersive, 7 corpus rows); same shape and same exclusion. "
        "Its two slots are served by the cuda_curl PML arm and the cuda_dispersive "
        "'dispersive' arm, and the two electric products partition on "
        "fields.polarizations, which the ordinary constitutive predicate refuses "
        "truthy and the dispersive one requires",
    "no_pml_dispersive_fused_electric_pair."
    "covers_no_pml_dispersive_fused_electric_pair":
        "a FUSED PRODUCT spanning step_D -> zero_metal_D -> update_E on a run with NO "
        "absorber, a registered susceptibility and stored E -- the last two "
        "POINTWISE-BUILDABLE D->E cells this board had (D_to_E cuda_conductive/"
        "conductive x cuda_no_pml_dispersive, 2 corpus rows, and D_to_E "
        "cuda_no_pml_curl/no-PML curl x the same, 1). ONE product on TWO cells, "
        "because conductive_kernels bakes the per-component conductivity into three "
        "COND defines and the (False, False, False) build IS no_pml_curl's certified "
        "lossless tail. Its two slots are served by the cuda_conductive and "
        "cuda_no_pml_curl arms at step_D and the cuda_no_pml_dispersive arm at "
        "update_E; it is the FIRST product on this track to declare a non-default "
        "REPAIR_PATHS, because update_E's plain overwrite (stepping.py:1019-1022) is "
        "not the split-field recurrence every other electric weld inverts",
    "folded_complex_offdiag_fused_electric_pair."
    "covers_folded_complex_offdiag_fused_electric_pair":
        "a FUSED PRODUCT spanning all five passes of the folded electric seam under "
        "COMPLEX64 storage on an OFF-DIAGONAL chi1inv run -- the COMPLEX twin of the "
        "two real stencil welds, and one of the last two cells this board recorded "
        "UNBUILDABLE, closed through the same SCRATCH-OUTPUT shape (the curl half "
        "writes a launch-local allocation the launcher rotates in afterwards, and "
        "every foreign sample the stencil reads is recomputed from pre-launch state "
        "rather than loaded from another block's output). Same shape and same "
        "exclusion as every fused product on this list: it bids at no slot, and its "
        "two are served by the cuda_complex_folded 'folded complex' curl arm and the "
        "cuda_complex_offdiag 'complex off-diagonal PML' arm. It is the FIRST product "
        "here whose two halves sit in different expansion-licence families, so its "
        "predicate takes one licence per half rather than one for both",
    "complex_no_pml_offdiag_fused_electric_pair."
    "covers_complex_no_pml_offdiag_fused_electric_pair":
        "the NO-ABSORBER sibling of the row above, spanning step_D -> zero_metal_D -> "
        "update_E under complex64 storage on an off-diagonal chi1inv run with an "
        "inert layer; same shape and same exclusion. Its seam is SHORTER rather than "
        "differently filled -- no split-field auxiliary, no constitutive recurrence "
        "and no fold in reach -- and its two slots are served by the "
        "cuda_complex_no_pml 'complex no-PML curl' arm and the cuda_complex_offdiag "
        "'complex off-diagonal no-PML' arm. The two complex welds partition on the "
        "ABSORBER, which one constitutive arm requires active and the other requires "
        "inert",
    "conductive_fused_electric_pair.covers_conductive_fused_electric_pair":
        "a FUSED PRODUCT spanning step_D -> zero_metal_D -> update_E where the curl "
        "is the THREE-HISTORY conductive-PML recurrence -- the board's D_to_E "
        "cuda_conductive/conductive x cuda_constitutive/ordinary cell, 1 corpus row "
        "(tests:TestAdjointSolver.test_damping). Same shape and same exclusion as the "
        "real twin two rows above; the two partition on the CONDUCTIVITY, which "
        "coverage.covers_real_pml_curl refuses by name and covers_conductive_curl "
        "requires. Servable only since the driver's condinv injection began "
        "rescaling SPARSELY at the published deposit indices",
    "complex_fused_electric_pair.covers_complex_fused_electric_pair":
        "a FUSED PRODUCT spanning step_D -> zero_metal_D -> update_E under complex64 "
        "storage; same shape and same exclusion. Its two slots are served by the "
        "cuda_complex family's own curl and constitutive arms -- the same label pair "
        "its magnetic twin absorbs one seam earlier -- and fused_pairs.FUSED_PAIR_ARMS "
        "is what requires the table to have given them to exactly those",
    "cylindrical_fused_electric_pair.covers_cylindrical_fused_electric_pair":
        "a FUSED PRODUCT spanning step_D -> zero_metal_D -> update_E on a Dcyl grid at "
        "|m| >= 1; same shape and same exclusion. Its two slots are served by the "
        "cuda_cyl_complex curl arm and the cuda_complex constitutive arm, and it may "
        "absorb them only where the table gave them to exactly those",
    "no_pml_complex_fused_electric_pair."
    "covers_no_pml_complex_fused_electric_pair":
        "a FUSED PRODUCT spanning step_D -> update_E under complex64 storage with NO "
        "ABSORBER; same shape and same exclusion as the three PML electric products "
        "above, and its seam is SHORTER than theirs rather than differently filled -- "
        "it refuses every walled, folded and electrically-sourced run, so nothing the "
        "driver puts between driver.py:3302 and :3313 runs inside it and REPLACES is a "
        "contiguous run of two. Its two slots are served by this family's own "
        "'complex no-PML curl' and 'complex no-PML stored E' arms -- the two rows "
        "directly above in this table -- and fused_pairs.FUSED_PAIR_ARMS is what "
        "requires the table to have given them to exactly those before it may absorb",
    "fused_polarization_pair.covers_fused_polarization_pair":
        "a FUSED PRODUCT spanning update_E -> update_P under an active layer -- the "
        "first on the E->P seam, whose two consults are ADJACENT driver statements "
        "(driver.py:3313/:3315) with no injection, fill or wall pass between them. "
        "It has no single slot to bid at, and both slots it spans are already served "
        "by the cuda_dispersive 'dispersive' arm and the cuda_ade 'ADE' arm -- the "
        "two predicates it conjoins -- so registering it would fire the composer's "
        "ambiguity refusal on every row it covers while moving the union census by "
        "zero. Reached instead by arms.plan_step(fuse=True) through "
        "fused_pairs.FUSED_PRODUCTS, which runs after the table has selected and "
        "may absorb only what the table gave to exactly those arms",
    "fused_polarization_pair.covers_no_pml_fused_polarization_pair":
        "a FUSED PRODUCT spanning update_E -> update_P with NO absorber; same shape "
        "and same exclusion as its PML twin above, disjoint from it on the absorber "
        "boolean alone. Its two slots are served by the cuda_no_pml_dispersive "
        "'no-PML dispersive store' arm and the cuda_no_pml_ade 'no-PML ADE' arm -- "
        "the two predicates it conjoins -- and fused_pairs.FUSED_PAIR_ARMS is what "
        "requires the table to have given them to exactly those before it may absorb",
    "three_slot_dispersive_weld.covers_three_slot_dispersive_weld":
        "a FUSED PRODUCT spanning THREE slots -- step_D -> update_E -> update_P -- "
        "and the first on any backend to do so. Same shape and same exclusion as "
        "every fused product on this list, one slot longer: it bids at no slot, and "
        "all THREE it spans are already served, by the cuda_curl 'PML' arm, the "
        "cuda_dispersive 'dispersive' arm and the cuda_ade 'ADE' arm -- the three "
        "labels of the two predicates it conjoins. Registering it would fire the "
        "composer's ambiguity refusal on every row it covers while moving the union "
        "census by zero. Reached instead by arms.plan_step(fuse=True), where its "
        "three-slot span SUPERSEDES the two-slot D->E product whose admission set "
        "strictly contains its own (fused_pairs._superseded_by_a_longer_span) rather "
        "than colliding with it",
    "no_pml_three_slot_dispersive_weld."
    "covers_no_pml_three_slot_dispersive_weld":
        "the NO-ABSORBER twin of the row above, spanning step_D -> zero_metal_D -> "
        "update_E -> update_P on a run with an inert layer and a registered "
        "susceptibility; same shape and same exclusion, disjoint from it on the "
        "absorber boolean alone. Its three slots are served by the cuda_conductive "
        "'conductive' arm (or the cuda_no_pml_curl 'no-PML curl' arm on its second "
        "cell, declared in fused_pairs.FUSED_PAIR_EXTRA_ARMS), the "
        "cuda_no_pml_dispersive 'no-PML dispersive store' arm and the cuda_no_pml_ade "
        "'no-PML ADE' arm",
    "complex_no_pml_three_slot_dispersive_weld."
    "covers_three_slot_complex_no_pml_dispersive_weld":
        "the COMPLEX no-absorber THREE-SLOT weld -- step_D -> update_E -> update_P "
        "under complex64 storage with no layer -- and the same exclusion as every "
        "fused product on this list: it bids at no slot, and all THREE it spans are "
        "already served, by this family's 'complex no-PML curl', 'complex no-PML "
        "stored E' and 'complex no-PML ADE' arms -- the labels of the two predicates "
        "it conjoins (no_pml_complex_fused_electric_pair and "
        "complex_no_pml_fused_polarization_pair, both themselves on this list). "
        "Registering it would fire the composer's ambiguity refusal on every row it "
        "covers while moving the union census by zero. Reached instead by "
        "arms.plan_step(fuse=True), where its three-slot span SUPERSEDES the two-slot "
        "D->E product whose admission set strictly contains its own "
        "(fused_pairs._superseded_by_a_longer_span) and holds update_E out of the "
        "E->P pair's reach (fused_pairs._pair_may_absorb)",
}


def register_all() -> Tuple[arms.ArmSpec, ...]:
    """Put every row of :data:`_TABLE` in the arm table. Called once, by
    :func:`.arms.ensure_registered`, which is what makes "once" true.

    A duplicate row is REFUSED by :func:`.arms.register` rather than replaced, so a
    (family, label) pair accidentally spelled twice on one slot fails here, loudly,
    instead of making which kernel runs depend on table order.
    """
    registered = []
    for row in _TABLE:
        predicate = _predicate(row["module"], row["name"])
        coverage = arms.coverage_adapter(predicate, row["shape"],
                                         row.get("license"))
        resolve = _RESOLVERS.get(row.get("resolve"))
        launch = _LAUNCHERS.get(row.get("resolve"))
        plan_class = _PLAN_CLASSES.get(row.get("plan_class"), arms.CudaSlotPlan)
        for slot in row["slots"]:
            kernel = row["kernel"]
            label = (kernel[slot] if isinstance(kernel, dict)
                     else kernel.format(slot=slot))
            plan = arms.plan_factory(row["family"], row["label"], label, resolve,
                                     launch, plan_class=plan_class)
            registered.append(arms.register(
                family=row["family"], slot=slot, label=row["label"],
                coverage=coverage, plan=plan,
                prefix=f"{row['label']}: ", noun=row["noun"],
                wired=True, shape=row["shape"],
                predicate_name=f"cuda_kernels.{row['module']}.{row['name']}"))
    return tuple(registered)
