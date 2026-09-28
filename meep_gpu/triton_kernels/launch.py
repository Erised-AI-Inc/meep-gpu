"""Host side of the Triton real-field PML step path.

Six plan builders live here, one per covered sub-step family; every specialized
family's builder lives in its own module and is reached through a lazy forwarder
below:

    plan_pml_curl(fields, pml, "step_B" | "step_D")   -> PmlCurlPlan | None
    plan_plain_curl(fields, pml, "step_B" | "step_D") -> PlainCurlPlan | None
    plan_constitutive(fields, pml, "H" | "E")         -> ConstitutivePlan | None
    plan_dispersive_constitutive(fields, pml)          -> DispersiveConstitutivePlan | None
    plan_ade_update_p(fields, state, component)       -> AdeUpdatePPlan | None
    plan_step(fields, pml)                            -> TritonStepPlan

:func:`plan_step` is the composition: it asks each predicate independently and
reports the SUBSET of sub-steps that are covered. COVERAGE IS A SET PER SUB-STEP,
not per run — a supported dispersive PML run gets the ordinary curl/H plans, the
specialized dispersive ``update_E`` plan and ``update_P``; a non-dispersive one gets
the ordinary E plan instead. Nothing about that mixes paths *within* a sub-step;
the driver's loop separates all of them with array-path boundary work.

A *plan* is built once for a (fields, pml, sub-step) triple and launched per
timestep. Everything that can be resolved ahead of the loop is resolved there:
the coverage verdict, the boundary constexprs, the Yee sub-lattice pairing, the
flattened coefficient views and the pointer adapters. Nothing on the launch path
allocates — the flattening in particular is a ``reshape(-1)`` VIEW taken once,
because doing it per launch is port-reference defect 5.14 (an allocation and a
copy, six times a timestep, of an array the plan already owns).

DISPATCH IS WIRED AND ON BY DEFAULT for a driver built with ``prefer_gpu=True``.
The driver-seam round gave ``meep_gpu.fastpath.plan_fast_path`` a real return and
the driver consults it at seven sites, so the planner is reached by a default
run, not by tests, gates and the benchmark alone. What bounds it is the ENABLE
and each arm's release entry, not the absence of a caller:
``fastpath.DISPATCH_BY_DEFAULT`` is True, ``MEEP_GPU_DISPATCH=0`` turns dispatch
off for a run, ``MEEP_GPU_FUSED=0`` vetoes it above that, and a
``prefer_gpu=False`` driver never consults a kernel table. Earlier text here
said dispatch was off by default; that described the package before release.
"""

from __future__ import annotations

from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Tuple

from .coverage import (
    CONSTITUTIVE_SIDES,
    FUSED_PAIRS,
    Coverage,
    ade_update_p_coverage,
    constitutive_coverage,
    dispersive_constitutive_coverage,
    fused_pair_coverage,
    pml_curl_coverage,
    sigma_is_volume,
    zero_metal_axes,
)

# The specialized builder is safe to import without Triton: its module keeps the
# optional dependency behind its launch path.  Importing it here also gives tests a
# concrete routing seam to replace without constructing device pointers.
from .dispersive_update_e import plan_dispersive_constitutive
from .. import deposit_repair as _deposit_repair
# The withdraw hoist is the H->D seam's counterpart to the deposit repair, and it is
# SHARED for the same reason: that seam carries no injection but does carry the
# driver's electric `withdraw`, and a second implementation of "perform the seam's
# pass before the launch" is a second thing to place wrong — the 2026-09-04
# campaign's own null control put it one pass later and diverged at step 2.
from .. import withdraw_hoist as _withdraw_hoist


def conductive_pml_curl_coverage(fields: Any, pml: Any, sub_step: str) -> Coverage:
    """Ask the conductive-PML predicate through its optional module."""
    from .conductivity import conductive_pml_curl_coverage as specialized_coverage  # noqa: PLC0415
    return specialized_coverage(fields, pml, sub_step)


def plan_conductive_pml_curl(fields: Any, pml: Any, sub_step: str,
                             block: Optional[int] = None) -> Any:
    """Build one conductive PML curl plan through its specialized module."""
    from .conductivity import plan_conductive_pml_curl as specialized_plan  # noqa: PLC0415
    return specialized_plan(fields, pml, sub_step, block)


def plain_curl_coverage(fields: Any, pml: Any, sub_step: str) -> Coverage:
    """Ask the no-absorber predicate without importing its module at package import."""
    from .no_pml import plain_curl_coverage as specialized_coverage  # noqa: PLC0415
    return specialized_coverage(fields, pml, sub_step)


def plan_plain_curl(fields: Any, pml: Any, sub_step: str,
                    block: Optional[int] = None) -> Any:
    """Build one no-absorber curl plan through its specialized module."""
    from .no_pml import plan_plain_curl as specialized_plan  # noqa: PLC0415
    return specialized_plan(fields, pml, sub_step, block)


def folded_grid_active(fields: Any) -> bool:
    """Whether the specialized folded composition family applies.

    An UNREADABLE fold consults the arm (see :func:`beta_active`); the folded
    predicates then refuse by name on the same unreadable attribute, which is
    the array path and is always correct.
    """
    from .symmetry import folded_grid_active as predicate  # noqa: PLC0415
    try:
        return predicate(fields)
    except Exception:  # noqa: BLE001 - an unreadable gate is consulted, not assumed
        return True


def folded_composition_curl_coverage(fields: Any, pml: Any,
                                     sub_step: str) -> Coverage:
    """Ask the folded curl's routing predicate through its optional module."""
    from .symmetry import folded_composition_curl_coverage as predicate  # noqa: PLC0415
    return predicate(fields, pml, sub_step)


def plan_folded_pml_curl(fields: Any, pml: Any, sub_step: str,
                         block: Optional[int] = None) -> Any:
    """Build one folded PML curl plan."""
    from .symmetry import plan_folded_pml_curl as builder  # noqa: PLC0415
    return builder(fields, pml, sub_step, block)


def mirror_ghost_fill_coverage(fields: Any, family: str) -> Coverage:
    """Ask whether one folded field family's source-adjacent fills are covered."""
    from .symmetry import mirror_ghost_fill_coverage as predicate  # noqa: PLC0415
    return predicate(fields, family)


def plan_mirror_ghost_fill(fields: Any, family: str,
                           block: Optional[int] = None) -> Any:
    """Build one combined near/far mirror-fill plan."""
    from .symmetry import plan_mirror_ghost_fill as builder  # noqa: PLC0415
    return builder(fields, family, block)


def folded_constitutive_coverage(fields: Any, pml: Any, side: str) -> Coverage:
    """Ask whether the ordinary constitutive arithmetic is valid on this fold."""
    from .symmetry import folded_constitutive_coverage as predicate  # noqa: PLC0415
    return predicate(fields, pml, side)


def plan_folded_constitutive(fields: Any, pml: Any, side: str,
                             block: Optional[int] = None) -> Any:
    """Build the ordinary constitutive kernel against a folded stored extent."""
    from .symmetry import plan_folded_constitutive as builder  # noqa: PLC0415
    return builder(fields, pml, side, block)


def cylindrical_grid_active(fields: Any) -> bool:
    """Whether the specialized real m=0 Dcyl composition family applies.

    An UNREADABLE attribute consults the arm rather than skipping it, for the
    reason spelled out at :func:`beta_active`: refusing to read is not evidence
    the family does not apply.

    ``fields.grid`` IS READ INSIDE THE TRY, and that is not a style choice.
    Hoisted out of it — as this gate and the two below it used to have it — a
    ``fields`` whose ``grid`` attribute raises took the WHOLE composition down
    from the gate block, before a single predicate ran: measured as
    ``RuntimeError`` escaping ``plan_step`` on every one of the six base
    configurations of the attribute fuzz, against a function whose first
    documented contract is that it never raises.
    """
    try:
        return bool(getattr(getattr(fields, "grid", None), "cylindrical", False))
    except Exception:  # noqa: BLE001 - an unreadable gate is consulted, not assumed
        return True


def cylindrical_curl_coverage(fields: Any, pml: Any) -> Coverage:
    """Ask the m=0 Dcyl curl predicate through its optional module."""
    from .cylindrical_triton import cylindrical_curl_coverage as predicate  # noqa: PLC0415
    return predicate(fields, pml)


def plan_cylindrical_curl(fields: Any, pml: Any, sub_step: str,
                          block: Optional[int] = None) -> Any:
    """Build one admitted m=0 Dcyl curl plan."""
    from .cylindrical_triton import plan_cylindrical_curl as builder  # noqa: PLC0415
    return builder(fields, pml, sub_step, block)


def cylindrical_constitutive_coverage(fields: Any, pml: Any,
                                       side: str) -> Coverage:
    """Ask the measured m=0 Dcyl constitutive predicate."""
    from .cylindrical_triton import cylindrical_constitutive_coverage as predicate  # noqa: PLC0415
    return predicate(fields, pml, side)


def plan_cylindrical_constitutive(fields: Any, pml: Any, side: str,
                                  block: Optional[int] = None) -> Any:
    """Build the existing elementwise constitutive plan for the Dcyl slice."""
    from .cylindrical_triton import plan_cylindrical_constitutive as builder  # noqa: PLC0415
    return builder(fields, pml, side, block)


def dispersive_fused_pair_coverage(fields: Any, pml: Any,
                                   sources: Any = None) -> Coverage:
    """Ask the pole-aware D/E pair predicate through its optional module."""
    from .dispersive_fused_pair import dispersive_fused_pair_coverage as predicate  # noqa: PLC0415
    return predicate(fields, pml, sources)


def plan_dispersive_fused_pair(fields: Any, pml: Any, sources: Any = None,
                               block: Optional[int] = None,
                               num_warps: Optional[int] = 1) -> Any:
    """Build the pole-aware D/E pair through its specialized module."""
    from .dispersive_fused_pair import plan_dispersive_fused_pair as builder  # noqa: PLC0415
    return builder(fields, pml, sources, block, num_warps)


def fused_ade_state_coverage(fields: Any, state: Any) -> Coverage:
    """Ask the one-launch susceptibility predicate through its optional module."""
    from .fused_ade_state import fused_ade_state_coverage as predicate  # noqa: PLC0415
    return predicate(fields, state)


def plan_fused_ade_state(fields: Any, state: Any,
                         block: Optional[int] = None) -> Any:
    """Build one all-component susceptibility plan through its optional module."""
    from .fused_ade_state import plan_fused_ade_state as builder  # noqa: PLC0415
    return builder(fields, state, block)


# ---------------------------------------------------------------------------
# The five certified specialized families — lazy forwarders
# ---------------------------------------------------------------------------
#
# Same shape as every forwarder above and for the same two reasons. The import
# lives INSIDE the body, so importing this package never pulls a family module
# in (each family's own test pins that on a machine with no Triton); and every
# name here is a MODULE-LEVEL attribute, so a routing test can replace a
# predicate or a builder without constructing a device pointer.
#
# Nothing here decides anything: the selection rule is :func:`plan_step`'s and
# the arithmetic is each family's. These are the seam between them.


def load_expansion_probe(path: Optional[str] = None) -> Any:
    """Read the complex-multiply expansion probe artifact, or None when absent.

    Three-valued like the reader it forwards to: a record, ``None`` for absent,
    or a ``RefusedExpansionProbe`` when this dispatch has already refused the
    artifact that variable points at. The re-export exists so callers need not
    import a family module; it must not flatten the third value into ``None``,
    which is what made a refusal survivable by a re-read.
    """
    from .complex_fields import load_expansion_probe as reader  # noqa: PLC0415
    return reader(path)


def complex_pml_curl_coverage(fields: Any, pml: Any, sub_step: str,
                              probe: Any = None) -> Coverage:
    """Ask the complex/Bloch split-field curl predicate."""
    from .complex_fields import complex_pml_curl_coverage as predicate  # noqa: PLC0415
    return predicate(fields, pml, sub_step, probe=probe)


def plan_complex_pml_curl(fields: Any, pml: Any, sub_step: str,
                          block: Optional[int] = None, probe: Any = None) -> Any:
    """Build one complex/Bloch split-field curl plan."""
    from .complex_fields import plan_complex_pml_curl as builder  # noqa: PLC0415
    return builder(fields, pml, sub_step, block, probe=probe)


def complex_constitutive_coverage(fields: Any, pml: Any, side: str,
                                  probe: Any = None) -> Coverage:
    """Ask the complex/Bloch constitutive predicate for one side."""
    from .complex_fields import complex_constitutive_coverage as predicate  # noqa: PLC0415
    return predicate(fields, pml, side, probe=probe)


def plan_complex_constitutive(fields: Any, pml: Any, side: str,
                              block: Optional[int] = None,
                              probe: Any = None) -> Any:
    """Build one complex/Bloch constitutive plan."""
    from .complex_fields import plan_complex_constitutive as builder  # noqa: PLC0415
    return builder(fields, pml, side, block, probe=probe)


def beta_pml_curl_coverage(fields: Any, pml: Any, sub_step: str) -> Coverage:
    """Ask the REAL special_kz curl predicate."""
    from .special_kz import beta_pml_curl_coverage as predicate  # noqa: PLC0415
    return predicate(fields, pml, sub_step)


def plan_beta_pml_curl(fields: Any, pml: Any, sub_step: str,
                       block: Optional[int] = None) -> Any:
    """Build one REAL special_kz curl plan."""
    from .special_kz import plan_beta_pml_curl as builder  # noqa: PLC0415
    return builder(fields, pml, sub_step, block)


def beta_bloch_pml_curl_coverage(fields: Any, pml: Any, sub_step: str,
                                 probe: Any = None) -> Coverage:
    """Ask the COMPLEX special_kz curl predicate (its extended pattern set)."""
    from .special_kz import beta_bloch_pml_curl_coverage as predicate  # noqa: PLC0415
    return predicate(fields, pml, sub_step, probe=probe)


def plan_beta_bloch_pml_curl(fields: Any, pml: Any, sub_step: str,
                             block: Optional[int] = None,
                             probe: Any = None) -> Any:
    """Build one COMPLEX special_kz curl plan."""
    from .special_kz import plan_beta_bloch_pml_curl as builder  # noqa: PLC0415
    return builder(fields, pml, sub_step, block, probe=probe)


def beta_run_constitutive_coverage(fields: Any, pml: Any, side: str) -> Coverage:
    """Ask whether the certified REAL constitutive kernel may step a beta run."""
    from .special_kz import beta_run_constitutive_coverage as predicate  # noqa: PLC0415
    return predicate(fields, pml, side)


def plan_beta_run_constitutive(fields: Any, pml: Any, side: str,
                               block: Optional[int] = None) -> Any:
    """Build the certified REAL constitutive plan for a beta run."""
    from .special_kz import plan_beta_run_constitutive as builder  # noqa: PLC0415
    return builder(fields, pml, side, block)


def beta_run_complex_constitutive_coverage(fields: Any, pml: Any, side: str,
                                           probe: Any = None) -> Coverage:
    """Ask whether the certified COMPLEX constitutive kernel may step a beta run.

    The base pattern set, not the curl's extended one: the kernel launched is
    the certified complex constitutive kernel unchanged, and its own contract is
    what binds here (``special_kz`` grouping choice 5).
    """
    from .special_kz import beta_run_complex_constitutive_coverage as predicate  # noqa: PLC0415
    return predicate(fields, pml, side, probe=probe)


def plan_beta_run_complex_constitutive(fields: Any, pml: Any, side: str,
                                       block: Optional[int] = None,
                                       probe: Any = None) -> Any:
    """Build the certified COMPLEX constitutive plan for a beta run."""
    from .special_kz import plan_beta_run_complex_constitutive as builder  # noqa: PLC0415
    return builder(fields, pml, side, block, probe=probe)


def bfast_pml_curl_coverage(fields: Any, pml: Any, sub_step: str) -> Coverage:
    """Ask the BFAST curl predicate."""
    from .bfast_curl import bfast_pml_curl_coverage as predicate  # noqa: PLC0415
    return predicate(fields, pml, sub_step)


def plan_bfast_pml_curl(fields: Any, pml: Any, sub_step: str,
                        block: Optional[int] = None) -> Any:
    """Build one BFAST curl plan.

    ONE launch, not two. The second additive term and the ``f_bfast_*`` IIR
    state both live inside this curl kernel, written in place and
    pointer-identical to the engine's arrays — which is why BFAST adds no slot
    to :data:`STEP_ORDER`.
    """
    from .bfast_curl import plan_bfast_pml_curl as builder  # noqa: PLC0415
    return builder(fields, pml, sub_step, block)


def bfast_run_constitutive_coverage(fields: Any, pml: Any, side: str) -> Coverage:
    """Ask whether the certified REAL constitutive kernel may step a BFAST run."""
    from .bfast_curl import bfast_run_constitutive_coverage as predicate  # noqa: PLC0415
    return predicate(fields, pml, side)


def plan_bfast_run_constitutive(fields: Any, pml: Any, side: str,
                                block: Optional[int] = None) -> Any:
    """Build the certified REAL constitutive plan for a BFAST run."""
    from .bfast_curl import plan_bfast_run_constitutive as builder  # noqa: PLC0415
    return builder(fields, pml, side, block)


def nonlinear_constitutive_coverage(fields: Any, pml: Any) -> Coverage:
    """Ask the chi2/chi3 Pade ``update_E`` predicate."""
    from .nonlinear_update_e import nonlinear_constitutive_coverage as predicate  # noqa: PLC0415
    return predicate(fields, pml)


def plan_nonlinear_constitutive(fields: Any, pml: Any,
                                block: Optional[int] = None) -> Any:
    """Build the chi2/chi3 Pade ``update_E`` plan."""
    from .nonlinear_update_e import plan_nonlinear_constitutive as builder  # noqa: PLC0415
    return builder(fields, pml, block)


def offdiag_constitutive_coverage(fields: Any, pml: Any) -> Coverage:
    """Ask the off-diagonal chi1inv row-product ``update_E`` predicate."""
    from .offdiag_update_e import offdiag_constitutive_coverage as predicate  # noqa: PLC0415
    return predicate(fields, pml)


def plan_offdiagonal_constitutive(fields: Any, pml: Any,
                                  block: Optional[int] = None) -> Any:
    """Build the off-diagonal chi1inv row-product ``update_E`` plan."""
    from .offdiag_update_e import plan_offdiagonal_constitutive as builder  # noqa: PLC0415
    return builder(fields, pml, block)


# ---------------------------------------------------------------------------
# The four further certified families — lazy forwarders
# ---------------------------------------------------------------------------
#
# THE IMPORT MUST STAY INSIDE THE BODY HERE FOR A SECOND REASON, on top of the
# lazy-import one every forwarder above carries: ``folded_complex`` and
# ``cylindrical_complex`` both import FROM THIS MODULE at module scope
# (folded_complex.py:283, cylindrical_complex.py:460 — ``SUB_STEPS``,
# ``CupyPointer``, ``_flat``, ``ConstitutivePlan``). A module-scope import here
# would close that into a cycle, and the failure would land on `import
# meep_gpu.triton_kernels.launch` rather than on anything a reader of this block
# is looking at.
#
# ``no_pml_constitutive`` is the one family here that ships NO KERNEL. Its
# product is a NULL plan, because under an inactive absorber ``stepping.update_H``
# returns at :916-917 before reading any array, and ``update_E`` does the same at
# :954-955 WHEN ``fields.stores_E`` IS ALSO FALSE — the guard there is
# ``if not pml_active and not fields.stores_E``, and dropping its second half
# (as three comments in this file used to) states something measurably untrue:
# on a grid with field storage switched on behind an inert layer ``update_E``
# moved 1536 uint32 words. The family's E-side predicate is what keeps that case
# out, by name and by that attribute. It still goes through the same forwarders,
# the same gate rule and the same :func:`_select_slot`; see
# :func:`absorber_inactive` for why it gets no preference of any kind.


def folded_complex_composition_curl_coverage(fields: Any, pml: Any, sub_step: str,
                                             probe: Any = None) -> Coverage:
    """Ask K1's COMPOSITION verdict — the one that requires an actual fold.

    ``folded_complex_pml_curl_coverage`` admits an unfolded grid deliberately so
    its own gate can prove reduction to the certified unfolded kernel; that is an
    equivalence product, not a routing rule. This is the routing verdict, and it
    is the same split the ``folded PML`` arm already makes.
    """
    from .folded_complex import folded_complex_composition_curl_coverage as predicate  # noqa: PLC0415
    return predicate(fields, pml, sub_step, probe=probe)


def plan_folded_complex_pml_curl(fields: Any, pml: Any, sub_step: str,
                                 block: Optional[int] = None,
                                 probe: Any = None) -> Any:
    """Build one folded complex/Bloch PML curl plan (K1)."""
    from .folded_complex import plan_folded_complex_pml_curl as builder  # noqa: PLC0415
    return builder(fields, pml, sub_step, block, probe=probe)


def folded_beta_pml_curl_coverage(fields: Any, pml: Any, sub_step: str) -> Coverage:
    """Ask the folded REAL special_kz curl predicate (K3a). A fold is mandatory."""
    from .folded_complex import folded_beta_pml_curl_coverage as predicate  # noqa: PLC0415
    return predicate(fields, pml, sub_step)


def plan_folded_beta_pml_curl(fields: Any, pml: Any, sub_step: str,
                              block: Optional[int] = None) -> Any:
    """Build one folded REAL special_kz curl plan (K3a)."""
    from .folded_complex import plan_folded_beta_pml_curl as builder  # noqa: PLC0415
    return builder(fields, pml, sub_step, block)


def folded_beta_bloch_pml_curl_coverage(fields: Any, pml: Any, sub_step: str,
                                        probe: Any = None) -> Coverage:
    """Ask the folded COMPLEX special_kz curl predicate (K3b). A fold is mandatory."""
    from .folded_complex import folded_beta_bloch_pml_curl_coverage as predicate  # noqa: PLC0415
    return predicate(fields, pml, sub_step, probe=probe)


def plan_folded_beta_bloch_pml_curl(fields: Any, pml: Any, sub_step: str,
                                    block: Optional[int] = None,
                                    probe: Any = None) -> Any:
    """Build one folded COMPLEX special_kz curl plan (K3b)."""
    from .folded_complex import plan_folded_beta_bloch_pml_curl as builder  # noqa: PLC0415
    return builder(fields, pml, sub_step, block, probe=probe)


def folded_mirror_ghost_fill_complex_coverage(fields: Any, family: str,
                                              probe: Any = None) -> Coverage:
    """Ask whether one folded family's ghost fills may run on COMPLEX storage (K2)."""
    from .folded_complex import folded_mirror_ghost_fill_complex_coverage as predicate  # noqa: PLC0415
    return predicate(fields, family, probe=probe)


def plan_folded_mirror_ghost_fill_complex(fields: Any, family: str,
                                          block: Optional[int] = None,
                                          probe: Any = None) -> Any:
    """Build one complex-storage mirror-fill plan (K2).

    ITS ADAPTER CONTRACT IS NOT ``MirrorGhostFillPlan``'S, and the only seam an
    adapter can branch on is ``selected[slot]``. ``symmetry.MirrorGhostFillPlan``
    fuses the near and far passes and is entitled to (the parity factors are
    +-1 and float32 multiplication by them is exact); this plan's two passes are
    separated in the driver by ``zero_metal_*`` (driver.py:3209-3211, :3222-3224)
    and complex float multiplication is NOT associative, so running the two
    passes as one is a measured wrong answer (5 words of By, 5 of Dx on a
    two-axis fold). Dispatch is disabled, so this is a documented contract today
    rather than a live hazard — documented HERE because inferring it from the
    plan class is exactly what a driver adapter would get wrong.
    """
    from .folded_complex import plan_folded_mirror_ghost_fill_complex as builder  # noqa: PLC0415
    return builder(fields, family, block, probe=probe)


def folded_complex_constitutive_coverage(fields: Any, pml: Any, side: str,
                                         probe: Any = None) -> Coverage:
    """Ask whether the CERTIFIED complex constitutive kernel may step a folded run.

    Carries NO beta clause, deliberately (``special_kz`` grouping choice 5): the
    constitutive kernel is storage-dependent, not beta-dependent, so this one arm
    serves the folded complex run with and without beta.
    """
    from .folded_complex import folded_complex_constitutive_coverage as predicate  # noqa: PLC0415
    return predicate(fields, pml, side, probe=probe)


def plan_folded_complex_constitutive(fields: Any, pml: Any, side: str,
                                     block: Optional[int] = None,
                                     probe: Any = None) -> Any:
    """Build the certified complex constitutive plan for a folded run."""
    from .folded_complex import plan_folded_complex_constitutive as builder  # noqa: PLC0415
    return builder(fields, pml, side, block, probe=probe)


def folded_beta_run_constitutive_coverage(fields: Any, pml: Any,
                                          side: str) -> Coverage:
    """Ask whether the certified REAL constitutive kernel may step a folded beta run."""
    from .folded_complex import folded_beta_run_constitutive_coverage as predicate  # noqa: PLC0415
    return predicate(fields, pml, side)


def plan_folded_beta_run_constitutive(fields: Any, pml: Any, side: str,
                                      block: Optional[int] = None) -> Any:
    """Build the certified REAL constitutive plan for a folded beta run."""
    from .folded_complex import plan_folded_beta_run_constitutive as builder  # noqa: PLC0415
    return builder(fields, pml, side, block)


def folded_offdiag_composition_coverage(fields: Any, pml: Any) -> Coverage:
    """Ask the folded off-diagonal ``update_E`` COMPOSITION verdict.

    Same standalone/composition split as K1's: the standalone predicate admits an
    unfolded grid so its gate can prove reduction, and the certified
    ``offdiag_update_e`` family owns the unfolded run.
    """
    from .folded_offdiag_update_e import folded_offdiag_composition_coverage as predicate  # noqa: PLC0415
    return predicate(fields, pml)


def plan_folded_offdiagonal_constitutive(fields: Any, pml: Any,
                                          block: Optional[int] = None) -> Any:
    """Build the folded off-diagonal chi1inv row-product ``update_E`` plan."""
    from .folded_offdiag_update_e import plan_folded_offdiagonal_constitutive as builder  # noqa: PLC0415
    return builder(fields, pml, block)


def complex_no_pml_offdiag_update_e_coverage(
        fields: Any, pml: Any, probe: Any = None) -> Coverage:
    """Ask whether the complex absorber-free row-product kernel covers ``update_E``."""
    from .complex_offdiag_update_e import (  # noqa: PLC0415
        complex_no_pml_offdiag_update_e_coverage as predicate)
    return predicate(fields, pml, probe)


def plan_complex_no_pml_offdiag_update_e(
        fields: Any, pml: Any, block: Optional[int] = None,
        probe: Any = None) -> Any:
    """Build the complex absorber-free row-product ``update_E`` plan."""
    from .complex_offdiag_update_e import (  # noqa: PLC0415
        plan_complex_no_pml_offdiag_update_e as builder)
    return builder(fields, pml, block=block, probe=probe)


def complex_folded_offdiag_update_e_coverage(
        fields: Any, pml: Any, probe: Any = None) -> Coverage:
    """Ask whether the complex folded PML row-product kernel covers ``update_E``."""
    from .complex_offdiag_update_e import (  # noqa: PLC0415
        complex_folded_offdiag_update_e_coverage as predicate)
    return predicate(fields, pml, probe)


def plan_complex_folded_offdiag_update_e(
        fields: Any, pml: Any, block: Optional[int] = None,
        probe: Any = None) -> Any:
    """Build the complex folded PML row-product ``update_E`` plan."""
    from .complex_offdiag_update_e import (  # noqa: PLC0415
        plan_complex_folded_offdiag_update_e as builder)
    return builder(fields, pml, block=block, probe=probe)


def cylindrical_complex_curl_coverage(fields: Any, pml: Any, sub_step: str,
                                      probe: Any = None) -> Coverage:
    """Ask the complex Dcyl curl predicate (|m| >= 1)."""
    from .cylindrical_complex import cylindrical_complex_curl_coverage as predicate  # noqa: PLC0415
    return predicate(fields, pml, sub_step, probe=probe)


def plan_cylindrical_complex_curl(fields: Any, pml: Any, sub_step: str,
                                  block: Optional[int] = None,
                                  probe: Any = None) -> Any:
    """Build one complex Dcyl curl plan."""
    from .cylindrical_complex import plan_cylindrical_complex_curl as builder  # noqa: PLC0415
    return builder(fields, pml, sub_step, block, probe=probe)


def cylindrical_complex_constitutive_coverage(fields: Any, pml: Any, side: str,
                                              probe: Any = None) -> Coverage:
    """Ask whether the certified complex constitutive kernel may step a Dcyl slice."""
    from .cylindrical_complex import cylindrical_complex_constitutive_coverage as predicate  # noqa: PLC0415
    return predicate(fields, pml, side, probe=probe)


def plan_cylindrical_complex_constitutive(fields: Any, pml: Any, side: str,
                                          block: Optional[int] = None,
                                          probe: Any = None) -> Any:
    """Build the certified complex constitutive plan for the admitted Dcyl slice."""
    from .cylindrical_complex import plan_cylindrical_complex_constitutive as builder  # noqa: PLC0415
    return builder(fields, pml, side, block, probe=probe)


def null_constitutive_coverage(fields: Any, pml: Any, side: str) -> Coverage:
    """Ask whether this constitutive sub-step is a NO-OP on the array path.

    The only family here whose product launches nothing. Under an inactive
    absorber ``stepping.update_H`` returns at :916-917 before reading any array;
    ``update_E`` returns at :954-955 under ``not pml_active and not
    fields.stores_E``, so the absorber alone is NOT the whole condition on that
    side and the predicate refuses stored E by name. Where both hold the covered
    product is ``NullConstitutivePlan``, not a kernel.
    """
    from .no_pml_constitutive import null_constitutive_coverage as predicate  # noqa: PLC0415
    return predicate(fields, pml, side)


def plan_null_constitutive(fields: Any, pml: Any, side: str,
                           block: Optional[int] = None) -> Any:
    """Build the null constitutive plan for one side, or None when refused."""
    from .no_pml_constitutive import plan_null_constitutive as builder  # noqa: PLC0415
    return builder(fields, pml, side, block)


# ---------------------------------------------------------------------------
# The residual-group ADMISSIONS — lazy forwarders
# ---------------------------------------------------------------------------
#
# Four predicate families, SIX entry points, and not one new kernel between them.
# Each restates a shipped clause set with the ONE clause that was over-broad
# INVERTED, so a body this campaign already certified may step a configuration it
# was refused for. The bodies are ``kernels.pml_curl_step``, ``ConstitutivePlan``,
# ``DispersiveConstitutivePlan``, ``AdeUpdatePPlan``, ``FoldedComplexPmlCurlPlan``
# and ``ComplexConstitutivePlan`` — every one of them unchanged.
#
# The identity that licenses each was measured on an RTX A6000 over eight complete
# cycles, non-vacuous on both sides with a live signed-zero census, THROUGH THESE
# BUILDERS with THESE predicates unpatched
# (results/residual_closure_2026-08-15/device/newpred/new_predicates.json). Each
# family module carries its own numbers; they are not repeated here.
#
# WHY EACH IS AN ARM RATHER THAN A NARROWING OF THE INCUMBENT. Every one of these
# predicates is disjoint from its incumbent BY THE INVERTED CLAUSE — the incumbent
# refuses exactly what this one requires. Narrowing the incumbent's clause instead
# would put TWO admitters on the same slot, which :func:`_select_slot` fails closed
# on: the slot would be left UNSELECTED and fall to the array path, which is the
# silent coverage LOSS this wiring exists to end. The scope change is therefore
# made by ADDING an arm, never by widening one.


def nonlinear_run_pml_curl_coverage(fields: Any, pml: Any,
                                    sub_step: str) -> Coverage:
    """Ask whether the CERTIFIED real PML curl may step a chi2/chi3 run's curls.

    ``coverage._grid_reasons`` clause 10 refuses a nonlinearity on EVERY sub-step;
    the Pade factor it names lives in ``update_E`` alone (stepping.py:999-1000) and
    the curls difference the stored E and H arrays without reading chi2 or chi3 at
    all. This is that clause inverted, and only that clause.
    """
    from .nonlinear_update_e import nonlinear_run_pml_curl_coverage as predicate  # noqa: PLC0415
    return predicate(fields, pml, sub_step)


def plan_nonlinear_run_pml_curl(fields: Any, pml: Any, sub_step: str,
                                block: Optional[int] = None) -> Any:
    """Build the certified real PML curl plan for a chi2/chi3 run."""
    from .nonlinear_update_e import plan_nonlinear_run_pml_curl as builder  # noqa: PLC0415
    return builder(fields, pml, sub_step, block)


def nonlinear_run_constitutive_coverage(fields: Any, pml: Any,
                                        side: str) -> Coverage:
    """Ask whether the CERTIFIED constitutive kernel may step a chi2/chi3 ``update_H``.

    ``side='E'`` is refused BY NAME by the family predicate — that is the side the
    Pade factor replaces, and it belongs to :func:`nonlinear_constitutive_coverage`,
    which is already an arm. Only ``update_H`` consults this one.
    """
    from .nonlinear_update_e import nonlinear_run_constitutive_coverage as predicate  # noqa: PLC0415
    return predicate(fields, pml, side)


def plan_nonlinear_run_constitutive(fields: Any, pml: Any, side: str,
                                    block: Optional[int] = None) -> Any:
    """Build the certified constitutive plan for a chi2/chi3 run's ``update_H``."""
    from .nonlinear_update_e import plan_nonlinear_run_constitutive as builder  # noqa: PLC0415
    return builder(fields, pml, side, block)


def folded_dispersive_constitutive_coverage(fields: Any, pml: Any) -> Coverage:
    """Ask whether the CERTIFIED dispersive ``update_E`` kernel may step a FOLDED run.

    Two inverted clauses, not one: a real fold is REQUIRED (so this cannot compete
    with ``dispersive_constitutive_coverage``, which refuses a fold) and at least
    one pole is REQUIRED (so it cannot compete with the ``folded`` arm, whose
    E-side predicate refuses a registered susceptibility). Disjoint from both
    incumbents in both directions.
    """
    from .folded_dispersive_update_e import (  # noqa: PLC0415
        folded_dispersive_constitutive_coverage as predicate)
    return predicate(fields, pml)


def plan_folded_dispersive_constitutive(fields: Any, pml: Any,
                                        block: Optional[int] = None) -> Any:
    """Build the certified dispersive ``update_E`` plan for a folded run."""
    from .folded_dispersive_update_e import (  # noqa: PLC0415
        plan_folded_dispersive_constitutive as builder)
    return builder(fields, pml, block)


def folded_offdiag_dispersive_constitutive_coverage(
        fields: Any, pml: Any) -> Coverage:
    """Ask whether the folded row-product plus pole-aware ``update_E`` is covered."""
    from .folded_offdiag_dispersive_update_e import (  # noqa: PLC0415
        folded_offdiag_dispersive_constitutive_coverage as predicate)
    return predicate(fields, pml)


def plan_folded_offdiag_dispersive_constitutive(
        fields: Any, pml: Any, block: Optional[int] = None) -> Any:
    """Build the folded row-product plus pole-aware ``update_E`` plan."""
    from .folded_offdiag_dispersive_update_e import (  # noqa: PLC0415
        plan_folded_offdiag_dispersive_constitutive as builder)
    return builder(fields, pml, block)


def folded_complex_offdiag_pml_curl_coverage(fields: Any, pml: Any, sub_step: str,
                                             probe: Any = None) -> Coverage:
    """Ask whether K1's curl may step a folded complex run carrying an off-diagonal row.

    THE SCOPE CHANGE TWO BACKENDS LICENSED. ``folded_complex._media_reasons``'
    off-diagonal clause is shared by the incumbent curl and constitutive
    predicates and refuses the row on ALL FOUR sub-steps; the row product it names
    (``stepping._offdiagonal_terms``, S:1190-1225) is reached from ``update_E``
    alone (S:972-979). This arm — with the incumbent's clause left exactly as it
    is — is what narrows that refusal to ``update_E``.
    """
    from .folded_complex import (  # noqa: PLC0415
        folded_complex_offdiag_pml_curl_coverage as predicate)
    return predicate(fields, pml, sub_step, probe=probe)


def plan_folded_complex_offdiag_pml_curl(fields: Any, pml: Any, sub_step: str,
                                         block: Optional[int] = None,
                                         probe: Any = None) -> Any:
    """Build K1's certified curl plan for a folded complex off-diagonal run."""
    from .folded_complex import (  # noqa: PLC0415
        plan_folded_complex_offdiag_pml_curl as builder)
    return builder(fields, pml, sub_step, block, probe=probe)


def folded_complex_offdiag_constitutive_coverage(fields: Any, pml: Any, side: str,
                                                 probe: Any = None) -> Coverage:
    """Ask whether the certified complex constitutive kernel may step that run's ``update_H``.

    ``side='E'`` is refused BY NAME and STAYS refused: 25041 of 110592 words
    differing over eight complete cycles, the largest divergence the closure round
    measured. That slot needs a complex folded off-diagonal kernel, which is not
    built, and this arm must never claim it.
    """
    from .folded_complex import (  # noqa: PLC0415
        folded_complex_offdiag_constitutive_coverage as predicate)
    return predicate(fields, pml, side, probe=probe)


def plan_folded_complex_offdiag_constitutive(fields: Any, pml: Any, side: str,
                                             block: Optional[int] = None,
                                             probe: Any = None) -> Any:
    """Build the certified complex constitutive plan for that run's ``update_H``."""
    from .folded_complex import (  # noqa: PLC0415
        plan_folded_complex_offdiag_constitutive as builder)
    return builder(fields, pml, side, block, probe=probe)


def no_pml_ade_update_p_coverage(fields: Any, pml: Any, state: Any,
                                 component: str) -> Coverage:
    """Ask whether the CERTIFIED ADE kernel may advance a pole with the absorber INERT.

    ``coverage.ade_update_p_coverage`` pins the EXISTENCE of ``f_w_*``; what the
    kernel needs is the LAYOUT of whatever ``Fields.drive_field`` hands it, which
    without an absorber is the stored E (fields.py:1158-1162). The inverted clause
    REQUIRES the layer to be inert, so the two are disjoint — and the closure
    round's CONTROL, which bound the stored E as the drive UNDER an active layer,
    diverged 3072/15360 words, which is what the inverted clause keeps out.
    """
    from .no_pml_ade import no_pml_ade_update_p_coverage as predicate  # noqa: PLC0415
    return predicate(fields, pml, state, component)


def plan_no_pml_ade_update_p(fields: Any, pml: Any, state: Any,
                             block: Optional[int] = None) -> Any:
    """Build the certified ADE ``update_P`` plan for an absorber-free run."""
    from .no_pml_ade import plan_no_pml_ade_update_p as builder  # noqa: PLC0415
    return builder(fields, pml, state, block)


def complex_ade_update_p_coverage(fields: Any, pml: Any, state: Any,
                                  component: str,
                                  probe: Any = None) -> Coverage:
    """Ask whether complex64 absorber-free ADE covers one driven component."""
    from .complex_ade import complex_ade_update_p_coverage as predicate  # noqa: PLC0415
    return predicate(fields, pml, state, component, probe)


def plan_complex_ade_update_p(fields: Any, pml: Any, state: Any,
                              block: Optional[int] = None,
                              probe: Any = None) -> Any:
    """Build one complex64 absorber-free susceptibility plan."""
    from .complex_ade import plan_complex_ade_update_p as builder  # noqa: PLC0415
    return builder(fields, pml, state, block, probe)


# ---------------------------------------------------------------------------
# The no-absorber closure — lazy forwarders
# ---------------------------------------------------------------------------
#
# These are distinct family modules, rather than variants inside ``no_pml`` or
# ``complex_fields``: two contain new Triton bodies and the complex curl carries
# a measured degenerate binding.  Keep every import inside its forwarder so a
# run that cannot meet the narrow gate neither imports nor compiles the family.


def complex_no_pml_curl_coverage(fields: Any, pml: Any, sub_step: str,
                                 probe: Any = None) -> Coverage:
    """Ask whether the complex no-absorber curl binding covers this sub-step."""
    from .complex_no_pml_curl import (  # noqa: PLC0415
        complex_no_pml_curl_coverage as predicate)
    return predicate(fields, pml, sub_step, probe=probe)


def plan_complex_no_pml_curl(fields: Any, pml: Any, sub_step: str,
                             block: Optional[int] = None,
                             probe: Any = None) -> Any:
    """Build the complex no-absorber curl binding."""
    from .complex_no_pml_curl import plan_complex_no_pml_curl as builder  # noqa: PLC0415
    return builder(fields, pml, sub_step, block, probe=probe)


def complex_conductive_no_pml_curl_coverage(
        fields: Any, pml: Any, sub_step: str,
        probe: Any = None) -> Coverage:
    """Ask whether the complex conductive no-absorber curl covers this sub-step."""
    from .complex_no_pml_conductive import (  # noqa: PLC0415
        complex_conductive_no_pml_curl_coverage as predicate)
    return predicate(fields, pml, sub_step, probe=probe)


def plan_complex_conductive_no_pml_curl(
        fields: Any, pml: Any, sub_step: str,
        block: Optional[int] = None, probe: Any = None) -> Any:
    """Build the complex conductive no-absorber curl plan."""
    from .complex_no_pml_conductive import (  # noqa: PLC0415
        plan_complex_conductive_no_pml_curl as builder)
    return builder(fields, pml, sub_step, block, probe=probe)


def conductive_plain_curl_coverage(fields: Any, pml: Any,
                                   sub_step: str) -> Coverage:
    """Ask whether the conductive no-absorber curl covers this sub-step."""
    from .no_pml_conductive import (  # noqa: PLC0415
        conductive_plain_curl_coverage as predicate)
    return predicate(fields, pml, sub_step)


def plan_conductive_plain_curl(fields: Any, pml: Any, sub_step: str,
                               block: Optional[int] = None) -> Any:
    """Build the conductive no-absorber curl plan."""
    from .no_pml_conductive import plan_conductive_plain_curl as builder  # noqa: PLC0415
    return builder(fields, pml, sub_step, block)


def stored_e_constitutive_coverage(fields: Any, pml: Any) -> Coverage:
    """Ask whether stored-E no-absorber ``update_E`` is covered."""
    from .no_pml_stored_e import stored_e_constitutive_coverage as predicate  # noqa: PLC0415
    return predicate(fields, pml)


def plan_stored_e_constitutive(fields: Any, pml: Any,
                               block: Optional[int] = None) -> Any:
    """Build the stored-E no-absorber constitutive plan."""
    from .no_pml_stored_e import plan_stored_e_constitutive as builder  # noqa: PLC0415
    return builder(fields, pml, block)


def complex_stored_e_coverage(fields: Any, pml: Any,
                              probe: Any = None) -> Coverage:
    """Ask whether complex pole-aware stored-E ``update_E`` is covered."""
    from .complex_no_pml_stored_e import (  # noqa: PLC0415
        complex_stored_e_coverage as predicate)
    return predicate(fields, pml, probe=probe)


def plan_complex_stored_e(fields: Any, pml: Any,
                          block: Optional[int] = None,
                          probe: Any = None) -> Any:
    """Build the complex pole-aware stored-E ``update_E`` plan."""
    from .complex_no_pml_stored_e import plan_complex_stored_e as builder  # noqa: PLC0415
    return builder(fields, pml, block, probe=probe)


# ---------------------------------------------------------------------------
# The gates: which arms are even CONSULTED for a configuration
# ---------------------------------------------------------------------------
#
# THE GATE RULE, stated so it stays checkable: a gate must be a NECESSARY
# condition of its arm's admission — not-gate implies the arm's own predicate
# would have refused anyway — and it must be computed from the SAME attribute
# the arm's inverted clause reads. Under that rule gating removes REFUSALS only,
# never admissions, so it can never be the reason a competing arm wins and can
# never suppress an ambiguity. Each gate below is its family's inverted clause
# transcribed, and a laptop test calls the family predicate directly on a
# gate-False double to check the implication rather than assume it.


def complex_storage_active(fields: Any) -> bool:
    """Complex64 storage — the complex family's inverted clause 2, transcribed.

    Unreadable consults the arm, exactly as :func:`beta_active` does. Every gate
    in this block follows that rule: the gate block runs BEFORE any predicate,
    so a gate that crashed on an unreadable attribute would take down the whole
    composition — including the seven slots that arm has nothing to do with —
    where the file's own rule is that an unreadable gate consults its arm and
    lets the arm's predicate refuse by name. EVERY attribute read a gate makes,
    ``fields.grid`` included, is therefore inside the try — see
    :func:`cylindrical_grid_active` for what a hoisted one measured.
    """
    try:
        grid = getattr(fields, "grid", None)
        return (bool(getattr(fields, "force_complex_fields", False))
                or bool(getattr(grid, "has_bloch", False)))
    except Exception:  # noqa: BLE001 - an unreadable gate is consulted, not assumed
        return True


def beta_active(fields: Any) -> bool:
    """Nonzero ``grid.beta`` — special_kz's inverted clause 12, transcribed.

    An UNREADABLE beta consults the arm rather than skipping it: refusing to
    read is not evidence the family does not apply, and the family predicate is
    the thing entitled to decide. That covers an unreadable ``fields.grid`` too,
    which is why the read is inside the try (:func:`cylindrical_grid_active`).
    """
    try:
        grid = getattr(fields, "grid", None)
        return float(getattr(grid, "beta", 0.0)) != 0.0
    except Exception:  # noqa: BLE001 - an unreadable beta is consulted, not assumed
        return True


def bfast_grid_active(fields: Any) -> bool:
    """``grid.bfast_active`` — the BFAST family's inverted clause 11, transcribed."""
    try:
        return bool(getattr(getattr(fields, "grid", None), "bfast_active", False))
    except Exception:  # noqa: BLE001 - an unreadable gate is consulted, not assumed
        return True


def nonlinear_active(fields: Any) -> bool:
    """chi2/chi3 installed — the nonlinear family's inverted clause 10, transcribed."""
    try:
        return bool(getattr(fields, "has_nonlinearity", False))
    except Exception:  # noqa: BLE001 - an unreadable gate is consulted, not assumed
        return True


def dispersion_active(fields: Any) -> bool:
    """A pole is REGISTERED — the folded dispersive family's inverted clause.

    ``folded_dispersive_constitutive_coverage`` inverts ``symmetry.py``'s
    susceptibility clause: with no state registered ``update_E``'s source is D
    rather than ``(D - sum P)``, and that configuration belongs to the ``folded``
    arm. So an empty polarization list means that predicate would have refused,
    which is what the gate rule requires, and the gate reads the SAME attribute
    the inverted clause reads.

    An unreadable list CONSULTS the arm, this file's rule everywhere: refusing to
    read is not evidence that no pole is installed.
    """
    try:
        return bool(tuple(getattr(fields, "polarizations", ()) or ()))
    except Exception:  # noqa: BLE001 - an unreadable gate is consulted, not assumed
        return True


def absorber_inactive(pml: Any) -> bool:
    """No ACTIVE absorber — the null family's inverted clause 2, transcribed.

    ``no_pml_constitutive._inactive_layer_reasons`` refuses an active layer
    (no_pml_constitutive.py:417-419) and refuses one that cannot answer
    (:413-416), reading exactly ``pml.is_active`` — the attribute
    ``stepping._pml_is_active`` (stepping.py:2498-2506) branches on at
    stepping.py:944 and :982. An unreadable layer CONSULTS the arm, this file's
    rule everywhere; the arm then refuses it by name.

    THE ONLY GATE HERE THAT READS THE LAYER RATHER THAN THE FIELDS, because the
    only family it guards is the one whose product is a NULL plan. Without it the
    null arm's refusal string would be appended to every unselected constitutive
    slot on every active-PML configuration in the suite; with it, only an
    inactive-layer run ever sees that arm at all.
    """
    try:
        return not bool(getattr(pml, "is_active", False))
    except Exception:  # noqa: BLE001 - an unreadable gate is consulted, not assumed
        return True


def offdiag_rows_possible(fields: Any) -> bool:
    """The DISJUNCTION of both off-diagonal triggers — deliberately the weaker gate.

    The family predicate counts live ROW SLOTS through
    ``Fields.chi1inv_offdiagonal_for``; the shipped E-side predicate reads the
    ``has_offdiagonal_epsilon`` FLAG. Where those two disagree BOTH admit, and
    that ambiguity has to survive to be refused: gating on the flag alone would
    suppress the specialized arm exactly there, the ordinary product would win
    unopposed, and the row would be silently dropped.

    REACHABILITY, MEASURED rather than assumed. Against the engine's own class
    the flag-False/slot-live direction CANNOT occur: ``has_offdiagonal_epsilon``
    is a read-only property returning ``bool(self._chi1inv_offdiagonal)`` and
    ``chi1inv_offdiagonal_for`` reads the same dict (fields.py:1313-1319), so
    they move together and the property has no setter. What IS reachable on the
    engine is the OPPOSITE direction — a row planted past
    ``_validated_offdiagonal_rows`` under a diagonal key sets the flag with
    every slot dead — and there every arm refuses (the shipped one on the flag,
    the specialized one on the slot count), which is safe.

    So the disjunction is kept for the callers that CAN produce the disagreement
    — the gates, the probes and every harness that builds a ``Fields`` double —
    and it is no longer the whole answer: :func:`live_offdiagonal_rows` vetoes a
    selected ``update_E`` product that does not implement a live row, so a
    disagreement that reaches a slot fails closed instead of dropping the
    coupling. The gate keeps the ambiguity VISIBLE; the veto is what makes the
    outcome safe when only one arm survives the other clauses.

    The implication the gate rule needs still holds in the other direction: a
    ``chi1inv_offdiagonal_for`` that is not callable makes every slot dead
    (the family reads it with a ``{}`` default), so not-gate implies the family
    predicate refuses.
    """
    try:
        return (bool(getattr(fields, "has_offdiagonal_epsilon", False))
                or callable(getattr(fields, "chi1inv_offdiagonal_for", None)))
    except Exception:  # noqa: BLE001 - an unreadable gate is consulted, not assumed
        return True


def live_offdiagonal_rows(fields: Any) -> bool:
    """Whether any off-diagonal chi1inv ROW SLOT is live, counted the family's way.

    Delegates to ``offdiag_update_e.row_volumes_for`` — the single place the slot
    binding is derived from ``Fields.chi1inv_offdiagonal_for`` — so this veto and
    the family predicate cannot disagree about what "a live row" means.

    Only ever consulted when the accessor exists, which is the same condition
    :func:`offdiag_rows_possible` gates its arm on, so this opens no import the
    composition did not already open. An accessor that RAISES answers True: the
    coupling cannot be shown dead, and the fail-closed answer for a product that
    would drop it is the array path.

    THE ACCESSOR LOOKUP IS INSIDE THE TRY, and the "unreadable is not dead"
    intent above is exactly why. It used to sit outside, so a ``fields`` whose
    ``chi1inv_offdiagonal_for`` attribute raised on ACCESS — rather than on call
    — propagated out of the veto and out of :func:`plan_step`, measured as
    ``RuntimeError`` escaping on five of the six base configurations of the
    attribute fuzz (the off-diagonal one survived only because the veto returns
    early there). The veto is the one block in ``plan_step`` with no
    ``_guarded_*`` wrapper around it, so this function is where that guard lives.
    """
    try:
        if not callable(getattr(fields, "chi1inv_offdiagonal_for", None)):
            return False
        from .offdiag_update_e import row_volumes_for  # noqa: PLC0415
        return any(volume is not None for volume in row_volumes_for(fields))
    except Exception:  # noqa: BLE001 - an unreadable row is not a dead row
        return True


#: Families whose predicates exist in this package but which :func:`plan_step`
#: does NOT consult: their gates are in flight or unbuilt. Adding one is a
#: coordinated change that re-runs the byte gate; until then their grids fall to
#: the array path because every consulted arm refuses them by name.
#:
#: EMPTY, and that is NOT the same claim as "every module in this package is an
#: arm" — which is what the four-family round's version of this comment said, and
#: which is false by one module. The whole-chain dispersive fusion (``step_D`` +
#: ``update_E`` + ``update_P`` in ONE launch, a level above the ``fuse`` and
#: ``fuse_ade`` seams below) is not an arm, has no gate, and has no slot in
#: :data:`STEP_ORDER` it could claim without a composition rule nothing has
#: measured. It cannot be NAMED here either: its own test asserts this file never
#: mentions it, which is the seam that keeps a deferred product deferred. So the
#: EXHAUSTIVE account of the package — every module in exactly one of "gated
#: family", "support import", "module-scope import" and "not an arm" — lives in
#: the planner composition test, where it is derived from the AST of this file
#: and from the package directory rather than from a list either one repeats.
#:
#: The name is kept, not deleted, because it is the seam a future uncertified
#: family declares itself on when nothing forbids naming it.
RESERVED_FAMILIES: Tuple[str, ...] = ()

#: The GATED family modules — one per family, each consulted
#: through a per-arm gate in :func:`plan_step` — named once so the lazy-import
#: seam can be checked against a list rather than against whatever happens to be
#: in ``sys.modules``. This is NOT the whole of what :func:`plan_step` imports;
#: :data:`SUPPORT_MODULES` is the rest, and the two together are what the AST
#: seam test compares against the shipped file.
#:
#: NONE of these may be imported at module scope. Two of them
#: (``folded_complex``, ``cylindrical_complex``) import FROM THIS MODULE at their
#: own module scope, so a module-scope import here closes an import cycle; and
#: every one of them must stay absent from ``sys.modules`` after a bare package
#: import, which each family's own test pins on a machine with no Triton. The
#: forwarders above are what keep both true: the import lives inside the body.
#:
#: New residual families remain separate from older support modules because
#: their gates are conditional and must stay lazy.
FAMILY_MODULES: Tuple[str, ...] = (
    "complex_fields",
    "special_kz",
    "bfast_curl",
    "nonlinear_update_e",
    "offdiag_update_e",
    "folded_complex",
    "folded_offdiag_update_e",
    "cylindrical_complex",
    "no_pml_constitutive",
    "folded_dispersive_update_e",
    "folded_offdiag_dispersive_update_e",
    "complex_offdiag_update_e",
    "complex_ade",
    "no_pml_ade",
    "complex_no_pml_curl",
    "complex_no_pml_conductive",
    "no_pml_conductive",
    "no_pml_stored_e",
    "complex_no_pml_stored_e",
)

#: The other sibling modules :func:`plan_step` imports from inside a function
#: body, and the reason :data:`FAMILY_MODULES` alone was never the whole
#: lazy-import seam. Four are arms with a gate of their own (``symmetry`` behind
#: the fold, ``cylindrical_triton`` behind the cylindrical axis) or with none
#: (``conductivity``, ``no_pml`` are consulted on every configuration); two are
#: reached only through the opt-in ``fuse`` / ``fuse_ade`` blocks; ``kernels`` is
#: the kernel source itself, imported by the builders here.
#:
#: They are separated from the families because the gated-import test measures a
#: different property of each — a family module must stay OUT of ``sys.modules``
#: when its gate is off, which is meaningless for a module with no gate — not
#: because the lazy-import rule is any weaker for them.
SUPPORT_MODULES: Tuple[str, ...] = (
    "kernels",
    "conductivity",
    "no_pml",
    "symmetry",
    "cylindrical_triton",
    "dispersive_fused_pair",
    "fused_ade_state",
    # THE TWO FOLDED PAIRS, routed 2026-08-27 by :func:`_install_folded_fused_pairs`.
    # They sit here beside ``dispersive_fused_pair`` rather than in
    # ``FAMILY_MODULES`` because a fused pair is not an arm: it never registers on
    # a slot, it is reached only from the opt-in ``fuse`` block, and it takes TWO
    # slots when it is reached. Both were in ``test_triton_planner_composition``'s
    # ``NOT_AN_ARM`` until this change, on the ground that no composition rule had
    # been measured for them; ``FOLDED_FUSED_PAIR_ARMS`` is that rule.
    "folded_fused_magnetic_pair",
    "folded_fused_pair",
    # THE INSTALLER WAVE, 2026-09-02. Twenty-five certified products routed
    # through :func:`_install_certified_fused_products`, which is one seam loop
    # over :data:`CERTIFIED_FUSED_PRODUCTS` rather than one branch per product.
    # They sit here for the reason the two folded pairs do and not a new one: a
    # fused pair is not an arm — it never registers on a slot, ``_select_slot``
    # never sees it, and it takes TWO slots when the fusion block reaches it — so
    # the "two admitters leave the slot UNSELECTED" hazard that kept several of
    # them out of ``FAMILY_MODULES`` does not apply to this route at all.
    #
    # THE LIST IS NOT MAINTAINED BY HAND AGAINST THE TABLE: the AST seam test
    # compares this declaration with the modules this file actually imports, and
    # ``test_triton_certified_fused_products`` additionally requires that the two
    # sets agree — a product added to the table without a line here fails both.
    "beta_fused_electric_pair",
    "beta_fused_magnetic_pair",
    "bfast_fused_electric_pair",
    "bfast_fused_magnetic_pair",
    "complex_beta_fused_electric_pair",
    "complex_beta_fused_magnetic_pair",
    "complex_conductive_fused_pair",
    "complex_fused_electric_pair",
    # THE COMPLEX H->D PRODUCT, 2026-09-07: routed for the reason the two
    # cylindrical H->D products are -- so the composer records its own
    # INSTALLABLE = False refusal by name on every complex row rather than leaving
    # an absence, and so the label boundary the Cartesian H->D product names is
    # closed for this cell of the seam too.
    "complex_fused_hd_pair",
    "complex_fused_magnetic_pair",
    "conductive_fused_electric_pair",
    "cylindrical_fused_electric_pair",
    "cylindrical_fused_magnetic_pair",
    "cylindrical_real_fused_electric_pair",
    "cylindrical_real_fused_magnetic_pair",
    # THE TWO CYLINDRICAL H->D PRODUCTS, 2026-09-07: routed so the composer records
    # their own refusal by name on every cylindrical row (both declare
    # INSTALLABLE = False for the arbitration reason each module states), and so
    # the label boundary the Cartesian H->D product names is closed for the seam.
    "cylindrical_fused_hd_pair",
    "cylindrical_real_fused_hd_pair",
    "folded_beta_complex_fused_magnetic_pair",
    "folded_beta_complex_fused_pair",
    "folded_beta_fused_electric_pair",
    "folded_beta_fused_magnetic_pair",
    "folded_complex_fused_magnetic_pair",
    "folded_complex_fused_pair",
    "folded_dispersive_fused_pair",
    # THE TWO SCRATCH-OUTPUT OFF-DIAGONAL D->E WELDS, 2026-09-15: routed once their
    # shared plan base gained a ``warm`` that never rotates. See the note above
    # :data:`CERTIFIED_FUSED_PAIR_SEAMS`.
    "folded_offdiag_fused_electric_pair",
    "no_pml_fused_electric_pair",
    "nonlinear_fused_magnetic_pair",
    "offdiag_fused_electric_pair",
)

#: Which axis pair the split-field recurrence reads for target 0, 1 and 2. Both
#: sub-steps take the same triple — vec.hpp's cycle_direction, X->Y->Z, with the
#: component's own axis picking dsig (next) and dsigu (next again).
DSIG_AXES: Tuple[Tuple[int, int], ...] = ((1, 2), (2, 0), (0, 1))

#: Measured policy for both cross-sub-step pairs at BLOCK=256. The complete
#: two-pair benchmark is recorded in
#: ``parity/meep_gpu/results/triton_two_pair_2026-08-11``: the fair 2-D/3-D
#: sweep selected one warp for B/H and D/E, and the 12-case shape/boundary policy
#: sweep had a 1.0739x median with a 0.99935x worst case. Passing ``None`` still
#: requests Triton's own default explicitly for diagnostic comparisons.
FUSED_DEFAULT_NUM_WARPS = 1

#: Targets, auxiliaries and sources per sub-step, in target order.
SUB_STEPS: Dict[str, Dict[str, Any]] = {
    "step_B": {
        "targets": ("Bx", "By", "Bz"),
        "sources": ("Ex", "Ey", "Ez"),
        "backward": 0,
        # The B curl reads HALF-INTEGER positions; getting this backwards is a
        # silent half-cell error, not a crash (stepping._curl_coefficients).
        "suffix": "_h",
    },
    "step_D": {
        "targets": ("Dx", "Dy", "Dz"),
        "sources": ("Hx", "Hy", "Hz"),
        "backward": 1,
        "suffix": "",
    },
}


class CupyPointer:
    """The whole CuPy-to-Triton interop layer.

    Triton's launcher resolves a pointer argument by calling ``arg.data_ptr()``
    and types it from ``arg.dtype``; a ``cupy.ndarray`` carries the address under
    a different name and is rejected outright. This adapter renames it. No torch
    tensor, no host round-trip, no device copy — the wrapped array is the same
    allocation the engine already holds, and the engine's own references to it
    stay live across the in-place update.
    """

    __slots__ = ("array",)

    def __init__(self, array: Any) -> None:
        self.array = array

    def data_ptr(self) -> int:
        return int(self.array.data.ptr)

    @property
    def dtype(self):
        return self.array.dtype


def require_triton() -> None:
    """Import Triton, or explain both ways it fails on a fresh box.

    The ``libcuda.so`` case is not a quirk of one machine: Triton JIT-compiles a
    small ``cuda_utils`` C extension at first launch and links it with ``-lcuda``,
    which needs a DEVELOPMENT symlink that a driver-only CUDA install does not
    ship (it has ``libcuda.so.1`` and nothing else). Without the hint the first
    launch dies in a ``/usr/bin/ld: cannot find -lcuda`` with no visible
    connection to what the caller did.
    """
    try:
        import triton  # noqa: F401, PLC0415
    except ImportError as exc:  # pragma: no cover - exercised by the absence test
        raise ImportError(
            "the Triton kernel track needs the optional `triton` package "
            "(pip install triton). The engine runs without it; only this fast "
            f"path is unavailable. Original error: {exc}"
        ) from exc
    import os  # noqa: PLC0415
    if "TRITON_LIBCUDA_PATH" not in os.environ and not _has_libcuda_dev_symlink():
        raise RuntimeError(
            "Triton links its cuda_utils extension with -lcuda, which needs a "
            "libcuda.so development symlink; this system appears to ship only "
            "libcuda.so.1. Create one in a directory you own and export "
            "TRITON_LIBCUDA_PATH to it:\n"
            "    mkdir -p ~/triton_libcuda_stub\n"
            "    ln -sf $(ldconfig -p | awk '/libcuda.so.1/ {print $NF; exit}') "
            "~/triton_libcuda_stub/libcuda.so\n"
            "    export TRITON_LIBCUDA_PATH=$HOME/triton_libcuda_stub"
        )


def _has_libcuda_dev_symlink() -> bool:
    import glob  # noqa: PLC0415
    import os  # noqa: PLC0415
    for root in ("/usr/lib/x86_64-linux-gnu", "/usr/lib64", "/usr/local/cuda/lib64"):
        if glob.glob(os.path.join(root, "libcuda.so")):
            return True
    return False


class PmlCurlPlan:
    """A launchable, allocation-free real-field PML curl sub-step.

    Built two ways and launched ONE way. :func:`plan_pml_curl` is the engine route
    — it reads a real ``Fields``/``PML`` and passes through the coverage predicate;
    :func:`plan_from_arrays` is the gate and benchmark route, which hands over bare
    device arrays. Both produce this object and both go through :meth:`run`, so the
    bytes the gate certifies are the bytes the engine would launch.
    """

    __slots__ = ("sub_step", "shape", "n_elem", "dtdx", "backward", "bc", "block",
                 "num_warps", "_targets", "_aux", "_sources", "_coefficients",
                 "_grid", "_kernel")

    def __init__(self, sub_step: str, shape, dtdx: float, bc, block: int,
                 targets, auxiliaries, sources, coefficients, kernel=None,
                 num_warps: Optional[int] = None) -> None:
        self.sub_step = sub_step
        self.shape = tuple(int(n) for n in shape)
        self.n_elem = self.shape[0] * self.shape[1] * self.shape[2]
        # float(): the array path multiplies a float32 volume by a Python float,
        # which NumPy/CuPy cast to float32 before the multiply. Triton types a
        # Python float argument as fp32, so the two scalars are the same bits.
        self.dtdx = float(dtdx)
        self.backward = int(SUB_STEPS[sub_step]["backward"])
        self.bc = tuple(int(code) for code in bc)
        self.block = int(block)
        self.num_warps = None if num_warps is None else int(num_warps)
        self._targets = tuple(CupyPointer(a) for a in targets)
        self._aux = tuple(CupyPointer(a) for a in auxiliaries)
        self._sources = tuple(CupyPointer(a) for a in sources)
        self._coefficients = tuple(CupyPointer(_flat(a)) for a in coefficients)
        self._grid = ((self.n_elem + self.block - 1) // self.block,)
        # An override exists for exactly one caller: the gate's source-mutation
        # leg, which compiles a deliberately broken copy of the shipped kernel.
        self._kernel = kernel

    def run(self, guard: Optional[bool] = None) -> None:
        """Launch the sub-step. In place; the driver's references stay valid.

        ``guard`` is not for callers: it exists so the gate can measure the
        contraction guard's effect (identical with, non-identical without) rather
        than assert it. Everything else takes the module constant.
        """
        from .kernels import ENABLE_FP_FUSION, pml_curl_step  # noqa: PLC0415

        nx, ny, nz = self.shape
        kernel = self._kernel if self._kernel is not None else pml_curl_step
        extra = {} if self.num_warps is None else {"num_warps": self.num_warps}
        kernel[self._grid](
            *self._targets, *self._aux, *self._sources, *self._coefficients,
            nx, ny, nz, self.n_elem, self.dtdx,
            BACKWARD=self.backward,
            BCX=self.bc[0], BCY=self.bc[1], BCZ=self.bc[2],
            BLOCK=self.block,
            enable_fp_fusion=ENABLE_FP_FUSION if guard is None else bool(guard),
            **extra,
        )

    def __repr__(self) -> str:
        return (f"PmlCurlPlan({self.sub_step}, shape={self.shape}, bc={self.bc}, "
                f"block={self.block}, num_warps={self.num_warps})")


def plan_pml_curl(fields: Any, pml: Any, sub_step: str,
                  block: Optional[int] = None,
                  num_warps: Optional[int] = None) -> Optional[PmlCurlPlan]:
    """Build a plan from the engine's own objects, or None when out of coverage.

    None is the only refusal: a configuration this kernel does not carry must fall
    back to the array path, never raise into a caller that would otherwise have
    stepped correctly.
    """
    if sub_step not in SUB_STEPS:
        raise ValueError(f"sub_step must be one of {tuple(SUB_STEPS)}, got {sub_step!r}")
    if not pml_curl_coverage(fields, pml, sub_step).covered:
        return None
    from ..stepping import _boundary_kinds  # noqa: PLC0415
    from .kernels import DEFAULT_BLOCK  # noqa: PLC0415

    spec = SUB_STEPS[sub_step]
    grid = fields.grid
    kinds = _boundary_kinds(grid, pml)
    return PmlCurlPlan(
        sub_step, grid.shape, grid.dt / grid.dx,
        [1 if kind == "metallic" else 0 for kind in kinds],
        DEFAULT_BLOCK if block is None else block,
        [getattr(fields, n) for n in spec["targets"]],
        [getattr(fields, "fu_" + n) for n in spec["targets"]],
        [getattr(fields, n) for n in spec["sources"]],
        [getattr(pml, f"{stem}_{axis}{spec['suffix']}")
         for axis in "xyz" for stem in ("kms", "sinv")],
        num_warps=num_warps,
    )


def plan_from_arrays(sub_step: str, arrays: Dict[str, Any], flat: Dict[str, Any],
                     codes, dtdx: float, block: Optional[int] = None,
                     kernel: Any = None,
                     num_warps: Optional[int] = None) -> PmlCurlPlan:
    """Build a plan from bare device arrays — the gate's and benchmark's route.

    ``arrays`` is keyed by component name, ``flat`` by ``kms_x``/``sinv_x``/... on
    the sub-lattice the caller already selected, and ``codes`` is the per-axis
    0/1 periodic/metallic triple. No coverage predicate runs here: the caller is a
    harness that has constructed the configuration deliberately, and refusing it
    would defeat the point of a mutation leg.
    """
    from .kernels import DEFAULT_BLOCK  # noqa: PLC0415

    spec = SUB_STEPS[sub_step]
    shape = tuple(int(n) for n in arrays[spec["targets"][0]].shape)
    return PmlCurlPlan(
        sub_step, shape, dtdx, codes, DEFAULT_BLOCK if block is None else block,
        [arrays[n] for n in spec["targets"]],
        [arrays["fu_" + n] for n in spec["targets"]],
        [arrays[n] for n in spec["sources"]],
        [flat[f"{stem}_{axis}"] for axis in "xyz" for stem in ("kms", "sinv")],
        kernel=kernel, num_warps=num_warps,
    )


# ---------------------------------------------------------------------------
# The constitutive sub-step (update_H / update_E)
# ---------------------------------------------------------------------------

class ConstitutivePlan:
    """A launchable, allocation-free ``dsigw`` constitutive sub-step.

    The Yee sub-lattice is chosen HERE and nowhere else: ``kps_a``/``kms_a`` for the
    H side, ``kps_a_h``/``kms_a_h`` for the E side (``stepping.py:948`` vs ``:986``).
    The kernel takes six coefficient pointers and never asks which lattice they came
    from, so a swap on this line is a silent half-cell error in the absorber profile
    — converged, smooth, and wrong. The gate carries a mutation for exactly it.
    """

    __slots__ = ("side", "shape", "n_elem", "scale", "block", "num_warps",
                 "_targets", "_aux", "_sources", "_inv_eps", "_coefficients",
                 "_grid", "_kernel")

    def __init__(self, side: str, shape, block: int, targets, auxiliaries, sources,
                 inverse_epsilon, coefficients, kernel=None,
                 num_warps: Optional[int] = None) -> None:
        self.side = side
        self.shape = tuple(int(n) for n in shape)
        self.n_elem = self.shape[0] * self.shape[1] * self.shape[2]
        self.scale = 1 if side == "E" else 0
        self.block = int(block)
        self.num_warps = None if num_warps is None else int(num_warps)
        self._targets = tuple(CupyPointer(a) for a in targets)
        self._aux = tuple(CupyPointer(a) for a in auxiliaries)
        self._sources = tuple(CupyPointer(a) for a in sources)
        # The H side has no inverse epsilon. Bind the sources as placeholders rather
        # than a null: the loads sit behind a `tl.constexpr` branch and are compiled
        # away, but a pointer argument still has to type, and passing None makes the
        # launcher's failure a TypeError far from its cause.
        self._inv_eps = (tuple(CupyPointer(a) for a in inverse_epsilon)
                         if inverse_epsilon is not None else self._sources)
        self._coefficients = tuple(CupyPointer(_flat(a)) for a in coefficients)
        self._grid = ((self.n_elem + self.block - 1) // self.block,)
        self._kernel = kernel

    def run(self, guard: Optional[bool] = None) -> None:
        """Launch the sub-step, in place. Same ``guard`` contract as the curl plan."""
        from .kernels import ENABLE_FP_FUSION, constitutive_step  # noqa: PLC0415

        nx, ny, nz = self.shape
        kernel = self._kernel if self._kernel is not None else constitutive_step
        extra = {} if self.num_warps is None else {"num_warps": self.num_warps}
        kernel[self._grid](
            *self._targets, *self._aux, *self._sources, *self._inv_eps,
            *self._coefficients,
            nx, ny, nz, self.n_elem,
            SCALE=self.scale,
            BLOCK=self.block,
            enable_fp_fusion=ENABLE_FP_FUSION if guard is None else bool(guard),
            **extra,
        )

    def __repr__(self) -> str:
        return (f"ConstitutivePlan({self.side}, shape={self.shape}, "
                f"block={self.block}, num_warps={self.num_warps})")


def plan_constitutive(fields: Any, pml: Any, side: str,
                      block: Optional[int] = None,
                      num_warps: Optional[int] = None) -> Optional[ConstitutivePlan]:
    """Build a constitutive plan from the engine's own objects, or None when refused."""
    if side not in CONSTITUTIVE_SIDES:
        raise ValueError(f"side must be one of {tuple(CONSTITUTIVE_SIDES)}, got {side!r}")
    if not constitutive_coverage(fields, pml, side).covered:
        return None
    from .kernels import DEFAULT_BLOCK  # noqa: PLC0415

    spec = CONSTITUTIVE_SIDES[side]
    suffix = "_h" if spec["half_integer"] else ""
    return ConstitutivePlan(
        side, fields.grid.shape, DEFAULT_BLOCK if block is None else block,
        [getattr(fields, n) for n in spec["targets"]],
        [getattr(fields, n) for n in spec["aux"]],
        [getattr(fields, n) for n in spec["sources"]],
        ([fields.inverse_epsilon_for(n) for n in spec["targets"]]
         if side == "E" else None),
        [getattr(pml, f"{stem}_{axis}{suffix}")
         for axis in "xyz" for stem in ("kps", "kms")],
        num_warps=num_warps,
    )


def plan_constitutive_from_arrays(side: str, arrays: Dict[str, Any],
                                  flat: Dict[str, Any], block: Optional[int] = None,
                                  kernel: Any = None,
                                  num_warps: Optional[int] = None) -> ConstitutivePlan:
    """Build a constitutive plan from bare device arrays — the gate's route.

    ``arrays`` is keyed by component name (plus ``inv_eps_Ex``... on the E side) and
    ``flat`` by ``kps_x``/``kms_x``/... on the sub-lattice THE CALLER already
    selected. No predicate runs: the caller is a harness that constructed the
    configuration deliberately, including the deliberately wrong ones.
    """
    from .kernels import DEFAULT_BLOCK  # noqa: PLC0415

    spec = CONSTITUTIVE_SIDES[side]
    shape = tuple(int(n) for n in arrays[spec["targets"][0]].shape)
    return ConstitutivePlan(
        side, shape, DEFAULT_BLOCK if block is None else block,
        [arrays[n] for n in spec["targets"]],
        [arrays[n] for n in spec["aux"]],
        [arrays[n] for n in spec["sources"]],
        ([arrays["inv_eps_" + n] for n in spec["targets"]] if side == "E" else None),
        [flat[f"{stem}_{axis}"] for axis in "xyz" for stem in ("kps", "kms")],
        # The gate's source-mutation leg compiles a deliberately broken copy of the
        # shipped kernel and hands it here. Dropping this argument is not a silent
        # slowdown, it is a silent DISARMING: every mutation leg then launches the
        # shipped kernel and reports the defect as uncaught. Measured — 4/4
        # constitutive source mutations came back 120/120 "identical" before this
        # line existed, which is what a gate that certifies nothing looks like.
        kernel=kernel, num_warps=num_warps,
    )


# ---------------------------------------------------------------------------
# The CROSS-SUB-STEP fused pair (step_B + zero_metal_B + update_H) — F1
# ---------------------------------------------------------------------------
#
# An ADDITION. `PmlCurlPlan` and `ConstitutivePlan` are untouched, still built by
# their own builders and still launchable; `plan_step(fuse=False)` — the default —
# still composes them separately. That composed-but-unfused path is the control
# this fusion is measured against and it has to stay runnable, so nothing below
# removes or reroutes it.

class FusedPairPlan:
    """One launch that performs one curl, its metallic wipe and its constitutive step.

    Holds BOTH halves' bindings: the curl's targets/auxiliaries/sources and
    half-integer ``kms``/``sinv``, and the constitutive side's targets,
    auxiliaries, optional inverse epsilon and the constitutive ``kps``/``kms``.
    The two sub-lattices are chosen here and nowhere else: B/H uses half-integer
    then integer; D/E uses integer then half-integer.

    ``zero_metal`` is the third binding and the one that is new. It is resolved by
    :func:`coverage.zero_metal_axes`, the same function the predicate checked with.
    """

    __slots__ = ("pair", "shape", "n_elem", "dtdx", "backward", "bc", "zero_metal",
                 "block", "num_warps", "_targets", "_aux", "_sources",
                 "_curl_coefficients", "_constitutive_targets", "_constitutive_aux",
                 "_inverse_epsilon", "_constitutive_coefficients", "_grid", "_kernel")

    def __init__(self, pair: str, shape, dtdx: float, bc, zero_metal, block: int,
                 targets, auxiliaries, sources, curl_coefficients,
                 constitutive_targets, constitutive_auxiliaries, inverse_epsilon,
                 constitutive_coefficients, kernel=None,
                 num_warps: Optional[int] = FUSED_DEFAULT_NUM_WARPS) -> None:
        if pair not in FUSED_PAIRS:
            raise ValueError(f"pair must be one of {tuple(FUSED_PAIRS)}, got {pair!r}")
        self.pair = pair
        self.shape = tuple(int(n) for n in shape)
        self.n_elem = self.shape[0] * self.shape[1] * self.shape[2]
        self.dtdx = float(dtdx)
        self.backward = int(SUB_STEPS[FUSED_PAIRS[pair]["curl"]]["backward"])
        self.bc = tuple(int(code) for code in bc)
        self.zero_metal = tuple(1 if flag else 0 for flag in zero_metal)
        self.block = int(block)
        self._targets = tuple(CupyPointer(a) for a in targets)
        self._aux = tuple(CupyPointer(a) for a in auxiliaries)
        self._sources = tuple(CupyPointer(a) for a in sources)
        self._curl_coefficients = tuple(CupyPointer(_flat(a)) for a in curl_coefficients)
        self._constitutive_targets = tuple(CupyPointer(a) for a in constitutive_targets)
        self._constitutive_aux = tuple(CupyPointer(a) for a in constitutive_auxiliaries)
        self._inverse_epsilon = (tuple(CupyPointer(a) for a in inverse_epsilon)
                                 if inverse_epsilon is not None else ())
        self._constitutive_coefficients = tuple(
            CupyPointer(_flat(a)) for a in constitutive_coefficients)
        self._grid = ((self.n_elem + self.block - 1) // self.block,)
        self._kernel = kernel
        # This is MEASURED RATHER THAN INFERRED. The separate plans carry the same
        # optional control so the benchmark compares an explicitly tuned fusion
        # against an explicitly tuned four-kernel control. Triton's default (4 for
        # a 1-D BLOCK=256 program) starves this
        # kernel: fusing doubles the dependency chain a program has to hide, and at
        # 4 warps each thread holds only 2 elements' worth of loads in flight. The
        # measurement, tier-4 shape, Mcell-steps/s on the pair alone
        # (results/triton_fusedpair_2026-08-09/results/fused_pair_micro.json):
        #
        #     num_warps    fused     unfused    fused/unfused
        #         1        5839.7     5585.1        1.046
        #         2        5149.7     5629.6        0.915
        #         4        4676.0     5596.4        0.836   <- Triton's default
        #         8        3236.1     5586.8        0.579
        #
        # It is NOT register pressure — the A1 census reads n_regs 40, no spills,
        # against a 64 ceiling, and 40 is the MAX of the two separate kernels' 40
        # and 36 rather than their sum. Left at the default the fusion is a 16 %
        # LOSS on the pair; at one warp it is a 3.7 % gain over the best unfused
        # configuration. The later two-pair fair gate measured 5.4 % on 2048^2 and
        # 7.0 % on 256x192x128 against independently tuned four-kernel controls.
        # Its shape/boundary policy sweep measured a 1.0739x median over 12 cases,
        # with one bandwidth-bound 5120^2 metallic case effectively tied at
        # 0.99935x. One warp is therefore the omitted-argument default; explicit
        # None remains "take Triton's default" for diagnostics.
        self.num_warps = None if num_warps is None else int(num_warps)

    def run(self, guard: Optional[bool] = None) -> None:
        """Launch both sub-steps. In place; same ``guard`` contract as the others."""
        from .kernels import (  # noqa: PLC0415
            ENABLE_FP_FUSION,
            fused_curl_constitutive_B,
            fused_curl_constitutive_D,
        )

        nx, ny, nz = self.shape
        shipped = (fused_curl_constitutive_B if self.pair == "B"
                   else fused_curl_constitutive_D)
        kernel = self._kernel if self._kernel is not None else shipped
        # Spelled as one dict so the contraction guard keeps its single spelling in
        # this file; an `options={...}` dict is banned by test and this is not one —
        # these are launch keywords, which RAISE on an unrecognised name.
        extra = {} if self.num_warps is None else {"num_warps": self.num_warps}
        common = (
            *self._targets, *self._aux, *self._sources, *self._curl_coefficients,
            *self._constitutive_targets, *self._constitutive_aux,
        )
        if self.pair == "B":
            kernel[self._grid](
                *common, *self._constitutive_coefficients,
                nx, ny, nz, self.n_elem, self.dtdx,
                BACKWARD=self.backward,
                BCX=self.bc[0], BCY=self.bc[1], BCZ=self.bc[2],
                ZM_X=self.zero_metal[0], ZM_Y=self.zero_metal[1], ZM_Z=self.zero_metal[2],
                BLOCK=self.block,
                enable_fp_fusion=ENABLE_FP_FUSION if guard is None else bool(guard),
                **extra,
            )
        else:
            kernel[self._grid](
                *common, *self._inverse_epsilon, *self._constitutive_coefficients,
                nx, ny, nz, self.n_elem, self.dtdx,
                BACKWARD=self.backward,
                BCX=self.bc[0], BCY=self.bc[1], BCZ=self.bc[2],
                ZM_X=self.zero_metal[0], ZM_Y=self.zero_metal[1], ZM_Z=self.zero_metal[2],
                BLOCK=self.block,
                enable_fp_fusion=ENABLE_FP_FUSION if guard is None else bool(guard),
                **extra,
            )

    def __repr__(self) -> str:
        return (f"FusedPairPlan({self.pair}, shape={self.shape}, bc={self.bc}, "
                f"zero_metal={self.zero_metal}, block={self.block}, "
                f"num_warps={self.num_warps})")


def _install_fused_pair(plans, selected, fields, pml, sources, pair_name,
                        curl_name, update_name, label, pair,
                        repair_paths=(_deposit_repair.SPLIT_FIELD_PATH,)):
    """Put a built fused pair into its two slots, with the deposit repair if it needs one.

    THE TWO SLOTS ARE THE MECHANISM, not an accounting detail. ``FastPath.dispatch``
    contracts that True means the whole sub-step ran, so a pair that owns ``step_D`` and
    ``update_E`` is consulted twice with the driver's inject, symmetry fill and wall clear
    in between. Without a deposit the second consult has nothing to do and holds a
    ``NoopPlan``; with one it holds the repair, which is the only place in the step where
    the injected field is final and the fused launch's pre-injection accumulation is still
    known. See ``deposit_repair``.

    ``repair_paths`` is the INSTALLING PRODUCT'S declaration of which recurrence its
    repair inverts, and it defaults to the split-field one — the only repair that
    existed before ``deposit_repair.PLAIN_PATH``, and the one every caller of this
    function ships today. A product whose halves refuse an active absorber runs
    ``update_E``'s plain overwrite instead and must say so: ``deposit_repair.repairable``
    refuses by name any configuration whose recurrence is not among the paths its caller
    declares, so a wrong declaration is a refusal rather than a silent mis-repair.

    THE H->D SEAM IS ROUTED TO THE WITHDRAW HOIST, 2026-09-04, AND THE BRANCH IS HERE
    RATHER THAN IN A FAMILY — the same argument that keeps the deposit bracket out of
    the families, so both mechanisms read as one protocol. That seam's ``pair_name``
    is ``withdraw_hoist.SEAM``, which is not a ``deposit_repair`` field letter: nothing
    is injected between ``update_H`` and ``step_D``, so there is no list to select and
    no bracket to build, and what a spanning product owes instead is the driver's
    electric ``withdraw``, performed immediately before its launch. There is no
    trailing plan — nothing undoes the hoist (the driver's own loop runs after the
    launch and no-ops), so the second slot holds the ``NoopPlan`` unconditionally.
    """
    if pair_name == _withdraw_hoist.SEAM:
        plans[curl_name] = _withdraw_hoist.LeadingWithdrawPlan(
            pair, fields, sources, span=(curl_name, update_name))
        plans[update_name] = NoopPlan(update_name, pair)
        selected[curl_name] = label
        selected[update_name] = label
        return
    seam = _deposit_repair.in_seam_sources(sources, pair_name)
    if not seam:
        plans[curl_name] = pair
        plans[update_name] = NoopPlan(update_name, pair)
    else:
        leading = _deposit_repair.LeadingRepairPlan(pair, fields, pml, seam, pair_name,
                                                    repair_paths)
        plans[curl_name] = leading
        plans[update_name] = _deposit_repair.TrailingRepairPlan(
            update_name, leading, fields, pml)
    selected[curl_name] = label
    selected[update_name] = label


class NoopPlan:
    """The sentinel a fused pair leaves in the slot of the sub-step it absorbed.

    ``TritonStepPlan.replaces`` must still report ``update_H``: the sub-step IS
    replaced, and a plan that reported only ``step_B`` would tell a reader — and a
    regression test — that ``update_H`` ran on the array path, which it did not.
    The substitution wrapper calls ``run()`` on whatever is in the slot, so the
    slot holds something whose ``run()`` does nothing.

    ``absorbed_by`` names the plan that did the work, so the composition can be
    inspected rather than inferred from a bare no-op.

    ONE SLOT IS NOT A PLAN AND NEVER WAS: ``update_P`` holds the LIST of
    per-susceptibility plans, because the driver advances the whole polarization
    list in one pass and it is that pass the slot replaces. So "``run()`` on
    whatever is in the slot" is the rule for six of the seven ``STEP_ORDER``
    names, and an adapter that applied it uniformly would raise
    ``AttributeError`` on ``update_P`` and on nothing else.

    That asymmetry is a CONTRACT rather than an oversight, and every installer in
    the tree already writes it out: ``probe_triton_cylindrical_composition``,
    ``probe_triton_source_seams``, ``probe_triton_engine_route``,
    ``probe_triton_conductivity_composition`` and
    ``probe_fused_kernel_bit_identity`` each branch on ``update_P`` and pass
    ``fields.drive_field`` to every entry — an argument a uniform ``run()`` has
    no way to supply. Two device gates additionally pin the slot's TYPE NAME as
    ``"list"`` (``probe_triton_dispersive_fused_pair``,
    ``probe_triton_fused_ade_state``). Both halves are pinned on the laptop by
    ``test_the_update_P_slot_is_the_one_list_and_every_other_slot_answers_run``.
    """

    __slots__ = ("absorbed_by", "sub_step")

    def __init__(self, sub_step: str, absorbed_by: Any) -> None:
        self.sub_step = sub_step
        self.absorbed_by = absorbed_by

    def run(self, *args: Any, **kwargs: Any) -> None:
        return None

    def __repr__(self) -> str:
        return f"NoopPlan({self.sub_step}, absorbed_by={self.absorbed_by!r})"




def plan_fused_pair(fields: Any, pml: Any, pair: str = "B", sources: Any = None,
                    block: Optional[int] = None,
                    num_warps: Optional[int] = FUSED_DEFAULT_NUM_WARPS,
                    ) -> Optional[FusedPairPlan]:
    """Build the fused pair from the engine's own objects, or None when refused.

    ``sources`` is the driver's source list and is REQUIRED in substance: the
    predicate refuses ``None`` rather than assuming an empty set, because ``Fields``
    does not hold the sources and a predicate that infers coverage from what it
    cannot see is the over-covering failure §10.4 names.
    """
    if pair not in FUSED_PAIRS:
        raise ValueError(f"pair must be one of {tuple(FUSED_PAIRS)}, got {pair!r}")
    if not fused_pair_coverage(fields, pml, pair, sources).covered:
        return None
    from ..stepping import _boundary_kinds  # noqa: PLC0415
    from .kernels import DEFAULT_BLOCK  # noqa: PLC0415

    spec = FUSED_PAIRS[pair]
    curl_spec = SUB_STEPS[spec["curl"]]
    side_spec = CONSTITUTIVE_SIDES[spec["constitutive"]]
    grid = fields.grid
    kinds = _boundary_kinds(grid, pml)
    return FusedPairPlan(
        pair, grid.shape, grid.dt / grid.dx,
        [1 if kind == "metallic" else 0 for kind in kinds],
        zero_metal_axes(grid),
        DEFAULT_BLOCK if block is None else block,
        [getattr(fields, n) for n in curl_spec["targets"]],
        [getattr(fields, "fu_" + n) for n in curl_spec["targets"]],
        [getattr(fields, n) for n in curl_spec["sources"]],
        [getattr(pml, f"{stem}_{axis}{curl_spec['suffix']}")
         for axis in "xyz" for stem in ("kms", "sinv")],
        [getattr(fields, n) for n in side_spec["targets"]],
        [getattr(fields, n) for n in side_spec["aux"]],
        ([fields.inverse_epsilon_for(n) for n in side_spec["targets"]]
         if spec["constitutive"] == "E" else None),
        [getattr(pml, f"{stem}_{axis}{'_h' if side_spec['half_integer'] else ''}")
         for axis in "xyz" for stem in ("kps", "kms")],
        num_warps=num_warps,
    )


def plan_fused_pair_from_arrays(pair: str, arrays: Dict[str, Any],
                                flat: Dict[str, Any], constitutive_flat: Dict[str, Any],
                                codes, zero_metal, dtdx: float,
                                block: Optional[int] = None,
                                kernel: Any = None,
                                num_warps: Optional[int] = FUSED_DEFAULT_NUM_WARPS,
                                ) -> FusedPairPlan:
    """Build the fused pair from bare device arrays — the gate's and benchmark's route.

    ``flat`` carries the curl's ``kms_x``/``sinv_x``/... and
    ``constitutive_flat`` the constitutive side's ``kps_x``/``kms_x``/... on the
    sub-lattices the caller selected. Both are deliberately caller-owned so the
    gate can hand over swapped tables and watch them be caught.
    """
    from .kernels import DEFAULT_BLOCK  # noqa: PLC0415

    spec = FUSED_PAIRS[pair]
    curl_spec = SUB_STEPS[spec["curl"]]
    side_spec = CONSTITUTIVE_SIDES[spec["constitutive"]]
    shape = tuple(int(n) for n in arrays[curl_spec["targets"][0]].shape)
    return FusedPairPlan(
        pair, shape, dtdx, codes, zero_metal,
        DEFAULT_BLOCK if block is None else block,
        [arrays[n] for n in curl_spec["targets"]],
        [arrays["fu_" + n] for n in curl_spec["targets"]],
        [arrays[n] for n in curl_spec["sources"]],
        [flat[f"{stem}_{axis}"] for axis in "xyz" for stem in ("kms", "sinv")],
        [arrays[n] for n in side_spec["targets"]],
        [arrays[n] for n in side_spec["aux"]],
        ([arrays["inv_eps_" + n] for n in side_spec["targets"]]
         if spec["constitutive"] == "E" else None),
        [constitutive_flat[f"{stem}_{axis}"]
         for axis in "xyz" for stem in ("kps", "kms")],
        kernel=kernel, num_warps=num_warps,
    )


# ---------------------------------------------------------------------------
# The ADE sub-step (update_P), one plan per susceptibility
# ---------------------------------------------------------------------------

class AdeUpdatePPlan:
    """One susceptibility's ADE recurrence for every component it drives.

    POINTERS ARE RESOLVED PER LAUNCH, DELIBERATELY, and this class exists mostly to
    say why. ``PolarizationState.update`` rotates three buffers per component per
    step (dispersion.py:687-691): the result lands in ``_scratch``, ``_scratch``
    takes over the array that held ``P_prev``, and ``P``/``P_prev`` shift down — and
    the retired history becomes the NEXT component's scratch inside the same call.
    A plan that cached ``P``/``P_prev``/``_scratch`` device views the way
    :class:`PmlCurlPlan` caches field views would be stale after the first component
    of the first step, and stale in a way that still computes: it would advance the
    same buffer twice and freeze another.

    The rotation below is that rotation, transcribed, not reimplemented.
    """

    __slots__ = ("state", "components", "shape", "n_elem", "block", "_sigma_is_volume",
                 "_grid", "_kernel")

    def __init__(self, state: Any, components: Sequence[str], shape, block: int,
                 kernel: Any = None) -> None:
        self.state = state
        self.components = tuple(components)
        self.shape = tuple(int(n) for n in shape)
        self.n_elem = self.shape[0] * self.shape[1] * self.shape[2]
        self.block = int(block)
        # Decided once, by the same function the predicate checked with, so the
        # constexpr and the clause cannot disagree (gate mutation m13).
        self._sigma_is_volume = {c: bool(sigma_is_volume(state, c)) for c in self.components}
        self._grid = ((self.n_elem + self.block - 1) // self.block,)
        self._kernel = kernel

    def run(self, drive, guard: Optional[bool] = None) -> None:
        """Advance every driven component one step.

        ``drive`` is ``Fields.drive_field`` — NOT a stored-E reader. Under PML the
        two differ, and taking E instead is "the single most likely silent wrong
        answer in dispersion" (fields.py:1149-1156): they agree exactly outside the
        absorber, so every no-PML case passes and only the PML case is wrong. The
        caller passes the bound method; this plan never reaches for a field itself.
        """
        from .kernels import ENABLE_FP_FUSION, ade_update_p  # noqa: PLC0415

        state = self.state
        kernel = self._kernel if self._kernel is not None else ade_update_p
        c_now, c_prev, c_drive = state._coefficients
        for component in self.components:
            w = drive(component)
            p = state.P[component]
            p_prev = state.P_prev[component]
            scratch = state._scratch
            volume = self._sigma_is_volume[component]
            sigma = state.sigma[component]
            kernel[self._grid](
                CupyPointer(scratch), CupyPointer(p), CupyPointer(p_prev),
                CupyPointer(sigma) if volume else float(sigma),
                CupyPointer(w),
                float(c_now), float(c_prev), float(c_drive), self.n_elem,
                SIGMA_IS_VOLUME=1 if volume else 0,
                BLOCK=self.block,
                enable_fp_fusion=ENABLE_FP_FUSION if guard is None else bool(guard),
            )
            # dispersion.py:687-691, verbatim. Every buffer is owned by exactly one
            # slot at a time, so no two names ever alias.
            state.P[component] = scratch
            state.P_prev[component] = p
            state._scratch = p_prev

    def __repr__(self) -> str:
        return (f"AdeUpdatePPlan({self.components}, shape={self.shape}, "
                f"block={self.block})")


def plan_ade_update_p(fields: Any, state: Any,
                      block: Optional[int] = None) -> Optional[AdeUpdatePPlan]:
    """Plan one susceptibility's ADE update, or None when any driven component is refused.

    ALL OR NOTHING PER STATE, on purpose. ``PolarizationState.update`` rotates a
    SHARED scratch buffer from component to component inside one call; covering two
    of three components and leaving the third to the array path would interleave two
    rotations over one buffer set. Sub-steps compose; halves of one sub-step do not.

    The Triton import stays BELOW the predicate, as it does in every builder here: a
    NumPy host must be able to plan (to ``None``) without the optional dependency of
    an optional package being importable at all.
    """
    components = tuple(state.driven()) if callable(getattr(state, "driven", None)) else ()
    if not components:
        return None
    for component in components:
        if not ade_update_p_coverage(fields, state, component).covered:
            return None
    from .kernels import DEFAULT_BLOCK  # noqa: PLC0415

    return AdeUpdatePPlan(state, components, fields.grid.shape,
                          DEFAULT_BLOCK if block is None else block)


# ---------------------------------------------------------------------------
# Composition — the covered SUBSET for one frozen configuration
# ---------------------------------------------------------------------------

#: The sub-step names a step plan may replace, in the driver's own step order.
#: ``fill_B``/``fill_D`` stand for the combined near-symmetry and folded-far
#: passes.  They sit after source injection; a driver adapter must therefore run
#: each plan from ``fill_symmetry_bc_*`` and suppress ``fill_folded_far_ghosts_*``.
STEP_ORDER: Tuple[str, ...] = (
    "step_B", "fill_B", "update_H",
    "step_D", "fill_D", "update_E", "update_P",
)


class TritonStepPlan:
    """Which sub-steps of one frozen configuration Triton covers, and the plans for them.

    This is the shape ``fastpath.FastPathPlan`` would hold, built here because
    dispatch is not wired: ``fastpath.plan_fast_path`` still returns ``None`` on
    every branch and this package does not touch it. The gate and the benchmark are
    the callers; the driver is not.

    ``replaces`` is built ADDITIVELY. A supported dispersive PML run gets all five
    named sub-steps: ``update_E`` is handled by its pole-aware specialized product,
    while a non-dispersive run selects the ordinary constitutive product. Partial
    coverage is legal and expected; what is NOT legal is replacing a sub-step whose
    own predicate refused, or selecting between overlapping predicates by order.

    ``selected`` names the ARM that won each filled slot. It is the only way to
    tell "ordinary" from "real beta run" from "BFAST run" on ``update_H``: all
    three build a :class:`ConstitutivePlan`, so the class name distinguishes
    nothing and a composition record that reported only the class would be
    reporting an ambiguity as a decision. It is an OPTIONAL fourth parameter —
    existing callers construct this object with three positional arguments.
    """

    __slots__ = ("plans", "polarization_plans", "reasons", "selected")

    def __init__(self, plans: Dict[str, Any], polarization_plans: Sequence[Any],
                 reasons: Dict[str, Tuple[str, ...]],
                 selected: Optional[Dict[str, str]] = None) -> None:
        self.plans = dict(plans)
        self.polarization_plans = tuple(polarization_plans)
        self.reasons = dict(reasons)
        self.selected = dict(selected or {})

    @property
    def replaces(self) -> Tuple[str, ...]:
        return tuple(name for name in STEP_ORDER if name in self.plans)

    def describe(self) -> str:
        covered = ", ".join(self.replaces) if self.replaces else "nothing"
        return (f"TritonStepPlan(replaces=[{covered}], "
                f"polarizations={len(self.polarization_plans)})")

    def __repr__(self) -> str:
        return self.describe()


class _Arm:
    """One candidate product for one sub-step slot.

    ``gate`` decides whether the arm is CONSULTED at all; ``predicate`` and
    ``builder`` are zero-argument closures so the module-level forwarder is
    resolved when they run (a routing test replaces the forwarder, not this).
    ``prefix`` is what the arm's refusals are labelled with in a combined
    refusal, and ``noun`` names the arm in a builder-refusal message.
    """

    __slots__ = ("label", "gate", "predicate", "builder", "prefix", "noun")

    def __init__(self, label: str, gate: bool, predicate: Any, builder: Any,
                 prefix: str, noun: str) -> None:
        self.label = label
        self.gate = bool(gate)
        self.predicate = predicate
        self.builder = builder
        self.prefix = prefix
        self.noun = noun


def _arm_verdict(arm: "_Arm") -> Coverage:
    """One consulted arm's verdict. Never raises and always returns a verdict.

    A predicate that RAISES is a REFUSAL, never an admission: this composer's
    contract is that it never raises into a caller that would otherwise have
    stepped correctly, and an exception is not evidence that a product applies.

    A predicate that RETURNS A NON-VERDICT is the same kind of event and gets
    the same treatment. Reading ``.covered`` off whatever came back is where the
    old spelling gave up its own contract — a predicate returning ``True`` (no
    ``.covered``) crashed the composition, and one returning ``None`` was
    silently indistinguishable from "not consulted", so its arm vanished from
    ``reasons`` with nothing to say it had been asked. Both are named refusals
    now; the gate decides consultation, and only the gate.

    The normalisation itself is :func:`_guarded_verdict`'s, shared with the three
    blocks that do not go through the arm table so the two cannot drift into
    treating a raising predicate differently depending on where it was called.
    """
    return _guarded_verdict(arm.predicate, f"{arm.label} predicate")


def _select_slot(slot: str, arms: Sequence["_Arm"], ambiguity: Any,
                 plans: Dict[str, Any], reasons: Dict[str, Tuple[str, ...]],
                 selected: Dict[str, str]) -> Tuple[str, ...]:
    """Fill one sub-step slot from its arm table, fail-closed. Returns the admitters.

    EVERY consulted arm's predicate runs before any selection: an overlap is a
    predicate defect and must fail closed rather than turn table order into a
    numerical-method choice. Two or more admitters leave the slot UNSELECTED —
    the array path, which is always correct — with a reason naming all of them.
    """
    consulted = [(arm, _arm_verdict(arm)) for arm in arms if arm.gate]
    admitted = [arm for arm, verdict in consulted if verdict.covered]
    labels = tuple(arm.label for arm in admitted)

    if len(admitted) > 1:
        reasons[slot] = (ambiguity(labels),)
        return labels
    if len(admitted) == 1:
        arm = admitted[0]
        try:
            built = arm.builder()
        except Exception as exc:  # noqa: BLE001 - a raising builder refuses the slot
            reasons[slot] = (f"{arm.noun} coverage admitted {slot} but its "
                             f"builder raised {exc!r}",)
            return labels
        if built is not None:
            plans[slot] = built
            selected[slot] = arm.label
        else:
            reasons[slot] = (f"{arm.noun} coverage admitted {slot} but its "
                             "builder refused",)
        return labels

    collected: List[str] = []
    for arm, verdict in consulted:
        collected.extend(f"{arm.prefix}{reason}" for reason in verdict.reasons)
    # An unselected slot must never be reported with a FALSY reason tuple: a
    # caller that tests `if plan.reasons.get(slot)` would read "nothing covers
    # this" as "nothing to report". Every refusal below carries a name.
    reasons[slot] = tuple(collected) or (
        f"no consulted product admitted {slot}, and none gave a reason",)
    return labels


def _reads_true(obj: Any, name: str) -> bool:
    """``bool(obj.name)``, with an unreadable attribute answering True.

    Used where the answer decides whether an UNMEASURED product is refused: an
    attribute this composer cannot read is not evidence that the thing it guards
    is absent, and the fail-closed answer is the one that refuses the product.
    """
    try:
        return bool(getattr(obj, name, False))
    except Exception:  # noqa: BLE001 - an unreadable attribute is not a False one
        return True


def _guarded_verdict(call: Any, noun: str) -> Coverage:
    """A coverage verdict from a call that may raise or return a non-verdict.

    The arm table gets this treatment from :func:`_arm_verdict`; the blocks that
    do NOT go through the arm table — the opt-in fusion and the polarization loop
    — get it from here. (The folded ghost fills were a third such block and are
    not any more: they go through the arm table, so they reach this function
    through :func:`_arm_verdict` like everything else.) Before this existed those
    blocks were the whole of the "never raises" contract's exception list, and
    on a host with no Triton the composer raised on every dispersive run and on
    every ``fuse=True`` run: a builder import error escaping into a caller that
    would otherwise have stepped correctly on the array path.
    """
    try:
        verdict = call()
    except Exception as exc:  # noqa: BLE001 - a raising predicate refuses
        return Coverage(False, (f"{noun} raised {exc!r}",))
    covered = getattr(verdict, "covered", None)
    reasons = getattr(verdict, "reasons", None)
    if covered is None or reasons is None:
        return Coverage(False, (f"{noun} returned {verdict!r}, which is not a "
                                "coverage verdict",))
    try:
        collected = tuple(str(reason) for reason in reasons)
        covered = bool(covered)
    except Exception as exc:  # noqa: BLE001 - an unreadable verdict refuses
        return Coverage(False, (f"{noun} returned a verdict this composer could "
                                f"not read ({exc!r})",))
    if covered:
        return Coverage(True, collected)
    return Coverage(False, collected or (f"{noun} refused without a reason",))


def _guarded_plan(call: Any, noun: str) -> Tuple[Any, Optional[str]]:
    """``(plan, None)`` from a builder, or ``(None, reason)`` when it raised.

    A tuple rather than a sentinel value because a plan is an arbitrary object —
    including, in the routing tests, a string — so no in-band value can mean
    "this raised" without also being a legal plan.
    """
    try:
        return call(), None
    except Exception as exc:  # noqa: BLE001 - a raising builder refuses
        return None, f"{noun} raised {exc!r}"


#: Which ARM each fused product's own kernel implements, per slot it absorbs.
#: A fused pair is the ordinary PML curl with one named constitutive product
#: inlined; absorbing a slot a DIFFERENT arm won — or a slot no arm won — would
#: substitute a numerical product nothing admitted, which is the over-covering
#: dispatch this composer exists to refuse.
FUSED_PAIR_ARMS: Dict[str, Tuple[str, str]] = {
    "B": ("PML", "ordinary"),
    "D": ("PML", "ordinary"),
    "dispersive_D": ("PML", "dispersive"),
}

#: THE FOLDED GRID'S TWO PAIRS, routed 2026-08-27 — the same two-slot protocol,
#: keyed by the seam name :mod:`..deposit_repair` knows.
#:
#: WHAT USED TO BE HERE WAS A BLANKET REFUSAL. ``fuse=True`` on a folded grid
#: returned "cross-sub-step fusion for this product is not implemented" for both
#: pairs, which was true of the ORDINARY pair — it carries no mirror fill — and
#: false of the two products that do. Each of these families welds the folded curl
#: to the folded constitutive update AND carries the near fill, the wall clear and
#: the far ghost image inline (their ``REPLACES``), so the seam the blanket refusal
#: named is the seam they close.
#:
#: THE ARM PAIR IS READ, NOT GUESSED, exactly as the ordinary rows above are.
#: ``folded_fused_magnetic_pair_coverage`` is literally
#: ``folded_composition_curl_coverage(fields, pml, "step_B")`` AND
#: ``folded_constitutive_coverage(fields, pml, "H")``
#: (folded_fused_magnetic_pair.py:807-812); those are the bodies behind the
#: ``folded PML`` curl arm and the ``folded`` constitutive arm in the tables below,
#: so a configuration this pair admits is one where those two arms won. The D row
#: is the same conjunction on ``step_D``/``E`` (folded_fused_pair.py:996-1001).
#:
#: NEITHER ROW LICENSES THE DEPOSIT REPAIR, AND THAT IS STILL TRUE AFTER THE FLIP.
#: A row here says which arm the kernel implements; whether the seam's deposit can be
#: repaired is the family's own ``CARRIES_DEPOSIT_REPAIR``, which both folded
#: families flipped to ``True`` on 2026-08-30.
#:
#: THE PREMISE THAT KEPT THEM ``False`` WAS RETIRED BY MEASUREMENT, NOT BY OPINION.
#: It read: the driver's post-injection ``fill_symmetry_bc_*`` writes the MIRROR
#: IMAGE of a deposited cell (stepping.py:1450-1451) and ``deposit_repair.apply``
#: only ever touches the deposit index itself. The second half is no longer the
#: case — ``deposit_repair._fill_image_rules`` (deposit_repair.py:217) and
#: ``repair_cells`` (:258) close the saved and restored set over exactly the cells
#: the near fill, the far fill and the wall clear image a deposit into, so
#: ``apply`` puts back every cell those passes touch. A point-only repair was
#: MEASURED wrong on a fold (images left holding a pre-injection field), which is
#: why the closure had to land before the flag could.
#:
#: SO THESE TWO NOW ROUTE THE CARRIED SEAM AS WELL AS THE CLEAN ONE, and every seam
#: the repair cannot invert is still refused BY NAME by ``repairable`` rather than
#: by the flag — off-diagonal, nonlinear, the cylindrical axis, a fold too short for
#: the near fill's source row, and an absorber that never ran its split-field
#: recurrence.
#:
#: THE ``fused pair`` PREFIX ON BOTH LABELS IS LOAD-BEARING, not a naming taste.
#: ``fastpath.arm_is_fused`` (fastpath.py:209-213) recognises a fused product by
#: ``FUSED_ARM_PREFIXES = ("fused pair",)``, and clause (8) of
#: ``fastpath.plan_fast_path`` (fastpath.py:2084-2091) refuses the WHOLE dispatch
#: when any slot carries such a label — the standing guard that a future flip of
#: ``fuse`` to default-on cannot silently reach a driver seam no gate drove. A
#: label reading ``folded fused pair B`` would NOT match that prefix, so routing
#: these two under one would have punched a hole straight through the guard at the
#: moment it started admitting rows. The suffix carries the family; the prefix
#: carries the refusal.
FOLDED_FUSED_PAIRS: Dict[str, Dict[str, str]] = {
    "B": {"curl": "step_B", "update": "update_H",
          "label": "fused pair B (folded)", "noun": "folded B"},
    "D": {"curl": "step_D", "update": "update_E",
          "label": "fused pair D (folded)", "noun": "folded D"},
}

#: The arms both folded pairs absorb, in ``_pair_may_absorb`` order. One entry
#: rather than one per pair because the folded curl arm registers the SAME label on
#: ``step_B`` and ``step_D`` and the folded constitutive arm the same on
#: ``update_H`` and ``update_E``.
FOLDED_FUSED_PAIR_ARMS: Tuple[str, str] = ("folded PML", "folded")


def _folded_fused_pair_entries():
    """``{pair: (coverage, builder)}`` for the two folded pairs, imported LAZILY.

    The import lives in a function body for the reason every family module's does:
    these modules import Triton at their own module scope (behind their own
    ``try``), and the package must stay importable — and free of them in
    ``sys.modules`` — on a host with none. ``FAMILY_MODULES``/``SUPPORT_MODULES``
    declare the seam and the AST test compares the declaration against this file.
    """
    from .folded_fused_magnetic_pair import (  # noqa: PLC0415
        folded_fused_magnetic_pair_coverage, plan_folded_fused_magnetic_pair)
    from .folded_fused_pair import (  # noqa: PLC0415
        folded_fused_pair_coverage, plan_folded_fused_pair)

    return {
        "B": (folded_fused_magnetic_pair_coverage,
              plan_folded_fused_magnetic_pair),
        "D": (folded_fused_pair_coverage, plan_folded_fused_pair),
    }


def _install_folded_fused_pairs(plans, reasons, selected, fields, pml, sources,
                                block, num_warps, offered=None):
    """Route the folded grid's two pairs through their OWN builders, or refuse by name.

    GUARDED EXACTLY AS AN ARM IS, and for the same reason the ordinary block is:
    ``plan_step`` never raises, the family predicates read a source list this
    composer did not build, and the builders import Triton, which the merge-bar host
    does not have. A raising predicate is a refusal and a raising builder leaves the
    seam on its separate plans with a named reason.

    WHAT STAYS ON THE ARRAY PATH WHEN A PAIR IS INSTALLED. The driver runs
    ``fill_symmetry_bc_*``, ``zero_metal_*`` and ``fill_folded_far_ghosts_*``
    UNCONDITIONALLY — driver.py:3285-3287 and :3298-3300 are plain calls with no
    ``fast.dispatch`` consult — so a folded pair that carries them inline has them
    run a SECOND time on the host after its launch. That is byte-safe only because
    all three are idempotent images of cells the pair did not change: the near fill
    copies cell 2 into cell 0 (stepping.py:1450-1451), the far fill copies the
    reflect row, and the wall clear zeroes a stored plane.

    SOMETHING DOES WRITE THE SEAM IN BETWEEN, SINCE 2026-08-30, AND THE REPAIR IS
    WHAT KEEPS IT BYTE-SAFE. Both families now declare ``CARRIES_DEPOSIT_REPAIR``,
    so an in-seam deposit is CARRIED rather than refused: ``_install_fused_pair``
    brackets the pair with ``LeadingRepairPlan``/``TrailingRepairPlan``, the save
    happens before the launch and the restore after the driver's inject and all
    three host passes. The three passes above still run a second time on the host
    and are still idempotent — what changed is that the cells they image a deposit
    into are inside the repair's restored set (``deposit_repair.repair_cells``),
    instead of being the reason the seam was refused. A seam the repair cannot
    invert is still refused, by ``repairable``, by name.
    """
    entries, import_refusal = _guarded_plan(
        _folded_fused_pair_entries, "the folded fused pair modules")
    if import_refusal is not None or entries is None:
        for pair_name in FOLDED_FUSED_PAIRS:
            reasons[f"fused_pair_{pair_name}"] = (
                import_refusal or "the folded fused pair modules were refused",)
        return

    for pair_name, spec in FOLDED_FUSED_PAIRS.items():
        key = f"fused_pair_{pair_name}"
        curl_name, update_name = spec["curl"], spec["update"]
        coverage, builder = entries[pair_name]
        verdict = _guarded_verdict(
            lambda coverage=coverage: coverage(fields, pml, sources),
            f"the {spec['noun']} pair predicate")
        if not verdict.covered:
            # A refused fusion is not a refused step: both sub-steps keep the
            # separate folded plans built above, and the reason is reported here.
            reasons[key] = verdict.reasons
            continue
        absorb = _pair_may_absorb(selected, curl_name, update_name,
                                  FOLDED_FUSED_PAIR_ARMS)
        if absorb is not None:
            reasons[key] = (absorb,)
            continue
        not_offered = _label_not_offered(spec["label"], offered)
        if not_offered is not None:
            reasons[key] = (not_offered,)
            continue
        pair, builder_refusal = _guarded_plan(
            lambda builder=builder: builder(fields, pml, sources, block, num_warps),
            f"the {spec['noun']} pair builder")
        if builder_refusal is not None:
            reasons[key] = (builder_refusal,)
        elif pair is not None:
            _install_fused_pair(plans, selected, fields, pml, sources, pair_name,
                                curl_name, update_name, spec["label"], pair)
        else:
            reasons[key] = (f"the {spec['noun']} pair was refused",)


def _label_not_offered(label: str,
                       offered: Optional[FrozenSet[str]]) -> Optional[str]:
    """``None`` when this caller will run ``label``, else why the seam stays unfused.

    WHY A CALLER MAY OFFER A SUBSET AT ALL, and it is the one thing this composer
    knows about its callers: ``fuse`` says WHETHER to attempt fusion, and a caller
    that will only RUN some of the products this table can build had, until
    2026-09-02, no way to say so. ``fuse=True`` then installed every product whose
    predicate admitted, and a caller that could not run one of them was left holding
    a composition it had to reject WHOLE — the fused label occupies both slots, so
    rejecting it rejects the separate arms that would otherwise have run.

    MEASURED, which is why this exists rather than being argued. On a conductive 2-D
    PML grid the 2026-09-02 product wave installed ``conductive_fused_electric_pair``
    over ``step_D``/``update_E``; its caller could run ``fused pair B`` and not that
    label, and the whole plan went to the array path — a grid that had been running
    a fused magnetic pair and four certified arms the day before. With the offer
    declared, the electric seam simply keeps its separate plans and the magnetic pair
    runs, which is the composition that grid had.

    ``None`` MEANS NO RESTRICTION, and that is the right default for every caller
    that composes in order to INSPECT — the gates, the probes, the census. They ask
    what the table can build, and an offer they did not make must not narrow it.

    THIS IS NOT A PREFERENCE AND IT IS NOT A TIE-BREAK. It is applied LAST, after
    every predicate, after the ambiguity rule and after :func:`_pair_may_absorb`, so
    a product that would have been refused for any of those reasons is still refused
    for that reason and with that wording. All this clause can do is leave a seam on
    its separate sub-step plans, which is the array path, which is always correct.
    """
    if offered is None or label in offered:
        return None
    return (f"{label} is not among the fused labels this caller offered to run "
            f"({', '.join(sorted(offered)) or 'none'}); the seam keeps its separate "
            f"sub-step plans")


def _offered_labels(fuse_labels: Any) -> Optional[FrozenSet[str]]:
    """Normalise the caller's offer, refusing to read a non-collection as empty.

    A caller that passes something unreadable is treated as having made NO offer
    rather than an empty one: an empty offer silently disables every product, which
    is a behaviour change wearing the shape of a typo.
    """
    if fuse_labels is None:
        return None
    if isinstance(fuse_labels, str):
        return frozenset({fuse_labels})
    try:
        return frozenset(str(label) for label in fuse_labels)
    except TypeError:
        return None


def _pair_may_absorb(selected: Dict[str, str], curl_name: str,
                     update_name: str, arms: Tuple[str, str]) -> Optional[str]:
    """``None`` when the fused product may take both slots, else why it may not.

    THE FUSION BLOCK IS THE ONE PLACE THAT FILLS A ``STEP_ORDER`` SLOT WITHOUT
    GOING THROUGH THE ARM TABLE, and it used to write both slots unconditionally.
    That defeated the fail-closed contract at its own point of decision: a slot
    left UNSELECTED because two predicates admitted it (clause (b)) or because
    its builder refused (clause (c)) was then claimed anyway — by the ordinary
    fused product, which is exactly the product the ambiguity refused to pick —
    and ``plan.reasons`` and ``plan.plans`` came back contradicting each other
    for the same slot.

    So the pair may absorb a slot only when the arm table already gave that slot
    to the arm the fused kernel implements. That is strictly narrower than the
    pair predicate and never narrower than what the fused gates measured: those
    ran ordinary PML grids, where the ``PML`` and ``ordinary``/``dispersive``
    arms are the ones that win.
    """
    for name, arm in ((curl_name, arms[0]), (update_name, arms[1])):
        won = selected.get(name)
        if won is None:
            return (f"{name} is not selected by any arm — the composition left it "
                    "on the array path, and a fused pair may not claim a slot the "
                    "arm table refused")
        if won != arm:
            return (f"{name} was selected by the {won!r} arm; this fused product "
                    f"implements the {arm!r} one and may not substitute it")
    return None


# ---------------------------------------------------------------------------
# THE CERTIFIED-PRODUCT SEAM LOOP — the installer wave of 2026-09-02
# ---------------------------------------------------------------------------
#
# WHAT THIS BLOCK IS FOR. Twenty-five modules in this package held a released
# device gate, a shipped predicate and a builder, and were reached by NO composer:
# ``launch.py`` imported none of them, so the reachability join could not even name
# them and their cells landed in the fusion board's "served by a product this file
# cannot name" bucket — 122 of the Triton board's 341 served seam-instances (of 387),
# the largest single bucket on any backend's board. Nothing about those products was
# missing except the composition rule; this is it.
#
# IT IS A PORT, NOT AN INVENTION, and porting rather than inventing is the whole
# reason it is a table and a loop instead of twenty-five hand-written branches.
# ``cuda_kernels/fused_pairs.install_fused_pairs`` is the same seam loop over the
# same three shared helpers (:func:`_guarded_verdict`, :func:`_guarded_plan`,
# :func:`_pair_may_absorb`), and it was itself ported from the two hand-written
# blocks above. Reading one is reading all three.
#
# WHY THE INCUMBENTS ARE NOT IN THE TABLE. ``fused_pair_B``/``fused_pair_D``,
# ``dispersive_fused_pair`` and the two folded pairs keep their hand-written
# branches, byte-for-byte. Their gates measured THOSE branches — including the
# ``specialized_family_owns_the_grid`` refusal and the fold/cylindrical refusals in
# front of them — and folding them into this loop would change what a released arm
# was released from. This loop runs AFTER them and reads the live ``selected``, so a
# slot an incumbent already absorbed carries a fused label rather than an arm label
# and :func:`_pair_may_absorb` refuses it by name. That is the whole of the
# cross-block precedence rule: there is no table order, only "the arm table's
# selection, as it stands when this loop is reached".
#
# WIRING IS NOT RELEASING. Every label below is refused BY NAME at
# ``fastpath.plan_fast_path``'s clause (8) — none is in
# ``fastpath.RELEASED_FUSED_ARMS`` — so a run that installs one of these products
# does not dispatch it: the whole plan is refused unless ``MEEP_GPU_FUSE_ARMS``
# opts into that exact label. What this block changes is that the product can be
# SELECTED at all, which is the precondition a release round measures against.
#
# THE TWO SCRATCH-OUTPUT OFF-DIAGONAL D->E WELDS ARE ROUTED SINCE 2026-09-15, and
# what kept them out until then was a mechanism rather than a preference. Their
# plan ROTATES the engine's own ``D``/``fu_D`` attributes after each launch, which
# the array path never does, and with no ``warm`` on the shared plan base
# ``fastpath.warm_plan`` fell to ``_warm_with_empty_grid``, which sets the settable
# ``_grid`` slot to ``(0,)`` and CALLS ``run`` — enqueueing nothing and rotating
# anyway, leaving the engine's D volumes on the plan's zero twins at PLAN TIME. The
# base now defines ``warm``: roles resolved, one launch at grid ``(0,)``, the grid
# restored, no rotation and no launch counted, so ``warm_plan`` never reaches
# ``run``. ``test_triton_certified_fused_products`` reads that mechanism off the
# tree, and the stencil gate's two ``warm`` legs measure it (a recording stub on
# the host, the shipped kernel on the device).
#
# NOTHING ELSE ABOUT THIS LOOP CHANGES FOR THEM. Each builder takes the loop's
# positional ``(fields, pml, sources, block, num_warps)``. Each predicate refuses
# every in-seam electric source (``CARRIES_DEPOSIT_REPAIR`` is False), so
# :func:`_install_fused_pair` puts a ``NoopPlan`` in ``update_E`` and brackets no
# repair. The off-diagonal veto runs BEFORE this loop and both constitutive arms
# are in :data:`ROW_PRODUCT_ARMS`, so it leaves the slot selected for them. And the
# two predicates PARTITION the off-diagonal space — the plain weld's curl half
# refuses a fold, the folded weld's composition variants refuse an unfolded grid —
# so the ambiguity rule never sees both on one seam.

#: Curl slot -> (constitutive slot, the ``deposit_repair`` seam name). The seam name
#: is ``deposit_repair``'s own: ``'B'`` selects the magnetic deposit list and
#: anything else the electric one (``deposit_repair._in_seam_indexed``).
#:
#: NO ``update_E -> update_P`` ROW, and its absence is a statement. The three E->P
#: chain products in this package are NOT in the table below (they are named in
#: ``test_triton_planner_composition``'s deferral tuple, which is where a deferred
#: product's name belongs — naming one here is what that seam reads): that seam's
#: second slot holds a LIST of per-susceptibility plans rather than a plan, its
#: first slot is a constitutive sub-step rather than a curl, and its products'
#: builders take the winning ARM rather than a source list — three differences from
#: the pair protocol, any one of which would make this loop's contract false about
#: them. A row here
#: with no product to fill it would report "no candidate" on every configuration,
#: which is the same defect ``cuda_kernels.fused_pairs.FUSED_PAIR_SEAMS`` records
#: for its own D/E row: a reason true of every run reads as a claim about the run.
#: AN ``update_H -> step_D`` ROW SINCE 2026-09-04, AND IT LANDED AHEAD OF ITS
#: PRODUCT, which is the one row here that did. What it buys before any H->D kernel
#: exists is the ARBITRATION (:func:`_neighbouring_seam_claimant`) and the KEY ORDER,
#: and an empty row costs nothing on this loop: it opens each seam with
#: ``if not candidates: continue`` and records no reason, so a seam nobody bids on is
#: SILENT rather than reporting "no candidate" on every configuration. That is the
#: defect the paragraph above cites, and it is the loop's behaviour that retired it.
#:
#: THE ORDER IS LOAD-BEARING AND THE ROW IS APPENDED LAST. This loop walks
#: ``.items()`` in dict order against the LIVE ``selected``. Placed first, the seam
#: would be offered ``update_H`` before the B->H products were, would find an ARM
#: label rather than a fused one there, and :func:`_pair_may_absorb` would let it take
#: both slots — displacing a released, gate-passing product on every row it reached.
#: ``test_the_h_to_d_seam_row_is_last`` pins the order with that reason.
#:
#: ITS SEAM NAME IS ``withdraw_hoist.SEAM``, NOT A ``deposit_repair`` FIELD LETTER.
#: Nothing is injected between ``update_H`` and ``step_D``, so there is no deposit
#: list to select, and ``_in_seam_indexed`` reads any string other than ``'B'`` as the
#: ELECTRIC list — which would bracket this seam with a repair for an injection that
#: happens in the NEXT seam. What the driver does put between the two consults is the
#: electric ``withdraw`` loop, and :func:`_install_fused_pair` routes this name to
#: :mod:`..withdraw_hoist` rather than to ``deposit_repair.in_seam_sources``.
CERTIFIED_FUSED_PAIR_SEAMS: Dict[str, Tuple[str, str]] = {
    "step_B": ("update_H", "B"),
    "step_D": ("update_E", "D"),
    "update_H": ("step_D", _withdraw_hoist.SEAM),
}

#: Which ARM each certified product's kernel implements, per slot it absorbs — the
#: bound in front of the loop, exactly as ``FUSED_PAIR_ARMS`` is in front of the
#: ordinary block and ``FOLDED_FUSED_PAIR_ARMS`` in front of the folded one. A
#: product with no row here is REFUSED BY NAME rather than absorbing a slot on a
#: mapping nobody established.
#:
#: EVERY ROW IS READ OFF THE PREDICATE THE PRODUCT CONJOINS, never assigned: each
#: module's ``*_coverage`` opens by calling the two arms' own predicates on its two
#: slots, and the pair below is those two arms' labels. The same derivation is
#: recorded per product, with the citation, in
#: ``parity/meep_gpu/build_triton_fusion_matrix.PRODUCTS[...]["cell"]``, and
#: ``test_triton_certified_fused_products`` asserts the two agree row for row — so
#: this table is measured against an independently written one rather than trusted.
CERTIFIED_FUSED_PAIR_ARMS: Dict[str, Tuple[str, str]] = {
    # B->H.
    "complex_fused_magnetic_pair": ("complex PML", "complex"),
    "cylindrical_fused_magnetic_pair": ("cylindrical complex PML",
                                        "cylindrical complex"),
    "cylindrical_real_fused_magnetic_pair": ("cylindrical PML", "cylindrical"),
    "folded_complex_fused_magnetic_pair": ("folded complex PML", "folded complex"),
    "folded_beta_complex_fused_magnetic_pair": ("folded complex beta PML",
                                                "folded complex"),
    "folded_beta_fused_magnetic_pair": ("folded real beta PML", "folded beta run"),
    "complex_beta_fused_magnetic_pair": ("complex beta PML", "complex beta run"),
    "beta_fused_magnetic_pair": ("real beta PML", "real beta run"),
    "bfast_fused_magnetic_pair": ("BFAST PML", "BFAST run"),
    "nonlinear_fused_magnetic_pair": ("nonlinear run PML", "nonlinear run"),
    # D->E.
    "complex_fused_electric_pair": ("complex PML", "complex"),
    "cylindrical_fused_electric_pair": ("cylindrical complex PML",
                                        "cylindrical complex"),
    "cylindrical_real_fused_electric_pair": ("cylindrical PML", "cylindrical"),
    "folded_complex_fused_pair": ("folded complex PML", "folded complex"),
    "folded_beta_complex_fused_pair": ("folded complex beta PML", "folded complex"),
    "folded_beta_fused_electric_pair": ("folded real beta PML", "folded beta run"),
    "complex_beta_fused_electric_pair": ("complex beta PML", "complex beta run"),
    "beta_fused_electric_pair": ("real beta PML", "real beta run"),
    "bfast_fused_electric_pair": ("BFAST PML", "BFAST run"),
    "folded_dispersive_fused_pair": ("folded PML", "folded dispersive"),
    "conductive_fused_electric_pair": ("conductive PML", "ordinary"),
    "complex_conductive_fused_pair": ("complex conductive no-PML curl",
                                      "complex no-PML stored E"),
    "no_pml_fused_electric_pair": ("conductive no-PML", "no-PML stored E"),
    # The two scratch-output off-diagonal welds (2026-09-15). The plain weld conjoins
    # ``coverage.pml_curl_coverage`` on ``step_D`` with the off-diagonal constitutive
    # predicate; the folded weld conjoins the COMPOSITION variants of the folded curl
    # and folded off-diagonal predicates, which is what keeps the two disjoint.
    "offdiag_fused_electric_pair": ("PML", "off-diagonal"),
    "folded_offdiag_fused_electric_pair": ("folded PML", "folded off-diagonal"),
    # H->D. The row is (update_H arm, step_D arm): ``_pair_may_absorb`` pairs the
    # first entry with the seam row's curl_name, which on this seam is ``update_H``.
    "cylindrical_real_fused_hd_pair": ("cylindrical", "cylindrical PML"),
    "cylindrical_fused_hd_pair": ("cylindrical complex", "cylindrical complex PML"),
    "complex_fused_hd_pair": ("complex", "complex PML"),
}

#: ADDITIONAL (curl arm, constitutive arm) pairs a product may absorb BESIDE its
#: primary row — a separate table rather than a widened value shape, so every
#: existing pin on a primary row stays a pin.
#:
#: TWO ROWS, and each is a SECOND CELL the SAME gate artifact released. The fusion
#: board scores each of them as its own product name against the same module and the
#: same ``gate.json`` (``no_pml_fused_electric_pair_lossless`` beside
#: ``no_pml_fused_electric_pair``, ``folded_complex_fused_magnetic_pair_offdiag``
#: beside ``folded_complex_fused_magnetic_pair``), because one predicate admits two
#: arm pairs; one launch serves both, so there is one label and two absorb rows.
#: Widening the primary row to a set instead would have made "which arm pair did
#: this product absorb" unanswerable from the table.
CERTIFIED_FUSED_PAIR_EXTRA_ARMS: Dict[str, Tuple[Tuple[str, str], ...]] = {
    # The lossless no-absorber cell: the same stored-E constitutive under the
    # plain no-PML curl arm rather than the conductive one. Its predicate reads
    # the two no-absorber curl arms' own partition and asks whichever serves the
    # run, so both arms are its arm.
    "no_pml_fused_electric_pair": (("no-PML", "no-PML stored E"),),
    # The off-diagonal folded-complex cell: the same launch under the two
    # off-diagonal-row arms of the same two families.
    "folded_complex_fused_magnetic_pair": (
        ("folded complex off-diagonal PML", "folded complex off-diagonal"),),
}

#: Product -> how to reach it and what it writes into ``selected``.
#:
#: ``module``/``coverage``/``builder`` are NAMES rather than objects because the
#: import must stay inside a function body: every one of these modules imports
#: Triton at its own module scope (behind its own ``try``), and this package must
#: stay importable — and free of them in ``sys.modules`` — on a host with none.
#: :data:`SUPPORT_MODULES` declares that seam and the AST test compares the
#: declaration against this file.
#:
#: THE LABEL CARRIES THE ``fused pair`` PREFIX ON PURPOSE, exactly as the two folded
#: labels do. ``fastpath.arm_is_fused`` recognises a fused product by
#: ``FUSED_ARM_PREFIXES = ("fused pair",)`` and clause (8) refuses the WHOLE dispatch
#: when any slot carries such a label. A label spelled any other way would punch a
#: hole straight through that guard at the moment this loop started admitting rows.
#: The suffix carries the family; the prefix carries the refusal.
CERTIFIED_FUSED_PRODUCTS: Dict[str, Dict[str, str]] = {
    "complex_fused_magnetic_pair": {
        "curl_slot": "step_B", "module": "complex_fused_magnetic_pair",
        "coverage": "complex_fused_magnetic_pair_coverage",
        "builder": "plan_complex_fused_magnetic_pair",
        "label": "fused pair B (complex)"},
    "cylindrical_fused_magnetic_pair": {
        "curl_slot": "step_B", "module": "cylindrical_fused_magnetic_pair",
        "coverage": "cylindrical_fused_magnetic_pair_coverage",
        "builder": "plan_cylindrical_fused_magnetic_pair",
        "label": "fused pair B (cylindrical complex)"},
    "cylindrical_real_fused_magnetic_pair": {
        "curl_slot": "step_B", "module": "cylindrical_real_fused_magnetic_pair",
        "coverage": "cylindrical_real_fused_magnetic_pair_coverage",
        "builder": "plan_cylindrical_real_fused_magnetic_pair",
        "label": "fused pair B (cylindrical)"},
    "folded_complex_fused_magnetic_pair": {
        "curl_slot": "step_B", "module": "folded_complex_fused_magnetic_pair",
        "coverage": "folded_complex_fused_magnetic_pair_coverage",
        "builder": "plan_folded_complex_fused_magnetic_pair",
        "label": "fused pair B (folded complex)"},
    "folded_beta_complex_fused_magnetic_pair": {
        "curl_slot": "step_B", "module": "folded_beta_complex_fused_magnetic_pair",
        "coverage": "folded_beta_complex_fused_magnetic_pair_coverage",
        "builder": "plan_folded_beta_complex_fused_magnetic_pair",
        "label": "fused pair B (folded complex beta)"},
    "folded_beta_fused_magnetic_pair": {
        "curl_slot": "step_B", "module": "folded_beta_fused_magnetic_pair",
        "coverage": "folded_beta_fused_magnetic_pair_coverage",
        "builder": "plan_folded_beta_fused_magnetic_pair",
        "label": "fused pair B (folded real beta)"},
    "complex_beta_fused_magnetic_pair": {
        "curl_slot": "step_B", "module": "complex_beta_fused_magnetic_pair",
        "coverage": "complex_beta_fused_magnetic_pair_coverage",
        "builder": "plan_complex_beta_fused_magnetic_pair",
        "label": "fused pair B (complex beta)"},
    "beta_fused_magnetic_pair": {
        "curl_slot": "step_B", "module": "beta_fused_magnetic_pair",
        "coverage": "beta_fused_magnetic_pair_coverage",
        "builder": "plan_beta_fused_magnetic_pair",
        "label": "fused pair B (real beta)"},
    "bfast_fused_magnetic_pair": {
        "curl_slot": "step_B", "module": "bfast_fused_magnetic_pair",
        "coverage": "bfast_fused_magnetic_pair_coverage",
        "builder": "plan_bfast_fused_magnetic_pair",
        "label": "fused pair B (BFAST)"},
    "nonlinear_fused_magnetic_pair": {
        "curl_slot": "step_B", "module": "nonlinear_fused_magnetic_pair",
        "coverage": "nonlinear_fused_magnetic_pair_coverage",
        "builder": "plan_nonlinear_fused_magnetic_pair",
        "label": "fused pair B (nonlinear)"},
    "complex_fused_electric_pair": {
        "curl_slot": "step_D", "module": "complex_fused_electric_pair",
        "coverage": "complex_fused_electric_pair_coverage",
        "builder": "plan_complex_fused_electric_pair",
        "label": "fused pair D (complex)"},
    "cylindrical_fused_electric_pair": {
        "curl_slot": "step_D", "module": "cylindrical_fused_electric_pair",
        "coverage": "cylindrical_fused_electric_pair_coverage",
        "builder": "plan_cylindrical_fused_electric_pair",
        "label": "fused pair D (cylindrical complex)"},
    "cylindrical_real_fused_electric_pair": {
        "curl_slot": "step_D", "module": "cylindrical_real_fused_electric_pair",
        "coverage": "cylindrical_real_fused_electric_pair_coverage",
        "builder": "plan_cylindrical_real_fused_electric_pair",
        "label": "fused pair D (cylindrical)"},
    "folded_complex_fused_pair": {
        "curl_slot": "step_D", "module": "folded_complex_fused_pair",
        "coverage": "folded_complex_fused_pair_coverage",
        "builder": "plan_folded_complex_fused_pair",
        "label": "fused pair D (folded complex)"},
    "folded_beta_complex_fused_pair": {
        "curl_slot": "step_D", "module": "folded_beta_complex_fused_pair",
        "coverage": "folded_beta_complex_fused_pair_coverage",
        "builder": "plan_folded_beta_complex_fused_pair",
        "label": "fused pair D (folded complex beta)"},
    "folded_beta_fused_electric_pair": {
        "curl_slot": "step_D", "module": "folded_beta_fused_electric_pair",
        "coverage": "folded_beta_fused_electric_pair_coverage",
        "builder": "plan_folded_beta_fused_electric_pair",
        "label": "fused pair D (folded real beta)"},
    "complex_beta_fused_electric_pair": {
        "curl_slot": "step_D", "module": "complex_beta_fused_electric_pair",
        "coverage": "complex_beta_fused_electric_pair_coverage",
        "builder": "plan_complex_beta_fused_electric_pair",
        "label": "fused pair D (complex beta)"},
    "beta_fused_electric_pair": {
        "curl_slot": "step_D", "module": "beta_fused_electric_pair",
        "coverage": "beta_fused_electric_pair_coverage",
        "builder": "plan_beta_fused_electric_pair",
        "label": "fused pair D (real beta)"},
    "bfast_fused_electric_pair": {
        "curl_slot": "step_D", "module": "bfast_fused_electric_pair",
        "coverage": "bfast_fused_electric_pair_coverage",
        "builder": "plan_bfast_fused_electric_pair",
        "label": "fused pair D (BFAST)"},
    "folded_dispersive_fused_pair": {
        "curl_slot": "step_D", "module": "folded_dispersive_fused_pair",
        "coverage": "folded_dispersive_fused_pair_coverage",
        "builder": "plan_folded_dispersive_fused_pair",
        "label": "fused pair D (folded dispersive)"},
    "conductive_fused_electric_pair": {
        "curl_slot": "step_D", "module": "conductive_fused_electric_pair",
        "coverage": "conductive_fused_electric_pair_coverage",
        "builder": "plan_conductive_fused_electric_pair",
        "label": "fused pair D (conductive)"},
    "complex_conductive_fused_pair": {
        "curl_slot": "step_D", "module": "complex_conductive_fused_pair",
        "coverage": "complex_conductive_fused_pair_coverage",
        "builder": "plan_complex_conductive_fused_pair",
        "label": "fused pair D (complex conductive no-PML)"},
    "no_pml_fused_electric_pair": {
        "curl_slot": "step_D", "module": "no_pml_fused_electric_pair",
        "coverage": "no_pml_fused_electric_pair_coverage",
        "builder": "plan_no_pml_fused_electric_pair",
        "label": "fused pair D (no-PML stored E)"},
    # THE TWO SCRATCH-OUTPUT OFF-DIAGONAL D->E WELDS, 2026-09-15. The note above
    # :data:`CERTIFIED_FUSED_PAIR_SEAMS` says why they were absent until then and why
    # this loop needs nothing new to route them.
    "offdiag_fused_electric_pair": {
        "curl_slot": "step_D", "module": "offdiag_fused_electric_pair",
        "coverage": "offdiag_fused_electric_pair_coverage",
        "builder": "plan_offdiag_fused_electric_pair",
        "label": "fused pair D (off-diagonal)"},
    "folded_offdiag_fused_electric_pair": {
        "curl_slot": "step_D", "module": "folded_offdiag_fused_electric_pair",
        "coverage": "folded_offdiag_fused_electric_pair_coverage",
        "builder": "plan_folded_offdiag_fused_electric_pair",
        "label": "fused pair D (folded off-diagonal)"},
    # THE H->D SEAM'S TWO CYLINDRICAL PRODUCTS, 2026-09-07. Both declare
    # INSTALLABLE = False, so ``_declared_uninstallable`` refuses each by name on
    # every row before its predicate is asked; routing them makes that refusal a
    # recorded reason rather than an absence, and the seam row (``update_H`` ->
    # ``withdraw_hoist.SEAM``) is already in the table for them.
    "cylindrical_real_fused_hd_pair": {
        "curl_slot": "update_H", "module": "cylindrical_real_fused_hd_pair",
        "coverage": "cylindrical_real_fused_hd_pair_coverage",
        "builder": "plan_cylindrical_real_fused_hd_pair",
        "label": "fused pair H->D (cylindrical)"},
    "cylindrical_fused_hd_pair": {
        "curl_slot": "update_H", "module": "cylindrical_fused_hd_pair",
        "coverage": "cylindrical_fused_hd_pair_coverage",
        "builder": "plan_cylindrical_fused_hd_pair",
        "label": "fused pair H->D (cylindrical complex)"},
    # THE CARTESIAN COMPLEX H->D PRODUCT, 17 seam-instances. INSTALLABLE = False,
    # so `_declared_uninstallable` refuses it by name on every row before its
    # predicate is asked; routing it makes that refusal a recorded reason rather
    # than an absence. The arbitration is MEASURED rather than assumed and it is a
    # LOSS on this cell: on a complex row the composer installs `fused pair B
    # (complex)` on step_B+update_H and `fused pair D (complex)` on
    # step_D+update_E, so a two-slot H->D span takes one slot from each and turns
    # two pairs into one (gate_triton_complex_fused_hd_pair.py's arbitration leg,
    # and per corpus row in its lift leg).
    "complex_fused_hd_pair": {
        "curl_slot": "update_H", "module": "complex_fused_hd_pair",
        "coverage": "complex_fused_hd_pair_coverage",
        "builder": "plan_complex_fused_hd_pair",
        "label": "fused pair H->D (complex)"},
}


def _certified_fused_product_modules():
    """``{module name: module}`` for the certified products, imported LAZILY.

    THE IMPORTS ARE WRITTEN OUT, one ``from .<module> import`` per product, rather
    than resolved through ``importlib`` — and that is not a style choice. Two
    separate readers measure what this composer can reach by walking THIS FILE'S
    PARSE TREE: ``build_triton_fusion_matrix.launch_imported_modules``, which
    decides the board's ``wired`` column, and
    ``test_triton_planner_composition._launch_imports``, which holds that reach
    against :data:`SUPPORT_MODULES`. BOTH read ``ImportFrom.module`` and skip a
    node where it is ``None``, so neither a dynamic ``import_module`` nor a
    ``from . import <module>`` would be seen — and the board would go on publishing
    ``wired: false`` for 122 seam-instances the planner now reaches, which is the
    same class of stale column ``launch_imported_modules``'s own docstring records
    having been built to fix.

    The import lives in a function BODY for the reason every family module's does:
    these modules import Triton at their own module scope, and the package must stay
    importable — and free of them in ``sys.modules`` — on a host with none.

    THE MODULE OBJECT IS RECOVERED FROM THE PREDICATE, because the import form that
    the two readers can see binds the FUNCTION and not the module. A plain
    function's ``__module__`` is the module that defines it, so the lookup is exact
    and — unlike a second hand-written list of module objects — cannot fall out of
    step with the imports above it.
    """
    import sys  # noqa: PLC0415 - see the module-object note above

    from .beta_fused_electric_pair import beta_fused_electric_pair_coverage  # noqa: PLC0415
    from .beta_fused_magnetic_pair import beta_fused_magnetic_pair_coverage  # noqa: PLC0415
    from .bfast_fused_electric_pair import bfast_fused_electric_pair_coverage  # noqa: PLC0415
    from .bfast_fused_magnetic_pair import bfast_fused_magnetic_pair_coverage  # noqa: PLC0415
    from .complex_beta_fused_electric_pair import complex_beta_fused_electric_pair_coverage  # noqa: PLC0415,E501
    from .complex_beta_fused_magnetic_pair import complex_beta_fused_magnetic_pair_coverage  # noqa: PLC0415,E501
    from .complex_conductive_fused_pair import complex_conductive_fused_pair_coverage  # noqa: PLC0415,E501
    from .complex_fused_electric_pair import complex_fused_electric_pair_coverage  # noqa: PLC0415
    from .complex_fused_hd_pair import complex_fused_hd_pair_coverage  # noqa: PLC0415
    from .complex_fused_magnetic_pair import complex_fused_magnetic_pair_coverage  # noqa: PLC0415
    from .conductive_fused_electric_pair import conductive_fused_electric_pair_coverage  # noqa: PLC0415,E501
    from .cylindrical_fused_electric_pair import cylindrical_fused_electric_pair_coverage  # noqa: PLC0415,E501
    from .cylindrical_fused_hd_pair import cylindrical_fused_hd_pair_coverage  # noqa: PLC0415,E501
    from .cylindrical_fused_magnetic_pair import cylindrical_fused_magnetic_pair_coverage  # noqa: PLC0415,E501
    from .cylindrical_real_fused_electric_pair import cylindrical_real_fused_electric_pair_coverage  # noqa: PLC0415,E501
    from .cylindrical_real_fused_hd_pair import cylindrical_real_fused_hd_pair_coverage  # noqa: PLC0415,E501
    from .cylindrical_real_fused_magnetic_pair import cylindrical_real_fused_magnetic_pair_coverage  # noqa: PLC0415,E501
    from .folded_beta_complex_fused_magnetic_pair import folded_beta_complex_fused_magnetic_pair_coverage  # noqa: PLC0415,E501
    from .folded_beta_complex_fused_pair import folded_beta_complex_fused_pair_coverage  # noqa: PLC0415,E501
    from .folded_beta_fused_electric_pair import folded_beta_fused_electric_pair_coverage  # noqa: PLC0415,E501
    from .folded_beta_fused_magnetic_pair import folded_beta_fused_magnetic_pair_coverage  # noqa: PLC0415,E501
    from .folded_complex_fused_magnetic_pair import folded_complex_fused_magnetic_pair_coverage  # noqa: PLC0415,E501
    from .folded_complex_fused_pair import folded_complex_fused_pair_coverage  # noqa: PLC0415
    from .folded_dispersive_fused_pair import folded_dispersive_fused_pair_coverage  # noqa: PLC0415,E501
    from .folded_offdiag_fused_electric_pair import folded_offdiag_fused_electric_pair_coverage  # noqa: PLC0415,E501
    from .no_pml_fused_electric_pair import no_pml_fused_electric_pair_coverage  # noqa: PLC0415
    from .nonlinear_fused_magnetic_pair import nonlinear_fused_magnetic_pair_coverage  # noqa: PLC0415,E501
    from .offdiag_fused_electric_pair import offdiag_fused_electric_pair_coverage  # noqa: PLC0415,E501

    predicates = (
        beta_fused_electric_pair_coverage, beta_fused_magnetic_pair_coverage,
        bfast_fused_electric_pair_coverage, bfast_fused_magnetic_pair_coverage,
        complex_beta_fused_electric_pair_coverage,
        complex_beta_fused_magnetic_pair_coverage,
        complex_conductive_fused_pair_coverage,
        complex_fused_electric_pair_coverage, complex_fused_hd_pair_coverage,
        complex_fused_magnetic_pair_coverage,
        conductive_fused_electric_pair_coverage,
        cylindrical_fused_electric_pair_coverage,
        cylindrical_fused_hd_pair_coverage,
        cylindrical_fused_magnetic_pair_coverage,
        cylindrical_real_fused_electric_pair_coverage,
        cylindrical_real_fused_hd_pair_coverage,
        cylindrical_real_fused_magnetic_pair_coverage,
        folded_beta_complex_fused_magnetic_pair_coverage,
        folded_beta_complex_fused_pair_coverage,
        folded_beta_fused_electric_pair_coverage,
        folded_beta_fused_magnetic_pair_coverage,
        folded_complex_fused_magnetic_pair_coverage,
        folded_complex_fused_pair_coverage, folded_dispersive_fused_pair_coverage,
        folded_offdiag_fused_electric_pair_coverage,
        no_pml_fused_electric_pair_coverage,
        nonlinear_fused_magnetic_pair_coverage,
        offdiag_fused_electric_pair_coverage)
    modules = {}
    for predicate in predicates:
        module = sys.modules[predicate.__module__]
        modules[predicate.__module__.rsplit(".", 1)[-1]] = module
    return modules


def _certified_fused_product_entry(modules, product: Dict[str, str]):
    """``(coverage, builder, repair_paths)`` for one product of that import.

    ``repair_paths`` is the MODULE'S OWN declaration of which recurrence its repair
    inverts, defaulting to the split-field one, which is what every module written
    before ``deposit_repair.PLAIN_PATH`` existed implicitly declares. Read here and
    handed to :func:`_install_fused_pair` rather than defaulted there, so a product
    whose repair inverts the plain branch is bracketed with the repair it actually
    carries. THE FLAG (``CARRIES_DEPOSIT_REPAIR``) IS NOT READ HERE, deliberately
    and as on the other two composers: it is enforced inside the product's own
    coverage predicate, which passes it to ``deposit_repair.seam_source_reasons``,
    so a product that has not declared it refuses every in-seam source and a
    configuration with a deposit never reaches the bracket. Checking it twice would
    let the two answers disagree.
    """
    module = modules[product["module"]]
    return (getattr(module, product["coverage"]),
            getattr(module, product["builder"]),
            tuple(getattr(module, "REPAIR_PATHS",
                          (_deposit_repair.SPLIT_FIELD_PATH,))))


def _declared_uninstallable(modules, name: str, product: Dict[str, str]):
    """``None`` when the product may be installed, else the reason it may not.

    ``INSTALLABLE`` IS A DECLARATION ABOUT THE PRODUCT, NOT ABOUT A RUN, which is why
    it is read here and reported on every configuration rather than folded into a
    coverage predicate. A predicate answers "can this weld serve this run" — a
    question about arithmetic, and the one its gate measures — and that answer must
    not change because of where the product sits in the driver's slot path. A module
    setting ``INSTALLABLE = False`` says the other kind of thing: the arithmetic is
    certified and the COMPOSITION is refused, on every row, for a measured reason of
    its own.

    IT IS BELT AND BRACES, NOT THE RULE. :func:`_neighbouring_seam_claimant` is what
    decides a seam collision from the run; this flag is what keeps a careless edit —
    a new seam row, a reordered table, a widened absorb declaration — from installing
    a product whose own module says it must not be.

    A module that is not in the imported set is INSTALLABLE: this is reached only
    after the guarded import already succeeded, so an absent entry is impossible
    rather than permissive, and fail-closed would turn an import problem into a
    silent permanent refusal that reads like a decision.
    """
    module = modules.get(product["module"])
    if module is None or getattr(module, "INSTALLABLE", True):
        return None
    declared = getattr(module, "INSTALLABLE_REASON", "")
    return (f"{name} declares INSTALLABLE = False: "
            f"{declared or 'its own module refuses the composition'}")


def _slot_degrees(seams: Dict[str, Tuple[str, Any]]) -> Dict[str, int]:
    """How many rows of ``seams`` name each slot -- its degree on the seam path.

    COMPUTED FROM THE TABLE, NEVER SPELLED. :func:`_neighbouring_seam_claimant` asks
    its question only of a span whose two END slots both have degree >= 2 -- an
    INTERIOR edge -- and the end/interior split has to follow the table so that a
    row added later moves it without anyone having to remember to.
    """
    degrees: Dict[str, int] = {}
    for curl_name, (update_name, _seam_name) in seams.items():
        for slot in (curl_name, update_name):
            degrees[slot] = degrees.get(slot, 0) + 1
    return degrees


def _is_end_edge_span(span: Sequence[str],
                      seams: Dict[str, Tuple[str, Any]]) -> bool:
    """True when either END slot of ``span`` sits in fewer than two rows of ``seams``."""
    span = tuple(span)
    degrees = _slot_degrees(seams)
    return min(degrees.get(span[0], 0), degrees.get(span[-1], 0)) < 2


def _neighbouring_seam_claimant(modules, fields, pml, sources, name: str,
                                curl_name: str, update_name: str,
                                span: Sequence[str] = ()):
    """``(claimant, refusal)`` for a product that would lose a slot to this one.

    THE RULE, SINCE 2026-09-05: the question is asked ONLY of a span that is an
    INTERIOR edge of the seam path -- both of its END slots sit in two or more rows
    of ``CERTIFIED_FUSED_PAIR_SEAMS`` -- and such a product is refused where another
    product claims the seam its LAST slot opens. An end-edge span is never asked and
    never yields. The degrees are COMPUTED from the table (:func:`_slot_degrees`),
    never spelled, so a seam row added later moves the split without anyone
    remembering to.

    WHY THAT IS THE RIGHT SHAPE IS A MATCHING ARGUMENT, not a corpus accident. Over a
    path of four slots the maximum matching has size 2 and it is always the two END
    edges; an interior edge belongs to no maximum matching unless both of its
    neighbours are absent. So an end-edge span must never yield, and an interior span
    must yield to a neighbour that serves. The other direction -- the seam a span's
    FIRST slot closes -- is answered exactly and cheaply by :func:`_pair_may_absorb`
    reading the live ``selected``: a fact about what installed rather than a
    prediction about a predicate, and a fact cannot annihilate with another fact.
    That is also what resolves the TIE below in the released incumbent's favour --
    the B->H product installs first and holds ``update_H`` -- so there is no
    tie-break helper and must not be.

    Degrees on this table today: ``step_B`` 1, ``update_H`` 2, ``step_D`` 2,
    ``update_E`` 1. So ``B_to_H`` (1, 2) and ``D_to_E`` (2, 1) are end edges and
    never yield, and ``H_to_D`` (2, 2) is interior and yields.

    WHAT THE GUARD CLOSED was measured 2026-09-04 through the METAL composer, whose
    seam table has the same three rows as this one: with the ``update_H`` row in the
    table, the later half ALONE refused the RELEASED B->H pair on every row an
    admitting H->D product reached -- even a stand-in with no absorb declaration,
    which can never install (3 launches / 1 seam against the shipped 2 / 2 on the
    155-row control cell, 4 / 0 against 3 / 1 on the 24 off-diagonal/folded rows).
    That tree was safe only because its H->D product declares ``INSTALLABLE = False``;
    the flag protects one product, this guard protects the seam. No H->D product is in
    this table yet, so the guard changes no plan this composer produces today.

    The earlier-neighbour direction was asked here briefly on 2026-09-04 and retired
    the same day; the block below records why, and the guard above cannot reproduce
    what it did because a terminal seam is an end edge by the table's own degrees.

    WHY IT EXISTS, AND IT IS ARITHMETIC RATHER THAN A CORPUS ACCIDENT. The driver's
    slot path is ``step_B — update_H — step_D — update_E``: four slots, three seams,
    and an installed pair is an edge of a matching on that path. Launches over those
    four slots are therefore

        launches = 4 - (number of installed pairs)

    so a product is worth installing only where it RAISES the matching size. A
    product spanning ``update_H``/``step_D`` takes one slot from each neighbour,
    which can never do that: where both neighbours serve it takes the matching from
    2 to 1 (one MORE launch per step and one FEWER seam served); where exactly one
    serves it ties at 3 launches and one seam; and where neither serves it would be a
    gain.

    GAIN, TIE OR LOSS FOR AN H->D PRODUCT, MEASURED 2026-09-04 at COMPOSER level
    (predicate AND absorb row) on Metal, over the 179 buildable rows of the 194-row
    ``h_to_d_seam_2026-09-04`` basis (Triton's predicate-level join is 154 / 25 /
    0 / 0 on the same basis, one row different from Metal's 155 / 24):

    ====  =========================================================  ====================
    rows  installing an H->D product there is                         under this rule
    ====  =========================================================  ====================
     133  a LOSS: both neighbours install, 2 pairs -> 1               B->H installs first;
                                                                      H->D refused by name
                                                                      (``_pair_may_absorb``)
      22  a TIE: only B->H serves, 3 launches / 1 seam either way     the same; the released
                                                                      incumbent keeps the slot
      22  a LOSS: only D->E serves                                    the later half names
                                                                      the D->E claimant
       2  a GAIN: NEITHER serves (the nonlinear cell, 4 / 0 -> 3 / 1) it installs, where its
                                                                      arm reaches the cell
    ====  =========================================================  ====================

    0 rows on which displacing the released B->H pair is a gain. The predicate-level
    join that preceded it had no "neither" row; the composer additionally requires an
    absorb row, and the 2 "neither" rows are why this rule must not be written as a
    blanket veto. The refusal this returns is therefore a RUN-SPECIFIC measured reason
    in the existing "who gets the slot" wording: where no neighbour's product admits,
    this returns ``None`` and the product installs.

    THE SCAN IS WITHIN THIS TABLE, and that boundary is the one the ambiguity rule
    already draws for the same reason. The hand-written incumbent block's products
    are not here, and they do not need to be: they run BEFORE this loop and write
    their label into ``selected``, so a slot one of them took is already refused to
    every candidate here by :func:`_pair_may_absorb`, by name. Extending this scan
    across the blocks would change what the incumbent gates measured, which is not a
    thing a composition round may do.

    IT IS SLOT ARBITRATION AND NOT A VERDICT ABOUT ARITHMETIC. Every predicate that
    reaches this admitted and every gate that released one certified it; what is
    decided here is who gets the slots.

    THE ONLY SHAPE THAT ESCAPES IT is a span that REACHES the neighbouring seam
    instead of colliding with it — a four-slot ``step_B -> update_H -> step_D ->
    update_E`` weld in ONE launch, which serves three seams where today's two pairs
    serve two in two launches. ``span`` is how that is said here: a product whose own
    span already covers the neighbouring seam's slots makes no trade.
    """
    span = tuple(span) or (curl_name, update_name)
    # R1: AN END-EDGE SPAN IS NEVER ASKED. The matching argument is in the docstring;
    # the degrees are read off the table so the answer moves with it.
    if _is_end_edge_span(span, CERTIFIED_FUSED_PAIR_SEAMS):
        return None
    neighbours = []
    # THE EARLIER-NEIGHBOUR HALF WAS REMOVED 2026-09-04, MEASURED, NOT ARGUED. It
    # asked the same question in the other direction -- "does a product claim the seam
    # my FIRST slot closes?" -- and it did two things, both wrong:
    #
    # * it was REDUNDANT where it was meant to help. An H->D span's last slot is
    #   ``step_D``, which opens the D->E seam, so the later half above already names a
    #   D->E claimant and returns before the earlier half is reached.
    # * it REVERSED A RELEASED RULING where it did fire. The only product class whose
    #   last slot opens no seam is E->P (``update_P`` is terminal), so the earlier half
    #   fired there and there alone -- refusing the polarization pair because a D->E
    #   product admits, while the later half was already refusing that D->E product
    #   because the polarization pair claims ``update_E``. Both neighbours annihilated
    #   and the seam went unserved. Measured on a dispersive PML run (poles=2): the
    #   later half names ``cuda_fused_polarization_pair`` for the D->E product, and the
    #   earlier half named ``cuda_dispersive_fused_electric_pair`` for the polarization
    #   pair. The 2026-09-02 trade ruling gives ``update_E`` to the E->P incumbent, and
    #   this function may not re-litigate it from the other side.
    #
    # What the earlier half was FOR -- an H->D product installing on a row where only
    # the B->H neighbour serves, which the launch algebra makes a TIE rather than a
    # gain -- was settled on 2026-09-04/05 with the product in hand (on Metal): it is
    # a TIE, the released incumbent keeps the slot, and the mechanism is not a half of
    # this function at all but `_pair_may_absorb` reading the live `selected` after
    # the B->H product installs first. What this function gained instead is the
    # end-edge guard above, which is what keeps the later half from refusing that
    # incumbent.
    if update_name in CERTIFIED_FUSED_PAIR_SEAMS:
        later_update = CERTIFIED_FUSED_PAIR_SEAMS[update_name][0]
        if later_update not in span:
            neighbours.append((update_name, later_update, update_name))
    for neighbour_curl, neighbour_update, shared in neighbours:
        for other, product in CERTIFIED_FUSED_PRODUCTS.items():
            if product["curl_slot"] != neighbour_curl or other == name:
                continue
            if _declared_uninstallable(modules, other, product) is not None:
                continue
            entry, read_refusal = _guarded_plan(
                lambda product=product: _certified_fused_product_entry(modules,
                                                                       product),
                f"the {other} module")
            if read_refusal is not None or entry is None:
                continue
            verdict = _guarded_verdict(
                lambda coverage=entry[0]: coverage(fields, pml, sources),
                f"the {other} pair predicate")
            if verdict.covered:
                return other, (
                    f"{other} admits this run's {neighbour_curl}/{neighbour_update} "
                    f"seam and would lose {shared} to this product; over the "
                    f"step_B..update_E slot path launches are 4 - (installed pairs), "
                    f"so a product taking a slot from a neighbouring seam can only "
                    f"tie or lose, and the slot is left with the product that already "
                    f"serves it. See _neighbouring_seam_claimant — this is slot "
                    f"arbitration, not a verdict about {name}'s arithmetic, which its "
                    f"own gate certifies")
    return None


def _install_certified_fused_products(plans, reasons, selected, fields, pml,
                                      sources, block, num_warps, offered=None):
    """At most one certified product per seam, refusing on every doubt.

    Guarded exactly as an arm is, and for the same three reasons the hand-written
    blocks above are: :func:`plan_step` never raises, the product predicates read a
    source list this composer did not build, and the builders import Triton, which
    the merge-bar host does not have. A raising predicate is a refusal; a raising or
    ``None``-returning builder leaves the seam unfused with a NAMED reason under
    ``fused_pair_<product>``.

    TWO ADMITTING PREDICATES ON ONE SEAM IS AN AMBIGUITY, NOT A PICK — clause (b)'s
    rule one level up, and the reason this loop takes every verdict before it
    installs anything. The wording deliberately avoids :func:`_select_slot`'s phrase
    "admitted the same configuration" so a probe scanning ``reasons`` for an
    ambiguous SLOT does not report a seam as one.

    THE AMBIGUITY RULE IS WITHIN THIS TABLE, and that boundary is deliberate rather
    than an oversight. An incumbent pair's predicate CAN admit a configuration one
    of these products also admits — ``fused_pair_coverage`` admits a conductive grid,
    and so does ``conductive_fused_electric_pair`` — but the two implement DIFFERENT
    arms, the arm table gives the slot to exactly one of them, and this loop reads
    the live ``selected``, so the incumbent's install is already in front of it.
    Extending the ambiguity rule across the blocks would change what the incumbent
    gates measured, which is not a thing a wiring round may do.
    """
    modules, import_refusal = _guarded_plan(_certified_fused_product_modules,
                                            "the certified fused product modules")
    if import_refusal is not None or modules is None:
        for name in CERTIFIED_FUSED_PRODUCTS:
            reasons[f"fused_pair_{name}"] = (
                import_refusal or "the certified fused product modules were refused",)
        return
    for curl_name, (update_name, pair_name) in CERTIFIED_FUSED_PAIR_SEAMS.items():
        candidates = []
        for name, product in CERTIFIED_FUSED_PRODUCTS.items():
            if product["curl_slot"] != curl_name:
                continue
            key = f"fused_pair_{name}"
            # THE ABSORB DECLARATION IS ASKED FIRST, BEFORE THE PREDICATE, and the
            # order is the point: a missing declaration is a fact about the TABLE,
            # true of every configuration, so it is reported on every configuration.
            # Asking the predicate first would make the reason appear and disappear
            # with the grid, which reads as a claim about the run.
            if CERTIFIED_FUSED_PAIR_ARMS.get(name) is None:
                reasons[key] = (
                    f"{name} has no absorb declaration: which arm each of its two "
                    f"slots implements has not been established, and a fused "
                    f"product may not substitute a numerical product no arm "
                    f"admitted",)
                continue
            # THE PRODUCT'S OWN INSTALLATION DECLARATION, ASKED BESIDE THE ABSORB
            # ONE AND FOR THE SAME REASON: it is a fact about the PRODUCT, true of
            # every configuration, so it is reported on every configuration. See
            # :func:`_declared_uninstallable`.
            uninstallable = _declared_uninstallable(modules, name, product)
            if uninstallable is not None:
                reasons[key] = (uninstallable,)
                continue
            entry, read_refusal = _guarded_plan(
                lambda product=product: _certified_fused_product_entry(modules,
                                                                       product),
                f"the {name} module")
            if read_refusal is not None or entry is None:
                reasons[key] = (read_refusal
                                or f"the {name} module was refused",)
                continue
            coverage, builder, repair_paths = entry
            verdict = _guarded_verdict(
                lambda coverage=coverage: coverage(fields, pml, sources),
                f"the {name} pair predicate")
            if not verdict.covered:
                reasons[key] = verdict.reasons
                continue
            candidates.append((name, builder, repair_paths))
        if not candidates:
            continue
        if len(candidates) > 1:
            named = ", ".join(sorted(name for name, _b, _r in candidates))
            for name, _builder, _paths in candidates:
                reasons[f"fused_pair_{name}"] = (
                    f"{named} all claim the {curl_name}/{update_name} seam; it is "
                    f"left unfused rather than assigned by table order",)
            continue
        name, builder, repair_paths = candidates[0]
        key = f"fused_pair_{name}"
        # THE PRIMARY ROW FIRST, THEN EVERY DECLARED EXTRA, and the pair absorbs on
        # the FIRST arm pair the table already gave both slots to. At most one can
        # ever match — the slots hold one selection each — so the loop is a lookup,
        # not a preference order. A product with no matching declaration is refused
        # with the PRIMARY row's reason, which names the arm the slots went to.
        refusal = _pair_may_absorb(selected, curl_name, update_name,
                                   CERTIFIED_FUSED_PAIR_ARMS[name])
        if refusal is not None:
            for extra_arms in CERTIFIED_FUSED_PAIR_EXTRA_ARMS.get(name, ()):
                if _pair_may_absorb(selected, curl_name, update_name,
                                    extra_arms) is None:
                    refusal = None
                    break
        if refusal is not None:
            reasons[key] = (refusal,)
            continue
        # A NEIGHBOURING SEAM'S CLAIM, ASKED BEFORE THIS PRODUCT IS BUILT. See
        # :func:`_neighbouring_seam_claimant`: over ``step_B..update_E`` launches are
        # ``4 - #pairs``, so a product taking a slot from a neighbouring seam can only
        # tie or lose -- and an END-EDGE span is never asked. No product in this table
        # spans the interior seam, so this refuses nothing that installs today.
        neighbour = _neighbouring_seam_claimant(modules, fields, pml, sources, name,
                                                curl_name, update_name,
                                                (curl_name, update_name))
        if neighbour is not None:
            reasons[key] = (neighbour[1],)
            continue
        not_offered = _label_not_offered(
            CERTIFIED_FUSED_PRODUCTS[name]["label"], offered)
        if not_offered is not None:
            reasons[key] = (not_offered,)
            continue
        pair, builder_refusal = _guarded_plan(
            lambda builder=builder: builder(fields, pml, sources, block,
                                            num_warps),
            f"the {name} pair builder")
        if builder_refusal is not None:
            reasons[key] = (builder_refusal,)
        elif pair is not None:
            _install_fused_pair(plans, selected, fields, pml, sources, pair_name,
                                curl_name, update_name,
                                CERTIFIED_FUSED_PRODUCTS[name]["label"], pair,
                                repair_paths=repair_paths)
        else:
            reasons[key] = (f"the {name} pair was refused",)


def _driven_components(state: Any) -> Tuple[str, ...]:
    """The components one susceptibility drives, or ``()`` when it will not say.

    ``driven()`` is engine code called from the composer, and a state whose
    ``driven()`` raises used to take the whole plan down with it while the very
    next line was about to record a refusal.
    """
    driven = getattr(state, "driven", None)
    if not callable(driven):
        return ()
    try:
        return tuple(driven())
    except Exception:  # noqa: BLE001 - an unreadable state drives nothing here
        return ()


def _ade_family_admits(states: Sequence[Any], ask: Any) -> bool:
    """Does ONE ``update_P`` family admit EVERY driven component of EVERY state?

    The unit of admission for this sub-step is the whole polarization pass, so the
    unit of the disjointness question is too. Anything less than unanimous is not
    an admission: ``plan_ade_update_p`` and ``plan_no_pml_ade_update_p`` are each
    all-or-nothing per state for the same reason
    (``PolarizationState.update`` rotates ONE shared scratch buffer across the
    components of a state), and the composer then keeps the whole sub-step on the
    array path if any state refuses.

    A STATE THAT DRIVES NOTHING IS NOT AN ADMISSION. An empty component list would
    make ``all(())`` true and hand the family a unanimous verdict over no
    evidence — the same vacuity the refusal path spells ``"<none driven>"``.

    A raising predicate is a refusal, through :func:`_guarded_verdict`, exactly as
    it is in the arm table.
    """
    if not states:
        return False
    for state in states:
        components = _driven_components(state)
        if not components:
            return False
        for component in components:
            verdict = _guarded_verdict(
                lambda component=component, state=state: ask(state, component),
                "an update_P predicate")
            if not verdict.covered:
                return False
    return True


#: The ``update_E`` arms that DO implement the off-diagonal row product, plus
#: ``None`` for an already-empty slot. Everything else is vetoed off a grid on
#: which the ARRAY PATH forms that product — the flag and a live row slot, both,
#: see :func:`_veto_dropped_offdiagonal_coupling`.
#:
#: ``no-PML null`` IS DELIBERATELY NOT HERE, and adding it would be the wrong
#: repair for the right complaint. Its product implements nothing, so listing it
#: among "the arms that implement the row product" makes one tuple mean two
#: things; and it is unreachable under the veto's conjunction anyway, because
#: ``no_pml_constitutive``'s own E-side predicate refuses an installed
#: off-diagonal row by name (no_pml_constitutive.py:514). What used to empty that
#: slot was the missing FLAG half of the trigger, and that is where it is fixed.
#:
#: THIS TUPLE IS PART OF THE SAME CHANGE AS THE ``folded off-diagonal`` ARM, and
#: it has to be: measured on both folded row-live grids, that arm ADMITS and
#: ``live_offdiagonal_rows`` is True, so with the old two-entry exemption the
#: veto fired on the one product that implements exactly the term the message
#: says is missing. The failure was silent in the worst way — the slot fell to
#: the array path, which is the correct answer, carrying a reason that was false
#: about the product it named.
ROW_PRODUCT_ARMS: Tuple[Optional[str], ...] = (
    None,
    "off-diagonal",
    "folded off-diagonal",
    "folded off-diagonal dispersive",
    "complex folded off-diagonal",
    "complex no-PML off-diagonal",
)


def _veto_dropped_offdiagonal_coupling(fields: Any, plans: Dict[str, Any],
                                       reasons: Dict[str, Tuple[str, ...]],
                                       selected: Dict[str, str]) -> None:
    """Refuse an ``update_E`` product that does not implement a LIVE row.

    THE HOLE THIS CLOSES. Thirteen of the eighteen ``update_E`` arms decide
    off-diagonal epsilon by the ``has_offdiagonal_epsilon`` FLAG; the five that own
    the row product (``off-diagonal`` and ``folded off-diagonal``, named together
    in :data:`ROW_PRODUCT_ARMS`) deliberately refuse to trust that flag and count
    LIVE ROW SLOTS instead. Where the flag and the slots disagree, a flag-reading
    arm can win a slot whose grid carries a coupling its kernel has no term for:
    the coupling would be neither refused, nor reported, nor computed.

    BOTH HALVES OF THAT DISAGREEMENT ARE TESTED, and the round that wired the
    four remaining families shipped only one of them. The trigger used to be the
    live ROW SLOTS alone, which is the wrong half on its own, because the
    ARRAY PATH THIS COMPOSER IS MEASURED AGAINST reads the FLAG:
    ``stepping.update_E`` sets ``offdiagonal = fields.has_offdiagonal_epsilon``
    (stepping.py:992) and forms the row product only under it. So on a
    flag-False/slot-live object the array path drops the row too, and a Triton
    product that also drops it is not wrong — it is byte-exact. MEASURED on
    identical-input copies of one such object, one with ``ROW_SLOTS[0]`` planted
    and one without: ``stepping.update_E`` writes 0 of 1536 differing uint32
    words under an active PML and 0 of 1536 under an inert layer, against 512 of
    1536 for the same pair with the real class's flag. The slot-only veto
    therefore removed CORRECT selections and installed a reason that was false
    about the product it named — including on the newly wired ``no-PML null``
    arm, whose product is a sub-step that executes no statement at all.

    So the veto fires on the CONJUNCTION: the array path would form the row
    product (the flag), a row is actually live (the slots), and the winning arm
    is not one of the two that implement it. Unreadable answers admit on both —
    :func:`_reads_true` and :func:`live_offdiagonal_rows` both answer True — so
    an object that cannot be read still fails closed.

    WHAT IT IS WORTH NOW. Under the shipped predicates the conjunction is
    unreachable: all thirteen non-exempt ``update_E`` predicates carry their own
    ``has_offdiagonal_epsilon`` refusal, so a flag-True grid leaves the slot to
    the row-product arms or to nobody (measured: BFAST, beta and complex flag-True
    grids select nothing for ``update_E`` and every arm refuses by name). That
    makes this a defence in depth rather than a live rescue, and it is kept
    because the thing it defends is one deleted clause away in any of thirteen
    files — a mutant that drops one is caught HERE, at the composition, rather
    than by whichever gate happens to run that family next.

    The fix cannot live in the gates. Gating those arms off would make a GATE
    the reason a competing arm wins, which is precisely what the gate rule
    forbids (clause (e)). So it lives here, after selection, as a veto: the slot
    goes to the array path with a named reason, which is clause (d) and is
    always correct.

    Applied AFTER the fusion block on purpose — a fused D/E pair inlines the
    same flag-reading constitutive product, so a veto placed before it would be
    overwritten by the very product it refused. When the vetoed ``update_E`` was
    absorbed into a fused pair, that pair's curl slot goes with it: the pair is
    ONE kernel and half of it cannot be kept.
    """
    if selected.get("update_E") in ROW_PRODUCT_ARMS:
        return
    if not _reads_true(fields, "has_offdiagonal_epsilon"):
        return
    if not live_offdiagonal_rows(fields):
        return
    arm = selected.pop("update_E")
    plan = plans.pop("update_E", None)
    reasons["update_E"] = (
        "has_offdiagonal_epsilon is set and a live off-diagonal chi1inv row "
        "slot is installed, so the array path forms the row product "
        f"(stepping.py:992); the {arm!r} product selected for update_E has no "
        "row-product term: it would drop the coupling silently. Only the "
        "off-diagonal arms implement it, and none of them won this slot",
    )
    if isinstance(plan, NoopPlan):
        for curl_name in ("step_D", "step_B"):
            if plans.get(curl_name) is getattr(plan, "absorbed_by", None):
                plans.pop(curl_name, None)
                selected.pop(curl_name, None)
                reasons[curl_name] = (
                    "the fused pair that carried this curl also carried the "
                    "vetoed update_E; one kernel cannot be half-kept",
                )


def _curl_ambiguity(labels: Sequence[str]) -> str:
    return (f"curl predicates {', '.join(labels)} admitted the same "
            "configuration; refusing an ambiguous curl product")


def _fill_ambiguity(labels: Sequence[str]) -> str:
    return (f"mirror-fill predicates {', '.join(labels)} admitted the same "
            "field family; refusing an ambiguous ghost-fill product")


def _update_h_ambiguity(labels: Sequence[str]) -> str:
    return (f"multiple update_H predicates ({', '.join(labels)}) admitted the "
            "same configuration; refusing an ambiguous numerical product")


def _update_e_ambiguity(labels: Sequence[str]) -> str:
    quantifier = "both" if len(labels) == 2 else "multiple"
    return (f"{quantifier} update_E predicates ({', '.join(labels)}) admitted "
            "the same configuration; refusing an ambiguous numerical product")


def plan_step(fields: Any, pml: Any, block: Optional[int] = None,
              fuse: bool = False, sources: Any = None,
              num_warps: Optional[int] = FUSED_DEFAULT_NUM_WARPS,
              fuse_ade: bool = False, probe: Any = None,
              fuse_labels: Any = None,
              ) -> TritonStepPlan:
    """Ask every predicate independently and report the covered subset with its refusals.

    Never raises and never returns None: a configuration nothing covers is a plan
    that replaces nothing, which is the array path, which is the correct answer.

    THE FAIL-CLOSED CONTRACT, for every slot and every ``(fields, pml)``:

    (a) a slot is filled only when EXACTLY ONE consulted predicate admits and
        its builder returns a plan;
    (b) two or more admitting predicates leave the slot UNSELECTED with an
        "ambiguous" reason naming every admitter — never a pick by table order,
        never a pick by specificity;
    (c) a predicate that raises counts as a refusal; a builder that raises or
        returns None leaves the slot unselected with a named reason;
    (d) an unselected slot is the ARRAY PATH, which is always correct — partial
        coverage of a step is legal and expected;
    (e) gating an arm off is permitted only when the gate is a NECESSARY
        condition of that arm's admission (see the gate rule above); a gate may
        never be the reason a competing arm wins;
    (f) a selected product that does not implement a LIVE feature of the grid is
        VETOED back to the array path with a named reason — see
        :func:`_veto_dropped_offdiagonal_coupling`, the one such feature measured
        today. A veto is not a selection rule and cannot hand the slot to anyone
        else: it only empties it.

    An over-covering dispatch is a silent wrong answer; a slot on the array path
    is a slower right answer. Every ambiguity resolves to the second.

    THE NULL FAMILY IS NOT AN EXCEPTION TO ANY OF THIS. ``no-PML null`` is the
    one arm whose product launches nothing — the array path's ``update_H``
    returns at stepping.py:944-945 under an inactive absorber, and ``update_E``
    at :983-984 under an inactive absorber AND ``fields.stores_E`` False, which
    is why :func:`absorber_inactive` gates the arm and the family's own predicate
    is what refuses stored E — and it goes through the same table, the same gate
    rule and the same :func:`_select_slot`. It is never preferred over a kernel
    (that would silently delete a real sub-step) and a kernel is never preferred
    over it (that would launch the ``dsigw`` accumulation on a run whose
    ``f_w_*`` arrays are not allocated). Both directions are wrong, so clause (b)
    applies unchanged — and here failing closed costs nothing at runtime, because
    the array-path call it falls back to is the one that returns immediately.

    ONE CONSEQUENCE FOR ``replaces``, named here because it is a meaning change
    and not an accident. Every other arm refuses a host whose array module is not
    CuPy at ``coverage._grid_reasons`` clause 1; the null family carries no such
    clause, deliberately (no_pml_constitutive.py:610-613 — a sub-step that
    returns before its first statement launches nothing, so it is correct on
    NumPy). So on a plain NumPy host with an inert layer ``plan.replaces`` is
    ``("update_H", "update_E")`` where it used to be ``()``. That is the true
    answer — those two sub-steps really are replaced, by a plan that does what
    the array path does, namely nothing — but a consumer reading
    ``bool(plan.replaces)`` as "a Triton KERNEL covers something here" is now
    reading it wrong. ``plan.selected`` names the arm and is the seam that still
    answers that question.

    ``fill_B``/``fill_D`` GO THROUGH THE TABLE TOO. They used to consult a single
    predicate in a hand-rolled block, which made them the last slots here whose
    second admitting product would have been decided by position.

    "NEVER RAISES" IS ENFORCED AT EVERY CALL SITE, not only inside the arm table.
    The opt-in fusion block and the polarization loop go
    through :func:`_guarded_verdict` / :func:`_guarded_plan` for the same reason
    :func:`_arm_verdict` exists, and every gate answers an unreadable attribute
    by consulting its arm rather than by aborting the plan. Before that, a
    Triton-less host — the stated merge bar — got a ``ModuleNotFoundError`` out
    of this function on every dispersive run and on every ``fuse=True`` run, from
    a builder import inside a block that had no guard.

    A FUSED PAIR MAY ABSORB A SLOT ONLY WHEN THE ARM TABLE GAVE THAT SLOT TO THE
    ARM THE FUSED KERNEL IMPLEMENTS (:func:`_pair_may_absorb`). The fusion block
    is the only place here that fills a ``STEP_ORDER`` slot without consulting the
    arm table, and writing it unconditionally was a hole straight through clauses
    (b) and (c): a slot left unselected by an ambiguity, or by a builder that
    refused, was claimed anyway by the ordinary fused product — the very product
    the ambiguity had refused to pick — leaving ``reasons`` and ``plans``
    contradicting each other for the same slot.

    ``probe`` is the complex-multiply expansion record, resolved ONCE here and
    passed explicitly to every complex-family predicate and builder. Reading it
    per predicate call would open a file up to eight times per plan and could
    see two different records inside one composition. It is the LAST parameter
    so every positional caller is unaffected.

    ``fuse`` IS OPT-IN AND DEFAULTS OFF. The composed-but-unfused plan — one launch
    per sub-step — is the control the cross-sub-step fusion is measured against,
    and it stays the default so every existing caller keeps measuring it. With
    ``fuse=True`` and ``fused_pair_coverage`` satisfied, ``step_B`` holds a
    :class:`FusedPairPlan` and ``update_H`` a :class:`NoopPlan`: ``replaces``
    still reports BOTH names, because both sub-steps really are replaced.

    ``fuse_labels`` IS THE CALLER'S OFFER, AND IT IS A SECOND QUESTION FROM
    ``fuse``. ``fuse`` asks whether to ATTEMPT fusion at all; ``fuse_labels`` names
    the fused labels this caller will actually RUN. ``None`` — the default, and what
    every inspecting caller passes — means no restriction. A label left out of a
    non-``None`` offer leaves its seam on the separate sub-step plans with a named
    reason, exactly as a refused predicate does; see :func:`_label_not_offered` for
    the measured failure this closes, which is that a fused label occupies BOTH
    slots, so a caller that has to reject one rejects the arms underneath it too.
    """
    plans: Dict[str, Any] = {}
    reasons: Dict[str, Tuple[str, ...]] = {}
    selected: Dict[str, str] = {}
    offered = _offered_labels(fuse_labels)
    has_fold = folded_grid_active(fields)
    has_cylindrical = cylindrical_grid_active(fields)
    has_complex = complex_storage_active(fields)
    has_beta = beta_active(fields)
    has_bfast = bfast_grid_active(fields)
    has_nonlinearity = nonlinear_active(fields)
    has_offdiag_rows = offdiag_rows_possible(fields)
    has_poles = dispersion_active(fields)
    # The one gate computed from the LAYER rather than from the fields, and the
    # only one whose arm builds no kernel. Read here with the others so the
    # whole gate block still runs before any predicate does.
    no_active_absorber = absorber_inactive(pml)

    # Resolved ONCE, and only when a complex-family arm is gated on — every one
    # is gated on complex storage, so an ordinary run never opens
    # the artifact and never imports that family's module.
    #
    # THE INVARIANT THIS RUNG BROKE AND NOW STATES. Reading the environment here
    # is legitimate for exactly ONE of the three things ``probe`` can be, and
    # this composer used to do it for two of them:
    #
    #   a record            -> use it, never re-read.
    #   None                -> NOTHING WAS OFFERED. Read the environment; a host
    #                          with no artifact makes the complex predicates
    #                          refuse by name, which is the right answer.
    #   RefusedExpansionProbe -> SOMETHING WAS OFFERED AND REFUSED. Re-reading is
    #                          how the refusal was defeated: the caller's drop
    #                          set only ITS OWN local name to None, this rung
    #                          re-opened the same file, and the refused artifact
    #                          was handed EXPLICITLY to all eight complex arms
    #                          below. Measured: 4/7 slots bound complex arms over
    #                          8004 launches from an artifact the policy clause
    #                          had already refused.
    #
    # Both of the last two used to arrive as ``None``. They no longer do, so the
    # ``is None`` test below means what it says — and the clause after it is what
    # keeps that true if someone reintroduces the conflation upstream.
    from ..expansion_refusal import is_expansion_refusal  # noqa: PLC0415

    resolved_probe = probe
    if resolved_probe is None and has_complex:
        resolved_probe = load_expansion_probe()
    if is_expansion_refusal(probe):
        # WHAT THIS DOES TODAY: NOTHING, and it is kept deliberately. Audited
        # 2026-08-16 — ``resolved_probe`` starts as ``probe`` two lines above,
        # and the re-read is guarded on ``resolved_probe is None``, which a
        # refusal never is; so the read cannot fire for a refusal and this
        # assignment restores what the name already holds. The line is a
        # BACKSTOP, not the enforcement, and the earlier claim that it is "what
        # the refusal has to survive" overstated it.
        #
        # WHEN IT BECOMES LOAD-BEARING: the moment the guard above stops being
        # conditional on ``resolved_probe is None`` — an unconditional
        # ``resolved_probe = load_expansion_probe()``, or a guard widened to
        # anything a refusal satisfies. Then the read would replace the refusal
        # with the very artifact it refused, and this line is what puts it back
        # before the composer hands ``resolved_probe`` EXPLICITLY to all eight
        # complex arms. That is the exact shape of the defect this value exists
        # to end, which is why the backstop stays.
        #
        # Restoring rather than raising because this function's whole contract is
        # that it never raises — a configuration nothing covers is a plan that
        # replaces nothing, which is the array path.
        resolved_probe = probe

    for name in ("step_B", "step_D"):
        _select_slot(name, (
            _Arm("PML", True,
                 lambda: pml_curl_coverage(fields, pml, name),
                 lambda: plan_pml_curl(fields, pml, name, block),
                 "PML: ", "PML curl"),
            _Arm("conductive PML", True,
                 lambda: conductive_pml_curl_coverage(fields, pml, name),
                 lambda: plan_conductive_pml_curl(fields, pml, name, block),
                 "conductive PML: ", "conductive PML"),
            _Arm("no-PML", True,
                 lambda: plain_curl_coverage(fields, pml, name),
                 lambda: plan_plain_curl(fields, pml, name, block),
                 "no-PML: ", "no-PML curl"),
            # The no-PML incumbent refuses any curl conductivity; this arm
            # inverts exactly that clause.  Its gate is the same layer question
            # as the predicate, so a closed gate can remove only a refusal.
            _Arm("conductive no-PML", no_active_absorber,
                 lambda: conductive_plain_curl_coverage(fields, pml, name),
                 lambda: plan_conductive_plain_curl(fields, pml, name, block),
                 "conductive no-PML: ", "conductive no-PML curl"),
            _Arm("folded PML", has_fold,
                 lambda: folded_composition_curl_coverage(fields, pml, name),
                 lambda: plan_folded_pml_curl(fields, pml, name, block),
                 "folded PML: ", "folded PML curl"),
            _Arm("cylindrical PML", has_cylindrical,
                 lambda: cylindrical_curl_coverage(fields, pml),
                 lambda: plan_cylindrical_curl(fields, pml, name, block),
                 "cylindrical PML: ", "cylindrical PML curl"),
            _Arm("complex PML", has_complex,
                 lambda: complex_pml_curl_coverage(fields, pml, name,
                                                   resolved_probe),
                 lambda: plan_complex_pml_curl(fields, pml, name, block,
                                               resolved_probe),
                 "complex PML: ", "complex PML curl"),
            # Complex storage plus a material conductivity needs both the
            # complex/Bloch stencil and the per-target conductive recurrence.
            # The lossless complex sibling below refuses every conductivity,
            # while this predicate requires at least one, so they are disjoint.
            _Arm("complex conductive no-PML curl",
                 has_complex and no_active_absorber,
                 lambda: complex_conductive_no_pml_curl_coverage(
                     fields, pml, name, resolved_probe),
                 lambda: plan_complex_conductive_no_pml_curl(
                     fields, pml, name, block, resolved_probe),
                 "complex conductive no-PML curl: ",
                 "complex conductive no-PML curl"),
            # The complex PML predicate requires an active layer.  This is its
            # no-absorber sibling, reusing the measured unit-coefficient binding
            # and deliberately covering curls only (not stored-E update_E).
            _Arm("complex no-PML curl", has_complex and no_active_absorber,
                 lambda: complex_no_pml_curl_coverage(fields, pml, name,
                                                       resolved_probe),
                 lambda: plan_complex_no_pml_curl(fields, pml, name, block,
                                                   resolved_probe),
                 "complex no-PML curl: ", "complex no-PML curl"),
            _Arm("real beta PML", has_beta,
                 lambda: beta_pml_curl_coverage(fields, pml, name),
                 lambda: plan_beta_pml_curl(fields, pml, name, block),
                 "real beta PML: ", "real beta PML curl"),
            _Arm("complex beta PML", has_beta and has_complex,
                 lambda: beta_bloch_pml_curl_coverage(fields, pml, name,
                                                      resolved_probe),
                 lambda: plan_beta_bloch_pml_curl(fields, pml, name, block,
                                                  resolved_probe),
                 "complex beta PML: ", "complex beta PML curl"),
            _Arm("BFAST PML", has_bfast,
                 lambda: bfast_pml_curl_coverage(fields, pml, name),
                 lambda: plan_bfast_pml_curl(fields, pml, name, block),
                 "BFAST PML: ", "BFAST PML curl"),
            # K1. The COMPOSITION verdict, not the standalone one: the standalone
            # predicate admits an unfolded grid so its own gate can prove
            # reduction to the certified unfolded kernel, and routing on that
            # would put two products on every unfolded complex grid.
            _Arm("folded complex PML", has_fold and has_complex,
                 lambda: folded_complex_composition_curl_coverage(
                     fields, pml, name, resolved_probe),
                 lambda: plan_folded_complex_pml_curl(fields, pml, name, block,
                                                      probe=resolved_probe),
                 "folded complex PML: ", "folded complex PML curl"),
            # K3a/K3b carry ``_has_real_fold`` inside their own predicates, so
            # their base verdicts ARE composition verdicts and need no split.
            _Arm("folded real beta PML", has_fold and has_beta,
                 lambda: folded_beta_pml_curl_coverage(fields, pml, name),
                 lambda: plan_folded_beta_pml_curl(fields, pml, name, block),
                 "folded real beta PML: ", "folded real beta PML curl"),
            _Arm("folded complex beta PML", has_fold and has_beta and has_complex,
                 lambda: folded_beta_bloch_pml_curl_coverage(fields, pml, name,
                                                             resolved_probe),
                 lambda: plan_folded_beta_bloch_pml_curl(fields, pml, name, block,
                                                         probe=resolved_probe),
                 "folded complex beta PML: ", "folded complex beta PML curl"),
            _Arm("cylindrical complex PML", has_cylindrical and has_complex,
                 lambda: cylindrical_complex_curl_coverage(fields, pml, name,
                                                           resolved_probe),
                 lambda: plan_cylindrical_complex_curl(fields, pml, name, block,
                                                       probe=resolved_probe),
                 "cylindrical complex PML: ", "cylindrical complex PML curl"),
            # The CERTIFIED real curl on a chi2/chi3 run. `pml_curl_coverage`
            # refuses it at `_grid_reasons` clause 10, which is written "refused
            # on every sub-step rather than only on update_E" — and the curls are
            # the sub-steps that never read chi2 or chi3. Disjoint from the `PML`
            # arm above in both directions by that one clause.
            _Arm("nonlinear run PML", has_nonlinearity,
                 lambda: nonlinear_run_pml_curl_coverage(fields, pml, name),
                 lambda: plan_nonlinear_run_pml_curl(fields, pml, name, block),
                 "nonlinear run PML: ", "nonlinear run PML curl"),
            # K1's curl on a folded complex run that ALSO carries an off-diagonal
            # row. The incumbent `folded complex PML` arm refuses the row through
            # the shared `_media_reasons` clause, which is left untouched: this
            # arm is what narrows that family's off-diagonal refusal to update_E,
            # where the row product actually lives.
            _Arm("folded complex off-diagonal PML",
                 has_fold and has_complex and has_offdiag_rows,
                 lambda: folded_complex_offdiag_pml_curl_coverage(
                     fields, pml, name, resolved_probe),
                 lambda: plan_folded_complex_offdiag_pml_curl(
                     fields, pml, name, block, probe=resolved_probe),
                 "folded complex off-diagonal PML: ",
                 "folded complex off-diagonal PML curl"),
        ), _curl_ambiguity, plans, reasons, selected)

    if has_fold:
        # THE FILL SLOTS GO THROUGH THE ARM TABLE TOO, as of the round that added
        # the complex-storage fill. Before that this block consulted ONE predicate
        # and hand-rolled its own guards, which made it the last place in the
        # function where a second admitting product would have been decided by
        # position rather than refused. The two arms split on STORAGE and were
        # measured never to co-admit, so the conversion changes no verdict on any
        # configuration in the disjointness sweep — it buys uniform fail-closure
        # and one spelling of "a raising predicate is a refusal".
        #
        # A PARTIAL FOLD COVERAGE IS LEGAL AND HAPPENS: neither fill predicate
        # carries an absorber clause, correctly — `fill_symmetry_bc_*` runs
        # whether or not a PML is installed — so on a folded no-PML run these two
        # slots are covered while every curl and constitutive slot is on the array
        # path. That is a substitution with no throughput benefit and a driver
        # adapter seam, and it is named here rather than discovered later.
        for family, slot in (("B", "fill_B"), ("D", "fill_D")):
            _select_slot(slot, (
                _Arm("mirror fill", has_fold,
                     lambda family=family: mirror_ghost_fill_coverage(fields, family),
                     lambda family=family: plan_mirror_ghost_fill(fields, family,
                                                                  block),
                     "mirror fill: ", "mirror-fill"),
                # K2. Its adapter contract is NOT MirrorGhostFillPlan's — the two
                # passes may not be fused, see plan_folded_mirror_ghost_fill_complex
                # — and ``selected[slot]`` naming the arm is the only seam an
                # adapter can branch on.
                _Arm("folded complex fill", has_fold and has_complex,
                     lambda family=family: folded_mirror_ghost_fill_complex_coverage(
                         fields, family, resolved_probe),
                     lambda family=family: plan_folded_mirror_ghost_fill_complex(
                         fields, family, block, probe=resolved_probe),
                     "folded complex fill: ", "folded complex mirror-fill"),
            ), _fill_ambiguity, plans, reasons, selected)

    # The three shipped update_H arms stay UNPREFIXED: that is what they report
    # today, and retrofitting prefixes onto them is a behaviour change with no
    # gate behind it. The specialized arms, which no gate has seen here before,
    # are labelled.
    _select_slot("update_H", (
        _Arm("ordinary", True,
             lambda: constitutive_coverage(fields, pml, "H"),
             lambda: plan_constitutive(fields, pml, "H", block),
             "", "ordinary constitutive"),
        _Arm("folded", has_fold,
             lambda: folded_constitutive_coverage(fields, pml, "H"),
             lambda: plan_folded_constitutive(fields, pml, "H", block),
             "", "folded constitutive"),
        _Arm("cylindrical", has_cylindrical,
             lambda: cylindrical_constitutive_coverage(fields, pml, "H"),
             lambda: plan_cylindrical_constitutive(fields, pml, "H", block),
             "", "cylindrical constitutive"),
        _Arm("complex", has_complex,
             lambda: complex_constitutive_coverage(fields, pml, "H",
                                                   resolved_probe),
             lambda: plan_complex_constitutive(fields, pml, "H", block,
                                               resolved_probe),
             "complex: ", "complex constitutive"),
        _Arm("real beta run", has_beta,
             lambda: beta_run_constitutive_coverage(fields, pml, "H"),
             lambda: plan_beta_run_constitutive(fields, pml, "H", block),
             "real beta run: ", "real beta run constitutive"),
        _Arm("complex beta run", has_beta and has_complex,
             lambda: beta_run_complex_constitutive_coverage(fields, pml, "H",
                                                            resolved_probe),
             lambda: plan_beta_run_complex_constitutive(fields, pml, "H", block,
                                                        resolved_probe),
             "complex beta run: ", "complex beta run constitutive"),
        _Arm("BFAST run", has_bfast,
             lambda: bfast_run_constitutive_coverage(fields, pml, "H"),
             lambda: plan_bfast_run_constitutive(fields, pml, "H", block),
             "BFAST run: ", "BFAST run constitutive"),
        # No beta clause on this arm, deliberately (special_kz grouping choice
        # 5): the constitutive kernel is storage-dependent, not beta-dependent,
        # so it serves the folded complex run with and without beta and the
        # `folded beta run` arm beside it refuses complex storage.
        _Arm("folded complex", has_fold and has_complex,
             lambda: folded_complex_constitutive_coverage(fields, pml, "H",
                                                          resolved_probe),
             lambda: plan_folded_complex_constitutive(fields, pml, "H", block,
                                                      probe=resolved_probe),
             "folded complex: ", "folded complex constitutive"),
        _Arm("folded beta run", has_fold and has_beta,
             lambda: folded_beta_run_constitutive_coverage(fields, pml, "H"),
             lambda: plan_folded_beta_run_constitutive(fields, pml, "H", block),
             "folded beta run: ", "folded beta run constitutive"),
        _Arm("cylindrical complex", has_cylindrical and has_complex,
             lambda: cylindrical_complex_constitutive_coverage(fields, pml, "H",
                                                               resolved_probe),
             lambda: plan_cylindrical_complex_constitutive(fields, pml, "H", block,
                                                           probe=resolved_probe),
             "cylindrical complex: ", "cylindrical complex constitutive"),
        # The CERTIFIED constitutive kernel on a chi2/chi3 run's MAGNETIC side.
        # `constitutive_coverage` refuses it through the same `_grid_reasons`
        # clause 10 the curls are refused by, and update_H reads B and mu with the
        # integer coefficient pair — never chi2, chi3 or inverse epsilon. The
        # family predicate refuses side='E' BY NAME, so this arm cannot appear in
        # the update_E table and the Pade product keeps that slot.
        _Arm("nonlinear run", has_nonlinearity,
             lambda: nonlinear_run_constitutive_coverage(fields, pml, "H"),
             lambda: plan_nonlinear_run_constitutive(fields, pml, "H", block),
             "nonlinear run: ", "nonlinear run constitutive"),
        # The update_H half of the folded complex off-diagonal scope change. Its
        # side='E' twin is refused by name and stays refused: 25041/110592 words
        # differing is the largest divergence the closure round measured.
        _Arm("folded complex off-diagonal",
             has_fold and has_complex and has_offdiag_rows,
             lambda: folded_complex_offdiag_constitutive_coverage(
                 fields, pml, "H", resolved_probe),
             lambda: plan_folded_complex_offdiag_constitutive(
                 fields, pml, "H", block, probe=resolved_probe),
             "folded complex off-diagonal: ",
             "folded complex off-diagonal constitutive"),
        # THE NULL ARM. Its product launches nothing because the array path's
        # `update_H` returns at stepping.py:944-945 before reading an array. It
        # gets NO preference of any kind, and the reason is symmetric: a null
        # preferred over a kernel silently deletes a real sub-step, and a kernel
        # preferred over the null launches the dsigw accumulation on a step that
        # allocates no `f_w_*` at all. Both directions are wrong, so an overlap
        # here fails closed exactly as everywhere else — and failing closed costs
        # nothing here, because the "slower right answer" is an array-path call
        # that returns immediately. LAST in the table so its refusal appends as a
        # suffix rather than reordering the frozen ones above it.
        _Arm("no-PML null", no_active_absorber,
             lambda: null_constitutive_coverage(fields, pml, "H"),
             lambda: plan_null_constitutive(fields, pml, "H", block),
             "no-PML null: ", "no-PML null constitutive"),
    ), _update_h_ambiguity, plans, reasons, selected)

    # update_E has the most products — twenty-one — and is where every E-only
    # family lives: the pole product, chi2/chi3 and BOTH row products. Every
    # consulted arm is evaluated before any is selected.
    e_admitted = _select_slot("update_E", (
        _Arm("ordinary", True,
             lambda: constitutive_coverage(fields, pml, "E"),
             lambda: plan_constitutive(fields, pml, "E", block),
             "ordinary: ", "ordinary constitutive"),
        _Arm("dispersive", True,
             lambda: dispersive_constitutive_coverage(fields, pml),
             lambda: plan_dispersive_constitutive(fields, pml, block),
             "dispersive: ", "dispersive constitutive"),
        _Arm("folded", has_fold,
             lambda: folded_constitutive_coverage(fields, pml, "E"),
             lambda: plan_folded_constitutive(fields, pml, "E", block),
             "folded: ", "folded constitutive"),
        _Arm("cylindrical", has_cylindrical,
             lambda: cylindrical_constitutive_coverage(fields, pml, "E"),
             lambda: plan_cylindrical_constitutive(fields, pml, "E", block),
             "cylindrical: ", "cylindrical constitutive"),
        _Arm("complex", has_complex,
             lambda: complex_constitutive_coverage(fields, pml, "E",
                                                   resolved_probe),
             lambda: plan_complex_constitutive(fields, pml, "E", block,
                                               resolved_probe),
             "complex: ", "complex constitutive"),
        _Arm("real beta run", has_beta,
             lambda: beta_run_constitutive_coverage(fields, pml, "E"),
             lambda: plan_beta_run_constitutive(fields, pml, "E", block),
             "real beta run: ", "real beta run constitutive"),
        _Arm("complex beta run", has_beta and has_complex,
             lambda: beta_run_complex_constitutive_coverage(fields, pml, "E",
                                                            resolved_probe),
             lambda: plan_beta_run_complex_constitutive(fields, pml, "E", block,
                                                        resolved_probe),
             "complex beta run: ", "complex beta run constitutive"),
        _Arm("BFAST run", has_bfast,
             lambda: bfast_run_constitutive_coverage(fields, pml, "E"),
             lambda: plan_bfast_run_constitutive(fields, pml, "E", block),
             "BFAST run: ", "BFAST run constitutive"),
        _Arm("nonlinear", has_nonlinearity,
             lambda: nonlinear_constitutive_coverage(fields, pml),
             lambda: plan_nonlinear_constitutive(fields, pml, block),
             "nonlinear: ", "nonlinear constitutive"),
        _Arm("off-diagonal", has_offdiag_rows,
             lambda: offdiag_constitutive_coverage(fields, pml),
             lambda: plan_offdiagonal_constitutive(fields, pml, block),
             "off-diagonal: ", "off-diagonal constitutive"),
        _Arm("folded complex", has_fold and has_complex,
             lambda: folded_complex_constitutive_coverage(fields, pml, "E",
                                                          resolved_probe),
             lambda: plan_folded_complex_constitutive(fields, pml, "E", block,
                                                      probe=resolved_probe),
             "folded complex: ", "folded complex constitutive"),
        _Arm("folded beta run", has_fold and has_beta,
             lambda: folded_beta_run_constitutive_coverage(fields, pml, "E"),
             lambda: plan_folded_beta_run_constitutive(fields, pml, "E", block),
             "folded beta run: ", "folded beta run constitutive"),
        _Arm("cylindrical complex", has_cylindrical and has_complex,
             lambda: cylindrical_complex_constitutive_coverage(fields, pml, "E",
                                                               resolved_probe),
             lambda: plan_cylindrical_complex_constitutive(fields, pml, "E", block,
                                                           probe=resolved_probe),
             "cylindrical complex: ", "cylindrical complex constitutive"),
        _Arm("complex folded off-diagonal",
             has_fold and has_complex and has_offdiag_rows,
             lambda: complex_folded_offdiag_update_e_coverage(
                 fields, pml, resolved_probe),
             lambda: plan_complex_folded_offdiag_update_e(
                 fields, pml, block, resolved_probe),
             "complex folded off-diagonal: ",
             "complex folded off-diagonal constitutive"),
        # The three-way intersection must precede neither parent semantically:
        # every predicate is evaluated and ambiguity fails closed. It is placed
        # first only so refusal reports read from the most specific product to
        # the two parent products it replaces.
        _Arm("folded off-diagonal dispersive",
             has_fold and has_offdiag_rows and has_poles,
             lambda: folded_offdiag_dispersive_constitutive_coverage(fields, pml),
             lambda: plan_folded_offdiag_dispersive_constitutive(
                 fields, pml, block),
             "folded off-diagonal dispersive: ",
             "folded off-diagonal dispersive constitutive"),
        # The second arm that counts LIVE ROW SLOTS rather than reading the flag,
        # and therefore the second entry in ROW_PRODUCT_ARMS. Adding it without
        # adding it there would have had the veto empty the one slot the correct
        # product had just won, with a message that was false about that product.
        _Arm("folded off-diagonal", has_fold and has_offdiag_rows,
             lambda: folded_offdiag_composition_coverage(fields, pml),
             lambda: plan_folded_offdiagonal_constitutive(fields, pml, block),
             "folded off-diagonal: ", "folded off-diagonal constitutive"),
        # The CERTIFIED dispersive update_E on a FOLDED run. Two inverted clauses
        # hold it apart from two incumbents at once: the `dispersive` arm above
        # refuses a fold (coverage._grid_reasons clause 5) and the `folded` arm
        # refuses a registered susceptibility (symmetry.py's E-side clause), so
        # this arm can only ever win a slot on which both of them refuse.
        _Arm("folded dispersive", has_fold and has_poles,
             lambda: folded_dispersive_constitutive_coverage(fields, pml),
             lambda: plan_folded_dispersive_constitutive(fields, pml, block),
             "folded dispersive: ", "folded dispersive constitutive"),
        _Arm("complex no-PML off-diagonal",
             has_complex and has_offdiag_rows and no_active_absorber,
             lambda: complex_no_pml_offdiag_update_e_coverage(
                 fields, pml, resolved_probe),
             lambda: plan_complex_no_pml_offdiag_update_e(
                 fields, pml, block, resolved_probe),
             "complex no-PML off-diagonal: ",
             "complex no-PML off-diagonal constitutive"),
        # Complex stored E with live polarization state is a distinct product:
        # the real no-PML sibling refuses complex storage, while the ordinary
        # complex and dispersive arms require an active PML layer. Conductivity
        # is deliberately not a gate because update_E does not read it.
        _Arm("complex no-PML stored E",
             has_complex and has_poles and no_active_absorber,
             lambda: complex_stored_e_coverage(
                 fields, pml, resolved_probe),
             lambda: plan_complex_stored_e(
                 fields, pml, block, resolved_probe),
             "complex no-PML stored E: ",
             "complex no-PML stored-E constitutive"),
        # Stored E without an absorber has neither the split-field update nor the
        # PML coefficient tail.  It precedes the null arm because the two are
        # alternatives on exactly this layer state, not because it has priority.
        _Arm("no-PML stored E", no_active_absorber,
             lambda: stored_e_constitutive_coverage(fields, pml),
             lambda: plan_stored_e_constitutive(fields, pml, block),
             "no-PML stored E: ", "no-PML stored-E constitutive"),
        # LAST, for the reason spelled out on the update_H copy of this arm.
        _Arm("no-PML null", no_active_absorber,
             lambda: null_constitutive_coverage(fields, pml, "E"),
             lambda: plan_null_constitutive(fields, pml, "E", block),
             "no-PML null: ", "no-PML null constitutive"),
    ), _update_e_ambiguity, plans, reasons, selected)

    # A specialized family owning this grid's sub-steps disables OPT-IN pair
    # fusion by name. This is load-bearing, not cosmetic: an ordinary pair's
    # kernel inlines the ORDINARY constitutive product, which is not the product
    # the family's own arm won, and installing one would overwrite the separately
    # built plans with a FusedPairPlan plus a NoopPlan — a composition no gate has
    # ever run. The nonlinear run is the case that was measured: `update_E` there
    # carries the Pade factor the pair's kernel has no term for, and the CONTROL
    # leg read the certified constitutive body at 1536 of 12288 words differing on
    # that side, so a fused D pair on a nonlinear run is exactly the composition
    # that diverges. The refusal is kept STRUCTURAL, ahead of `_pair_may_absorb`
    # rather than left to it, so that it holds whether or not a pair predicate
    # would admit — a family's curls becoming coverable must not be able to reach
    # the ordinary pair through the back door.
    #
    # THIS DISJUNCTION IS THE WHOLE-GRID HALF ONLY, as of 2026-09-13. It used to
    # carry two further terms, both about an off-diagonal chi1inv, and both have
    # moved to `off_diagonal_owns_update_e` below because that property is NOT a
    # whole-grid one: it owns exactly one seam. See the note there for what the
    # kernels read, which is what decides it.
    specialized_family_owns_the_grid = (
        has_complex or has_beta or has_bfast or has_nonlinearity
    )

    # THE OFF-DIAGONAL HALF OF THAT GUARD, SPLIT OUT AND NARROWED TO ONE SEAM.
    # Until 2026-09-13 these two terms sat in the disjunction above, so an
    # off-diagonal grid never reached the `elif fuse:` block at all: every pair
    # was refused one branch earlier, by name. That was conservatism rather than a
    # correctness bound — the comment it replaces said so itself, in the words
    # "this guard's job is to refuse an UNMEASURED fusion" — and the asymmetry it
    # was hiding is decidable from the kernels' own signatures rather than
    # argued:
    #
    #   * THE MAGNETIC PAIR NEVER RECEIVES chi1inv. :func:`plan_fused_pair`
    #     passes the inverse-epsilon volumes only when the pair's constitutive
    #     side is "E" (`if spec["constitutive"] == "E" else None`), and
    #     `FUSED_PAIRS["B"]` is constitutive "H"; `FusedPairPlan.__init__` stores
    #     `()` for that None; `FusedPairPlan.run`'s `self.pair == "B"` branch
    #     passes `self._inverse_epsilon` to no launch, while the D branch does;
    #     and `kernels.fused_curl_constitutive_B` carries no inverse-epsilon
    #     pointer in its signature at all. An off-diagonal chi1inv therefore
    #     cannot change one value the B seam computes. The narrowing is what the
    #     kernel reads, not a judgement about how much the coupling matters.
    #   * THE ELECTRIC PAIR DOES READ IT, AND READS IT AS THREE DIAGONAL
    #     VOLUMES. `kernels.fused_curl_constitutive_D` takes `ie0, ie1, ie2` and
    #     applies them pointwise, with no row-product term, and the dispersive
    #     D/E pair's kernel takes the same three and applies them the same way
    #     (`dispersive_fused_pair.py:72,215`). Fusing either on a grid whose
    #     array path forms the row product (stepping.py:992) would drop the
    #     coupling silently — which is exactly what
    #     :func:`_veto_dropped_offdiagonal_coupling` refuses one slot later. So
    #     both E-side pairs keep the refusal, BY NAME and in the same words,
    #     inside the `elif fuse:` block below.
    #
    # THE NARROWING IS NOT SELF-LICENSING, and it is not the same KIND of edit as
    # widening a release row: what it changes is which compositions the engine
    # INSTALLS on a real run, not only which ones a table admits. Its device
    # evidence is therefore owed rather than inherited, and it is the unfolded
    # off-diagonal route case the campaign that ships this edit drives: a
    # PASS-FUSED verdict with the substitution proved exact AND the electric pair
    # refused by name on the same row. Until that record exists the composition
    # is one no Triton gate has exercised. The analogous composition HAS been
    # measured on another backend — the Metal round of 2026-09-12 composed the
    # ordinary magnetic pair on `offdiag_2d`, 4 -> 3 launches per step,
    # byte-identical to the array path at every rung to 192 steps — but that
    # backend's composer holds no guard of this kind, so its measurement is
    # evidence about the ARITHMETIC and not a licence for this file.
    #
    # BOTH TRIGGERS ARE KEPT VERBATIM, and for the reason they were written. The
    # term is DELIBERATELY NARROWER than the arm's gate: `offdiag_rows_possible`
    # is true for every real Fields (it holds the accessor), so using it here
    # would silently disable fusion for every engine-route run and change the
    # control the fused gates were measured against. The gate's job is to
    # preserve an ambiguity; this term's job is to refuse an unmeasured fusion on
    # the seam that has one. So it fires on the installed FLAG, plus on
    # `update_E` having actually admitted a row-product arm — which is exactly
    # the flag-False-but-row-live case the gate exists to keep visible.
    off_diagonal_owns_update_e = (
        _reads_true(fields, "has_offdiagonal_epsilon")
        or any(arm in ROW_PRODUCT_ARMS for arm in e_admitted)
    )

    if fuse and has_fold:
        # THE ORDINARY PAIR STILL MAY NOT RUN HERE, and that half of the old
        # blanket refusal is unchanged: its kernel carries no mirror fill, so the
        # folded ghost-fill passes the driver runs after injection would be work
        # the launch never did. What changed 2026-08-27 is that the two products
        # which DO carry those passes are now routed through their own builders
        # instead of being refused alongside it. See :func:`_install_folded_fused_pairs`.
        _install_folded_fused_pairs(plans, reasons, selected, fields, pml, sources,
                                    block, num_warps, offered)
        reasons["fused_pair_dispersive_D"] = (
            "folded dispersive D/E fusion is not implemented",
        )
    elif fuse and has_cylindrical:
        for pair_name in FUSED_PAIRS:
            reasons[f"fused_pair_{pair_name}"] = (
                "cylindrical m=0 composition keeps the radial prefix and axis rules "
                "in separate curl plans; cross-sub-step fusion is not implemented",
            )
        reasons["fused_pair_dispersive_D"] = (
            "cylindrical dispersive D/E fusion is not implemented",
        )
    elif fuse and specialized_family_owns_the_grid:
        for pair_name in FUSED_PAIRS:
            reasons[f"fused_pair_{pair_name}"] = (
                "a specialized family owns this grid's sub-steps; cross-sub-step "
                "fusion for it is not implemented",
            )
        reasons["fused_pair_dispersive_D"] = (
            "specialized-family dispersive D/E fusion is not implemented",
        )
    elif fuse:
        # Every call in this block is guarded exactly as an arm's is: the pair
        # predicates read a source list this composer did not build (a
        # non-sequence `sources` raises inside `tuple(sources)`) and the pair
        # builders import Triton, which the merge-bar host does not have. An
        # opt-in optimisation that CRASHES the plan is strictly worse than one
        # that refuses it.
        ordinary_pair_verdicts = {
            name: _guarded_verdict(
                lambda name=name: fused_pair_coverage(fields, pml, name, sources),
                f"the {name} pair predicate")
            for name in FUSED_PAIRS
        }
        dispersive_pair_verdict = _guarded_verdict(
            lambda: dispersive_fused_pair_coverage(fields, pml, sources),
            "the dispersive D/E pair predicate")
        for pair_name, spec in FUSED_PAIRS.items():
            if off_diagonal_owns_update_e and spec["constitutive"] == "E":
                # THE HALF OF THE OLD BLANKET REFUSAL THAT SURVIVES THE SPLIT,
                # keyed on the constitutive side rather than on the pair's letter
                # because the side is the reason: this is the seam whose kernel
                # takes the inverse-epsilon pointers and applies them as three
                # diagonal volumes. The magnetic pair falls through and is now
                # composed here for the first time.
                #
                # THE REASON TEXT IS UNCHANGED ON PURPOSE. It is the string the
                # frozen gate artifacts under parity/meep_gpu/results recorded,
                # and the refusal it names is the same refusal arriving from a
                # different branch — `_pair_may_absorb` would also decline this
                # slot once an off-diagonal arm won `update_E`, but it would say
                # so in terms of which arm won rather than in terms of the
                # coupling, and a reader comparing artifacts across this edit
                # needs the sentence to hold still.
                reasons[f"fused_pair_{pair_name}"] = (
                    "a specialized family owns this grid's sub-steps; cross-sub-step "
                    "fusion for it is not implemented",
                )
                continue
            pair_verdict = ordinary_pair_verdicts[pair_name]
            if (pair_name == "D" and pair_verdict.covered
                    and dispersive_pair_verdict.covered):
                reasons["fused_pair_dispersive_D"] = (
                    "both ordinary and dispersive D/E pair predicates admitted the "
                    "same configuration; retaining the separate sub-step plans",
                )
                continue
            curl_name = spec["curl"]
            update_name = "update_" + spec["constitutive"]
            if not pair_verdict.covered:
                # A refused fusion is not a refused step: both sub-steps keep the
                # separate plans built above, and the reason is reported alongside.
                reasons[f"fused_pair_{pair_name}"] = pair_verdict.reasons
                continue
            refusal = _pair_may_absorb(selected, curl_name, update_name,
                                       FUSED_PAIR_ARMS[pair_name])
            if refusal is not None:
                reasons[f"fused_pair_{pair_name}"] = (refusal,)
                continue
            not_offered = _label_not_offered(f"fused pair {pair_name}", offered)
            if not_offered is not None:
                reasons[f"fused_pair_{pair_name}"] = (not_offered,)
                continue
            pair, builder_refusal = _guarded_plan(
                lambda: plan_fused_pair(fields, pml, pair_name, sources, block,
                                        num_warps),
                f"the {pair_name} pair builder")
            if builder_refusal is not None:
                reasons[f"fused_pair_{pair_name}"] = (builder_refusal,)
            elif pair is not None:
                _install_fused_pair(plans, selected, fields, pml, sources, pair_name,
                                    curl_name, update_name,
                                    f"fused pair {pair_name}", pair)
            else:
                reasons[f"fused_pair_{pair_name}"] = ("the fused pair was refused",)

        if off_diagonal_owns_update_e:
            # THE DISPERSIVE D/E PAIR COMPOSES THE SAME SEAM, so the narrowing
            # above does not reach it: its kernel takes the same three
            # inverse-epsilon volumes and applies them pointwise. Refused here,
            # ahead of the builder, rather than left to the `_pair_may_absorb`
            # check inside the branch below — the artifact then carries this
            # reason by name instead of one about which arm won the slot, and the
            # pair's own coverage predicate is never asked to stand in for a
            # composition rule.
            reasons["fused_pair_dispersive_D"] = (
                "specialized-family dispersive D/E fusion is not implemented",
            )
        elif (dispersive_pair_verdict.covered
                and not ordinary_pair_verdicts["D"].covered):
            refusal = _pair_may_absorb(selected, "step_D", "update_E",
                                       FUSED_PAIR_ARMS["dispersive_D"])
            if refusal is None:
                refusal = _label_not_offered("dispersive fused pair", offered)
            pair, builder_refusal = (None, None)
            if refusal is None:
                pair, builder_refusal = _guarded_plan(
                    lambda: plan_dispersive_fused_pair(fields, pml, sources,
                                                       block, num_warps),
                    "the dispersive D/E pair builder")
            if refusal is not None:
                reasons["fused_pair_dispersive_D"] = (refusal,)
            elif builder_refusal is not None:
                reasons["fused_pair_dispersive_D"] = (builder_refusal,)
            elif pair is not None:
                _install_fused_pair(plans, selected, fields, pml, sources, "D",
                                    "step_D", "update_E", "dispersive fused pair", pair)
            else:
                reasons["fused_pair_dispersive_D"] = (
                    "dispersive D/E pair coverage admitted the configuration but "
                    "its builder refused",
                )
        elif not dispersive_pair_verdict.covered:
            reasons["fused_pair_dispersive_D"] = dispersive_pair_verdict.reasons

    _veto_dropped_offdiagonal_coupling(fields, plans, reasons, selected)

    if fuse:
        # THE CERTIFIED-PRODUCT SEAM LOOP, and it runs AFTER the veto rather than
        # inside the four-branch chain above. Two reasons, both about correctness
        # rather than tidiness:
        #
        #   * THE VETO WOULD OTHERWISE RIP OUT A CORRECT PRODUCT. It early-returns
        #     when ``selected["update_E"]`` is one of ``ROW_PRODUCT_ARMS``; a fused
        #     label is not one, so an off-diagonal product installed BEFORE it
        #     would be vetoed for having "no row-product term" — which is false of
        #     exactly the two products that implement the term.
        #   * A VETOED SLOT IS THEN UNSELECTED, which is the fail-closed half:
        #     the veto pops ``selected["update_E"]``, so :func:`_pair_may_absorb`
        #     refuses every candidate on that seam by name. There is no path on
        #     which this loop installs a product the veto would have refused.
        #
        # It runs on EVERY grid, not only the ones the four-branch chain leaves
        # unfused: these products exist for the folded, cylindrical and
        # specialized-family grids that chain refuses by name, and those refusals
        # are unchanged — they are about the ORDINARY and DISPERSIVE pairs, whose
        # kernels carry none of what those grids need.
        _install_certified_fused_products(plans, reasons, selected, fields, pml,
                                          sources, block, num_warps, offered)

    try:
        states = tuple(getattr(fields, "polarizations", ()) or ())
    except Exception as exc:  # noqa: BLE001 - an unreadable list is not an empty one
        states = ()
        reasons["update_P"] = (f"the polarization list could not be read ({exc!r})",)
    polarization_plans: List[Any] = []
    if states:
        # WHICH ADE FAMILY OWNS ``update_P`` — decided ONCE for the whole sub-step,
        # not per state, because the families are held apart by facts about
        # the RUN (is the split-field absorber active?) rather than about a pole.
        # The driver advances every polarization in one pass and it is that pass
        # this slot replaces, so a plan that took state 0 from one family and
        # state 1 from the other would describe a pass nothing runs.
        #
        # THE AMBIGUITY IS CHECKED BEFORE ANYTHING IS BUILT, which is the whole
        # reason the predicates are consulted here at all: this block is
        # BUILDER-FIRST (each builder asks its own predicate and answers None),
        # and a builder-first fallback would silently prefer the incumbent if both
        # families ever admitted — clause (b)'s failure mode, spelled "pick by
        # order". So every verdict is taken first and any multiple admission
        # leaves the slot UNSELECTED, exactly as :func:`_select_slot` would.
        ade_label = "fused ADE state" if fuse_ade else "ADE update_P"
        ade_builder = (
            (lambda state: plan_fused_ade_state(fields, state, block))
            if fuse_ade else
            (lambda state: plan_ade_update_p(fields, state, block)))
        ade_predicates: Tuple[Tuple[str, Any], ...] = ()
        ambiguity: Optional[Tuple[str, ...]] = None
        if not fuse_ade:
            candidates: List[Tuple[str, str, Any, Any]] = [
                ("ADE update_P", "",
                 lambda state, component:
                     ade_update_p_coverage(fields, state, component),
                 lambda state: plan_ade_update_p(fields, state, block)),
            ]
            if no_active_absorber:
                # The gate is the null family's, and it is NECESSARY here for the
                # same reason: ``no_pml_ade._inert_layer_reasons`` refuses an
                # active layer by name, so a closed gate can only remove refusals.
                candidates.append(
                    ("no-PML ADE update_P", "no-PML ADE: ",
                     lambda state, component:
                         no_pml_ade_update_p_coverage(
                             fields, pml, state, component),
                     lambda state:
                         plan_no_pml_ade_update_p(fields, pml, state, block)))
                if has_complex:
                    candidates.append(
                        ("complex ADE update_P", "complex ADE: ",
                         lambda state, component:
                             complex_ade_update_p_coverage(
                                 fields, pml, state, component, resolved_probe),
                         lambda state:
                             plan_complex_ade_update_p(
                                 fields, pml, state, block, resolved_probe)))
            ade_predicates = tuple((prefix, ask)
                                   for _label, prefix, ask, _builder in candidates)
            admitted = [candidate for candidate in candidates
                        if _ade_family_admits(states, candidate[2])]
            if len(admitted) > 1:
                quantifier = "both" if len(admitted) == 2 else str(len(admitted))
                labels = ", ".join(candidate[0] for candidate in admitted)
                ambiguity = (
                    f"{quantifier} update_P predicates ({labels}) admitted the "
                    "same configuration; refusing an ambiguous numerical product",
                )
            elif admitted:
                ade_label, _prefix, _ask, ade_builder = admitted[0]
    if states and ambiguity is not None:
        # Nothing is built and nothing is explained per state: two products
        # admitted, so the slot is the array path and the reason names both.
        reasons["update_P"] = ambiguity
    elif states:
        refused: List[str] = []
        for index, state in enumerate(states):
            if refused:
                # Already on the array path: explain this state, do not plan it.
                plan, builder_refusal = None, None
            else:
                plan, builder_refusal = _guarded_plan(
                    (lambda state=state: ade_builder(state)),
                    f"the polarization {index} builder")
            if builder_refusal is None and plan is not None:
                polarization_plans.append(plan)
                continue
            # EVERY refusing state is explained, not just the first. The sub-step
            # falls to the array path for all of them — the driver advances the
            # whole list in one pass — so a record naming only state 0 documents
            # a fraction of what actually happened and sends a reader looking in
            # the wrong place.
            if builder_refusal is not None:
                refused.append(f"polarization {index}: {builder_refusal}")
                continue
            if fuse_ade:
                verdict = _guarded_verdict(
                    lambda state=state: fused_ade_state_coverage(fields, state),
                    f"the polarization {index} predicate")
                refused.extend(f"polarization {index}: {reason}"
                               for reason in verdict.reasons)
                continue
            components = _driven_components(state)
            for component in components or ("<none driven>",):
                # EVERY consulted family explains itself. Reporting only the
                # incumbent on an absorber-free run would name the clause that is
                # least relevant there and hide the one that actually decided it.
                for prefix, ask in ade_predicates:
                    verdict = _guarded_verdict(
                        lambda component=component, state=state, ask=ask:
                            ask(state, component),
                        f"the polarization {index} {component} predicate")
                    refused.extend(f"polarization {index} {component}: {prefix}{r}"
                                   for r in verdict.reasons)
        if refused or len(polarization_plans) != len(states):
            # One refused state keeps the whole update_P sub-step on the array path:
            # the driver advances every polarization in one pass, and it is that pass
            # this plan replaces.
            polarization_plans = []
            reasons["update_P"] = tuple(refused) or ("a polarization was refused",)
        else:
            plans["update_P"] = polarization_plans
            selected["update_P"] = ade_label

    return TritonStepPlan(plans, polarization_plans, reasons, selected)


def explain(fields: Any, pml: Any) -> Coverage:
    """The curl coverage verdict with its reasons — for reports, tests and refusals."""
    return pml_curl_coverage(fields, pml)


def _flat(array: Any) -> Any:
    """A contiguous 1-D VIEW of a broadcast-shaped coefficient column.

    ``reshape(-1)`` on an ``(n, 1, 1)`` / ``(1, n, 1)`` / ``(1, 1, n)`` C-contiguous
    array is a view, so this costs nothing and — critically — is done once at plan
    time rather than six times a timestep.
    """
    flat = array.reshape(-1)
    if not flat.flags.c_contiguous:
        raise ValueError("PML coefficient column did not flatten to a contiguous view")
    if str(flat.dtype) != "float32":
        raise ValueError(f"PML coefficient column dtype {flat.dtype} is not float32")
    return flat
