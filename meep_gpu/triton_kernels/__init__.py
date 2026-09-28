"""Triton implementation of the real-field PML curl sub-step (Phase 1, track 2).

The same kernel the hand-written CUDA track implements, written in Triton so the
two can be compared on one gate and one benchmark: bit-identity against
``stepping.py``, throughput in Mcell-steps/s, and lines of code.

    from meep_gpu.triton_kernels import plan_pml_curl
    plan = plan_pml_curl(fields, pml, "step_B")   # None when out of coverage
    if plan is not None:
        plan.run()

NOTHING HERE IS ON THE STEP PATH. ``meep_gpu.fastpath.plan_fast_path`` returns
None on every branch and this package does not touch it; wiring dispatch waits on
the measurement.

IMPORTING THIS PACKAGE MUST NOT NEED TRITON. Triton is an optional dependency of
an optional package, and a missing optional dependency must not break the engine —
so ``coverage``/``availability`` are importable anywhere, and ``triton`` itself is
imported only when a plan is built or launched. ``test_triton_kernels`` pins that
by importing this module with ``triton`` blocked from ``sys.modules``.
"""

from __future__ import annotations

from typing import Any, Optional

#: The sentinel that means "take ``launch.FUSED_DEFAULT_NUM_WARPS``".
#:
#: These wrappers used to spell their ``num_warps`` default as ``None``, which is
#: not the same value: ``None`` means "let Triton choose" (4 warps for a 1-D
#: BLOCK=256 program) while ``launch``'s own default is the MEASURED 1, and
#: launch.py records 4 warps as a 16% throughput loss on the pair. The same call
#: through this package and through ``.launch`` therefore built two different
#: launch configurations from identical inputs — and every gate and probe imports
#: these names from the package.
#:
#: A sentinel rather than a duplicated ``1``: the measured value has ONE home,
#: and passing ``num_warps=None`` explicitly must still mean Triton's own default.
_INHERIT_LAUNCH_DEFAULT: Any = object()


def _num_warps(value: Any) -> Optional[int]:
    """Resolve the sentinel against ``launch``'s measured default."""
    if value is not _INHERIT_LAUNCH_DEFAULT:
        return value
    from .launch import FUSED_DEFAULT_NUM_WARPS  # noqa: PLC0415
    return FUSED_DEFAULT_NUM_WARPS

from .coverage import (  # noqa: E402 - wrappers above define the shared sentinel first
    CONSTITUTIVE_SIDES,
    COVERED_BOUNDARIES,
    COVERED_SUSCEPTIBILITY_KINDS,
    FUSED_PAIRS,
    Coverage,
    ade_update_p_coverage,
    constitutive_coverage,
    dispersive_constitutive_coverage,
    fused_pair_coverage,
    pml_curl_coverage,
    zero_metal_axes,
)

__all__ = [
    "CONSTITUTIVE_SIDES",
    "COVERED_BOUNDARIES",
    "COVERED_SUSCEPTIBILITY_KINDS",
    "FUSED_PAIRS",
    "Coverage",
    "pml_curl_coverage",
    "folded_pml_curl_coverage",
    "conductive_pml_curl_coverage",
    "plain_curl_coverage",
    "mirror_ghost_fill_coverage",
    "constitutive_coverage",
    "folded_constitutive_coverage",
    "dispersive_constitutive_coverage",
    "dispersive_fused_pair_coverage",
    "fused_ade_state_coverage",
    "ade_update_p_coverage",
    "fused_pair_coverage",
    "zero_metal_axes",
    "triton_available",
    "plan_pml_curl",
    "plan_folded_pml_curl",
    "plan_conductive_pml_curl",
    "plan_plain_curl",
    "plan_mirror_ghost_fill",
    "plan_constitutive",
    "plan_folded_constitutive",
    "cylindrical_curl_coverage",
    "plan_cylindrical_curl",
    "cylindrical_constitutive_coverage",
    "plan_cylindrical_constitutive",
    "plan_dispersive_constitutive",
    "plan_dispersive_fused_pair",
    "plan_fused_ade_state",
    "plan_ade_update_p",
    "plan_fused_pair",
    # The five certified specialized families the planner now consults. Every
    # one is a lazy wrapper: naming them here must not make importing this
    # package import their modules.
    "load_expansion_probe",
    "complex_pml_curl_coverage",
    "plan_complex_pml_curl",
    "complex_constitutive_coverage",
    "plan_complex_constitutive",
    "beta_pml_curl_coverage",
    "plan_beta_pml_curl",
    "beta_bloch_pml_curl_coverage",
    "plan_beta_bloch_pml_curl",
    "beta_run_constitutive_coverage",
    "plan_beta_run_constitutive",
    "beta_run_complex_constitutive_coverage",
    "plan_beta_run_complex_constitutive",
    "bfast_pml_curl_coverage",
    "plan_bfast_pml_curl",
    "bfast_run_constitutive_coverage",
    "plan_bfast_run_constitutive",
    "nonlinear_constitutive_coverage",
    "plan_nonlinear_constitutive",
    "offdiag_constitutive_coverage",
    "plan_offdiagonal_constitutive",
    # The four further certified families, wired in the round after those five.
    # Same lazy shape and the same rule: naming them here must not make importing
    # this package import their modules.
    "folded_complex_composition_curl_coverage",
    "plan_folded_complex_pml_curl",
    "folded_beta_pml_curl_coverage",
    "plan_folded_beta_pml_curl",
    "folded_beta_bloch_pml_curl_coverage",
    "plan_folded_beta_bloch_pml_curl",
    "folded_mirror_ghost_fill_complex_coverage",
    "plan_folded_mirror_ghost_fill_complex",
    "folded_complex_constitutive_coverage",
    "plan_folded_complex_constitutive",
    "folded_beta_run_constitutive_coverage",
    "plan_folded_beta_run_constitutive",
    "folded_offdiag_composition_coverage",
    "plan_folded_offdiagonal_constitutive",
    "cylindrical_complex_curl_coverage",
    "plan_cylindrical_complex_curl",
    "cylindrical_complex_constitutive_coverage",
    "plan_cylindrical_complex_constitutive",
    "null_constitutive_coverage",
    "plan_null_constitutive",
    # The residual-group ADMISSIONS, wired in the round after those four. Not one
    # new kernel between them: each restates a shipped clause set with the one
    # over-broad clause inverted, so a CERTIFIED body may step a configuration it
    # was refused for. Same lazy shape and the same rule as everything above.
    "nonlinear_run_pml_curl_coverage",
    "plan_nonlinear_run_pml_curl",
    "nonlinear_run_constitutive_coverage",
    "plan_nonlinear_run_constitutive",
    "folded_dispersive_constitutive_coverage",
    "plan_folded_dispersive_constitutive",
    "folded_offdiag_dispersive_constitutive_coverage",
    "plan_folded_offdiag_dispersive_constitutive",
    "folded_complex_offdiag_pml_curl_coverage",
    "plan_folded_complex_offdiag_pml_curl",
    "folded_complex_offdiag_constitutive_coverage",
    "plan_folded_complex_offdiag_constitutive",
    "no_pml_ade_update_p_coverage",
    "plan_no_pml_ade_update_p",
    "complex_ade_update_p_coverage",
    "plan_complex_ade_update_p",
    # No-absorber closure families.  These stay lazy for the same reason as the
    # older specialized entries: importing the optional package must not import
    # a Triton body merely because its public name exists.
    "complex_no_pml_curl_coverage",
    "plan_complex_no_pml_curl",
    "complex_conductive_no_pml_curl_coverage",
    "plan_complex_conductive_no_pml_curl",
    "conductive_plain_curl_coverage",
    "plan_conductive_plain_curl",
    "stored_e_constitutive_coverage",
    "plan_stored_e_constitutive",
    "complex_stored_e_coverage",
    "plan_complex_stored_e",
    "complex_no_pml_offdiag_update_e_coverage",
    "plan_complex_no_pml_offdiag_update_e",
    "complex_folded_offdiag_update_e_coverage",
    "plan_complex_folded_offdiag_update_e",
    "plan_step",
    "explain",
    "require_triton",
]


def triton_available() -> bool:
    """Is the optional Triton dependency importable? Never raises."""
    try:
        import triton  # noqa: F401, PLC0415
    except Exception:  # noqa: BLE001 - a broken install is as unavailable as a missing one
        return False
    return True


def plan_pml_curl(fields: Any, pml: Any, sub_step: str,
                  block: Optional[int] = None) -> Any:
    """Plan one curl sub-step, or None when the configuration is out of coverage."""
    from .launch import plan_pml_curl as _plan  # noqa: PLC0415
    return _plan(fields, pml, sub_step, block)


def folded_pml_curl_coverage(fields: Any, pml: Any) -> Coverage:
    """Fold-capable PML curl coverage, including its unfolded equivalence case."""
    from .symmetry import folded_pml_curl_coverage as _coverage  # noqa: PLC0415
    return _coverage(fields, pml)


def plan_folded_pml_curl(fields: Any, pml: Any, sub_step: str,
                         block: Optional[int] = None) -> Any:
    """Plan one folded PML curl sub-step, or None when refused."""
    from .symmetry import plan_folded_pml_curl as _plan  # noqa: PLC0415
    return _plan(fields, pml, sub_step, block)


def conductive_pml_curl_coverage(fields: Any, pml: Any, sub_step: str) -> Coverage:
    """Conductive-PML curl coverage, importable without loading Triton."""
    from .conductivity import conductive_pml_curl_coverage as _coverage  # noqa: PLC0415
    return _coverage(fields, pml, sub_step)


def plan_conductive_pml_curl(fields: Any, pml: Any, sub_step: str,
                             block: Optional[int] = None) -> Any:
    """Plan one conductive PML curl sub-step, or None when refused."""
    from .conductivity import plan_conductive_pml_curl as _plan  # noqa: PLC0415
    return _plan(fields, pml, sub_step, block)


def plain_curl_coverage(fields: Any, pml: Any, sub_step: str) -> Coverage:
    """No-absorber curl coverage, available without importing Triton eagerly."""
    from .no_pml import plain_curl_coverage as _coverage  # noqa: PLC0415
    return _coverage(fields, pml, sub_step)


def plan_plain_curl(fields: Any, pml: Any, sub_step: str,
                    block: Optional[int] = None) -> Any:
    """Plan one no-absorber curl sub-step, or None when refused."""
    from .no_pml import plan_plain_curl as _plan  # noqa: PLC0415
    return _plan(fields, pml, sub_step, block)


def mirror_ghost_fill_coverage(fields: Any, family: str) -> Coverage:
    """Coverage for one folded B/D family's combined near/far ghost fills."""
    from .symmetry import mirror_ghost_fill_coverage as _coverage  # noqa: PLC0415
    return _coverage(fields, family)


def plan_mirror_ghost_fill(fields: Any, family: str,
                           block: Optional[int] = None) -> Any:
    """Plan one folded B/D family's combined near/far ghost fills."""
    from .symmetry import plan_mirror_ghost_fill as _plan  # noqa: PLC0415
    return _plan(fields, family, block)


def plan_constitutive(fields: Any, pml: Any, side: str,
                      block: Optional[int] = None) -> Any:
    """Plan ``update_H`` (side='H') or ``update_E`` ('E'), or None when refused."""
    from .launch import plan_constitutive as _plan  # noqa: PLC0415
    return _plan(fields, pml, side, block)


def folded_constitutive_coverage(fields: Any, pml: Any, side: str) -> Coverage:
    """Coverage for ordinary constitutive arithmetic on a folded stored extent."""
    from .symmetry import folded_constitutive_coverage as _coverage  # noqa: PLC0415
    return _coverage(fields, pml, side)


def plan_folded_constitutive(fields: Any, pml: Any, side: str,
                             block: Optional[int] = None) -> Any:
    """Plan update_H/update_E on a folded stored extent, or None when refused."""
    from .symmetry import plan_folded_constitutive as _plan  # noqa: PLC0415
    return _plan(fields, pml, side, block)


def cylindrical_curl_coverage(fields: Any, pml: Any) -> Coverage:
    """Coverage for the strict real m=0 cylindrical PML curl slice."""
    from .cylindrical_triton import cylindrical_curl_coverage as _coverage  # noqa: PLC0415
    return _coverage(fields, pml)


def plan_cylindrical_curl(fields: Any, pml: Any, sub_step: str,
                          block: Optional[int] = None) -> Any:
    """Plan one strict real m=0 cylindrical curl, or None when refused."""
    from .cylindrical_triton import plan_cylindrical_curl as _plan  # noqa: PLC0415
    return _plan(fields, pml, sub_step, block)


def cylindrical_constitutive_coverage(fields: Any, pml: Any,
                                       side: str) -> Coverage:
    """Coverage for the measured elementwise Dcyl constitutive sub-step."""
    from .cylindrical_triton import cylindrical_constitutive_coverage as _coverage  # noqa: PLC0415
    return _coverage(fields, pml, side)


def plan_cylindrical_constitutive(fields: Any, pml: Any, side: str,
                                  block: Optional[int] = None) -> Any:
    """Plan the existing constitutive kernel for the admitted Dcyl slice."""
    from .cylindrical_triton import plan_cylindrical_constitutive as _plan  # noqa: PLC0415
    return _plan(fields, pml, side, block)


def plan_dispersive_constitutive(fields: Any, pml: Any,
                                 block: Optional[int] = None) -> Any:
    """Plan pole-aware dispersive ``update_E``, or None when refused."""
    from .dispersive_update_e import plan_dispersive_constitutive as _plan  # noqa: PLC0415
    return _plan(fields, pml, block)


def dispersive_fused_pair_coverage(fields: Any, pml: Any,
                                   sources: Any = None) -> Coverage:
    """Coverage for the fused D curl plus pole-aware electric update."""
    from .dispersive_fused_pair import dispersive_fused_pair_coverage as _coverage  # noqa: PLC0415
    return _coverage(fields, pml, sources)


def plan_dispersive_fused_pair(fields: Any, pml: Any, sources: Any = None,
                               block: Optional[int] = None,
                               num_warps: Any = _INHERIT_LAUNCH_DEFAULT) -> Any:
    """Plan the fused dispersive D/E pair, or ``None`` when refused."""
    from .dispersive_fused_pair import plan_dispersive_fused_pair as _plan  # noqa: PLC0415
    return _plan(fields, pml, sources, block, _num_warps(num_warps))


def fused_ade_state_coverage(fields: Any, state: Any) -> Coverage:
    """Coverage for one launch spanning a susceptibility's driven components."""
    from .fused_ade_state import fused_ade_state_coverage as _coverage  # noqa: PLC0415
    return _coverage(fields, state)


def plan_fused_ade_state(fields: Any, state: Any,
                         block: Optional[int] = None) -> Any:
    """Plan one all-component susceptibility update, or ``None`` when refused."""
    from .fused_ade_state import plan_fused_ade_state as _plan  # noqa: PLC0415
    return _plan(fields, state, block)


def plan_ade_update_p(fields: Any, state: Any, block: Optional[int] = None) -> Any:
    """Plan one susceptibility's ADE update, or None when any driven component is refused."""
    from .launch import plan_ade_update_p as _plan  # noqa: PLC0415
    return _plan(fields, state, block)


def plan_fused_pair(fields: Any, pml: Any, pair: str = "B", sources: Any = None,
                    block: Optional[int] = None,
                    num_warps: Any = _INHERIT_LAUNCH_DEFAULT) -> Any:
    """Plan the fused ``step_B`` + ``zero_metal_B`` + ``update_H`` launch, or None.

    ``sources`` is the driver's source list; ``None`` is a refusal, not an empty
    set — see :func:`coverage.fused_pair_coverage`.
    """
    from .launch import plan_fused_pair as _plan  # noqa: PLC0415
    return _plan(fields, pml, pair, sources, block, _num_warps(num_warps))


# ---------------------------------------------------------------------------
# The five certified specialized families — lazy public wrappers
# ---------------------------------------------------------------------------
#
# Each body imports inside itself, exactly like every wrapper above: the five
# family modules must stay out of ``sys.modules`` after a bare package import,
# and each family's own test pins that on a machine with no Triton.


def load_expansion_probe(path: Optional[str] = None) -> Any:
    """Read the complex-multiply expansion probe artifact, or None when absent.

    Three-valued: a record, ``None`` when nothing is offered, or a
    ``RefusedExpansionProbe`` when this dispatch has already refused the artifact
    the variable points at. This is the outermost link of the re-export chain
    ``fastpath`` reads through, and flattening the third value into ``None`` here
    would hand a refused artifact straight back to the caller that refused it.
    """
    from .launch import load_expansion_probe as _reader  # noqa: PLC0415
    return _reader(path)


def complex_pml_curl_coverage(fields: Any, pml: Any, sub_step: str,
                              probe: Any = None) -> Coverage:
    """Coverage for one complex/Bloch split-field PML curl sub-step."""
    from .launch import complex_pml_curl_coverage as _coverage  # noqa: PLC0415
    return _coverage(fields, pml, sub_step, probe)


def plan_complex_pml_curl(fields: Any, pml: Any, sub_step: str,
                          block: Optional[int] = None, probe: Any = None) -> Any:
    """Plan one complex/Bloch split-field curl, or None when refused."""
    from .launch import plan_complex_pml_curl as _plan  # noqa: PLC0415
    return _plan(fields, pml, sub_step, block, probe)


def complex_constitutive_coverage(fields: Any, pml: Any, side: str,
                                  probe: Any = None) -> Coverage:
    """Coverage for the complex/Bloch constitutive sub-step on one side."""
    from .launch import complex_constitutive_coverage as _coverage  # noqa: PLC0415
    return _coverage(fields, pml, side, probe)


def plan_complex_constitutive(fields: Any, pml: Any, side: str,
                              block: Optional[int] = None,
                              probe: Any = None) -> Any:
    """Plan the complex/Bloch constitutive sub-step, or None when refused."""
    from .launch import plan_complex_constitutive as _plan  # noqa: PLC0415
    return _plan(fields, pml, side, block, probe)


def beta_pml_curl_coverage(fields: Any, pml: Any, sub_step: str) -> Coverage:
    """Coverage for one REAL special_kz curl sub-step."""
    from .launch import beta_pml_curl_coverage as _coverage  # noqa: PLC0415
    return _coverage(fields, pml, sub_step)


def plan_beta_pml_curl(fields: Any, pml: Any, sub_step: str,
                       block: Optional[int] = None) -> Any:
    """Plan one REAL special_kz curl, or None when refused."""
    from .launch import plan_beta_pml_curl as _plan  # noqa: PLC0415
    return _plan(fields, pml, sub_step, block)


def beta_bloch_pml_curl_coverage(fields: Any, pml: Any, sub_step: str,
                                 probe: Any = None) -> Coverage:
    """Coverage for one COMPLEX special_kz curl sub-step."""
    from .launch import beta_bloch_pml_curl_coverage as _coverage  # noqa: PLC0415
    return _coverage(fields, pml, sub_step, probe)


def plan_beta_bloch_pml_curl(fields: Any, pml: Any, sub_step: str,
                             block: Optional[int] = None,
                             probe: Any = None) -> Any:
    """Plan one COMPLEX special_kz curl, or None when refused."""
    from .launch import plan_beta_bloch_pml_curl as _plan  # noqa: PLC0415
    return _plan(fields, pml, sub_step, block, probe)


def beta_run_constitutive_coverage(fields: Any, pml: Any, side: str) -> Coverage:
    """Coverage for the certified REAL constitutive kernel on a beta run."""
    from .launch import beta_run_constitutive_coverage as _coverage  # noqa: PLC0415
    return _coverage(fields, pml, side)


def plan_beta_run_constitutive(fields: Any, pml: Any, side: str,
                               block: Optional[int] = None) -> Any:
    """Plan the certified REAL constitutive sub-step for a beta run."""
    from .launch import plan_beta_run_constitutive as _plan  # noqa: PLC0415
    return _plan(fields, pml, side, block)


def beta_run_complex_constitutive_coverage(fields: Any, pml: Any, side: str,
                                           probe: Any = None) -> Coverage:
    """Coverage for the certified COMPLEX constitutive kernel on a beta run."""
    from .launch import beta_run_complex_constitutive_coverage as _coverage  # noqa: PLC0415
    return _coverage(fields, pml, side, probe)


def plan_beta_run_complex_constitutive(fields: Any, pml: Any, side: str,
                                       block: Optional[int] = None,
                                       probe: Any = None) -> Any:
    """Plan the certified COMPLEX constitutive sub-step for a beta run."""
    from .launch import plan_beta_run_complex_constitutive as _plan  # noqa: PLC0415
    return _plan(fields, pml, side, block, probe)


def bfast_pml_curl_coverage(fields: Any, pml: Any, sub_step: str) -> Coverage:
    """Coverage for one BFAST curl sub-step."""
    from .launch import bfast_pml_curl_coverage as _coverage  # noqa: PLC0415
    return _coverage(fields, pml, sub_step)


def plan_bfast_pml_curl(fields: Any, pml: Any, sub_step: str,
                        block: Optional[int] = None) -> Any:
    """Plan one BFAST curl, or None when refused."""
    from .launch import plan_bfast_pml_curl as _plan  # noqa: PLC0415
    return _plan(fields, pml, sub_step, block)


def bfast_run_constitutive_coverage(fields: Any, pml: Any, side: str) -> Coverage:
    """Coverage for the certified REAL constitutive kernel on a BFAST run."""
    from .launch import bfast_run_constitutive_coverage as _coverage  # noqa: PLC0415
    return _coverage(fields, pml, side)


def plan_bfast_run_constitutive(fields: Any, pml: Any, side: str,
                                block: Optional[int] = None) -> Any:
    """Plan the certified REAL constitutive sub-step for a BFAST run."""
    from .launch import plan_bfast_run_constitutive as _plan  # noqa: PLC0415
    return _plan(fields, pml, side, block)


def nonlinear_constitutive_coverage(fields: Any, pml: Any) -> Coverage:
    """Coverage for the chi2/chi3 Pade ``update_E`` product."""
    from .launch import nonlinear_constitutive_coverage as _coverage  # noqa: PLC0415
    return _coverage(fields, pml)


def plan_nonlinear_constitutive(fields: Any, pml: Any,
                                block: Optional[int] = None) -> Any:
    """Plan the chi2/chi3 Pade ``update_E``, or None when refused."""
    from .launch import plan_nonlinear_constitutive as _plan  # noqa: PLC0415
    return _plan(fields, pml, block)


def offdiag_constitutive_coverage(fields: Any, pml: Any) -> Coverage:
    """Coverage for the off-diagonal chi1inv row-product ``update_E``."""
    from .launch import offdiag_constitutive_coverage as _coverage  # noqa: PLC0415
    return _coverage(fields, pml)


def plan_offdiagonal_constitutive(fields: Any, pml: Any,
                                  block: Optional[int] = None) -> Any:
    """Plan the off-diagonal row-product ``update_E``, or None when refused."""
    from .launch import plan_offdiagonal_constitutive as _plan  # noqa: PLC0415
    return _plan(fields, pml, block)


# ---------------------------------------------------------------------------
# The four further certified specialized families — lazy public wrappers
# ---------------------------------------------------------------------------
#
# EVERY ONE OF THESE GOES THROUGH ``.launch``, not through the family module.
# That is deliberate and it is not merely stylistic: ``folded_complex`` and
# ``cylindrical_complex`` import FROM ``.launch`` at module scope, so the package
# has exactly one edge into each family and it is the planner's forwarder. A
# wrapper that reached into the family directly would give the package a second,
# unlazy one.
#
# ``null_constitutive_coverage``/``plan_null_constitutive`` are the odd pair: the
# product they build is a NULL plan for a sub-step the array path skips entirely,
# so ``plan_null_constitutive`` is the one builder in this file that needs
# neither Triton nor a device.


def folded_complex_composition_curl_coverage(fields: Any, pml: Any, sub_step: str,
                                             probe: Any = None) -> Coverage:
    """Routing coverage for one folded complex/Bloch PML curl sub-step (K1)."""
    from .launch import folded_complex_composition_curl_coverage as _coverage  # noqa: PLC0415
    return _coverage(fields, pml, sub_step, probe)


def plan_folded_complex_pml_curl(fields: Any, pml: Any, sub_step: str,
                                 block: Optional[int] = None,
                                 probe: Any = None) -> Any:
    """Plan one folded complex/Bloch PML curl (K1), or None when refused."""
    from .launch import plan_folded_complex_pml_curl as _plan  # noqa: PLC0415
    return _plan(fields, pml, sub_step, block, probe)


def folded_beta_pml_curl_coverage(fields: Any, pml: Any, sub_step: str) -> Coverage:
    """Coverage for one folded REAL special_kz curl sub-step (K3a)."""
    from .launch import folded_beta_pml_curl_coverage as _coverage  # noqa: PLC0415
    return _coverage(fields, pml, sub_step)


def plan_folded_beta_pml_curl(fields: Any, pml: Any, sub_step: str,
                              block: Optional[int] = None) -> Any:
    """Plan one folded REAL special_kz curl (K3a), or None when refused."""
    from .launch import plan_folded_beta_pml_curl as _plan  # noqa: PLC0415
    return _plan(fields, pml, sub_step, block)


def folded_beta_bloch_pml_curl_coverage(fields: Any, pml: Any, sub_step: str,
                                        probe: Any = None) -> Coverage:
    """Coverage for one folded COMPLEX special_kz curl sub-step (K3b)."""
    from .launch import folded_beta_bloch_pml_curl_coverage as _coverage  # noqa: PLC0415
    return _coverage(fields, pml, sub_step, probe)


def plan_folded_beta_bloch_pml_curl(fields: Any, pml: Any, sub_step: str,
                                    block: Optional[int] = None,
                                    probe: Any = None) -> Any:
    """Plan one folded COMPLEX special_kz curl (K3b), or None when refused."""
    from .launch import plan_folded_beta_bloch_pml_curl as _plan  # noqa: PLC0415
    return _plan(fields, pml, sub_step, block, probe)


def folded_mirror_ghost_fill_complex_coverage(fields: Any, family: str,
                                              probe: Any = None) -> Coverage:
    """Coverage for one folded family's ghost fills on COMPLEX storage (K2)."""
    from .launch import folded_mirror_ghost_fill_complex_coverage as _coverage  # noqa: PLC0415
    return _coverage(fields, family, probe)


def plan_folded_mirror_ghost_fill_complex(fields: Any, family: str,
                                          block: Optional[int] = None,
                                          probe: Any = None) -> Any:
    """Plan one complex-storage mirror-fill (K2), or None when refused.

    ITS TWO PASSES MAY NOT BE FUSED — the driver puts ``zero_metal_*`` between
    them and complex float multiplication is not associative. The contract is
    written out on ``launch.plan_folded_mirror_ghost_fill_complex``; it is
    repeated as a pointer here because this is the name a caller outside the
    package binds to.
    """
    from .launch import plan_folded_mirror_ghost_fill_complex as _plan  # noqa: PLC0415
    return _plan(fields, family, block, probe)


def folded_complex_constitutive_coverage(fields: Any, pml: Any, side: str,
                                         probe: Any = None) -> Coverage:
    """Coverage for the certified complex constitutive kernel on a folded run."""
    from .launch import folded_complex_constitutive_coverage as _coverage  # noqa: PLC0415
    return _coverage(fields, pml, side, probe)


def plan_folded_complex_constitutive(fields: Any, pml: Any, side: str,
                                     block: Optional[int] = None,
                                     probe: Any = None) -> Any:
    """Plan the complex constitutive sub-step for a folded run, or None."""
    from .launch import plan_folded_complex_constitutive as _plan  # noqa: PLC0415
    return _plan(fields, pml, side, block, probe)


def folded_beta_run_constitutive_coverage(fields: Any, pml: Any,
                                          side: str) -> Coverage:
    """Coverage for the certified REAL constitutive kernel on a folded beta run."""
    from .launch import folded_beta_run_constitutive_coverage as _coverage  # noqa: PLC0415
    return _coverage(fields, pml, side)


def plan_folded_beta_run_constitutive(fields: Any, pml: Any, side: str,
                                      block: Optional[int] = None) -> Any:
    """Plan the REAL constitutive sub-step for a folded beta run, or None."""
    from .launch import plan_folded_beta_run_constitutive as _plan  # noqa: PLC0415
    return _plan(fields, pml, side, block)


def folded_offdiag_composition_coverage(fields: Any, pml: Any) -> Coverage:
    """Routing coverage for the folded off-diagonal row-product ``update_E``."""
    from .launch import folded_offdiag_composition_coverage as _coverage  # noqa: PLC0415
    return _coverage(fields, pml)


def plan_folded_offdiagonal_constitutive(fields: Any, pml: Any,
                                         block: Optional[int] = None) -> Any:
    """Plan the folded off-diagonal row-product ``update_E``, or None."""
    from .launch import plan_folded_offdiagonal_constitutive as _plan  # noqa: PLC0415
    return _plan(fields, pml, block)


def cylindrical_complex_curl_coverage(fields: Any, pml: Any, sub_step: str,
                                      probe: Any = None) -> Coverage:
    """Coverage for one complex Dcyl curl sub-step (|m| >= 1)."""
    from .launch import cylindrical_complex_curl_coverage as _coverage  # noqa: PLC0415
    return _coverage(fields, pml, sub_step, probe)


def plan_cylindrical_complex_curl(fields: Any, pml: Any, sub_step: str,
                                  block: Optional[int] = None,
                                  probe: Any = None) -> Any:
    """Plan one complex Dcyl curl, or None when refused."""
    from .launch import plan_cylindrical_complex_curl as _plan  # noqa: PLC0415
    return _plan(fields, pml, sub_step, block, probe)


def cylindrical_complex_constitutive_coverage(fields: Any, pml: Any, side: str,
                                              probe: Any = None) -> Coverage:
    """Coverage for the certified complex constitutive kernel on a Dcyl slice."""
    from .launch import cylindrical_complex_constitutive_coverage as _coverage  # noqa: PLC0415
    return _coverage(fields, pml, side, probe)


def plan_cylindrical_complex_constitutive(fields: Any, pml: Any, side: str,
                                          block: Optional[int] = None,
                                          probe: Any = None) -> Any:
    """Plan the complex constitutive sub-step for the Dcyl slice, or None."""
    from .launch import plan_cylindrical_complex_constitutive as _plan  # noqa: PLC0415
    return _plan(fields, pml, side, block, probe)


def null_constitutive_coverage(fields: Any, pml: Any, side: str) -> Coverage:
    """Is this constitutive sub-step a NO-OP on the array path for this run?"""
    from .launch import null_constitutive_coverage as _coverage  # noqa: PLC0415
    return _coverage(fields, pml, side)


def plan_null_constitutive(fields: Any, pml: Any, side: str,
                           block: Optional[int] = None) -> Any:
    """Plan the null constitutive sub-step, or None when refused.

    The one builder here that launches nothing: under an inactive absorber
    ``stepping.update_H`` returns at :916-917 and ``update_E`` at :954-955 before
    reading any array, so the covered product is a ``NullConstitutivePlan``.
    """
    from .launch import plan_null_constitutive as _plan  # noqa: PLC0415
    return _plan(fields, pml, side, block)


def nonlinear_run_pml_curl_coverage(fields: Any, pml: Any,
                                    sub_step: str) -> Coverage:
    """May the CERTIFIED real PML curl step a chi2/chi3 run's ``step_B``/``step_D``?"""
    from .launch import nonlinear_run_pml_curl_coverage as _coverage  # noqa: PLC0415
    return _coverage(fields, pml, sub_step)


def plan_nonlinear_run_pml_curl(fields: Any, pml: Any, sub_step: str,
                                block: Optional[int] = None) -> Any:
    """Plan the certified real PML curl for a chi2/chi3 run, or None."""
    from .launch import plan_nonlinear_run_pml_curl as _plan  # noqa: PLC0415
    return _plan(fields, pml, sub_step, block)


def nonlinear_run_constitutive_coverage(fields: Any, pml: Any,
                                        side: str) -> Coverage:
    """May the CERTIFIED constitutive kernel step a chi2/chi3 run's ``update_H``?"""
    from .launch import nonlinear_run_constitutive_coverage as _coverage  # noqa: PLC0415
    return _coverage(fields, pml, side)


def plan_nonlinear_run_constitutive(fields: Any, pml: Any, side: str,
                                    block: Optional[int] = None) -> Any:
    """Plan the certified constitutive sub-step for a chi2/chi3 run, or None."""
    from .launch import plan_nonlinear_run_constitutive as _plan  # noqa: PLC0415
    return _plan(fields, pml, side, block)


def folded_dispersive_constitutive_coverage(fields: Any, pml: Any) -> Coverage:
    """May the CERTIFIED dispersive ``update_E`` kernel step a FOLDED run?"""
    from .launch import folded_dispersive_constitutive_coverage as _coverage  # noqa: PLC0415
    return _coverage(fields, pml)


def plan_folded_dispersive_constitutive(fields: Any, pml: Any,
                                        block: Optional[int] = None) -> Any:
    """Plan the certified dispersive ``update_E`` for a folded run, or None."""
    from .launch import plan_folded_dispersive_constitutive as _plan  # noqa: PLC0415
    return _plan(fields, pml, block)


def folded_offdiag_dispersive_constitutive_coverage(
        fields: Any, pml: Any) -> Coverage:
    """Coverage for folded, tensor-row, pole-aware ``update_E``."""
    from .launch import (  # noqa: PLC0415
        folded_offdiag_dispersive_constitutive_coverage as _coverage)
    return _coverage(fields, pml)


def plan_folded_offdiag_dispersive_constitutive(
        fields: Any, pml: Any, block: Optional[int] = None) -> Any:
    """Plan folded, tensor-row, pole-aware ``update_E``, or ``None``."""
    from .launch import (  # noqa: PLC0415
        plan_folded_offdiag_dispersive_constitutive as _plan)
    return _plan(fields, pml, block)


def folded_complex_offdiag_pml_curl_coverage(fields: Any, pml: Any, sub_step: str,
                                             probe: Any = None) -> Coverage:
    """Coverage for K1's curl on a folded complex run carrying an off-diagonal row."""
    from .launch import folded_complex_offdiag_pml_curl_coverage as _coverage  # noqa: PLC0415
    return _coverage(fields, pml, sub_step, probe)


def plan_folded_complex_offdiag_pml_curl(fields: Any, pml: Any, sub_step: str,
                                         block: Optional[int] = None,
                                         probe: Any = None) -> Any:
    """Plan K1's certified curl for that run, or None."""
    from .launch import plan_folded_complex_offdiag_pml_curl as _plan  # noqa: PLC0415
    return _plan(fields, pml, sub_step, block, probe)


def folded_complex_offdiag_constitutive_coverage(fields: Any, pml: Any, side: str,
                                                 probe: Any = None) -> Coverage:
    """Coverage for that run's ``update_H``. ``side='E'`` is refused by name."""
    from .launch import folded_complex_offdiag_constitutive_coverage as _coverage  # noqa: PLC0415
    return _coverage(fields, pml, side, probe)


def plan_folded_complex_offdiag_constitutive(fields: Any, pml: Any, side: str,
                                             block: Optional[int] = None,
                                             probe: Any = None) -> Any:
    """Plan the certified complex constitutive sub-step for that run, or None."""
    from .launch import plan_folded_complex_offdiag_constitutive as _plan  # noqa: PLC0415
    return _plan(fields, pml, side, block, probe)


def no_pml_ade_update_p_coverage(fields: Any, pml: Any, state: Any,
                                 component: str) -> Coverage:
    """May the CERTIFIED ADE kernel advance this pole with the absorber INERT?"""
    from .launch import no_pml_ade_update_p_coverage as _coverage  # noqa: PLC0415
    return _coverage(fields, pml, state, component)


def plan_no_pml_ade_update_p(fields: Any, pml: Any, state: Any,
                             block: Optional[int] = None) -> Any:
    """Plan the certified ADE ``update_P`` for an absorber-free run, or None."""
    from .launch import plan_no_pml_ade_update_p as _plan  # noqa: PLC0415
    return _plan(fields, pml, state, block)


def complex_ade_update_p_coverage(fields: Any, pml: Any, state: Any,
                                  component: str,
                                  probe: Any = None) -> Coverage:
    """Coverage for one complex64 absorber-free ADE component."""
    from .launch import complex_ade_update_p_coverage as _coverage  # noqa: PLC0415
    return _coverage(fields, pml, state, component, probe)


def plan_complex_ade_update_p(fields: Any, pml: Any, state: Any,
                              block: Optional[int] = None,
                              probe: Any = None) -> Any:
    """Plan one complex64 absorber-free susceptibility, or ``None``."""
    from .launch import plan_complex_ade_update_p as _plan  # noqa: PLC0415
    return _plan(fields, pml, state, block, probe)


# ---------------------------------------------------------------------------
# No-absorber closure families — lazy public wrappers
# ---------------------------------------------------------------------------


def complex_no_pml_curl_coverage(fields: Any, pml: Any, sub_step: str,
                                 probe: Any = None) -> Coverage:
    """Coverage for a complex/Bloch curl without an active absorber."""
    from .launch import complex_no_pml_curl_coverage as _coverage  # noqa: PLC0415
    return _coverage(fields, pml, sub_step, probe)


def plan_complex_no_pml_curl(fields: Any, pml: Any, sub_step: str,
                             block: Optional[int] = None,
                             probe: Any = None) -> Any:
    """Plan a complex/Bloch curl without an active absorber."""
    from .launch import plan_complex_no_pml_curl as _plan  # noqa: PLC0415
    return _plan(fields, pml, sub_step, block, probe)


def complex_conductive_no_pml_curl_coverage(
        fields: Any, pml: Any, sub_step: str,
        probe: Any = None) -> Any:
    """Ask whether the complex conductive no-absorber curl covers this sub-step."""
    from .launch import (  # noqa: PLC0415
        complex_conductive_no_pml_curl_coverage as _coverage)
    return _coverage(fields, pml, sub_step, probe)


def plan_complex_conductive_no_pml_curl(
        fields: Any, pml: Any, sub_step: str,
        block: Optional[int] = None, probe: Any = None) -> Any:
    """Build the complex conductive no-absorber curl plan."""
    from .launch import plan_complex_conductive_no_pml_curl as _plan  # noqa: PLC0415
    return _plan(fields, pml, sub_step, block, probe)


def conductive_plain_curl_coverage(fields: Any, pml: Any,
                                   sub_step: str) -> Coverage:
    """Coverage for one conductive no-absorber curl."""
    from .launch import conductive_plain_curl_coverage as _coverage  # noqa: PLC0415
    return _coverage(fields, pml, sub_step)


def plan_conductive_plain_curl(fields: Any, pml: Any, sub_step: str,
                               block: Optional[int] = None) -> Any:
    """Plan one conductive no-absorber curl."""
    from .launch import plan_conductive_plain_curl as _plan  # noqa: PLC0415
    return _plan(fields, pml, sub_step, block)


def stored_e_constitutive_coverage(fields: Any, pml: Any) -> Coverage:
    """Coverage for the stored-E no-absorber electric update."""
    from .launch import stored_e_constitutive_coverage as _coverage  # noqa: PLC0415
    return _coverage(fields, pml)


def plan_stored_e_constitutive(fields: Any, pml: Any,
                               block: Optional[int] = None) -> Any:
    """Plan the stored-E no-absorber electric update."""
    from .launch import plan_stored_e_constitutive as _plan  # noqa: PLC0415
    return _plan(fields, pml, block)


def complex_stored_e_coverage(fields: Any, pml: Any,
                              probe: Any = None) -> Any:
    """Ask whether complex pole-aware stored-E ``update_E`` is covered."""
    from .launch import complex_stored_e_coverage as _coverage  # noqa: PLC0415
    return _coverage(fields, pml, probe)


def plan_complex_stored_e(fields: Any, pml: Any,
                          block: Optional[int] = None,
                          probe: Any = None) -> Any:
    """Build the complex pole-aware stored-E ``update_E`` plan."""
    from .launch import plan_complex_stored_e as _plan  # noqa: PLC0415
    return _plan(fields, pml, block, probe)


def complex_no_pml_offdiag_update_e_coverage(
        fields: Any, pml: Any, probe: Any = None) -> Coverage:
    """Coverage for complex tensor-row ``update_E`` without an absorber."""
    from .launch import (  # noqa: PLC0415
        complex_no_pml_offdiag_update_e_coverage as _coverage)
    return _coverage(fields, pml, probe)


def plan_complex_no_pml_offdiag_update_e(
        fields: Any, pml: Any, block: Optional[int] = None,
        probe: Any = None) -> Any:
    """Plan complex tensor-row ``update_E`` without an absorber."""
    from .launch import plan_complex_no_pml_offdiag_update_e as _plan  # noqa: PLC0415
    return _plan(fields, pml, block, probe)


def complex_folded_offdiag_update_e_coverage(
        fields: Any, pml: Any, probe: Any = None) -> Coverage:
    """Coverage for complex folded tensor-row PML ``update_E``."""
    from .launch import (  # noqa: PLC0415
        complex_folded_offdiag_update_e_coverage as _coverage)
    return _coverage(fields, pml, probe)


def plan_complex_folded_offdiag_update_e(
        fields: Any, pml: Any, block: Optional[int] = None,
        probe: Any = None) -> Any:
    """Plan complex folded tensor-row PML ``update_E``."""
    from .launch import plan_complex_folded_offdiag_update_e as _plan  # noqa: PLC0415
    return _plan(fields, pml, block, probe)


def plan_step(fields: Any, pml: Any, block: Optional[int] = None,
              fuse: bool = False, sources: Any = None,
              num_warps: Any = _INHERIT_LAUNCH_DEFAULT,
              fuse_ade: bool = False, probe: Any = None,
              fuse_labels: Any = None) -> Any:
    """The covered SUBSET of sub-steps for one frozen configuration, plus refusal reasons.

    Never None: a configuration nothing covers replaces nothing, which is the array
    path. ``fastpath.plan_fast_path`` reaches this composer from the driver now and
    is held shut by its ENABLE rather than by having no caller — see the note in
    ``launch``'s header; the gate and the benchmark are still callers here too.

    ``fuse=False`` (the default) composes one launch per sub-step; that path is the
    control the cross-sub-step fusion is measured against and it stays the default.

    ``probe`` is the complex-multiply expansion record. Forwarding it is not
    optional bookkeeping: without it the complex-family arms see ``None``, refuse
    with "the EXPANSION constexpr may not be guessed", and every complex slot
    falls to the array path with a reason that reads like a platform bug rather
    than a wiring one.

    ``fuse_labels`` is the caller's OFFER — the fused labels it will actually run —
    and ``None`` (the default) means no restriction. THIS WRAPPER IS WHY IT IS
    SPELLED OUT HERE RATHER THAN LEFT TO ``**kwargs``: every argument is forwarded
    POSITIONALLY below, so a keyword the wrapper does not name is a ``TypeError``
    at the one call site that matters. Measured 2026-09-02 on a device: the
    dispatcher passed it, this wrapper did not, and every case of a driver-route
    campaign reported ``DID-NOT-FUSE`` with the real reason buried in an internal
    error string.
    """
    from .launch import plan_step as _plan  # noqa: PLC0415
    return _plan(fields, pml, block, fuse, sources, _num_warps(num_warps),
                 fuse_ade, probe, fuse_labels)


def explain(fields: Any, pml: Any) -> Coverage:
    """Curl coverage verdict plus reasons. Needs no Triton."""
    return pml_curl_coverage(fields, pml)


def require_triton() -> None:
    """Raise a diagnosable error when Triton cannot be used on this machine."""
    from .launch import require_triton as _require  # noqa: PLC0415
    _require()
