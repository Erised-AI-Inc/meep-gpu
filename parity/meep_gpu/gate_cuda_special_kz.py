"""Sub-step byte-identity gate for the hand-CUDA special_kz (``grid.beta``) family.

THE QUESTION, in three parts, each measured separately:

1. **The curl.** Is ``cuda_kernels/special_kz_curl.py``'s pair byte-identical, as
   uint32 words, to ``stepping.step_B`` / ``step_D`` on a grid whose
   ``grid.beta`` is nonzero -- at one launch and at 60, at an exactly
   representable courant and at one that is not, on physical-band and
   subnormal-band operands, under both float32 subnormal policies?
2. **The constitutive sides.** ``covers_special_kz_constitutive`` admits the
   CERTIFIED constitutive pair on a beta run on the reading that
   ``stepping.update_H`` / ``update_E`` read nothing beta-dependent. A reading is
   not evidence. The constitutive arm runs the certified kernels against the array
   path on a beta grid whose two curls have ALREADY moved the state, per side, and
   the two sides are scored SEPARATELY -- because the Metal track's tranche 6 ended
   at 757/759 with the two open slots being exactly one row's constitutive sides
   while its curls were served, so "the curl works therefore the constitutive works"
   is a shape that has already been wrong once on this family.
3. **Is the beta term doing anything at all?** Every case measures the array path
   TWICE from the same frozen state -- once with beta and once with beta forced to
   zero -- and refuses itself if the two agree. That floor also measures the term's
   COMPONENT PATTERN against the oracle rather than asserting it: Bx and By (Dx and
   Dy) must move, and Bz (Dz) must NOT, because MEEP's ``cc`` loop runs over
   ``d_c`` in {X, Y} only (step_db.cpp:148-176). A kernel that put a term on the z
   component would pass a gate that only asked "did anything change".

WHAT THE FIXTURE CAN AND CANNOT BE, and it is not a choice
----------------------------------------------------------
``Grid`` refuses ``dimensions=2`` with a z extent ("an invariant axis is INFINITE
and uniform"), and ``_resolve_beta`` refuses beta off a 2-D Cartesian grid
(grid.py:668-697, MEEP fields.cpp:546-547). So EVERY beta grid has ``nz == 1``,
and this gate sweeps nothing else -- which is also what the corpus has: the two
real-storage beta rows are 1200x1x1 and 420x212x1. The consequence is stated in
``does_not_claim``: the z-axis ghost rule and the z-axis wall mask are exercised
only at a single cell here, and they are the CERTIFIED pair's, already gated
elsewhere on grids that are thick in z.

THE FOLD IS IN THE SWEEP, at both terminations, because one of the two corpus rows
has one: ``TestSpecialKz.test_eigsrc_kz_1_real_imag`` is a Y fold with a PERIODIC
termination (stored 212 > owned 211). The beta term reuses a CENTRE register the
fold never touches, so the composition is expected to be free -- and expecting is
why it is swept rather than argued.

WHAT WOULD MAKE THIS GATE VACUOUS, AND THE FLOORS THAT REFUSE IT
----------------------------------------------------------------
* ``oracle_moved`` -- the fraction of output words the array path changed from the
  frozen input. A case that moved nothing is refused, not passed.
* ``beta_term_is_live`` -- the beta-on and beta-off oracle runs must differ, and
  differ on the components the transcription says carry a term and on no other. A
  case where the term changes no word measures the certified kernel, not this one.
* ``absorber_is_not_the_identity`` -- ``max|kms-1|`` and ``max|sinv-1|`` over every
  axis. An identity profile hides a coefficient-index error.
* The mutation battery, including legs that MUST BE UNCAUGHT. A battery of
  only-must-be-caught legs scores identically whether the comparator works or has
  degenerated into failing everything. The two beta nulls are the family's own:
  ``curl + (-(c*g))`` for ``curl - (c*g)`` (IEEE-754 defines subtraction as
  addition of the negation) and ``(g * c)`` for ``(c * g)`` (float multiply
  commutes exactly). Each is paired with a must-catch leg on the same expression.

RUNNING IT
----------
Device (the GPU host, ONE verified-empty GPU; the cache dir MUST carry the policy
token because CuPy's disk-cache key is computed above the strip seam)::

    CUDA_VISIBLE_DEVICES=$GPU CUPY_CACHE_DIR=$OUT/cupy_cache/ftz_stripped \\
        python -u gate_cuda_special_kz.py \\
        --subnormal-policy keep --out $OUT/keep/gate.json

Laptop (no CUDA). The NumPy backend runs the SAME oracle, the same fixture, the
same floors and the same HOST mutations against a transcription of the device
tree. IT COMPILES NOTHING AND CERTIFIES NOTHING::

    python -u gate_cuda_special_kz.py --backend numpy --out /tmp/skz.json

One flushed line per case (the progress-reporting rule); the artifact is rewritten atomically
after every case, so an interrupted run keeps everything up to the failure.
Correctness only -- no throughput claim is made or possible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_API = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _REPO_API not in sys.path:
    sys.path.insert(0, _REPO_API)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

try:
    import cupy as cp
except ImportError:  # laptop: the NumPy backend still runs
    cp = None

import probe_fused_kernel_bit_identity as probe  # noqa: E402
import gate_provenance  # noqa: E402

from meep_gpu import stepping  # noqa: E402
from meep_gpu.fields import Fields, IYEE_SHIFTS  # noqa: E402
from meep_gpu.grid import Grid, Mirror  # noqa: E402
from meep_gpu.pml import PML  # noqa: E402
from meep_gpu.cuda_kernels import coverage  # noqa: E402
from meep_gpu.cuda_kernels import special_kz_curl as family  # noqa: E402

log = probe.log
bit_compare = probe.bit_compare
combine = probe.combine
to_host = probe.to_host
operand_census = probe.operand_census
subnormal_band_hosts = probe.subnormal_band_hosts

SEED = 20260820
MULTI_STEP_BUDGET = 60
_ADVANCE = np.float32(0.97)

BC_PERIODIC = 0
BC_METALLIC = 1
BC_MIRROR_PERIODIC = 2

SUB_STEPS: Tuple[str, ...] = ("step_B", "step_D")
SIDES: Tuple[str, ...] = ("H", "E")

SUB_STEP_ARRAYS: Dict[str, Dict[str, Any]] = {
    "step_B": {"targets": ("Bx", "By", "Bz"), "aux": ("fu_Bx", "fu_By", "fu_Bz"),
               "sources": ("Ex", "Ey", "Ez"), "half_integer": True,
               "backward": False},
    "step_D": {"targets": ("Dx", "Dy", "Dz"), "aux": ("fu_Dx", "fu_Dy", "fu_Dz"),
               "sources": ("Hx", "Hy", "Hz"), "half_integer": False,
               "backward": True},
}

CONSTITUTIVE_ARRAYS: Dict[str, Dict[str, Any]] = {
    "H": {"targets": ("Hx", "Hy", "Hz"), "aux": ("f_w_Hx", "f_w_Hy", "f_w_Hz"),
          "sources": ("Bx", "By", "Bz")},
    "E": {"targets": ("Ex", "Ey", "Ez"), "aux": ("f_w_Ex", "f_w_Ey", "f_w_Ez"),
          "sources": ("Dx", "Dy", "Dz")},
}

#: The six curl terms as the SHIPPED KERNEL spells them, plus WHICH OPERAND the
#: beta term takes and at WHICH SIGN. Read as ``(target, aux, first, first_axis,
#: second, second_axis, dsig, dsigu, beta_operand, beta_sign)`` where
#: ``beta_operand`` is ``"first"``, ``"second"`` or ``None``.
#:
#: Written out here as a transcription of the DEVICE source rather than derived
#: from ``stepping``: the NumPy leg consumes it, and deriving it from the oracle
#: would make that leg compare the oracle with itself.
#: :func:`check_terms_against_stepping` pins it against ``stepping``.
CURL_TERMS: Dict[str, Tuple[Tuple[Any, ...], ...]] = {
    "step_B": (
        ("Bx", "fu_Bx", "Ez", 1, "Ey", 2, 1, 2, "second", +1.0),
        ("By", "fu_By", "Ex", 2, "Ez", 0, 2, 0, "first", -1.0),
        ("Bz", "fu_Bz", "Ey", 0, "Ex", 1, 0, 1, None, 0.0),
    ),
    "step_D": (
        ("Dx", "fu_Dx", "Hz", 1, "Hy", 2, 1, 2, "second", +1.0),
        ("Dy", "fu_Dy", "Hx", 2, "Hz", 0, 2, 0, "first", -1.0),
        ("Dz", "fu_Dz", "Hy", 0, "Hx", 1, 0, 1, None, 0.0),
    ),
}

#: Every fixture is nz == 1: ``Grid`` refuses a z extent on a ``dimensions=2``
#: grid and ``_resolve_beta`` refuses beta anywhere else. Both corpus rows are of
#: this shape.
BETA_SPECS: Tuple[Dict[str, Any], ...] = (
    # The two corpus shapes, in kind. refl-angular-kz2d.py is an all-periodic
    # single row; eigsrc_kz is a Y fold with a PERIODIC termination.
    {"label": "plain_periodic", "axes": "", "phase": 1, "beta": 0.3321611318837033,
     "boundaries": ("periodic", "periodic", "periodic"), "cell": (12.0, 8.0, 0.0)},
    {"label": "plain_metallic_x", "axes": "", "phase": 1, "beta": 0.3321611318837033,
     "boundaries": ("metallic", "periodic", "periodic"), "cell": (12.0, 8.0, 0.0)},
    {"label": "plain_metallic_xy", "axes": "", "phase": 1, "beta": -0.685,
     "boundaries": ("metallic", "metallic", "periodic"), "cell": (12.0, 8.0, 0.0)},
    # NEGATIVE beta: the coefficient's sign is a separate draw from the sign the
    # call site carries, and a kernel that folded the two would only show here.
    {"label": "plain_periodic_negative_beta", "axes": "", "phase": 1, "beta": -0.39073112848927377,
     "boundaries": ("periodic", "periodic", "periodic"), "cell": (10.0, 12.0, 0.0)},
    # THE FOLD, both terminations, both plane parities. The corpus row is the
    # PERIODIC one; the metallic one is swept because a family must not release on
    # the half of a split it happens to need.
    {"label": "fold_Y_periodic", "axes": "Y", "phase": 1, "beta": 0.2,
     "boundaries": ("periodic", "periodic", "periodic"), "cell": (8.0, 16.0, 0.0)},
    {"label": "fold_Y_metallic", "axes": "Y", "phase": 1, "beta": 0.2,
     "boundaries": ("periodic", "metallic", "periodic"), "cell": (8.0, 16.0, 0.0)},
    {"label": "fold_Y_periodic_odd_plane", "axes": "Y", "phase": -1, "beta": 0.2,
     "boundaries": ("periodic", "periodic", "periodic"), "cell": (8.0, 16.0, 0.0)},
    # A FOLD ON THE SLOWEST-STRIDE AXIS: the beta partner for By is the Ex CENTRE
    # and for Bx the Ey centre, so an index defect that confused a centre with a
    # neighbour would move with the stride.
    {"label": "fold_X_periodic", "axes": "X", "phase": 1, "beta": -0.42,
     "boundaries": ("periodic", "periodic", "periodic"), "cell": (16.0, 8.0, 0.0)},
    {"label": "fold_X_metallic", "axes": "X", "phase": 1, "beta": -0.42,
     "boundaries": ("metallic", "periodic", "periodic"), "cell": (16.0, 8.0, 0.0)},
)

COURANTS: Tuple[float, ...] = (0.5, 0.35)
INEXACT_COURANT = 0.35
VALUE_CLASSES: Tuple[str, ...] = ("uniform", "subnormal_band")

GUARD_SETS: Tuple[Tuple[str, Tuple[str, ...], bool], ...] = (
    ("fmad_false", ("--fmad=false",), True),
    ("default_no_options", (), False),
)


# ---------------------------------------------------------------------------
# THE FAMILY'S OWN SOURCE MUTATIONS
# ---------------------------------------------------------------------------
#
# Each is a text transform on the device string, returning ``(mutated, sites)``.
# ``sites`` is asserted nonzero and the mutated text asserted different, because a
# mutation that matched nothing and a mutation that cannot bite look identical in
# a verdict.

_BETA_LINES = (
    ("        curl = curl - (beta_plus * f2);\n", "plus", "f2"),
    ("        curl = curl - (beta_minus * f1);\n", "minus", "f1"),
)


def _replace_all(source: str, pairs: Sequence[Tuple[str, str]]) -> Tuple[str, int]:
    sites = 0
    out = source
    for old, new in pairs:
        sites += out.count(old)
        out = out.replace(old, new)
    return out, sites


def _drop_beta_term(source: str) -> Tuple[str, int]:
    """The whole feature deleted: the certified kernel, on a beta run."""
    return _replace_all(source, [(line, "") for line, _s, _o in _BETA_LINES])


def _beta_partner_shifted(source: str) -> Tuple[str, int]:
    """NULL, AND THE NULL IS THE FINDING. MEASURED 2026-08-20, 0/4 caught.

    This began as the family's headline must-catch leg -- "the SHIFTED operand
    instead of the CENTRE one", the transcription's most likely silent slip. It is
    UNDETECTABLE on a special_kz grid, and the reason is structural rather than a
    property of the fixture:

    * Bx's beta partner is Ey and the only shifted Ey in that block is ``ss``, the
      neighbour along Z; By's partner is Ex and its only shifted companion is
      ``sf``, also along Z (``B_CURL_TERMS``: Bx pairs Ez/y with Ey/z, By pairs
      Ex/z with Ez/x). The D side is the same pair of axes.
    * EVERY beta grid has ``nz == 1``: ``Grid`` refuses a z extent at
      ``dimensions=2`` and ``_resolve_beta`` refuses beta anywhere else. With one
      cell on a periodic axis ``shift_up``/``shift_dn`` return ``g[idx]`` itself.

    So on the only grids this family can ever be handed, the centre operand and the
    z-shifted operand ARE THE SAME WORD, and swapping them is not a defect at all.
    Recorded as a NULL that must be UNCAUGHT rather than deleted, because a
    mutation quietly dropped and a mutation that cannot bite look identical in a
    verdict -- and because if a future grid ever made z thick, this leg would start
    failing and say so. The discriminating leg for "the wrong operand" is
    :func:`_beta_partner_wrong_component`, which IS caught.
    """
    return _replace_all(source, [
        ("curl = curl - (beta_plus * f2);", "curl = curl - (beta_plus * ss);"),
        ("curl = curl - (beta_minus * f1);", "curl = curl - (beta_minus * sf);"),
    ])


def _beta_partner_wrong_component(source: str) -> Tuple[str, int]:
    """The OTHER source component as the beta partner, at the same sign.

    Bx takes Ez (its ``first``) instead of Ey (its ``second``), and By takes Ez
    instead of Ex -- the cross product's two slots exchanged. This is the
    observable half of "the wrong operand" once
    :func:`_beta_partner_shifted` is measured inert: the two components are
    genuinely different volumes, so the swap survives ``nz == 1``.
    """
    return _replace_all(source, [
        ("curl = curl - (beta_plus * f2);", "curl = curl - (beta_plus * f1);"),
        ("curl = curl - (beta_minus * f1);", "curl = curl - (beta_minus * f2);"),
    ])


def _beta_swap_signs(source: str) -> Tuple[str, int]:
    """``sign = (d_c == X ? +1 : -1)`` read backwards (step_db.cpp:148-176)."""
    return _replace_all(source, [
        ("curl = curl - (beta_plus * f2);", "curl = curl - (beta_MINUS * f2);"),
        ("curl = curl - (beta_minus * f1);", "curl = curl - (beta_plus * f1);"),
        ("beta_MINUS", "beta_minus"),
    ])


def _beta_added_not_subtracted(source: str) -> Tuple[str, int]:
    """``stepping`` negates the PRODUCT (:784) and the caller ADDS (:361); a
    kernel that adds the unnegated product has the term at the wrong sign."""
    return _replace_all(source, [
        ("curl = curl - (beta_plus * f2);", "curl = curl + (beta_plus * f2);"),
        ("curl = curl - (beta_minus * f1);", "curl = curl + (beta_minus * f1);"),
    ])


def _beta_scaled_by_dtdx(source: str) -> Tuple[str, int]:
    """The term treated as a finite difference. stepping.py:758-762 says it is an
    ANALYTIC derivative and carries no 1/dx; scaling it is a plausible, smooth,
    entirely wrong answer."""
    return _replace_all(source, [
        ("curl = curl - (beta_plus * f2);",
         "curl = curl - (dtdx * (beta_plus * f2));"),
        ("curl = curl - (beta_minus * f1);",
         "curl = curl - (dtdx * (beta_minus * f1));"),
    ])


def _beta_on_z_component(source: str) -> Tuple[str, int]:
    """A term on the z component, which MEEP's ``cc`` loop never gives it.

    Inserted into the THIRD block only, matched on that block's own stencil line
    (``sf`` there is the first source shifted along x), so the two blocks that do
    carry a term are untouched.
    """
    for target in ("Bz", "Dz"):
        anchor = source.find(f"pml_apply({target}, fu_{target},")
        if anchor < 0:
            continue
        stencil = "        float curl = dtdx * ((sf - f1) + (f2 - ss));\n"
        position = source.rfind(stencil, 0, anchor)
        if position < 0:
            continue
        cut = position + len(stencil)
        return (source[:cut] + "        curl = curl - (beta_plus * f2);\n"
                + source[cut:]), 1
    return source, 0


def _beta_after_mask(source: str) -> Tuple[str, int]:
    """The term added AFTER the ownership mask instead of before it.

    ``stepping`` adds at :361/:363 and masks at :369 (the D side at :443/:445
    against :450), so a masked cell's curl is exactly zero. Move the term below
    the mask and the masked plane carries a beta increment MEEP never wrote.
    Scored only on legs whose grid carries a mask code -- on an all-periodic
    unfolded grid no mask line fires and this is not a defect at all.
    """
    out = source
    sites = 0
    for line, sign, operand in _BETA_LINES:
        if line not in out:
            continue
        # Delete it from above the mask block and re-insert it below, immediately
        # before the pml_apply for the same component.
        head, _sep, tail = out.partition(line)
        marker = "        pml_apply("
        position = tail.find(marker)
        if position < 0:
            continue
        out = head + tail[:position] + line + tail[position:]
        sites += 1
    return out, sites


def _beta_subtract_as_add_negative(source: str) -> Tuple[str, int]:
    """NULL. IEEE-754 defines ``a - b`` as ``a + (-b)``, so the two spellings are
    the same bits on every input including signed zeros. This is grouping choice 2
    stated as an experiment: if it is CAUGHT, the identity this family rests on is
    false on this platform and the transcription has to be respelled."""
    return _replace_all(source, [
        ("curl = curl - (beta_plus * f2);", "curl = curl + (-(beta_plus * f2));"),
        ("curl = curl - (beta_minus * f1);", "curl = curl + (-(beta_minus * f1));"),
    ])


def _beta_product_operands_commuted(source: str) -> Tuple[str, int]:
    """NULL. float32 multiply commutes exactly, so ``g * c`` and ``c * g`` are the
    same word. stepping.py:811 puts the coefficient on the LEFT; this leg measures
    that the choice is a transcription convention rather than an arithmetic fact."""
    return _replace_all(source, [
        ("(beta_plus * f2)", "(f2 * beta_plus)"),
        ("(beta_minus * f1)", "(f1 * beta_minus)"),
    ])


BETA_SOURCE_MUTATIONS: Dict[str, Any] = {
    "drop_beta_term": _drop_beta_term,
    "beta_partner_shifted": _beta_partner_shifted,
    "beta_partner_wrong_component": _beta_partner_wrong_component,
    "beta_swap_signs": _beta_swap_signs,
    "beta_added_not_subtracted": _beta_added_not_subtracted,
    "beta_scaled_by_dtdx": _beta_scaled_by_dtdx,
    "beta_on_z_component": _beta_on_z_component,
    "beta_after_mask": _beta_after_mask,
    "beta_subtract_as_add_negative": _beta_subtract_as_add_negative,
    "beta_product_operands_commuted": _beta_product_operands_commuted,
}

#: MUST BE UNCAUGHT. The first two are IEEE identities; the third is a MEASURED
#: structural fact about every grid this family can be handed -- see
#: :func:`_beta_partner_shifted`, which is where the reasoning lives.
BETA_NULL_MUTATIONS: Tuple[str, ...] = ("beta_subtract_as_add_negative",
                                        "beta_product_operands_commuted",
                                        "beta_partner_shifted")

#: Legs whose grid must carry a mask code for the mutation's lines to fire.
SOURCE_MUTATION_REQUIRES_MASK: Dict[str, bool] = {"beta_after_mask": True}

#: The shared curl defects, borrowed rather than respelled so this gate cannot own
#: a second spelling of a defect the certified record was cut against.
SHARED_SOURCE_MUTATIONS: Tuple[str, ...] = (
    "regroup_stencil", "drop_metallic_mask", "drop_folded_periodic_top_mask",
    "drop_folded_periodic_near_mask", "swap_dsig_dsigu", "drop_fu_store",
    "read_fprev_after_store", "fortran_order_index_decomposition",
    "commute_dtdx_scale", "reload_fu_from_memory",
)
SHARED_NULL_MUTATIONS: Tuple[str, ...] = probe.PML_NULL_MUTATIONS
SHARED_REQUIRES_CODE: Dict[str, int] = {
    "drop_metallic_mask": BC_METALLIC,
    "drop_folded_periodic_top_mask": BC_MIRROR_PERIODIC,
    "drop_folded_periodic_near_mask": BC_MIRROR_PERIODIC,
}

#: HOST mutations: they corrupt the SCALARS or the TABLES rather than the device
#: text, so they run on both backends.
HOST_MUTATIONS: Tuple[str, ...] = (
    "swap_beta_coefficients",
    "zero_beta_coefficients",
    "beta_coefficient_scaled_by_dtdx",
    "beta_coefficient_rounded_twice",
    "swap_curl_sublattice",
    "beta_coefficient_negated_from_plus",
)
#: MUST BE UNCAUGHT: float64 negation and float32 rounding commute exactly, so
#: computing the minus coefficient as ``-plus`` in float64 and rounding once is
#: the same word as computing it from ``sign = -1`` (grouping choice 1). A gate
#: whose every host leg must be caught cannot tell a working comparator from one
#: that fails everything.
HOST_NULL_MUTATIONS: Tuple[str, ...] = ("beta_coefficient_negated_from_plus",)


# ---------------------------------------------------------------------------
# The fixture
# ---------------------------------------------------------------------------

class _NumpyWearingCupysName:
    """NumPy behind CuPy's ``__name__`` -- the one clause a laptop cannot satisfy."""

    __name__ = "cupy"

    def __getattr__(self, item):
        return getattr(np, item)


def build(xp, spec: Dict[str, Any], courant: float):
    """A frozen ``(fields, layer, grid)`` triple for one beta spec.

    ``dimensions=2`` is passed explicitly and ``cell_size``'s z extent is 0: those
    are the only grids ``_resolve_beta`` admits, so they are the only grids a
    dispatch could ever hand this pair.
    """
    planes = tuple(Mirror(name, spec["phase"]) for name in spec["axes"])
    grid = Grid(resolution=1.0, cell_size=tuple(spec["cell"]),
                boundaries=tuple(spec["boundaries"]), symmetry=planes,
                xp=xp, courant=courant, dimensions=2, beta=spec["beta"])
    thickness = tuple(
        (0, 0) if grid.shape[axis] < 6
        else (0, 2) if grid.is_mirrored(axis)
        else (2, 2)
        for axis in range(3))
    layer = PML(grid=grid, thickness=thickness)
    fields = Fields(grid=grid)
    fields.enable_field_storage()
    fields.enable_pml_storage()
    return fields, layer, grid


def state_names(sub_step: str) -> Tuple[str, ...]:
    spec = SUB_STEP_ARRAYS[sub_step]
    return tuple(spec["targets"]) + tuple(spec["aux"]) + tuple(spec["sources"])


def outputs(sub_step: str) -> Tuple[str, ...]:
    spec = SUB_STEP_ARRAYS[sub_step]
    return tuple(spec["targets"]) + tuple(spec["aux"])


def constitutive_state_names(side: str) -> Tuple[str, ...]:
    spec = CONSTITUTIVE_ARRAYS[side]
    return tuple(spec["targets"]) + tuple(spec["aux"]) + tuple(spec["sources"])


def constitutive_outputs(side: str) -> Tuple[str, ...]:
    spec = CONSTITUTIVE_ARRAYS[side]
    return tuple(spec["targets"]) + tuple(spec["aux"])


def seed_state(fields, grid, names: Sequence[str], value_class: str,
               rng) -> Dict[str, np.ndarray]:
    """Physical-band or subnormal-band values in every array the sub-step touches.

    THE AUXILIARIES START NONZERO in both classes: a zero ``fu`` makes ``kms*prev``
    exactly zero on the first launch whatever ``kms`` holds, which would hide a
    mis-indexed coefficient until step two.
    """
    xp = grid.xp
    if value_class == "uniform":
        host = {name: rng.uniform(-1.0, 1.0, size=grid.shape).astype(np.float32)
                for name in names}
    elif value_class == "subnormal_band":
        host = subnormal_band_hosts(tuple(names), tuple(grid.shape), rng)
    else:
        raise ValueError(f"value class {value_class!r} is not one of {VALUE_CLASSES}")
    for name, values in host.items():
        getattr(fields, name)[...] = xp.asarray(np.ascontiguousarray(values))
    return host


def snapshot(fields, names: Sequence[str]) -> Dict[str, np.ndarray]:
    return {name: to_host(getattr(fields, name)).copy() for name in names}


def restore(fields, frozen: Dict[str, np.ndarray]) -> None:
    xp = fields.grid.xp
    for name, values in frozen.items():
        getattr(fields, name)[...] = xp.asarray(values)


def advance_sources(fields, names: Sequence[str]) -> None:
    """Move the operands between launches, identically on both paths."""
    for name in names:
        getattr(fields, name)[...] = getattr(fields, name) * _ADVANCE


# ---------------------------------------------------------------------------
# Codes, tables and coefficients
# ---------------------------------------------------------------------------

def boundary_codes_for(grid) -> Tuple[Tuple[int, int, int], Tuple[str, str, str]]:
    """The SHIPPED code triple, from the function the predicate itself consults."""
    kinds = tuple(coverage.real_pml_boundary_kinds(grid))
    codes, refusal = coverage.real_curl_boundary_codes(grid)
    if refusal is not None:
        raise ValueError(f"the shipped resolver refuses this grid: {refusal}")
    return tuple(int(code) for code in codes), kinds


def tables_for(sub_step: str, layer) -> Dict[str, Any]:
    suffix = "_h" if SUB_STEP_ARRAYS[sub_step]["half_integer"] else ""
    return {f"{stem}_{axis}": getattr(layer, f"{stem}_{axis}{suffix}").reshape(-1)
            for axis in "xyz" for stem in ("kms", "sinv")}


def wrong_sub_lattice_tables(sub_step: str, layer) -> Dict[str, Any]:
    suffix = "" if SUB_STEP_ARRAYS[sub_step]["half_integer"] else "_h"
    return {f"{stem}_{axis}": getattr(layer, f"{stem}_{axis}{suffix}").reshape(-1)
            for axis in "xyz" for stem in ("kms", "sinv")}


def host_mutated_inputs(name: Optional[str], sub_step: str, layer, grid,
                        dtdx: float) -> Tuple[Dict[str, Any], Tuple[float, float]]:
    """``(tables, (plus, minus))`` with one host-level defect applied, or neither."""
    tables = tables_for(sub_step, layer)
    plus, minus = family.beta_curl_coefficients(grid.beta, grid.dt)
    if name is None:
        return tables, (plus, minus)
    if name == "swap_curl_sublattice":
        return wrong_sub_lattice_tables(sub_step, layer), (plus, minus)
    if name == "swap_beta_coefficients":
        return tables, (minus, plus)
    if name == "zero_beta_coefficients":
        return tables, (0.0, 0.0)
    if name == "beta_coefficient_scaled_by_dtdx":
        return tables, (float(np.float32(plus * dtdx)),
                        float(np.float32(minus * dtdx)))
    if name == "beta_coefficient_rounded_twice":
        # float32 -> float16 -> float32: a second rounding of the same value, which
        # is what a transcription that let the coefficient through a narrower
        # intermediate would produce.
        return tables, (float(np.float32(np.float16(plus))),
                        float(np.float32(np.float16(minus))))
    if name == "beta_coefficient_negated_from_plus":
        # THE NULL: the minus coefficient computed as float64 -(sign*2*pi*b*dt)
        # and rounded once, instead of from sign = -1. Must be UNCAUGHT.
        exact = 2.0 * np.pi * float(grid.beta) * float(grid.dt)
        return tables, (float(np.float32(exact)), float(np.float32(-exact)))
    raise KeyError(f"unknown host mutation {name!r}")


# ---------------------------------------------------------------------------
# The two kernel-side backends
# ---------------------------------------------------------------------------

def run_kernel_cuda(sub_step: str, fields, tables, codes, dtdx: float,
                    beta: Tuple[float, float]) -> None:
    family.step_special_kz(sub_step, fields, tables,
                           tuple(np.int32(c) for c in codes), dtdx, beta)
    cp.cuda.runtime.deviceSynchronize()


def _plane(axis: int, index: int, shape) -> Tuple[Any, ...]:
    key: List[Any] = [slice(None)] * len(shape)
    key[axis] = index
    return tuple(key)


def _shift_up_numpy(field: np.ndarray, axis: int, bc: int) -> np.ndarray:
    shifted = np.roll(field, -1, axis=axis)
    if bc != BC_PERIODIC:
        shifted[_plane(axis, -1, field.shape)] = np.float32(0.0)
    return shifted


def _shift_dn_numpy(field: np.ndarray, axis: int, bc: int) -> np.ndarray:
    shifted = np.roll(field, 1, axis=axis)
    if bc != BC_PERIODIC:
        shifted[_plane(axis, 0, field.shape)] = np.float32(0.0)
    return shifted


def _broadcast(vector, axis: int) -> np.ndarray:
    shape = [1, 1, 1]
    shape[axis] = int(np.asarray(vector).size)
    return np.asarray(vector, dtype=np.float32).reshape(shape)


def run_kernel_numpy(sub_step: str, fields, tables, codes, dtdx: float,
                     beta: Tuple[float, float]) -> None:
    """The device tree, transcribed, in float32 -- the laptop backend.

    THE GROUPING IS THE SHIPPED ONE and it is load-bearing::

        float curl = dtdx * ((sf - f1) + (f2 - ss));
        curl = curl - (beta_plus * f2);          // Bx / Dx  (centre, sign +1)
        curl = curl - (beta_minus * f1);         // By / Dy  (centre, sign -1)
        <the three mask lines>
        pml_apply(...)

    THIS COMPILES NOTHING AND CERTIFIES NOTHING. It cannot see a defect the NVRTC
    contraction guard exists for and it is not the shipped bytes; what it settles
    is whether the harness can localize a divergence and whether it can fail.
    """
    spec = SUB_STEP_ARRAYS[sub_step]
    shift = _shift_dn_numpy if spec["backward"] else _shift_up_numpy
    scale = np.float32(dtdx)
    coefficients = {"second": np.float32(beta[0]), "first": np.float32(beta[1])}
    names = "xyz"
    for (target, aux, first, first_axis, second, second_axis, dsig, dsigu,
         beta_operand, _sign) in CURL_TERMS[sub_step]:
        f = getattr(fields, target)
        fu = getattr(fields, aux)
        f1 = getattr(fields, first)
        f2 = getattr(fields, second)
        sf = shift(f1, first_axis, codes[first_axis])
        ss = shift(f2, second_axis, codes[second_axis])
        curl = (scale * ((sf - f1) + (f2 - ss))).astype(np.float32)
        if beta_operand is not None:
            partner = f2 if beta_operand == "second" else f1
            curl = (curl - (coefficients[beta_operand] * partner)).astype(np.float32)
        iyee = IYEE_SHIFTS[target]
        for axis in range(3):
            if iyee[axis] == 0:
                if codes[axis] in (BC_METALLIC, BC_MIRROR_PERIODIC):
                    curl[_plane(axis, 0, curl.shape)] = np.float32(0.0)
            elif codes[axis] == BC_MIRROR_PERIODIC:
                curl[_plane(axis, -1, curl.shape)] = np.float32(0.0)
        kms = _broadcast(tables[f"kms_{names[dsig]}"], dsig)
        sinv = _broadcast(tables[f"sinv_{names[dsig]}"], dsig)
        kms_u = _broadcast(tables[f"kms_{names[dsigu]}"], dsigu)
        sinv_u = _broadcast(tables[f"sinv_{names[dsigu]}"], dsigu)
        fprev = fu.copy()
        fu_new = (((fprev * kms).astype(np.float32) - curl).astype(np.float32)
                  * sinv).astype(np.float32)
        fu[...] = fu_new
        a = ((f * kms_u).astype(np.float32) + fu_new).astype(np.float32)
        f[...] = ((a - fprev).astype(np.float32) * sinv_u).astype(np.float32)


# ---------------------------------------------------------------------------
# The floors
# ---------------------------------------------------------------------------

def moved_fraction(before: Dict[str, np.ndarray], after: Dict[str, np.ndarray],
                   names: Sequence[str]) -> float:
    moved = total = 0
    for name in names:
        a = np.ascontiguousarray(before[name], dtype=np.float32).ravel().view(np.uint32)
        b = np.ascontiguousarray(after[name], dtype=np.float32).ravel().view(np.uint32)
        moved += int(np.count_nonzero(a != b))
        total += int(a.size)
    return moved / total if total else 0.0


def differing_words(a: Dict[str, np.ndarray], b: Dict[str, np.ndarray],
                    name: str) -> int:
    x = np.ascontiguousarray(a[name], dtype=np.float32).ravel().view(np.uint32)
    y = np.ascontiguousarray(b[name], dtype=np.float32).ravel().view(np.uint32)
    return int(np.count_nonzero(x != y))


def absorber_is_not_the_identity(sub_step: str, layer) -> Dict[str, Any]:
    tables = tables_for(sub_step, layer)
    worst = 0.0
    for key, vector in tables.items():
        values = to_host(vector).astype(np.float64)
        worst = max(worst, float(np.max(np.abs(values - 1.0))))
    return {"max_deviation_from_identity": worst, "meets_floor": worst > 0.0}


def check_terms_against_stepping() -> Dict[str, Any]:
    """Pin :data:`CURL_TERMS` -- stencil AND beta assignment -- against ``stepping``.

    The stencil half is the certified gate's check. The BETA half is this gate's:
    ``stepping.step_B`` gives Bx the ``electric["Ey"]`` partner at +1 and By the
    ``electric["Ex"]`` partner at -1 (:361/:363), and Bz none. Ey is Bx's SECOND
    source and Ex is By's FIRST, so the table's ``beta_operand`` column has to
    agree with that mapping or the NumPy leg transcribes a different kernel from
    the one the device compiles.
    """
    out: Dict[str, Any] = {"agreed": True, "checked": [], "disagreements": []}
    axis_of = {"x": 0, "y": 1, "z": 2}
    #: What stepping.py:384-391 / :467-474 pair with each target, read off the
    #: source: (partner component, sign). Bz/Dz absent.
    beta_call_sites = {"step_B": {"Bx": ("Ey", +1.0), "By": ("Ex", -1.0)},
                       "step_D": {"Dx": ("Hy", +1.0), "Dy": ("Hx", -1.0)}}
    for sub_step, source in (("step_B", stepping.B_CURL_TERMS),
                             ("step_D", stepping.D_CURL_TERMS)):
        for mine, theirs in zip(CURL_TERMS[sub_step], source):
            record = {
                "sub_step": sub_step, "target": mine[0],
                "mine": [mine[0], mine[2], mine[3], mine[4], mine[5], mine[6], mine[7]],
                "stepping": [theirs.target, theirs.first, theirs.first_axis,
                             theirs.second, theirs.second_axis,
                             axis_of[theirs.dsig], axis_of[theirs.dsigu]],
            }
            expected = beta_call_sites[sub_step].get(mine[0])
            if expected is None:
                record["beta_agrees"] = mine[8] is None
                record["beta"] = [None, None]
            else:
                partner, sign = expected
                component = theirs.second if mine[8] == "second" else theirs.first
                record["beta"] = [component, mine[9]]
                record["beta_expected"] = [partner, sign]
                record["beta_agrees"] = (component == partner and mine[9] == sign)
            record["agrees"] = (record["mine"] == record["stepping"]
                                and record["beta_agrees"])
            out["checked"].append(record)
            if not record["agrees"]:
                out["agreed"] = False
                out["disagreements"].append(record)
    return out


class _BetaZeroGrid:
    """The fixture's grid with beta forced to zero -- the floor's control arm.

    Only ``stepping``'s beta branch reads it (``grid.beta != 0.0`` at :356/:438 and
    ``_special_kz_beta_term`` at :770), so an array-path run through this proxy is
    the SAME sub-step with the term deleted and nothing else changed.
    """

    __slots__ = ("_grid",)

    def __init__(self, grid) -> None:
        object.__setattr__(self, "_grid", grid)

    @property
    def beta(self) -> float:
        return 0.0

    def __getattr__(self, item):
        return getattr(object.__getattribute__(self, "_grid"), item)


def beta_term_is_live(fields, layer, grid, sub_step: str,
                      frozen: Dict[str, np.ndarray],
                      reference: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Run the ORACLE again with beta zeroed, and measure what the term moved.

    Two floors in one measurement, and the second is the one that matters:

    * the beta-on and beta-off runs must DIFFER at all -- otherwise this case
      measures the certified kernel and says nothing about this family;
    * they must differ on exactly the components the transcription says carry a
      term. Bz (Dz) moving would mean the ORACLE puts a term where this kernel does
      not; Bx or By (Dx, Dy) not moving would mean the fixture cannot see the term
      on the component it is supposed to test. Both are refusals, and the pattern
      is measured against ``stepping`` rather than asserted from the reading.
    """
    saved = fields.grid
    try:
        object.__setattr__(fields, "grid", _BetaZeroGrid(saved))
        restore(fields, frozen)
        if sub_step == "step_B":
            stepping.step_B(fields, layer)
        else:
            stepping.step_D(fields, layer)
        without = snapshot(fields, state_names(sub_step))
    finally:
        object.__setattr__(fields, "grid", saved)
    per_component = {name: differing_words(reference, without, name)
                     for name in outputs(sub_step)}
    expected = {term[0]: term[8] is not None for term in CURL_TERMS[sub_step]}
    pattern_ok = True
    for target, carries in expected.items():
        aux = "fu_" + target
        moved = per_component[target] > 0 or per_component[aux] > 0
        if moved != carries:
            pattern_ok = False
    total = sum(per_component.values())
    return {"differing_words_vs_beta_zero": per_component,
            "total": total,
            "component_pattern_matches_stepping": pattern_ok,
            "expected_carries_term": expected,
            "meets_floor": total > 0 and pattern_ok}


def structure_facts(grid) -> Dict[str, Any]:
    return {
        "shape": [int(n) for n in grid.shape],
        "dimensions": int(getattr(grid, "dimensions", 0)),
        "beta": float(grid.beta),
        "mirrored": [bool(grid.is_mirrored(a)) for a in range(3)],
        "metallic": [bool(grid.is_metallic(a)) for a in range(3)],
        "stored_cells": [int(grid.stored_cells(a)) for a in range(3)],
        "owned_cells": [int(grid.owned_cells(a)) for a in range(3)],
        "boundary_kinds": list(coverage.real_pml_boundary_kinds(grid)),
    }


# ---------------------------------------------------------------------------
# One curl case
# ---------------------------------------------------------------------------

def one_case(backend: str, spec: Dict[str, Any], sub_step: str, courant: float,
             value_class: str, guard: str,
             host_mutation: Optional[str] = None) -> Dict[str, Any]:
    """One frozen state, run twice: the array path, then the kernel."""
    started = time.time()
    xp = cp if backend == "cuda" else _NumpyWearingCupysName()
    rng = np.random.default_rng(
        SEED + int.from_bytes(hashlib.sha256(
            f"{spec['label']}|{sub_step}|{courant}|{value_class}".encode()).digest()[:4], "big"))

    fields, layer, grid = build(xp, spec, courant)
    seed_state(fields, grid, state_names(sub_step), value_class, rng)
    dtdx = float(grid.dt / grid.dx)
    codes, kinds = boundary_codes_for(grid)

    case: Dict[str, Any] = {
        "label": spec["label"], "sub_step": sub_step, "courant": courant,
        "value_class": value_class, "guard": guard, "backend": backend,
        "host_mutation": host_mutation, "fold_axes": spec["axes"],
        "boundaries": list(spec["boundaries"]),
        "boundary_codes": [int(c) for c in codes], "resolved_kinds": list(kinds),
        "dtdx": dtdx, "structure": structure_facts(grid),
    }
    covered, reason = family.covers_special_kz_curl(fields, layer, grid, sub_step)
    case["predicate"] = {"covered": bool(covered), "reason": reason}
    certified, certified_reason = coverage.covers_real_pml_curl(
        fields, layer, grid, sub_step)
    case["certified_predicate"] = {"covered": bool(certified),
                                   "reason": certified_reason}

    absorbs = absorber_is_not_the_identity(sub_step, layer)
    case["absorber_is_not_the_identity"] = absorbs
    if not absorbs["meets_floor"]:
        case["skipped"] = ("every absorber coefficient is the identity; a "
                           "coefficient-index error would be invisible here")
        case["seconds"] = time.time() - started
        return case

    frozen = snapshot(fields, state_names(sub_step))
    case["operand_census"] = operand_census(frozen)

    # Leg 1: the oracle.
    if sub_step == "step_B":
        stepping.step_B(fields, layer)
    else:
        stepping.step_D(fields, layer)
    reference = snapshot(fields, state_names(sub_step))

    moved = moved_fraction(frozen, reference, outputs(sub_step))
    case["oracle_moved"] = moved
    if moved == 0.0:
        case["skipped"] = ("the array path changed no output word from the frozen "
                           "input; a case that moved nothing certifies nothing")
        case["seconds"] = time.time() - started
        return case

    live = beta_term_is_live(fields, layer, grid, sub_step, frozen, reference)
    case["beta_term_is_live"] = live
    if not live["meets_floor"]:
        case["skipped"] = ("the beta term moved no output word, or moved the wrong "
                           "components: this case measures the certified kernel, "
                           "not this family")
        case["seconds"] = time.time() - started
        return case

    # Leg 2: the kernel, from the SAME frozen state.
    restore(fields, frozen)
    tables, beta = host_mutated_inputs(host_mutation, sub_step, layer, grid, dtdx)
    case["beta_coefficients"] = [float(beta[0]), float(beta[1])]
    runner = run_kernel_cuda if backend == "cuda" else run_kernel_numpy
    runner(sub_step, fields, tables, codes, dtdx, beta)

    parts = {name: bit_compare(reference[name], getattr(fields, name))
             for name in outputs(sub_step)}
    case["single_launch"] = combine(parts)
    case["single_launch_per_array"] = {
        name: parts[name]["differing_floats"] for name in outputs(sub_step)}

    # Leg 3: the multi-step. ``fu`` IS STATE.
    if host_mutation is None:
        restore(fields, frozen)
        for _ in range(MULTI_STEP_BUDGET):
            if sub_step == "step_B":
                stepping.step_B(fields, layer)
            else:
                stepping.step_D(fields, layer)
            advance_sources(fields, SUB_STEP_ARRAYS[sub_step]["sources"])
        oracle_multi = snapshot(fields, state_names(sub_step))

        restore(fields, frozen)
        for _ in range(MULTI_STEP_BUDGET):
            runner(sub_step, fields, tables, codes, dtdx, beta)
            advance_sources(fields, SUB_STEP_ARRAYS[sub_step]["sources"])
        multi = {name: bit_compare(oracle_multi[name], getattr(fields, name))
                 for name in outputs(sub_step)}
        case["multi_step"] = combine(multi)
        case["multi_step"]["launches"] = MULTI_STEP_BUDGET

    case["seconds"] = time.time() - started
    return case


# ---------------------------------------------------------------------------
# The constitutive arm
# ---------------------------------------------------------------------------

def one_constitutive_case(backend: str, spec: Dict[str, Any], side: str,
                          courant: float, value_class: str,
                          guard: str) -> Dict[str, Any]:
    """The CERTIFIED constitutive kernel against the array path, on a BETA run.

    THE CURLS RUN FIRST, both of them, so the state this sub-step reads is a state
    the beta term produced. A constitutive arm seeded with fresh random numbers
    would measure the certified kernel on a grid that merely HAS a beta attribute;
    this measures it on a beta run's own history.
    """
    started = time.time()
    if backend != "cuda":
        return {"label": spec["label"], "side": side, "courant": courant,
                "value_class": value_class, "backend": backend,
                "skipped": ("the constitutive arm runs the CERTIFIED CUDA kernel; "
                            "there is no NumPy transcription of it in this gate "
                            "and inventing one would compare two transcriptions")}
    from meep_gpu.cuda_kernels import constitutive_kernels  # noqa: PLC0415

    rng = np.random.default_rng(
        SEED + 977 + int.from_bytes(hashlib.sha256(
            f"{spec['label']}|{side}|{courant}|{value_class}".encode()).digest()[:4], "big"))
    fields, layer, grid = build(cp, spec, courant)
    every = tuple(dict.fromkeys(
        state_names("step_B") + state_names("step_D")
        + constitutive_state_names("H") + constitutive_state_names("E")))
    seed_state(fields, grid, every, value_class, rng)

    case: Dict[str, Any] = {
        "label": spec["label"], "side": side, "courant": courant,
        "value_class": value_class, "guard": guard, "backend": backend,
        "structure": structure_facts(grid),
    }
    covered, reason = family.covers_special_kz_constitutive(fields, layer, grid, side)
    case["predicate"] = {"covered": bool(covered), "reason": reason}

    # THE BETA RUN'S OWN HISTORY: both curls, on the array path, before anything
    # is frozen. The constitutive sub-step then reads what beta produced.
    stepping.step_B(fields, layer)
    stepping.step_D(fields, layer)

    frozen = snapshot(fields, constitutive_state_names(side))
    if side == "H":
        stepping.update_H(fields, layer)
    else:
        stepping.update_E(fields, layer)
    reference = snapshot(fields, constitutive_state_names(side))
    moved = moved_fraction(frozen, reference, constitutive_outputs(side))
    case["oracle_moved"] = moved
    if moved == 0.0:
        case["skipped"] = ("the array path changed no output word; a constitutive "
                           "case that moved nothing certifies nothing")
        case["seconds"] = time.time() - started
        return case

    restore(fields, frozen)
    constitutive_kernels.update_fused_pml_real(side, fields, layer)
    cp.cuda.runtime.deviceSynchronize()
    parts = {name: bit_compare(reference[name], getattr(fields, name))
             for name in constitutive_outputs(side)}
    case["single_launch"] = combine(parts)

    # THE MULTI-STEP LEG, because ``f_w`` is state on this sub-step too.
    restore(fields, frozen)
    for _ in range(MULTI_STEP_BUDGET):
        if side == "H":
            stepping.update_H(fields, layer)
        else:
            stepping.update_E(fields, layer)
        advance_sources(fields, CONSTITUTIVE_ARRAYS[side]["sources"])
    oracle_multi = snapshot(fields, constitutive_state_names(side))
    restore(fields, frozen)
    for _ in range(MULTI_STEP_BUDGET):
        constitutive_kernels.update_fused_pml_real(side, fields, layer)
        advance_sources(fields, CONSTITUTIVE_ARRAYS[side]["sources"])
    cp.cuda.runtime.deviceSynchronize()
    multi = {name: bit_compare(oracle_multi[name], getattr(fields, name))
             for name in constitutive_outputs(side)}
    case["multi_step"] = combine(multi)
    case["multi_step"]["launches"] = MULTI_STEP_BUDGET
    case["seconds"] = time.time() - started
    return case


# ---------------------------------------------------------------------------
# The sweeps
# ---------------------------------------------------------------------------

def case_product(product: str):
    specs = BETA_SPECS if product == "full" else BETA_SPECS[:3]
    courants = COURANTS if product == "full" else (INEXACT_COURANT,)
    classes = VALUE_CLASSES if product == "full" else ("uniform",)
    return [(spec, sub_step, courant, value_class)
            for spec in specs for sub_step in SUB_STEPS
            for courant in courants for value_class in classes]


def run_sweep(results, out_path, backend, product, guard):
    cases = []
    plan = case_product(product)
    for index, (spec, sub_step, courant, value_class) in enumerate(plan, start=1):
        case = one_case(backend, spec, sub_step, courant, value_class, guard)
        cases.append(case)
        results.setdefault("sweep", {})[guard] = cases
        save(results, out_path)
        if case.get("skipped"):
            log(f"[{guard}] case {index}/{len(plan)} {spec['label']} {sub_step} "
                f"SKIPPED: {case['skipped'][:70]}")
            continue
        single = case["single_launch"]["bit_identical"]
        multi = case.get("multi_step", {}).get("bit_identical")
        log(f"[{guard}] case {index}/{len(plan)} {spec['label']} {sub_step} "
            f"c={courant} {value_class} shape={case['structure']['shape']} "
            f"single={'IDENTICAL' if single else 'DIVERGED'} "
            f"multi={'IDENTICAL' if multi else ('DIVERGED' if multi is False else '-')} "
            f"diff={case['single_launch']['differing_floats']} "
            f"beta_moved={case['beta_term_is_live']['total']} "
            f"({case['seconds']:.1f} s)")
    return cases


def run_constitutive(results, out_path, backend, product, guard):
    cases = []
    specs = BETA_SPECS if product == "full" else BETA_SPECS[:3]
    courants = COURANTS if product == "full" else (INEXACT_COURANT,)
    classes = VALUE_CLASSES if product == "full" else ("uniform",)
    plan = [(spec, side, courant, value_class)
            for spec in specs for side in SIDES
            for courant in courants for value_class in classes]
    for index, (spec, side, courant, value_class) in enumerate(plan, start=1):
        case = one_constitutive_case(backend, spec, side, courant, value_class, guard)
        cases.append(case)
        results.setdefault("constitutive", {})[guard] = cases
        save(results, out_path)
        if case.get("skipped"):
            log(f"[const {guard}] {index}/{len(plan)} {spec['label']} update_{side} "
                f"SKIPPED: {case['skipped'][:70]}")
            continue
        log(f"[const {guard}] {index}/{len(plan)} {spec['label']} update_{side} "
            f"c={courant} {value_class} "
            f"single={'IDENTICAL' if case['single_launch']['bit_identical'] else 'DIVERGED'} "
            f"multi={'IDENTICAL' if case['multi_step']['bit_identical'] else 'DIVERGED'} "
            f"diff={case['single_launch']['differing_floats']} "
            f"({case['seconds']:.1f} s)")
    return cases


MUTATION_SPEC_LABELS: Tuple[str, ...] = (
    "plain_periodic", "plain_metallic_xy", "fold_Y_periodic", "fold_Y_metallic",
    "fold_X_periodic", "plain_periodic_negative_beta",
)


def spec_codes(spec: Dict[str, Any]) -> Tuple[int, int, int]:
    _f, _l, grid = build(_NumpyWearingCupysName(), spec, COURANTS[0])
    codes, refusal = coverage.real_curl_boundary_codes(grid)
    if refusal is not None:
        raise ValueError(f"the shipped resolver refuses fixture {spec['label']!r}: "
                         f"{refusal}")
    return tuple(int(c) for c in codes)


def scorable_baselines(results) -> Tuple[set, List[Dict[str, Any]]]:
    """``(label, sub_step)`` pairs whose UNMUTATED case was bit-identical.

    A mutation leg only means something where the baseline is identical: on an arm
    that already diverges, every leg -- including one that must be UNCAUGHT --
    scores CAUGHT for free.
    """
    baseline = {}
    for case in results.get("sweep", {}).get("fmad_false", []):
        if (case.get("courant") != INEXACT_COURANT
                or case.get("value_class") != "uniform"):
            continue
        baseline[(case["label"], case["sub_step"])] = case
    scorable, excluded = set(), []
    for key, case in baseline.items():
        if case.get("skipped"):
            excluded.append({"label": key[0], "sub_step": key[1],
                             "why": f"baseline skipped: {case['skipped'][:80]}"})
            continue
        if (case["single_launch"]["bit_identical"]
                and case.get("multi_step", {}).get("bit_identical", True)):
            scorable.add(key)
        else:
            excluded.append({"label": key[0], "sub_step": key[1],
                             "why": "the unmutated baseline already diverges here",
                             "differing_words":
                                 case["single_launch"]["differing_floats"]})
    return scorable, excluded


def mutation_plan(product: str, scorable: Optional[set] = None):
    by_label = {spec["label"]: spec for spec in BETA_SPECS}
    labels = MUTATION_SPEC_LABELS if product == "full" else MUTATION_SPEC_LABELS[:3]
    plan = [(by_label[label], sub_step, INEXACT_COURANT, "uniform")
            for label in labels for sub_step in SUB_STEPS]
    if scorable is None:
        return plan
    return [e for e in plan if (e[0]["label"], e[1]) in scorable]


def leg_caught(case: Dict[str, Any]) -> bool:
    if case.get("skipped"):
        return False
    if not case["single_launch"]["bit_identical"]:
        return True
    multi = case.get("multi_step")
    return bool(multi is not None and not multi["bit_identical"])


def _score(legs, must_be_caught: bool) -> Dict[str, Any]:
    scored = [c for c in legs if not c.get("skipped")]
    caught = sum(1 for c in scored if leg_caught(c))
    if not scored:
        verdict = "NO LEGS"
    elif must_be_caught:
        verdict = ("CAUGHT" if caught == len(scored)
                   else "PARTIAL" if caught else "UNCAUGHT")
    else:
        verdict = "NULL CONFIRMED" if caught == 0 else "NULL VIOLATED"
    return {"ran": len(scored), "caught": caught, "uncaught": len(scored) - caught,
            "must_be_caught": must_be_caught, "verdict": verdict}


def run_host_mutations(results, out_path, backend, product, scorable):
    out: Dict[str, Any] = {}
    plan = mutation_plan(product, scorable)
    for name in HOST_MUTATIONS:
        legs = [one_case(backend, spec, sub_step, courant, value_class,
                         "fmad_false", host_mutation=name)
                for spec, sub_step, courant, value_class in plan]
        out[name] = dict(_score(legs, name not in HOST_NULL_MUTATIONS), cases=legs)
        log(f"[host-mut] {name}: caught {out[name]['caught']}/{out[name]['ran']} "
            f"-> {out[name]['verdict']}")
        results["host_mutations"] = out
        save(results, out_path)
    return out


def _grid_carries_a_mask(codes: Tuple[int, int, int]) -> bool:
    return any(code in (BC_METALLIC, BC_MIRROR_PERIODIC) for code in codes)


def run_source_mutations(results, out_path, product, scorable):
    """Every device-text defect, applied to the SHIPPED strings and recompiled."""
    from meep_gpu.cuda_kernels import compile_cache  # noqa: PLC0415

    originals = {name: family.kernel_source(name)
                 for name in family.SPECIAL_KZ_KERNELS}
    transforms = dict(BETA_SOURCE_MUTATIONS)
    for name in SHARED_SOURCE_MUTATIONS:
        transforms[name] = probe.SOURCE_MUTATIONS[name]
    nulls = set(BETA_NULL_MUTATIONS) | set(SHARED_NULL_MUTATIONS)
    out: Dict[str, Any] = {}
    plan = mutation_plan(product, scorable)

    try:
        for sub_step in SUB_STEPS:
            kernel_name = family.KERNEL_FOR_SUB_STEP[sub_step]
            for name, transform in transforms.items():
                mutated, sites = transform(originals[kernel_name])
                key = f"{sub_step}:{name}"
                required = SHARED_REQUIRES_CODE.get(name)
                leg_plan = [e for e in plan if e[1] == sub_step]
                if required is not None:
                    leg_plan = [e for e in leg_plan
                                if required in spec_codes(e[0])]
                if SOURCE_MUTATION_REQUIRES_MASK.get(name):
                    leg_plan = [e for e in leg_plan
                                if _grid_carries_a_mask(spec_codes(e[0]))]
                if sites == 0 or mutated == originals[kernel_name]:
                    out[key] = {"armed": False,
                                "why": (f"matched {sites} site(s) and changed "
                                        f"nothing; the mutation and the kernel "
                                        f"have drifted apart")}
                    log(f"[src-mut] {key}: NOT ARMED ({sites} sites)")
                    results["source_mutations"] = out
                    save(results, out_path)
                    continue
                family.set_kernel_source(kernel_name, mutated)
                family._clear_kernel_cache()
                compile_cache.clear_compile_log()
                digest = hashlib.sha256(mutated.encode("utf-8")).hexdigest()
                legs = [one_case("cuda", spec, sub_step, courant, value_class,
                                 "fmad_false")
                        for spec, _s, courant, value_class in leg_plan]
                family.set_kernel_source(kernel_name, originals[kernel_name])
                family._clear_kernel_cache()
                from_mutated = sum(1 for entry in compile_cache.compile_log()
                                   if entry["source_sha256"] == digest)
                scored = _score(legs, name not in nulls)
                out[key] = dict(scored, armed=True, sites=sites,
                                requires_code=required,
                                legs_in_plan=len([e for e in plan
                                                  if e[1] == sub_step]),
                                legs_scored=len(leg_plan),
                                mutated_source_sha256=digest,
                                kernel_constructions_from_mutated_bytes=from_mutated,
                                cases=legs)
                if from_mutated == 0 and scored["ran"]:
                    out[key]["verdict"] = "UNACCOUNTED"
                    out[key]["why"] = ("no kernel construction used the mutated "
                                       "bytes; this leg did not exercise the "
                                       "mutation")
                log(f"[src-mut] {key}: caught {scored['caught']}/{scored['ran']} "
                    f"builds_from_mutated={from_mutated} -> {out[key]['verdict']}")
                results["source_mutations"] = out
                save(results, out_path)
    finally:
        for name in family.SPECIAL_KZ_KERNELS:
            family.set_kernel_source(name, originals[name])
        family._clear_kernel_cache()
    return out


# ---------------------------------------------------------------------------
# The verdict
# ---------------------------------------------------------------------------

def summarize(results) -> Dict[str, Any]:
    sweep = results.get("sweep", {})
    primary = sweep.get("fmad_false", [])
    scored = [c for c in primary if not c.get("skipped")]
    reasons: List[str] = []

    arms: Dict[str, Dict[str, Any]] = {}
    for case in scored:
        key = "folded" if case["fold_axes"] else "unfolded"
        entry = arms.setdefault(key, {"cases": 0, "single_identical": 0,
                                      "multi_cases": 0, "multi_identical": 0,
                                      "differing_words": 0, "sub_steps": set(),
                                      "labels": set()})
        entry["cases"] += 1
        entry["sub_steps"].add(case["sub_step"])
        entry["labels"].add(case["label"])
        entry["single_identical"] += bool(case["single_launch"]["bit_identical"])
        if "multi_step" in case:
            entry["multi_cases"] += 1
            entry["multi_identical"] += bool(case["multi_step"]["bit_identical"])
        entry["differing_words"] += case["single_launch"]["differing_floats"]
    for entry in arms.values():
        entry["sub_steps"] = sorted(entry["sub_steps"])
        entry["labels"] = sorted(entry["labels"])
        entry["all_identical"] = (entry["single_identical"] == entry["cases"]
                                  and entry["multi_identical"] == entry["multi_cases"])

    if not scored:
        reasons.append("no curl case was scored at all")
    for arm in ("unfolded", "folded"):
        if arm not in arms:
            reasons.append(f"the sweep contained no {arm} case; the predicate "
                           f"admits both, so releasing on one is releasing on "
                           f"half the admitted set")
            continue
        entry = arms[arm]
        if entry["single_identical"] != entry["cases"]:
            reasons.append(f"{arm}: single-launch divergence on "
                           f"{entry['cases'] - entry['single_identical']} of "
                           f"{entry['cases']} cases")
        if entry["multi_identical"] != entry["multi_cases"]:
            reasons.append(f"{arm}: multi-step divergence on "
                           f"{entry['multi_cases'] - entry['multi_identical']} of "
                           f"{entry['multi_cases']} cases")
        if sorted(entry["sub_steps"]) != sorted(SUB_STEPS):
            reasons.append(f"{arm}: only sub-steps {entry['sub_steps']} were scored")

    terms = results.get("curl_terms_vs_stepping", {})
    if not terms.get("agreed", False):
        reasons.append("the transcribed curl/beta term table disagrees with "
                       "stepping's")

    # THE PREDICATE MUST AGREE WITH THE SWEEP. Every case swept is one the
    # predicate admits, and every one is a case the CERTIFIED predicate refuses --
    # so an overlap between the two families would show here as a case both admit.
    for case in scored:
        if not case["predicate"]["covered"]:
            reasons.append(f"{case['label']} {case['sub_step']}: the shipped "
                           f"predicate REFUSES a case this gate scored "
                           f"({case['predicate']['reason'][:80]})")
        if case["certified_predicate"]["covered"]:
            reasons.append(f"{case['label']} {case['sub_step']}: the CERTIFIED "
                           f"predicate also admits this case; the two families "
                           f"are meant to partition on beta")

    # THE CONSTITUTIVE ARM, scored PER SIDE.
    constitutive = results.get("constitutive", {}).get("fmad_false", [])
    per_side: Dict[str, Dict[str, int]] = {}
    for case in constitutive:
        entry = per_side.setdefault(case["side"], {"cases": 0, "single": 0,
                                                   "multi_cases": 0, "multi": 0,
                                                   "skipped": 0})
        if case.get("skipped"):
            entry["skipped"] += 1
            continue
        entry["cases"] += 1
        entry["single"] += bool(case["single_launch"]["bit_identical"])
        entry["multi_cases"] += 1
        entry["multi"] += bool(case["multi_step"]["bit_identical"])
    for side in SIDES:
        entry = per_side.get(side)
        if entry is None or entry["cases"] == 0:
            reasons.append(f"update_{side} was never scored on a beta run; the "
                           f"constitutive admission would rest on the reading "
                           f"alone, which is what the Metal tranche-6 split warns "
                           f"against")
            continue
        if entry["single"] != entry["cases"] or entry["multi"] != entry["multi_cases"]:
            reasons.append(f"update_{side}: divergence on "
                           f"{entry['cases'] - entry['single']} of "
                           f"{entry['cases']} single-launch cases and "
                           f"{entry['multi_cases'] - entry['multi']} of "
                           f"{entry['multi_cases']} multi-step cases")

    for name, leg in results.get("host_mutations", {}).items():
        if leg["verdict"] == "NO LEGS":
            reasons.append(f"host mutation {name} was never scored; unasked is "
                           f"not inert")
        elif leg["verdict"] not in ("CAUGHT", "NULL CONFIRMED"):
            reasons.append(f"host mutation {name} is {leg['verdict']} "
                           f"({leg['caught']}/{leg['ran']})")
    for key, leg in results.get("source_mutations", {}).items():
        if not leg.get("armed"):
            reasons.append(f"source mutation {key} was not armed: {leg.get('why')}")
        elif leg["verdict"] not in ("CAUGHT", "NULL CONFIRMED"):
            reasons.append(f"source mutation {key} is {leg['verdict']}")

    control = [c for c in sweep.get("default_no_options", [])
               if not c.get("skipped") and c["courant"] == INEXACT_COURANT]
    guarded_identical = {(c["label"], c["sub_step"], c["courant"], c["value_class"])
                         for c in scored if c["single_launch"]["bit_identical"]}
    comparable = [c for c in control
                  if (c["label"], c["sub_step"], c["courant"], c["value_class"])
                  in guarded_identical]
    control_diverged = [c for c in comparable
                        if not c["single_launch"]["bit_identical"]]
    guard_control = {
        "scored_at_inexact_courant": len(control),
        "comparable_where_guarded_leg_was_identical": len(comparable),
        "diverged": len(control_diverged),
        "reading": ("NOT MEASURED on this run" if not comparable else
                    "the contraction guard is load-bearing on this pair"
                    if control_diverged else
                    "MEASURED DECORATIVE on these cases: the unguarded leg was "
                    "bit-identical too, so this run carries no evidence that "
                    "--fmad=false changed an answer here"),
    }

    return {
        "released": not reasons,
        "reasons": reasons,
        "arms": arms,
        "constitutive_per_side": per_side,
        "guard_control": guard_control,
        "scored_cases": len(scored),
        "claim": ("on a real 2-D Cartesian grid with grid.beta != 0 and real "
                  "float32 storage under an active PML, "
                  "cuda_kernels/special_kz_curl.py's pair is byte-identical to "
                  "stepping.step_B / step_D per sub-step at one launch and at 60, "
                  "unfolded and on a mirror fold at both terminations, at an "
                  "exactly representable courant and at one that is not, on "
                  "physical-band and subnormal-band operands; and the CERTIFIED "
                  "constitutive pair is byte-identical to stepping.update_H / "
                  "update_E on the same beta runs, scored per side"),
        "does_not_claim": [
            "nothing dispatches these kernels; this gate licenses a predicate "
            "clause, not a wiring",
            "COMPLEX storage with beta is a different kernel and is untouched: "
            "four of the six beta corpus rows are complex and none is reached",
            "every beta grid has nz == 1 by construction (Grid refuses a z extent "
            "on a dimensions=2 grid), so the z-axis ghost rule and z wall mask are "
            "exercised at a single cell only -- they are the certified pair's and "
            "are gated elsewhere on grids thick in z",
            "off-diagonal epsilon with beta in real storage is refused by name "
            "(stepping raises, MEEP aborts) and is not measured",
            "a conductivity, BFAST, a Bloch phase, cylindrical coordinates and "
            "three simultaneous fold planes are refused by inherited clauses and "
            "untested here",
            "no throughput claim: this is a correctness gate and times nothing",
        ],
    }


def save(results, out_path: str) -> None:
    gate_provenance.stamp(results)
    tmp = out_path + ".tmp"
    with open(tmp, "w") as handle:
        json.dump(results, handle, indent=2, sort_keys=False, default=str)
    os.replace(tmp, out_path)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backend", choices=("cuda", "numpy"), default="cuda")
    parser.add_argument("--product", choices=("full", "reduced"), default="full")
    parser.add_argument("--subnormal-policy", choices=("keep", "flush"), default=None)
    parser.add_argument("--import-meep-for-host-policy", action="store_true")
    parser.add_argument("--out", required=True)
    parser.add_argument("--skip-mutations", action="store_true")
    args = parser.parse_args(argv)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    results: Dict[str, Any] = {
        "gate": "cuda_special_kz",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": os.uname().nodename,
        "argv": list(argv if argv is not None else sys.argv[1:]),
        "backend": args.backend,
        "product": args.product,
        "question": ("is the hand-CUDA special_kz curl pair byte-identical to "
                     "stepping.step_B / step_D on a beta run, and is the CERTIFIED "
                     "constitutive pair byte-identical to update_H / update_E on "
                     "the same run, per sub-step and per side?"),
        "curl_terms_vs_stepping": check_terms_against_stepping(),
    }

    if args.backend == "cuda":
        if cp is None:
            log("[fatal] --backend cuda but CuPy did not import")
            results["status"] = "refused: no CuPy"
            save(results, args.out)
            return 2
        if args.import_meep_for_host_policy:
            results["meep_host_import"] = probe.import_meep_for_host_policy()
        observer = probe.install_nvrtc_binary_observer()
        results["nvrtc_observer"] = observer
        if args.subnormal_policy:
            results["subnormal_policy_install"] = \
                probe.install_subnormal_policy_for_run(args.subnormal_policy, _REPO_API)
        results["environment"] = probe.device_info()
        results["subnormal_policy_stamp"] = probe.subnormal_policy_stamp(_REPO_API)
    else:
        results["environment"] = {"python": sys.version.split()[0],
                                  "numpy_version": np.__version__,
                                  "note": ("NumPy backend: compiles nothing, "
                                           "certifies nothing")}
    save(results, args.out)

    for guard, options, _primary in GUARD_SETS:
        if args.backend == "numpy" and guard != "fmad_false":
            continue
        if args.backend == "cuda":
            from meep_gpu.cuda_kernels import constitutive_kernels  # noqa: PLC0415
            family._COMPILE_OPTIONS = tuple(options)
            family._clear_kernel_cache()
            constitutive_kernels._COMPILE_OPTIONS = tuple(options)
            constitutive_kernels._clear_kernel_cache()
        log(f"[guard] {guard} options={options}")
        run_sweep(results, args.out, args.backend, args.product, guard)
        run_constitutive(results, args.out, args.backend, args.product, guard)

    if args.backend == "cuda":
        from meep_gpu.cuda_kernels import constitutive_kernels  # noqa: PLC0415
        family._COMPILE_OPTIONS = ("--fmad=false",)
        family._clear_kernel_cache()
        constitutive_kernels._COMPILE_OPTIONS = ("--fmad=false",)
        constitutive_kernels._clear_kernel_cache()

    if not args.skip_mutations:
        scorable, excluded = scorable_baselines(results)
        results["mutation_scope"] = {
            "scorable": sorted("::".join(k) for k in scorable),
            "excluded": excluded,
            "why": ("a mutation leg answers 'can this gate see this defect', and "
                    "only an arm whose unmutated baseline is bit-identical can "
                    "answer it"),
        }
        log(f"[mut-scope] {len(scorable)} scorable pairs; {len(excluded)} excluded")
        save(results, args.out)
        run_host_mutations(results, args.out, args.backend, args.product, scorable)
        if args.backend == "cuda":
            run_source_mutations(results, args.out, args.product, scorable)

    if args.backend == "cuda":
        results["nvrtc_binary_report"] = probe.nvrtc_binary_report()
    results["summary"] = summarize(results)
    results["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save(results, args.out)

    verdict = results["summary"]
    log(f"[verdict] released={verdict['released']} scored={verdict['scored_cases']}")
    for name, arm in sorted(verdict["arms"].items()):
        log(f"[verdict]   {name}: single {arm['single_identical']}/{arm['cases']} "
            f"multi {arm['multi_identical']}/{arm['multi_cases']} "
            f"diff={arm['differing_words']}")
    for side, entry in sorted(verdict["constitutive_per_side"].items()):
        log(f"[verdict]   update_{side}: single {entry['single']}/{entry['cases']} "
            f"multi {entry['multi']}/{entry['multi_cases']} "
            f"skipped {entry['skipped']}")
    for reason in verdict["reasons"]:
        log(f"[verdict]   - {reason}")
    return 0 if verdict["released"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
