"""Gate: ``meep_gpu.cuda_kernels.complex_beta_fused_hd_pair`` -- the COMPLEX beta H->D weld.

ONE CUDA kernel welding ``stepping.update_H`` into the special_kz ``stepping.step_D``
under COMPLEX64 storage on a ``grid.beta != 0`` run, measured BIT-IDENTICAL as uint32
words per COMPLETE DRIVER STEP against FOUR independent arrangements of the seam,
under both float32 subnormal policies, on synthetic fixtures and on the four lifted
corpus rows of its cell.

THE CELL, AND WHY IT IS ONE TEXT
=============================================================================
``results/fusion_matrix_cuda_2026-09-07_wired/fusion_matrix_cuda.json``,
``h_to_d_seam.instances`` filtered to ``buildable_not_built``:

* ``(cuda_complex_beta/complex beta -> cuda_complex_beta/complex beta)`` -- **4
  instances**: ``tests:TestSpecialKz.test_special_kz``,
  ``tests:TestSpecialKz.test_eigsrc_kz_0_complex`` and the two
  ``tests:TestEigCoeffs.test_binary_grating_special_kz_*`` legs.

Three of the four are folded (Mirror(Y), periodic termination) and one is not, and
ONE TEXT serves all four: ``complex_beta_kernels``' device source is the FOLDED
complex curl with the beta insert, so a fold is a runtime boundary code here. The
released complex H->D product needed two variants; this one needs none, and the
fixtures below still carry both folded and unfolded grids because the RUNTIME CODES
differ even where the text does not.

THE FOUR REFERENCES, and what each one rules out
=============================================================================
``array``               ``stepping``'s own ten passes. THE ORACLE.
``singles``             the certified ``step_D_pml_complex_beta`` and the certified
                        complex constitutive kernel at their own slots, everything
                        else on the array path. Rules out a divergence that is the
                        certified halves' rather than the weld's.
``composition_today``   what the composer BUILDS on these rows today: the released
                        ``cuda_complex_beta_fused_magnetic_pair`` at ``step_B`` and
                        ``cuda_complex_beta_fused_electric_pair`` at ``step_D``.
                        Rules out "identical to a composition nobody runs".
``unfused``             all four slot-path slots as certified singles.
``weld``                the subject.

Plus ``weld_composed`` on the launch-structure leg only.

WHAT THIS SEAM ADDS THAT NO SIBLING H->D GATE CAN MEASURE
=============================================================================
THE BETA PARTNER. ``stepping._special_kz_beta_term`` consumes the SAME-CELL magnetic
snapshot, which the array path takes AFTER ``update_H``; welded, that snapshot must be
this launch's own ``own_h[...]`` register. ``beta_partner_reads_stale_h`` is the armed
defect and it puts target 0's beta pass one sub-step behind and nothing else.

WHAT THE ARITHMETIC LEGS ADD THAT A WHOLE-STEP IDENTITY CANNOT
=============================================================================
The family's real-by-complex product is PART-WISE (``part_products.scale``: one float32
product per part). The full complex product
against ``(c + 0j)`` the family carried before differs from it on the SIGN OF A ZERO
only -- a class a random battery does not reach. The ``spelling`` leg is the released
complex H->D gate's (``gate_cuda_complex_fused_hd_pair.leg_spelling``: the family's
multiply against the array path's product on a signed-zero class beside the ordinary
ones), and the mutation battery arms the old full product as a device defect on a
PLANTED row-0 tiny-normal fixture, where it bites under ``flush``.

**THIS FAMILY PERFORMS NO DIVISION** and the ``spelling`` leg records that as a
measured absence rather than omitting it: the reciprocal the recurrence needs is
``sinv``, computed host-side by ``PML``. The absence is asserted against the emitted
text, with a planted divide as the control.

THE BETA COEFFICIENT IS ONE REAL WORD PER CALL SITE. The beta pass forms
``-quarter_turn(c, g)`` part by part from a host-rounded float32 coefficient, so no
complex coefficient -- and no signed-zero real word -- exists anywhere to be right
about.

NOT WIRED, AND THE ARBITRATION LEG SAYS SO IN BOTH ARRANGEMENTS
=============================================================================
The family is not in ``fused_pairs.FUSED_PRODUCTS`` / ``registry`` / ``arms`` yet.
The ``arbitration`` leg measures the composer AS SHIPPED and again WITH the wiring
rows patched in-process, and it ASSERTS the substantive claims only -- refused BY
NAME, the refusal names ``INSTALLABLE = False``, the selection unchanged, both
released neighbours keeping their slots -- while RECORDING the positional ones, which
a wiring round inverts by design.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_API = str(next(parent for parent in Path(_HERE).parents
                     if (parent / "meep_gpu" / "cuda_kernels").is_dir()))
for _path in (_REPO_API, _HERE):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import beta_hd_gate_common as kit  # noqa: E402
# The released complex H->D gate: this weld splices the same constitutive half and the
# same curl transform, so its weld-clause mutations, its record mutation and its
# spelling leg are imported rather than respelled.
import gate_cuda_complex_fused_hd_pair as cx_hd  # noqa: E402

cp = kit.cp
log = kit.log
differing = kit.differing
to_host = kit.to_host

FAMILY_NAME = "cuda_complex_beta_fused_hd_pair"
BLOCK_KEY = "cuda_complex_beta_fused_hd_pair_gate"
GATE_FILE = "gate_complex_beta_fused_hd_pair.py"

#: The board cell, in the census's ``(update_H arm, step_D arm)`` spelling.
CELL_ARMS: Tuple[str, str] = ("cuda_complex_beta/complex beta",
                              "cuda_complex_beta/complex beta")

#: The betas of the four corpus rows this cell holds, from
#: ``complex_beta_kernels``' own admission: ``test_special_kz`` -0.3907,
#: ``test_eigsrc_kz_0_complex`` 0.2, and the two binary-grating legs -0.685 / -0.912.
#: Fixtured across BOTH SIGNS, because the coefficient's sign is what the swapped
#: call-site words corrupt.
CORPUS_BETAS: Tuple[float, ...] = (-0.3907, 0.2, -0.685, -0.912)

#: THE PLANT. The part-wise product and the old full complex product against
#: ``(c + 0j)`` differ only where a part the multiply sees is a SIGNED ZERO, which on
#: the battery's data means a product that underflows: the sign of a flushed zero. The
#: class is ``|re|`` and ``|im|`` tiny NORMALS with both parts negative, so ``z.im * c``
#: with a small coefficient falls into the subnormal range and ``flush`` zeroes it
#: while ``keep`` does not. Planted on row 0 of EVERY stored volume, because
#: ``mul_field_left`` -- the helper whose spelling the defect changes -- is called by
#: ``pml_apply`` on ``D``, ``fu_D`` and the curl, NOT on ``H`` or ``B``.
PLANT_MAGNITUDE = np.float32(1.5e-38)

#: The synthetic fixtures. Both fold terminations, both full-count parities, a fully
#: periodic grid (the ONLY specialisation on which the shifted tap's PERIODIC WRAP is
#: observable), a live Bloch phase on an UNFOLDED axis, and both signs of beta.
#:
#: A PHASE ON A FOLDED AXIS IS NOT FIXTURED. ``complex_folded_kernels._fold_reasons``
#: refuses that configuration by name and ``complex_beta_kernels`` inherits the
#: refusal; the folded fixtures below phase unfolded axes only, which is what the four
#: corpus rows carry. The invariant z axis is forced periodic by ``Grid`` at
#: ``dimensions=2``, so no fixture walls, folds or phases it.
SPECS: Tuple[Dict[str, Any], ...] = (
    {"label": "periodic_k0", "cell": (1.6, 1.7, 0.0), "beta": CORPUS_BETAS[0],
     "boundaries": ("periodic", "periodic", "periodic"), "symmetry": (),
     "k_point": (0.0, 0.0, 0.0)},
    {"label": "periodic_kx", "cell": (1.6, 1.7, 0.0), "beta": CORPUS_BETAS[1],
     "boundaries": ("periodic", "periodic", "periodic"), "symmetry": (),
     "k_point": (0.3, 0.0, 0.0)},
    # THE BRILLOUIN EDGE, where ``Grid._axis_bloch_phase`` returns EXACTLY -1+0j: the
    # rotation is then a sign flip with an exact zero imaginary part, which is where a
    # dropped zero cross term hides.
    {"label": "brillouin_edge_kx", "cell": (1.6, 1.7, 0.0), "beta": CORPUS_BETAS[2],
     "boundaries": ("periodic", "periodic", "periodic"), "symmetry": (),
     "k_point": (0.5, 0.0, 0.0)},
    {"label": "walls_xy", "cell": (1.6, 1.7, 0.0), "beta": CORPUS_BETAS[3],
     "boundaries": ("metallic", "metallic", "periodic"), "symmetry": (),
     "k_point": (0.0, 0.0, 0.0)},
    {"label": "wall_x_only", "cell": (1.6, 1.6, 0.0), "beta": 0.3321611318837033,
     "boundaries": ("metallic", "periodic", "periodic"), "symmetry": (),
     "k_point": (0.0, 0.0, 0.0)},
    # ---- the folded grids: three of this cell's four rows are Mirror(Y) periodic ----
    {"label": "fold_y_even_k0", "cell": (1.6, 2.0, 0.0), "beta": CORPUS_BETAS[0],
     "boundaries": None, "symmetry": (("Y", 1),), "k_point": (0.0, 0.0, 0.0)},
    {"label": "fold_y_odd_kx", "cell": (1.6, 2.0, 0.0), "beta": CORPUS_BETAS[2],
     "boundaries": None, "symmetry": (("Y", -1),), "k_point": (0.3, 0.0, 0.0)},
    {"label": "fold_y_odd_count", "cell": (1.6, 2.1, 0.0), "beta": CORPUS_BETAS[1],
     "boundaries": None, "symmetry": (("Y", 1),), "k_point": (0.0, 0.0, 0.0)},
    {"label": "fold_y_metallic_termination", "cell": (1.6, 2.0, 0.0),
     "beta": CORPUS_BETAS[3], "boundaries": ("periodic", "metallic", "periodic"),
     "symmetry": (("Y", 1),), "k_point": (0.0, 0.0, 0.0)},
)

#: The fixtures the mutations are scored on: a live Bloch phase, a wall on each
#: in-plane axis, both fold terminations, and both signs of beta.
MUTATION_SPEC_LABELS: Tuple[str, ...] = ("periodic_kx", "walls_xy", "fold_y_odd_kx",
                                         "fold_y_metallic_termination")

REDUCED_LABELS: Tuple[str, ...] = ("periodic_kx", "walls_xy", "brillouin_edge_kx",
                                   "fold_y_odd_kx", "fold_y_metallic_termination")


# ---------------------------------------------------------------------------
# The mutations
# ---------------------------------------------------------------------------

#: The released complex H->D gate's weld clauses this weld inherits whole (its
#: constitutive half, its curl transform, its connection and its per-chunk forms),
#: armed under the released gate's own spellings.
_INHERITED_WELD_CLAUSES: Tuple[str, ...] = (
    "foreign_tap_reads_stale_H", "foreign_tap_reads_B", "own_cell_reads_stale_H",
    "f_w_H_written_in_place", "foreign_history_reads_the_post_store_value",
    "the_window_edge_is_not_connected", "the_shifted_operand_is_connected",
    "old_package_ghost_phase", "old_package_constitutive_grouping",
    "coefficient_index_is_the_origin", "full_complex_coefficient_products",
    "fmad_default_build", "select_removed", "predicate_on_coefficient_identity",
    "swap_dsig_and_dsigu", "auxiliary_stored_on_a_plain_cell",
)


def _inherited(name: str) -> Dict[str, Any]:
    entry = dict(cx_hd.DEVICE_MUTATIONS[name])
    entry.pop("expect", None)
    return entry


DEVICE_MUTATIONS: Dict[str, Dict[str, Any]] = {
    # ---- THE BETA CLAUSES: this seam's own ----
    "beta_partner_reads_stale_h": {
        "old": "        cf inc_beta = owned_cell ? beta_increment(bp, f_2) : cf_zero();",
        "new": ("        cf inc_beta = owned_cell ? beta_increment(bp, "
                "cconnect(cf_load(g1, idx), k, ph_z, pz)) : cf_zero();"),
        "why": ("THE DEFECT THIS WELD CREATES AND NO SIBLING H->D SEAM HAS. The beta "
                "partner is the SAME-CELL magnetic snapshot, which the array path "
                "takes AFTER update_H; this reverts it to the PRE-LAUNCH volume, "
                "leaving the finite-difference stencil correct and putting exactly ONE "
                "beta pass one sub-step behind. Every magnitude stays plausible and "
                "the run converges"),
        "live_on": "every fixture (beta is nonzero on all of them)",
    },
    "drop_the_beta_term": {
        "old": ("        cfold_apply(f0, u0, idx, inc_beta, sinv_y[j], sinv_z[k], "
                "in_i_z);\n"),
        "new": "        // beta pass removed\n",
        "why": ("the analytic d/dz pass is dropped from target 0 entirely -- a "
                "beta = 0 run on one component, which is a DIFFERENT physical problem "
                "that still converges"),
        "live_on": "every fixture",
    },
    "swap_the_beta_signs": {
        "old": "        cf inc_beta = owned_cell ? beta_increment(bp, f_2) : cf_zero();",
        "new": "        cf inc_beta = owned_cell ? beta_increment(bm, f_2) : cf_zero();",
        "why": ("the +1 call site takes the -1 coefficient (step_db.cpp:161-175's "
                "sign = (d_c == X ? +1 : -1) read backwards). The two words differ "
                "only in sign -- and the cross-polarisation coupling reverses"),
        "live_on": "every fixture",
    },
    "beta_added_not_subtracted": {
        "old": "    o.re = c * g.im;\n    o.im = -(c * g.re);\n",
        "new": "    o.re = -(c * g.im);\n    o.im = c * g.re;\n",
        "why": ("stepping._special_kz_beta_term returns -quarter_turn(c, partner) and "
                "the pass's ladder SUBTRACTS it; the quarter turn without its "
                "negation puts the term at the wrong sign"),
        "live_on": "every fixture",
    },
    "beta_scaled_by_dtdx": {
        "old": "        cf inc_beta = owned_cell ? beta_increment(bp, f_2) : cf_zero();",
        "new": ("        cf inc_beta = owned_cell ? beta_increment((dtdx * bp), f_2) "
                ": cf_zero();"),
        "why": ("the beta term is an ANALYTIC derivative and carries NO dtdx "
                "(stepping._special_kz_beta_term). Scaling it by the finite-difference "
                "factor rescales the TE/TM coupling by the Courant number"),
        "live_on": "every fixture",
    },
    "beta_on_the_third_component": {
        # TARGET 2 gets a beta pass it must not have. The anchor is target 2's own
        # absorber line, which is unique.
        "old": ("        pml_apply(f2, u2, idx, curl, kms_x[i], sinv_x[i], kms_y[j], "
                "sinv_y[j], in_i_y);\n"),
        "new": ("        pml_apply(f2, u2, idx, curl, kms_x[i], sinv_x[i], kms_y[j], "
                "sinv_y[j], in_i_y);\n"
                "        cfold_apply(f2, u2, idx, beta_increment(bp, f_2), sinv_x[i], "
                "sinv_y[j], in_i_y);\n"),
        "why": ("a beta pass on the component MEEP gives none (step_db.cpp:161-175 "
                "runs cc over d_c in {X, Y} only). The component PATTERN is part of "
                "the transcription: targets 0 and 1 must move and target 2 must not"),
        "live_on": "every fixture",
    },
    "beta_on_an_unowned_cell": {
        # THE COMPOSITION NEEDLE the certified single's own gate carries, re-armed on
        # the weld: the increment formed on every cell. TARGET 1 (Dy), NOT TARGET 0:
        # under MEEP's per-chunk forms the beta pass is a ladder of its own on the
        # stored word, so an unowned cell's increment survives a complete driver step
        # only where it reaches an AUXILIARY (every unowned plane's f word is rewritten
        # by the wall pass or a fill). Dx's auxiliary axis is the invariant z, so target
        # 0's increment is inert on every 2-D fixture; Dy's is x, and Dy's far-ghost row
        # of a PERIODIC-terminated y fold lies in the x layer's columns (measured on the
        # real twin's text on the host: target 0 moves 0 words on 6 of 6 grids, target
        # 1 moves fu_Dy on the periodic fold only; gate_special_kz_fused_hd_pair.py).
        "old": "        cf inc_beta = owned_cell ? beta_increment(bm, f_1) : cf_zero();",
        "new": "        cf inc_beta = beta_increment(bm, f_1);",
        "why": ("MEEP's beta loop visits owned cells only, and the array path masks "
                "the increment to them; formed on every cell it lands on the masked "
                "wall and fold planes, and on the far-ghost row of a periodic fold it "
                "reaches fu_Dy in the x layer"),
        "live_on": "fold_y_odd_kx (a periodic-terminated y fold with an x layer)",
    },
    # ---- THE WELD CLAUSES (the released complex H->D gate's) ----
    **{name: _inherited(name) for name in _INHERITED_WELD_CLAUSES},
    "auxiliary_load_unguarded": {
        **_inherited("auxiliary_load_unguarded"),
        "predicted_null": ("a DECLARED NULL (KERNELS.md section 7.2): the loaded word "
                           "feeds only the forms of the cells whose chunk carries the "
                           "layer, so no stored word changes; the static needle in the "
                           "module's own test pins the guard"),
    },
    "the_periodic_wrap_reads_the_near_row": {
        "old": ("    if (bc == BC_METALLIC || bc == BC_MIRROR_PERIODIC) return cf_zero();\n"
                "    return resolve_H(comp, idx + (na - 1) * stride, weld);"),
        "new": ("    if (bc == BC_METALLIC || bc == BC_MIRROR_PERIODIC) return cf_zero();\n"
                "    return resolve_H(comp, idx + stride, weld);"),
        "why": ("the periodic wrap reads the row ABOVE instead of the far row. Unlike "
                "the metallic ghost this IS observable: on a periodic axis the "
                "ownership mask does not zero the curl the wrapped tap feeds"),
        "live_on": "every fixture (their x axis wraps)",
    },
    "the_ghost_serves_the_clamped_cells_constitutive": {
        # ANCHORED ON THE RECOMPUTE'S OWN GHOST, with the wrap line carried along: the
        # certified ``cshift_up`` and ``cshift_dn`` ride along in the lifted prelude and
        # spell the ghost line identically.
        "old": ("    if (bc == BC_METALLIC || bc == BC_MIRROR_PERIODIC) return cf_zero();\n"
                "    return resolve_H(comp, idx + (na - 1) * stride, weld);"),
        "new": ("    if (bc == BC_METALLIC || bc == BC_MIRROR_PERIODIC) "
                "return resolve_H(comp, idx, weld);\n"
                "    return resolve_H(comp, idx + (na - 1) * stride, weld);"),
        "why": ("the metallic and mirror ghosts serve update_H's value at the clamped "
                "cell instead of the exact cf_zero(). A DERIVED NULL for BOTH codes: "
                "``ghost_observability`` reads off the emitted text that every "
                "redirected tap is masked at exactly the plane its constant ghost "
                "fires on, so nothing either ghost serves can reach step_D's output. "
                "The control that MUST bite is the PERIODIC sibling above"),
        "live_on": "no fixture -- derived null; see ghost_observability",
        "predicted_null": ("derived from the emitted text by the ghost_observability "
                           "leg, for BOTH constant-ghost codes: the set of (tap, "
                           "plane) pairs a ghost fires on and the set the ownership "
                           "mask zeroes COINCIDE"),
    },
}

#: The one armed edit required NOT to diverge: the own-cell register replaced by a
#: RELOAD of the scratch word this thread has just stored -- the same value by a
#: different route (the released complex H->D gate's).
BYTE_NEUTRAL: Dict[str, str] = {
    **cx_hd.BYTE_NEUTRAL,
    "why": ("the own-cell register is replaced by a RELOAD of the scratch word pair "
            "this thread has just stored -- the same value by a different route. Two "
            "float32 words stored to global memory and loaded back are the identity "
            "on the bits, so an edit that diverged here would mean the harness rather "
            "than the weld decides the answer"),
}

HOST_MUTATIONS: Tuple[str, ...] = ("rotation_skipped", "scratch_aliased_to_storage",
                                   "swap_constitutive_sublattice",
                                   "the_other_direction_s_connection",
                                   "swap_beta_coefficient_words",
                                   "record_bound_off_by_one", "record_lattices_swapped")


# ---------------------------------------------------------------------------
# The adapter
# ---------------------------------------------------------------------------

class ComplexBetaAdapter(kit.Adapter):
    """The COMPLEX beta H->D weld's answers to the shared legs' questions."""

    complex_storage = True
    cell_arms = CELL_ARMS
    specs = SPECS
    mutation_spec_labels = MUTATION_SPEC_LABELS
    reduced_labels = REDUCED_LABELS
    value_classes = ("uniform", "subnormal_band", "planted_row0")
    device_mutations = DEVICE_MUTATIONS
    host_mutations = HOST_MUTATIONS
    byte_neutral = BYTE_NEUTRAL
    #: (update_H's arm, step_D's arm). Both are this family's own ``complex beta``
    #: arm: ``covers_complex_beta_fused_hd_pair`` opens with
    #: ``complex_beta_kernels.covers_complex_beta_constitutive(..., "H")`` and
    #: continues into ``covers_complex_beta_curl(..., "step_D")``, which ARE that
    #: arm's two predicates. The board's cell is the same pair of labels.
    wiring_arms_row = ("complex beta", "complex beta")
    wiring_module = "complex_beta_fused_hd_pair"
    released_neighbours = ("cuda_complex_beta_fused_magnetic_pair",
                           "cuda_complex_beta_fused_electric_pair")
    env_stem = "MEEP_GPU_CXBHD_GATE"

    def __init__(self) -> None:
        from meep_gpu.cuda_kernels import complex_beta_fused_hd_pair as family  # noqa: PLC0415
        self.family = family

    # -- fixtures -------------------------------------------------------------
    def plant(self, fields, grid, value_class: str) -> Dict[str, Any]:
        """Plant the tiny-normal row-0 class on EVERY stored volume.

        WHERE THE PLANT HAS TO GO IS A MEASUREMENT, not a guess, and the released
        complex H->D gate got it wrong on its first cut: planting only the MAGNETIC
        volumes left ``naive_four_product_multiply`` uncaught under ``flush``, because
        ``mul_field_left`` is called by ``pml_apply`` on ``D``, ``fu_D`` and the curl,
        NOT on ``H`` or ``B``. The plant also varies the SIGNS across the plane: the
        defect is the sign of a FLUSHED zero, so both parts must appear with both signs
        for the class to be reached at all.
        """
        if value_class != "planted_row0":
            return {"planted": False}
        shape = tuple(int(n) for n in grid.shape)
        magnitude = float(PLANT_MAGNITUDE)
        j = np.arange(shape[1])[:, None]
        k = np.arange(shape[2])[None, :]
        real_sign = np.where((j + k) % 2 == 0, -1.0, 1.0)
        imag_sign = np.where(j % 2 == 0, -1.0, 1.0) * np.ones_like(k)
        plane = np.empty(shape[1:], dtype=np.complex64)
        plane.real = (real_sign * magnitude).astype(np.float32)
        plane.imag = (imag_sign * magnitude).astype(np.float32)
        planted: List[str] = []
        for name in kit.STATE_NAMES:
            array = getattr(fields, name, None)
            if array is None:
                continue
            host = to_host(array).copy()
            host[0, :, :] = plane
            array[...] = cp.asarray(np.ascontiguousarray(host))
            planted.append(name)
        return {"planted": True, "volumes": planted,
                "cells": len(planted) * int(np.prod(shape[1:])),
                "magnitude": magnitude,
                "why": ("row 0 of EVERY stored volume set to tiny NORMALS of all four "
                        "sign combinations. Multiplied by any coefficient below 1 -- "
                        "every sinv, and kms inside the absorber -- the product "
                        "underflows, and under flush the sign of the flushed zero is "
                        "what separates the naive four-product multiply from the "
                        "contracted one. A random battery reaches this class with "
                        "probability ~0")}

    # -- the product ----------------------------------------------------------
    def predicate(self, fields, pml, grid, sources) -> Tuple[bool, str]:
        return self.family.covers_complex_beta_fused_hd_pair(fields, pml, grid, sources)

    def resolve(self, fields, grid, pml, dtdx: float, **overrides) -> Dict[str, Any]:
        # The chunk record the kernel binds as ``cb``: the one-rank division the array
        # path steps with, unless a host mutation hands a corrupted one.
        if overrides.get("chunk_bounds") is None:
            overrides["chunk_bounds"] = kit.hd_gate.chunk_record(grid, pml)
        return self.family.resolve(fields, grid, pml, dtdx=dtdx, **overrides)

    def assert_disjoint(self, fields, state) -> int:
        return self.family.assert_bindings_are_disjoint(fields, state)

    def launch(self, fields, state, kernel=None, threads: Optional[int] = None):
        return self.family.launch_complex_beta_fused_hd_pair(
            fields, state, kernel,
            self.family._FUSED_THREADS if threads is None else int(threads))  # noqa: SLF001

    def rotate(self, fields, state) -> None:
        state["scratch"] = self.family.rotate_into_fields(fields, state["scratch"])

    def boundary_codes(self, grid):
        from meep_gpu.cuda_kernels import complex_folded_kernels  # noqa: PLC0415
        return complex_folded_kernels.folded_complex_boundary_codes(grid)

    def kernel_source(self, source: Optional[str] = None) -> str:
        return self.family.kernel_source() if source is None else source

    def compile(self, source: Optional[str] = None,
                options: Optional[Tuple[str, ...]] = None):
        kernel = cp.RawKernel(
            self.kernel_source(source), self.kernel_name(),
            options=self.compile_options() if options is None else tuple(options))
        kernel.compile()
        return kernel

    # -- the references -------------------------------------------------------
    def singles(self, fields, grid, pml, dtdx: float) -> Dict[str, Callable[[], None]]:
        """The certified single-slot launchers for a COMPLEX beta run.

        The two CURL slots go to ``complex_beta_kernels``' certified beta pair, and
        the two CONSTITUTIVE slots to the certified complex pair -- which is what
        ``covers_complex_beta_constitutive`` admits and what the census records at
        ``update_H``/``update_E`` on all four corpus rows, because that family ships no
        constitutive kernel of its own.
        """
        from meep_gpu.cuda_kernels import (complex_beta_kernels,  # noqa: PLC0415
                                           complex_emitter, complex_pml_kernels)

        constitutive = {side: complex_pml_kernels.complex_constitutive_tables(
            pml, complex_emitter.HALF_INTEGER[side]) for side in ("H", "E")}
        record = kit.hd_gate.chunk_record(grid, pml)

        def curl(sub_step: str):
            return lambda: complex_beta_kernels.step_complex_beta(
                sub_step, fields, grid=grid, pml=pml, dtdx=dtdx, chunk_bounds=record)

        def update_h() -> None:
            # MEEP's start of H before the run's first update_H
            # (stepping.separate_h_from_b): the shipped single launcher runs it and the
            # bare launcher does not.
            kit.stepping.separate_h_from_b(fields, pml)
            complex_pml_kernels.update_fused_pml_complex(
                "H", fields, tables=constitutive["H"], chunk_bounds=record)

        return {
            "step_B": curl("step_B"),
            "step_D": curl("step_D"),
            "update_H": update_h,
            "update_E": lambda: complex_pml_kernels.update_fused_pml_complex(
                "E", fields, tables=constitutive["E"], chunk_bounds=record),
        }

    def released_pairs(self, fields, grid, pml, dtdx: float):
        from meep_gpu.cuda_kernels import (  # noqa: PLC0415
            complex_beta_fused_electric_pair as electric,
            complex_beta_fused_magnetic_pair as magnetic)

        record = kit.hd_gate.chunk_record(grid, pml)

        def run_magnetic():
            return magnetic.run_complex_beta_fused_magnetic_pair(
                fields, grid, pml, dtdx, chunk_bounds=record, sources=())

        def run_electric():
            return electric.run_complex_beta_fused_electric_pair(
                fields, grid, pml, dtdx, chunk_bounds=record, sources=())

        return magnetic, electric, run_magnetic, run_electric

    # -- the host legs --------------------------------------------------------
    def leg_transcription(self) -> Dict[str, Any]:
        """Is every welded byte the certified family's, and does the lift INVERT?

        ONE TEXT AND TWO BYTE EQUALITIES. The equalities are the strongest thing this
        leg has and they are NEW measurements rather than restatements: this family's
        curl transform applied to the CERTIFIED PLAIN complex ``step_D`` text returns
        exactly what ``complex_fused_hd_pair.curl_pieces("plain")`` returns, and applied
        to the CERTIFIED FOLDED text exactly what its ``"folded"`` variant returns --
        byte for byte. The beta weld is therefore the RELEASED, GATED transform plus the
        beta insert and the beta-partner assertion.
        """
        from meep_gpu.cuda_kernels import (complex_emitter,  # noqa: PLC0415
                                           complex_folded_kernels,
                                           complex_fused_hd_pair)

        family = self.family
        rows: List[Dict[str, Any]] = []
        equalities: List[Dict[str, Any]] = []
        source = family.kernel_source()
        prelude = family.constitutive_prelude()
        certified_prelude, _ = family._curl_source().split(  # noqa: SLF001
            '\nextern "C" __global__ void ', 1)
        inverted = prelude
        for old, new in complex_fused_hd_pair.CONSTITUTIVE_LIFT_EDITS:
            inverted = inverted.replace(new, old, 1)
        rows.append({
            "kernel": family.KERNEL_NAME,
            "lines": len(source.splitlines()),
            "sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
            "ascii": kit.cyl.encodes_ascii(source),
            "the_lift_inverts_to_the_certified_prelude":
                inverted == certified_prelude,
            "the_part_products_block_is_present_whole":
                complex_emitter._PART_PRODUCTS.strip() in source,  # noqa: SLF001
            "the_kernel_name_is_declared":
                f'extern "C" __global__ void {family.KERNEL_NAME}(' in source,
            "cshift_dn_recompute_keeps_the_near_neighbour_branch":
                "if (ia > 0) return resolve_H(comp, idx - stride, weld);" in source,
            "cshift_dn_recompute_keeps_the_wrap_arithmetic":
                "resolve_H(comp, idx + (na - 1) * stride, weld)" in source,
            # THE STORED WINDOW: a shifted tap is a plain wrap and carries no
            # connection; the connection rides on the operand read at the cell
            # itself, which here is the own-cell register.
            "cshift_dn_recompute_carries_no_connection":
                "rotate_field_left" not in source.split(
                    "cshift_dn_recompute(", 1)[1].split("\n}\n", 1)[0],
            "every_own_cell_register_is_connected":
                source.count("cconnect(own_h[") == family.OWN_LOAD_EDITS,
            "the_ghost_is_still_an_exact_cf_zero":
                ("if (bc == BC_METALLIC || bc == BC_MIRROR_PERIODIC) "
                 "return cf_zero();") in source,
            "no_magnetic_pointer_survives_in_the_curl_half":
                "cshift_dn(g" not in source,
            # MINUS THE DECLARATION: the helper's own signature spells its name
            # too, and a count that included it would report seven taps and pass a
            # weld that had lost one.
            "shifted_taps": source.count("cshift_dn_recompute(") - 1,
            "own_cell_registers": source.count("own_h["),
            # THE FOLDED DELTAS, which this family inherits whole because its text
            # IS the folded one with an insert.
            "the_third_boundary_code_is_defined":
                "#define BC_MIRROR_PERIODIC 2" in source,
            "the_ghost_is_widened_to_the_metallic_zero":
                "bc == BC_METALLIC || bc == BC_MIRROR_PERIODIC" in source,
            "the_mask_splits_by_termination":
                source.count("BC_MIRROR_PERIODIC && ") >= 6,
            # THE BETA DELTAS: MEEP's pass, after each component's curl.
            "the_two_beta_increments_are_present":
                source.count("cf inc_beta = owned_cell ? beta_increment(b")
                == family.BETA_INSERT_LINES,
            "the_beta_helper_is_the_part_wise_quarter_turn":
                "    o.re = c * g.im;\n    o.im = -(c * g.re);\n" in source,
            "the_two_beta_words_are_bound": "float bp, float bm," in source,
            "both_beta_partners_are_weld_registers":
                "beta_increment(bp, f_2)" in source
                and "beta_increment(bm, f_1)" in source
                and "cf f_2 = cconnect(own_h[1], k, ph_z, pz);" in source
                and "cf f_1 = cconnect(own_h[0], k, ph_z, pz);" in source,
        })
        mine_plain = family.curl_pieces(complex_emitter.complex_source("step_D"))
        mine_folded = family.curl_pieces(complex_folded_kernels.folded_source("step_D"))
        equalities.append({
            "the_transform_on_the_certified_plain_text_is_the_released_transform":
                mine_plain == complex_fused_hd_pair.curl_pieces("plain"),
            "the_transform_on_the_certified_folded_text_is_the_released_transform":
                mine_folded == complex_fused_hd_pair.curl_pieces("folded"),
        })
        needles = kit.mutation_needles_resolve(self)
        keys = [key for key in rows[0] if isinstance(rows[0][key], bool)]
        checks = {key: all(bool(row.get(key, True)) for row in rows) for key in keys}
        checks["every_text_redirects_six_taps_and_six_own_loads"] = all(
            row["shifted_taps"] == family.HALO_TAPS
            and row["own_cell_registers"] >= family.OWN_LOAD_EDITS for row in rows)
        checks["the_transform_is_the_released_one_on_both_certified_texts"] = all(
            row["the_transform_on_the_certified_plain_text_is_the_released_transform"]
            and row["the_transform_on_the_certified_folded_text_is_the_released_transform"]
            for row in equalities)
        checks["every_armed_needle_resolves_to_exactly_one_site"] = needles["passed"]
        return {"texts": rows, "denominator": len(rows),
                "released_transform_equalities": equalities,
                "mutation_needles": needles, "checks": checks,
                "device_sources": sorted(family.device_sources()),
                "source_digest": family.source_digest(),
                "what_the_equalities_license": (
                    "that this family's curl transform IS the released complex H->D "
                    "weld's, over BOTH of that product's certified input texts, with "
                    "the beta text as its only other input -- so every "
                    "clause gate_cuda_complex_fused_hd_pair.py measured about that "
                    "transform carries, and what THIS gate adds is the beta insert and "
                    "the beta partner"),
                "passed": all(checks.values())}

    def leg_refusal(self) -> Dict[str, Any]:
        """Every configuration this predicate must refuse, BY NAME, on a host."""
        from meep_gpu import withdraw_hoist  # noqa: PLC0415

        family = self.family
        named: List[Dict[str, Any]] = []

        class _HostGrid:
            xp = np
            beta = 0.0
            dimensions = 2
            cylindrical = False

            @staticmethod
            def has_symmetry():
                return False

            @staticmethod
            def is_mirrored(axis):  # noqa: ARG004
                return False

        class _BetaGrid(_HostGrid):
            beta = 0.25

        class _Fields:
            force_complex_fields = True

        def check(label: str, thunk, expect_covered: bool, names: str = "") -> None:
            try:
                covered, reason = thunk()
            except Exception as error:  # noqa: BLE001
                named.append({"case": label, "raised": repr(error)[:300],
                              "passed": False})
                return
            ok = bool(covered) == expect_covered and (
                not names or names.lower() in str(reason).lower())
            named.append({"case": label, "covered": bool(covered),
                          "reason": str(reason)[:300],
                          "expected_covered": expect_covered, "names": names,
                          "passed": ok})

        # 1. beta = 0 belongs to the certified complex pair or the folded one.
        check("beta_is_zero",
              lambda: family.covers_complex_beta_fused_hd_pair(
                  _Fields(), None, _HostGrid(), ()),
              False, "grid.beta is zero")

        # 2. REAL storage with beta is special_kz_curl's, named.
        class _RealFields:
            force_complex_fields = False

        check("real_storage_with_beta",
              lambda: family.covers_complex_beta_fused_hd_pair(
                  _RealFields(), None, _BetaGrid(), ()),
              False, "special_kz_curl")

        # 3. CYLINDRICAL with beta: MEEP aborts.
        class _Cylindrical(_BetaGrid):
            cylindrical = True

        check("cylindrical_with_beta",
              lambda: family.covers_complex_beta_fused_hd_pair(
                  _Fields(), None, _Cylindrical(), ()),
              False, "cylindrical")

        # 4. A 3-D grid with beta.
        class _ThreeD(_BetaGrid):
            dimensions = 3

        check("three_dimensional_with_beta",
              lambda: family.covers_complex_beta_fused_hd_pair(
                  _Fields(), None, _ThreeD(), ()),
              False, "dimensions")

        # 5. An undeclared source set: the seam clause cannot infer an absence.
        check("sources_undeclared",
              lambda: family.covers_complex_beta_fused_hd_pair(
                  _Fields(), None, _BetaGrid(), None),
              False, "")

        # 6. The withdraw refusal's text, on a synthetic standing source.
        class _Withdrawing:
            field_type = "electric"
            is_integrated = True
            _n_source_points = 1

            def withdraw(self, fields):  # noqa: ARG002
                return None

        seam_reasons = withdraw_hoist.seam_withdraw_reasons(
            _Fields(), (_Withdrawing(),), undeclared="undeclared",
            refusal=lambda index, source: (
                f"source {index} ({type(source).__name__}) has a standing integrated "
                f"electric withdraw"),
            hoists_the_withdraw=family.HOISTS_THE_WITHDRAW, span=family.REPLACES)
        withdraw_case = {
            "reasons": [str(text)[:220] for text in seam_reasons],
            "passed": bool(seam_reasons)
                      and "standing integrated" in str(seam_reasons[0])}

        # 7. The transform refuses a text whose beta partner is NOT the weld register.
        stale = {"passed": False}
        try:
            certified = family._curl_source()  # noqa: SLF001
            corrupted = certified.replace(
                "        cf inc_beta = owned_cell ? beta_increment(bp, f_2) : cf_zero();",
                "        cf f_9 = cf_load(g1, idx);\n"
                "        cf inc_beta = owned_cell ? beta_increment(bp, f_9) : cf_zero();",
                1)
            if corrupted == certified:
                raise RuntimeError("the corruption's anchor is gone from the "
                                   "certified complex beta step_D text")
            family.curl_pieces(corrupted)
            stale = {"passed": False,
                     "note": "the transform accepted a beta partner it did not redirect"}
        except AssertionError as error:
            stale = {"raised": str(error)[:300], "passed": True}
        except Exception as error:  # noqa: BLE001
            stale = {"raised": repr(error)[:300], "passed": False}

        record = {"named": named, "withdraw_text": withdraw_case,
                  "the_transform_refuses_a_stale_beta_partner": stale,
                  "denominator": len(named) + 2}
        record["passed"] = bool(all(row["passed"] for row in named)
                                and withdraw_case["passed"] and stale["passed"])
        return record

    def leg_spelling(self) -> Dict[str, Any]:
        """A real coefficient times a complex field, and the divide this family lacks.

        THE MULTIPLY IS THE RELEASED COMPLEX H->D GATE'S LEG
        (``gate_cuda_complex_fused_hd_pair.leg_spelling``): this weld lifts the same
        ``mul_field_left`` / ``mul_coefficient_left`` from the same emitter, measured
        against the array path's product (``part_products.scale``), with the earlier
        full complex products required to bite somewhere.

        THE DIVIDE IS RECORDED AS AN ABSENCE WITH A CONTROL THAT BITES. This family has
        no division site; the assertion is armed by planting one and requiring the
        family's own emitter to refuse it.
        """
        if cp is None:
            return {"passed": False, "error": "no CuPy"}
        family = self.family
        record = cx_hd.leg_spelling(self.policy)
        armed = {"refused": None, "passed": False}
        try:
            family._assert_the_measured_absence_of_a_divide(  # noqa: SLF001
                family.kernel_source().replace(
                    "    cf own_h[3];", "    float bad = 1.0f / dtdx;", 1))
        except AssertionError as error:
            armed = {"refused": str(error)[:300], "passed": True}
        record["the_divide_has_no_site_here"] = {
            "planted_divide_is_refused": armed,
            "why": ("the reciprocal this recurrence needs is sinv, computed host-side "
                    "by PML, so a complex divide has nothing to be right about here. "
                    "Asserted against the emitted text, with a planted divide as the "
                    "control")}
        record["passed"] = bool(record.get("passed") and armed["passed"])
        return record

    # -- host mutations -------------------------------------------------------
    def host_mutation(self, name: str, spec: Mapping[str, Any],
                      steps: int) -> Dict[str, Any]:
        family = self.family
        if name == "rotation_skipped":
            record = kit.drive(self, spec, "uniform", steps,
                               modes=("array", "weld"), rotate=False)
            return {"caught": not record["identical"]["weld"],
                    "first_divergence": record["first_divergence"]["weld"],
                    "why": ("the launcher never swaps the H/f_w_H bindings, so the "
                            "driver keeps stepping the PRE-launch magnetic field")}
        if name == "scratch_aliased_to_storage":
            fields, grid, pml, dtdx, _plant = self.build(
                spec, "uniform", kit.case_rng(self.seed, "alias"))
            state = self.resolve(fields, grid, pml, dtdx)
            state["scratch"] = {volume: getattr(fields, volume)
                                for volume in family.SCRATCH_VOLUMES}
            try:
                self.assert_disjoint(fields, state)
                return {"caught": False, "refused": None,
                        "why": "the disjointness check accepted the in-place weld"}
            except ValueError as error:
                return {"caught": True, "refused": str(error)[:400],
                        "why": ("the scratch IS the storage, which makes the launch "
                                "the IN-PLACE weld the board refused")}
        if name == "swap_constitutive_sublattice":
            from meep_gpu.cuda_kernels import (complex_emitter,  # noqa: PLC0415
                                               complex_pml_kernels)
            fields, grid, pml, dtdx, _plant = self.build(
                spec, "uniform", kit.case_rng(self.seed, "swap"))
            half = complex_pml_kernels.complex_constitutive_tables(pml, True)
            tables = complex_pml_kernels.complex_curl_tables(
                pml, complex_emitter.HALF_INTEGER["step_D"])
            record = kit.drive(self, spec, "uniform", min(steps, 8),
                               modes=("array", "weld"),
                               state_overrides={"constitutive": half,
                                                "tables": tables})
            return {"caught": not record["identical"]["weld"],
                    "first_divergence": record["first_divergence"]["weld"],
                    "why": ("update_H's kps/kms taken from the HALF-INTEGER "
                            "sub-lattice -- a half-cell error in the absorber "
                            "profile, converged and smooth")}
        if name == "the_other_direction_s_connection":
            from meep_gpu.cuda_kernels import complex_pml_kernels  # noqa: PLC0415
            phased = next((case for case in self.specs if any(case["k_point"])), spec)
            fields, grid, pml, dtdx, _plant = self.build(
                phased, "uniform", kit.case_rng(self.seed, "phase"))
            # THE OTHER DIRECTION'S CONNECTION (the forward curl's) where the D curl's
            # belongs: the classic sign error, in which every magnitude stays plausible
            # and only the phase moves.
            flags, values = complex_pml_kernels.bloch_phase_arguments(grid, False)
            record = kit.drive(self, phased, "uniform", min(steps, 8),
                               modes=("array", "weld"),
                               state_overrides={"phase_flags": flags,
                                                "phase_values": values})
            return {"caught": not record["identical"]["weld"],
                    "first_divergence": record["first_divergence"]["weld"],
                    "fixture": phased["label"],
                    "why": ("the forward curl's Bloch connection bound where the D "
                            "curl's belongs -- the classic sign error")}
        if name == "swap_beta_coefficient_words":
            fields, grid, pml, dtdx, _plant = self.build(
                spec, "uniform", kit.case_rng(self.seed, "beta"))
            plus, minus = family.beta_coefficients(grid)
            record = kit.drive(self, spec, "uniform", min(steps, 8),
                               modes=("array", "weld"),
                               state_overrides={"beta": (minus, plus)})
            return {"caught": not record["identical"]["weld"],
                    "first_divergence": record["first_divergence"]["weld"],
                    "coefficients": [float(plus), float(minus)],
                    "why": ("the two host-rounded call-site coefficients are handed "
                            "over in the wrong order. Only the HOST transcription can "
                            "be wrong here, which is why this is a host mutation")}
        if name in ("record_bound_off_by_one", "record_lattices_swapped"):
            # KERNELS.md section 7.2: the chunk record the WELD binds, corrupted (the
            # oracle's division is untouched); the released complex H->D gate's.
            fields, grid, pml, dtdx, _plant = self.build(
                spec, "uniform", kit.case_rng(self.seed, "record"))
            bound = cx_hd.mutated_record(name, grid, pml)
            record = kit.drive(self, spec, "uniform", min(steps, 8),
                               modes=("array", "weld"),
                               state_overrides={"chunk_bounds": bound})
            return {"caught": not record["identical"]["weld"],
                    "first_divergence": record["first_divergence"]["weld"],
                    "why": ("the boundary plane of a layer changes form (off by one) "
                            "or moves by one cell (lattices swapped)")}
        raise ValueError(f"no such host mutation {name!r}")

    # -- wiring ---------------------------------------------------------------
    def wiring_product_row(self) -> Dict[str, Any]:
        from meep_gpu.cuda_kernels import fused_pairs  # noqa: PLC0415
        from meep_gpu.triton_kernels.coverage import Coverage  # noqa: PLC0415

        family = self.family

        def coverage(context: Any) -> Coverage:
            covered, reason = family.covers_complex_beta_fused_hd_pair(
                context.fields, context.pml, context.grid, context.sources)
            return Coverage(bool(covered), (str(reason),))

        def plan(context: Any):
            held: Dict[str, Any] = {}

            def resolve(ctx: Any) -> Dict[str, Any]:
                if "state" not in held:
                    held["state"] = family.resolve(
                        ctx.fields, ctx.grid, ctx.pml,
                        chunk_bounds=ctx.chunk_bounds_device())
                return {"state": held["state"]}

            def launch(fields: Any, arguments: Dict[str, Any]) -> Dict[str, Any]:
                return family.run_complex_beta_fused_hd_pair(
                    fields, context.grid, context.pml,
                    sources=context.sources, state=arguments["state"], check=False)

            return fused_pairs.CudaFusedPairPlan(
                family=family.FAMILY, label="complex beta fused H/D pair",
                kernel_label=family.KERNEL_NAME, replaces=tuple(family.REPLACES),
                slots=("update_H", "step_D"), context=context,
                resolve_launch_args=resolve, launch=launch)

        return {"curl_slot": "update_H", "coverage": coverage, "plan": plan,
                "module": self.wiring_module}


LICENCES_WHAT = (
    "ONE CUDA kernel measured byte-identical, per COMPLETE DRIVER STEP and as uint32 "
    "words over every stored volume, to four independent arrangements of the "
    "update_H -> step_D seam on a COMPLEX64 special_kz (grid.beta != 0) run, at every "
    "block size and at three seed scales, on the fixtures and corpus rows this record "
    "names, under the installed float32 subnormal policy. It licenses NO throughput "
    "claim, NO dispatch claim and "
    "NO composition claim: the family declares INSTALLABLE = False, is not in the "
    "composer's tables on disk, and a credited seam-instance is PREDICATE ADMISSION.")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = kit.make_parser(__doc__.splitlines()[0])
    args = parser.parse_args(argv)
    adapter = ComplexBetaAdapter()

    if args.lift_child:
        return kit.lift_child(
            adapter, args.lift_child, args.lift_child_target,
            json.loads(args.lift_child_cases), args.lift_child_out,
            args.lift_steps, args.lift_child_progress, args.subnormal_policy,
            bool(args.import_meep_for_host_policy), BLOCK_KEY)

    if not args.out:
        parser.error("--out is required unless --lift-child is given")
    return kit.run_campaign(adapter, args, GATE_FILE, FAMILY_NAME, BLOCK_KEY,
                            LICENCES_WHAT)


if __name__ == "__main__":
    raise SystemExit(main())
