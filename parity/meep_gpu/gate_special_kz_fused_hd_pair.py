"""Gate: ``meep_gpu.cuda_kernels.special_kz_fused_hd_pair`` -- the REAL beta H->D weld.

ONE CUDA kernel welding ``stepping.update_H`` into the special_kz ``stepping.step_D``
under REAL float32 storage on a ``grid.beta != 0`` run, measured BIT-IDENTICAL as
uint32 words per COMPLETE DRIVER STEP against FOUR independent arrangements of the
seam, under both float32 subnormal policies, on synthetic fixtures and on the two
lifted corpus rows of its cell.

THE CELL
=============================================================================
``results/fusion_matrix_cuda_2026-09-07_wired/fusion_matrix_cuda.json``,
``h_to_d_seam.instances`` filtered to ``buildable_not_built``:

* ``(cuda_special_kz/real beta -> cuda_special_kz/real beta)`` -- **2 instances**,
  ``examples:refl-angular-kz2d.py`` and
  ``tests:TestSpecialKz.test_eigsrc_kz_1_real_imag``.

Those are the only two REAL-storage beta rows in the corpus. The other four beta rows
are complex storage and belong to ``gate_complex_beta_fused_hd_pair.py``, the sibling
gate built in the same round over the same shared kit.

THE FOUR REFERENCES, and what each one rules out
=============================================================================
``array``               ``stepping``'s own ten passes. THE ORACLE.
``singles``             the certified ``update_H_pml_real`` and
                        ``step_D_special_kz_real`` kernels at their own slots,
                        everything else on the array path. Rules out a divergence that
                        is the certified halves' rather than the weld's.
``composition_today``   what the composer BUILDS on these rows today: the released
                        ``cuda_special_kz_fused_magnetic_pair`` at ``step_B`` and
                        ``cuda_special_kz_fused_electric_pair`` at ``step_D``. Rules
                        out "identical to a composition nobody runs".
``unfused``             all four slot-path slots as certified singles. The composition
                        a composer with fusion vetoed would build.
``weld``                the subject.

Plus ``weld_composed`` on the launch-structure leg only: the weld with its two
neighbours as certified singles, so launch counts compare like with like.

WHAT THIS SEAM ADDS THAT NO SIBLING H->D GATE CAN MEASURE
=============================================================================
THE BETA PARTNER. ``stepping._special_kz_beta_term`` consumes the SAME-CELL magnetic
snapshot, which the array path takes AFTER ``update_H``; welded, that snapshot must be
this launch's own ``own_h[...]`` register. ``beta_partner_reads_stale_h`` is the armed
defect: it reverts exactly the two beta partners to a load of the PRE-launch volume
and leaves the other four own-cell loads alone, so it is one sub-step behind on two of
the six curl terms and on nothing else. Every magnitude stays plausible.

WHAT THE ARITHMETIC LEGS ADD, AND WHAT THEY RECORD AS ABSENT
=============================================================================
This family is REAL float32 throughout: ``complex64 * float32``'s contracted
four-product form -- the spelling the complex families must get right -- HAS NO SITE
here, and neither does any division (the reciprocal the recurrence needs is ``sinv``,
computed host-side by ``PML``). Both absences are ASSERTED against the emitted text
rather than omitted, because a reader arriving from the complex or cylindrical H->D
products will look for each. What IS live is ``--fmad=false``: ``curl - (beta * f)``
is a multiply feeding a subtract, exactly the shape a compiler contracts, and the
attribution sub-leg requires the guarded and unguarded builds to differ.

NOT WIRED, AND THE ARBITRATION LEG SAYS SO IN BOTH ARRANGEMENTS
=============================================================================
The family is not in ``fused_pairs.FUSED_PRODUCTS`` / ``registry`` / ``arms`` yet; the
wiring is a separate change. Every arrangement here is PLANNED FROM ARRAYS by this
gate and the array path is driven by this gate. The ``arbitration`` leg measures the
composer AS SHIPPED and again WITH the wiring rows patched in-process, and it ASSERTS
the substantive claims only -- refused BY NAME, the refusal names ``INSTALLABLE =
False``, the selection unchanged, both released neighbours keeping their slots --
while RECORDING the positional ones, which a wiring round inverts by design.
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

cp = kit.cp
log = kit.log

FAMILY_NAME = "cuda_special_kz_fused_hd_pair"
BLOCK_KEY = "cuda_special_kz_fused_hd_pair_gate"
GATE_FILE = "gate_special_kz_fused_hd_pair.py"

#: The board cell, in the census's ``(update_H arm, step_D arm)`` spelling.
CELL_ARMS: Tuple[str, str] = ("cuda_special_kz/real beta", "cuda_special_kz/real beta")

#: The corpus betas this family's admission names, used so the fixtures cover the
#: SIGN of the coefficient as well as its magnitude: ``examples/refl-angular-kz2d.py``
#: carries +0.3321611318837033 and ``TestSpecialKz.test_eigsrc_kz_1_real_imag`` +0.2.
#: A NEGATIVE beta is fixtured too, because the coefficient's sign is what
#: ``swap_beta_signs`` corrupts and a battery of one sign could not tell the two
#: call-site signs apart.
CORPUS_BETA = 0.3321611318837033

#: The synthetic fixtures. Built so that every in-plane axis is walled somewhere,
#: folded somewhere, BOTH fold terminations appear, both full-count parities appear,
#: and a fully periodic grid appears -- which is the ONLY specialisation on which the
#: shifted tap's PERIODIC WRAP is observable at all, because on a metallic axis the
#: ownership mask zeroes the very curl the wrapped tap feeds.
#:
#: EVERY FIXTURE CARRIES A NONZERO BETA and the builder refuses one that does not: a
#: beta-free grid measures a kernel this cell does not contain. The invariant z axis is
#: forced periodic by ``Grid`` at ``dimensions=2``, so no fixture walls or folds it.
SPECS: Tuple[Dict[str, Any], ...] = (
    {"label": "periodic", "cell": (1.6, 1.7, 0.0), "beta": CORPUS_BETA,
     "boundaries": ("periodic", "periodic", "periodic"), "symmetry": (),
     "k_point": (0.0, 0.0, 0.0)},
    {"label": "periodic_negative_beta", "cell": (1.6, 1.7, 0.0), "beta": -0.3907,
     "boundaries": ("periodic", "periodic", "periodic"), "symmetry": (),
     "k_point": (0.0, 0.0, 0.0)},
    {"label": "walls_xy", "cell": (1.6, 1.7, 0.0), "beta": CORPUS_BETA,
     "boundaries": ("metallic", "metallic", "periodic"), "symmetry": (),
     "k_point": (0.0, 0.0, 0.0)},
    {"label": "wall_x_only", "cell": (1.6, 1.6, 0.0), "beta": -0.2,
     "boundaries": ("metallic", "periodic", "periodic"), "symmetry": (),
     "k_point": (0.0, 0.0, 0.0)},
    {"label": "fold_y_even", "cell": (1.6, 2.0, 0.0), "beta": 0.2,
     "boundaries": None, "symmetry": (("Y", 1),), "k_point": (0.0, 0.0, 0.0)},
    {"label": "fold_y_odd_phase", "cell": (1.6, 2.0, 0.0), "beta": CORPUS_BETA,
     "boundaries": None, "symmetry": (("Y", -1),), "k_point": (0.0, 0.0, 0.0)},
    {"label": "fold_y_odd_count", "cell": (1.6, 2.1, 0.0), "beta": -0.3907,
     "boundaries": None, "symmetry": (("Y", 1),), "k_point": (0.0, 0.0, 0.0)},
    {"label": "fold_y_metallic_termination", "cell": (1.6, 2.0, 0.0),
     "beta": CORPUS_BETA, "boundaries": ("periodic", "metallic", "periodic"),
     "symmetry": (("Y", 1),), "k_point": (0.0, 0.0, 0.0)},
    {"label": "fold_x_wall_y", "cell": (2.0, 1.6, 0.0), "beta": 0.2,
     "boundaries": ("periodic", "metallic", "periodic"), "symmetry": (("X", 1),),
     "k_point": (0.0, 0.0, 0.0)},
)

#: The fixtures the mutations are scored on. Chosen so every armed defect has at least
#: one fixture where the machinery it disables is LIVE: a periodic wrap that survives
#: the mask, a wall on each in-plane axis, both fold terminations, and both signs of
#: the beta coefficient.
MUTATION_SPEC_LABELS: Tuple[str, ...] = ("periodic", "walls_xy", "fold_y_odd_count",
                                         "fold_y_metallic_termination")

#: The subset ``--product reduced`` runs: one periodic, one walled, one of each fold
#: termination, and one negative beta.
REDUCED_LABELS: Tuple[str, ...] = ("periodic", "periodic_negative_beta", "walls_xy",
                                   "fold_y_even", "fold_y_metallic_termination")


# ---------------------------------------------------------------------------
# The mutations
# ---------------------------------------------------------------------------

DEVICE_MUTATIONS: Dict[str, Dict[str, Any]] = {
    # ---- THE BETA CLAUSES: this seam's own, and the reason the product exists ----
    "beta_partner_reads_stale_h": {
        "old": ("        float f2 = own_h[1];\n"
                "        float ss = shift_dn_recompute(1, idx, k, nz, sz, bc_z, weld);\n"
                "        float curl = dtdx * ((sf - f1) + (f2 - ss));\n"
                "        curl = curl - (beta_plus * f2);"),
        "new": ("        float f2 = own_h[1];\n"
                "        float ss = shift_dn_recompute(1, idx, k, nz, sz, bc_z, weld);\n"
                "        float curl = dtdx * ((sf - f1) + (f2 - ss));\n"
                "        curl = curl - (beta_plus * Hy[idx]);"),
        "why": ("THE DEFECT THIS WELD CREATES AND NO SIBLING H->D SEAM HAS. The beta "
                "partner is the SAME-CELL magnetic snapshot, which the array path "
                "takes AFTER update_H; this reverts it to the PRE-launch volume, "
                "leaving the finite-difference stencil correct and putting exactly ONE "
                "of the six curl terms one sub-step behind. Every magnitude stays "
                "plausible and the run converges"),
        "live_on": "every fixture (beta is nonzero on all of them)",
    },
    "drop_the_beta_term": {
        "old": "        curl = curl - (beta_plus * f2);\n",
        "new": "        // beta term removed\n",
        "why": ("the analytic d/dz factor is dropped from target 0 entirely -- the "
                "run is then a beta = 0 run on one component, which is a DIFFERENT "
                "physical problem that still converges"),
        "live_on": "every fixture",
    },
    "swap_the_beta_signs": {
        "old": "        curl = curl - (beta_plus * f2);",
        "new": "        curl = curl - (beta_minus * f2);",
        "why": ("the +1 call site takes the -1 coefficient (stepping.py:443 against "  # stepping.py live lines for the frozen device-text citation(s) in this string: 443->472
                ":445). float64 negation and float32 rounding commute exactly, so the "
                "two words differ only in sign -- and the cross-polarisation coupling "
                "reverses"),
        "live_on": "every fixture",
    },
    "beta_added_not_subtracted": {
        "old": "        curl = curl - (beta_plus * f2);",
        "new": "        curl = curl + (beta_plus * f2);",
        "why": ("stepping.py:784 returns -(c * partner) and :443 ADDS it; this drops "  # stepping.py live lines for the frozen device-text citation(s) in this string: 784->811, 443->472
                "the negation. IEEE defines subtraction as addition of the negation, "
                "so the single subtract in the certified text is the same bits -- and "
                "this control is what shows that reading is load-bearing"),
        "live_on": "every fixture",
    },
    "beta_scaled_by_dtdx": {
        "old": "        curl = curl - (beta_plus * f2);",
        "new": "        curl = curl - (dtdx * (beta_plus * f2));",
        "why": ("the beta term is an ANALYTIC derivative and carries NO dtdx "
                "(stepping.py:758-762, :797). Scaling it by the finite-difference "
                "factor is the classic transcription error and rescales the "
                "TE/TM coupling by the Courant number"),
        "live_on": "every fixture",
    },
    "beta_on_the_third_component": {
        # TARGET 2 (Dz) GETS A BETA INCREMENT IT MUST NOT HAVE. MEEP's ``cc`` loop
        # runs over ``d_c`` in {X, Y} only (step_db.cpp:148-176) and
        # stepping.py:467-474 has no Dz branch, so the third component's curl carries
        # no analytic term at all. The anchor is target 2's curl line together with
        # its FIRST mask, which is unique: targets 0 and 1 carry their beta line
        # between those two statements.
        "old": ("        float curl = dtdx * ((sf - f1) + (f2 - ss));\n"
                "        if (bc_x == BC_METALLIC && i == 0) curl = 0.0f;\n"
                "        if (bc_y == BC_METALLIC && j == 0) curl = 0.0f;"),
        "new": ("        float curl = dtdx * ((sf - f1) + (f2 - ss));\n"
                "        curl = curl - (beta_plus * f2);\n"
                "        if (bc_x == BC_METALLIC && i == 0) curl = 0.0f;\n"
                "        if (bc_y == BC_METALLIC && j == 0) curl = 0.0f;"),
        "why": ("a beta increment on the component MEEP gives none. The component "
                "PATTERN is part of the transcription and not a detail: targets 0 and "
                "1 must move and target 2 must not"),
        "live_on": "every fixture",
    },
    # ---- THE WELD CLAUSES: the shape, shared with the released real sibling ----
    "foreign_tap_reads_stale_H": {
        "old": "    if (ia > 0) return resolve_H(comp, idx - stride, weld);",
        "new": ("    if (ia > 0) return (comp == 0 ? weld.Hx\n"
                "                        : comp == 1 ? weld.Hy : weld.Hz)"
                "[idx - stride];"),
        "why": ("the foreign tap loads the PRE-LAUNCH magnetic field instead of "
                "recomputing update_H there -- which is what an unfused step_D would "
                "read and is exactly one sub-step behind"),
        "live_on": "every fixture",
    },
    "foreign_tap_reads_B": {
        "old": "    if (ia > 0) return resolve_H(comp, idx - stride, weld);",
        "new": ("    if (ia > 0) return (comp == 0 ? weld.Bx\n"
                "                        : comp == 1 ? weld.By : weld.Bz)"
                "[idx - stride];"),
        "why": ("the tap reads the constitutive SOURCE rather than its result. Under "
                "mu = 1 outside the absorber H and B are close, so this is the "
                "plausible wrong answer rather than an obvious one"),
        "live_on": "every fixture",
    },
    "own_cell_reads_stale_H": {
        "old": "        float f1 = own_h[2];\n        float sf = shift_dn_recompute(2,",
        "new": "        float f1 = Hz[idx];\n        float sf = shift_dn_recompute(2,",
        "why": ("the thread's OWN cell load reverts to the pre-launch volume, so Dx's "
                "curl differences an H one sub-step behind on one of its two terms"),
        "live_on": "every fixture",
    },
    "f_w_H_written_in_place": {
        "old": "    *fw_out = src;",
        "new": "    const_cast<float*>(fw)[idx] = src;",
        "why": ("the split-field history is stored IN PLACE, so a foreign recompute "
                "reading fw at a cell another block already wrote gets B where its "
                "recurrence needs B_prev. RACY BY CONSTRUCTION: whether it fires is a "
                "schedule, which is why the deterministic twin below is armed beside "
                "it"),
        "live_on": "every fixture (schedule-dependent)",
        "predicted_null": ("schedule-dependent by construction; the deterministic twin "
                           "foreign_history_reads_the_post_store_value carries the "
                           "same arithmetic consequence and MUST be caught"),
    },
    "foreign_history_reads_the_post_store_value": {
        "old": "    float prev = fw[idx];",
        "new": "    float prev = src;",
        "why": ("THE DETERMINISTIC TWIN of the in-place store: the recurrence reads "
                "the value the in-place arrangement would have handed it, with no "
                "race to depend on"),
        "live_on": "every fixture with an active absorber (kms != 0)",
    },
    "accumulations_regrouped": {
        "old": "    return a - kms * prev;",
        "new": "    return f[idx] + (kps * src - kms * prev);",
        "why": ("the two SEPARATE accumulations become one expression, which is a "
                "different float32 number because addition is not associative. MEEP's "
                "step_update_EDHB accumulates left to right and the certified body "
                "transcribes that; flattening it reads as a slightly weaker absorber "
                "rather than as a bug"),
        "live_on": "every fixture with an active absorber",
    },
    "coefficient_index_is_the_threads_own": {
        "old": ("    int k = idx % nz;\n    int j = (idx / nz) % ny;\n"
                "    int i = idx / (ny * nz);\n\n    // Hx <- Bx"),
        "new": ("    int k = 0;\n    int j = 0;\n    int i = 0;\n\n    // Hx <- Bx"),
        "why": ("the recomputed cell's absorber coefficients are taken from the origin "
                "instead of from the recomputed cell -- the half-cell class of error, "
                "converged and smooth"),
        "live_on": "every fixture with an active absorber",
    },
    "the_metallic_ghost_is_the_clamped_cells_constitutive": {
        "old": "    return (bc == BC_PERIODIC) ? resolve_H(comp, idx + (na - 1) * stride, weld) : 0.0f;",
        "new": ("    return (bc == BC_PERIODIC) ? resolve_H(comp, idx + (na - 1) * "
                "stride, weld)\n                               : resolve_H(comp, idx, weld);"),
        "why": ("the metallic ghost serves update_H's value at the clamped cell "
                "instead of the exact 0.0f. RECORDED AS A NULL: the ghost_observability "
                "leg DERIVES from the emitted text that every tap is masked at exactly "
                "the plane its metallic ghost fires on, so nothing the ghost serves can "
                "reach step_D's output. What must bite is the PERIODIC sibling below"),
        "live_on": "no fixture -- derived null; see ghost_observability",
        "predicted_null": ("derived from the emitted text by the ghost_observability "
                           "leg: on the D side the set of (tap, plane) pairs the "
                           "metallic ghost fires on and the set the ownership mask "
                           "zeroes COINCIDE"),
    },
    "the_periodic_wrap_reads_the_near_row": {
        "old": "    return (bc == BC_PERIODIC) ? resolve_H(comp, idx + (na - 1) * stride, weld) : 0.0f;",
        "new": ("    return (bc == BC_PERIODIC) ? resolve_H(comp, idx + stride, weld) "
                ": 0.0f;"),
        "why": ("the periodic wrap reads the row ABOVE instead of the far row. Unlike "
                "the metallic ghost this IS observable: on a periodic axis the "
                "ownership mask does not zero the curl the wrapped tap feeds"),
        "live_on": "periodic, periodic_negative_beta, fold_y_* (their x axis wraps)",
    },
    "fmad_default_build": {
        "old": None,
        "new": None,
        "options": (),
        "why": ("the same source built WITHOUT --fmad=false, so NVRTC may contract the "
                "accidental multiply-adds the transcription does not spell -- of which "
                "this family has FOUR, including the beta insert's curl - (beta * f)"),
        "live_on": "every fixture",
    },
}

#: The one armed edit required NOT to diverge: the own-cell register replaced by a
#: RELOAD of the scratch word this thread has just stored -- the same value by a
#: different route. A byte-neutral control that DID diverge would mean the harness,
#: not the weld, decides the answer.
BYTE_NEUTRAL: Dict[str, str] = {
    "old": "        float f1 = own_h[2];\n        float sf = shift_dn_recompute(2,",
    "new": "        float f1 = Hz_out[idx];\n        float sf = shift_dn_recompute(2,",
    "why": ("the own-cell register is replaced by a RELOAD of the scratch word this "
            "thread has just stored -- the same value by a different route. A float32 "
            "stored to global memory and loaded back is the identity on the bits, so "
            "an edit that diverged here would mean the harness rather than the weld "
            "decides the answer"),
}

HOST_MUTATIONS: Tuple[str, ...] = ("rotation_skipped", "scratch_aliased_to_storage",
                                   "swap_constitutive_sublattice",
                                   "swap_beta_coefficient_words",
                                   "drop_the_metallic_wall_code")


# ---------------------------------------------------------------------------
# The adapter
# ---------------------------------------------------------------------------

class SpecialKzAdapter(kit.Adapter):
    """The REAL beta H->D weld's answers to the shared legs' questions."""

    complex_storage = False
    cell_arms = CELL_ARMS
    specs = SPECS
    mutation_spec_labels = MUTATION_SPEC_LABELS
    reduced_labels = REDUCED_LABELS
    value_classes = ("uniform", "subnormal_band")
    device_mutations = DEVICE_MUTATIONS
    host_mutations = HOST_MUTATIONS
    byte_neutral = BYTE_NEUTRAL
    #: The tuple order for an H->D span is (update_H's arm, step_D's arm). Both are
    #: this family's own ``real beta`` arm: ``covers_special_kz_fused_hd_pair`` opens
    #: with ``special_kz_curl.covers_special_kz_constitutive(..., "H")`` and continues
    #: into ``covers_special_kz_curl(..., "step_D")``, which ARE that arm's two
    #: predicates. The board's cell is the same pair of labels.
    wiring_arms_row = ("real beta", "real beta")
    wiring_module = "special_kz_fused_hd_pair"
    released_neighbours = ("cuda_special_kz_fused_magnetic_pair",
                           "cuda_special_kz_fused_electric_pair")
    env_stem = "MEEP_GPU_SKZHD_GATE"

    def __init__(self) -> None:
        from meep_gpu.cuda_kernels import special_kz_fused_hd_pair as family  # noqa: PLC0415
        self.family = family
        self.arm = "real"

    # -- the product ----------------------------------------------------------
    def predicate(self, fields, pml, grid, sources) -> Tuple[bool, str]:
        return self.family.covers_special_kz_fused_hd_pair(fields, pml, grid, sources)

    def resolve(self, fields, grid, pml, dtdx: float, **overrides) -> Dict[str, Any]:
        state = {
            "tables": overrides.get("tables")
                      or self.family.special_kz_fused_hd_pair_tables(pml),
            "codes": overrides.get("codes")
                     or self.family.special_kz_fused_hd_pair_codes(grid),
            "beta": overrides.get("beta") or self.family.beta_scalars(grid),
            "scratch": overrides.get("scratch")
                       or self.family.special_kz_fused_hd_pair_scratch(fields),
            "dtdx": float(overrides.get("dtdx", dtdx)),
        }
        return state

    def assert_disjoint(self, fields, state) -> int:
        return self.family.assert_scratch_is_disjoint(fields, state["scratch"],
                                                      state["tables"])

    def launch(self, fields, state, kernel=None, threads: Optional[int] = None):
        return self.family.launch_special_kz_fused_hd_pair(
            fields, state["scratch"], state["tables"], state["codes"], state["beta"],
            state["dtdx"], kernel,
            self.family._FUSED_THREADS if threads is None else int(threads))  # noqa: SLF001

    def rotate(self, fields, state) -> None:
        state["scratch"] = self.family.rotate_into_fields(fields, state["scratch"])

    def boundary_codes(self, grid):
        from meep_gpu.cuda_kernels import coverage  # noqa: PLC0415
        return coverage.real_curl_boundary_codes(grid)

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
        """The certified single-slot launchers for a REAL beta run.

        The two CURL slots go to ``special_kz_curl``'s certified pair -- the beta
        kernels -- and the two CONSTITUTIVE slots to the certified ordinary pair,
        which is what ``covers_special_kz_constitutive`` admits and what the census
        records at ``update_H``/``update_E`` on both corpus rows.
        """
        from meep_gpu.cuda_kernels import (constitutive_kernels,  # noqa: PLC0415
                                           coverage, special_kz_curl,
                                           step_curl_kernels)

        codes, refusal = coverage.real_curl_boundary_codes(grid)
        if refusal is not None:
            raise SystemExit(f"the certified singles refuse this fixture: {refusal}")
        codes = tuple(np.int32(int(code)) for code in codes)
        curl_tables = {sub: step_curl_kernels.real_pml_curl_tables(pml, half)
                       for sub, half in (("step_B", True), ("step_D", False))}
        constitutive = {
            side: constitutive_kernels.real_constitutive_tables(
                pml, coverage.constitutive_sub_lattice(side))
            for side in ("H", "E")}
        beta = special_kz_curl.beta_curl_coefficients(grid.beta, grid.dt)

        def curl(sub_step: str):
            return lambda: special_kz_curl.step_special_kz(
                sub_step, fields, curl_tables[sub_step], codes, dtdx, beta)

        return {
            "step_B": curl("step_B"),
            "step_D": curl("step_D"),
            "update_H": lambda: constitutive_kernels.update_fused_pml_real(
                "H", fields, tables=constitutive["H"]),
            "update_E": lambda: constitutive_kernels.update_fused_pml_real(
                "E", fields, tables=constitutive["E"]),
        }

    def released_pairs(self, fields, grid, pml, dtdx: float):
        from meep_gpu.cuda_kernels import (  # noqa: PLC0415
            special_kz_fused_electric_pair as electric,
            special_kz_fused_magnetic_pair as magnetic)

        def run_magnetic():
            return magnetic.run_special_kz_fused_magnetic_pair(
                fields, grid, pml, dtdx, sources=())

        def run_electric():
            return electric.run_special_kz_fused_electric_pair(
                fields, grid, pml, dtdx, sources=())

        return magnetic, electric, run_magnetic, run_electric

    # -- the host legs --------------------------------------------------------
    def leg_transcription(self) -> Dict[str, Any]:
        """Is every welded byte the certified family's, and does the lift INVERT?

        SIX CHECKS AND ONE EQUALITY. The equality is the strongest thing this leg has
        and it is a NEW measurement rather than a restatement: this family's curl
        transform applied to the CERTIFIED (beta-free) ``step_D_pml_real`` text
        returns exactly what the RELEASED ``fused_hd_pair.welded_curl_tail`` returns,
        byte for byte. The beta weld is therefore the released, gated transform plus
        two beta assertions -- measured, not claimed.
        """
        from meep_gpu.cuda_kernels import (constitutive_kernels,  # noqa: PLC0415
                                           fused_hd_pair, step_curl_kernels)

        family = self.family
        source = family.kernel_source()
        prelude = fused_hd_pair.constitutive_prelude()
        inverted = prelude
        for edit in fused_hd_pair.CONSTITUTIVE_LIFT_EDITS:
            del edit  # the table is prose; the anchors are the module's constants
        for old, new in ((fused_hd_pair._APPLY_SIGNATURE,  # noqa: SLF001
                          fused_hd_pair._APPLY_SIGNATURE_PURE),  # noqa: SLF001
                         (fused_hd_pair._APPLY_PARAMETERS,  # noqa: SLF001
                          fused_hd_pair._APPLY_PARAMETERS_PURE),  # noqa: SLF001
                         (fused_hd_pair._APPLY_FW_STORE,  # noqa: SLF001
                          fused_hd_pair._APPLY_FW_STORE_PURE),  # noqa: SLF001
                         (fused_hd_pair._APPLY_F_STORE,  # noqa: SLF001
                          fused_hd_pair._APPLY_F_STORE_PURE)):  # noqa: SLF001
            inverted = inverted.replace(new, old, 1)
        mine = family.welded_curl_tail(
            step_curl_kernels._step_D_pml_real_kernel_code)  # noqa: SLF001
        released = fused_hd_pair.welded_curl_tail()
        checks = {
            "ascii": kit.cyl.encodes_ascii(source),
            "the_lift_inverts_to_the_certified_prelude":
                inverted == constitutive_kernels._REAL_CONSTITUTIVE_PRELUDE,  # noqa: SLF001
            "the_kernel_name_is_declared":
                f'extern "C" __global__ void {family.KERNEL_NAME}(' in source,
            "shift_dn_recompute_keeps_the_near_neighbour_branch":
                "if (ia > 0) return resolve_H(comp, idx - stride, weld);" in source,
            "shift_dn_recompute_keeps_the_wrap_arithmetic":
                "resolve_H(comp, idx + (na - 1) * stride, weld)" in source,
            "the_metallic_ghost_is_still_an_exact_zero": ": 0.0f;" in source,
            "no_magnetic_pointer_survives_in_the_curl_half": "shift_dn(H" not in source,
            "the_transform_on_the_certified_text_is_the_released_transform":
                mine == released,
            "the_two_beta_increments_are_present":
                source.count("curl = curl - (beta_") == family.BETA_INSERT_LINES,
            "both_beta_partners_are_weld_registers": all(
                f"float {register} = own_h[" in source
                for register in ("f1", "f2")),
            "the_beta_scalars_are_bound": (
                "float beta_plus, float beta_minus" in source),
            "no_complex_type_appears": "cf " not in source and "cf_load" not in source,
        }
        needles = kit.mutation_needles_resolve(self)
        checks["every_armed_needle_resolves_to_exactly_one_site"] = needles["passed"]
        return {
            "kernel": family.KERNEL_NAME,
            "lines": len(source.splitlines()),
            "sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
            "source_digest": family.source_digest(),
            "shifted_taps": source.count("shift_dn_recompute(") - 1,
            "own_cell_registers": source.count("own_h["),
            "mutation_needles": needles,
            "checks": checks,
            "what_the_equality_licenses": (
                "that this family's curl transform IS the released real H->D weld's, "
                "with the beta text as its only other input -- so every clause "
                "gate_cuda_fused_hd_pair.py measured about that transform carries, and "
                "what this gate adds is the beta insert and the beta partner"),
            "passed": all(checks.values())
                      and source.count("shift_dn_recompute(") - 1 == family.HALO_TAPS,
        }

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

        class _BetaGrid(_HostGrid):
            beta = 0.25

        class _Fields:
            pass

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

        # 1. beta = 0 belongs to the certified real pair, and is refused BY NAME.
        check("beta_is_zero",
              lambda: family.covers_special_kz_fused_hd_pair(_Fields(), None,
                                                             _HostGrid(), ()),
              False, "grid.beta is zero")

        # 2. COMPLEX storage with beta is the complex-beta family's, named.
        class _ComplexFields:
            force_complex_fields = True

        check("complex_storage_with_beta",
              lambda: family.covers_special_kz_fused_hd_pair(_ComplexFields(), None,
                                                             _BetaGrid(), ()),
              False, "complex64 storage with beta")

        # 3. An OFF-DIAGONAL epsilon with beta in real storage: stepping RAISES and
        #    MEEP aborts, so there is no such run.
        class _OffDiagonal:
            force_complex_fields = False
            has_offdiagonal_epsilon = True

        check("offdiagonal_epsilon_with_beta",
              lambda: family.covers_special_kz_fused_hd_pair(_OffDiagonal(), None,
                                                             _BetaGrid(), ()),
              False, "off-diagonal epsilon")

        # 4. A CYLINDRICAL grid with beta: MEEP aborts.
        class _Cylindrical(_BetaGrid):
            cylindrical = True

        check("cylindrical_with_beta",
              lambda: family.covers_special_kz_fused_hd_pair(_Fields(), None,
                                                             _Cylindrical(), ()),
              False, "cylindrical")

        # 5. A 3-D grid with beta.
        class _ThreeD(_BetaGrid):
            dimensions = 3

        check("three_dimensional_with_beta",
              lambda: family.covers_special_kz_fused_hd_pair(_Fields(), None,
                                                             _ThreeD(), ()),
              False, "dimensions")

        # 6. An undeclared source set: the seam clause cannot infer an absence.
        check("sources_undeclared",
              lambda: family.covers_special_kz_fused_hd_pair(_Fields(), None,
                                                             _BetaGrid(), None),
              False, "")

        # 7. The withdraw refusal's text, on a synthetic standing source.
        class _Withdrawing:
            # The three conditions ``withdraw_hoist._withdraw_does_work`` gates on --
            # a callable ``withdraw``, ``is_integrated``, and at least one source
            # point. ``_applied_dipole`` is the PER-STEP gate and is deliberately
            # absent.
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

        # 8. The transform refuses a text whose beta partner is NOT the weld register.
        stale = {"passed": False}
        try:
            corrupted = family._certified_curl_source().replace(  # noqa: SLF001
                "        float f2 = Hy[idx];\n"
                "        float ss = shift_dn(Hy, idx, k, nz, sz, bc_z);\n"
                "        float curl = dtdx * ((sf - f1) + (f2 - ss));\n"
                "        curl = curl - (beta_plus * f2);",
                "        float f2 = Hy[idx];\n"
                "        float ss = shift_dn(Hy, idx, k, nz, sz, bc_z);\n"
                "        float curl = dtdx * ((sf - f1) + (f2 - ss));\n"
                "        float f9 = Hy[idx];\n"
                "        curl = curl - (beta_plus * f9);", 1)
            family.welded_curl_tail(corrupted)
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
        """``float32 / float32`` and ``complex64 * float32``, RECORDED AS ABSENCES.

        THE MEASURED FACTS THIS FAMILY MUST RESPECT AND DOES NOT EXERCISE. On this
        backend ``complex64 * float32`` is the CONTRACTED four-product form and
        ``complex64 / float32`` is CuPy's SCALED complex/complex algorithm -- neither
        is the numpy spelling, and carrying one backend's or one storage class's
        prescription to another is the defect those measurements exist to prevent.
        REAL ``float32 / float32`` is a plain true divide.

        This family has NO SITE for any of the three: its text is real float32
        throughout and it performs no division at all. That is asserted against the
        emitted text -- with the integer index decode enumerated as the one exception
        and its count checked -- and the assertion is ARMED: a planted floating-point
        divide must be refused by the family's own emitter.
        """
        family = self.family
        source = family.kernel_source()
        armed = {"refused": None, "passed": False}
        try:
            family._assert_the_measured_absences(  # noqa: SLF001
                source.replace("        float curl = dtdx * ((sf - f1) + (f2 - ss));",
                               "        float curl = ((sf - f1) + (f2 - ss)) / dtdx;",
                               1))
        except AssertionError as error:
            armed = {"refused": str(error)[:300], "passed": True}
        complex_armed = {"refused": None, "passed": False}
        try:
            family._assert_the_measured_absences(  # noqa: SLF001
                source.replace("    float own_h[3];", "    cf own_h[3];", 1))
        except AssertionError as error:
            complex_armed = {"refused": str(error)[:300], "passed": True}
        checks = {
            "the_shipped_text_carries_no_complex_type":
                "cf " not in source and "cf_load" not in source,
            "the_shipped_text_passes_its_own_absence_assertion": True,
            "a_planted_float_divide_is_refused": armed["passed"],
            "a_planted_complex_type_is_refused": complex_armed["passed"],
        }
        return {
            "kernel": family.KERNEL_NAME,
            "permitted_divide_lines": list(family._PERMITTED_DIVIDE_LINES),  # noqa: SLF001
            "permitted_divides": family._PERMITTED_DIVIDES,  # noqa: SLF001
            "planted_float_divide": armed,
            "planted_complex_type": complex_armed,
            "checks": checks,
            "what_this_records": (
                "an ABSENCE, twice, each with a control that bites. The complex "
                "multiply's contraction and the complex divide's scaled algorithm are "
                "the two spellings a reader arriving from the complex or cylindrical "
                "H->D products will look for; this family has no site for either, and "
                "REAL float32 division -- which is an ordinary true divide, a THIRD "
                "spelling again -- has no site either"),
            "passed": all(checks.values()),
        }

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
                                "the IN-PLACE weld the board refused -- every foreign "
                                "recompute would read words other blocks had already "
                                "overwritten and the answer would be a schedule")}
        if name == "swap_constitutive_sublattice":
            from meep_gpu.cuda_kernels import (constitutive_kernels,  # noqa: PLC0415
                                               step_curl_kernels)
            fields, grid, pml, dtdx, _plant = self.build(
                spec, "uniform", kit.case_rng(self.seed, "swap"))
            curl = step_curl_kernels.real_pml_curl_tables(pml, False)
            half = constitutive_kernels.real_constitutive_tables(pml, True)
            tables = {"kms": {axis: curl[f"kms_{axis}"] for axis in "xyz"},
                      "sinv": {axis: curl[f"sinv_{axis}"] for axis in "xyz"},
                      "kps": {axis: half[f"kps_{axis}"] for axis in "xyz"}}
            record = kit.drive(self, spec, "uniform", min(steps, 8),
                               modes=("array", "weld"),
                               state_overrides={"tables": tables})
            return {"caught": not record["identical"]["weld"],
                    "first_divergence": record["first_divergence"]["weld"],
                    "why": ("update_H's kps taken from the HALF-INTEGER sub-lattice "
                            "-- a half-cell error in the absorber profile, converged "
                            "and smooth")}
        if name == "swap_beta_coefficient_words":
            fields, grid, pml, dtdx, _plant = self.build(
                spec, "uniform", kit.case_rng(self.seed, "beta"))
            plus, minus = family.beta_scalars(grid)
            record = kit.drive(self, spec, "uniform", min(steps, 8),
                               modes=("array", "weld"),
                               state_overrides={"beta": (minus, plus)})
            return {"caught": not record["identical"]["weld"],
                    "first_divergence": record["first_divergence"]["weld"],
                    "coefficients": [float(plus), float(minus)],
                    "why": ("the two host-rounded call-site coefficients are handed "
                            "over in the wrong order. The kernel cannot tell them "
                            "apart; only the host transcription can be wrong here, "
                            "which is why this is a HOST mutation")}
        if name == "drop_the_metallic_wall_code":
            # SCORED ON A WALLED FIXTURE, and the fixture is chosen rather than
            # inherited: binding every axis PERIODIC on an already-periodic grid
            # changes nothing at all, and this mutation came back UNCAUGHT on the
            # kit's first host-mutation smoke because it ran on ``periodic``.
            walled = next(case for case in self.specs
                          if case["boundaries"] and "metallic" in case["boundaries"])
            fields, grid, pml, dtdx, _plant = self.build(
                walled, "uniform", kit.case_rng(self.seed, "codes"))
            codes = tuple(0 for _ in family.special_kz_fused_hd_pair_codes(grid))
            record = kit.drive(self, walled, "uniform", min(steps, 8),
                               modes=("array", "weld"),
                               state_overrides={"codes": codes})
            return {"caught": not record["identical"]["weld"],
                    "first_divergence": record["first_divergence"]["weld"],
                    "fixture": walled["label"],
                    "why": ("every axis bound as PERIODIC, so a metallic wall's mask "
                            "and its exact-zero ghost both vanish and the wrap reads "
                            "the far row instead")}
        raise ValueError(f"no such host mutation {name!r}")

    # -- wiring ---------------------------------------------------------------
    def wiring_product_row(self) -> Dict[str, Any]:
        from meep_gpu.cuda_kernels import fused_pairs  # noqa: PLC0415
        from meep_gpu.triton_kernels.coverage import Coverage  # noqa: PLC0415

        family = self.family

        def coverage(context: Any) -> Coverage:
            covered, reason = family.covers_special_kz_fused_hd_pair(
                context.fields, context.pml, context.grid, context.sources)
            return Coverage(bool(covered), (str(reason),))

        def plan(context: Any):
            held: Dict[str, Any] = {}

            def resolve(ctx: Any) -> Dict[str, Any]:
                if "tables" not in held:
                    held["tables"] = family.special_kz_fused_hd_pair_tables(ctx.pml)
                    held["codes"] = family.special_kz_fused_hd_pair_codes(ctx.grid)
                    held["beta"] = family.beta_scalars(ctx.grid)
                    held["scratch"] = family.special_kz_fused_hd_pair_scratch(ctx.fields)
                return dict(held)

            def launch(fields: Any, arguments: Dict[str, Any]) -> Dict[str, Any]:
                dtdx = float(context.grid.dt / context.grid.dx)
                record = family.launch_special_kz_fused_hd_pair(
                    fields, arguments["scratch"], arguments["tables"],
                    arguments["codes"], arguments["beta"], dtdx)
                held["scratch"] = family.rotate_into_fields(fields,
                                                            arguments["scratch"])
                return record

            return fused_pairs.CudaFusedPairPlan(
                family=family.FAMILY, label="special_kz fused H/D pair",
                kernel_label=family.KERNEL_NAME, replaces=tuple(family.REPLACES),
                slots=("update_H", "step_D"), context=context,
                resolve_launch_args=resolve, launch=launch)

        return {"curl_slot": "update_H", "coverage": coverage, "plan": plan,
                "module": self.wiring_module}


LICENCES_WHAT = (
    "ONE CUDA kernel measured byte-identical, per COMPLETE DRIVER STEP and as uint32 "
    "words over every stored volume, to four independent arrangements of the "
    "update_H -> step_D seam on a REAL float32 special_kz (grid.beta != 0) run, at "
    "every block size and at three seed scales, on the fixtures and corpus rows this "
    "record names, under the installed float32 subnormal policy. It licenses NO "
    "throughput claim, NO dispatch claim and NO composition claim: the family declares "
    "INSTALLABLE = False, is not in the composer's tables on disk, and a credited "
    "seam-instance is PREDICATE ADMISSION.")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = kit.make_parser(__doc__.splitlines()[0])
    args = parser.parse_args(argv)
    adapter = SpecialKzAdapter()

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
