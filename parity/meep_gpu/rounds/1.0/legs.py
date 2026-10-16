#!/usr/bin/env python3
"""The 1.0 certification round's leg tables: one home for every gate the round runs.

Every NVIDIA leg the round runs is spelled here once, with the expected wall time it
took on one RTX A6000 (the 1.0 round's CERT-3 for the CUDA gate legs, the Triton fleet
and the hand legs; the 1.0 final pre-certification pass for the route legs and the
end-to-end licence), and the shell drivers ask this file for their plans (``plan``),
their checks (``check``) and their estimates (``estimate``).
It imports nothing from the package for the tables; ``check`` imports the checkout
being certified, read-only.

WHERE THE TABLES COME FROM.

* CUDA ``cited``: the 70 legs of ``rounds/h200_2026-10-03/run_cuda_welds.sh`` (35 gate
  pairs, keep and flush).
* CUDA ``residue``: the 46 legs of the 0.9.2 residue lane (23 families, keep and
  flush) and its ADE ``update_P`` re-cut (2 legs, a SIBLING root, never inside a
  campaign: ``record_cuda_regate.py`` refuses a subdirectory no block claims).
* Triton: ``rounds/h200_2026-10-03/run_round.sh``'s shards A (33), B (11), C (1), D
  (5), the two stencil welds and the bit identity; the 0.9.2 close round's three
  (``fused_ade_chain``, ``probe_triton_folded_deposit_closure``,
  ``complex_fused_ade_chain``); and the fleet driver's nine other gates (tier 2). 62
  fleet gates, every one in ``drive_triton_weld_gates.GATES``.
* Triton H->D: the ten ``gate_triton_*fused_hd_pair.py`` gates under keep, every leg,
  lifts on, one root per gate (``TRITON_HD``): the gates the Triton board's GATE_BOUND
  table binds and no other step runs. Hand legs, not fleet gates.
* Route: the 0.9.2 close round's ``run_route_lane.sh`` (blocking preflight, the
  ``cuda_alone`` preflight, the six CUDA legs; the Triton preflight, four legs and the
  end-to-end licence).

WHAT IS NEW FOR 1.0 (DESIGN.md R13, the flush dispatch licence).

* BOTH POLICIES ON BOTH NVIDIA TABLES. CUDA already ran keep and flush legs. Triton ran
  keep only; 1.0 runs a second fleet under
  ``--policy flush`` into ``<fleet>_flush`` roots, which
  ``rebind_triton_welds.py --flush-campaign`` reads.
* THE CUDA LANE SPLITS ACROSS GPUs by family subdirectory: one campaign root per GPU
  (``cuda_regate_<stamp>_g<k>``), each recorded by its own ``record_cuda_regate.py
  --campaign`` and rebound by its own ``rebind_cuda_welds.py --campaign``. MEASURED:
  the 0.9.2 ledger at fa59c2f binds 31 welds from ``cuda_regate_2026-10-03_cc86`` and
  23 from ``cuda_regate_2026-10-05_cc86_residue`` on compute capability 8.6, i.e. two
  campaigns, two blocks, two rebinds. The constraint is the join: a block claims a
  WHOLE subdirectory and the rebind requires the re-gate to have cut exactly the legs
  the entry stands on, so both policies of a subdirectory (and every family sharing
  it) land in the same campaign.
* NO EXPANSION LICENCE. The complex families take no expansion record since the
  licence was deleted, so no leg here carries a probe argument, the fleet runs no
  ``unified_expansion`` gate, and the route runs no probe legs.

THE MEEP CORPUS. The four CUDA H->D legs (``--lift-steps``) and the ten Triton H->D hand
legs re-run MEEP corpus rows, read from ``<root>/examples`` and ``<root>/tests``. The round
names the root once (:func:`resolve_corpus`): ``--corpus-root``, else the root the reference
environment's build record names (``<sys.prefix of --python>/share/meep-gpu-build/
reference_build.json``, ``corpus``: the release tarball's ``python/`` tree the build keeps in
the prefix, and the list of every file's sha256 beside it), and either way checks the tree
against that list, and the corpus siblings the record names (files outside ``python/`` that
corpus rows read, kept beside it) by name and sha256; ``check``
refuses a gate stage whose lift legs would find no verified corpus, and ``round.py`` exports
the root to every lane as ``MEEP_GPU_CORPUS_ROOT``. The gates' own lookup chain
(:func:`corpus_root`) is unchanged and is not what the round reads. ``metal_round.py`` reads
the same record by the same rule.

A NEW CAPABILITY (``--new-capability``). A family gate does not go through dispatch admission,
so it runs on any card; only the route and licence legs need the ledgers to admit the card.
``check`` refuses a gate stage on a capability a table does not admit unless the flag says the
capability is new (one the ledgers hold no live record for: an H200 at 9.0), refuses the flag
on a capability both tables already admit, and refuses it on the route stage, which runs only
once the bind has added the capability's records. A new capability runs chunked
(``chunks.py partition --capability 9.0 --new-device``), so every root it writes carries its
device; its licence lands in ``<licence campaign>_cc<capability>`` as every capability's does.

THE TRITON H->D INPUTS. The ten Triton H->D lifts also join their rows to a census and a
seam record under ``parity/meep_gpu/results`` (each gate's ``CENSUS`` and ``SEAM_RECORD``,
read from its source by :func:`triton_hd_inputs`), which a checkout does not carry; ``check``
refuses a gate stage on which either is absent or lacks a file the lift reads
(:data:`TRITON_HD_INPUT_FILES`). Measured on the 1.0 round: with the census absent the first
such leg ran every device leg and then stopped at its lift, 113 s into a 3,052 s leg.

usage:
  legs.py check     --repo R [--capability 8.6] [--new-capability] [--route]
                    [--corpus-root DIR | --python <reference environment>/bin/python]
  legs.py plan cuda --repo R --results RES --stamp S --split K --out DIR [--only id,..]
  legs.py plan triton --repo R --results RES --stamp S --out DIR
  legs.py plan route --table cuda|triton --repo R --results RES --stamp S --out DIR
  legs.py estimate  [--gpus N] [--tables cuda,triton]
  legs.py names     --repo R --stamp S --capability 8.6           (route campaign names)
  legs.py bind      --results RES --stamp S [--cuda-split K] [--write]
  legs.py bind      --chunks INDEX --selection F [--write]   (a chunked round: chunks.py bind)
  legs.py check     --repo R --chunk-index INDEX --chunk c<k> ...  (that chunk's roots fresh)
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.machinery
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent

#: The order every explicit route case list leads with: the lazy-install
#: process-history hazard (a CUDA case fold-refused before rung 8b leaves the array
#: legs' CuPy kernels compiled before ``keep`` was installed) is avoided only when the
#: first case dispatches.
LEAD_CASE = "pml_2d"

# ---------------------------------------------------------------------------- CUDA ----
#: (subdirectory, leg template with @P@, "file"|"dir", expected seconds keep, flush,
#:  script, args...).
#:
#: EXPECTED SECONDS ARE MEASURED, for this table, CUDA_RESIDUE and CUDA_ADE_EXPECT: each
#: leg's wall in the 1.0 round's CERT-3 NVIDIA (stamp 2026-10-11_10, frozen commit
#: 2b75bef8; one RTX A6000 per lane, two CUDA campaigns beside the two Triton core
#: lanes, a host other jobs loaded), rounded up. The 0.9.2 walls they replace priced the
#: CUDA work at 22,006 s against 29,843 s measured, so the planner gave the CUDA lanes
#: too much and two Triton lanes sat idle for about two hours. A price is RAISED above its
#: wall where run_lane.py's step watchdog, max(1,800 s, 4 x expect_s), would fall below 3x
#: the larger of the leg's CERT-3 wall and its wall in the final pre-certification pass
#: (five CUDA campaigns on one card): there it is ceil(0.75 x that larger wall), 13 of 118
#: legs (``cuda_conductive`` keep 954 -> 1,316 s against 1,754 s on the shared card). A
#: watchdog kill costs a re-run of the leg.
CURL = "@REPO@/meep_gpu/cuda_kernels/step_curl_kernels.py"
CUDA_CITED: Tuple[tuple, ...] = (
    ("fused_pml_bit_identity_hand_2026-08-22_rename", "@P@/probe.json", "file", 95, 113,
     "probe_fused_kernel_bit_identity.py", "--module", CURL, "--track", "hand",
     "--experiments", "pml,multistep", "--coefficients", "synthetic,real"),
    ("cuda_constitutive_recut_2026-08-22_rename", "@P@/cgate.json", "file", 112, 125,
     "probe_fused_kernel_bit_identity.py", "--module", CURL, "--track", "hand",
     "--experiments", "constitutive,constitutive_multistep", "--coefficients", "synthetic,real"),
    ("cuda_offdiag_2026-08-21_rename", "@P@/gate.json", "file", 326, 250,
     "gate_cuda_offdiag.py", "--backend", "cupy", "--product", "full"),
    ("cuda_dispersive_offdiag_2026-08-20b", "@P@/gate.json", "file", 676, 642,
     "gate_cuda_dispersive_offdiag.py", "--backend", "cupy", "--product", "full"),
    ("cuda_folded_offdiag_kernel_2026-08-21_rename", "@P@/gate.json", "file", 148, 162,
     "gate_cuda_folded_offdiag_kernel.py", "--backend", "cupy", "--product", "full"),
    ("cuda_in_seam_2026-08-21_final", "fill_folded_far/@P@/gate.json", "file", 33, 36,
     "gate_cuda_in_seam_passes.py", "--pass", "fill_folded_far", "--backend", "cuda"),
    ("cuda_in_seam_2026-08-21_final", "fill_symmetry/@P@/gate.json", "file", 30, 36,
     "gate_cuda_in_seam_passes.py", "--pass", "fill_symmetry", "--backend", "cuda"),
    ("cuda_in_seam_2026-08-21_final", "zero_metal/@P@/gate.json", "file", 27, 29,
     "gate_cuda_in_seam_passes.py", "--pass", "zero_metal", "--backend", "cuda"),
    ("cuda_fused_magnetic_pair_2026-08-27", "@P@/gate.json", "file", 120, 153,
     "gate_cuda_fused_magnetic_pair.py"),
    ("cuda_fused_electric_pair_2026-08-31_fillcarry_r3", "@P@/gate.json", "file", 142, 161,
     "gate_cuda_fused_electric_pair.py"),
    ("cuda_fused_polarization_pair_2026-09-01_pml", "@P@/gate.json", "file", 366, 770,
     "gate_cuda_fused_polarization_pair.py", "--arm", "pml", "--product", "full"),
    ("cuda_fused_polarization_pair_2026-09-01_no_pml", "@P@/gate.json", "file", 110, 109,
     "gate_cuda_fused_polarization_pair.py", "--arm", "no_pml", "--product", "full"),
    ("cuda_conductive_fused_electric_pair_2026-09-02b", "@P@/gate.json", "file", 73, 80,
     "gate_cuda_conductive_fused_electric_pair.py"),
    ("cuda_offdiag_stencil_welds_2026-09-02", "@P@/gate.json", "file", 155, 213,
     "gate_cuda_offdiag_stencil_welds.py", "--product", "full", "--steps", "60", "--repeats", "4"),
    ("cuda_dispersive_offdiag_polarization_pair_2026-09-02", "@P@/gate.json", "file", 246, 336,
     "gate_cuda_dispersive_offdiag_polarization_pair.py", "--product", "full", "--steps", "60",
     "--repeats", "4"),
    ("cuda_three_slot_weld_2026-09-11_regate", "@P@_pml/gate.json", "dir", 80, 92,
     "gate_cuda_three_slot_weld.py", "--arms", "pml"),
    ("cuda_three_slot_weld_2026-09-11_regate", "@P@_no_pml/gate.json", "dir", 48, 57,
     "gate_cuda_three_slot_weld.py", "--arms", "no_pml"),
    ("cuda_residue_complex_polarization_pair_2026-09-02b", "@P@/gate.json", "file", 30, 37,
     "gate_cuda_complex_polarization_pair.py", "--product", "full"),
    ("cuda_complex_three_slot_weld_2026-09-04_gatefix", "@P@/gate.json", "file", 134, 128,
     "gate_cuda_complex_three_slot_weld.py"),
    ("cuda_fused_complex_pairs_2026-09-11_regate", "complex_@P@/gate.json", "file", 148, 58,
     "gate_cuda_fused_complex_pairs.py", "--family", "complex"),
    ("cuda_fused_complex_pairs_2026-09-11_regate", "complex_electric_@P@/gate.json", "file",
     89, 149, "gate_cuda_fused_complex_pairs.py", "--family", "complex_electric"),
    ("cuda_fused_complex_pairs_2026-09-11_regate", "no_pml_complex_electric_@P@/gate.json",
     "file", 75, 30, "gate_cuda_fused_complex_pairs.py", "--family",
     "no_pml_complex_electric"),
    ("cuda_fused_complex_pairs_2026-09-13_cyl", "cylindrical_@P@/gate.json", "file", 113, 104,
     "gate_cuda_fused_complex_pairs.py", "--family", "cylindrical"),
    ("cuda_fused_complex_pairs_2026-09-13_cyl", "cylindrical_electric_@P@/gate.json", "file",
     115, 100, "gate_cuda_fused_complex_pairs.py", "--family", "cylindrical_electric"),
) + tuple(
    (f"cuda_fused_residual_{fam}_2026-09-02", "@P@/gate.json", "file", keep, flush,
     "gate_cuda_fused_complex_pairs.py", "--family", fam, "--product", "full")
    for fam, keep, flush in (("bfast", 35, 45), ("complex_beta", 100, 93),
                             ("complex_folded", 88, 92), ("cylindrical_real", 71, 50),
                             ("special_kz", 52, 55))
) + tuple(
    (f"cuda_residue_{fam}_electric_2026-09-02", "@P@/gate.json", "file", keep, flush,
     "gate_cuda_fused_complex_pairs.py", "--family", f"{fam}_electric", "--product", "full")
    for fam, keep, flush in (("bfast", 43, 58), ("complex_beta", 68, 72),
                             ("cylindrical_real", 61, 59), ("special_kz", 86, 95))
) + (
    ("cuda_residue_folded_complex_electric_2026-09-02", "@P@/gate.json", "file", 90, 91,
     "gate_cuda_fused_complex_pairs.py", "--family", "folded_complex_electric", "--product",
     "full"),
    ("cuda_complex_offdiag_stencil_welds_2026-09-17", "@P@/gate.json", "file", 434, 308,
     "gate_cuda_complex_offdiag_stencil_welds.py"),
)

#: The residue lane: (subdirectory, wrap the flush leg in meep_host_import.py, peak
#: processes, expected keep, flush, script, args with @P@/@HOST@). @HOST@ is
#: --import-meep-for-host-policy on a flush leg that is not wrapped, dropped otherwise.
CUDA_RESIDUE: Tuple[tuple, ...] = (
    ("cuda_ade_2026-08-19", 0, 1, 78, 105, "gate_cuda_ade.py", "--backend", "cupy", "--product",
     "full", "--subnormal-policy", "@P@", "@HOST@"),
    ("cuda_complex_regate_2026-08-26", 0, 1, 619, 668, "gate_cuda_complex.py", "--backend", "cupy",
     "--product", "full", "--subnormal-policy", "@P@", "@HOST@"),
    ("cuda_bfast_2026-08-21_rename_r2", 0, 1, 156, 178, "gate_cuda_bfast.py", "--subnormal-policy",
     "@P@", "@HOST@"),
    ("cuda_complex_beta_2026-09-01", 1, 1, 191, 184, "gate_cuda_complex_beta.py", "--backend",
     "cupy", "--product", "full", "--subnormal-policy", "@P@"),
    ("cuda_complex_folded_2026-09-01", 1, 1, 151, 176, "gate_cuda_complex_folded.py", "--backend",
     "cupy", "--product", "full", "--subnormal-policy", "@P@"),
    ("cuda_complex_no_pml_2026-08-27", 0, 1, 624, 402, "gate_cuda_complex_no_pml.py",
     "--subnormal-policy", "@P@", "@HOST@"),
    ("cuda_residue_complex_polarization_pair_2026-09-02", 0, 1, 73, 43,
     "gate_cuda_complex_polarization_pair.py", "--product", "full", "--subnormal-policy", "@P@",
     "@HOST@"),
    ("cuda_conductive_2026-08-21_rename", 0, 1, 1316, 1277, "gate_cuda_conductive.py",
     "--subnormal-policy", "@P@", "@HOST@"),
    ("cuda_cylindrical_complex_2026-08-26", 0, 1, 859, 744, "gate_cuda_cylindrical_complex.py",
     "--subnormal-policy", "@P@", "@HOST@"),
    ("cuda_cylindrical_fused_hd_pair_2026-09-07", 0, 2, 527, 560,
     "gate_cuda_cylindrical_fused_hd_pair.py", "--subnormal-policy", "@P@", "@HOST@", "--steps",
     "60", "--lift-steps", "60"),
    ("cuda_cylindrical_real_2026-08-26", 0, 1, 102, 105, "gate_cuda_cylindrical_real.py",
     "--subnormal-policy", "@P@", "@HOST@"),
    ("cuda_cylindrical_real_fused_hd_pair_2026-09-07", 0, 2, 95, 108,
     "gate_cuda_cylindrical_real_fused_hd_pair.py", "--subnormal-policy", "@P@", "@HOST@",
     "--steps", "60", "--lift-steps", "60"),
    ("cuda_dispersive_2026-08-26", 0, 1, 384, 463, "gate_cuda_dispersive.py", "--subnormal-policy",
     "@P@", "@HOST@"),
    ("cuda_fused_magnetic_pair_2026-08-28_fillcarry", 0, 1, 152, 137,
     "gate_cuda_fused_magnetic_pair.py", "--subnormal-policy", "@P@", "@HOST@", "--steps", "60"),
    ("cuda_fused_magnetic_pair_2026-09-01_nonlinear", 0, 1, 126, 209,
     "gate_cuda_fused_magnetic_pair.py", "--subnormal-policy", "@P@", "@HOST@", "--product",
     "full"),
    ("cuda_no_pml_curl_2026-08-21_rename", 0, 1, 650, 617, "gate_cuda_no_pml_curl.py",
     "--subnormal-policy", "@P@", "@HOST@"),
    ("cuda_nonlinear_2026-08-21_rename", 0, 1, 161, 172, "gate_cuda_nonlinear.py",
     "--subnormal-policy", "@P@", "@HOST@"),
    ("cuda_special_kz_2026-08-21_rename_r2", 0, 1, 192, 147, "gate_cuda_special_kz.py",
     "--subnormal-policy", "@P@", "@HOST@"),
    ("cuda_residue_special_kz_fillcarry_2026-09-02", 0, 1, 50, 54, "gate_cuda_fused_complex_pairs.py",
     "--family", "special_kz", "--product", "full", "--subnormal-policy", "@P@", "@HOST@"),
    ("cuda_folded_constitutive_2026-08-19", 0, 1, 226, 274, "gate_cuda_folded_constitutive.py",
     "--subnormal-policy", "@P@", "@HOST@", "--backend", "cuda", "--product", "full"),
    ("cuda_folded_constitutive_2026-08-21_rename", 0, 1, 183, 101,
     "gate_cuda_folded_constitutive.py", "--subnormal-policy", "@P@", "@HOST@"),
    ("cuda_complex_fused_hd_pair_2026-09-07", 0, 2, 556, 469, "gate_cuda_complex_fused_hd_pair.py",
     "--subnormal-policy", "@P@", "@HOST@", "--steps", "60", "--lift-steps", "60"),
    ("cuda_fused_hd_pair_2026-09-06", 0, 2, 3316, 3925, "gate_cuda_fused_hd_pair.py",
     "--subnormal-policy", "@P@", "@HOST@", "--steps", "60", "--lift-steps", "60", "--product",
     "full", "--lift-timeout", "3600"),
)
#: The ADE update_P re-cut, read by record_cuda_ade_recut.py from a sibling root.
CUDA_ADE_EXPECT = {"keep": 139, "flush": 142}

# -------------------------------------------------------------------------- Triton ----
TRITON_A = (
    "complex,fused_electric,no_pml_constitutive,offdiag,nonlinear,bfast,"
    "special_kz,folded_complex,cylindrical_complex,complex_ade,complex_no_pml_stored_e,"
    "no_pml_conductive,no_pml_stored_e,folded_offdiag,cylindrical_fused_electric_pair,"
    "cylindrical_fused_magnetic_pair,complex_conductive_fused_pair,folded_dispersive_fused_pair,"
    "cylindrical_real_fused_electric_pair,beta_fused_electric_pair,folded_beta_fused_electric_pair,"
    "folded_beta_fused_magnetic_pair,no_pml_fused_electric_pair,bfast_fused_electric_pair,"
    "conductive_fused_electric_pair,probe_triton_cylindrical_real_fused_magnetic_pair,"
    "probe_triton_dispersive_fused_pair,probe_triton_folded_fused_magnetic_pair,"
    "probe_triton_folded_fused_pair,probe_triton_fused_ade_state,beta_fused_magnetic_pair,"
    "bfast_fused_magnetic_pair").split(",")
TRITON_R092 = ["fused_ade_chain", "probe_triton_folded_deposit_closure"]
TRITON_TIER2 = ["conductivity", "symmetry", "cylindrical", "folded_offdiag_dispersive",
                "folded_offdiag_fused_ade_chain", "cylindrical_complex_composition",
                "fused_offdiag_electric", "probe_triton_fused_dispersive_chain", "source_seams"]
TRITON_B = (
    "complex_beta_fused_electric_pair,complex_beta_fused_magnetic_pair,complex_fused_electric_pair,"
    "folded_beta_complex_fused_magnetic_pair,folded_beta_complex_fused_pair,folded_complex_fused_pair,"
    "nonlinear_fused_magnetic_pair,probe_triton_complex_fused_magnetic_pair,complex_no_pml_conductive,"
    "complex_no_pml_curl,probe_triton_folded_complex_fused_magnetic_pair").split(",")
TRITON_R092_ENV = ["complex_fused_ade_chain"]
TRITON_C = ["complex_offdiag"]
TRITON_D = ["cylindrical_composition", "symmetry_composition", "conductivity_composition",
            "engine_route", "no_pml"]
#: The ``recut_composition_records.LEGS`` keys whose legs the composition step runs
#: (:data:`TRITON_D`), which the bind re-cuts from the composition roots with ``--only``.
#: The tool's LEGS table holds nine keys and refuses, all or nothing, a run root that
#: lacks a leg of a key it is asked to re-cut; the other four keys' legs run in the fleet
#: roots, not in the composition root. :func:`check` derives this set from the checkout's
#: LEGS and refuses a difference.
TRITON_COMPOSITION_KEYS = ("cylindrical_composition_gate", "symmetry_composition_gate",
                           "conductivity_composition_gate", "dispersive_composition_gate",
                           "no_pml_composition_gate")
TRITON_STENCIL = ["offdiag_fused_electric_pair", "folded_offdiag_fused_electric_pair"]
TRITON_BITID_EXPERIMENTS = ("pml,multistep,constitutive,constitutive_multistep,fused_pair,"
                            "fused_pair_multistep,wholestep")
#: Expected seconds per gate: MEASURED, the mean of the gate's keep and flush walls in the
#: 1.0 round's CERT-3 NVIDIA (stamp 2026-10-11_10, frozen commit 2b75bef8; one RTX A6000
#: per lane, the fleet driver running one gate at a time, every leg released), rounded
#: up. One number prices both policies: the two walls differ by noise, not by policy
#: (over the 61 gates, log(flush/keep) in CERT-3 and in the final pre-certification pass
#: correlate at 0.10 and agree in sign for 33), so the larger of the two would only add
#: noise. The pre-certification prices they replace were measured with four fleet gates
#: sharing a card (33,081 s per policy against CERT-3's 23,124 s keep and 23,319 s
#: flush), so the planner packed the two Triton core lanes at 23,864 s and they ran
#: 15,650 s and 17,305 s while the CUDA lanes ran 23,079 s and 24,675 s. The planner packs
#: lanes by these walls (``pack_lanes``) and ``run_lane.py`` derives each step's timeout
#: from them (4x plus 3,600 s per gate for a fleet step). Each fleet gate's own timeout is
#: the fleet driver's, which ``fleet_launch.py`` can only RAISE: its floors hold every
#: gate at 3x or more the larger of its CERT-3 and pre-certification walls.
TRITON_EXPECT: Dict[str, int] = {
    "complex": 377, "fused_electric": 63, "no_pml_constitutive": 34, "offdiag": 109,
    "nonlinear": 196, "bfast": 1082, "special_kz": 1640, "folded_complex": 3435,
    "cylindrical_complex": 130, "complex_ade": 22, "complex_no_pml_stored_e": 35,
    "no_pml_conductive": 25, "no_pml_stored_e": 25, "folded_offdiag": 252,
    "cylindrical_fused_electric_pair": 341, "cylindrical_fused_magnetic_pair": 511,
    "complex_conductive_fused_pair": 112, "folded_dispersive_fused_pair": 422,
    "cylindrical_real_fused_electric_pair": 106, "beta_fused_electric_pair": 205,
    "folded_beta_fused_electric_pair": 289, "folded_beta_fused_magnetic_pair": 208,
    "no_pml_fused_electric_pair": 49, "bfast_fused_electric_pair": 141,
    "conductive_fused_electric_pair": 122,
    "probe_triton_cylindrical_real_fused_magnetic_pair": 36,
    "probe_triton_dispersive_fused_pair": 32,
    "probe_triton_folded_fused_magnetic_pair": 154, "probe_triton_folded_fused_pair": 117,
    "probe_triton_fused_ade_state": 35, "beta_fused_magnetic_pair": 78,
    "bfast_fused_magnetic_pair": 119, "complex_beta_fused_electric_pair": 203,
    "complex_beta_fused_magnetic_pair": 148, "complex_fused_electric_pair": 146,
    "folded_beta_complex_fused_magnetic_pair": 338, "folded_beta_complex_fused_pair": 399,
    "folded_complex_fused_pair": 487, "nonlinear_fused_magnetic_pair": 167,
    "probe_triton_complex_fused_magnetic_pair": 194, "complex_no_pml_conductive": 109,
    "complex_no_pml_curl": 47, "probe_triton_folded_complex_fused_magnetic_pair": 533,
    "complex_offdiag": 59, "cylindrical_composition": 73, "symmetry_composition": 47,
    "conductivity_composition": 77, "engine_route": 156, "no_pml": 206,
    "fused_ade_chain": 172, "probe_triton_folded_deposit_closure": 66,
    "complex_fused_ade_chain": 124, "conductivity": 3580, "symmetry": 4337,
    "cylindrical": 114, "folded_offdiag_dispersive": 54,
    "folded_offdiag_fused_ade_chain": 398, "cylindrical_complex_composition": 51,
    "fused_offdiag_electric": 288, "probe_triton_fused_dispersive_chain": 119,
    "source_seams": 56,
}
#: The hand legs (stencil welds, bit identity), priced from CERT-3 the same way. One is
#: RAISED, as the CUDA legs are: ``stencil_offdiag_fused_electric_pair`` 529 -> 597 s, so
#: its step watchdog, max(1,800 s, 4 x 597 s), is at least 3x its 794.7 s wall in the
#: pre-certification pass.
TRITON_HAND_EXPECT = {"bit_identity": 924, "stencil_offdiag_fused_electric_pair": 597,
                      "stencil_folded_offdiag_fused_electric_pair": 731}

#: THE TEN TRITON H->D GATES (``gate_triton_<gate>.py``), the update_H -> step_D fused
#: pairs. ``INSTALLABLE`` is False on all ten, so no route leg dispatches them and no
#: ledger welds them; the Triton board credits their seam-instances only through
#: ``build_triton_fusion_matrix.GATE_BOUND``, which re-hashes each gate's artifact against
#: the tree and refuses one that drifts, so a board cut on a frozen tree needs these gates
#: run on it. They run here as hand legs (``run_lane.py`` kind ``hd``): every leg, lifts on
#: (``--lift-timeout`` :data:`TRITON_HD_LIFT_TIMEOUT`), each into its own root
#: ``triton_<gate>_<stamp>/<policy>/gate.json``, the layout GATE_BOUND names. They are
#: NOT fleet gates: ``drive_triton_weld_gates.GATES`` is the weld fleet the Triton rebind
#: and the composition policy read, and these gates weld nothing.
#:
#: UNDER KEEP ONLY (:data:`TRITON_HD_POLICIES`). The board binds the keep run (one policy
#: per process), and nothing dispatches these products, so no flush evidence is owed; a
#: flush run is a second policy of the same measurement, added here when it is wanted.
TRITON_HD = ("fused_hd_pair", "folded_fused_hd_pair", "complex_fused_hd_pair",
             "cylindrical_fused_hd_pair", "folded_complex_fused_hd_pair",
             "beta_complex_fused_hd_pair", "conductive_bfast_fused_hd_pair",
             "cylindrical_real_fused_hd_pair", "beta_real_fused_hd_pair",
             "nonlinear_fused_hd_pair")
TRITON_HD_POLICIES: Tuple[str, ...] = ("keep",)
TRITON_HD_LIFT_TIMEOUT = 3600
#: Expected seconds per H->D gate (keep): MEASURED, each gate's wall in the 1.0
#: pre-certification pass (2026-10-10, series entries 01 to 105; the ten gates in two
#: streams of five on ONE RTX A6000, beside that pass's seven other streams on the same
#: card, every gate released), rounded up. Seven were measured again on the frozen tree of the 1.0 round
#: (2026-10-12, tree 729b161c, two streams on one card, every one released): fused 1,315 s,
#: folded 1,363, complex 482, folded_complex 206, beta_complex 173, conductive_bfast 127,
#: beta_real 100 -- 0.21 to 0.43 of the walls here. The price is the larger of the two, so
#: the step watchdog, max(1,800 s, 4 x expect_s), is at least 4x every wall measured; the
#: planner over-packs these units by the difference until the round measures them one gate
#: at a time on a card.
TRITON_HD_EXPECT: Dict[str, int] = {
    "fused_hd_pair": 3052, "folded_fused_hd_pair": 3523, "complex_fused_hd_pair": 1893,
    "cylindrical_fused_hd_pair": 1111, "folded_complex_fused_hd_pair": 929,
    "beta_complex_fused_hd_pair": 639, "conductive_bfast_fused_hd_pair": 602,
    "cylindrical_real_fused_hd_pair": 388, "beta_real_fused_hd_pair": 352,
    "nonlinear_fused_hd_pair": 257,
}

# --------------------------------------------------------------------------- route ----
TRITON_PRE = "pml_2d,folded_2d,folded_dispersive_2d,pml_3d,offdiag_2d,folded_complex_offdiag_2d"
CUDA_PRE = "pml_2d,complex_no_pml_3d,dispersive_2d,absorber_1d,folded_2d,pml_3d"
CUDA_PRE_ALONE = "pml_2d,folded_2d,pml_3d,offdiag_2d,folded_offdiag_dispersive2_2d"
#: (leg, cache kind, expected s, args...). "keep" caches are minted by
#: the package; "plain" is the flush leg's.
#:
#: EXPECTED SECONDS ARE MEASURED: each leg's wall in the 1.0 final pre-certification pass
#: (2026-10-10, series entries 01 to 111, tree b676c5f5; every route and licence leg on
#: ONE RTX A6000 shared by three CUDA route lanes, two Triton route lanes and the Triton
#: hand lane, under the round's own run_lane.py step code), rounded up. The two flush
#: legs ran before the bind (every case refused by policy, so neither released); the round
#: runs them after it, where they dispatch, and the dispatching keep legs' walls are of
#: the same size (Triton shipped 3,181 s against flush 3,066 s). Shared cards make these
#: over-estimates of a leg run alone. They are the base of the route stage's watchdogs,
#: so a stale price is a watchdog below the leg it guards: the 0.9.2 prices these replace
#: had shipped, harness_keep and flush at 450 s against measured 3,181, 1,573 and
#: 3,066 s, and a leg its watchdog kills has used its one-shot route name.
ROUTE_CUDA = (
    ("c_preflight", "keep", 470, "--backend", "cuda", "--cases", CUDA_PRE, "--out",
     "@PF@/cuda_shipped"),
    ("c_preflight_alone", "keep", 345, "--backend", "cuda", "--cases", CUDA_PRE_ALONE,
     "--table-preference", "unset", "--without-triton", "--out", "@PF@/cuda_alone"),
    ("cuda_shipped", "keep", 2527, "--backend", "cuda", "--out", "@OUT@/cuda_shipped"),
    ("cuda_harness_keep", "keep", 1959, "--backend", "cuda", "--install-policy", "keep",
     "--out", "@OUT@/cuda_harness_keep"),
    ("cuda_flush", "plain", 2792, "--backend", "cuda", "--install-policy", "flush", "--out",
     "@OUT@/cuda_flush"),
    ("cuda_default_precedence", "keep", 2659, "--backend", "cuda", "--table-preference",
     "unset", "--out", "@OUT@/cuda_default_precedence"),
    ("cuda_alone", "keep", 1981, "--backend", "cuda", "--table-preference", "unset",
     "--without-triton", "--out", "@OUT@/cuda_alone"),
)
ROUTE_TRITON = (
    ("t_preflight", "keep", 672, "--cases", TRITON_PRE, "--out", "@PF@/shipped"),
    ("shipped", "keep", 3181, "--out", "@OUT@/shipped"),
    ("harness_keep", "keep", 1573, "--install-policy", "keep", "--out", "@OUT@/harness_keep"),
    ("flush", "plain", 3066, "--install-policy", "flush", "--out", "@OUT@/flush"),
)
#: The blocking preflights: every case PASS-FUSED and the gate released, or the
#: campaign stops before its long legs. The cuda_alone preflight is recorded only.
ROUTE_BLOCKING = {"c_preflight", "t_preflight"}
#: The legs recut_driver_dispatch_record.py requires released, per table.
ROUTE_REQUIRED = {
    "cuda": ("cuda_shipped", "cuda_harness_keep", "cuda_flush", "cuda_default_precedence"),
    "triton": ("shipped", "harness_keep", "flush"),
}
#: The end-to-end licence's three legs, MEASURED in the same pass the same way.
E2E_EXPECT = {"preflight": 61, "ship": 377, "null": 305}

#: The argument every CUDA H->D leg passes that LIFTS MEEP corpus rows (its lift leg).
LIFT_FLAG = "--lift-steps"
#: The variable the lifting gates read the corpus from (``measure_predicate_coverage``).
CORPUS_VARIABLE = "MEEP_GPU_CORPUS_ROOT"
#: The corpus directories a lift reads under the root.
CORPUS_DIRECTORIES = ("examples", "tests")
#: Where the reference environment keeps its build records, under its prefix.
BUILD_RECORD_DIR = ("share", "meep-gpu-build")
#: The reference build record there. Its ``corpus`` entry names the kept MEEP ``python/``
#: tree (``root``, relative to the prefix) and the list of its files' sha256 (``list``,
#: relative to the record's directory, one ``<sha256>  <path>`` line per file), pinned by
#: that list's own sha256 (``list_sha256``) and line count (``files``), and the corpus
#: siblings (``siblings``: ``[{"path", "sha256"}]``, each a file outside ``python/`` that a
#: corpus row reads, its path relative to the root's parent, where the build keeps it); the
#: build writes it from the sha256-pinned release tarball. A record without ``siblings`` was
#: written before the build kept them, and is refused (:data:`SIBLINGS_MISSING`).
BUILD_RECORD = "reference_build.json"
#: Why a record without a ``siblings`` entry is refused, and what brings it up to date.
SIBLINGS_MISSING = ("names no corpus siblings (\"siblings\": the files outside python/ that "
                    "corpus rows read, which the build keeps beside it): a record written "
                    "before the build kept them, whose corpus measures two test_simulation.py "
                    "rows wrongly; refresh it with the build script's --refresh-corpus")
#: An unlisted file under the corpus root with one of these suffixes is refused, since it
#: could shadow an import of a corpus row: every suffix Python imports
#: (``importlib.machinery.all_suffixes()``: source, byte code, extension modules). Byte
#: code inside ``__pycache__`` is a cache of a listed source and is not read here.
CORPUS_REFUSED_EXTRAS = tuple(sorted(set(importlib.machinery.all_suffixes())))
#: THE TRITON LAUNCHER'S TOOLCHAIN. Triton compiles a C launcher for each kernel signature
#: it first runs (``triton/runtime/build.py``, ``_build``) with ``$CC``, else the first
#: ``gcc`` on ``PATH``, else ``clang``, and links it ``-lcuda`` from ``$TRITON_LIBCUDA_PATH``,
#: else from the directory of the dynamic linker's ``libcuda.so.1``, where a driver installs
#: no ``libcuda.so``. MEASURED on the GPU host in the linux-64 reference environment with
#: ``CC`` unset: the host's ``/usr/bin/gcc`` compiled the launcher, a compiler outside the
#: environment, and with ``PATH`` holding the environment alone Triton refused ("Failed to
#: find C compiler"). The environment carries its own compiler under this name (no plain
#: ``gcc``: ``environments/locks/README.md``), and its build links the driver's
#: ``libcuda.so.1`` as ``libcuda.so`` into a directory it records in :data:`LIBCUDA_RECORD`.
#: Every round child on these platforms gets both (:func:`launcher_toolchain`).
LAUNCHER_PLATFORMS = ("linux",)
LAUNCHER_COMPILER = ("bin", "x86_64-conda-linux-gnu-cc")
#: The build's record of that directory, beside :data:`BUILD_RECORD`: ``directory``,
#: relative to the prefix, or ``null`` when the build host had no NVIDIA driver.
LIBCUDA_RECORD = "libcuda.json"


# ------------------------------------------------------------------------ helpers -----
def cuda_legs(repo: Path, results: Path, stamp: str) -> List[Dict[str, Any]]:
    """Every CUDA gate leg (cited, residue, ADE), unassigned to a lane."""
    g = repo / "parity" / "meep_gpu"
    legs: List[Dict[str, Any]] = []
    for sub, tmpl, kind, ek, ef, script, *args in CUDA_CITED:
        for policy, expect in (("keep", ek), ("flush", ef)):
            leg = tmpl.replace("@P@", policy)
            argv = [a.replace("@REPO@", str(repo)) for a in args]
            argv += ["--subnormal-policy", policy]
            if policy == "flush":
                argv.append("--import-meep-for-host-policy")
            legs.append({"id": f"cited/{sub}/{leg}", "group": "cited", "subdir": sub,
                         "leg": leg, "policy": policy, "script": str(g / script),
                         "args": argv, "out_kind": kind, "expect_s": expect, "peak": 1,
                         "wrap": 0, "cache": "keep" if policy == "keep" else "plain"})
    for sub, wrap, peak, ek, ef, script, *args in CUDA_RESIDUE:
        for policy, expect in (("keep", ek), ("flush", ef)):
            argv: List[str] = []
            for item in args:
                item = item.replace("@P@", policy)
                if item == "@HOST@":
                    if policy == "flush" and not wrap:
                        argv.append("--import-meep-for-host-policy")
                    continue
                argv.append(item)
            legs.append({"id": f"residue/{sub}/{policy}", "group": "residue", "subdir": sub,
                         "leg": f"{policy}/gate.json", "policy": policy,
                         "script": str(g / script), "args": argv, "out_kind": "file",
                         "expect_s": expect, "peak": peak,
                         "wrap": wrap if policy == "flush" else 0,
                         "cache": "keep" if policy == "keep" else "plain"})
    for policy in ("keep", "flush"):
        argv = ["--backend", "cupy", "--product", "full", "--subnormal-policy", policy]
        if policy == "flush":
            argv.append("--import-meep-for-host-policy")
        legs.append({"id": f"ade_recut/{policy}", "group": "ade", "subdir": None,
                     "leg": f"{policy}/gate.json", "policy": policy,
                     "script": str(g / "gate_cuda_ade.py"), "args": argv, "out_kind": "file",
                     "expect_s": CUDA_ADE_EXPECT[policy], "peak": 1, "wrap": 0,
                     "cache": "keep" if policy == "keep" else "plain"})
    return legs


def corpus_root(explicit: str = "", environ: Optional[Dict[str, str]] = None) -> str:
    """The MEEP corpus root the lift legs walk, resolved as the lifting gates resolve it.

    ``explicit`` (``--corpus-root``), else ``$MEEP_GPU_CORPUS_ROOT``, else
    ``$MGPU_SITE_MEEP_SOURCE/python``, else ``~/meep/python`` -- the chain
    ``measure_predicate_coverage.EXAMPLES_DIR``/``TESTS_DIR`` follow. The round exports the
    answer as ``MEEP_GPU_CORPUS_ROOT`` to every child, so the gates and this check read one
    root.
    """
    environ = os.environ if environ is None else environ
    if explicit:
        return str(Path(explicit).expanduser())
    if environ.get(CORPUS_VARIABLE):
        return str(Path(environ[CORPUS_VARIABLE]).expanduser())
    site = environ.get("MGPU_SITE_MEEP_SOURCE")
    if site:
        return str(Path(site).expanduser() / "python")
    return str(Path("~/meep/python").expanduser())


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_corpus_record(record_dir: Path) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """``(corpus, None)`` from the reference build record in ``record_dir``, else
    ``(None, why)``. ``corpus``: ``root`` (relative to the prefix), ``files`` (path under
    the root -> sha256, read from the record's list, whose own sha256 and line count the
    record pins) and ``source`` (the tarball's sha256)."""
    path = Path(record_dir) / BUILD_RECORD
    if not path.is_file():
        return None, (f"no reference build record at {path}: the reference environment's "
                      f"build writes it (environments/locks/README.md)")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return None, f"the reference build record {path} cannot be read: {exc}"
    corpus = document.get("corpus") if isinstance(document, dict) else None
    if (not isinstance(corpus, dict) or not isinstance(corpus.get("root"), str)
            or not corpus["root"] or not isinstance(corpus.get("list"), str)
            or not isinstance(corpus.get("list_sha256"), str)
            or not isinstance(corpus.get("files"), int)):
        return None, (f"{path} names no corpus (\"corpus\": {{\"root\", \"list\", "
                      f"\"list_sha256\", \"files\"}})")
    listing = Path(record_dir) / corpus["list"]
    if not listing.is_file():
        return None, f"the corpus list {listing} that {path} names is absent"
    if sha256_of(listing) != corpus["list_sha256"].lower():
        return None, (f"the corpus list {listing} is not the one {path} recorded "
                      f"(sha256 differs)")
    files: Dict[str, str] = {}
    for number, line in enumerate(listing.read_text(encoding="utf-8").splitlines(), 1):
        digest, sep, rel = line.partition("  ")
        if not sep or len(digest) != 64 or not rel:
            return None, f"{listing}:{number} is not \"<sha256>  <path>\""
        files[rel] = digest
    if len(files) != corpus["files"]:
        return None, (f"{listing} lists {len(files)} files, {path} recorded "
                      f"{corpus['files']}")
    siblings, why = corpus_siblings(corpus, path)
    if why:
        return None, why
    return {"root": corpus["root"], "files": files, "siblings": siblings,
            "source": (document.get("meep") or {}).get("tarball_sha256")}, None


def corpus_siblings(corpus: Dict[str, Any], record: Path) -> Tuple[Dict[str, str], Optional[str]]:
    """``({path: sha256}, None)`` from a record's ``corpus`` entry ``siblings``, else
    ``({}, why)``: the entry absent (:data:`SIBLINGS_MISSING`), not a list of
    ``{"path", "sha256"}``, or a path that is absolute, climbs (``..``), lies under
    ``python/`` (the list holds those) or is named twice. The one reading both
    :func:`read_corpus_record` and ``l3_corpus.recorded_corpus`` use."""
    entries = corpus.get("siblings")
    if entries is None:
        return {}, f"{record} {SIBLINGS_MISSING}"
    if not isinstance(entries, list):
        return {}, f"{record}: the corpus siblings are not a list of {{\"path\", \"sha256\"}}"
    siblings: Dict[str, str] = {}
    for entry in entries:
        rel = entry.get("path") if isinstance(entry, dict) else None
        digest = entry.get("sha256") if isinstance(entry, dict) else None
        parts = rel.split("/") if isinstance(rel, str) else []
        if (not isinstance(digest, str) or len(digest) != 64 or not parts
                or rel.startswith("/") or any(part in ("", ".", "..") for part in parts)
                or parts[0] == "python" or rel in siblings):
            return {}, (f"{record}: the corpus sibling {entry!r} is not a file path relative "
                        f"to the corpus root's parent, outside python/, named once, with its "
                        f"sha256")
        siblings[rel] = digest.lower()
    return siblings, None


def verify_corpus(root: Path, files: Dict[str, str],
                  siblings: Dict[str, str]) -> Tuple[List[str], List[str]]:
    """``(problems, notes)``: every listed file present with its sha256; every sibling
    (``{path: sha256}``, the path relative to the root's parent) present there with its
    sha256, a problem naming it otherwise; an unlisted file with a suffix in
    :data:`CORPUS_REFUSED_EXTRAS` refused (byte code outside ``__pycache__`` included:
    Python imports it without a source); other unlisted files noted. Byte-code caches
    inside ``__pycache__`` are ignored."""
    root = Path(root)
    if not root.is_dir():
        return [f"the corpus root {root} is not a directory"], []
    missing, changed = [], []
    for rel, digest in sorted(files.items()):
        path = root / rel
        if not path.is_file():
            missing.append(rel)
        elif sha256_of(path) != digest.lower():
            changed.append(rel)
    extra = sorted(path.relative_to(root).as_posix() for path in root.rglob("*")
                   if path.is_file() and "__pycache__" not in path.parts
                   and path.relative_to(root).as_posix() not in files)
    refused = [rel for rel in extra if rel.endswith(CORPUS_REFUSED_EXTRAS)]
    problems = []
    for what, names in (("absent", missing), ("changed (sha256)", changed),
                        ("unlisted and importable", refused)):
        if names:
            problems.append(f"{len(names)} file(s) under {root} {what}: "
                            f"{', '.join(names[:5])}{', ...' if len(names) > 5 else ''}")
    beside, bad = root.parent, 0
    for rel, digest in sorted(siblings.items()):
        path = beside / rel
        if not path.is_file():
            problems.append(f"the corpus sibling {rel} is absent from {beside}")
        elif sha256_of(path) != digest.lower():
            problems.append(f"the corpus sibling {rel} under {beside} is changed (sha256)")
        else:
            continue
        bad += 1
    notes = [f"{len(files) - len(missing) - len(changed)} of {len(files)} recorded files "
             f"match under {root}"
             + (f"; {len(siblings) - bad} of {len(siblings)} recorded sibling(s) match beside "
                f"it ({', '.join(sorted(siblings))})" if siblings else "")]
    others = [rel for rel in extra if rel not in refused]
    if others:
        notes.append(f"{len(others)} unlisted non-code file(s) under {root}, not refused: "
                     f"{', '.join(others[:5])}{', ...' if len(others) > 5 else ''}")
    return problems, notes


def resolve_corpus(python: str, explicit: str, env: Dict[str, str]) -> Dict[str, Any]:
    """The corpus the lift legs walk: ``{"root", "record", "problems", "notes"}``.

    The record is ``<sys.prefix of python>/share/meep-gpu-build/reference_build.json``;
    the root is ``explicit`` (``--corpus-root``) or the record's, and is verified against
    the record's list and siblings (:func:`verify_corpus`) either way. The calling shell's variables
    name no corpus here: the round exports its answer as ``MEEP_GPU_CORPUS_ROOT`` to every
    child itself.
    """
    out: Dict[str, Any] = {"root": None, "record": None, "problems": [], "notes": []}
    import subprocess  # noqa: PLC0415
    try:
        done = subprocess.run([python, "-c", "import sys; print(sys.prefix)"],
                              env=dict(env), capture_output=True, text=True, timeout=300)
        lines = done.stdout.strip().splitlines() if done.returncode == 0 else []
    except (OSError, subprocess.TimeoutExpired):
        lines = []
    if explicit:
        out["root"] = str(Path(explicit).expanduser().resolve())
    if not lines:
        out["problems"].append(f"{python} reported no sys.prefix, so the reference build "
                               f"record cannot be found")
        return out
    prefix = Path(lines[-1])
    record_dir = prefix.joinpath(*BUILD_RECORD_DIR)
    out["record"] = str(record_dir / BUILD_RECORD)
    record, why = read_corpus_record(record_dir)
    if record is None:
        out["problems"].append(why)
        return out
    if not explicit:
        out["root"] = str((prefix / record["root"]).resolve())
    if record.get("source"):
        out["notes"].append(f"recorded source tarball sha256: {record['source']}")
    problems, notes = verify_corpus(Path(out["root"]), record["files"], record["siblings"])
    out["problems"] += problems
    out["notes"] += notes
    return out


def interpreter_prefix(python: str, env: Optional[Dict[str, str]] = None) -> Optional[str]:
    """``sys.prefix`` as ``python`` reports it, or None when it does not answer."""
    import subprocess  # noqa: PLC0415
    try:
        done = subprocess.run([python, "-c", "import sys; print(sys.prefix)"],
                              env=dict(os.environ if env is None else env),
                              capture_output=True, text=True, timeout=300)
    except (OSError, subprocess.TimeoutExpired):
        return None
    lines = done.stdout.strip().splitlines() if done.returncode == 0 else []
    return lines[-1] if lines else None


def launcher_toolchain(prefix: Optional[str], environ: Dict[str, str],
                       libcuda_stub: str = "") -> Dict[str, Any]:
    """The compiler and the libcuda directory a round child builds Triton's launchers with:
    ``{"cc", "cc_source", "libcuda", "libcuda_source", "problems", "notes"}``.

    What the caller set wins: ``CC`` from ``environ`` and ``TRITON_LIBCUDA_PATH`` from
    ``libcuda_stub`` (``--libcuda-stub``), else from ``environ``; an empty value is unset.
    Else, on :data:`LAUNCHER_PLATFORMS`, when ``prefix`` is a reference environment (it
    holds :data:`BUILD_RECORD`): ``CC`` is its own compiler (:data:`LAUNCHER_COMPILER`),
    and a reference environment without it is a problem, named; ``TRITON_LIBCUDA_PATH`` is
    the directory its :data:`LIBCUDA_RECORD` names, and a recorded directory without
    ``libcuda.so`` is a problem, named, while no recorded directory (a build host without
    a driver) leaves it unset, noted. Any other interpreter gets only what the caller set.
    """
    out: Dict[str, Any] = {"cc": None, "cc_source": None, "libcuda": None,
                           "libcuda_source": None, "problems": [], "notes": []}
    if environ.get("CC"):
        out["cc"], out["cc_source"] = environ["CC"], "caller (CC)"
    if libcuda_stub:
        out["libcuda"], out["libcuda_source"] = libcuda_stub, "caller (--libcuda-stub)"
    elif environ.get("TRITON_LIBCUDA_PATH"):
        out["libcuda"] = environ["TRITON_LIBCUDA_PATH"]
        out["libcuda_source"] = "caller (TRITON_LIBCUDA_PATH)"
    if not sys.platform.startswith(LAUNCHER_PLATFORMS) or (out["cc"] and out["libcuda"]):
        return out
    record_dir = Path(prefix).joinpath(*BUILD_RECORD_DIR) if prefix else None
    if record_dir is None or not (record_dir / BUILD_RECORD).is_file():
        out["notes"].append(f"{prefix or 'the interpreter'} is not a reference environment "
                            f"(no {'/'.join(BUILD_RECORD_DIR)}/{BUILD_RECORD}): CC and "
                            f"TRITON_LIBCUDA_PATH only as the caller sets them")
        return out
    if not out["cc"]:
        compiler = Path(prefix).joinpath(*LAUNCHER_COMPILER)
        if compiler.is_file() and os.access(compiler, os.X_OK):
            out["cc"], out["cc_source"] = str(compiler), "reference environment"
        else:
            out["problems"].append(
                f"the reference environment {prefix} has no C compiler at {compiler}: "
                f"Triton compiles its kernel launchers with $CC, else the host's gcc, which "
                f"is outside the environment; rebuild the environment "
                f"(parity/meep_gpu/build_meep_133_linux.sh) or set CC")
    if not out["libcuda"]:
        path = record_dir / LIBCUDA_RECORD
        try:
            directory = json.loads(path.read_text(encoding="utf-8")).get("directory")
        except (OSError, ValueError, AttributeError):
            directory = None
        if not directory:
            out["notes"].append(f"{path} names no libcuda directory (the build host had no "
                                f"NVIDIA driver): TRITON_LIBCUDA_PATH unset, so Triton links "
                                f"against the directory of the driver's libcuda.so.1")
        elif (Path(prefix) / directory / "libcuda.so").exists():
            out["libcuda"] = str(Path(prefix) / directory)
            out["libcuda_source"] = "reference environment"
        else:
            out["problems"].append(
                f"{path} names {Path(prefix) / directory}, which holds no libcuda.so: "
                f"Triton's launchers would not link; rebuild the environment on the GPU host "
                f"or pass --libcuda-stub")
    return out


def launcher_environment(env: Dict[str, str], toolchain: Dict[str, Any]) -> Dict[str, str]:
    """``env`` with the toolchain set (:func:`launcher_toolchain`): ``CC``, and the libcuda
    directory on ``TRITON_LIBCUDA_PATH`` and first on ``LD_LIBRARY_PATH``."""
    if toolchain.get("cc"):
        env["CC"] = toolchain["cc"]
    directory = toolchain.get("libcuda")
    if directory:
        env["TRITON_LIBCUDA_PATH"] = directory
        rest = [p for p in env.get("LD_LIBRARY_PATH", "").split(":") if p and p != directory]
        env["LD_LIBRARY_PATH"] = ":".join([directory] + rest)
    return env


def launcher_line(toolchain: Dict[str, Any]) -> str:
    """One line naming the toolchain and where each part came from."""
    return (f"Triton launcher toolchain: CC={toolchain.get('cc') or '(unset)'} "
            f"[{toolchain.get('cc_source') or 'none'}], TRITON_LIBCUDA_PATH="
            f"{toolchain.get('libcuda') or '(unset)'} "
            f"[{toolchain.get('libcuda_source') or 'none'}]"
            + "".join(f"; {note}" for note in toolchain.get("notes") or ()))


def lifting_legs(steps: Sequence[Dict[str, Any]]) -> List[str]:
    """The ids of the steps that lift MEEP corpus rows: the CUDA H->D gate legs
    (``--lift-steps``) and the Triton H->D hand legs (kind ``hd``, lifts always on)."""
    return [str(step.get("id")) for step in steps
            if (step.get("kind", "gate") == "gate" and LIFT_FLAG in (step.get("args") or ()))
            or step.get("kind") == "hd"]


def triton_hd_steps(policies: Sequence[str] = TRITON_HD_POLICIES) -> List[Dict[str, Any]]:
    """One hand leg per Triton H->D gate and policy, unassigned to a lane.

    ``@RESULTS@`` and ``@STAMP@`` are the round's placeholders (``round.py subs_for``):
    the root is ``results/triton_<gate>_<stamp>`` and the artifact ``<root>/<policy>/
    gate.json``, the layout ``build_triton_fusion_matrix.GATE_BOUND`` names.
    """
    return [{"id": f"triton/{policy}/hd/{gate}", "kind": "hd", "gate": gate,
             "script": f"gate_triton_{gate}.py", "root": f"@RESULTS@/triton_{gate}_@STAMP@",
             "policy": policy, "lift_timeout": TRITON_HD_LIFT_TIMEOUT,
             "expect_s": TRITON_HD_EXPECT[gate]}
            for policy in policies for gate in TRITON_HD]


def corpus_problems(steps: Sequence[Dict[str, Any]], root: Optional[str],
                    record_problems: Sequence[str] = ()) -> List[str]:
    """Why these steps' lifts would find no verified corpus at ``root``; empty when they
    would. ``record_problems`` are :func:`resolve_corpus`'s (no record, a tree that does not
    match it)."""
    lifting = lifting_legs(steps)
    if not lifting:
        return []
    legs_named = (f"{len(lifting)} H->D leg(s) lift MEEP corpus rows ({', '.join(lifting[:4])}"
                  f"{', ...' if len(lifting) > 4 else ''})")
    out = [f"{legs_named} and the MEEP corpus does not verify: {problem}"
           for problem in record_problems]
    if not root:
        return out or [f"{legs_named} and no corpus is named: pass --corpus-root <MEEP 1.33.0 "
                       f"source>/python (the root the round exports as {CORPUS_VARIABLE}, "
                       f"holding examples/tests); without it every lift refuses"]
    missing = [name for name in CORPUS_DIRECTORIES if not (Path(root) / name).is_dir()]
    if not missing:
        return out
    return out + [f"{legs_named} and the corpus root {root} has no {'/'.join(missing)} "
                  f"directory: pass --corpus-root <MEEP 1.33.0 source>/python (the root the "
                  f"round exports as {CORPUS_VARIABLE}, holding examples/tests); without it "
                  f"every lift refuses"]


#: The two module constants a Triton H->D gate's lift reads its inputs by: the census the
#: rows are joined to and the seam record that puts them in the product's cell, each a
#: directory under ``parity/meep_gpu/results`` (``lift_basis`` in
#: ``gate_triton_fused_hd_pair.py``, ``gate_triton_folded_fused_hd_pair.py``,
#: ``gate_triton_complex_fused_hd_pair.py`` and ``triton_cylindrical_hd_gate_kit.py``;
#: the tail kit reuses the first).
TRITON_HD_INPUTS = ("CENSUS", "SEAM_RECORD")
#: The files ``lift_basis`` reads under each input. A census leg file that is absent is
#: skipped there without a word, and every seam row it would have named a module for then
#: lifts nothing; the seam record's rows are read unconditionally.
TRITON_HD_INPUT_FILES = {
    "CENSUS": ("examples.jsonl", "tests.jsonl", "tests_param_matched.jsonl"),
    "SEAM_RECORD": ("h_to_d_seam.jsonl",),
}


def _harness_constant(harness: Path, module: str, name: str,
                      seen: Tuple[str, ...] = ()) -> Optional[str]:
    """The string a harness module binds ``name`` to at top level, read from its source.

    Read, never imported: the H->D gates import the Triton kernel modules, and the check
    runs before a GPU is taken. Follows ``name = alias.OTHER`` into the harness module
    ``alias`` names, and, where the module does not bind ``name`` at all, the first
    harness module it imports that does (the tail and cylindrical gates take theirs from
    their kit). ``None`` when no string is reached.
    """
    import ast  # noqa: PLC0415

    path = harness / f"{module}.py"
    if module in seen or len(seen) > 4 or not path.is_file():
        return None
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    aliases: Dict[str, str] = {}
    local: List[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            for item in node.names:
                if (harness / f"{item.name}.py").is_file():
                    aliases[item.asname or item.name] = item.name
                    local.append(item.name)
    bound = None
    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id == name):
            bound = node.value
    chain = seen + (module,)
    if isinstance(bound, ast.Constant) and isinstance(bound.value, str):
        return bound.value
    if (isinstance(bound, ast.Attribute) and isinstance(bound.value, ast.Name)
            and bound.value.id in aliases):
        return _harness_constant(harness, aliases[bound.value.id], bound.attr, chain)
    if bound is not None:
        return None
    for other in local:
        found = _harness_constant(harness, other, name, chain)
        if found is not None:
            return found
    return None


def triton_hd_inputs(harness: Path, steps: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Per Triton H->D gate among ``steps``, the census and seam record its lift reads."""
    gates = sorted({str(step["gate"]) for step in steps if step.get("kind") == "hd"})
    return {gate: {name: _harness_constant(harness, f"gate_triton_{gate}", name)
                   for name in TRITON_HD_INPUTS} for gate in gates}


def triton_hd_input_problems(harness: Path, steps: Sequence[Dict[str, Any]],
                             results: Optional[Path] = None
                             ) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
    """Why these steps' Triton H->D lifts would find their inputs missing; empty when not.

    The inputs live under ``results`` (default ``<harness>/results``, the directory the
    gates read as ``HERE / "results"``), which a checkout does not carry: the round stages
    them. A gate stopped by ``the lift leg needs ...`` has run every device leg first, so
    a missing input is refused here, by name, before a GPU is taken.
    """
    results = harness / "results" if results is None else results
    inputs = triton_hd_inputs(harness, steps)
    problems: List[str] = []
    unread = sorted(gate for gate, names in inputs.items()
                    if any(value is None for value in names.values()))
    if unread:
        problems.append(f"the Triton H->D gate(s) {unread} name no census or seam record "
                        f"this check can read ({'/'.join(TRITON_HD_INPUTS)} in their "
                        f"source or their kit's), so their lift inputs cannot be checked")
    needed: Dict[Tuple[str, str], List[str]] = {}
    for gate, names in inputs.items():
        for kind, value in names.items():
            if value is not None:
                needed.setdefault((kind, value), []).append(gate)
    for (kind, value), gates in sorted(needed.items()):
        root = results / value
        if not root.is_dir():
            missing = "does not exist"
        else:
            empty = [name for name in TRITON_HD_INPUT_FILES[kind]
                     if not (root / name).is_file() or (root / name).stat().st_size == 0]
            if not empty:
                continue
            missing = f"lacks {', '.join(empty)}"
        problems.append(
            f"{len(gates)} Triton H->D leg(s) ({', '.join(gates[:4])}"
            f"{', ...' if len(gates) > 4 else ''}) lift from {root} ({kind}), which "
            f"{missing}: stage the round's H->D inputs under {results} before the gate "
            f"stage; without them each such leg runs its device legs and then stops at "
            f"'the lift leg needs ...'")
    return problems, inputs


def split_cuda(legs: Sequence[Dict[str, Any]], lanes: int) -> List[List[Dict[str, Any]]]:
    """Assign whole subdirectories to ``lanes`` lanes, longest first (LPT).

    A subdirectory is the unit a certification block claims, so it never straddles two
    campaigns; the ADE re-cut (its own sibling root) joins the lightest lane last.
    Within a lane, legs run longest subdirectory first, keep before flush, so the long
    pole starts first and a wide (lifting) leg never waits for the end.
    """
    if lanes < 1:
        raise ValueError("lanes must be >= 1")
    by_sub: Dict[str, List[Dict[str, Any]]] = {}
    ade: List[Dict[str, Any]] = []
    for leg in legs:
        (ade if leg["subdir"] is None else by_sub.setdefault(leg["subdir"], [])).append(leg)
    weight = {sub: sum(l["expect_s"] for l in group) for sub, group in by_sub.items()}
    out: List[List[Dict[str, Any]]] = [[] for _ in range(lanes)]
    load = [0] * lanes
    for sub in sorted(by_sub, key=lambda s: (-weight[s], s)):
        k = min(range(lanes), key=lambda i: (load[i], i))
        out[k].extend(sorted(by_sub[sub], key=lambda l: (l["policy"] != "keep", l["id"])))
        load[k] += weight[sub]
    if ade:
        k = min(range(lanes), key=lambda i: (load[i], i))
        out[k].extend(ade)
    return out


def materialise_cuda(lane: List[Dict[str, Any]], campaign: Path, ade_root: Path) -> List[Dict[str, Any]]:
    """Fill each leg's --out and artifact for the lane's campaign root."""
    done = []
    for leg in lane:
        leg = dict(leg)
        if leg["group"] == "ade":
            artifact = ade_root / leg["leg"]
            leg["subjects"] = list(ADE_SUBJECTS)
            leg["subjects_file"] = str(ade_root / "SUBJECT_DIGESTS.txt")
        else:
            artifact = campaign / leg["subdir"] / leg["leg"]
        out = artifact.parent if leg["out_kind"] == "dir" else artifact
        leg["args"] = list(leg["args"]) + ["--out", str(out)]
        leg["artifact"] = str(artifact)
        leg["leg_dir"] = str(artifact.parent)
        leg["kind"] = "gate"
        done.append(leg)
    return done


#: The tier-2 gates split off the core fleet so they can be placed on any lane.
TRITON_MOVABLE = (("t2_conductivity", ["conductivity"]), ("t2_symmetry", ["symmetry"]),
                  ("t2_rest", ["cylindrical", "folded_offdiag_dispersive",
                               "folded_offdiag_fused_ade_chain", "fused_offdiag_electric",
                               "probe_triton_fused_dispersive_chain", "source_seams"]))


def triton_units(policies: Sequence[str] = ("keep", "flush")) -> List[Dict[str, Any]]:
    """The Triton work as placeable units: per policy one ordered CORE unit and three
    MOVABLE tier-2 units, and under each policy of :data:`TRITON_HD_POLICIES` one
    movable unit per H->D gate.

    CORE (one lane, in order): the fleet (A + the 0.9.2 three +
    ``cylindrical_complex_composition``), shard B (its ``_env_probe`` root, the 0.9.2
    shard name), shard C (``_offdiag_probe``), the composition shard D (after its staged
    manifest), the two stencil welds and the bit identity. Every step writes a fresh
    root; ``_flush`` roots are R13's second policy (``rebind_triton_welds.py
    --flush-campaign``). The H->D units are hand legs into ``triton_<gate>_<stamp>``
    (:func:`triton_hd_steps`).
    """
    movable_gates = {g for _name, gates in TRITON_MOVABLE for g in gates}
    core_fleet = [g for g in TRITON_A + TRITON_R092 + TRITON_TIER2 if g not in movable_gates]
    b = TRITON_B + TRITON_R092_ENV
    units: List[Dict[str, Any]] = []
    for policy in policies:
        suffix = "" if policy == "keep" else "_flush"
        steps = [
            {"id": f"triton/{policy}/fleet", "kind": "fleet", "root": f"@FLEET@{suffix}",
             "gates": list(core_fleet), "policy": policy, "env": {},
             "expect_s": sum(TRITON_EXPECT[g] for g in core_fleet)},
            {"id": f"triton/{policy}/env_probe", "kind": "fleet",
             "root": f"@FLEET@{suffix}_env_probe", "gates": list(b), "policy": policy,
             "env": {}, "expect_s": sum(TRITON_EXPECT[g] for g in b)},
            {"id": f"triton/{policy}/offdiag_probe", "kind": "fleet",
             "root": f"@FLEET@{suffix}_offdiag_probe", "gates": list(TRITON_C),
             "policy": policy, "env": {}, "expect_s": TRITON_EXPECT["complex_offdiag"]},
            {"id": f"triton/{policy}/composition", "kind": "fleet",
             "root": f"@COMPOSITION@{suffix}", "gates": list(TRITON_D), "policy": policy,
             "env": {}, "manifest": True,
             "expect_s": sum(TRITON_EXPECT[g] for g in TRITON_D)},
        ]
        steps += [{"id": f"triton/{policy}/stencil_{fam}", "kind": "stencil",
                   "root": f"@STENCIL@{suffix}", "family": fam, "policy": policy,
                   "expect_s": TRITON_HAND_EXPECT[f"stencil_{fam}"]} for fam in TRITON_STENCIL]
        steps.append({"id": f"triton/{policy}/bit_identity", "kind": "bitid",
                      "root": f"@BITID@{suffix}", "policy": policy,
                      "experiments": TRITON_BITID_EXPERIMENTS,
                      "expect_s": TRITON_HAND_EXPECT["bit_identity"]})
        units.append({"name": f"triton_{policy}_core", "steps": steps})
        for name, gates in TRITON_MOVABLE:
            units.append({"name": f"triton_{policy}_{name}", "steps": [
                {"id": f"triton/{policy}/{name}", "kind": "fleet",
                 "root": f"@FLEET@{suffix}_{name}", "gates": list(gates), "policy": policy,
                 "env": {}, "expect_s": sum(TRITON_EXPECT[g] for g in gates)}]})
        # The H->D gates: one movable unit per gate, so the planner can place each.
        if policy in TRITON_HD_POLICIES:
            units += [{"name": f"triton_{policy}_hd_{step['gate']}", "steps": [step]}
                      for step in triton_hd_steps((policy,))]
    return units


def _unit_seconds(unit: Dict[str, Any]) -> int:
    return sum(int(s["expect_s"]) for s in unit["steps"])


def pack_lanes(units: List[Dict[str, Any]], gpus: int) -> List[List[Dict[str, Any]]]:
    """Longest-processing-time packing of whole units onto ``gpus`` lanes."""
    lanes: List[List[Dict[str, Any]]] = [[] for _ in range(gpus)]
    load = [0] * gpus
    for unit in sorted(units, key=lambda u: (-_unit_seconds(u), u["name"])):
        k = min(range(gpus), key=lambda i: (load[i], i))
        lanes[k].append(unit)
        load[k] += _unit_seconds(unit)
    return lanes


def plan_lanes(repo: Path, results: Path, stamp: str, *, gpus: int, tables: Sequence[str],
               cuda_split: Optional[int] = None,
               only: Optional[set] = None) -> Tuple[List[List[Dict[str, Any]]], Dict[str, Any]]:
    """The round's gate lanes for ``tables`` on at most ``gpus`` GPUs (the account cap).

    CUDA splits into ``cuda_split`` campaigns (default: 2 when both tables share four
    GPUs, else as many as GPUs), each one unit pinned to one lane, because one campaign
    is one environment and one recorder block. The Triton units pack around them.
    """
    units: List[Dict[str, Any]] = []
    if "cuda" in tables:
        legs = cuda_legs(repo, results, stamp)
        if only:
            legs = [l for l in legs if l["id"] in only]
        if cuda_split is None:
            cuda_split = (2 if gpus >= 4 else 1) if "triton" in tables else gpus
        for k, lane in enumerate(split_cuda(legs, max(1, cuda_split))):
            if lane:
                units.append({"name": f"cuda_g{k}", "steps": materialise_cuda(
                    lane, results / f"cuda_regate_{stamp}_g{k}",
                    results / f"cuda_ade_recut_{stamp}")})
    if "triton" in tables:
        for unit in triton_units():
            if only:
                unit = dict(unit, steps=[s for s in unit["steps"] if s["id"] in only])
            if unit["steps"]:
                units.append(unit)
    lanes = pack_lanes(units, gpus)
    plan = [[step for unit in lane for step in unit["steps"]] for lane in lanes]
    summary = {"gpus": gpus, "tables": list(tables),
               "cuda_campaigns": cuda_split if "cuda" in tables else 0,
               "lanes": [{"units": [u["name"] for u in lane], "steps": len(p),
                          "expect_s": sum(int(s["expect_s"]) for s in p)}
                         for lane, p in zip(lanes, plan)]}
    summary["wall_s"] = max((l["expect_s"] for l in summary["lanes"]), default=0)
    return plan, summary


def route_legs(table: str) -> List[Dict[str, Any]]:
    rows = ROUTE_CUDA if table == "cuda" else ROUTE_TRITON
    out = []
    for name, cache, expect, *args in rows:
        cases = args[args.index("--cases") + 1] if "--cases" in args else "all"
        if cases != "all" and cases.split(",")[0] != LEAD_CASE:
            raise AssertionError(f"{name}: an explicit case list must lead with {LEAD_CASE}")
        out.append({"id": name, "kind": "route", "cache": cache, "env": {},
                    "args": ["--gpu-id", "0", *args], "expect_s": expect,
                    "blocking": name in ROUTE_BLOCKING,
                    "cases_required": (len(cases.split(",")) if name in ROUTE_BLOCKING else None)})
    return out


def required_route_legs(table: str) -> Tuple[str, ...]:
    return tuple(ROUTE_REQUIRED[table])


# -------------------------------------------------------------------------- check -----
def capability_device(capability: str) -> str:
    """``9.0`` -> ``cc90``: the device a compute capability names in a route campaign
    (``fastpath.route_campaign``), a licence directory and a chunk stamp. Refused (ValueError)
    when it is not spelled ``major.minor``."""
    text = str(capability or "")
    major, dot, minor = text.partition(".")
    if not (dot and major.isdigit() and not major.startswith("0") and len(minor) == 1
            and minor.isdigit()):
        raise ValueError(f"{capability!r} is not a compute capability spelled major.minor "
                         f"(8.6, 9.0)")
    return f"cc{major}{minor}"


def licence_directory(licence_campaign: str, capability: str) -> str:
    """The licence's results directory on one capability: ``<licence campaign>_cc<NN>``.

    One per device, so the licences of two capabilities of one round never share a
    directory, nor does either extend the other by ``_``; ``route_stamps.licence_refusal``
    accepts the name for that device.
    """
    return f"{licence_campaign}_{capability_device(capability)}"


def route_campaign_names(capability: str) -> Tuple[Dict[str, Any], List[str]]:
    """``(facts, problems)``: the round's NVIDIA route campaigns and licence, by name.

    THE NAMES ARE THE REGISTRY'S (``meep_gpu/route_stamps.py``), read from the checkout
    :func:`check` put first on the path, and a name it refuses is a problem: the stamp
    still the placeholder the freeze replaces (``route_stamp_is_unset``), or a campaign
    that has already run (``route_campaign_is_spent``). The licence directory is the
    primary table's (``route_stamps.licence_campaign``), the only one the re-cut accepts
    a licence from.
    """
    from meep_gpu import fastpath, route_stamps  # noqa: PLC0415
    facts: Dict[str, Any] = {}
    problems: List[str] = []
    for table in ("triton", "cuda"):
        key = f"{table}_route_campaign"
        facts[key] = fastpath.route_campaign(route_stamps.route_gate(table), capability)
        refused = route_stamps.campaign_refusal(table, facts[key])
        if refused:
            problems.append(f"{key} is {facts[key]}: {refused} (RUNBOOK.md, CERT-1)")
    facts["licence_campaign"] = route_stamps.licence_campaign(
        fastpath.primary_table(capability) or "triton")
    return facts, problems


def check(repo: Path, capability: Optional[str], route: bool, *,
          git_expect: Optional[str] = None, fresh: Sequence[Path] = (),
          corpus: str = "", python: Optional[str] = None,
          new_capability: bool = False) -> Tuple[List[str], Dict[str, Any]]:
    """Every name this file spells, checked against the checkout. Returns (problems, facts).

    No GPU: it imports the package and the gates and reads the ledgers. What it refuses:
    a leg script or tool the checkout lacks (the Triton H->D hand legs' gates included); a
    fleet gate the round does not run, or one it runs that the fleet driver does not know,
    or an H->D gate the fleet driver also runs; composition re-cut keys
    (:data:`TRITON_COMPOSITION_KEYS`) other than the ones ``recut_composition_records.LEGS``
    covers the composition step's legs with; the flush licence absent (R13);
    for the route stage, a route campaign name ``meep_gpu/route_stamps.py`` refuses (the
    stamp unset, or a campaign that has already run; a gate stage records it in the facts
    and goes on); a route leg list that differs from
    the legs ``recut_driver_dispatch_record.BACKENDS`` requires; a preflight case the
    route gate's drive table lacks; a capability either NVIDIA table does not admit, unless
    ``new_capability`` says it is new (on the route stage always); ``new_capability`` on a
    capability both tables admit, without a capability, or on the route stage;
    ``MEEP_GPU_HARDWARE`` or the retired ``MEEP_GPU_ALLOW_UNCERTIFIED`` exported for a
    route stage; a hit of the hygiene guard (``tools/ci/check_hygiene.py``: a host name,
    account name, local path or personal e-mail address) in the checkout's tracked files
    or the coverage board's subject, and, for the route stage, in the four ledgers the
    bind wrote (:func:`hygiene_problems`); with ``git_expect``, a
    HEAD other than the frozen commit or an unclean checkout; any results root in
    ``fresh`` that already exists (every name the round writes is one-shot); and, for the
    gate stage, a MEEP corpus (``corpus``, else the one the reference build record of
    ``python``'s environment names, verified against its list: :func:`resolve_corpus`) the
    H->D lift legs
    would find unverified, empty or not named at all, and a census or seam record the
    Triton H->D lift legs read
    (:func:`triton_hd_input_problems`) absent from ``parity/meep_gpu/results``.
    """
    problems: List[str] = []
    g = repo / "parity" / "meep_gpu"
    facts: Dict[str, Any] = {}
    every_cuda_leg = cuda_legs(repo, g / "results", "check")
    for leg in every_cuda_leg:
        if not Path(leg["script"]).is_file():
            problems.append(f"CUDA leg {leg['id']}: {leg['script']} is not in the checkout")
    every_hd_leg = triton_hd_steps()
    for leg in every_hd_leg:
        if not (g / leg["script"]).is_file():
            problems.append(f"Triton H->D leg {leg['id']}: {leg['script']} is not in the "
                            f"checkout")
    facts["triton_hd_legs"] = len(every_hd_leg)
    if not route:
        found = resolve_corpus(python or sys.executable, corpus, dict(os.environ))
        facts.update(corpus_root=found["root"], corpus_record=found["record"],
                     corpus_notes=found["notes"])
        facts["corpus_lifting_legs"] = len(lifting_legs(every_cuda_leg + every_hd_leg))
        problems.extend(corpus_problems(every_cuda_leg + every_hd_leg, found["root"],
                                        found["problems"]))
        hd_problems, facts["triton_hd_inputs"] = triton_hd_input_problems(g, every_hd_leg)
        problems.extend(hd_problems)
    for name in ("drive_triton_weld_gates.py", "gate_triton_offdiag_stencil_welds.py",
                 "probe_fused_kernel_bit_identity.py", "track_triton_pml.py",
                 "triton_device_identity.py", "gate_dispatch_fused_route.py",
                 "gate_dispatch_end_to_end.py", "record_cuda_regate.py", "rebind_cuda_welds.py",
                 "rebind_triton_welds.py", "recut_driver_dispatch_record.py",
                 "record_cuda_ade_recut.py", "recut_composition_records.py",
                 "neutral_labels.py", "neutralize_ledger_labels.py"):
        if not (g / name).is_file():
            problems.append(f"{name} is not in the checkout")
    if git_expect:
        import subprocess  # noqa: PLC0415
        head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                              capture_output=True, text=True)
        if head.returncode != 0 or head.stdout.strip() != git_expect:
            problems.append(f"HEAD is {(head.stdout or head.stderr).strip()[:64]}, not "
                            f"the frozen commit {git_expect}")
        dirty = subprocess.run(["git", "-C", str(repo), "status", "--porcelain"],
                               capture_output=True, text=True)
        lines = [l for l in dirty.stdout.splitlines() if l.strip()]
        allowed = set(ROUTE_STAGE_LEDGERS) if route else set()
        stray = [l for l in lines if l[3:] not in allowed]
        if dirty.returncode != 0 or stray:
            problems.append(f"the checkout is not clean: {stray[:5]}")
        facts["rebound_ledgers_present"] = sorted(l[3:] for l in lines if l[3:] in allowed)
    hygiene = hygiene_problems(repo, ("tracked", "subject", "ledgers") if route
                               else ("tracked", "subject"))
    facts["hygiene"] = "clean" if not hygiene else f"{len(hygiene)} problem(s)"
    problems.extend(hygiene)
    for target in fresh:
        if target.exists() or target.is_symlink():
            problems.append(f"{target} exists; every results root the round writes is "
                            f"one-shot (a gate appends rows to a directory it finds)")
    sys.path[:0] = [str(repo), str(g)]
    try:
        import drive_triton_weld_gates as fleet  # noqa: PLC0415
        wanted = set(TRITON_A + TRITON_R092 + TRITON_TIER2 + TRITON_B + TRITON_R092_ENV
                     + TRITON_C + TRITON_D)
        missing = sorted(wanted - set(fleet.GATES))
        if missing:
            problems.append(f"the fleet driver has no gate named {missing}")
        facts["fleet_gates"] = len(fleet.GATES)
        facts["round_gates"] = len(wanted)
        unrun = sorted(set(fleet.GATES) - wanted)
        if unrun:
            problems.append(f"the fleet driver carries gates this round does not run: {unrun}")
        in_fleet = sorted(set(TRITON_HD) & set(fleet.GATES))
        if in_fleet:
            problems.append(f"the H->D gates {in_fleet} are fleet gates in this checkout; the "
                            f"round runs them as hand legs and would run them twice")
        import recut_composition_records as recut_composition  # noqa: PLC0415
        derived = tuple(key for key, (leg_names, _probe) in recut_composition.LEGS.items()
                        if set(leg_names) & set(TRITON_D))
        uncovered = sorted(set(TRITON_D) - {name for key in derived
                                            for name in recut_composition.LEGS[key][0]})
        facts["triton_composition_keys"] = list(derived)
        if derived != TRITON_COMPOSITION_KEYS or uncovered:
            problems.append(f"the composition re-cut keys are {list(TRITON_COMPOSITION_KEYS)}; "
                            f"recut_composition_records.LEGS covers the composition step's "
                            f"legs with {list(derived)}"
                            + (f", and none of its keys covers {uncovered}" if uncovered else ""))
        from meep_gpu import fastpath  # noqa: PLC0415
        facts["dispatch_by_default"] = bool(getattr(fastpath, "DISPATCH_BY_DEFAULT", None))
        admitted = getattr(fastpath, "TABLE_DISPATCH_POLICIES", None)
        facts["flush_licence"] = bool(admitted and "flush" in admitted.get("triton", ())
                                      and "flush" in admitted.get(fastpath.CUDA_TABLE, ()))
        if not facts["flush_licence"]:
            problems.append("fastpath.TABLE_DISPATCH_POLICIES does not admit flush on both "
                            "NVIDIA tables: CP-licence.patch is not in this checkout, so the "
                            "flush route legs would refuse at rung 8b (R13)")
        import recut_driver_dispatch_record as recut  # noqa: PLC0415
        for table in ("triton", "cuda"):
            required = tuple(sorted(recut.BACKENDS[table]["required_legs"]))
            ours = tuple(sorted(ROUTE_REQUIRED[table]))
            if required != ours:
                problems.append(f"the {table} route legs the recut requires are {required}; "
                                f"this round runs {ours}")
        if tuple(recut.BACKENDS["cuda"].get("optional_legs", ())) != ("cuda_alone",):
            problems.append(f"the CUDA optional legs are "
                            f"{recut.BACKENDS['cuda'].get('optional_legs')}, not cuda_alone")
        if route:
            import gate_dispatch_fused_route as gate  # noqa: PLC0415
            for backend, cases in (("triton", TRITON_PRE), ("cuda", CUDA_PRE),
                                   ("cuda", CUDA_PRE_ALONE)):
                drive = gate.drive_table(backend)
                absent = [c for c in cases.split(",") if c not in drive]
                if absent:
                    problems.append(f"preflight cases {absent} are not in the {backend} "
                                    f"route gate's drive table")
        facts["capability_mode"] = "new" if new_capability else "certified"
        if new_capability and not capability:
            problems.append("--new-capability names no capability: pass --capability")
        # A CAPABILITY IS SPELLED major.minor OR REFUSED BY NAME, before any table or
        # registry reads it: a typo is never a new capability.
        try:
            spelled = bool(capability) and bool(capability_device(capability))
        except ValueError as refused:
            spelled = False
            problems.append(f"--capability is refused: {refused}")
        if spelled:
            admitting = []
            for table in ("triton", "cuda"):
                admission = fastpath.capability_admission(table)
                listed = admission.get("admitted") or ()
                facts[f"{table}_admits"] = list(listed)
                if capability in listed:
                    admitting.append(table)
                elif route or not new_capability:
                    problems.append(
                        f"capability_admission({table!r}) does not admit {capability}: "
                        f"{list(listed)}" + ("" if route else
                                             "; a capability the ledgers hold no live record "
                                             "for is gated with --new-capability"))
            # A NEW CAPABILITY IS SAID, NEVER INFERRED. The gate stage runs on any card,
            # because no family gate goes through admission; the flag is what lets it run on
            # one the ledgers do not admit, and it is refused where it would be untrue.
            if new_capability and route:
                problems.append("--new-capability is a gate-stage mode: the route stage "
                                "runs once the bind has added the capability's records "
                                "and both tables admit it")
            elif new_capability and len(admitting) == 2:
                problems.append(f"--new-capability: both NVIDIA tables already admit "
                                f"{capability}; it is a certified capability, so run "
                                f"without the flag")
            # THE ROUTE STAGE STOPS ON A NAME THE REGISTRY REFUSES; a gate stage only
            # records it. The stamp edit moves a file no gate holds, so gates may run
            # before the freeze sets it, and their roots carry the round's own stamp.
            named, refusals = route_campaign_names(capability)
            facts.update(named)
            facts["licence_directory"] = licence_directory(named["licence_campaign"],
                                                           capability)
            facts["route_campaign_refusals"] = list(refusals)
            if route:
                problems.extend(refusals)
    except Exception as exc:  # noqa: BLE001 - a check that cannot import is a problem
        problems.append(f"could not import the checkout: {exc!r}")
    for name in ("MEEP_GPU_HARDWARE", "MEEP_GPU_ALLOW_UNCERTIFIED"):
        if route and os.environ.get(name) is not None:
            problems.append(f"{name} is set; the route gate refuses a leg when it is "
                            "exported at all")
    return problems, facts


#: Why a new capability's gate stage runs chunked: an unchunked stamp carries no device, so
#: its roots could equal, or be extended by, another device's roots of the same round.
NEW_CAPABILITY_UNCHUNKED = (
    "--new-capability runs chunked: partition an index with chunks.py partition --capability "
    "<cc> --new-device and launch its chunks (--chunk-index, --chunk), whose stamps carry the "
    "device")


def index_device_problems(index: Dict[str, Any], capability: Optional[str],
                          new_capability: bool) -> List[str]:
    """Why a chunk index does not certify this capability in this mode: its device is
    another card's, or its ``device_mode`` disagrees with ``--new-capability``."""
    problems: List[str] = []
    if index.get("machine") != "nvidia":
        return [f"the chunk index is a {index.get('machine')} index"]
    try:
        device = capability_device(capability) if capability else None
    except ValueError as refused:
        return [str(refused)]
    if device is None:
        problems.append(f"the chunk index certifies {index.get('device')}: pass --capability")
    elif index.get("device") != device:
        problems.append(f"the chunk index certifies {index.get('device')}, and this card is "
                        f"{device} (--capability {capability})")
    mode = "new" if new_capability else "certified"
    if index.get("device_mode") != mode:
        problems.append(f"the chunk index is of a {index.get('device_mode')} device and this "
                        f"launch says {mode} ("
                        + ("drop --new-capability" if new_capability else
                           "pass --new-capability") + ")")
    return problems


def hygiene_problems(repo: Path, legs: Sequence[str]) -> List[str]:
    """The hygiene guard's hits on ``repo``, one problem per leg that has any.

    Owner ruling R22: a host name, account name, local path or personal e-mail address
    in a certified byte cannot be scrubbed at export, so the round refuses to start on a
    checkout whose tracked files or board subject hold one (CERT-1), and refuses the
    route stage while the ledgers the bind wrote hold one: the bind tools write neutral
    labels, and ``neutralize_ledger_labels.py`` rewrites what an earlier bind left.
    """
    import importlib.util  # noqa: PLC0415
    path = repo / "tools" / "ci" / "check_hygiene.py"
    if not path.is_file():
        return ["tools/ci/check_hygiene.py is not in the checkout"]
    spec = importlib.util.spec_from_file_location("check_hygiene", path)
    guard = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = guard
    spec.loader.exec_module(guard)
    hits, unreadable = guard.run_legs(guard.load_rules(path.parent / "hygiene_rules.json"),
                                      repo, list(legs))
    out = [f"the hygiene guard cannot read {problem}" for problem in unreadable]
    for leg in legs:
        mine = [hit for hit in hits if hit.leg == leg]
        if mine:
            out.append(f"the hygiene guard finds {len(mine)} hit(s) in its {leg} leg "
                       f"(python tools/ci/check_hygiene.py --legs {leg}): "
                       + "; ".join(f"{hit.path}:{hit.where} {hit.klass}" for hit in mine[:5]))
    return out


#: The files a route stage may find changed in the frozen checkout: the ledgers the
#: bind wrote after the gates (copied in uncommitted, as the 0.9.2 per-architecture
#: Metal round did), because the flush route legs dispatch only on the
#: ``certified_policies`` evidence the bind records. Nothing else may differ.
ROUTE_STAGE_LEDGERS = ("meep_gpu/triton_kernels/fingerprints.json",
                       "meep_gpu/cuda_kernels/fingerprints.json",
                       "meep_gpu/cuda_kernels/certification.json")


def flush_coverage(repo: Path, capability: str) -> Tuple[Dict[str, Any], List[str]]:
    """How many dispatchable arms the flush licence covers on ``capability``: N of D.

    For every arm of the Triton and CUDA certification maps, asks
    ``fastpath.policy_evidence_reasons`` (the rule dispatch itself applies) whether a
    caller-installed ``flush`` would be admitted. Before the bind writes the evidence
    every arm answers no; the route stage's go/no-go is every arm answering yes. Returns
    the per-table counts and the distinct gates that lack the evidence.
    """
    g = repo / "parity" / "meep_gpu"
    sys.path[:0] = [str(repo), str(g)]
    from meep_gpu import fastpath, fastpath_cuda  # noqa: PLC0415
    out: Dict[str, Any] = {"capability": capability}
    lacking: List[str] = []
    for table, arms in (("triton", fastpath.ARM_CERTIFICATION),
                        (fastpath.CUDA_TABLE, fastpath_cuda.CUDA_ARM_CERTIFICATION)):
        covered, gates_lacking = 0, set()
        for arm, (_family, gate) in sorted(arms.items()):
            reasons = fastpath.policy_evidence_reasons([arm], table, capability, "flush")
            if reasons:
                gates_lacking.add(gate)
            else:
                covered += 1
        out[table] = {"arms_covered": covered, "arms": len(arms),
                      "gates_lacking": sorted(gates_lacking)}
        lacking += [f"{table}:{gate}" for gate in sorted(gates_lacking)]
    return out, lacking


def _roots_newest_first(results: Path, name: str) -> List[str]:
    """``name`` and its resume continuations (``name_r2``, ``_r3`` ...), newest first.

    A lane that resumes a fleet step runs the gates left into a fresh continuation
    root (``run_lane.py``), and both re-cut tools take the FIRST root holding a
    released artifact for a gate, so the re-run must be searched before the root it
    repairs.
    """
    found = sorted((p.name for p in results.glob(f"{name}_r[0-9]*") if p.is_dir()),
                   key=lambda n: int(n.rsplit("_r", 1)[1]), reverse=True)
    return [f"results/{n}" for n in found] + [f"results/{name}"]


def bind_plan(results: Path, stamp: str, cuda_campaigns: int, *, write: bool) -> List[str]:
    """The ordered commands that bind the round's evidence, for the bind host.

    Report mode by default: none of these writes unless ``write`` adds ``--write``
    (the integrator's call, on the owner's tree). Run from ``parity/meep_gpu`` with
    the round's result roots under ``results/``. The CUDA campaigns are recorded and
    rebound one at a time (``record_cuda_regate.py`` and ``rebind_cuda_welds.py`` each
    take one ``--campaign``; the 0.9.2 ledger already binds two campaigns on one
    capability); the Triton keep roots are searched in order with their flush twins
    beside them; the composition and family records take the flush composition run
    and the flush fleet roots; the bit identity takes its flush twin. The composition
    re-cut names its keys (``--only``, :data:`TRITON_COMPOSITION_KEYS`): the composition
    root holds the legs of five of the tool's nine keys, and the tool refuses a run
    that lacks a selected key's leg, all or nothing. The H->D gates' roots bind nothing
    here: no ledger welds them, and the Triton board reads them at the records step
    (``repoint_board_bindings.py``).
    """
    w = " --write" if write else ""
    out = []
    for k in range(cuda_campaigns):
        out.append(f"python record_cuda_regate.py --campaign results/cuda_regate_{stamp}_g{k}{w}")
    for k in range(cuda_campaigns):
        out.append(f"python rebind_cuda_welds.py --campaign results/cuda_regate_{stamp}_g{k}{w}")
    out.append(f"python record_cuda_ade_recut.py --campaign results/cuda_ade_recut_{stamp}{w}")
    fleet = f"triton_fleet_{stamp}"
    keep_names = ([fleet, f"{fleet}_env_probe", f"{fleet}_offdiag_probe"]
                  + [f"{fleet}_{name}" for name, _gates in TRITON_MOVABLE]
                  + [f"triton_offdiag_stencil_welds_{stamp}"])

    def flush_name(name: str) -> str:
        return (name.replace(fleet, f"{fleet}_flush", 1) if name.startswith(fleet)
                else f"{name}_flush")

    keep = [r for name in keep_names for r in _roots_newest_first(results, name)]
    flush = [r for name in keep_names for r in _roots_newest_first(results, flush_name(name))]
    out.append("python rebind_triton_welds.py " + " ".join(f"--campaign {r}" for r in keep)
               + " " + " ".join(f"--flush-campaign {r}" for r in flush) + w)
    out.append(f"python rebind_triton_welds.py --bind-bit-identity "
               f"results/triton_bit_identity_{stamp}/gate.json --bit-identity-flush "
               f"results/triton_bit_identity_{stamp}_flush/gate.json{w}")
    out.append(f"python recut_composition_records.py --run results/triton_composition_{stamp} "
               f"--flush-run results/triton_composition_{stamp}_flush "
               f"--only {','.join(TRITON_COMPOSITION_KEYS)}{w}")
    family_keep = _roots_newest_first(results, fleet)
    family_flush = _roots_newest_first(results, f"{fleet}_flush")
    out.append(f"python recut_composition_records.py --family-recert {' '.join(family_keep)} "
               f"--flush-family-recert {' '.join(family_flush)}{w}")
    # LAST: the label leaves no re-bind above replaces (the dated certification blocks
    # accumulate) get their neutral labels, so the guard's ledger leg reads clean.
    out.append(f"python neutralize_ledger_labels.py{w}")
    return out


#: The 13 files the ADE update_P re-cut manifests BEFORE its legs compile
#: (``record_cuda_ade_recut.py`` re-hashes them file by file and refuses undeclared
#: drift); the 0.9.2 residue lane carried the same list.
ADE_SUBJECTS = (
    "meep_gpu/cuda_kernels/ade_kernels.py", "meep_gpu/cuda_kernels/coverage.py",
    "meep_gpu/cuda_kernels/compile_cache.py", "meep_gpu/cuda_kernels/test_ade_update_p.py",
    "meep_gpu/dispersion.py", "meep_gpu/stepping.py", "meep_gpu/fields.py", "meep_gpu/pml.py",
    "meep_gpu/grid.py", "meep_gpu/subnormal_policy.py", "parity/meep_gpu/gate_cuda_ade.py",
    "parity/meep_gpu/probe_fused_kernel_bit_identity.py", "parity/meep_gpu/gate_provenance.py")


# ----------------------------------------------------------------------- estimate -----
def estimate(gpus: int, tables: Sequence[str] = ("cuda", "triton")) -> Dict[str, Any]:
    """Serial seconds per lane from the expected walls (one A6000 per lane, the tables above)."""
    repo = Path("/repo")
    _plan, summary = plan_lanes(repo, repo / "results", "x", gpus=gpus, tables=tables)
    route = {t: sum(l["expect_s"] for l in route_legs(t)) for t in ("cuda", "triton")}
    route["triton"] += sum(E2E_EXPECT.values())
    return {"cert3": summary, "route_s": route,
            "cert3_wall_s": summary["wall_s"], "cert4_wall_s": max(route.values())}


# ---------------------------------------------------------------------------- main ----
def round_roots(stamp: str, cuda_campaigns: int, capability: Optional[str] = None,
                facts: Optional[Dict[str, Any]] = None) -> List[str]:
    """Every results root name the round creates for ``stamp`` (one-shot names)."""
    names = [f"cuda_regate_{stamp}_g{k}" for k in range(cuda_campaigns)]
    names.append(f"cuda_ade_recut_{stamp}")
    for suffix in ("", "_flush"):
        fleet = f"triton_fleet_{stamp}{suffix}"
        names += [fleet, f"{fleet}_env_probe", f"{fleet}_offdiag_probe"]
        names += [f"{fleet}_{name}" for name, _gates in TRITON_MOVABLE]
        names += [f"triton_composition_{stamp}{suffix}",
                  f"triton_offdiag_stencil_welds_{stamp}{suffix}",
                  f"triton_bit_identity_{stamp}{suffix}"]
    names += [f"triton_{gate}_{stamp}" for gate in TRITON_HD]
    licence = (facts or {}).get("licence_campaign") or f"dispatch_end_to_end_{stamp}"
    names.append(licence)
    if capability:
        names.append((facts or {}).get("licence_directory")
                     or licence_directory(licence, capability))
    for key in ("triton_route_campaign", "cuda_route_campaign"):
        if facts and facts.get(key):
            names += [facts[key], f"{facts[key]}_preflight"]
    return names


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("--repo", required=True)
    c.add_argument("--capability", default=None)
    c.add_argument("--route", action="store_true")
    c.add_argument("--git-expect", default=None, help="the frozen commit (full hash)")
    c.add_argument("--fresh-stamp", default=None,
                   help="refuse when any results root of this stamp already exists")
    c.add_argument("--chunk-index", default=None,
                   help="with --chunk: refuse when any results root of that chunk exists, or "
                        "when the index's chunks could collide (chunks.py)")
    c.add_argument("--chunk", default=None)
    c.add_argument("--cuda-split", type=int, default=2)
    c.add_argument("--corpus-root", default="",
                   help="the MEEP corpus the H->D lifts walk (default: the root the "
                        "reference build record of --python's environment names; verified "
                        "against its list either way)")
    c.add_argument("--python", default=sys.executable,
                   help="the reference environment's interpreter (default: this one)")
    c.add_argument("--new-capability", action="store_true",
                   help="gate stage: the capability is one the ledgers hold no live record "
                        "for (an H200 at 9.0); runs chunked, from an index partitioned with "
                        "--new-device")
    c.add_argument("--json", default=None)
    p = sub.add_parser("plan")
    p.add_argument("what", choices=("lanes", "route"))
    p.add_argument("--repo", required=True)
    p.add_argument("--results", required=True)
    p.add_argument("--stamp", required=True)
    p.add_argument("--gpus", type=int, default=4)
    p.add_argument("--tables", default="cuda,triton")
    p.add_argument("--cuda-split", type=int, default=None)
    p.add_argument("--table", choices=("cuda", "triton"), default="cuda")
    p.add_argument("--only", default="")
    p.add_argument("--out", required=True)
    e = sub.add_parser("estimate")
    e.add_argument("--gpus", type=int, default=4)
    e.add_argument("--tables", default="cuda,triton")
    n = sub.add_parser("names")
    n.add_argument("--repo", required=True)
    n.add_argument("--capability", required=True)
    f = sub.add_parser("flush-coverage")
    f.add_argument("--repo", required=True)
    f.add_argument("--capability", required=True)
    f.add_argument("--json", default=None)
    b = sub.add_parser("bind")
    b.add_argument("--results", default="results")
    b.add_argument("--stamp", default=None)
    b.add_argument("--chunks", default=None,
                   help="a chunked round's index (chunks.py, nvidia): the bind of the "
                        "selection --selection names (chunks.py verify --selection)")
    b.add_argument("--selection", default=None,
                   help="with --chunks: the per-unit choice chunks.py verify wrote")
    b.add_argument("--cuda-split", type=int, default=2)
    b.add_argument("--write", action="store_true",
                   help="print the commands WITH --write (they are printed, never run)")
    args = ap.parse_args(argv)

    if args.cmd == "check":
        repo = Path(args.repo).resolve()
        _p, facts0 = check(repo, args.capability, False) if args.fresh_stamp else ([], {})
        fresh = ([repo / "parity" / "meep_gpu" / "results" / name for name in
                  round_roots(args.fresh_stamp, args.cuda_split, args.capability, facts0)]
                 if args.fresh_stamp else [])
        chunk_problems: List[str] = []
        if args.chunk_index or args.chunk:
            import chunks as chunk_tool  # noqa: PLC0415 - beside this file
            if not (args.chunk_index and args.chunk):
                ap.error("--chunk-index and --chunk go together")
            index = chunk_tool.read_index(Path(args.chunk_index))
            chunk = chunk_tool.find_chunk(index, args.chunk)
            results = repo / "parity" / "meep_gpu" / "results"
            fresh += [results / name for name in chunk_tool.chunk_roots(chunk, index["machine"])]
            chunk_problems = [f"chunk index: {line}" for line in
                              chunk_tool.collisions(index, results=results, repo=repo)]
            chunk_problems += chunk_tool.round_problems(repo, index)
            chunk_problems += index_device_problems(index, args.capability, args.new_capability)
        elif args.new_capability and not args.route:
            chunk_problems.append(NEW_CAPABILITY_UNCHUNKED)
        problems, facts = check(repo, args.capability, args.route,
                                git_expect=args.git_expect, fresh=fresh,
                                corpus=args.corpus_root, python=args.python,
                                new_capability=args.new_capability)
        problems += chunk_problems
        for line in problems:
            print(f"FAIL {line}", flush=True)
        print("FACTS " + json.dumps(facts, sort_keys=True), flush=True)
        print(f"CHECK {'OK' if not problems else 'REFUSED'} ({len(problems)} problem(s))",
              flush=True)
        if args.json:
            Path(args.json).write_text(json.dumps({"problems": problems, "facts": facts},
                                                  indent=1, sort_keys=True) + "\n")
        return 1 if problems else 0
    if args.cmd == "estimate":
        print(json.dumps(estimate(args.gpus, tuple(t for t in args.tables.split(",") if t)),
                         indent=1))
        return 0
    if args.cmd == "names":
        _problems, facts = check(Path(args.repo).resolve(), args.capability, False)
        print(facts.get("triton_route_campaign", ""), facts.get("cuda_route_campaign", ""))
        return 0
    if args.cmd == "flush-coverage":
        out, lacking = flush_coverage(Path(args.repo).resolve(), args.capability)
        for table in ("triton", "cuda"):
            row = out[table]
            print(f"{table}: flush admitted for {row['arms_covered']} of {row['arms']} arms; "
                  f"{len(row['gates_lacking'])} gate(s) lack evidence"
                  + (f": {row['gates_lacking']}" if row["gates_lacking"] else ""), flush=True)
        print(f"FLUSH_COVERAGE {'COMPLETE' if not lacking else 'INCOMPLETE'}", flush=True)
        if args.json:
            Path(args.json).write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
        return 0 if not lacking else 1
    if args.cmd == "bind":
        if args.chunks:
            import chunks as chunk_tool  # noqa: PLC0415 - beside this file
            index = chunk_tool.read_index(Path(args.chunks))
            if index["machine"] != "nvidia":
                ap.error(f"{args.chunks} is a {index['machine']} index; its bind is "
                         f"chunks.py bind --index {args.chunks}")
            problems = chunk_tool.collisions(index)
            for line in problems:
                print(f"FAIL {line}", flush=True)
            if problems:
                return 1
            if not args.selection:
                ap.error("bind --chunks takes --selection (the file chunks.py verify "
                         "--selection wrote)")
            results = Path(args.results).resolve()
            lines = chunk_tool.bind_commands(index, results, args.chunks, args.selection,
                                             write=args.write)
            if chunk_tool.bind_withheld(lines):
                for line in lines:
                    print(line, flush=True)
                return 1
            lines += chunk_tool.records_lines(
                index, chunk_tool.read_selection(Path(args.selection), index), results)
        elif args.stamp:
            lines = bind_plan(Path(args.results).resolve(), args.stamp, args.cuda_split,
                              write=args.write)
        else:
            ap.error("bind takes --stamp (one round) or --chunks (a chunked round)")
        for line in lines:
            print(line, flush=True)
        return 0
    repo, results = Path(args.repo).resolve(), Path(args.results).resolve()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    only = {s for s in args.only.split(",") if s} or None
    if args.what == "lanes":
        tables = tuple(t for t in args.tables.split(",") if t)
        plan, summary = plan_lanes(repo, results, args.stamp, gpus=args.gpus, tables=tables,
                                   cuda_split=args.cuda_split, only=only)
        for k, steps in enumerate(plan):
            if steps:
                (out / f"lane_{k}.json").write_text(json.dumps(steps, indent=1) + "\n")
        (out / "lanes_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
        for k, lane in enumerate(summary["lanes"]):
            print(f"lane_{k}: {lane['steps']} steps, {lane['expect_s']} s expected: "
                  f"{', '.join(lane['units'])}", flush=True)
        print(f"wall {summary['wall_s']} s (the longest lane)", flush=True)
        return 0
    legs = route_legs(args.table)
    if only:
        legs = [l for l in legs if l["id"] in only]
    (out / f"route_{args.table}.json").write_text(json.dumps(legs, indent=1) + "\n")
    print(f"route {args.table}: {len(legs)} legs; the recut requires "
          f"{list(required_route_legs(args.table))}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
