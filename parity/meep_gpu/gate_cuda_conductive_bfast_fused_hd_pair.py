#!/usr/bin/env python3
"""Device byte gate for the CUDA H->D weld on the conductive and BFAST curl tails.

WHAT IS BEING CERTIFIED. ``meep_gpu/cuda_kernels/conductive_bfast_fused_hd_pair.py``
computes the certified ``update_H`` constitutive into launch-local SCRATCH, takes its
own cell's magnetic field from registers, RECOMPUTES every one of the curl half's six
foreign taps from pre-launch state, steps ``D``/``fu_D`` and the variant's third
history in place, and rotates the ``H``/``f_w_H`` bindings afterwards. ONE PRODUCT,
TWO VARIANTS, TWO BOARD CELLS -- the variant is the ``step_D`` tail and nothing else:

    conductive   H_to_D (cuda_constitutive/ordinary -> cuda_conductive/conductive)
    bfast        H_to_D (cuda_bfast/BFAST           -> cuda_bfast/BFAST)

The claim is per COMPLETE DRIVER STEP -- the driver's own pass order -- over a stated
budget of steps, as uint32 WORDS over EVERY stored volume the engine allocates
(primaries, the split-field PML auxiliaries, both ``f_w`` histories, and the variant's
own ``f_cond_D*`` / ``f_bfast_D*``); never ``allclose``, because ``-0.0 == 0.0`` lies.
The references are stepped from ONE seed beside the weld:

  1. the ARRAY PATH -- ``stepping``'s passes in the driver's order;
  2. the CERTIFIED SINGLES -- ``update_H`` alone then the variant's own certified
     ``step_D`` alone, as two dispatched launches, everything else on the array path;
  3. the COMPOSITION THE COMPOSER INSTALLS ON THESE ROWS TODAY -- the released B->H
     pair for the variant plus the released D->E pair for the variant, each ABSORBING
     the driver passes its own ``REPLACES`` names. This is the reference the whole
     arbitration finding rests on. Where a neighbour refuses a fixture, the
     arrangement falls back to certified singles for that half and RECORDS which;
  4. the same slots DISPATCHED UNFUSED -- one certified single per slot, with every
     in-seam pass on the array path.

All four must agree with the weld word for word.

LEGS
  host legs (no device launch)
  driver_order         REPLACES is exactly the driver's two adjacent consults, the only
                       statement between them is the electric withdraw loop, and the
                       sync channel's containment rule excludes step_D -- all read off
                       the tree with ``ast``, never spelled
  transcription        per VARIANT and, on the conductive one, per MASK: the certified
                       curl's parameter list is carried across CHARACTER FOR CHARACTER,
                       the certified tail differs in EXACTLY the twelve redirected
                       magnetic reads, the lifted constitutive pieces are byte-equal to
                       the released sibling's own, every declared edit's anchor
                       resolves, every mutation needle resolves exactly once, the text
                       is pure ASCII (the NVRTC locale trap)
  refusal              by name: a run that is neither conductive nor BFAST; a run that
                       is BOTH; a lossless step_D; a no-absorber conductive run; a
                       folded BFAST grid; complex storage; an undeclared source list; a
                       standing INTEGRATED electric withdraw; a non-integrated electric
                       source NOT refused; a magnetic source NOT refused
  arbitration          the shipped composer, asked. On both variants: which product
                       won each slot, that THIS product is refused BY NAME, that the
                       refusal text names INSTALLABLE False, that the selection is
                       unchanged by this product's presence, and that the released
                       neighbours keep the slots they had. POSITIONAL facts -- whether
                       a row for this product exists in the composer's tables at all --
                       are RECORDED, never asserted: this gate is written for the
                       UNWIRED tree and the wiring inverts exactly those
  ghost_observability  whether a metallic ghost's exact 0.0f is observable in step_D's
                       output at all, per variant. What it buys is an honest mutation
                       set: where every tap is masked at the plane its ghost fires on,
                       the ghost mutation is a RECORDED NULL and the periodic-wrap one
                       is the armed sibling
  device legs
  product              the five arrangements over every fixture x value class, byte
                       identity per complete driver step, with the movement floor, the
                       subnormal census, and per-variant liveness floors: a conductive
                       case must have moved f_cond and a BFAST case must have moved
                       f_bfast, or the case measured nothing about the tail that makes
                       it its own cell
  purity_and_race      per fixture, on the array path: of the curl's valid foreign
                       taps, how many land on a cell ``update_H`` MOVED -- the count of
                       taps whose recomputed value differs from what an in-place weld
                       would have raced on. Near zero would mean the identity licenses
                       nothing. Beside it the RACE LEDGER: the same identity at six
                       block sizes, because the design's whole claim is that the answer
                       does not depend on which block ran first
  seed_scale           the same verdict at three seed exponents, with the denormal
                       census at each, so the shipped scale is a measurement
  launch_structure     launches per step at the seam and over the whole step, for the
                       weld and every reference, two independent counters
  sync                 the product FORCE-installed; a flux accessor called mid-run so
                       ``synchronize_magnetic_fields`` consults ``update_H_synchronize``.
                       An arm whose plan answers that consult MUST diverge in D/fu_D --
                       that divergence IS the citation for the composer refusal -- and
                       the arm that declines by the containment rule MUST NOT
  withdraw             an integrated electric source in the seam (the configuration the
                       predicate refuses today): the hoisted arrangement identical, the
                       un-hoisted launch and the ``after_step_D`` placement both diverge
  byte_neutral         the own-cell register reads replaced by reloads of the scratch
                       just stored: the one armed edit required NOT to diverge
  mutation             the armed device and host defects, each with its expectation
                       DECLARED. Every entry is scored on a fixture where the machinery
                       it disables is LIVE, and an entry predicted null is recorded
                       WITH its reason rather than dropped
  disarm               the identical harness on the shipped bytes, required not to
                       diverge, so a mutation leg's catches are not the harness's
  compiler             read AFTER every in-process device leg: the NVRTC observer saw
                       at least one compile, the CuPy cache held ZERO entries when the
                       policy was installed, and the policy reached the compiler
  lift                 the corpus rows the standing seam record puts in the two cells,
                       re-lifted ON THE DEVICE in their own children and driven as
                       THREE arrangements from one captured state per complete step --
                       the weld, the certified singles and the array path. The weld's
                       claim is against the CERTIFIED SINGLES, because a defect in this
                       weld cannot appear in them

Rule 7: one flushed line per case, and the artifact is rewritten after every leg, so
an interrupted run keeps everything that landed.

RUNNING IT (the GPU host, ONE verified-empty GPU)::

    CUDA_VISIBLE_DEVICES=$GPU \\
    CUPY_CACHE_DIR=$OUT/cupy_cache/ftz_stripped \\
    MEEP_GPU_CORPUS_ROOT=<staging-root>/meep_133/src/meep-1.33.0/python \\
        python -u gate_cuda_conductive_bfast_fused_hd_pair.py \\
            --subnormal-policy keep --out $OUT/keep/gate.json
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import inspect
import json
import os
import platform
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
# BY NAME, never by parents[N]: a moved harness resolving a wrong root measures nothing.
_REPO_API = str(next(parent for parent in Path(_HERE).parents
                     if (parent / "meep_gpu" / "cuda_kernels").is_dir()))
for _path in (_REPO_API, _HERE):
    if _path not in sys.path:
        sys.path.insert(0, _path)

try:
    import cupy as cp
except ImportError:  # laptop: only the host legs run
    cp = None

import gate_provenance  # noqa: E402
import probe_fused_kernel_bit_identity as probe  # noqa: E402

# THE TWO SIBLING GATES, imported for the machinery all three share: the launch counter
# that wraps the shipped memo, the word comparator, the read-only snapshot, the driver
# dispatch shim and the corpus-lift capture/restore. ONE copy of each -- a second
# transcription of a launch counter is a second place for a counter to stop counting.
import gate_cuda_offdiag_stencil_welds as scratch_gate  # noqa: E402
import gate_cuda_fused_hd_pair as hd_gate  # noqa: E402

from meep_gpu import stepping, withdraw_hoist  # noqa: E402
from meep_gpu.fastpath import (SYNC_PASS_OWNERS, SYNC_PATH_SLOTS,  # noqa: E402
                               SYNC_UPDATE_H_PASS)

log = probe.log
to_host = scratch_gate.to_host
words = scratch_gate.words
differing = scratch_gate.differing
MemoLaunchCounter = scratch_gate.MemoLaunchCounter
needle = scratch_gate.needle
Shim = hd_gate.Shim
SEED_SCALE_BITS = hd_gate.SEED_SCALE_BITS

# ---------------------------------------------------------------------------
# The battery contract, so the census driver can lift a corpus row into this file
# ---------------------------------------------------------------------------

SUBJECT_PACKAGE = "cuda_kernels"

#: Per-case RNG seed base, from the case LABEL rather than from ``hash()``:
#: ``PYTHONHASHSEED`` salts the hash of a string, so a hash-seeded case could not be
#: replayed from the record that names it.
SEED = 20260907

#: Complete DRIVER STEPS a fixture runs -- the budget every hand-CUDA record on this
#: track is cut at.
STEPS = 60

#: The steps at which the sync leg calls the energy accessor. Two, so the hazard is
#: measured to persist and is not a one-step transient.
SYNC_STEPS: Tuple[int, ...] = (3, 7)

#: The block sizes the race ledger runs. 32 is one warp and 1024 the maximum; the
#: shipped launcher's own 256 sits between them, and they put wildly different numbers
#: of blocks on the grid.
BLOCK_SIZES: Tuple[int, ...] = (32, 64, 128, 256, 512, 1024)

#: Every stored volume a complete step can touch, PLUS the two third histories this
#: product's variants write. The sibling's tuple carries neither, and a comparison
#: that omitted the output that makes each cell its own cell would be the vacuous
#: pass this file exists to refuse. Volumes that are not allocated on a fixture are
#: dropped by :func:`state_of`, so one tuple serves both variants.
STATE_NAMES: Tuple[str, ...] = tuple(
    list(scratch_gate.STATE_NAMES)
    + [f"f_cond_D{axis}" for axis in "xyz"]
    + [f"f_bfast_D{axis}" for axis in "xyz"])

IMMUTABLE_PML: Tuple[str, ...] = scratch_gate.IMMUTABLE_PML
STEP_PASSES: Tuple[str, ...] = scratch_gate.STEP_PASSES
VALUE_CLASSES: Tuple[str, ...] = ("uniform", "subnormal_band")

#: The group that is EXEMPT from the movement floor under the band class, and why.
#: Under ``flush`` a state seeded entirely in the band drives ``constitutive_apply``
#: to ``f[idx] = (f[idx] + 0) - 0`` because both products flush, so the magnetic
#: PRIMARY need not move; its split-field history is written unconditionally and must.
#: The floor itself is per GROUP and is measured against the ARRAY PATH -- see
#: :func:`drive` -- rather than being a second list of volume names.
BAND_EXEMPT_GROUPS: Tuple[str, ...] = ("H",)

#: The five arrangements every product case drives in lockstep.
MODES: Tuple[str, ...] = ("array", "singles", "composition_today", "unfused", "weld")

LEG_GROUPS: Dict[str, Tuple[str, ...]] = {
    "host": ("driver_order", "transcription", "refusal", "arbitration",
             "ghost_observability"),
    # ``compiler`` is LAST among the in-process device legs on purpose: it reads the
    # NVRTC observer and the policy's strip counters AFTER every kernel this process
    # will launch has been compiled.
    "device": ("product", "purity_and_race", "seed_scale", "launch_structure", "sync",
               "withdraw", "byte_neutral", "mutation", "disarm", "compiler"),
    "corpus": ("lift",),
}
ALL_LEGS: Tuple[str, ...] = tuple(leg for group in LEG_GROUPS.values() for leg in group)

#: The census this gate lifts its corpus rows from, and the seam record whose
#: ``withdraw_in_seam`` flag names the rows the predicate must refuse.
CENSUS = "cuda_predicate_coverage_2026-09-03_extended"
SEAM_RECORD = "h_to_d_seam_2026-09-04"

#: The two board cells this product serves, in the census's own arm spelling
#: ``(update_H arm, step_D arm)``, keyed by the variant that serves each. The lift
#: leg's basis is DERIVED from this map against the seam record rather than listed.
CELL_ARMS: Dict[str, Tuple[str, str]] = {
    "conductive": ("cuda_constitutive/ordinary", "cuda_conductive/conductive"),
    "bfast": ("cuda_bfast/BFAST", "cuda_bfast/BFAST"),
}

#: HOW MANY CLEAN COMPLETE STEPS A LIFTED CORPUS ROW MUST REACH TO COUNT.
LIFT_CLEAN_STEP_FLOOR = 8

# ---------------------------------------------------------------------------
# The fixtures
# ---------------------------------------------------------------------------
#
# THE CONDUCTIVE SET SWEEPS THE MASK, because the mask is a COMPILE-TIME constant and
# a run that swept only 'CCC' would say nothing about the configuration MEEP's own
# allocation granularity makes ordinary -- the certified family's own gate swept the
# same five and this one inherits the argument. Every axis is walled somewhere, every
# axis is folded somewhere, both fold terminations appear and a fully periodic grid
# appears, which is the ONLY specialisation on which the shifted tap's PERIODIC WRAP
# is observable at all.
#
# THE BFAST SET REFUSES A FOLD BY CONSTRUCTION (``bfast_curl``'s clause 4) and carries
# the shape the corpus row has -- a reduced-dimension, fully periodic grid, where the
# HOST-SIDE invariance gates are what decide k1/k2 -- beside a 3-D one where neither
# axis is invariant, so a kernel that re-derived those gates on the device would be
# wrong on one and right on the other.
SPECS: Tuple[Dict[str, Any], ...] = (
    {"label": "cond_periodic_CCC", "variant": "conductive", "cell": (1.6, 1.7, 1.5),
     "boundaries": ("periodic", "periodic", "periodic"), "symmetry": (),
     "mask": (True, True, True)},
    {"label": "cond_walls_CCC", "variant": "conductive", "cell": (1.6, 1.7, 1.5),
     "boundaries": ("metallic", "metallic", "metallic"), "symmetry": (),
     "mask": (True, True, True)},
    {"label": "cond_walls_xy_Cmm", "variant": "conductive", "cell": (1.6, 1.6, 1.6),
     "boundaries": ("metallic", "metallic", "periodic"), "symmetry": (),
     "mask": (True, False, False)},
    {"label": "cond_wall_z_mCm", "variant": "conductive", "cell": (1.6, 1.6, 1.6),
     "boundaries": ("periodic", "periodic", "metallic"), "symmetry": (),
     "mask": (False, True, False)},
    {"label": "cond_periodic_mmC", "variant": "conductive", "cell": (1.6, 1.7, 1.5),
     "boundaries": ("periodic", "periodic", "periodic"), "symmetry": (),
     "mask": (False, False, True)},
    {"label": "cond_fold_y_even_CmC", "variant": "conductive", "cell": (1.6, 2.0, 1.6),
     "boundaries": None, "symmetry": (("Y", 1),), "mask": (True, False, True)},
    {"label": "cond_fold_y_odd_phase_CCC", "variant": "conductive",
     "cell": (1.6, 2.0, 1.6), "boundaries": None, "symmetry": (("Y", -1),),
     "mask": (True, True, True)},
    {"label": "cond_fold_y_metallic_CCC", "variant": "conductive",
     "cell": (1.6, 2.0, 1.6), "boundaries": ("periodic", "metallic", "periodic"),
     "symmetry": (("Y", 1),), "mask": (True, True, True)},
    {"label": "bfast_periodic_3d", "variant": "bfast", "cell": (1.6, 1.7, 1.5),
     "boundaries": ("periodic", "periodic", "periodic"), "symmetry": (),
     "bfast_k": (0.8169576958985646, 0.31, 0.17)},
    {"label": "bfast_walls_all", "variant": "bfast", "cell": (1.6, 1.7, 1.5),
     "boundaries": ("metallic", "metallic", "metallic"), "symmetry": (),
     "bfast_k": (0.8169576958985646, 0.31, 0.17)},
    {"label": "bfast_wall_x_only", "variant": "bfast", "cell": (1.6, 1.6, 1.6),
     "boundaries": ("metallic", "periodic", "periodic"), "symmetry": (),
     "bfast_k": (0.41, 0.0, 0.63)},
    # THE CORPUS ROW'S OWN SHAPE: reduced dimensions, fully periodic, one nonzero k.
    # Two axes are INVARIANT here, so ``have_p``/``have_m`` zero four of the six
    # coefficients on the host -- the transcription's whole point, and a shape no
    # 3-D fixture reaches.
    {"label": "bfast_reduced_1d", "variant": "bfast", "cell": (0.0, 0.0, 4.0),
     "boundaries": ("periodic", "periodic", "periodic"), "symmetry": (),
     "bfast_k": (0.8169576958985646, 0.0, 0.0), "dimensions": 1},
)

BY_LABEL: Dict[str, Dict[str, Any]] = {spec["label"]: spec for spec in SPECS}

#: The fixtures the mutations are scored on. Chosen so every armed defect has at least
#: one fixture where the machinery it disables is LIVE: a wall on every axis, a
#: periodic wrap that survives the mask, a fold, and both variants.
MUTATION_SPEC_LABELS: Tuple[str, ...] = ("cond_periodic_CCC", "cond_walls_CCC",
                                         "cond_fold_y_odd_phase_CCC",
                                         "bfast_periodic_3d", "bfast_walls_all")

#: The subset ``--product reduced`` runs. It carries at least one fixture of EACH
#: variant, so a reduced run cannot pass the product leg's per-variant floors
#: vacuously.
REDUCED_LABELS: Tuple[str, ...] = ("cond_periodic_CCC", "cond_walls_CCC",
                                   "cond_fold_y_even_CmC", "bfast_periodic_3d",
                                   "bfast_reduced_1d")

#: The conductivity profile every conductive fixture installs, as a GRADED volume
#: rather than a constant: MEEP's own ``Absorber`` is graded, ``condfac``/``condinv``
#: are full volumes, and a constant sigma would make every cell's coefficient the same
#: word -- which cannot distinguish a kernel that indexed the profile at the wrong
#: cell from one that indexed it at the right one.
CONDUCTIVITY_SCALE = 0.75


def case_rng(label: str) -> "np.random.Generator":
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    return np.random.default_rng(SEED + int.from_bytes(digest[:4], "big"))


def _family():
    from meep_gpu.cuda_kernels import (conductive_bfast_fused_hd_pair  # noqa: PLC0415
                                       as family)
    return family


# ---------------------------------------------------------------------------
# The fixture
# ---------------------------------------------------------------------------

def _graded_sigma(shape: Tuple[int, ...], rng) -> Any:
    """A graded float32 sigma volume on the device."""
    host = (CONDUCTIVITY_SCALE
            * (0.25 + rng.uniform(0.0, 1.0, size=shape))).astype(np.float32)
    return cp.asarray(np.ascontiguousarray(host))


def _install_conductivity(fields, grid, spec: Mapping[str, Any], rng) -> None:
    """A D conductivity on exactly the components the mask names.

    D-SIDE ONLY, and that is the corpus row's own shape:
    ``TestAdjointSolver.test_damping`` carries a D conductivity and no B one, so its
    ``step_B`` is the certified curl's and its ``step_D`` is the conductive family's.
    Installing a B conductivity too would move ``step_B`` to a different arm and the
    fixture would stop being this cell's.
    """
    shape = tuple(int(n) for n in grid.shape)
    mapping = {component: (_graded_sigma(shape, rng) if live else None)
               for component, live in zip(("Dx", "Dy", "Dz"), spec["mask"])}
    fields.set_d_conductivity(mapping)


def build(spec: Mapping[str, Any], value_class: str, rng):
    """A frozen ``(fields, grid, pml, dtdx)`` on the DEVICE, seeded for this class."""
    from meep_gpu.fields import Fields  # noqa: PLC0415
    from meep_gpu.grid import Grid, Mirror  # noqa: PLC0415
    from meep_gpu.pml import PML  # noqa: PLC0415

    extra: Dict[str, Any] = {}
    if spec.get("dimensions") is not None:
        extra["dimensions"] = int(spec["dimensions"])
    if spec.get("bfast_k") is not None:
        extra["bfast_scaled_k"] = tuple(spec["bfast_k"])
    grid = Grid(resolution=10.0, cell_size=spec["cell"], courant=0.35, xp=cp,
                boundaries=spec["boundaries"],
                symmetry=tuple(Mirror(axis, phase)
                               for axis, phase in spec["symmetry"]),
                **extra)
    thickness = tuple((0, 0) if grid.shape[axis] < 6
                      else (0, 2) if grid.is_mirrored(axis)
                      else (2, 2) for axis in range(3))
    pml = PML(grid=grid, thickness=thickness)
    fields = Fields(grid=grid)
    fields.enable_field_storage()
    fields.enable_pml_storage()
    if spec["variant"] == "conductive":
        _install_conductivity(fields, grid, spec, rng)
    shape = tuple(int(n) for n in grid.shape)
    epsilon = {name: cp.full(shape, value, cp.float32)
               for name, value in zip(("Ex", "Ey", "Ez"), (2.0, 2.5, 3.0))}
    inverse = {name: cp.full(shape, np.float32(1.0 / value), cp.float32)
               for name, value in zip(("Ex", "Ey", "Ez"), (2.0, 2.5, 3.0))}
    fields.set_epsilon_volumes(epsilon, inverse)
    hosts = (probe.pml_field_hosts(shape, rng, "uniform") if value_class == "uniform"
             else probe.subnormal_band_hosts(STATE_NAMES, shape, rng))
    for name in STATE_NAMES:
        array = getattr(fields, name, None)
        if array is None:
            continue
        source = hosts.get(name)
        if source is None:
            source = rng.normal(0.0, 0.37, size=shape).astype(np.float32)
        array[...] = cp.asarray(np.ascontiguousarray(
            np.asarray(source, dtype=np.float32)))
    return fields, grid, pml, float(grid.dt / grid.dx)


def state_of(fields) -> Dict[str, Any]:
    return {name: getattr(fields, name) for name in STATE_NAMES
            if getattr(fields, name, None) is not None}


def frozen(fields) -> Dict[str, np.ndarray]:
    return {name: to_host(value).copy() for name, value in state_of(fields).items()}


def compare(left: Mapping[str, np.ndarray],
            right: Mapping[str, np.ndarray]) -> Dict[str, int]:
    if set(left) != set(right):
        return {"__volume_set__": len(set(left) ^ set(right))}
    return {name: n for name in sorted(left)
            if (n := differing(left[name], right[name]))}


def word_total(snapshot: Mapping[str, np.ndarray]) -> int:
    return int(sum(words(value).size for value in snapshot.values()))


def read_only_snapshot(pml) -> Dict[str, np.ndarray]:
    return {f"pml.{name}": to_host(array).copy()
            for name in IMMUTABLE_PML
            if (array := getattr(pml, name, None)) is not None}


def read_only_drift(before: Mapping[str, np.ndarray], pml) -> List[str]:
    after = read_only_snapshot(pml)
    return sorted(name for name in before
                  if name in after and differing(before[name], after[name]))


def run_pass(name: str, fields, pml) -> None:
    scratch_gate.run_pass(name, fields, pml)


# ---------------------------------------------------------------------------
# The arrangements
# ---------------------------------------------------------------------------

class Arrangement:
    """One way of performing a complete driver step, and its own launch counter."""

    __slots__ = ("name", "fields", "grid", "pml", "dtdx", "step", "launches",
                 "state", "notes")

    def __init__(self, name: str, fields, grid, pml, dtdx: float) -> None:
        self.name = name
        self.fields, self.grid, self.pml, self.dtdx = fields, grid, pml, dtdx
        self.launches = 0
        self.state = None
        self.notes: Dict[str, Any] = {}

    def run_step(self) -> None:
        raise NotImplementedError


def _certified_single_launchers(fields, grid, pml, dtdx, variant: str):
    """``{slot: callable}`` -- the certified single-slot launchers for this variant.

    RESOLVED THROUGH EACH CERTIFIED FAMILY'S OWN ENTRY POINT, never re-derived: the
    conductive variant's ``step_D`` is ``conductive_kernels.step_conductive_curl`` and
    its ``step_B`` is the certified lossless curl (a D conductivity leaves step_B's
    targets lossless, which is exactly the arm split the census records on
    ``test_damping``); the BFAST variant's two curls are ``bfast_curl.step_bfast`` on
    the HALF-INTEGER tables for ``step_B`` and the INTEGER ones for ``step_D``, and its
    two constitutive slots are the shipped certified pair, which is what
    ``covers_bfast_constitutive`` admits.
    """
    from meep_gpu.cuda_kernels import (bfast_curl, conductive_kernels,  # noqa: PLC0415
                                       constitutive_kernels, step_curl_kernels)

    curl_b = step_curl_kernels.real_pml_curl_tables(pml, True)
    curl_d = step_curl_kernels.real_pml_curl_tables(pml, False)
    codes = step_curl_kernels.real_curl_boundary_codes(grid)
    tables_h = constitutive_kernels.constitutive_tables_for("H", pml)
    tables_e = constitutive_kernels.constitutive_tables_for("E", pml)
    plans = {
        "update_H": lambda: constitutive_kernels.update_fused_pml_real(
            "H", fields, tables=tables_h),
        "update_E": lambda: constitutive_kernels.update_fused_pml_real(
            "E", fields, tables=tables_e),
    }
    if variant == "conductive":
        plans["step_B"] = lambda: step_curl_kernels._step_B_fused_pml_real(  # noqa: SLF001
            fields, curl_b, codes, dtdx)
        plans["step_D"] = lambda: conductive_kernels.step_conductive_curl(
            fields, pml, "step_D", codes, dtdx, tables=curl_d)
    else:
        k_b = bfast_curl.bfast_curl_coefficients(grid, "step_B")
        k_d = bfast_curl.bfast_curl_coefficients(grid, "step_D")
        plans["step_B"] = lambda: bfast_curl.step_bfast(
            "step_B", fields, curl_b, codes, dtdx, k_b)
        plans["step_D"] = lambda: bfast_curl.step_bfast(
            "step_D", fields, curl_d, codes, dtdx, k_d)
    return plans


#: The released pairs either side of this seam, per variant, WITH THEIR ENTRY POINTS
#: SPELLED. Named rather than discovered: a ``dir()`` scan for the first ``covers_*``
#: on a module picks whatever sorts first -- measured 2026-09-07, it picked
#: ``covers_fill_folded_far`` off the conductive electric pair and died on the arity.
#: A neighbour resolved by accident is a reference arrangement that measures the wrong
#: composition, which is worse than dying.
NEIGHBOURS: Dict[str, Dict[str, Tuple[str, str, str]]] = {
    "conductive": {
        # A D-side conductivity leaves step_B's targets LOSSLESS, so the B->H
        # neighbour on this cell is the certified magnetic pair -- which is the same
        # arm split the census records on `TestAdjointSolver.test_damping`.
        "magnetic": ("fused_magnetic_pair", "covers_fused_magnetic_pair",
                     "run_fused_magnetic_pair"),
        "electric": ("conductive_fused_electric_pair",
                     "covers_conductive_fused_electric_pair",
                     "run_conductive_fused_electric_pair"),
    },
    "bfast": {
        "magnetic": ("bfast_fused_magnetic_pair",
                     "covers_bfast_fused_magnetic_pair",
                     "run_bfast_fused_magnetic_pair"),
        "electric": ("bfast_fused_electric_pair",
                     "covers_bfast_fused_electric_pair",
                     "run_bfast_fused_electric_pair"),
    },
}


def _neighbour_module(variant: str, role: str):
    import importlib  # noqa: PLC0415

    name, _covers, _run = NEIGHBOURS[variant][role]
    return importlib.import_module(f"meep_gpu.cuda_kernels.{name}")


def _neighbour_products(variant: str):
    """``(B->H product, D->E product)`` -- the released pairs either side of this seam."""
    return _neighbour_module(variant, "magnetic"), _neighbour_module(variant, "electric")


def _neighbour_entry(variant: str, role: str, kind: str):
    module = _neighbour_module(variant, role)
    name, covers, run = NEIGHBOURS[variant][role]
    del name
    return getattr(module, covers if kind == "covers" else run)


def _run_neighbour(variant: str, role: str, fields, grid, pml, dtdx) -> Dict[str, Any]:
    """Launch one neighbouring released pair through ITS OWN named entry point."""
    return _neighbour_entry(variant, role, "run")(fields, grid, pml, dtdx, sources=())


def _neighbour_verdict(variant: str, role: str, fields, pml, grid) -> Tuple[bool, str]:
    """One neighbouring pair's own predicate, by name."""
    return _neighbour_entry(variant, role, "covers")(fields, pml, grid, ())


def _holds(label: Any, product) -> bool:
    """Does this selected arm label name that product?

    THE COMPOSER SELECTS BY LABEL AND THE MODULE DECLARES A FAMILY, and the two are
    different spellings of one thing (``cuda_bfast_fused_magnetic_pair`` against
    ``BFAST fused magnetic pair``). Normalised here in ONE place rather than matched by
    substring at each call site -- measured 2026-09-07: a substring match reported
    "neither neighbour holds a slot" on a composition where both did.
    """
    text = str(label).replace("_", " ").casefold().strip()
    family = product.FAMILY.replace("cuda_", "", 1).replace("_", " ").casefold()
    return text == family


def make_arrangement(mode: str, spec: Mapping[str, Any], fields, grid, pml,
                     dtdx: float, *, kernel: Optional[Any] = None,
                     rotate: bool = True, threads: Optional[int] = None,
                     own_load_reads_scratch: bool = False) -> Arrangement:
    """One of :data:`MODES`, built against an already-seeded engine."""
    family = _family()
    variant = spec["variant"]
    arrangement = Arrangement(mode, fields, grid, pml, dtdx)

    if mode == "array":
        def step() -> None:
            for name in STEP_PASSES:
                run_pass(name, fields, pml)

    elif mode in ("singles", "unfused"):
        singles = _certified_single_launchers(fields, grid, pml, dtdx, variant)
        # `singles` dispatches ONLY the seam's two slots; `unfused` dispatches all four
        # of the slot path's. The first isolates the seam, the second is the
        # composition the composer builds with fusion vetoed, and both must agree.
        dispatched = (("update_H", "step_D") if mode == "singles"
                      else ("step_B", "update_H", "step_D", "update_E"))

        def step() -> None:
            for name in STEP_PASSES:
                if name in dispatched:
                    singles[name]()
                    arrangement.launches += 1
                else:
                    run_pass(name, fields, pml)

    elif mode == "composition_today":
        magnetic_product, electric_product = _neighbour_products(variant)
        singles = _certified_single_launchers(fields, grid, pml, dtdx, variant)
        # WHICH NEIGHBOURS ACTUALLY ADMIT THIS FIXTURE IS MEASURED, NOT ASSUMED. Each
        # released pair is asked ONCE, before the first step; where it refuses, the
        # arrangement falls back to certified singles for that half AND RECORDS THE
        # REFUSAL, so a fixture no neighbour reaches is visible in the artifact rather
        # than silently becoming `unfused` under another name.
        installed: Dict[str, Any] = {}
        absorbed: List[str] = []
        for role, product in (("magnetic", magnetic_product),
                              ("electric", electric_product)):
            covered, reason = _neighbour_verdict(variant, role, fields, pml, grid)
            arrangement.notes[f"{role}_neighbour"] = {
                "family": product.FAMILY, "admits": bool(covered),
                "reason": None if covered else reason,
                "replaces": list(product.REPLACES)}
            if covered:
                installed[role] = product
                absorbed.extend(product.REPLACES)
        arrangement.notes["absorbed_passes"] = sorted(set(absorbed))

        def step() -> None:
            for name in STEP_PASSES:
                for role, product in installed.items():
                    if name == product.REPLACES[0]:
                        record = _run_neighbour(variant, role, fields, grid, pml, dtdx)
                        if not record.get("launched"):
                            raise SystemExit(
                                f"the released {role} pair refused mid-run a fixture "
                                f"it admitted at plan time: {record.get('reason')}")
                        arrangement.launches += 1
                if name in absorbed:
                    continue
                if name in ("step_B", "update_H", "step_D", "update_E"):
                    singles[name]()
                    arrangement.launches += 1
                else:
                    run_pass(name, fields, pml)

    elif mode in ("weld", "weld_composed"):
        arrangement.state = family.resolve(fields, grid, pml, dtdx=dtdx)
        family.assert_bindings_are_disjoint(fields, arrangement.state)
        block = family._FUSED_THREADS if threads is None else int(threads)  # noqa: SLF001
        neighbours = (_certified_single_launchers(fields, grid, pml, dtdx, variant)
                      if mode == "weld_composed" else None)
        launch_kernel = kernel
        if own_load_reads_scratch and launch_kernel is None:
            launch_kernel = _byte_neutral_kernel(variant, arrangement.state["cond"])

        def step() -> None:
            for name in STEP_PASSES:
                if name == "step_D":
                    continue           # absorbed by the launch at update_H
                if name != "update_H":
                    if neighbours is not None and name in ("step_B", "update_E"):
                        neighbours[name]()
                        arrangement.launches += 1
                    else:
                        run_pass(name, fields, pml)
                    continue
                family.launch_conductive_bfast_fused_hd_pair(
                    fields, arrangement.state, kernel=launch_kernel, threads=block)
                arrangement.launches += 1
                if rotate:
                    arrangement.state["scratch"] = family.rotate_into_fields(
                        fields, arrangement.state["scratch"])
    else:
        raise ValueError(f"unknown arrangement {mode!r}")

    arrangement.step = step  # type: ignore[assignment]
    return arrangement


def drive(spec: Mapping[str, Any], value_class: str, steps: int, *,
          modes: Sequence[str] = MODES, kernel: Optional[Any] = None,
          rotate: bool = True, threads: Optional[int] = None,
          own_load_reads_scratch: bool = False, seed_scale_bits: int = 0,
          progress: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
    """Step every arrangement side by side from ONE seed, compared per COMPLETE step.

    The comparison is against the ARRAY PATH for every other arrangement, and the
    record carries the first divergence per arrangement rather than stopping the
    others: a leg that stopped at the first disagreement could not say whether the
    weld or a certified single was the one that moved.
    """
    label = f"{spec['label']}/{value_class}"
    engines: Dict[str, Arrangement] = {}
    scale = np.float32(2.0) ** int(seed_scale_bits)
    for mode in modes:
        fields, grid, pml, dtdx = build(spec, value_class, case_rng(label))
        if seed_scale_bits:
            for name, array in state_of(fields).items():
                del name
                array *= scale
        engines[mode] = make_arrangement(
            mode, spec, fields, grid, pml, dtdx,
            kernel=kernel if mode == "weld" else None,
            rotate=rotate if mode == "weld" else True,
            threads=threads if mode == "weld" else None,
            own_load_reads_scratch=own_load_reads_scratch if mode == "weld" else False)
    reference = engines[modes[0]]
    seeded = frozen(reference.fields)
    for mode in modes[1:]:
        mismatch = compare(seeded, frozen(engines[mode].fields))
        if mismatch:
            raise SystemExit(f"{label}: {mode} was not seeded identically: {mismatch}")

    read_only = read_only_snapshot(reference.pml)
    census = probe.operand_census({name: to_host(value)
                                   for name, value in state_of(reference.fields).items()})
    first: Dict[str, Optional[Dict[str, Any]]] = {mode: None for mode in modes[1:]}
    compared = 0
    # WHAT THE ARRAY PATH MOVED AT ANY STEP, accumulated rather than read off the final
    # state. A seed-versus-final comparison is not a liveness measurement: on the
    # reduced-dimension BFAST fixture two of the three IIR histories carry `total == 0`
    # exactly (their k pair is zeroed by the HOST-SIDE invariance gates), so their
    # recurrence is `fb <- -fb` and an even step budget returns them to the seed
    # bit for bit. Measured 2026-09-07, and it failed the floor on a case where every
    # arrangement was byte-identical at every one of the four steps.
    moved_ever: set = set()
    started = time.perf_counter()
    with MemoLaunchCounter() as counter:
        for step in range(steps):
            for mode in modes:
                engines[mode].step()
            cp.cuda.runtime.deviceSynchronize()
            state = {mode: frozen(engines[mode].fields) for mode in modes}
            compared = step + 1
            moved_ever |= {name for name in seeded
                           if differing(seeded[name], state[modes[0]][name])}
            for mode in modes[1:]:
                if first[mode] is not None:
                    continue
                moved = compare(state[modes[0]], state[mode])
                if moved:
                    first[mode] = {"step": step + 1, "volumes": moved}
            if all(first[mode] is not None for mode in modes[1:]):
                break
            if progress is not None and (step + 1) % 10 == 0:
                progress(f"{label} step {step + 1}/{steps}")
        memo_named, memo_total = counter.named(), counter.total

    # THE FLOOR IS A PROPERTY OF THE ARRAY PATH, and it is a floor rather than a list:
    # a fixture may legitimately freeze a component (a reduced dimension does), and
    # demanding that every named volume move would refuse the corpus row's own shape.
    # What may NOT happen is that a case measured nothing, so each GROUP the seam
    # writes must have moved somewhere: the electric primary, its split-field
    # auxiliary, the magnetic half the weld computes, and the variant's own third
    # history -- the last being what makes this cell its own cell rather than the
    # lossless product under another name.
    history = ([f"f_cond_D{axis}" for axis in "xyz"] if spec["variant"] == "conductive"
               else [f"f_bfast_D{axis}" for axis in "xyz"])
    groups = {
        "D": [f"D{axis}" for axis in "xyz"],
        "fu_D": [f"fu_D{axis}" for axis in "xyz"],
        "H": [f"H{axis}" for axis in "xyz"],
        "f_w_H": [f"f_w_H{axis}" for axis in "xyz"],
        "third_history": history,
    }
    exempt = set(BAND_EXEMPT_GROUPS) if value_class == "subnormal_band" else set()
    live = {group: sorted(set(names) & moved_ever) for group, names in groups.items()}
    present = [name for name in history if name in seeded]
    unmoved = sorted(group for group, names in groups.items()
                     if group not in exempt
                     and any(name in seeded for name in names) and not live[group])
    tail_is_live = bool(present) and bool(live["third_history"])
    return {
        "label": label, "spec": spec["label"], "variant": spec["variant"],
        "value_class": value_class, "mask": list(spec.get("mask") or ()),
        "steps_requested": steps, "steps_compared": compared,
        "shape": [int(n) for n in reference.grid.shape],
        "threads": int(threads or _family()._FUSED_THREADS),  # noqa: SLF001
        "rotated": bool(rotate), "seed_scale_bits": int(seed_scale_bits),
        "identical": {mode: first[mode] is None for mode in modes[1:]},
        "first_divergence": {mode: first[mode] for mode in modes[1:]},
        "words_compared_per_step": word_total(seeded),
        "words_compared": word_total(seeded) * compared * max(len(modes) - 1, 1),
        "launches": {mode: engines[mode].launches for mode in modes},
        "arrangement_notes": {mode: engines[mode].notes for mode in modes
                              if engines[mode].notes},
        "memo_launches": {"named": memo_named, "total": memo_total},
        "read_only_drift": read_only_drift(read_only, reference.pml),
        "operand_census": census,
        "volumes_the_array_path_moved_at_some_step": sorted(moved_ever),
        "live_groups": live, "groups_exempt_under_this_value_class": sorted(exempt),
        "unmoved_groups": unmoved, "non_vacuous": not unmoved,
        "third_history_volumes": present,
        "third_history_moved": live["third_history"],
        "the_variants_own_tail_is_live": tail_is_live,
        "seconds": round(time.perf_counter() - started, 2),
        "passed": (not unmoved
                   and not read_only_drift(read_only, reference.pml)
                   and all(first[mode] is None for mode in modes[1:])
                   and compared == steps
                   and tail_is_live),
    }


# ---------------------------------------------------------------------------
# Host leg: the driver's order
# ---------------------------------------------------------------------------

def leg_driver_order() -> Dict[str, Any]:
    """``REPLACES`` is the driver's two ADJACENT consults, and the seam holds one pass."""
    from meep_gpu import driver as driver_module  # noqa: PLC0415

    family = _family()
    source = textwrap.dedent(inspect.getsource(driver_module.FdtdDriver.step))
    tree = ast.parse(source)
    called: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = getattr(node.func, "id", None)
            if name in STEP_PASSES and name not in called:
                called.append(name)
    order = [name for name in STEP_PASSES if name in called]
    replaces = list(family.REPLACES)
    start, end = order.index(replaces[0]), order.index(replaces[-1])
    span = order[start:end + 1]

    body = ast.parse(source).body[0]
    statements = list(ast.walk(body))
    consults = [node for node in statements
                if isinstance(node, ast.Call)
                and getattr(node.func, "attr", None) == "dispatch"]
    consulted = [node.args[0].value for node in consults
                 if node.args and isinstance(node.args[0], ast.Constant)]
    between = [name for name in consulted
               if consulted.index(name) > consulted.index("update_H")
               and consulted.index(name) < consulted.index("step_D")]
    withdraw_calls = sum(
        1 for node in statements
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Call)
        and getattr(node.func.func, "id", None) == "getattr")
    record = {
        "driver_calls": called, "walked_order": order,
        "replaces": replaces, "driver_span": span,
        "replaces_is_the_whole_span": span == replaces,
        "adjacent": order.index("step_D") - order.index("update_H") == 1,
        "consulted_slots_in_order": consulted,
        "slots_consulted_between_the_two": between,
        "the_seam_holds_no_other_consult": between == [],
        "withdraw_getattr_call_sites_in_step": withdraw_calls,
        "seam_module": family.SEAM,
        "seam_module_is_the_withdraw_hoist": family.SEAM == withdraw_hoist.SEAM,
        "carries_deposit_repair": bool(family.CARRIES_DEPOSIT_REPAIR),
        "hoists_the_withdraw": bool(family.HOISTS_THE_WITHDRAW),
        "sync_path_slots": list(SYNC_PATH_SLOTS),
        "sync_pass_owner": dict(SYNC_PASS_OWNERS),
        "step_D_is_outside_the_sync_window": "step_D" not in SYNC_PATH_SLOTS,
        "the_sync_consult_owner_is_this_products_leading_slot":
            SYNC_PASS_OWNERS.get(SYNC_UPDATE_H_PASS) == family.SLOT,
        # THE SPAN IS THE SIBLING'S, exactly. Two products on one seam that disagreed
        # about what the seam IS would be two different claims wearing one name.
        "the_span_is_the_released_siblings_span":
            tuple(family.REPLACES) == tuple(hd_gate._family().REPLACES),  # noqa: SLF001
    }
    record["passed"] = bool(
        record["replaces_is_the_whole_span"] and record["adjacent"]
        and record["the_seam_holds_no_other_consult"]
        and record["seam_module_is_the_withdraw_hoist"]
        and not record["carries_deposit_repair"]
        and record["step_D_is_outside_the_sync_window"]
        and record["the_sync_consult_owner_is_this_products_leading_slot"]
        and record["the_span_is_the_released_siblings_span"]
        and withdraw_calls >= 2)
    return record


# ---------------------------------------------------------------------------
# Host leg: the transcription
# ---------------------------------------------------------------------------

#: Every ``(variant, mask)`` the transcription and the mutation battery emit. The
#: conductive variant is emitted at the ALL-CONDUCTIVE mask and at one MIXED mask, so
#: the check sees both a text where every ``#if COND*`` branch is live and one where
#: the preprocessor drops two of them back to the lossless family's own tail line.
EMITTED: Tuple[Tuple[str, Optional[Tuple[bool, bool, bool]]], ...] = (
    ("conductive", (True, True, True)),
    ("conductive", (True, False, True)),
    ("bfast", None),
)


def _certified_curl_pieces(variant: str,
                           cond: Optional[Tuple[bool, bool, bool]]
                           ) -> Tuple[str, str, str]:
    """``(prelude, parameters, body)`` of the CERTIFIED curl this variant welds."""
    family = _family()
    return family._split_certified(  # noqa: SLF001
        family.certified_curl_text(variant, cond), family.CERTIFIED_CURL[variant])


def leg_transcription() -> Dict[str, Any]:
    """Both halves are the certified emitters' own bytes, with the declared edits only."""
    family = _family()
    per_text: Dict[str, Any] = {}
    findings: List[str] = []

    agreement = family.sibling_agreement()
    for variant, cond in EMITTED:
        key = f"{variant}:{'' .join('C' if f else '-' for f in cond) if cond else 'k'}"
        source = family.kernel_source(variant, cond)
        certified_prelude, certified_params, certified_body = _certified_curl_pieces(
            variant, cond)
        pieces = family.welded_curl_pieces(variant, cond)

        # THE CERTIFIED PARAMETER LIST IS CARRIED ACROSS CHARACTER FOR CHARACTER. Not a
        # substring argument about names: equality against the certified text, which is
        # the strongest form this check can take and the reason this product does not
        # hand-write a signature.
        parameters_verbatim = pieces["parameters"] == certified_params
        parameters_in_source = certified_params in source

        # THE CURL PRELUDE, WHOLE AND UNEDITED -- including whatever helper the owning
        # family put in front of its kernel (`cond_pml_apply` on one variant, nothing
        # on the other).
        prelude_verbatim = pieces["prelude"] == certified_prelude

        # THE CURL BODY, LINE BY LINE. The certified tail below the decode and the
        # welded one must differ in EXACTLY the twelve magnetic reads and nothing else.
        decode = family._hd._INDEX_DECODE  # noqa: SLF001
        certified_tail = certified_body[certified_body.index(decode) + len(decode):]
        left = certified_tail.splitlines()
        right = pieces["tail"].splitlines()
        changed = [(a, b) for a, b in zip(left, right) if a != b]
        edits_are_the_reads = (
            len(left) == len(right)
            and len(changed) == family.HALO_TAPS + family.OWN_LOAD_EDITS
            and all(("shift_dn(" in a and "shift_dn_recompute(" in b)
                    or (any(f"{t}[idx]" in a for t in family.H_TARGETS)
                        and "own_h[" in b)
                    for a, b in changed))

        # NO STALE READ SURVIVES. In the fused signature Hx/Hy/Hz are the PRE-LAUNCH
        # magnetic field, so any indexing of them below the weld would be the very
        # defect the design removes.
        below = source.split("raw_update_H_cell(idx, weld, own_h, own_w);", 1)[1]
        residue = {name: below.count(f"{name}[") for name in family.H_TARGETS}

        # THE VARIANT'S OWN TAIL IS PRESENT WHOLE, and named rather than inferred: a
        # weld that dropped the conductive recurrence or the BFAST insert would still
        # compile and would be the lossless product wearing this one's name.
        if variant == "conductive":
            # THE MASK IS A PREPROCESSOR CHOICE, so BOTH branches are in the text and
            # the three ``#define`` lines are what select them. Checked as the two
            # facts that actually matter: the defines say exactly what the mask says,
            # and each target carries both a conductive and a lossless tail so the
            # ``#else`` really is the certified family's own line.
            defines = [f"#define COND{index} {int(bool(flag))}"
                       for index, flag in enumerate(cond or ())]
            tail_markers = {
                "cond_pml_apply_is_defined_once": source.count(
                    "__device__ __forceinline__ void cond_pml_apply(") == 1,
                # ``(km1 != 1.0f) || (si1 != 1.0f)`` and its dsigu twin: FOUR exact
                # comparisons, and a tolerance in place of any one of them is the
                # armed mutation m_conductive_partition_is_a_tolerance.
                "the_exact_partition_survives": source.count("!= 1.0f") == 4,
                "the_three_store_predicates_survive": (
                    "if (dsigu) u[idx] = u_new;" in source
                    and "if (dsig) c[idx] = c_new;" in source),
                "the_mask_defines_say_what_the_mask_says":
                    all(source.count(line + "\n") == 1 for line in defines)
                    and source.startswith(defines[0]),
                "every_target_carries_both_tails": all(
                    f"cond_pml_apply({target}, fu_{target}," in source
                    and f"pml_apply({target}, fu_{target}," in source
                    for target in family.D_TARGETS),
                "the_four_case_recurrence_is_present_whole": all(
                    marker in source for marker in (
                        "float c_new    = ((cv * cfv) - curl) * civ;",
                        "float u_cond   = ((uv * cfv) - curl) * civ;",
                        "float u_split  = (((uv * km1) + c_new) - cv) * si1;",
                        "float f_split  = (((fv * km2) + u_new) - uv) * si2;",
                        "float f_first  = (((fv * km1) + c_new) - cv) * si1;",
                        "float f_direct = ((fv * cfv) - curl) * civ;")),
            }
        else:
            tail_markers = {
                "the_bfast_sum_survives": source.count("* (sf + f1)") == 3,
                "the_tustin_term_survives": source.count(
                    "float advance = total - (2.0f * bprev);") == 3,
                "the_state_store_survives": all(
                    f"fb_D{axis}[idx] = bprev + advance;" in source
                    for axis in "xyz"),
                "the_double_mask_survives": source.count(
                    "advance = 0.0f;") == source.count("curl = 0.0f;"),
                "the_six_scalars_are_declared": all(
                    f"float k{n}_{g}" in source
                    for n in (1, 2) for g in ("a", "b", "c")),
            }

        record = {
            "kernel": family.kernel_name(variant),
            "lines": source.count("\n"),
            "bytes": len(source),
            "sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
            "certified_parameter_list_is_verbatim": parameters_verbatim,
            "certified_parameter_list_appears_in_the_emitted_source":
                parameters_in_source,
            "certified_curl_prelude_is_verbatim": prelude_verbatim,
            "curl_tail_edits_are_exactly_the_magnetic_reads": edits_are_the_reads,
            "curl_tail_changed_lines": len(changed),
            "expected_changed_lines": family.HALO_TAPS + family.OWN_LOAD_EDITS,
            "shifted_taps": source.count("shift_dn_recompute("),
            "own_cell_registers": below.count("own_h["),
            "no_stale_magnetic_read_below_the_weld": residue,
            "ascii": source.isascii(),
            "encodes_under_ascii": _encodes_ascii(source),
            "pml_apply_defined_once":
                source.count("__device__ __forceinline__ void pml_apply(") == 1,
            "the_kernel_name_is_declared":
                f'void {family.kernel_name(variant)}(' in source,
            "variant_tail": tail_markers,
            "mutation_needles_resolve": _needles_resolve(source, variant),
        }
        record["passed"] = bool(
            parameters_verbatim and parameters_in_source and prelude_verbatim
            and edits_are_the_reads and sum(residue.values()) == 0
            and record["ascii"] and record["encodes_under_ascii"]
            and record["pml_apply_defined_once"]
            and record["the_kernel_name_is_declared"]
            and all(tail_markers.values())
            and record["mutation_needles_resolve"]["passed"])
        if not record["passed"]:
            findings.append(f"{key}: transcription")
        per_text[key] = record

    # THE CONSTITUTIVE HALF IS THE SIBLING'S OWN. Where the released product can emit,
    # the four lifted pieces must be byte-equal to its; where it cannot, that is a
    # RECORDED fact rather than a pass.
    if agreement["available"] and not agreement["agree"]:
        findings.append(f"the lifted constitutive pieces disagree with the released "
                        f"sibling: {agreement['disagreements']}")

    # THE DECLARED EDIT TABLES resolve: every ADDED parameter name appears in the
    # emitted signature and in no certified list.
    added = _added_parameters_resolve()
    if not added["passed"]:
        findings.append("the declared added parameters do not resolve")

    return {
        "texts": len(per_text), "per_text": per_text,
        "sibling_agreement": agreement,
        "added_parameters": added,
        "source_sha256": family.source_digest(),
        "device_string_names": sorted(family.device_sources()),
        "two_device_strings": len(family.device_sources()) == 2,
        "findings": findings,
        "passed": (not findings and all(r["passed"] for r in per_text.values())
                   and len(family.device_sources()) == 2),
    }


def _added_parameters_resolve() -> Dict[str, Any]:
    """Every declared added parameter is in the emitted list and in NO certified one."""
    family = _family()
    rows: Dict[str, Any] = {}
    passed = True
    for variant, cond in EMITTED:
        _prelude, certified, _body = _certified_curl_pieces(variant, cond)
        emitted = family.signature(variant, certified)
        seen: Dict[str, Dict[str, bool]] = {}
        for group, names in family.ADDED_PARAMETERS.items():
            for name in names:
                in_emitted = f" {name},"  in emitted or f" {name}\n" in emitted
                in_certified = (f" {name}," in certified
                                or f" {name}\n" in certified)
                seen[f"{group}.{name}"] = {"emitted": in_emitted,
                                           "in_the_certified_list": in_certified}
                if not in_emitted or in_certified:
                    passed = False
        rows[f"{variant}:{cond}"] = seen
    return {"rows": rows, "passed": passed}


def _encodes_ascii(source: str) -> bool:
    """The NVRTC locale trap, PERFORMED rather than scanned."""
    try:
        source.encode("ascii")
    except UnicodeEncodeError:
        return False
    return True


def _needles_resolve(source: str, variant: str) -> Dict[str, Any]:
    """Every armed device mutation's anchor appears exactly once in the shipped text."""
    counts: Dict[str, int] = {}
    for name, spec in DEVICE_MUTATIONS.items():
        if spec.get("variant") not in (None, variant):
            continue
        counts[name] = source.count(spec["old"])
    return {"counts": counts,
            "passed": all(count == 1 for count in counts.values())}


# ---------------------------------------------------------------------------
# Host leg: the refusals
# ---------------------------------------------------------------------------

class _NoConductivity:
    """The run's fields with every conductivity hidden. A refusal fixture, not a proxy."""

    __slots__ = ("_fields",)

    def __init__(self, fields: Any) -> None:
        object.__setattr__(self, "_fields", fields)

    def condfac_for(self, component: str) -> Any:  # noqa: ARG002
        return None

    def condinv_for(self, component: str) -> Any:  # noqa: ARG002
        return None

    def __getattr__(self, item: str) -> Any:
        return getattr(object.__getattribute__(self, "_fields"), item)


class _BfastGrid:
    """The run's grid claiming an active BFAST. The 'both at once' fixture."""

    __slots__ = ("_grid", "_k")

    def __init__(self, grid: Any, k=(0.4, 0.0, 0.0)) -> None:
        object.__setattr__(self, "_grid", grid)
        object.__setattr__(self, "_k", tuple(k))

    @property
    def bfast_active(self) -> bool:
        return True

    @property
    def bfast_scaled_k(self):
        return object.__getattribute__(self, "_k")

    def __getattr__(self, item: str) -> Any:
        return getattr(object.__getattribute__(self, "_grid"), item)


class _Source:
    """A source object shaped exactly as ``withdraw_hoist`` reads one.

    THE THREE THINGS ``_withdraw_does_work`` GATES ON, and all three are needed: a
    callable ``withdraw``, ``is_integrated`` True, and a positive
    ``_n_source_points``. A stub carrying only the first two is a source whose
    withdraw does NOTHING, and a refusal fixture built from it measures the
    predicate's silence rather than its refusal -- which is what the first run of this
    leg reported (2026-09-07: `standing_integrated_electric_withdraw` came back
    `covered: True`).
    """

    def __init__(self, field_type: str, is_integrated: bool, points: int = 5) -> None:
        self.field_type = field_type
        self.is_integrated = is_integrated
        self._n_source_points = int(points)
        self._applied_dipole = None

    def withdraw(self, fields: Any) -> None:  # pragma: no cover - never called here
        del fields


def leg_refusal() -> Dict[str, Any]:
    """Every configuration this launch cannot serve, refused BY NAME."""
    family = _family()
    cases: Dict[str, Any] = {}
    findings: List[str] = []

    def verdict(name: str, fields, pml, grid, sources, expect: bool,
                marker: Optional[str] = None) -> None:
        covered, reason = family.covers_conductive_bfast_fused_hd_pair(
            fields, pml, grid, sources)
        row = {"covered": bool(covered), "reason": reason, "expected": expect,
               "marker": marker,
               # WHICH HALF SPOKE, RECORDED. A conjunction refuses at its FIRST clause,
               # so a configuration this product has its own clause for may be named by
               # a certified half instead -- which is a correct refusal and a fact worth
               # having in the artifact rather than a failure.
               "named_by": (reason.split(":", 1)[0] if not covered and ":" in reason
                            else None),
               "marker_present": marker is None or marker.lower() in reason.lower()}
        cases[name] = row
        if bool(covered) != expect or not row["marker_present"]:
            findings.append(name)

    if cp is None:
        return {"passed": False, "findings": ["no device"],
                "reason": "the refusal fixtures are built on the device"}

    cond_spec = BY_LABEL["cond_periodic_CCC"]
    bfast_spec = BY_LABEL["bfast_periodic_3d"]
    c_fields, c_grid, c_pml, _ = build(cond_spec, "uniform", case_rng("refusal_cond"))
    b_fields, b_grid, b_pml, _ = build(bfast_spec, "uniform",
                                       case_rng("refusal_bfast"))

    # THE TWO ADMISSIONS FIRST, so a leg that refused everything cannot pass.
    verdict("conductive_admitted", c_fields, c_pml, c_grid, (), True)
    verdict("bfast_admitted", b_fields, b_pml, b_grid, (), True)

    # A non-integrated electric source is NOT refused; nor is a magnetic one.
    verdict("plain_electric_source_is_not_refused", c_fields, c_pml, c_grid,
            (_Source("D", False),), True)
    verdict("magnetic_source_is_not_refused", c_fields, c_pml, c_grid,
            (_Source("B", True),), True)

    # The refusals.
    verdict("undeclared_source_list", c_fields, c_pml, c_grid, None, False,
            "the source set was not declared")
    verdict("standing_integrated_electric_withdraw", c_fields, c_pml, c_grid,
            (_Source("D", True),), False, "standing integrated")
    verdict("lossless_step_D_belongs_to_the_released_sibling",
            _NoConductivity(c_fields), c_pml, c_grid, (), False,
            "neither a BFAST grid nor a conductivity")
    verdict("conductive_and_bfast_together", c_fields, c_pml, _BfastGrid(c_grid), (),
            False, "There is no such cell")
    verdict("bfast_without_the_conductivity_is_still_bfast",
            _NoConductivity(b_fields), b_pml, b_grid, (), True)

    # A FOLDED BFAST GRID, refused by name -- ``bfast_curl``'s clause 4.
    fold_fields, fold_grid, fold_pml, _ = build(
        {**bfast_spec, "label": "bfast_fold", "cell": (1.6, 2.0, 1.6),
         "boundaries": None, "symmetry": (("Y", 1),)},
        "uniform", case_rng("refusal_bfast_fold"))
    verdict("folded_bfast_grid", fold_fields, fold_pml, fold_grid, (), False,
            "mirror symmetry with BFAST")

    # A NO-ABSORBER CONDUCTIVE RUN. This product's signature binds an f_cond history
    # that a run without an absorber never allocates.
    class _InertLayer:
        def __init__(self, pml: Any) -> None:
            self._pml = pml

        is_active = False

        def __getattr__(self, item: str) -> Any:
            return getattr(self._pml, item)

    # THE MARKER IS "active" AND NOT THIS PRODUCT'S OWN SENTENCE, because the
    # CONSTITUTIVE half is asked first and refuses an inert layer by name before this
    # product's ACTIVE_LAYER_ONLY clause is reached. That ordering is the driver's --
    # update_H runs first -- and the refusal is correct either way; ``named_by``
    # records which half spoke, so a later reordering is visible rather than silent.
    verdict("conductive_without_an_active_layer", c_fields, _InertLayer(c_pml), c_grid,
            (), False, "active")

    # THE MISSING ROTATION TARGET. The predicate refuses a run whose H storage is not
    # allocated, because the rotation has nothing to rebind.
    class _NoStoredH:
        def __init__(self, fields: Any) -> None:
            object.__setattr__(self, "_fields", fields)

        Hx = None

        def __getattr__(self, item: str) -> Any:
            return getattr(object.__getattribute__(self, "_fields"), item)

    verdict("no_stored_H", _NoStoredH(c_fields), c_pml, c_grid, (), False,
            "Hx is not allocated")

    facts = {
        "cases": cases, "findings": findings,
        "installable": bool(family.INSTALLABLE),
        "installable_reason_names_the_algebra":
            "4 - (installed pairs)" in family.INSTALLABLE_REASON,
        "hoists_the_withdraw": bool(family.HOISTS_THE_WITHDRAW),
        "carries_deposit_repair": bool(family.CARRIES_DEPOSIT_REPAIR),
        "active_layer_only": bool(family.ACTIVE_LAYER_ONLY),
    }
    facts["passed"] = bool(not findings and not family.INSTALLABLE
                           and facts["installable_reason_names_the_algebra"])
    return facts


# ---------------------------------------------------------------------------
# Host leg: the arbitration
# ---------------------------------------------------------------------------

def leg_arbitration() -> Dict[str, Any]:
    """The shipped composer, asked -- on both variants.

    WRITTEN FOR THE UNWIRED TREE, WHICH IS WHY THE POSITIONAL FACTS ARE RECORDED AND
    THE SUBSTANTIVE ONES ASSERTED. Whether a row for this product exists in
    ``fused_pairs``' tables at all is exactly what a wiring change inverts, so it is
    reported and never required. What IS required is what a wiring change must not
    change:

    * the composer's SELECTION is the same whether or not this product is present;
    * the released neighbours keep the slots they had;
    * where the composer names this product at all, it names it as REFUSED, and the
      refusal text carries ``INSTALLABLE`` False.
    """
    from meep_gpu.cuda_kernels import arms  # noqa: PLC0415

    if cp is None:
        return {"passed": False, "findings": ["no device"],
                "reason": "the composer is asked against a real engine"}
    family = _family()
    rows: Dict[str, Any] = {}
    findings: List[str] = []
    for label in ("cond_periodic_CCC", "bfast_periodic_3d"):
        spec = BY_LABEL[label]
        fields, grid, pml, dtdx = build(spec, "uniform", case_rng(f"arb/{label}"))
        del dtdx
        plan = arms.plan_step(fields=fields, pml=pml, grid=grid, sources=(), fuse=True)
        selected = dict(getattr(plan, "selected", {}) or {})
        reasons = {key: list(texts)
                   for key, texts in (getattr(plan, "reasons", {}) or {}).items()}
        named = {key: texts for key, texts in reasons.items() if family.FAMILY in key}
        unfused = arms.plan_step(fields=fields, pml=pml, grid=grid, sources=(),
                                 fuse=False)
        magnetic_product, electric_product = _neighbour_products(spec["variant"])
        neighbours = {
            "magnetic": {"family": magnetic_product.FAMILY,
                         "holds": [slot for slot, arm in selected.items()
                                   if _holds(arm, magnetic_product)]},
            "electric": {"family": electric_product.FAMILY,
                         "holds": [slot for slot, arm in selected.items()
                                   if _holds(arm, electric_product)]},
        }
        row = {
            "spec": label, "variant": spec["variant"],
            "selected": {slot: str(arm) for slot, arm in selected.items()},
            "selected_unfused": {slot: str(arm) for slot, arm
                                 in (getattr(unfused, "selected", {}) or {}).items()},
            # POSITIONAL, RECORDED NOT ASSERTED: the unwired tree has no row for this
            # product, so the composer cannot name it. A wiring change makes this True
            # and must not change anything below it.
            "the_composer_names_this_product": bool(named),
            "refusal_texts": {key: texts for key, texts in named.items()},
            "this_product_holds_no_slot": not any(
                _holds(arm, family) for arm in selected.values()),
            # THE SELECTION IS THE INVARIANT A WIRING CHANGE MUST NOT MOVE, and the
            # two are compared here rather than remembered: fusing must change WHICH
            # products hold the four slots and must never leave a slot unfilled.
            "every_slot_is_filled_fused_and_unfused": (
                sorted(selected) == sorted(getattr(unfused, "selected", {}) or {})),
            "the_refusal_names_installable_false": (
                all("INSTALLABLE" in text or "installable" in text
                    for texts in named.values() for text in texts)
                if named else None),
            "neighbours": neighbours,
            "the_released_neighbours_keep_their_slots": bool(
                neighbours["magnetic"]["holds"] or neighbours["electric"]["holds"]),
        }
        rows[label] = row
        if not row["this_product_holds_no_slot"]:
            findings.append(f"{label}: the composer gave this product a slot while it "
                            f"declares INSTALLABLE False")
        if named and row["the_refusal_names_installable_false"] is False:
            findings.append(f"{label}: the composer names this product but its refusal "
                            f"does not cite INSTALLABLE")
        if not row["the_released_neighbours_keep_their_slots"]:
            findings.append(f"{label}: neither released neighbour holds a slot, so "
                            f"this fixture measures nothing about arbitration")
    return {"rows": rows, "findings": findings,
            "installable": bool(family.INSTALLABLE),
            "installable_reason": family.INSTALLABLE_REASON,
            "what_is_recorded_rather_than_asserted": (
                "whether the composer's tables carry a row for this product at all. "
                "This gate is written for the UNWIRED tree; the wiring change inverts "
                "exactly that fact and must leave every asserted clause here "
                "unchanged"),
            "passed": not findings}


# ---------------------------------------------------------------------------
# Host leg: is a metallic ghost observable at all?
# ---------------------------------------------------------------------------

def leg_ghost_observability() -> Dict[str, Any]:
    """Whether a metallic ghost's exact 0.0f can reach ``step_D``'s output.

    THE COINCIDENCE, MEASURED, AND WHAT IT COSTS THE MUTATION SET, STATED. Each
    ``shift_dn`` tap's ghost fires on the plane ``ia == 0`` of the axis it steps along;
    the certified body then zeroes that component's curl on exactly that plane under
    ``BC_METALLIC``. Read off the emitted text per variant: for every tap, is there a
    mask line naming the SAME boundary code and the SAME index?

    Where the answer is yes for all six, no edit to what the metallic ghost serves can
    move a byte, the ghost mutation is a RECORDED NULL, and the PERIODIC WRAP is the
    armed sibling -- which is observable, because the mask does not fire there.
    """
    family = _family()
    per_variant: Dict[str, Any] = {}
    findings: List[str] = []
    for variant, cond in (("conductive", (True, True, True)), ("bfast", None)):
        pieces = family.welded_curl_pieces(variant, cond)
        tail = pieces["tail"]
        taps: List[Dict[str, Any]] = []
        for line in tail.splitlines():
            if "shift_dn_recompute(" not in line:
                continue
            arguments = line.split("shift_dn_recompute(", 1)[1].split(")", 1)[0]
            parts = [part.strip() for part in arguments.split(",")]
            # (comp, idx, <axis index>, <extent>, <stride>, <bc>, weld)
            axis_index, boundary = parts[2], parts[5]
            block = tail[tail.rindex("{", 0, tail.index(line)):]
            block = block[:block.index("}")] if "}" in block else block
            masked = f"if ({boundary} == BC_METALLIC && {axis_index} == 0)" in block
            taps.append({"tap": line.strip(), "axis_index": axis_index,
                         "boundary": boundary, "masked_where_the_ghost_fires": masked})
        row = {
            "taps": taps,
            "tap_count": len(taps),
            "every_tap_is_masked_where_its_ghost_fires":
                bool(taps) and all(t["masked_where_the_ghost_fires"] for t in taps),
            "ownership_mask_clauses": tail.count("BC_METALLIC &&"),
        }
        if len(taps) != family.HALO_TAPS:
            findings.append(f"{variant}: {len(taps)} taps, not {family.HALO_TAPS}")
        per_variant[variant] = row
    return {
        "per_variant": per_variant, "findings": findings,
        "what_this_licenses": (
            "where every tap is masked at the plane its metallic ghost fires on, no "
            "edit to what that ghost serves can move a byte of step_D's output -- so "
            "this gate's ghost mutation is a RECORDED NULL and its PERIODIC sibling is "
            "the armed one. It licenses nothing about the periodic wrap, which is "
            "observable and is measured by m_periodic_wrap_reads_the_near_row"),
        "passed": not findings,
    }


# ---------------------------------------------------------------------------
# Device leg: the purity ledger and the race ledger
# ---------------------------------------------------------------------------

def leg_purity_and_race(specs: Sequence[Mapping[str, Any]],
                        steps: int) -> Dict[str, Any]:
    """How much the recompute is worth, and whether the schedule can be seen.

    THE PURITY LEDGER. On the ARRAY PATH, per fixture: run ``update_H`` from a frozen
    state and count, over the curl's valid foreign taps, how many land on a cell where
    ``H`` MOVED. That count is the number of taps whose recomputed value DIFFERS from
    what an in-place weld would have raced on. Near zero would mean the byte identity
    licenses nothing, because the hazard was never live.

    THE RACE LEDGER. The same byte identity at six block sizes. The design's whole
    claim is that the answer does not depend on which block ran first, and a block
    size that never moves cannot expose the opposite.
    """
    ledger: List[Dict[str, Any]] = []
    for spec in specs:
        fields, grid, pml, dtdx = build(spec, "uniform", case_rng(spec["label"]))
        del dtdx
        before = {name: to_host(getattr(fields, name)).copy()
                  for name in ("Hx", "Hy", "Hz")}
        stepping.update_H(fields, pml)
        cp.cuda.runtime.deviceSynchronize()
        after = {name: to_host(getattr(fields, name)).copy()
                 for name in ("Hx", "Hy", "Hz")}
        moved = {name: (words(before[name]) != words(after[name]))
                 for name in before}
        shape = tuple(int(n) for n in grid.shape)
        total = int(np.prod(shape))
        moved_cells = {name: int(mask.sum()) for name, mask in moved.items()}
        ledger.append({
            "spec": spec["label"], "variant": spec["variant"], "shape": list(shape),
            "cells": total,
            "cells_update_H_moved": moved_cells,
            "fraction_moved": {name: round(count / max(total, 1), 4)
                               for name, count in moved_cells.items()},
            # EVERY foreign tap this kernel makes reads a cell update_H may have moved,
            # so the ledger's headline is the per-component fraction: a fixture where
            # H moved nowhere would make the recompute decorative.
            "the_hazard_is_live": all(count > 0 for count in moved_cells.values()),
        })
        log(f"[purity] {spec['label']:28s} moved="
            f"{ledger[-1]['cells_update_H_moved']} of {total}")

    race: List[Dict[str, Any]] = []
    for spec in specs:
        for threads in BLOCK_SIZES:
            record = drive(spec, "uniform", min(steps, 12),
                           modes=("array", "weld"), threads=threads)
            race.append({"spec": spec["label"], "variant": spec["variant"],
                         "threads": threads,
                         "identical": record["identical"]["weld"],
                         "steps_compared": record["steps_compared"],
                         "words_compared": record["words_compared"]})
            log(f"[race] {spec['label']:28s} b{threads:<5d} "
                f"{'IDENTICAL' if race[-1]['identical'] else 'DIVERGED'}")
    return {
        "purity_ledger": ledger, "race_ledger": race,
        "block_sizes": list(BLOCK_SIZES),
        "the_hazard_is_live_on_every_fixture": bool(ledger) and all(
            row["the_hazard_is_live"] for row in ledger),
        "identical_at_every_block_size": bool(race) and all(
            row["identical"] for row in race),
        "passed": bool(ledger and race
                       and all(row["the_hazard_is_live"] for row in ledger)
                       and all(row["identical"] for row in race)),
    }


# ---------------------------------------------------------------------------
# Device leg: the seed scale
# ---------------------------------------------------------------------------

def leg_seed_scale(steps: int) -> Dict[str, Any]:
    """The seed exponent is a change of exponent and nothing else.

    DRIVEN, not asserted. One fixture of each variant runs at three scales and the
    verdict must be the same at each; the record carries the denormal census at each so
    a reader can see why the shipped scale is the one it is.
    """
    rows: Dict[str, Any] = {}
    findings: List[str] = []
    for label in ("cond_walls_CCC", "bfast_periodic_3d"):
        spec = BY_LABEL[label]
        per_scale: Dict[str, Any] = {}
        for bits in (0, 20, 40):
            record = drive(spec, "uniform", min(steps, 12),
                           modes=("array", "weld"), seed_scale_bits=bits)
            per_scale[f"2^{bits}"] = {
                "identical": record["identical"]["weld"],
                "first_divergence": record["first_divergence"]["weld"],
                "steps_compared": record["steps_compared"],
                "subnormals": record["operand_census"]["subnormals"]}
            log(f"[seed_scale] {label:28s} 2^{bits:<3d} "
                f"{'IDENTICAL' if per_scale[f'2^{bits}']['identical'] else 'DIVERGED'}")
        rows[label] = per_scale
        verdicts = {row["identical"] for row in per_scale.values()}
        if verdicts != {True}:
            findings.append(f"{label}: the verdict is not the same at every scale")
    return {"rows": rows, "findings": findings,
            "what_this_measures": (
                "the solver is LINEAR in the field state and every coefficient it "
                "multiplies by is field-independent, so scaling every stored volume "
                "by a POWER OF TWO shifts each float32 exponent and leaves every "
                "mantissa and every rounding decision untouched. A verdict that "
                "changed with the exponent would be a claim about the band rather "
                "than about the kernel"),
            "passed": not findings}


# ---------------------------------------------------------------------------
# Device leg: the launch structure
# ---------------------------------------------------------------------------

def leg_launch_structure(steps: int) -> Dict[str, Any]:
    """Launches per step at the seam AND over the whole step, two counters.

    THE HONEST NUMBER IS REPORTED WHETHER OR NOT IT FLATTERS THE WELD: against the
    composition the composer installs today the whole-step launch count does not fall.
    That is the arbitration ruling as a measurement.
    """
    rows: Dict[str, Any] = {}
    findings: List[str] = []
    for label in ("cond_periodic_CCC", "bfast_periodic_3d"):
        spec = BY_LABEL[label]
        record = drive(spec, "uniform", min(steps, 8),
                       modes=("array", "singles", "composition_today", "unfused",
                              "weld", "weld_composed"))
        per_step = {mode: round(count / max(record["steps_compared"], 1), 3)
                    for mode, count in record["launches"].items()}
        rows[label] = {
            "variant": spec["variant"],
            "launches_total": record["launches"],
            "launches_per_step": per_step,
            "memo_launches": record["memo_launches"],
            "arrangement_notes": record["arrangement_notes"],
            "identical": record["identical"],
            "the_weld_is_one_launch_at_the_seam": per_step["weld"] == 1.0,
            "the_certified_singles_are_two":
                per_step["singles"] == 2.0,
            "the_weld_composed_against_the_composition_today": {
                "weld_composed": per_step["weld_composed"],
                "composition_today": per_step["composition_today"],
                "the_weld_does_not_reduce_the_step":
                    per_step["weld_composed"] >= per_step["composition_today"]},
        }
        if not rows[label]["the_weld_is_one_launch_at_the_seam"]:
            findings.append(f"{label}: the weld is not one launch per step")
        if not rows[label]["the_certified_singles_are_two"]:
            findings.append(f"{label}: the certified singles are not two launches")
        if not all(record["identical"].values()):
            findings.append(f"{label}: an arrangement diverged inside the launch leg")
        log(f"[launch_structure] {label:28s} {per_step}")
    return {"rows": rows, "findings": findings,
            "two_counters": ("the arrangements' own counts and an independent wrapper "
                             "around the shipped compile memo"),
            "passed": not findings}


# ---------------------------------------------------------------------------
# The driver-level legs
# ---------------------------------------------------------------------------

def _build_driver(spec: Mapping[str, Any], *, integrated: bool = False,
                  seed: int = SEED, scale_bits: int = SEED_SCALE_BITS):
    """One seeded ``FdtdDriver`` on the device, with one electric source."""
    from meep_gpu.driver import FdtdDriver  # noqa: PLC0415

    boundaries = ({axis: kind for axis, kind in zip("xyz", spec["boundaries"])}
                  if spec["boundaries"] else None)
    extra: Dict[str, Any] = {}
    if spec.get("bfast_k") is not None:
        extra["bfast_scaled_k"] = tuple(spec["bfast_k"])
    driver = FdtdDriver(cell_size=spec["cell"], resolution=10.0, courant=0.35,
                        force_complex_fields=False, boundaries=boundaries,
                        symmetry=[{"direction": axis, "phase": phase}
                                  for axis, phase in spec["symmetry"]],
                        dimensions=int(spec.get("dimensions") or 3),
                        prefer_gpu=True, gpu_id=0, **extra)
    driver.setup_pml(2)
    shape = tuple(int(n) for n in driver.grid.shape)
    driver.fields.set_epsilon_volumes(
        {name: cp.full(shape, np.float32(value), cp.float32)
         for name, value in (("Ex", 2.0), ("Ey", 2.5), ("Ez", 3.0))},
        {name: cp.full(shape, np.float32(1.0 / value), cp.float32)
         for name, value in (("Ex", 2.0), ("Ey", 2.5), ("Ez", 3.0))})
    if spec["variant"] == "conductive":
        _install_conductivity(driver.fields, driver.grid, spec,
                              np.random.default_rng(seed + 7))
    driver.add_source({"component": "Ez", "center": (0.0, 0.0, 0.0),
                       "size": (0.0, 0.0, 0.0), "frequency": 1.0,
                       "source_type": "continuous" if integrated else "gaussian",
                       **({"is_integrated": True} if integrated
                          else {"fwidth": 0.2})})
    rng = np.random.default_rng(seed)
    scale = np.float32(2.0) ** int(scale_bits)
    for name in STATE_NAMES:
        array = getattr(driver.fields, name, None)
        if array is None:
            continue
        host = (0.37 * rng.standard_normal(array.shape)).astype(np.float32) * scale
        array[...] = cp.asarray(np.ascontiguousarray(host))
    driver.invalidate_fast_path()
    return driver


def _weld_plan_for(driver, *, hoisted: bool = False,
                   placement: str = withdraw_hoist.BEFORE_UPDATE_H):
    """``(plans, state)`` -- the weld installed at ``update_H`` with ``step_D`` absorbed."""
    family = _family()
    fields, pml, grid = driver.fields, driver.pml, driver.grid
    dtdx = float(grid.dt / grid.dx)
    resolved = family.resolve(fields, grid, pml, dtdx=dtdx)
    state = {"resolved": resolved, "launches": 0, "withdrawn": 0}

    def run() -> None:
        if hoisted:
            state["withdrawn"] += withdraw_hoist.hoist(
                fields, driver._sources, span=family.REPLACES,  # noqa: SLF001
                placement=placement)
        family.launch_conductive_bfast_fused_hd_pair(fields, resolved)
        state["launches"] += 1
        resolved["scratch"] = family.rotate_into_fields(fields, resolved["scratch"])

    return {"update_H": run, "step_D": lambda: None}, state


def _driver_state(driver) -> Dict[str, np.ndarray]:
    return {name: to_host(array).copy()
            for name in STATE_NAMES
            if (array := getattr(driver.fields, name, None)) is not None}


def leg_sync(steps: int) -> Dict[str, Any]:
    """The magnetic half-step's channel, with the product FORCE-INSTALLED.

    ``synchronize_magnetic_fields`` repeats the magnetic half of ``step`` and then
    UNDOES it, but its backup is ``_SYNC_FIELDS`` + ``_SYNC_AUXILIARY`` -- magnetic
    names only. ``D``, ``fu_D``, ``f_cond_D`` and ``f_bfast_D`` are in NEITHER, so a
    product spanning ``update_H`` and ``step_D`` that answered the
    ``update_H_synchronize`` consult would advance the electric state inside a window
    nothing can undo, on every flux and energy call.

    THIS LEG IS THE CITATION FOR THE COMPOSER REFUSAL and it has its own null control:
    the hazard arm MUST diverge in D/fu_D against a run that never synchronizes, and
    the guarded arm -- which declines the consult by the containment rule -- MUST NOT.
    """
    from meep_gpu.driver import FdtdDriver  # noqa: PLC0415

    rows: Dict[str, Any] = {}
    findings: List[str] = []
    sync_fields = list(getattr(FdtdDriver, "_SYNC_FIELDS"))
    sync_auxiliary = list(getattr(FdtdDriver, "_SYNC_AUXILIARY"))
    electric_backed_up = [name for name in sync_fields + sync_auxiliary
                          if name.startswith(("D", "fu_D", "f_cond_D", "f_bfast_D"))]
    for label in ("cond_periodic_CCC", "bfast_periodic_3d"):
        spec = BY_LABEL[label]
        arms_out: Dict[str, Any] = {}
        baselines: Dict[str, Dict[str, np.ndarray]] = {}
        for arm, answers, synchronizes in (("no_sync", False, False),
                                           ("guarded", False, True),
                                           ("hazard", True, True)):
            driver = _build_driver(spec)
            plans, state = _weld_plan_for(driver)
            driver._fast_path = Shim(plans,  # noqa: SLF001
                                     answers_the_sync_consult=answers)
            driver._fast_path_stale = False  # noqa: SLF001
            for step in range(steps):
                driver.step()
                if synchronizes and (step + 1) in SYNC_STEPS:
                    driver.synchronize_magnetic_fields()
                    driver.field_energy_in_box()
                    driver.restore_magnetic_fields()
            cp.cuda.runtime.deviceSynchronize()
            baselines[arm] = _driver_state(driver)
            arms_out[arm] = {"launches": state["launches"],
                             "consults": dict(driver._fast_path.calls)}  # noqa: SLF001
            driver.close()
        electric = tuple([f"{stem}{axis}" for stem in ("D", "fu_D") for axis in "xyz"]
                         + [f"f_cond_D{axis}" for axis in "xyz"]
                         + [f"f_bfast_D{axis}" for axis in "xyz"])
        for arm in ("guarded", "hazard"):
            moved = compare(baselines["no_sync"], baselines[arm])
            arms_out[arm]["differing_volumes"] = moved
            arms_out[arm]["differing_electric_volumes"] = {
                name: n for name, n in moved.items() if name in electric}
        row = {
            "variant": spec["variant"], "arms": arms_out,
            "the_guarded_arm_is_identical":
                not arms_out["guarded"]["differing_volumes"],
            "the_hazard_arm_diverges_in_D":
                bool(arms_out["hazard"]["differing_electric_volumes"]),
        }
        rows[label] = row
        if not row["the_guarded_arm_is_identical"]:
            findings.append(f"{label}: the guarded arm diverged")
        if not row["the_hazard_arm_diverges_in_D"]:
            findings.append(f"{label}: the hazard arm did NOT diverge, so this leg "
                            f"cites nothing")
        log(f"[sync] {label:28s} guarded_identical="
            f"{row['the_guarded_arm_is_identical']} hazard_diverges="
            f"{row['the_hazard_arm_diverges_in_D']}")
    return {"rows": rows, "findings": findings,
            "steps": steps, "sync_steps": list(SYNC_STEPS),
            "sync_fields": sync_fields, "sync_auxiliary": sync_auxiliary,
            "electric_volumes_in_a_backup_list": electric_backed_up,
            "the_electric_state_is_in_neither_backup_list":
                not electric_backed_up,
            "what_this_cites": (
                "a product spanning update_H and step_D advances D, fu_D and the "
                "variant's third history inside synchronize_magnetic_fields, whose "
                "backup list holds none of them, so the restore cannot reach them. The "
                "containment rule in FastPathPlan.dispatch is what stops it, and this "
                "leg is the measurement behind that rule for THIS product"),
            "passed": bool(not findings and not electric_backed_up)}


def leg_withdraw(steps: int) -> Dict[str, Any]:
    """The seam's one pass, on a device, with both null controls.

    The configuration this predicate REFUSES today -- a standing integrated electric
    withdraw -- driven anyway with the hoist wired, so what the refusal costs is
    measured rather than assumed. Three arrangements, each stepped in lockstep with its
    own array-path reference and compared per COMPLETE step:

    ``hoisted``      the withdraw performed immediately before the launch. IDENTICAL.
    ``not_hoisted``  the same launch with no hoist: the curl reads a D still holding
                     the previous step's standing dipole. MUST diverge.
    ``after_step_D`` the campaign's own null control, at the wrong placement. MUST be
                     refused by name or diverge -- never launch and agree.

    THIS LEG SEEDS AT 2^0 AND THE OTHER DRIVER LEGS AT 2^80, AND THE EXPONENT IS THE
    MEASUREMENT HERE RATHER THAN A SETTING. Every other driver leg scales the seed away
    from the denormal band; this one compares a run whose ONLY difference is a source
    dipole of order one, and at 2^80 that dipole is eighty binades below the state it
    is added to -- ``(f + d) == f`` exactly in float32, so the withdraw becomes a no-op
    and BOTH null controls report "did not diverge" for a reason that has nothing to do
    with the seam. Measured 2026-09-07: at 2^80 the un-hoisted arm was byte-identical
    to the array path on both variants over six complete steps while the hoist itself
    performed six withdraws.
    """
    rows: Dict[str, Any] = {}
    findings: List[str] = []
    for label in ("cond_periodic_CCC", "bfast_periodic_3d"):
        spec = BY_LABEL[label]
        # THE PREDICATE'S OWN VERDICT ON THIS CONFIGURATION, first: it must REFUSE.
        family = _family()
        probe_driver = _build_driver(spec, integrated=True, scale_bits=0)
        covered, reason = family.covers_conductive_bfast_fused_hd_pair(
            probe_driver.fields, probe_driver.pml, probe_driver.grid,
            tuple(probe_driver._sources))  # noqa: SLF001
        probe_driver.close()

        arms_out: Dict[str, Any] = {}
        for arm, hoisted, placement in (
                ("hoisted", True, withdraw_hoist.BEFORE_UPDATE_H),
                ("not_hoisted", False, withdraw_hoist.BEFORE_UPDATE_H),
                ("after_step_D", True, withdraw_hoist.AFTER_STEP_D)):
            subject = _build_driver(spec, integrated=True, scale_bits=0)
            reference = _build_driver(spec, integrated=True, scale_bits=0)
            plans, state = _weld_plan_for(subject, hoisted=hoisted,
                                          placement=placement)
            subject._fast_path = Shim(plans)  # noqa: SLF001
            subject._fast_path_stale = False  # noqa: SLF001
            reference._fast_path = None  # noqa: SLF001
            reference._fast_path_stale = False  # noqa: SLF001
            first: Optional[Dict[str, Any]] = None
            refused: Optional[str] = None
            try:
                for step in range(steps):
                    subject.step()
                    reference.step()
                    cp.cuda.runtime.deviceSynchronize()
                    if first is None:
                        moved = compare(_driver_state(reference),
                                        _driver_state(subject))
                        if moved:
                            first = {"step": step + 1, "volumes": moved}
            except withdraw_hoist.WithdrawNotHoistable as error:
                # THE STRONGEST FORM THE NULL CONTROL CAN TAKE, and it is the module's
                # own doing rather than this leg's: `after_step_D` is not merely wrong,
                # it is REFUSED BY NAME before a single launch, with the campaign that
                # measured its divergence quoted in the refusal. A leg that swallowed
                # this and reported "did not diverge" would be scoring the wrong thing.
                refused = str(error)[:600]
            arms_out[arm] = {"identical": first is None and refused is None,
                             "first_divergence": first,
                             "refused_by_name": refused,
                             "launches": state["launches"],
                             "withdraws_hoisted": state["withdrawn"]}
            subject.close()
            reference.close()
        row = {
            "variant": spec["variant"],
            "the_predicate_refuses_this_configuration": not covered,
            "refusal_reason": reason,
            "arms": arms_out,
            "the_hoisted_arrangement_is_identical": arms_out["hoisted"]["identical"],
            "the_unhoisted_arrangement_diverges":
                not arms_out["not_hoisted"]["identical"],
            # EITHER OUTCOME SATISFIES IT, and the record says which: the wrong
            # placement is refused by name before it launches, or (if some future
            # widening admits it) it launches and diverges. What is forbidden is that
            # it launches and agrees.
            "the_wrong_placement_is_refused_or_diverges":
                not arms_out["after_step_D"]["identical"],
            "the_wrong_placement_was_refused_by_name":
                arms_out["after_step_D"]["refused_by_name"] is not None,
        }
        rows[label] = row
        for key in ("the_predicate_refuses_this_configuration",
                    "the_hoisted_arrangement_is_identical",
                    "the_unhoisted_arrangement_diverges",
                    "the_wrong_placement_is_refused_or_diverges"):
            if not row[key]:
                findings.append(f"{label}: {key} is False")
        log(f"[withdraw] {label:28s} refused={row['the_predicate_refuses_this_configuration']} "
            f"hoisted={row['the_hoisted_arrangement_is_identical']} "
            f"unhoisted_diverges={row['the_unhoisted_arrangement_diverges']}")
    return {"rows": rows, "findings": findings,
            "what_this_measures": (
                "what the HOISTS_THE_WITHDRAW = False refusal costs, driven rather "
                "than assumed. The hoist is correct arithmetic on this seam; what "
                "makes the flag False is that no wiring performs it while the product "
                "declares INSTALLABLE False"),
            "passed": not findings}


# ---------------------------------------------------------------------------
# Device leg: the byte-neutral edit
# ---------------------------------------------------------------------------

_OWN_LOAD_STORES = (
    "    Hx_out[idx] = own_h[0]; Hy_out[idx] = own_h[1]; Hz_out[idx] = own_h[2];\n")


def _byte_neutral_kernel(variant: str, cond):
    """The shipped body with the own-cell registers replaced by RELOADS of the scratch.

    A float32 stored to global memory and loaded back is the identity on the bits, so
    this edit MUST NOT change an answer -- and it is the one armed edit in this gate
    required not to diverge. It also proves the scratch store lands before the curl
    reads it, which no mutation that MUST diverge can show.
    """
    family = _family()
    source = family.kernel_source(variant, cond)
    if _OWN_LOAD_STORES not in source:
        raise AssertionError("the scratch stores are not where this edit expects them")
    source = source.replace(
        _OWN_LOAD_STORES,
        _OWN_LOAD_STORES
        + "    __threadfence_block();\n"
          "    own_h[0] = Hx_out[idx]; own_h[1] = Hy_out[idx];\n"
          "    own_h[2] = Hz_out[idx];\n", 1)
    return family._get_kernel(variant, cond, source)  # noqa: SLF001


def leg_byte_neutral(steps: int) -> Dict[str, Any]:
    """The one armed edit required NOT to diverge."""
    rows: Dict[str, Any] = {}
    findings: List[str] = []
    for label in ("cond_walls_CCC", "bfast_walls_all"):
        spec = BY_LABEL[label]
        record = drive(spec, "uniform", steps, modes=("array", "weld"),
                       own_load_reads_scratch=True)
        rows[label] = {"variant": spec["variant"],
                       "identical": record["identical"]["weld"],
                       "first_divergence": record["first_divergence"]["weld"],
                       "steps_compared": record["steps_compared"],
                       "words_compared": record["words_compared"]}
        if not record["identical"]["weld"]:
            findings.append(f"{label}: the byte-neutral edit changed an answer")
        log(f"[byte_neutral] {label:28s} "
            f"{'IDENTICAL' if rows[label]['identical'] else 'DIVERGED'}")
    return {"rows": rows, "findings": findings,
            "edit": ("the own-cell registers are replaced by RELOADS of the scratch "
                     "words just stored, behind a block fence"),
            "passed": not findings}


# ---------------------------------------------------------------------------
# The mutations
# ---------------------------------------------------------------------------

#: ``tag -> {old, new, expected, variant, why}``. ``expected`` is CAUGHT or NULL and is
#: DECLARED before the run; an entry predicted NULL is recorded WITH its reason rather
#: than dropped, because a battery that quietly dropped its nulls could not be audited.
DEVICE_MUTATIONS: Dict[str, Dict[str, Any]] = {
    "m_foreign_tap_reads_the_stale_value": {
        "expected": "CAUGHT", "variant": None,
        "old": "    if (ia > 0) return resolve_H(comp, idx - stride, weld);\n",
        "new": ("    if (ia > 0) {\n"
                "        if (comp == 0) return weld.Hx[idx - stride];\n"
                "        if (comp == 1) return weld.Hy[idx - stride];\n"
                "        return weld.Hz[idx - stride];\n"
                "    }\n"),
        "why": "the near-neighbour tap reads the PRE-LAUNCH H instead of recomputing "
               "it -- exactly the value an in-place weld would have read at a "
               "neighbour the launch had not yet written. This is the defect the whole "
               "design removes, so it MUST be caught; a null here would mean the "
               "recompute is decorative on this fixture and the purity ledger is the "
               "leg that would have said so first",
    },
    "m_periodic_wrap_reads_the_near_row": {
        "expected": "CAUGHT", "variant": None,
        "old": ("    return (bc == BC_PERIODIC) ? resolve_H(comp, "
                "idx + (na - 1) * stride, weld) : 0.0f;\n"),
        "new": ("    return (bc == BC_PERIODIC) ? resolve_H(comp, idx, weld) : 0.0f;\n"),
        "why": "the periodic wrap reads the thread's own cell instead of the top "
               "plane. The wrap is OBSERVABLE -- the ownership mask does not fire on a "
               "periodic axis -- which is what makes this the armed sibling of the "
               "metallic-ghost null",
    },
    "m_ghost_is_the_clamped_cells_constitutive": {
        "expected": "NULL", "variant": None,
        "old": ("    return (bc == BC_PERIODIC) ? resolve_H(comp, "
                "idx + (na - 1) * stride, weld) : 0.0f;\n"),
        "new": ("    return (bc == BC_PERIODIC) ? resolve_H(comp, "
                "idx + (na - 1) * stride, weld) : resolve_H(comp, idx, weld);\n"),
        "why": "the METALLIC ghost serves the thread's own recomputed value instead of "
               "the exact 0.0f. Predicted NULL and the prediction is measured by "
               "leg_ghost_observability: every tap is masked at the plane its metallic "
               "ghost fires on, so no edit to what that ghost serves can move a byte. "
               "Recorded rather than dropped",
    },
    "m_own_cell_reads_stale": {
        "expected": "CAUGHT", "variant": None,
        "old": "    raw_update_H_cell(idx, weld, own_h, own_w);\n",
        "new": ("    raw_update_H_cell(idx, weld, own_h, own_w);\n"
                "    own_h[0] = weld.Hx[idx]; own_h[1] = weld.Hy[idx];\n"
                "    own_h[2] = weld.Hz[idx];\n"),
        "why": "the curl's own-cell magnetic loads read the PRE-launch H. The array "
               "path's step_D runs AFTER update_H and reads the stepped value, so this "
               "is a whole sub-step of skew",
    },
    "m_scratch_store_dropped": {
        "expected": "CAUGHT", "variant": None,
        "old": _OWN_LOAD_STORES,
        "new": "    // the scratch store, removed\n",
        "why": "the stepped H is never written to scratch, so the rotation hands the "
               "driver an uninitialised volume. Caught on the FIRST step",
    },
    "m_split_field_history_not_stored": {
        "expected": "CAUGHT", "variant": None,
        "old": ("    f_w_Hx_out[idx] = own_w[0]; f_w_Hy_out[idx] = own_w[1];\n"
                "    f_w_Hz_out[idx] = own_w[2];\n"),
        "new": "    // the f_w_H store, removed\n",
        "why": "update_H's split-field history is never written. Its consumer is the "
               "NEXT step's constitutive recurrence, so a comparison that stopped at "
               "one step would miss it",
    },
    "m_constitutive_regrouped": {
        "expected": "CAUGHT", "variant": None,
        "old": "    return a - kms * prev;\n",
        "new": "    return f[idx] + (kps * src - kms * prev);\n",
        "why": "the two accumulations are REASSOCIATED: the array path forms "
               "`f + kps*src` and rounds it, then subtracts `kms*prev` and rounds "
               "again (constitutive_kernels' note 1). This spelling rounds the "
               "difference of the two products first. Same value in exact arithmetic, "
               "a different float32",
    },
    "m_constitutive_inlined": {
        "expected": "NULL", "variant": None,
        "old": "    return a - kms * prev;\n",
        "new": "    return (f[idx] + kps * src) - kms * prev;\n",
        "why": "`a` is SUBSTITUTED BY ITS DEFINITION and nothing else moves -- same "
               "operands, same operators, same associativity, so the expression tree "
               "is identical and `--fmad=false` keeps it from contracting either way. "
               "PREDICTED NULL and MEASURED NULL, and it is the control that makes "
               "m_constitutive_regrouped's catch a statement about the GROUPING rather "
               "than about touching that line at all",
    },
    "m_recompute_indexes_the_threads_own_profile": {
        "expected": "CAUGHT", "variant": None,
        # THE ANCHOR CARRIES THE COMMENT THAT PRECEDES IT, because the index decode
        # itself appears TWICE in the emitted text -- once inside raw_update_H_cell and
        # once as the fused kernel's own preamble -- and an anchor that matched both
        # would be refused by the needle check rather than editing the one it means.
        "old": ("    // the thread's own.\n"
                "\n"
                "    int k = idx % nz;\n"
                "    int j = (idx / nz) % ny;\n"
                "    int i = idx / (ny * nz);\n"),
        "new": ("    // the thread's own.\n"
                "\n"
                "    int k = 0;\n"
                "    int j = 0;\n"
                "    int i = 0;\n"),
        "why": "the recomputed cell's decode is clamped to the origin, so the absorber "
               "profile is indexed at the WRONG cell. Converged, smooth and wrong -- "
               "the failure mode a tolerance comparison cannot see",
    },
    "m_conductive_partition_is_a_tolerance": {
        "expected": "NULL", "variant": "conductive", "premise": "near_one_entries",
        "old": "    bool dsig = (km1 != 1.0f) || (si1 != 1.0f);\n",
        "new": ("    bool dsig = (fabsf(km1 - 1.0f) > 1e-6f) "
                "|| (fabsf(si1 - 1.0f) > 1e-6f);\n"),
        "why": "the four-case partition becomes a tolerance where the array path spells "
               "an EXACT comparison (stepping.py:2055-2058). PREDICTED NULL ON THESE "
               "FIXTURES AND THE PREDICTION IS MEASURED: the ``premise`` field counts "
               "the coefficient entries that are within 1e-6 of 1.0 WITHOUT being "
               "exactly 1.0, which is the only band where the two spellings can "
               "disagree, and a real absorber profile has no such entry -- the "
               "certified conductive family measured the same thing on its own tables "
               "(561/640 entries exactly 1.0, 561 within 1e-7) and drives this defect "
               "with a SYNTHETIC near-one table instead "
               "(gate_cuda_conductive.py, coefficient_tables_swept 'near_one'). "
               "Recorded here with its premise rather than dropped, and a fixture that "
               "GREW a near-one entry would flip the premise and the expectation "
               "together",
    },
    "m_conductive_history_not_stored": {
        "expected": "CAUGHT", "variant": "conductive",
        "old": "    if (dsig) c[idx] = c_new;\n",
        "new": "    // the f_cond store, removed\n",
        "why": "the conductive PML history is never written. Its consumer is the NEXT "
               "step's recurrence, which is why the product leg compares f_cond as an "
               "OUTPUT and drives sixty steps rather than one",
    },
    "m_bfast_sum_becomes_the_difference": {
        "expected": "CAUGHT", "variant": "bfast",
        "old": "        float total = (k1_a * (sf + f1)) - (k2_a * (ss + f2));\n",
        "new": "        float total = (k1_a * (sf - f1)) - (k2_a * (ss + f2));\n",
        "why": "the BFAST term SUMS the same operands the curl DIFFERENCES. Turning "
               "one sum into a difference is the single easiest way to lose the term "
               "silently, and on an invariant axis it takes 2*g to exactly zero",
    },
    "m_bfast_state_not_stored": {
        "expected": "CAUGHT", "variant": "bfast",
        "old": "        fb_Dx[idx] = bprev + advance;\n",
        "new": "        // the IIR store, removed\n",
        "why": "the BFAST IIR state is never advanced. Its homogeneous mode is UNDAMPED "
               "forever, so a kernel that gets the field right and the state wrong is "
               "correct for exactly one launch and wrong from the second",
    },
}

#: Host mutations: defects in the LAUNCHER rather than in the device text.
HOST_MUTATIONS: Tuple[str, ...] = ("rotation_skipped", "scratch_aliased_to_storage",
                                   "swapped_sub_lattice", "boundary_codes_dropped")


def _mutated_kernel(tag: str, variant: str, cond):
    family = _family()
    spec = DEVICE_MUTATIONS[tag]
    source = family.kernel_source(variant, cond)
    if source.count(spec["old"]) != 1:
        raise AssertionError(
            f"{tag}: its anchor appears {source.count(spec['old'])} times in the "
            f"{variant} text, not once; a mutation that did not land would be "
            f"reported UNCAUGHT")
    return family._get_kernel(  # noqa: SLF001
        variant, cond, source.replace(spec["old"], spec["new"], 1))


def _host_mutation(tag: str, spec: Mapping[str, Any], steps: int) -> Dict[str, Any]:
    """One launcher-level defect, driven against the array path."""
    family = _family()
    if tag == "rotation_skipped":
        record = drive(spec, "uniform", min(steps, 12), modes=("array", "weld"),
                       rotate=False)
        return {"diverged": not record["identical"]["weld"],
                "first_divergence": record["first_divergence"]["weld"],
                "why": ("the launcher never rotates, so the driver keeps reading the "
                        "PRE-launch H for the whole run")}
    fields, grid, pml, dtdx = build(spec, "uniform", case_rng(f"host/{tag}/{spec['label']}"))
    state = family.resolve(fields, grid, pml, dtdx=dtdx)
    if tag == "scratch_aliased_to_storage":
        state["scratch"]["Hx"] = fields.Hx
        try:
            family.assert_bindings_are_disjoint(fields, state)
        except ValueError as error:
            return {"diverged": True, "refused_by_name": str(error)[:400],
                    "why": ("the scratch is bound to the live storage, which would "
                            "make the launch the IN-PLACE weld the board refused. The "
                            "launcher refuses it BY NAME before any launch")}
        return {"diverged": False,
                "why": "assert_bindings_are_disjoint did not refuse an aliased scratch"}
    if tag == "swapped_sub_lattice":
        from meep_gpu.cuda_kernels import step_curl_kernels  # noqa: PLC0415
        half = step_curl_kernels.real_pml_curl_tables(pml, True)
        wrong = dict(state)
        wrong["tables"] = {"kms": {axis: half[f"kms_{axis}"] for axis in "xyz"},
                           "sinv": {axis: half[f"sinv_{axis}"] for axis in "xyz"},
                           "kps": state["tables"]["kps"]}
        return _one_step_divergence(spec, fields, grid, pml, wrong, tag)
    if tag == "boundary_codes_dropped":
        wrong = dict(state)
        wrong["boundary_codes"] = (0, 0, 0)
        return _one_step_divergence(spec, fields, grid, pml, wrong, tag)
    raise ValueError(tag)


def _one_step_divergence(spec, fields, grid, pml, state, tag: str) -> Dict[str, Any]:
    """Launch the weld with a corrupted binding for ONE step against the array path."""
    family = _family()
    reference, r_grid, r_pml, r_dtdx = build(
        spec, "uniform", case_rng(f"host/{tag}/{spec['label']}"))
    del r_grid, r_dtdx
    for name in STEP_PASSES:
        run_pass(name, reference, r_pml)
    for name in STEP_PASSES:
        if name == "step_D":
            continue
        if name != "update_H":
            run_pass(name, fields, pml)
            continue
        family.launch_conductive_bfast_fused_hd_pair(fields, state)
        state["scratch"] = family.rotate_into_fields(fields, state["scratch"])
    cp.cuda.runtime.deviceSynchronize()
    moved = compare(frozen(reference), frozen(fields))
    del grid
    return {"diverged": bool(moved), "volumes": moved,
            "why": f"{tag} corrupts a binding the launcher resolves"}


def _near_one_entries(specs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """The premise behind the tolerance mutation's declared NULL, MEASURED.

    The exact ``!= 1.0f`` partition and a ``|x - 1| > 1e-6`` tolerance can only
    disagree on a coefficient that is NEAR one without BEING one. This counts those
    entries on every conductive fixture's own tables, so the declared null rests on a
    number rather than on a recollection.
    """
    from meep_gpu.cuda_kernels import step_curl_kernels  # noqa: PLC0415

    rows: Dict[str, Any] = {}
    for spec in specs:
        if spec["variant"] != "conductive":
            continue
        _fields, _grid, pml, _dtdx = build(spec, "uniform",
                                           case_rng(f"premise/{spec['label']}"))
        tables = step_curl_kernels.real_pml_curl_tables(pml, False)
        exact = near = total = 0
        for name, array in tables.items():
            if not name.startswith(("kms_", "sinv_")):
                continue
            host = to_host(array).astype(np.float64)
            total += int(host.size)
            exact += int(np.count_nonzero(host == 1.0))
            near += int(np.count_nonzero((host != 1.0)
                                         & (np.abs(host - 1.0) <= 1e-6)))
        rows[spec["label"]] = {"entries": total, "exactly_one": exact,
                               "near_one_but_not_one": near}
    return {"per_fixture": rows,
            "no_fixture_carries_a_near_one_entry":
                bool(rows) and all(row["near_one_but_not_one"] == 0
                                   for row in rows.values())}


def leg_mutation(specs: Sequence[Mapping[str, Any]], steps: int) -> Dict[str, Any]:
    """Every armed defect, each scored on a fixture where its machinery is LIVE."""
    family = _family()
    rows: Dict[str, Any] = {}
    findings: List[str] = []
    premises = {"near_one_entries": _near_one_entries(specs)}
    for tag, spec in DEVICE_MUTATIONS.items():
        wanted = spec.get("variant")
        cases = [s for s in specs if wanted in (None, s["variant"])]
        if not cases:
            findings.append(f"{tag}: no fixture of its variant is in the scored set")
            continue
        outcomes: Dict[str, Any] = {}
        for case in cases:
            cond = case.get("mask") if case["variant"] == "conductive" else None
            try:
                kernel = _mutated_kernel(tag, case["variant"], cond)
            except AssertionError as error:
                outcomes[case["label"]] = {"armed": False, "error": str(error)[:400]}
                continue
            record = drive(case, "uniform", min(steps, 12),
                           modes=("array", "weld"), kernel=kernel)
            outcomes[case["label"]] = {
                "armed": True,
                "diverged": not record["identical"]["weld"],
                "first_divergence": record["first_divergence"]["weld"],
                "steps_compared": record["steps_compared"]}
        caught = any(o.get("diverged") for o in outcomes.values())
        armed = all(o.get("armed") for o in outcomes.values())
        verdict = "CAUGHT" if caught else "NULL"
        rows[tag] = {"expected": spec["expected"], "verdict": verdict,
                     "armed_everywhere": armed, "why": spec["why"],
                     "variant": wanted, "outcomes": outcomes,
                     "premise": premises.get(spec.get("premise"))}
        if not armed:
            findings.append(f"{tag}: did not arm on every scored fixture")
        elif verdict != spec["expected"]:
            findings.append(f"{tag}: expected {spec['expected']}, measured {verdict}")
        log(f"[mutation] {tag:48s} {verdict:6s} (expected {spec['expected']})")

    host_rows: Dict[str, Any] = {}
    # A WALLED FIXTURE, and that is not a preference. ``boundary_codes_dropped``
    # substitutes ``(0, 0, 0)`` -- three BC_PERIODIC -- and on a FULLY PERIODIC fixture
    # that is the run's own triple, so the mutation is a no-op and the leg reports
    # UNCAUGHT for a reason about the fixture. Measured 2026-09-07 on
    # ``cond_periodic_CCC``.
    host_case = next((s for s in specs
                      if s["variant"] == "conductive"
                      and (s["boundaries"] or ("periodic",))[0] == "metallic"),
                     next(s for s in specs if s["variant"] == "conductive"))
    for tag in HOST_MUTATIONS:
        outcome = _host_mutation(tag, host_case, steps)
        host_rows[tag] = {**outcome, "expected": "CAUGHT",
                          "verdict": "CAUGHT" if outcome["diverged"] else "NULL"}
        if not outcome["diverged"]:
            findings.append(f"{tag}: host mutation was NOT caught")
        log(f"[mutation/host] {tag:42s} "
            f"{host_rows[tag]['verdict']:6s} (expected CAUGHT)")

    caught_count = sum(1 for row in rows.values() if row["verdict"] == "CAUGHT")
    null_count = sum(1 for row in rows.values() if row["verdict"] == "NULL")
    # A DECLARED NULL IS ONLY WORTH ITS PREMISE, so a null whose premise stopped
    # holding fails the leg rather than passing quietly.
    if not premises["near_one_entries"]["no_fixture_carries_a_near_one_entry"]:
        findings.append(
            "a conductive fixture now carries a coefficient NEAR one without BEING "
            "one, so m_conductive_partition_is_a_tolerance is an ARMED mutation that "
            "must be caught rather than a structural null; flip its expectation and "
            "re-run")
    return {
        "device": rows, "host": host_rows, "findings": findings,
        "premises": premises,
        "armed": len(rows) + len(host_rows),
        "caught": caught_count + sum(1 for r in host_rows.values()
                                     if r["verdict"] == "CAUGHT"),
        "declared_nulls": [tag for tag, row in rows.items()
                           if row["expected"] == "NULL"],
        "measured_nulls": null_count,
        "family": family.FAMILY,
        "passed": not findings,
    }


def leg_disarm(steps: int) -> Dict[str, Any]:
    """The identical harness on the SHIPPED bytes, required not to diverge.

    A mutation battery that ran through a harness of its own could catch the harness
    rather than the defect. This runs the same ``drive`` over the same fixtures with
    the shipped kernel handed in EXPLICITLY through the same ``kernel=`` door, so the
    only thing that differs from a mutation case is the bytes.
    """
    family = _family()
    rows: Dict[str, Any] = {}
    findings: List[str] = []
    for label in MUTATION_SPEC_LABELS:
        spec = BY_LABEL[label]
        cond = spec.get("mask") if spec["variant"] == "conductive" else None
        kernel = family._get_kernel(spec["variant"], cond)  # noqa: SLF001
        record = drive(spec, "uniform", min(steps, 12), modes=("array", "weld"),
                       kernel=kernel)
        rows[label] = {"identical": record["identical"]["weld"],
                       "first_divergence": record["first_divergence"]["weld"],
                       "steps_compared": record["steps_compared"]}
        if not record["identical"]["weld"]:
            findings.append(f"{label}: the shipped bytes diverged through the "
                            f"mutation harness")
        log(f"[disarm] {label:28s} "
            f"{'IDENTICAL' if rows[label]['identical'] else 'DIVERGED'}")
    return {"rows": rows, "findings": findings, "passed": not findings}


# ---------------------------------------------------------------------------
# Device leg: the compiler
# ---------------------------------------------------------------------------

#: The cache snapshot, taken from the sibling gate rather than transcribed: it is the
#: number the compiler leg's clause 2 rests on, and a second reader of a directory is a
#: second place for a reading to drift.
_cupy_cache_snapshot = hd_gate._cupy_cache_snapshot  # noqa: SLF001


def leg_compiler(policy: Optional[str], at_install: Mapping[str, Any]) -> Dict[str, Any]:
    """Did THIS process compile the kernels it ran, and did the policy reach the compiler?

    FOUR CLAUSES, each measured after every in-process leg has run, and each is the
    released sibling's own -- the reasoning behind them is written once, in
    ``gate_cuda_fused_hd_pair.leg_compiler``, and is not restated here:

    1. the NVRTC binary observer recorded at least one compile;
    2. the cache directory held ZERO entries when the policy was installed, so every
       binary this process ran, it compiled;
    3. the policy reached the compiler -- under ``keep`` the strip removed
       ``-ftz=true`` on every call it saw and no observed tuple still carried it, under
       ``flush`` every observed tuple did;
    4. informative: THIS family's emitted device strings appear among the observed
       sources.
    """
    report = probe.nvrtc_binary_report()
    outer = probe.subnormal_policy_stamp(_REPO_API)
    stamp = dict(outer.get("stamp") or {})
    stamp.setdefault("policy", outer.get("policy"))
    now = _cupy_cache_snapshot()
    family = _family()
    try:
        sources = family.device_sources()
        digests = {name: hashlib.sha256(text.encode("utf-8")).hexdigest()
                   for name, text in sources.items()}
    except Exception as exc:  # noqa: BLE001 - recorded; clause 4 is informative
        digests = {"unavailable": repr(exc)[:200]}
    observed = int(report.get("nvrtc_calls_observed") or 0)
    counters = {"nvrtc_calls": stamp.get("nvrtc_calls"),
                "ftz_removed": stamp.get("ftz_removed")}
    checks: Dict[str, bool] = {
        "the_compiler_was_exercised": observed > 0,
        "every_binary_this_process_ran_was_compiled_by_it":
            at_install.get("entries") == 0 and at_install.get("error") is None,
    }
    if policy == "keep":
        checks["the_strip_reached_every_compile"] = bool(
            (counters["ftz_removed"] or 0) > 0
            and counters["nvrtc_calls"] == observed
            and not report.get("any_ftz_true_reached_nvrtc"))
    elif policy == "flush":
        checks["ftz_true_reached_every_compile"] = bool(
            report.get("all_ftz_true_reached_nvrtc")
            and (counters["ftz_removed"] or 0) == 0)
    else:
        checks["a_policy_was_requested"] = False
    seen = {entry.get("source_sha256")
            for entry in (report.get("observations") or [])}
    return {
        "policy_requested": policy, "policy_stamp": stamp.get("policy"),
        "counters_after_every_in_process_leg": counters,
        "observed": {key: report.get(key) for key in
                     ("nvrtc_calls_observed", "distinct_binaries", "distinct_sources",
                      "any_ftz_true_reached_nvrtc", "all_ftz_true_reached_nvrtc")},
        "observations": report.get("observations"),
        "cache": {"dir": at_install.get("dir"),
                  "entries_at_install": at_install.get("entries"),
                  "preexisting_at_install": at_install.get("preexisting"),
                  "entries_after_every_in_process_leg": now["entries"]},
        "family_device_string_sha256": digests,
        "family_sources_observed_at_nvrtc": {
            name: (digest in seen) for name, digest in digests.items()},
        "checks": checks,
        "what_this_licenses": (
            "that the binaries this process executed were compiled by this process, "
            "under the installed policy, with the policy's option treatment observed "
            "at the NVRTC seam. It says nothing about the lift children, which run in "
            "their own interpreters against the same directory"),
        "passed": all(checks.values()),
    }


# ---------------------------------------------------------------------------
# The corpus lift
# ---------------------------------------------------------------------------

def _child_progress(message: str) -> None:
    path = os.environ.get("MEEP_GPU_CBHD_GATE_PROGRESS")
    label = os.environ.get("MEEP_GPU_CBHD_GATE_LABEL", "")
    line = f"[{time.strftime('%H:%M:%S')}] {label:<46} {message}"
    print(line, flush=True)
    if path:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()


def _drive_lifted_row(driver: Any, steps: int) -> Dict[str, Any]:
    """One lifted corpus row, driven through the driver's own consult order.

    THE SIBLING'S SHAPE AND ITS ARGUMENT, with this product's plans: one driver,
    captured and restored, three arrangements per complete step from a common state,
    and the weld's claim measured against the CERTIFIED SINGLES because a defect in
    this weld cannot appear in them. A step where both dispatched arrangements agree
    with each other and leave the array path is recorded under
    ``backend_disagreement_steps`` with its subnormal census and is a finding about
    the backend rather than about the weld.
    """
    family = _family()
    weld_plans, state = _weld_plan_for(driver)
    weld_shim = Shim(weld_plans)
    singles_shim: Optional[Shim] = None
    singles_error: Optional[str] = None
    variant, refusal = family.variant_for(driver.fields, driver.grid)
    try:
        if variant is None:
            raise RuntimeError(refusal)
        singles = _certified_single_launchers(
            driver.fields, driver.grid, driver.pml,
            float(driver.grid.dt / driver.grid.dx), variant)
        singles_shim = Shim({"update_H": singles["update_H"],
                             "step_D": singles["step_D"]})
    except Exception as error:  # noqa: BLE001
        singles_error = repr(error)[:400]

    detached = {"dft": list(getattr(driver, "_dft_monitors", ()) or ()),
                "flux": list(getattr(driver, "_flux_monitors", ()) or ())}
    driver._dft_monitors = []  # noqa: SLF001
    driver._flux_monitors = []  # noqa: SLF001

    first: Optional[Dict[str, Any]] = None
    backend_steps: List[Dict[str, Any]] = []
    clean = scored = 0
    per_step_words = 0
    banded_at: Optional[int] = None
    error: Optional[str] = None
    not_replayable: Optional[Dict[str, Any]] = None
    started = time.perf_counter()
    for step in range(steps):
        before = hd_gate._capture(driver)  # noqa: SLF001
        per_step_words = sum(words(value).size for value in before["volumes"].values())
        if step == 0:
            # THE REPLAY CONTROL, AND IT IS THE PRECONDITION OF EVERY COMPARISON BELOW.
            try:
                driver._fast_path = None  # noqa: SLF001
                driver._fast_path_stale = False  # noqa: SLF001
                driver.step()
                cp.cuda.runtime.deviceSynchronize()
                replay_a = hd_gate._capture(driver)  # noqa: SLF001
                hd_gate._restore(driver, before)  # noqa: SLF001
                driver.step()
                cp.cuda.runtime.deviceSynchronize()
                replay_b = hd_gate._capture(driver)  # noqa: SLF001
                hd_gate._restore(driver, before)  # noqa: SLF001
            except Exception as exc:  # noqa: BLE001
                error = repr(exc)[:400]
                break
            moved = compare(replay_a["volumes"], replay_b["volumes"])
            if moved:
                not_replayable = {
                    "volumes": moved,
                    "reason": ("one complete driver step, run twice from ONE captured "
                               "state on the ARRAY PATH, did not reproduce itself, so "
                               "byte identity ACROSS arrangements is undefined here -- "
                               "not false, undefined. The row is refused by name")}
                break
        try:
            driver._fast_path_stale = False  # noqa: SLF001
            driver._fast_path = weld_shim  # noqa: SLF001
            driver.step()
            cp.cuda.runtime.deviceSynchronize()
            weld_state = hd_gate._capture(driver)  # noqa: SLF001
            hd_gate._restore(driver, before)  # noqa: SLF001
            singles_state = None
            if singles_shim is not None:
                driver._fast_path = singles_shim  # noqa: SLF001
                driver.step()
                cp.cuda.runtime.deviceSynchronize()
                singles_state = hd_gate._capture(driver)  # noqa: SLF001
                hd_gate._restore(driver, before)  # noqa: SLF001
            driver._fast_path = None  # noqa: SLF001
            driver.step()
            cp.cuda.runtime.deviceSynchronize()
            array_state = hd_gate._capture(driver)  # noqa: SLF001
        except Exception as exc:  # noqa: BLE001
            error = repr(exc)[:400]
            break
        census = {
            "array_path": sum(hd_gate._subnormal_words(v)  # noqa: SLF001
                              for v in array_state["volumes"].values()),
            "weld": sum(hd_gate._subnormal_words(v)  # noqa: SLF001
                        for v in weld_state["volumes"].values()),
            "certified_singles": (None if singles_state is None else
                                  sum(hd_gate._subnormal_words(v)  # noqa: SLF001
                                      for v in singles_state["volumes"].values())),
        }
        if any(value for value in census.values() if value is not None):
            banded_at = step + 1
            break
        scored = step + 1
        against_singles = ({} if singles_state is None else
                           compare(singles_state["volumes"], weld_state["volumes"]))
        if against_singles:
            first = {"step": step + 1, "against": "the certified singles",
                     "volumes": against_singles, "subnormal_census": census}
            break
        against_array = compare(array_state["volumes"], weld_state["volumes"])
        if against_array:
            if singles_state is None:
                first = {"step": step + 1, "against": "the array path",
                         "volumes": against_array, "subnormal_census": census,
                         "note": ("no certified-singles arrangement was available on "
                                  "this row, so this divergence is unattributed")}
                break
            backend_steps.append({"step": step + 1, "volumes": against_array,
                                  "subnormal_census": census})
            continue
        clean = step + 1

    driver._dft_monitors = detached["dft"]  # noqa: SLF001
    driver._flux_monitors = detached["flux"]  # noqa: SLF001
    return {
        "variant": variant, "variant_refusal": refusal,
        "steps_compared": clean, "steps_scored": scored,
        "first_divergence": first, "step_error": error,
        "not_replayable": not_replayable,
        "certified_singles_error": singles_error,
        "backend_disagreement_steps": backend_steps,
        "monitors_detached": {key: len(value) for key, value in detached.items()},
        "entered_the_denormal_band_at_step": banded_at,
        "words_compared_per_step": per_step_words,
        "words_compared": per_step_words * scored * (2 if singles_shim else 1),
        "launches": state["launches"],
        "seconds": round(time.perf_counter() - started, 2),
        "clean_step_floor": LIFT_CLEAN_STEP_FLOOR,
        "below_the_floor": (scored < LIFT_CLEAN_STEP_FLOOR and first is None
                            and not_replayable is None),
        "passed": bool(first is None and error is None and not_replayable is None
                       and scored >= LIFT_CLEAN_STEP_FLOOR),
    }


def evaluate(driver: Any, probe_module: Any) -> Dict[str, Any]:  # noqa: ARG001
    """The census driver's battery hook: the identity on ONE lifted row."""
    from meep_gpu.cuda_kernels import arms  # noqa: PLC0415

    family = _family()
    steps = int(os.environ.get("MEEP_GPU_CBHD_GATE_STEPS", STEPS))
    max_cells = os.environ.get("MEEP_GPU_CBHD_GATE_MAX_CELLS")
    sources = tuple(getattr(driver, "_sources", ()) or ())
    variant, refusal = family.variant_for(driver.fields, driver.grid)
    block: Dict[str, Any] = {
        "family": family.FAMILY, "steps_requested": steps, "n_sources": len(sources),
        "variant": variant, "variant_refusal": refusal,
        "sources": [{"type": type(s).__name__,
                     "field_type": str(getattr(s, "field_type", "")),
                     "is_integrated": bool(getattr(s, "is_integrated", False)),
                     "withdraw_does_work": bool(
                         withdraw_hoist._withdraw_does_work(s))}  # noqa: SLF001
                    for s in sources]}
    covered, reason = family.covers_conductive_bfast_fused_hd_pair(
        driver.fields, driver.pml, driver.grid, sources)
    block["predicate_admits"] = bool(covered)
    block["predicate_reason"] = reason
    try:
        plan = arms.plan_step(fields=driver.fields, pml=driver.pml, grid=driver.grid,
                              sources=sources, fuse=True)
        block["composer_selected"] = {slot: str(arm) for slot, arm
                                      in (getattr(plan, "selected", {}) or {}).items()}
        block["composer_refuses_this_product"] = [
            text for key, texts in (getattr(plan, "reasons", {}) or {}).items()
            if family.FAMILY in key for text in texts]
    except Exception as error:  # noqa: BLE001
        block["composer_error"] = repr(error)[:400]
    cells = int(np.prod(driver.grid.shape))
    block["grid_cells"] = cells
    if not covered:
        block.update({"driven": False,
                      "why_not_driven": "the predicate refused this row"})
        return {"cuda_conductive_bfast_fused_hd_pair_gate": block}
    if max_cells and cells > int(max_cells):
        block.update({"driven": False,
                      "why_not_driven": (f"{cells} cells exceeds "
                                         f"MEEP_GPU_CBHD_GATE_MAX_CELLS={max_cells}; "
                                         f"refused rather than run partially")})
        return {"cuda_conductive_bfast_fused_hd_pair_gate": block}
    block["driven"] = True
    block.update(_drive_lifted_row(driver, steps))
    _child_progress(f"passed={block.get('passed')} words={block.get('words_compared')}")
    return {"cuda_conductive_bfast_fused_hd_pair_gate": block}


def _lift_child(leg: str, target: str, cases: Sequence[str], out_json: str,
                steps: int, progress: str) -> int:
    """Lift ONE corpus row onto the device and drive it. Writes ``out_json`` always."""
    import meep_gpu  # noqa: PLC0415

    def mark(message: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {Path(target).name:<38} {message}"
        print(line, flush=True)
        if progress:
            with open(progress, "a", encoding="utf-8") as handle:
                handle.write(line + "\n")
                handle.flush()

    records: List[Dict[str, Any]] = []

    def finish() -> int:
        for record in records:
            gate_provenance.stamp(record)
        Path(out_json).write_text(json.dumps(records, default=str), encoding="utf-8")
        return 0

    if leg == "examples":
        from parity.meep_gpu.sweep_corpus_lift_parity import (  # noqa: PLC0415
            capture_simulation)

        record, sim, restore = capture_simulation(target)
        record["row"] = Path(target).name
        records.append(record)
        if sim is None:
            record.update({"measured": False, "note": "no mp.Simulation to lift"})
            return finish()
        restore()
        pairs = [(record, sim)]
    else:
        from parity.meep_gpu import survey_meep_tests as harness  # noqa: PLC0415

        namespace = harness.build_child_namespace()
        module = namespace["_import_module"](target)
        wanted = set(cases)
        pairs = []
        for class_name, method_name in namespace["_enumerate_cases"](module):
            case_id = f"{class_name}.{method_name}"
            record, sim, restore, _case = namespace["_run_case"](
                module, target, class_name, method_name, 900.0)
            restore()
            if case_id not in wanted:
                continue
            record["row"] = case_id
            records.append(record)
            if sim is None:
                record.update({"measured": False,
                               "note": "no mp.Simulation to lift on this replay"})
                continue
            pairs.append((record, sim))

    for record, sim in pairs:
        started = time.time()
        try:
            driver = meep_gpu.lift_simulation(sim, prefer_gpu=True, gpu_id=0)
        except BaseException as exc:  # noqa: BLE001
            record.update({"measured": False,
                           "lift_error": f"{type(exc).__name__}: {exc}"[:600]})
            mark(f"LIFT FAILED {type(exc).__name__}")
            continue
        record["lift_s"] = round(time.time() - started, 2)
        record["grid_shape"] = [int(v) for v in driver.shape]
        record["grid_cells"] = int(np.prod([int(v) for v in driver.shape]))
        try:
            record.update(evaluate(driver, None))
            record["measured"] = True
        except BaseException as exc:  # noqa: BLE001
            record.update({"measured": False,
                           "battery_error": f"{type(exc).__name__}: {exc}"[:600]})
        finally:
            with contextlib.suppress(BaseException):
                driver.close()
        block = record.get("cuda_conductive_bfast_fused_hd_pair_gate") or {}
        mark(f"measured={record.get('measured')} "
             f"admits={block.get('predicate_admits')} "
             f"passed={block.get('passed')} steps={block.get('steps_compared')}")
    return finish()


def _load_jsonl(path: Path) -> List[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def lift_basis(results: Path) -> Tuple[List[dict], Dict[str, Any]]:
    """The rows the standing seam record puts in this product's TWO cells.

    DERIVED from the seam record's own per-backend arm selection -- the arms the census
    chose for each row -- never from a list here. The join to a tests row's MODULE and
    to the shim's own case name is the sibling gate's, reused rather than re-derived.
    """
    seam = results / SEAM_RECORD
    if not seam.is_dir():
        raise SystemExit(f"the lift leg needs {seam}")
    census = results / CENSUS
    modules: Dict[str, str] = {}
    replay_case: Dict[str, str] = {}
    for leg in ("tests", "tests_param_matched"):
        path = census / f"{leg}.jsonl"
        if not path.is_file():
            continue
        for record in _load_jsonl(path):
            case = record.get("case") or record.get("row")
            label = f"tests:{record.get('row') or case}"
            if case and record.get("module"):
                modules[label] = record["module"]
            if case and record.get("row") and case != record["row"]:
                replay_case[label] = case
    wanted = {tuple(arms): variant for variant, arms in CELL_ARMS.items()}
    rows: List[dict] = []
    for record in _load_jsonl(seam / "h_to_d_seam.jsonl"):
        cuda = (record.get("arms") or {}).get("cuda") or {}
        cell = (cuda.get("update_H"), cuda.get("step_D"))
        if cell not in wanted:
            continue
        rows.append({
            "label": record["label"], "leg": record["leg"], "row": record["row"],
            "module": record.get("module") or modules.get(record["label"]),
            "replay_case": replay_case.get(record["label"], record["row"]),
            "interpreter": sys.executable,
            "cell": list(cell), "variant": wanted[cell],
            "update_H_kernel": cuda.get("update_H_kernel"),
            "step_D_kernel": cuda.get("step_D_kernel"),
            "withdraw_in_seam": bool((record.get("h_to_d_seam") or {})
                                     .get("withdraw_in_seam")),
            "would_be_bucket": cuda.get("would_be_bucket"),
        })
    facts = {
        "seam_record": SEAM_RECORD, "census": CENSUS,
        "cells": {variant: list(arms) for variant, arms in CELL_ARMS.items()},
        "rows_per_cell": {variant: sum(1 for row in rows if row["variant"] == variant)
                          for variant in CELL_ARMS},
        "rows_total": len(rows),
        # THE PAIRING, READ OFF THE CERTIFIED ARMS RATHER THAN OFF THE CELL LABEL. The
        # conductive cell's two halves live in DIFFERENT modules and the record says so
        # per row; this is the fact the whole product rests on and it is carried into
        # the artifact rather than argued in a docstring.
        "entry_points_per_row": {row["label"]: [row["update_H_kernel"],
                                                row["step_D_kernel"]] for row in rows},
        "rows_with_a_standing_withdraw": sorted(row["label"] for row in rows
                                                if row["withdraw_in_seam"]),
        "module_join": {"source": f"{CENSUS}/tests*.jsonl",
                        "cases_resolved": len(modules),
                        "tests_rows_without_a_module": sorted(
                            row["label"] for row in rows
                            if row["leg"] != "examples" and not row["module"])},
    }
    return rows, facts


def leg_lift(out_dir: Path, steps: int, max_cells: Optional[int], timeout: float,
             resume: bool, only: Optional[Sequence[str]]) -> Dict[str, Any]:
    """Every corpus row in the two cells, re-lifted in its own interpreter and driven."""
    import measure_predicate_coverage as census  # noqa: PLC0415

    out_dir = Path(out_dir).resolve()
    rows, facts = lift_basis(Path(_HERE) / "results")
    if only:
        wanted = set(only)
        rows = [row for row in rows if row["label"] in wanted]
    lift_dir = out_dir / "lift"
    per_row = lift_dir / "per_row"
    per_row.mkdir(parents=True, exist_ok=True)
    progress_log = lift_dir / "steps.progress.log"
    examples_dir = Path(census.EXAMPLES_DIR)
    tests_dir = Path(census.TESTS_DIR)
    if not examples_dir.is_dir() or not tests_dir.is_dir():
        return {"passed": False, "facts": facts,
                "reason": (f"the MEEP corpus is not at {examples_dir} / {tests_dir}; "
                           f"set MEEP_GPU_CORPUS_ROOT to the checkout the census was "
                           f"cut over. Refused rather than measured on nothing")}
    environment = dict(os.environ)
    environment.update({"KMP_DUPLICATE_LIB_OK": "TRUE", "MPLBACKEND": "Agg",
                        "PYTHONPATH": _REPO_API,
                        "MEEP_GPU_CBHD_GATE_STEPS": str(steps),
                        "MEEP_GPU_CBHD_GATE_PROGRESS": str(progress_log)})
    if max_cells is not None:
        environment["MEEP_GPU_CBHD_GATE_MAX_CELLS"] = str(max_cells)
    work = Path(tempfile.mkdtemp(prefix="cuda_cbhd_lift_"))
    for entry in examples_dir.iterdir():
        if entry.suffix in (".py", ".ipynb"):
            continue
        link = work / entry.name
        if not link.exists():
            with contextlib.suppress(OSError):
                link.symlink_to(entry)
    shim_path = Path(census.__file__).resolve().parent / "shim"
    needs_shim = set()
    for row in rows:
        module_name = row.get("module")
        if row["leg"] == "examples" or not module_name:
            continue
        module_file = tests_dir / module_name
        if module_file.is_file() and "import parameterized" in module_file.read_text(
                encoding="utf-8", errors="replace"):
            needs_shim.add(module_name)

    measured: List[dict] = []
    for index, row in enumerate(rows, start=1):
        record_path = per_row / (row["label"].replace(":", "__").replace(".", "_")
                                 + ".json")
        environment["MEEP_GPU_CBHD_GATE_LABEL"] = row["label"]
        environment["PYTHONPATH"] = (f"{shim_path}{os.pathsep}{_REPO_API}"
                                     if row.get("module") in needs_shim
                                     else _REPO_API)
        started = time.time()
        if not (resume and record_path.exists()):
            if row["leg"] == "examples":
                command = [sys.executable, "-u", str(Path(__file__).resolve()),
                           "--lift-child", "examples",
                           "--lift-child-target", str(examples_dir / row["row"]),
                           "--lift-child-out", str(record_path),
                           "--lift-child-progress", str(progress_log),
                           "--lift-steps", str(steps)]
            elif not row["module"]:
                log(f"lift {index}/{len(rows)} {row['label']}: REFUSED, no module "
                    f"recorded for this tests row")
                measured.append({**row, "measured": False,
                                 "note": "no module recorded"})
                continue
            else:
                command = [sys.executable, "-u", str(Path(__file__).resolve()),
                           "--lift-child", "tests",
                           "--lift-child-target", str(tests_dir / row["module"]),
                           "--lift-child-cases",
                           json.dumps([row.get("replay_case") or row["row"]]),
                           "--lift-child-out", str(record_path),
                           "--lift-child-progress", str(progress_log),
                           "--lift-steps", str(steps)]
            log(f"lift {index}/{len(rows)} {row['label']} start")
            stderr_text, note = "", "child died"
            try:
                completed = subprocess.run(command, cwd=str(work), env=environment,
                                           timeout=timeout, stdout=subprocess.DEVNULL,
                                           stderr=subprocess.PIPE, check=False)
                stderr_text = (completed.stderr or b"").decode("utf-8", "replace")
            except subprocess.TimeoutExpired as expired:
                note = "child timeout"
                stderr_text = ((expired.stderr or b"").decode("utf-8", "replace")
                               if expired.stderr else "")
            if not record_path.exists():
                died = {"row": row["row"], "measured": False, "note": note,
                        "stderr_tail": stderr_text[-1500:]}
                gate_provenance.stamp(died)
                record_path.write_text(json.dumps(died, default=str),
                                       encoding="utf-8")
        payload = json.loads(record_path.read_text(encoding="utf-8"))
        candidates = payload if isinstance(payload, list) else [payload]
        wanted_case = row.get("replay_case") or row["row"]
        record = next((r for r in candidates
                       if r.get("row") in (row["row"], wanted_case)),
                      candidates[0] if candidates else {})
        record.update({key: value for key, value in row.items() if key != "label"})
        record["label"] = row["label"]
        measured.append(record)
        block = record.get("cuda_conductive_bfast_fused_hd_pair_gate") or {}
        log(f"lift {index}/{len(rows)} {row['label']}: "
            f"measured={record.get('measured')} "
            f"variant={block.get('variant')} "
            f"admits={block.get('predicate_admits')} driven={block.get('driven')} "
            f"passed={block.get('passed')} steps={block.get('steps_compared')} "
            f"words={block.get('words_compared')} ({time.time() - started:.1f} s)")

    (lift_dir / "rows.json").write_text(
        json.dumps(measured, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8")

    blocks = {r["label"]: (r.get("cuda_conductive_bfast_fused_hd_pair_gate") or {})
              for r in measured}
    admitted = sorted(label for label, b in blocks.items() if b.get("predicate_admits"))
    refused = sorted(label for label, b in blocks.items()
                     if b.get("predicate_admits") is False)
    refused_by_the_withdraw = sorted(
        label for label in refused
        if "standing integrated" in str(blocks[label].get("predicate_reason", "")))
    _ABSENT_DEPENDENCY = ("ModuleNotFoundError", "ImportError",
                          "must be configured/compiled with")

    def _unlifted_reason(record: Mapping[str, Any]) -> Optional[str]:
        if record.get("has_simulation"):
            return None
        for key in ("error", "lift_error", "note", "battery_error"):
            text = str(record.get(key) or "")
            if any(marker in text for marker in _ABSENT_DEPENDENCY):
                return text[:300]
        return None

    by_label = {r["label"]: r for r in measured}
    unlifted = {label: reason for label, record in by_label.items()
                if "predicate_admits" not in (blocks.get(label) or {})
                and (reason := _unlifted_reason(record))}
    unmeasured = sorted(label for label, b in blocks.items()
                        if "predicate_admits" not in b and label not in unlifted)
    driven = sorted(label for label in admitted if blocks[label].get("driven"))
    diverged = sorted(label for label in driven if blocks[label].get("first_divergence"))
    backend_rows = {label: len(blocks[label].get("backend_disagreement_steps") or ())
                    for label in driven
                    if blocks[label].get("backend_disagreement_steps")}
    below_floor = sorted(label for label in driven
                         if blocks[label].get("below_the_floor"))
    errored = sorted(label for label in driven if blocks[label].get("step_error"))
    unreplayable = {label: blocks[label]["not_replayable"]["volumes"]
                    for label in driven if blocks[label].get("not_replayable")}
    passed_rows = sorted(label for label in driven if blocks[label].get("passed"))
    expected_refused = sorted(set(facts["rows_with_a_standing_withdraw"])
                              - set(unlifted))
    if only:
        expected_refused = sorted(set(expected_refused) & set(blocks))
    # THE VARIANT THE PRODUCT PICKED ON EACH ROW MUST BE THE ONE THE SEAM RECORD'S CELL
    # NAMES. A row served by the wrong tail would be byte-identical to nothing and this
    # is where a mis-resolution shows.
    variant_agrees = {label: (blocks[label].get("variant")
                              == by_label[label].get("variant"))
                      for label in driven}
    return {
        "facts": facts, "rows_measured": len(measured),
        "admitted": len(admitted), "refused": len(refused),
        "refused_by_the_withdraw_clause": refused_by_the_withdraw,
        "expected_refused": expected_refused,
        "the_refusals_are_exactly_the_withdraw_rows":
            refused_by_the_withdraw == expected_refused and refused == expected_refused,
        "unmeasured": unmeasured,
        "unlifted_on_this_host": unlifted,
        "rows_the_engine_was_asked_about": len(blocks) - len(unlifted),
        "driven": len(driven),
        "passed_rows": len(passed_rows), "diverged": diverged,
        "backend_disagreement_rows": backend_rows,
        "the_variant_matches_the_cell_on_every_driven_row":
            all(variant_agrees.values()) if variant_agrees else False,
        "variant_per_row": {label: blocks[label].get("variant") for label in driven},
        "steps_scored": sum(int(blocks[label].get("steps_scored") or 0)
                            for label in driven),
        "below_the_floor": below_floor, "errored": errored,
        "not_replayable_rows": unreplayable,
        "complete_driver_steps": sum(int(blocks[label].get("steps_compared") or 0)
                                     for label in driven),
        "words_compared": sum(int(blocks[label].get("words_compared") or 0)
                              for label in driven),
        "passed": bool(measured and not diverged and not errored and not unmeasured
                       and refused == expected_refused
                       and (all(variant_agrees.values()) if variant_agrees else False)),
    }


# ---------------------------------------------------------------------------
# The runner
# ---------------------------------------------------------------------------

def save(results: Dict[str, Any], out: str) -> None:
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    gate_provenance.stamp(results)
    path.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str),
                    encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=None, help="artifact FILE path")
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--product", choices=("full", "reduced"), default="full")
    parser.add_argument("--subnormal-policy", choices=("keep", "flush"), default=None)
    parser.add_argument("--import-meep-for-host-policy", action="store_true")
    parser.add_argument("--legs", default=None,
                        help="comma-separated subset of " + ",".join(ALL_LEGS))
    parser.add_argument("--lift-steps", type=int, default=STEPS)
    parser.add_argument("--lift-max-cells", type=int, default=None)
    parser.add_argument("--lift-timeout", type=float, default=1800.0)
    parser.add_argument("--lift-only", default=None, help="comma-separated row labels")
    parser.add_argument("--lift-resume", action="store_true")
    parser.add_argument("--no-device", action="store_true")
    parser.add_argument("--lift-child", choices=("examples", "tests"), default=None,
                        help="internal: lift and drive ONE corpus row")
    parser.add_argument("--lift-child-target", default=None)
    parser.add_argument("--lift-child-cases", default="[]")
    parser.add_argument("--lift-child-out", default=None)
    parser.add_argument("--lift-child-progress", default="")
    args = parser.parse_args(argv)

    if args.lift_child:
        return _lift_child(args.lift_child, args.lift_child_target,
                           json.loads(args.lift_child_cases), args.lift_child_out,
                           args.lift_steps, args.lift_child_progress)

    if not args.out:
        parser.error("--out is required unless --lift-child is given")
    started = time.perf_counter()
    legs = tuple(args.legs.split(",")) if args.legs else ALL_LEGS
    results: Dict[str, Any] = {
        "gate": Path(__file__).name,
        "family": "cuda_conductive_bfast_fused_hd_pair",
        "seam": withdraw_hoist.SEAM,
        "cells": {variant: list(arms) for variant, arms in CELL_ARMS.items()},
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "argv": list(argv if argv is not None else sys.argv[1:]),
        "legs_requested": list(legs),
        "steps": args.steps,
        "what_a_release_does_not_license": None,
    }
    if cp is None or args.no_device:
        results["device_mode"] = False
        results["status"] = "refused: no CuPy" if cp is None else "structural only"
        # THE HOST LEGS THAT NEED NO ENGINE still run, so a laptop can check the
        # transcription without claiming anything about a device.
        for name, thunk in (("driver_order", leg_driver_order),
                            ("transcription", leg_transcription),
                            ("ghost_observability", leg_ghost_observability)):
            if name in legs:
                try:
                    results[name] = thunk()
                except Exception as error:  # noqa: BLE001
                    results[name] = {"passed": False, "error": repr(error)[:2000]}
                log(f"[{name}] passed={results[name].get('passed')}")
        results["canonical_verdict"] = {
            "released": False,
            "reasons": ["no device: a release needs the device legs"]}
        save(results, args.out)
        log(results["status"])
        return 1

    results["device_mode"] = True
    if args.import_meep_for_host_policy:
        results["meep_host_import"] = probe.import_meep_for_host_policy()
    results["nvrtc_observer"] = probe.install_nvrtc_binary_observer()
    if args.subnormal_policy:
        results["subnormal_policy_install"] = probe.install_subnormal_policy_for_run(
            args.subnormal_policy, _REPO_API)
    results["cupy_cache"] = _cupy_cache_snapshot()
    results["host"] = platform.node()
    results["environment"] = probe.device_info()
    results["subnormal_policy_stamp"] = probe.subnormal_policy_stamp(_REPO_API)

    family = _family()
    results["what_a_release_does_not_license"] = family.WHAT_A_RELEASE_DOES_NOT_LICENSE
    results["source_sha256"] = family.source_digest()
    save(results, args.out)

    specs = list(SPECS) if args.product == "full" else [
        spec for spec in SPECS if spec["label"] in REDUCED_LABELS]
    mutation_specs = [BY_LABEL[label] for label in MUTATION_SPEC_LABELS
                      if args.product == "full" or label in REDUCED_LABELS]
    if not any(s["variant"] == "bfast" for s in mutation_specs):
        mutation_specs.append(BY_LABEL["bfast_periodic_3d"])

    def run_leg(name: str, thunk: Callable[[], Dict[str, Any]]) -> None:
        if name not in legs:
            return
        started_leg = time.perf_counter()
        try:
            results[name] = thunk()
        except Exception as error:  # noqa: BLE001
            results[name] = {"passed": False, "error": repr(error)[:2000]}
        results[name]["seconds"] = round(time.perf_counter() - started_leg, 1)
        log(f"[{name}] passed={results[name].get('passed')} "
            f"({results[name]['seconds']} s)")
        save(results, args.out)

    run_leg("driver_order", leg_driver_order)
    run_leg("transcription", leg_transcription)
    run_leg("refusal", leg_refusal)
    run_leg("arbitration", leg_arbitration)
    run_leg("ghost_observability", leg_ghost_observability)

    if "product" in legs:
        cases: List[Dict[str, Any]] = []
        for spec in specs:
            for value_class in VALUE_CLASSES:
                record = drive(spec, value_class, args.steps,
                               progress=lambda m: log(f"  {m}"))
                cases.append(record)
                log(f"[product] {record['label']:38s} "
                    f"{'IDENTICAL' if record['passed'] else 'DIVERGED'} "
                    f"steps={record['steps_compared']} "
                    f"words={record['words_compared']} "
                    f"subnormals={record['operand_census']['subnormals']}")
                results["product"] = {"cases": cases}
                save(results, args.out)
        band = [r for r in cases if r["value_class"] == "subnormal_band"]
        uniform = [r for r in cases if r["value_class"] == "uniform"]
        per_variant = {variant: [r for r in cases if r["variant"] == variant]
                       for variant in ("conductive", "bfast")}
        results["product"] = {
            "cases": cases, "denominator": len(cases),
            "complete_driver_steps": sum(r["steps_compared"] for r in cases),
            "words_compared": sum(r["words_compared"] for r in cases),
            "cases_per_variant": {v: len(rows) for v, rows in per_variant.items()},
            "masks_swept": sorted({"".join("C" if f else "-" for f in r["mask"])
                                   for r in cases if r["mask"]}),
            "the_band_class_really_contains_subnormals":
                bool(band) and all(r["operand_census"]["subnormals"] > 0 for r in band),
            "the_uniform_class_contains_none":
                bool(uniform) and all(r["operand_census"]["subnormals"] == 0
                                      for r in uniform),
            # BOTH CELLS ARE A FLOOR, NOT A BONUS: a product leg that scored no case of
            # one variant cannot release, so neither cell's credit can rest on the
            # other's measurement.
            "both_variants_were_driven": all(bool(rows)
                                             for rows in per_variant.values()),
            "the_variants_own_tail_was_live_on_every_case":
                all(r["the_variants_own_tail_is_live"] for r in cases),
            "passed": (bool(cases) and all(r["passed"] for r in cases)
                       and all(bool(rows) for rows in per_variant.values())),
        }
        save(results, args.out)

    run_leg("purity_and_race", lambda: leg_purity_and_race(specs, args.steps))
    run_leg("seed_scale", lambda: leg_seed_scale(args.steps))
    run_leg("launch_structure", lambda: leg_launch_structure(args.steps))
    run_leg("sync", lambda: leg_sync(min(args.steps, 12)))
    run_leg("withdraw", lambda: leg_withdraw(min(args.steps, 12)))
    run_leg("byte_neutral", lambda: leg_byte_neutral(args.steps))
    run_leg("mutation", lambda: leg_mutation(mutation_specs, args.steps))
    run_leg("disarm", lambda: leg_disarm(args.steps))
    run_leg("compiler", lambda: leg_compiler(args.subnormal_policy,
                                             results.get("cupy_cache") or {}))
    run_leg("lift", lambda: leg_lift(
        Path(args.out).parent, args.lift_steps, args.lift_max_cells,
        args.lift_timeout, args.lift_resume,
        args.lift_only.split(",") if args.lift_only else None))

    clauses = {name: bool(results.get(name, {}).get("passed"))
               for name in legs if name in results}
    results["verdict"] = {
        "clauses": clauses, "passed": bool(clauses) and all(clauses.values()),
        "legs_run": sorted(clauses), "legs_requested": list(legs),
        "legs_missing": sorted(set(legs) - set(clauses)),
    }
    results["canonical_verdict"] = {
        "released": bool(results["verdict"]["passed"]
                         and not results["verdict"]["legs_missing"]),
        "reasons": [name for name, value in clauses.items() if not value]
                   + [f"leg did not run: {name}"
                      for name in results["verdict"]["legs_missing"]],
        "what_it_licenses": (
            "ONE CUDA product in TWO variants, measured byte-identical per COMPLETE "
            "DRIVER STEP and as uint32 words over every stored volume -- including the "
            "variant's own third history -- to four independent arrangements of the "
            "update_H -> step_D seam, at every block size, on the fixtures and corpus "
            "rows this record names. It licenses NO throughput claim, NO dispatch "
            "claim and NO composition claim: the family declares INSTALLABLE = False "
            "and the composer refuses to install it on every configuration, so a "
            "credited seam-instance is PREDICATE ADMISSION and the product executes "
            "nowhere"),
    }
    results["elapsed_s"] = round(time.perf_counter() - started, 1)
    save(results, args.out)
    log("=" * 78)
    for clause, value in clauses.items():
        log(f"  {'PASS' if value else 'FAIL'}  {clause}")
    log(f"RELEASED: {results['canonical_verdict']['released']}  "
        f"({results['elapsed_s']} s)")
    return 0 if results["canonical_verdict"]["released"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
