#!/usr/bin/env python3
"""Device byte gate for the CUDA H->D weld: ``update_H`` welded into ``step_D``.

THE FOURTH SEAM'S FIRST DEVICE GATE ON THIS BACKEND, and the first hand-CUDA fused
gate whose subject is a product the composer must NOT install. That shapes every leg:
the arithmetic is certified the way the sibling gates certify theirs, and the
arbitration -- who holds ``update_H`` -- is MEASURED here as a fact about what the
shipped composer builds, never asserted from a docstring.

WHAT IS BEING CERTIFIED. ``meep_gpu/cuda_kernels/fused_hd_pair.py`` computes the
``update_H`` constitutive into launch-local SCRATCH, takes its own cell's magnetic
field from registers, RECOMPUTES every one of the curl half's six foreign taps from
pre-launch state, steps ``D``/``fu_D`` in place, and rotates the ``H``/``f_w_H``
bindings afterwards. The claim is per COMPLETE DRIVER STEP -- the driver's own pass
order -- over a stated budget of steps, as uint32 WORDS over EVERY stored volume the
engine allocates (primaries, split-field PML auxiliaries, both ``f_w`` histories;
never ``allclose``, because ``-0.0 == 0.0`` lies), against FOUR reference
arrangements from one seed:

  1. the ARRAY PATH -- ``stepping``'s passes in the driver's order;
  2. the CERTIFIED SINGLES -- ``update_H`` alone then ``step_D`` alone, as two
     dispatched launches, everything else on the array path;
  3. the COMPOSITION THE COMPOSER INSTALLS ON THESE ROWS TODAY -- the released B->H
     fused magnetic pair holding ``step_B``..``update_H`` and the released D->E fused
     electric pair holding ``step_D``..``update_E``. This is the reference the whole
     arbitration finding rests on;
  4. the same slots DISPATCHED UNFUSED -- one certified single per slot, with the
     in-seam passes on the array path.

All four must agree with the weld word for word. What differs is the LAUNCH COUNT,
which :func:`leg_launch_structure` reports twice over (the launchers' own reports and
an independent wrapper around the shipped compile memo) at the seam AND over the whole
step -- where the honest number is that the weld does NOT reduce the step's launches
against the composition installed today. That is the arbitration ruling reproduced as
a measurement rather than re-litigated.

LEGS
  host legs (no device launch)
  driver_order       REPLACES is exactly the driver's two adjacent consults, the only
                     statement between them is the electric withdraw loop, and the
                     sync channel's containment rule excludes step_D -- all read off
                     the tree with ``ast``, never spelled
  transcription      both halves are the certified emitters' own bytes: the curl tail
                     differs in EXACTLY the twelve redirected magnetic reads, the
                     constitutive prelude in EXACTLY the declared lift edits, every
                     declared edit's anchor resolves, every mutation needle resolves
                     exactly once, the text is pure ASCII (the NVRTC locale trap)
  refusal            by name: an undeclared source list; a standing INTEGRATED
                     electric withdraw (the reason 5 corpus rows are refused); a
                     non-integrated electric source NOT refused; a magnetic source NOT
                     refused; complex storage; an inactive absorber naming the
                     constitutive half; a Dcyl grid; a conductive step_D target; a
                     BFAST run; a nonlinear run ADMITTED through the spine arm
  arbitration        the composer, asked: on a row both incumbents reach, the released
                     B->H pair holds step_B AND update_H, this product is refused BY
                     NAME, and the H_to_D seam row routes to the withdraw hoist
  device legs
  product            the four-reference identity over complete driver steps on every
                     fixture, with the movement floor and the subnormal census. Two
                     fixtures carry an instantaneous chi2/chi3 (2026-09-06): on them
                     the composition that ships is the released B->H pair plus the
                     PML curl single and the nonlinear update_E arm -- the census's
                     own selection on the cell's two corpus rows -- and each case
                     also records that the chi MOVED the array path in one step,
                     that the predicate admitted the fixture through the nonlinear
                     spine arm while the ordinary one refused it, and that the
                     (nonlinear, PML) extra-arm row rather than the primary row is
                     what the composer selected. A product leg with no nonlinear
                     case does not pass
  purity_ledger      per fixture, on the array path: of the curl's valid foreign taps,
                     how many land on a cell ``update_H`` MOVED -- the count of taps
                     whose recomputed value differs from what an in-place weld would
                     have raced on. Near zero would mean the identity licenses nothing
  block_sizes        the same identity at six block sizes. The design's whole claim is
                     that the answer does not depend on which block ran first, and a
                     block size that never moves cannot expose the opposite
  launch_structure   launches per step at the seam and over the whole step, for the
                     weld and all four references, two independent counters
  sync               the product FORCE-installed; a flux accessor called mid-run so
                     ``synchronize_magnetic_fields`` consults ``update_H_synchronize``.
                     An arm whose plan answers that consult MUST diverge in D/fu_D --
                     that divergence IS the citation for the composer refusal -- and
                     the arm that declines by the containment rule MUST NOT
  withdraw           an integrated electric source in the seam (the configuration the
                     predicate refuses today): the hoisted arrangement identical, the
                     un-hoisted launch and the ``after_step_D`` placement both diverge
                     -- the device counterpart of ``results/h_to_d_withdraw_order``
  byte_neutral       the own-cell register reads replaced by reloads of the scratch
                     just stored: the one armed edit required NOT to diverge
  mutation           the armed device and host defects, each of which MUST diverge;
                     entries predicted null are recorded WITH the reason, never dropped
  lift               every corpus row the standing census puts in the cell, re-lifted
                     ON THE DEVICE in its own child and driven as THREE arrangements
                     from one captured state per complete step -- the weld, the
                     certified singles and the array path. The weld's claim is against
                     the CERTIFIED SINGLES, because a defect in this weld cannot appear
                     in them; a step where both dispatched arrangements agree with each
                     other and leave the array path is recorded under
                     backend_disagreement_steps with its subnormal census and is a
                     finding about the backend rather than about the weld. A row this
                     HOST cannot lift -- the corpus's lift records were cut across
                     three interpreters and the validation host has one -- is named
                     under unlifted_on_this_host with its import error and counted out
                     of the denominator; anything unmeasured for another reason is a
                     hole and fails the leg. Every row is gated by a REPLAY CONTROL
                     first -- one complete step run twice from one captured state on
                     the array path, required to reproduce itself -- because a row
                     whose source draws fresh randomness per evaluation gives each
                     arrangement a different current and byte identity ACROSS
                     arrangements is undefined there, not false
  disarm             the identical harness on the shipped bytes, required not to
                     diverge, so a mutation leg's catches are not the harness's
  compiler           read AFTER every in-process device leg: the NVRTC observer saw
                     at least one compile, the CuPy cache held ZERO entries when the
                     policy was installed (so every binary this process ran, it
                     compiled -- the disk cache is keyed above the strip seam and a
                     warm directory can serve the other policy's bytes silently),
                     and the policy reached the compiler: under keep the strip
                     removed -ftz=true on every call it saw and no observed option
                     tuple still carried it, under flush every observed tuple did.
                     The 2026-09-06 release legs recorded nvrtc_calls 0 because the
                     counters were stamped at install time, on caches that were
                     already warm; this leg is what turns that into a measurement

Rule 7: one flushed line per case, and the artifact is rewritten after every leg, so
an interrupted run keeps everything that landed.
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
import h_to_d_seam  # noqa: E402
import probe_fused_kernel_bit_identity as probe  # noqa: E402

# THE SIBLING SCRATCH-WELD GATE, imported for the machinery both gates share: the
# launch counter that wraps the shipped memo, the word comparator and the read-only
# snapshot. ONE copy of each -- a second transcription of a launch counter is a second
# place for a counter to stop counting.
import gate_cuda_offdiag_stencil_welds as scratch_gate  # noqa: E402

from meep_gpu import stepping, withdraw_hoist  # noqa: E402
from meep_gpu.fastpath import (SYNC_PASS_OWNERS, SYNC_PATH_SLOTS,  # noqa: E402
                               SYNC_UPDATE_H_PASS)

log = probe.log
to_host = scratch_gate.to_host
words = scratch_gate.words
differing = scratch_gate.differing
MemoLaunchCounter = scratch_gate.MemoLaunchCounter
needle = scratch_gate.needle

# ---------------------------------------------------------------------------
# The battery contract, so the census driver can lift a corpus row into this file
# ---------------------------------------------------------------------------

#: ``measure_predicate_coverage`` digests this package into every lifted row's
#: ``subject_manifest_sha256``. Declared, as every battery declares it.
SUBJECT_PACKAGE = "cuda_kernels"

#: Per-case RNG seed base, from the case LABEL rather than from ``hash()``:
#: ``PYTHONHASHSEED`` salts the hash of a string, so a hash-seeded case could not be
#: replayed from the record that names it.
SEED = 20260906

#: Complete DRIVER STEPS a fixture runs. Sixty -- the budget every hand-CUDA record on
#: this track is cut at, and the budget the withdraw-order campaign this gate is the
#: device counterpart of ran.
STEPS = 60

#: The steps at which the sync leg calls the energy accessor. Two, so the hazard is
#: measured to persist and is not a one-step transient.
SYNC_STEPS: Tuple[int, ...] = (3, 7)

#: The block sizes the schedule sweep runs. 32 is one warp and 1024 the maximum; the
#: shipped launcher's own 256 sits between them, and they put wildly different numbers
#: of blocks on the grid.
BLOCK_SIZES: Tuple[int, ...] = (32, 64, 128, 256, 512, 1024)

#: Every stored volume a complete step can touch. All of them are compared: the
#: electric half is here even though this weld writes only D/fu_D and the magnetic
#: pair, because a weld that corrupted H reaches E through ``step_D`` and ``update_E``
#: on the very same step.
STATE_NAMES: Tuple[str, ...] = scratch_gate.STATE_NAMES

#: The volumes that must be BIT-UNCHANGED by a launch: the PML coefficient vectors on
#: both Yee sub-lattices. Every table argument in this signature is ``__restrict__``,
#: and a kernel that wrote through a ``const`` binding is UB NVRTC does not diagnose.
IMMUTABLE_PML: Tuple[str, ...] = scratch_gate.IMMUTABLE_PML

VALUE_CLASSES: Tuple[str, ...] = ("uniform", "subnormal_band")

#: The volumes a launch must MOVE for a case to be non-vacuous under the band class.
#: Under ``flush`` a state seeded entirely in the band drives ``constitutive_apply`` to
#: ``f[idx] = (f[idx] + 0) - 0``, so H need not move; D, fu_D and f_w_H are written
#: unconditionally and must.
BAND_MUST_MOVE: Tuple[str, ...] = tuple(
    [f"D{axis}" for axis in "xyz"] + [f"fu_D{axis}" for axis in "xyz"]
    + [f"f_w_H{axis}" for axis in "xyz"])

#: The driver's own ten-pass step, in its order. Read from the driver by
#: :func:`leg_driver_order` rather than trusted here.
STEP_PASSES: Tuple[str, ...] = scratch_gate.STEP_PASSES

#: The five arrangements every product case drives in lockstep. ``weld`` is the
#: subject; the other four are the references named in the module docstring.
MODES: Tuple[str, ...] = ("array", "singles", "composition_today", "unfused", "weld")

LEG_GROUPS: Dict[str, Tuple[str, ...]] = {
    "host": ("driver_order", "transcription", "refusal", "arbitration"),
    # ``compiler`` is LAST among the in-process device legs on purpose: it reads the
    # NVRTC observer and the policy's strip counters AFTER every kernel this process
    # will launch has been compiled, so what it records is the whole run's compiler
    # traffic and not the zero every counter reads at install time.
    "device": ("product", "purity_ledger", "block_sizes", "launch_structure", "sync",
               "withdraw", "byte_neutral", "mutation", "disarm", "compiler"),
    "corpus": ("lift",),
}
ALL_LEGS: Tuple[str, ...] = tuple(leg for group in LEG_GROUPS.values() for leg in group)

#: The census this gate lifts its corpus rows from, and the seam record whose
#: ``withdraw_in_seam`` flag names the rows the predicate must refuse.
CENSUS = "cuda_predicate_coverage_2026-09-03_extended"
SEAM_RECORD = "h_to_d_seam_2026-09-04"

#: The board cell this product serves, in the census's own arm spelling:
#: ``(update_H arm, step_D arm)``.
CELL_ARMS: Tuple[str, str] = ("cuda_constitutive/ordinary", "cuda_curl/PML")

#: The SECOND cell, reached through ``FUSED_PAIR_EXTRA_ARMS``. On this backend the
#: widening is a MODULE IDENTITY rather than a text comparison -- ``nonlinear_
#: constitutive`` ships no ``update_H`` kernel at all and admits the shipped
#: ``update_H_pml_real`` unchanged -- and the census agrees from the other side, giving
#: both cells the same ``update_H_kernel``.
EXTRA_CELL_ARMS: Tuple[str, str] = ("cuda_nonlinear/nonlinear", "cuda_curl/PML")

#: HOW MANY CLEAN COMPLETE STEPS A LIFTED CORPUS ROW MUST REACH TO COUNT. The
#: synthetic fixtures' amplitude is this gate's to choose and is placed clear of the
#: denormal band for the whole budget; a lifted row's state is the ROW'S -- it starts
#: at the engine's zeros and is driven by its own sources, so its wavefront's leading
#: cells are arbitrarily small and it enters the band on its own schedule. This is a
#: FLOOR below which a row measured too little to be worth a verdict, not a target:
#: the record carries each row's own step count and the words it compared.
LIFT_CLEAN_STEP_FLOOR = 8

#: The synthetic fixtures. Built so that every axis is walled somewhere, every axis is
#: folded somewhere, BOTH fold terminations appear, both full-count parities appear,
#: and a fully periodic grid appears -- which is the ONLY specialisation on which the
#: shifted tap's PERIODIC WRAP is observable at all, because on a metallic axis
#: ``_mask_non_owned_cells`` zeroes the very curl the wrapped tap feeds.
SPECS: Tuple[Dict[str, Any], ...] = (
    {"label": "periodic", "cell": (1.6, 1.7, 1.5),
     "boundaries": ("periodic", "periodic", "periodic"), "symmetry": ()},
    {"label": "walls_all", "cell": (1.6, 1.7, 1.5),
     "boundaries": ("metallic", "metallic", "metallic"), "symmetry": ()},
    {"label": "wall_x_only", "cell": (1.6, 1.6, 1.6),
     "boundaries": ("metallic", "periodic", "periodic"), "symmetry": ()},
    {"label": "wall_z_only", "cell": (1.6, 1.6, 1.6),
     "boundaries": ("periodic", "periodic", "metallic"), "symmetry": ()},
    {"label": "fold_y_even", "cell": (1.6, 2.0, 1.6),
     "boundaries": None, "symmetry": (("Y", 1),)},
    {"label": "fold_y_odd_phase", "cell": (1.6, 2.0, 1.6),
     "boundaries": None, "symmetry": (("Y", -1),)},
    {"label": "fold_y_odd_count", "cell": (1.6, 2.1, 1.6),
     "boundaries": None, "symmetry": (("Y", 1),)},
    {"label": "fold_y_metallic_termination", "cell": (1.6, 2.0, 1.6),
     "boundaries": ("periodic", "metallic", "periodic"), "symmetry": (("Y", 1),)},
    {"label": "fold_xy_mixed", "cell": (2.0, 2.0, 1.6),
     "boundaries": None, "symmetry": (("X", 1), ("Y", -1))},
    # THE NONLINEAR CELL, DRIVEN. Two seam-instances on the board reach this product
    # through ``FUSED_PAIR_EXTRA_ARMS``'s (``nonlinear``, ``PML``) row rather than the
    # primary one, and until 2026-09-06 no synthetic fixture in this leg carried a
    # chi2/chi3: the credit rested on the module identity (``nonlinear_constitutive``
    # ships no ``update_H`` kernel) and on the two lifted corpus rows the lift leg
    # drives. These two fixtures put the identity under a chi ON THIS LEG as well --
    # a walled axis and a fully periodic grid, UNFOLDED, because the nonlinear spine
    # arm refuses a fold by name (``covers_real_pml_nonlinear_constitutive``, clause
    # 2) and a fixture the predicate refuses would measure nothing about the cell.
    {"label": "wall_z_nonlinear", "cell": (1.6, 1.6, 1.6),
     "boundaries": ("periodic", "periodic", "metallic"), "symmetry": (),
     "nonlinear": True},
    {"label": "periodic_nonlinear", "cell": (1.6, 1.7, 1.5),
     "boundaries": ("periodic", "periodic", "periodic"), "symmetry": (),
     "nonlinear": True},
)

#: The fixtures that carry an instantaneous chi2/chi3, read off the table rather than
#: listed twice.
NONLINEAR_LABELS: Tuple[str, ...] = tuple(spec["label"] for spec in SPECS
                                          if spec.get("nonlinear"))

#: The fixtures the mutations are scored on. Chosen so every armed defect has at least
#: one fixture where the machinery it disables is LIVE: a wall on every axis, a
#: periodic wrap that survives the mask, and both fold terminations.
MUTATION_SPEC_LABELS: Tuple[str, ...] = ("periodic", "walls_all", "fold_y_odd_phase",
                                         "fold_y_metallic_termination")

#: The subset ``--product reduced`` runs. It carries one nonlinear fixture so that a
#: reduced run cannot pass the product leg's nonlinear floor vacuously.
REDUCED_LABELS: Tuple[str, ...] = ("periodic", "walls_all", "fold_y_odd_phase",
                                   "fold_xy_mixed", "wall_z_nonlinear")


def case_rng(label: str) -> "np.random.Generator":
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    return np.random.default_rng(SEED + int.from_bytes(digest[:4], "big"))


# ---------------------------------------------------------------------------
# The fixture
# ---------------------------------------------------------------------------

def build(spec: Mapping[str, Any], value_class: str, rng):
    """A frozen ``(fields, grid, pml, dtdx)`` on the DEVICE, seeded for this class."""
    from meep_gpu.fields import Fields  # noqa: PLC0415
    from meep_gpu.grid import Grid, Mirror  # noqa: PLC0415
    from meep_gpu.pml import PML  # noqa: PLC0415

    grid = Grid(resolution=10.0, cell_size=spec["cell"], courant=0.35, xp=cp,
                boundaries=spec["boundaries"],
                symmetry=tuple(Mirror(axis, phase)
                               for axis, phase in spec["symmetry"]))
    thickness = tuple((0, 0) if grid.shape[axis] < 6
                      else (0, 2) if grid.is_mirrored(axis)
                      else (2, 2) for axis in range(3))
    pml = PML(grid=grid, thickness=thickness)
    fields = Fields(grid=grid)
    fields.enable_field_storage()
    fields.enable_pml_storage()
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
    if spec.get("nonlinear"):
        # AN INSTANTANEOUS chi2/chi3 ON EVERY E COMPONENT, drawn as VOLUMES so the
        # array path's per-cell chi lookup is exercised rather than a broadcast
        # scalar (``gate_cuda_nonlinear.py``'s own fixture rule), and drawn AFTER the
        # state so every linear label's sequence is untouched. The chi feeds
        # ``stepping.update_E`` ONLY -- the weld binds neither chi nor epsilon -- so
        # what a nonlinear case measures is the spine arm's admission, the
        # (``nonlinear``, ``PML``) absorb row, and the weld's bit identity inside a
        # step whose D/E half runs MEEP's Pade branch on every arrangement.
        # :func:`drive` holds the liveness floor that keeps the chi from being
        # decorative.
        chi2 = {component: cp.asarray(
                    (np.float32(0.02) * rng.uniform(0.4, 1.6, size=shape)
                     ).astype(np.float32)) for component in ("Ex", "Ey", "Ez")}
        chi3 = {component: cp.asarray(
                    (np.float32(0.05) * rng.uniform(0.4, 1.6, size=shape)
                     ).astype(np.float32)) for component in ("Ex", "Ey", "Ez")}
        fields.set_nonlinear_volumes(chi2, chi3)
    return fields, grid, pml, float(grid.dt / grid.dx)


def _nonlinear(fields) -> bool:
    """Both chi maps by name -- the spelling every nonlinear predicate on this track uses."""
    return bool(getattr(fields, "_chi2_components", None)
                or getattr(fields, "_chi3_components", None))


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


# ---------------------------------------------------------------------------
# The five arrangements
# ---------------------------------------------------------------------------

def _family():
    from meep_gpu.cuda_kernels import fused_hd_pair as family  # noqa: PLC0415
    return family


def run_pass(name: str, fields, pml) -> None:
    scratch_gate.run_pass(name, fields, pml)


class Arrangement:
    """One way of performing a complete driver step, and its own launch counter.

    Every arrangement performs the SAME ten driver passes; what differs is which of
    them run on the array path and which are absorbed into a launch. ``launches`` is
    the arrangement's own count, reported beside the memo counter's so two independent
    numbers have to agree.
    """

    __slots__ = ("name", "fields", "grid", "pml", "dtdx", "step", "launches",
                 "scratch", "tables", "codes")

    def __init__(self, name: str, fields, grid, pml, dtdx: float) -> None:
        self.name = name
        self.fields, self.grid, self.pml, self.dtdx = fields, grid, pml, dtdx
        self.launches = 0
        self.scratch = None
        self.tables = None
        self.codes = None

    def run_step(self) -> None:
        raise NotImplementedError


def _certified_single_launchers(fields, grid, pml, dtdx):
    """``{slot: callable}`` -- the certified single-slot launchers, resolved once."""
    from meep_gpu.cuda_kernels import (constitutive_kernels,  # noqa: PLC0415
                                       step_curl_kernels)

    curl_b = step_curl_kernels.real_pml_curl_tables(pml, True)
    curl_d = step_curl_kernels.real_pml_curl_tables(pml, False)
    codes = step_curl_kernels.real_curl_boundary_codes(grid)
    tables_h = constitutive_kernels.constitutive_tables_for("H", pml)
    tables_e = constitutive_kernels.constitutive_tables_for("E", pml)
    if _nonlinear(fields):
        # THE CERTIFIED update_E ON A NONLINEAR RUN IS THE NONLINEAR FAMILY'S, not the
        # ordinary kernel: ``coverage.covers_real_pml_constitutive(..., "E")`` refuses a
        # chi by name and the composer gives ``update_E`` to the ``cuda_nonlinear``
        # arm (the census records ``update_E: nonlinear`` on both corpus rows of the
        # cell). Dispatching the ordinary kernel here would compare the weld against
        # a reference that steps a DIFFERENT equation and call the disagreement the
        # weld's. ``update_H`` is unchanged: the nonlinear family's H arm is a null
        # widening that admits the shipped ``update_H_pml_real`` (that module's own
        # docstring), which is the whole fact the extra-arm row rests on.
        from meep_gpu.cuda_kernels import nonlinear_constitutive  # noqa: PLC0415
        update_e = lambda: nonlinear_constitutive.update_E_fused_pml_real_nonlinear(  # noqa: E731
            fields, pml)
    else:
        update_e = lambda: constitutive_kernels.update_fused_pml_real(  # noqa: E731
            "E", fields, tables=tables_e)
    return {
        "step_B": lambda: step_curl_kernels._step_B_fused_pml_real(  # noqa: SLF001
            fields, curl_b, codes, dtdx),
        "step_D": lambda: step_curl_kernels._step_D_fused_pml_real(  # noqa: SLF001
            fields, curl_d, codes, dtdx),
        "update_H": lambda: constitutive_kernels.update_fused_pml_real(
            "H", fields, tables=tables_h),
        "update_E": update_e,
    }


def make_arrangement(mode: str, fields, grid, pml, dtdx: float, *,
                     kernel: Optional[Any] = None, rotate: bool = True,
                     threads: Optional[int] = None,
                     own_load_reads_scratch: bool = False) -> Arrangement:
    """One of :data:`MODES`, built against an already-seeded engine."""
    family = _family()
    arrangement = Arrangement(mode, fields, grid, pml, dtdx)

    if mode == "array":
        def step() -> None:
            for name in STEP_PASSES:
                run_pass(name, fields, pml)
    elif mode in ("singles", "unfused"):
        singles = _certified_single_launchers(fields, grid, pml, dtdx)
        # `singles` dispatches ONLY the seam's two slots; `unfused` dispatches all
        # four of the slot path's. The distinction is the point: the first isolates
        # the seam, the second is the composition the composer builds with fusion
        # vetoed, and both must agree with the weld.
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
        from meep_gpu.cuda_kernels import (fused_electric_pair,  # noqa: PLC0415
                                           fused_magnetic_pair)

        # ON A NONLINEAR RUN THE COMPOSITION THAT SHIPS IS A DIFFERENT ONE, and the
        # census says which: the released B->H pair still holds step_B/update_H
        # (through its own (PML, nonlinear) extra-arm row), but the released D->E
        # pair refuses a chi by name and ``step_D`` / ``update_E`` go to the PML curl
        # single and the ``cuda_nonlinear`` arm respectively -- ``composer_selected``
        # on both corpus rows of the cell reads exactly that. So this arrangement
        # dispatches THAT composition (three launches) rather than raising on the
        # refusal the D->E pair correctly returns.
        #
        # THE ELECTRIC HALF'S BOUNDARY PASSES STILL RUN. A fused pair absorbs the
        # driver passes between its two slots (``fill_symmetry_bc_D``, ``zero_metal_D``,
        # ``fill_folded_far_ghosts_D`` for the D->E pair); a ``step_D`` SINGLE absorbs
        # none of them, so on this composition they run on the array path exactly as
        # ``unfused`` runs them. Measured 2026-09-07 before this was written: without
        # them the tangential D on a metallic z wall was never zeroed and the
        # arrangement diverged from the array path in exactly one 16x16 plane of Dx
        # and Dy at step 1, on the walled nonlinear fixture and on no periodic one.
        # The electric half is DERIVED from the magnetic pair's own REPLACES rather
        # than typed, and REPLACES is asserted to be the prefix of the step it claims.
        nonlinear_neighbours = (_certified_single_launchers(fields, grid, pml, dtdx)
                                if _nonlinear(fields) else None)
        magnetic_half = tuple(fused_magnetic_pair.REPLACES)
        if STEP_PASSES[:len(magnetic_half)] != magnetic_half:
            raise SystemExit(
                f"the released B->H pair claims REPLACES={magnetic_half}, which is not "
                f"the prefix of the driver step {STEP_PASSES}; the composition this "
                f"arrangement builds would not be the one the composer builds")
        electric_half = STEP_PASSES[len(magnetic_half):]

        def step() -> None:
            magnetic = fused_magnetic_pair.run_fused_magnetic_pair(
                fields, grid, pml, dtdx, sources=())
            if not magnetic.get("launched"):
                raise SystemExit(
                    f"the released B->H pair refused a fixture the H->D weld admits: "
                    f"{magnetic.get('reason')}")
            arrangement.launches += 1
            if nonlinear_neighbours is not None:
                for name in electric_half:
                    if name in ("step_D", "update_E"):
                        nonlinear_neighbours[name]()
                        arrangement.launches += 1
                    else:
                        run_pass(name, fields, pml)
                return
            electric = fused_electric_pair.run_fused_electric_pair(
                fields, grid, pml, dtdx, sources=())
            if not electric.get("launched"):
                raise SystemExit(
                    f"the released D->E pair refused a fixture the H->D weld admits: "
                    f"{electric.get('reason')}")
            arrangement.launches += 1
    elif mode in ("weld", "weld_composed"):
        arrangement.tables = family.fused_hd_pair_tables(pml)
        arrangement.codes = family.fused_hd_pair_codes(grid)
        arrangement.scratch = family.fused_hd_pair_scratch(fields)
        family.assert_scratch_is_disjoint(fields, arrangement.scratch,
                                          arrangement.tables)
        block = family._FUSED_THREADS if threads is None else int(threads)  # noqa: SLF001
        # ``weld`` ISOLATES the seam: every other pass runs on the array path, so a
        # divergence there is this weld's and not a neighbouring single's.
        # ``weld_composed`` is what a composer that INSTALLED this product would build
        # -- the weld plus the two neighbours as certified singles -- and it exists so
        # the launch-structure leg compares like with like against the composition
        # that ships. Using ``weld`` there would compare one launch against two and
        # report a saving the composition does not have.
        neighbours = (_certified_single_launchers(fields, grid, pml, dtdx)
                      if mode == "weld_composed" else None)

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
                report = family.launch_fused_hd_pair(
                    fields, arrangement.scratch, arrangement.tables,
                    arrangement.codes, dtdx, kernel, block)
                arrangement.launches += 1
                if rotate:
                    arrangement.scratch = family.rotate_into_fields(
                        fields, arrangement.scratch)
                del report
    else:
        raise ValueError(f"unknown arrangement {mode!r}")

    arrangement.step = step  # type: ignore[assignment]
    return arrangement


def drive(specs_label: str, spec: Mapping[str, Any], value_class: str, steps: int, *,
          modes: Sequence[str] = MODES, kernel: Optional[Any] = None,
          rotate: bool = True, threads: Optional[int] = None,
          progress: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
    """Step every arrangement side by side from ONE seed, compared per COMPLETE step.

    The comparison is against the ARRAY PATH for every other arrangement, and the
    record carries the first divergence per arrangement rather than stopping the
    others: a leg that stopped at the first disagreement could not say whether the
    weld or a certified single was the one that moved.
    """
    label = f"{spec['label']}/{value_class}"
    engines: Dict[str, Arrangement] = {}
    for mode in modes:
        fields, grid, pml, dtdx = build(spec, value_class, case_rng(label))
        engines[mode] = make_arrangement(
            mode, fields, grid, pml, dtdx, kernel=kernel if mode == "weld" else None,
            rotate=rotate if mode == "weld" else True,
            threads=threads if mode == "weld" else None)
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
    started = time.perf_counter()
    with MemoLaunchCounter() as counter:
        for step in range(steps):
            for mode in modes:
                engines[mode].step()
            cp.cuda.runtime.deviceSynchronize()
            state = {mode: frozen(engines[mode].fields) for mode in modes}
            compared = step + 1
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

    final = frozen(engines[modes[-1]].fields)
    must_move = STATE_NAMES if value_class == "uniform" else BAND_MUST_MOVE
    unmoved = sorted(name for name in must_move
                     if name in seeded and not differing(seeded[name], final[name]))
    nonlinear = (_nonlinear_facts(spec, value_class, label) if spec.get("nonlinear")
                 else None)
    return {
        "label": label, "spec": spec["label"], "value_class": value_class,
        "nonlinear": nonlinear,
        "steps_requested": steps, "steps_compared": compared,
        "shape": [int(n) for n in reference.grid.shape],
        "threads": int(threads or _family()._FUSED_THREADS),  # noqa: SLF001
        "rotated": bool(rotate),
        "identical": {mode: first[mode] is None for mode in modes[1:]},
        "first_divergence": {mode: first[mode] for mode in modes[1:]},
        "words_compared_per_step": word_total(seeded),
        "words_compared": word_total(seeded) * compared * max(len(modes) - 1, 1),
        "launches": {mode: engines[mode].launches for mode in modes},
        "memo_launches": {"named": memo_named, "total": memo_total},
        "read_only_drift": read_only_drift(read_only, reference.pml),
        "operand_census": census,
        "unmoved_volumes": unmoved, "non_vacuous": not unmoved,
        "seconds": round(time.perf_counter() - started, 2),
        "passed": (not unmoved
                   and not read_only_drift(read_only, reference.pml)
                   and all(first[mode] is None for mode in modes[1:])
                   and compared == steps
                   and (nonlinear is None or nonlinear["passed"])),
    }


def _nonlinear_facts(spec: Mapping[str, Any], value_class: str,
                     label: str) -> Dict[str, Any]:
    """What a nonlinear case measures BESIDE the identity, so the chi is not decorative.

    Four facts, each of which the identity alone cannot supply:

    1. **THE CHI IS LIVE.** The same seed built twice, the chi cleared on one copy, and
       ONE complete array-path step run on both: the two must diverge in the uniform
       class. A fixture whose chi moved nothing would score the identity on the linear
       equation and call it the nonlinear cell. In the subnormal-band class the
       divergence is NOT required: ``chi3 * E^2`` underflows to an exact zero on a
       state seeded in the band, so the Pade factor is exactly 1 there by float32
       arithmetic and not by any defect. The count is recorded either way.
    2. **THE PREDICATE ADMITS IT THROUGH THE SPINE ARM.** ``covers_fused_hd_pair`` says
       yes, the ordinary constitutive predicate says no, and the nonlinear one says
       yes -- the disjunction is a lookup between two disjoint predicates and both
       halves of it are recorded.
    3. **THE COMPOSER SELECTS THE NONLINEAR ARMS**, unfused: ``update_H`` goes to the
       ``cuda_nonlinear`` family's H arm (the null widening that admits the shipped
       kernel) and ``update_E`` to its E arm -- the census's own ``composer_selected``
       on both corpus rows of the cell, reproduced on the synthetic fixture.
    4. **THE EXTRA-ARM ROW IS THE ONE THAT ADMITS IT**: the selected
       (``update_H``, ``step_D``) arm labels equal ``FUSED_PAIR_EXTRA_ARMS``'s row for
       this family and NOT the primary row.
    """
    from meep_gpu.cuda_kernels import arms, fused_pairs  # noqa: PLC0415
    from meep_gpu.cuda_kernels.coverage import (  # noqa: PLC0415
        covers_real_pml_constitutive)
    from meep_gpu.cuda_kernels.nonlinear_constitutive import (  # noqa: PLC0415
        covers_real_pml_nonlinear_constitutive)

    family = _family()
    with_chi, _grid_a, pml_a, _ = build(spec, value_class, case_rng(label))
    without_chi, _grid_b, pml_b, _ = build(spec, value_class, case_rng(label))
    without_chi.set_nonlinear_volumes({}, {})
    seed_identical = not compare(frozen(with_chi), frozen(without_chi))
    for name in STEP_PASSES:
        run_pass(name, with_chi, pml_a)
        run_pass(name, without_chi, pml_b)
    cp.cuda.runtime.deviceSynchronize()
    moved = compare(frozen(with_chi), frozen(without_chi))
    chi_moved_words = int(sum(moved.values()))

    fields, grid, pml, _ = build(spec, value_class, case_rng(label))
    ordinary = covers_real_pml_constitutive(fields, pml, grid, "H")
    spine = covers_real_pml_nonlinear_constitutive(fields, pml, grid, "H")
    weld = family.covers_fused_hd_pair(fields, pml, grid, ())
    unfused = arms.plan_step(fields=fields, pml=pml, grid=grid, sources=(), fuse=False)
    selected = {slot: unfused.selected.get(slot)
                for slot in ("step_B", "update_H", "step_D", "update_E")}
    extra_rows = [tuple(row) for row in
                  fused_pairs.FUSED_PAIR_EXTRA_ARMS.get(family.FAMILY, ())]
    primary_row = tuple(fused_pairs.FUSED_PAIR_ARMS.get(family.FAMILY, ()))
    seam_arms = (selected.get("update_H"), selected.get("step_D"))
    liveness_required = value_class == "uniform"
    return {
        "seed_identical_before_chi_clear": seed_identical,
        "chi_moved_words_in_one_array_step": chi_moved_words,
        "chi_moved_arrays": dict(sorted(moved.items())),
        "chi_liveness_required_in_this_class": liveness_required,
        "weld_predicate_admits": bool(weld[0]),
        "weld_predicate_reason": weld[1],
        "the_ordinary_constitutive_predicate_refuses_it": not ordinary[0],
        "ordinary_reason": ordinary[1],
        "the_nonlinear_spine_arm_admits_it": bool(spine[0]),
        "spine_reason": spine[1],
        "the_two_are_disjoint": bool(ordinary[0]) != bool(spine[0]),
        "composer_selected_unfused": selected,
        "seam_arms_selected": list(seam_arms),
        "extra_arm_rows": [list(row) for row in extra_rows],
        "primary_arm_row": list(primary_row),
        "admitted_through_the_extra_arm_row": (seam_arms in extra_rows
                                              and seam_arms != primary_row),
        "passed": bool(
            seed_identical
            and (chi_moved_words > 0 or not liveness_required)
            and weld[0] and not ordinary[0] and spine[0]
            and selected.get("update_E") == "nonlinear"
            and seam_arms in extra_rows and seam_arms != primary_row),
    }


# ---------------------------------------------------------------------------
# Host leg: the driver's order, read from the driver
# ---------------------------------------------------------------------------

def leg_driver_order() -> Dict[str, Any]:
    """``REPLACES`` is the driver's two ADJACENT consults, and the seam holds one pass.

    Three things, all read off the tree with ``ast`` rather than spelled here:

    * the ordered ``stepping`` calls in ``FdtdDriver.step`` put ``update_H``
      immediately before ``step_D``;
    * the ONLY statement between the two CONSULT sites is the electric ``withdraw``
      loop -- which is what makes ``withdraw_hoist`` the seam's module and
      ``deposit_repair`` irrelevant here;
    * ``step_D`` is NOT in ``fastpath.SYNC_PATH_SLOTS``, so a plan spanning it must
      decline the ``update_H_synchronize`` consult. That is the containment rule the
      sync leg then measures on a device.
    """
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

    # THE SEAM'S OWN STATEMENTS, between the two CONSULTS rather than between the two
    # array calls. Walked as the module body's statement list so a second statement
    # arriving there is a named failure rather than an assumption that ages.
    body = ast.parse(source).body[0]
    statements = [node for node in ast.walk(body)]
    consults = [node for node in statements
                if isinstance(node, ast.Call)
                and getattr(getattr(node.func, "attr", None), "__str__", str)() == "dispatch"]
    consulted = [node.args[0].value for node in consults
                 if node.args and isinstance(node.args[0], ast.Constant)]
    between = [name for name in consulted
               if consulted.index(name) > consulted.index("update_H")
               and consulted.index(name) < consulted.index("step_D")]
    withdraw_calls = sum(
        1 for node in statements
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Call)
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
    }
    record["passed"] = bool(
        record["replaces_is_the_whole_span"] and record["adjacent"]
        and record["the_seam_holds_no_other_consult"]
        and record["seam_module_is_the_withdraw_hoist"]
        and not record["carries_deposit_repair"]
        and record["step_D_is_outside_the_sync_window"]
        and record["the_sync_consult_owner_is_this_products_leading_slot"]
        and withdraw_calls >= 2)
    return record


# ---------------------------------------------------------------------------
# Host leg: the transcription
# ---------------------------------------------------------------------------

def _emitted_pieces() -> Dict[str, str]:
    family = _family()
    head, tail = family.welded_curl_tail()
    return {"curl_prelude": family.curl_prelude(),
            "constitutive_prelude": family.constitutive_prelude(),
            "raw_update_H_cell": family.raw_update_H_cell_source(),
            "resolve_H": family.resolution_source(),
            "shift_dn_recompute": family.shift_dn_recompute_source(),
            "signature": family.signature(),
            "curl_head": head, "curl_tail": tail,
            "kernel": family.kernel_source()}


def leg_transcription() -> Dict[str, Any]:
    """Both halves are the certified emitters' own bytes, with the declared edits only."""
    from meep_gpu.cuda_kernels import (constitutive_kernels,  # noqa: PLC0415
                                       step_curl_kernels)

    family = _family()
    pieces = _emitted_pieces()
    source = pieces["kernel"]
    certified_curl = step_curl_kernels._step_D_pml_real_kernel_code  # noqa: SLF001
    certified_h = constitutive_kernels._update_H_pml_real_kernel_code  # noqa: SLF001

    # THE CURL PRELUDE IS CARRIED WHOLE AND UNEDITED, which is the strongest form this
    # check can take: no substring argument, just equality.
    curl_prelude_verbatim = (pieces["curl_prelude"]
                             == step_curl_kernels._REAL_PML_PRELUDE)  # noqa: SLF001

    # THE CURL BODY, LINE BY LINE. The certified tail below the decode and the welded
    # one must differ in EXACTLY the twelve magnetic reads and nothing else.
    certified_tail = certified_curl.split("\n) {\n", 1)[1]
    certified_tail = certified_tail[certified_tail.index("    int i = idx / (ny * nz);\n")
                                    + len("    int i = idx / (ny * nz);\n"):]
    certified_tail = certified_tail[: -len("}\n")]
    left = certified_tail.splitlines()
    right = pieces["curl_tail"].splitlines()
    changed = [(a, b) for a, b in zip(left, right) if a != b]
    curl_edits_are_the_reads = (
        len(left) == len(right) and len(changed) == family.HALO_TAPS
        + family.OWN_LOAD_EDITS
        and all(("shift_dn(" in a and "shift_dn_recompute(" in b)
                or (("Hx[idx]" in a or "Hy[idx]" in a or "Hz[idx]" in a)
                    and "own_h[" in b)
                for a, b in changed))

    # THE CONSTITUTIVE BODY. Every certified line survives except the preamble and the
    # three call lines, and the DECODE is present verbatim.
    h_body = certified_h.split("\n) {\n", 1)[1][: -len("}\n")]
    decode = ("    int k = idx % nz;\n    int j = (idx / nz) % ny;\n"
              "    int i = idx / (ny * nz);\n")
    kept = [line for line in h_body.splitlines()
            if line.strip() and not line.strip().startswith("//")
            and "blockIdx" in line or False]
    del kept
    constitutive_facts = {
        "decode_is_carried_verbatim": decode in pieces["raw_update_H_cell"],
        "thread_preamble_is_gone":
            "blockIdx.x" not in pieces["raw_update_H_cell"],
        "three_calls_captured":
            pieces["raw_update_H_cell"].count("constitutive_apply_pure(") == 3,
        "the_two_accumulations_are_separate":
            "float a = f[idx] + kps * src;" in pieces["constitutive_prelude"]
            and "return a - kms * prev;" in pieces["constitutive_prelude"],
        "the_helper_stores_nothing":
            "fw[idx] = src;" not in pieces["constitutive_prelude"]
            and "f[idx] = a" not in pieces["constitutive_prelude"],
        "prev_is_read_before_the_store_moves":
            pieces["constitutive_prelude"].index("float prev = fw[idx];")
            < pieces["constitutive_prelude"].index("*fw_out = src;"),
    }

    # THE GHOST RULE. The recomputed shift keeps the exact 0.0f and the certified
    # branch, and reads no volume.
    ghost_facts = {
        "keeps_the_exact_zero": ": 0.0f;" in pieces["shift_dn_recompute"],
        "keeps_the_periodic_test":
            "(bc == BC_PERIODIC) ?" in pieces["shift_dn_recompute"],
        "keeps_the_near_branch": "if (ia > 0) return" in pieces["shift_dn_recompute"],
        "reads_no_volume": "g[" not in pieces["shift_dn_recompute"],
        "index_expressions_are_the_certified_ones":
            "idx - stride" in pieces["shift_dn_recompute"]
            and "idx + (na - 1) * stride" in pieces["shift_dn_recompute"],
    }

    # THE DECLARED EDIT TABLES resolve: every "line" that is a literal anchor is found
    # in the certified text and every "became" in the emitted text.
    def anchors(table, certified: str) -> Dict[str, Any]:
        resolved, missing = 0, []
        for entry in table:
            line = entry["line"]
            if "*" in line or "..." in line or "<" in line or "/" in line:
                continue           # a SHAPE rather than a literal anchor
            if line.strip() and all(part.strip() in certified
                                    for part in line.split("\n")):
                resolved += 1
            else:
                missing.append(line)
        return {"resolved": resolved, "unresolved": missing}

    edit_facts = {
        "constitutive": anchors(family.CONSTITUTIVE_LIFT_EDITS,
                                constitutive_kernels._REAL_CONSTITUTIVE_PRELUDE  # noqa: SLF001
                                + certified_h),
        "curl": anchors(family.CURL_LIFT_EDITS,
                        step_curl_kernels._REAL_PML_PRELUDE + certified_curl),  # noqa: SLF001
    }

    # NO STALE READ SURVIVES. In the fused signature `Hx`/`Hy`/`Hz` are the PRE-LAUNCH
    # magnetic field, so any indexing of them below the weld would be the very defect
    # the design removes. They appear ONLY in the signature, the pack construction and
    # raw_update_H_cell's own binding.
    below = source.split("raw_update_H_cell(idx, weld, own_h, own_w);", 1)[1]
    residue = {name: below.count(f"{name}[") for name in family.H_TARGETS}

    facts = {
        "curl_prelude_is_verbatim": curl_prelude_verbatim,
        "curl_tail_edits_are_exactly_the_magnetic_reads": curl_edits_are_the_reads,
        "curl_tail_changed_lines": len(changed),
        "expected_changed_lines": family.HALO_TAPS + family.OWN_LOAD_EDITS,
        "constitutive": constitutive_facts,
        "ghost": ghost_facts,
        "declared_edits": edit_facts,
        "no_stale_magnetic_read_below_the_weld": residue,
        "source_is_ascii": source.isascii(),
        "source_encodes_under_ascii": _encodes_ascii(source),
        "one_device_string": list(family.device_sources()) == [family.KERNEL_NAME],
        "source_sha256": family.source_digest(),
        "source_bytes": len(source),
        "pml_apply_defined_once":
            source.count("__device__ __forceinline__ void pml_apply(") == 1,
        "mutation_needles_resolve": _mutation_needles_resolve(source),
    }
    facts["passed"] = bool(
        facts["curl_prelude_is_verbatim"]
        and facts["curl_tail_edits_are_exactly_the_magnetic_reads"]
        and all(constitutive_facts.values()) and all(ghost_facts.values())
        and not edit_facts["constitutive"]["unresolved"]
        and not edit_facts["curl"]["unresolved"]
        and sum(residue.values()) == 0
        and facts["source_is_ascii"] and facts["source_encodes_under_ascii"]
        and facts["one_device_string"] and facts["pml_apply_defined_once"]
        and facts["mutation_needles_resolve"]["passed"])
    return facts


def _encodes_ascii(source: str) -> bool:
    """The NVRTC locale trap, PERFORMED rather than scanned.

    ``cupy.cuda.compiler.compile_using_nvrtc`` writes the source with a bare
    ``open(..., 'w')``, so the bytes go through the interpreter's LOCALE encoding --
    ASCII under C/POSIX, which is what a non-interactive shell on the validation host
    gets. Two em-dashes in a comment once killed a kernel at its first launch.
    """
    try:
        source.encode("ascii")
    except UnicodeEncodeError:
        return False
    return True


def _mutation_needles_resolve(source: str) -> Dict[str, Any]:
    """Every armed device mutation's anchor appears exactly once in the shipped text.

    A mutation that did not land is a mutation that was not tested, and it would be
    reported UNCAUGHT -- the one way a mutation battery lies.
    """
    counts: Dict[str, int] = {}
    for name, spec in DEVICE_MUTATIONS.items():
        if spec["old"] is _DROP_NEAR_FACE_MASKS:
            # NOT A SINGLE ANCHOR: this one edits twelve lines and asserts its own
            # count, so the check is that applying it LANDS and removes exactly those
            # twelve rather than that a needle resolves once.
            counts[name] = (len(source.splitlines())
                            - len(_drop_near_face_masks(source).splitlines())) // 12
            continue
        counts[name] = source.count(spec["old"])
        for also_old, _also_new in spec.get("also", ()):
            counts[name] = min(counts[name], source.count(also_old))
    return {"counts": counts,
            "passed": all(count == 1 for count in counts.values())}


# ---------------------------------------------------------------------------
# Host leg: the refusals
# ---------------------------------------------------------------------------

def leg_refusal() -> Dict[str, Any]:
    """Every configuration this launch cannot serve, refused BY NAME."""
    from meep_gpu.fields import Fields  # noqa: PLC0415
    from meep_gpu.grid import Grid, Mirror  # noqa: PLC0415
    from meep_gpu.pml import PML  # noqa: PLC0415
    from meep_gpu.sources import (ContinuousEnvelope, GaussianEnvelope,  # noqa: PLC0415
                                  VolumeSource)

    family = _family()
    checks: Dict[str, Any] = {}
    cell = (1.6, 1.6, 1.6)

    def plain(**kwargs):
        grid = Grid(resolution=10.0, cell_size=cell, courant=0.35, xp=cp,
                    boundaries=("periodic", "periodic", "periodic"), **kwargs)
        fields = Fields(grid=grid)
        fields.enable_field_storage()
        fields.enable_pml_storage()
        return fields, PML(grid=grid, thickness=2), grid

    # 0. THE POSITIVE CONTROL. A refusal leg whose every case refuses proves nothing:
    #    it would look identical if the predicate refused everything.
    fields, pml, grid = plain()
    checks["admits_a_plain_run"] = _verdict(family, fields, pml, grid, (), True)

    # 1. THE FOLD IS ADMITTED, and by the CURL's runtime boundary code rather than by a
    #    second product. This is the whole reason the CUDA cell is 127 rows where
    #    Metal's is 49.
    folded_fields, _folded_pml, folded_grid = plain(symmetry=(Mirror("Y", 1),))
    folded_pml = PML(grid=folded_grid, thickness=((2, 2), (0, 2), (2, 2)))
    checks["admits_a_folded_grid"] = _verdict(
        family, folded_fields, folded_pml, folded_grid, (), True)

    # 2. THE SEAM'S ONE PASS.
    checks["refuses_an_undeclared_source_set"] = _verdict(
        family, fields, pml, grid, None, False, "was not declared")
    integrated = VolumeSource(grid=grid, component="Ez", center=(0.0, 0.0, 0.0),
                              size=(0.0, 0.0, 0.0),
                              envelope=ContinuousEnvelope(frequency=1.0,
                                                          is_integrated=True))
    checks["refuses_a_standing_integrated_electric_withdraw"] = _verdict(
        family, fields, pml, grid, [integrated], False, "standing integrated")
    plain_source = VolumeSource(grid=grid, component="Ez", center=(0.0, 0.0, 0.0),
                                size=(0.0, 0.0, 0.0),
                                envelope=GaussianEnvelope(frequency=1.0, fwidth=0.2))
    checks["admits_a_non_integrated_electric_source"] = _verdict(
        family, fields, pml, grid, [plain_source], True)
    magnetic = VolumeSource(grid=grid, component="Hz", center=(0.0, 0.0, 0.0),
                            size=(0.0, 0.0, 0.0),
                            envelope=ContinuousEnvelope(frequency=1.0,
                                                        is_integrated=True))
    checks["admits_an_integrated_magnetic_source"] = _verdict(
        family, fields, pml, grid, [magnetic], True)

    # 3. THE TWO HALVES' OWN CLAUSES, asked through this predicate so a widening is
    #    visible here rather than only in a half's own test.
    complex_grid = Grid(resolution=10.0, cell_size=cell, courant=0.35, xp=cp,
                        boundaries=("periodic", "periodic", "periodic"))
    complex_fields = Fields(grid=complex_grid, force_complex_fields=True)
    complex_fields.enable_field_storage()
    complex_fields.enable_pml_storage()
    checks["refuses_complex_storage"] = _verdict(
        family, complex_fields, PML(grid=complex_grid, thickness=2), complex_grid,
        (), False, "complex64")
    checks["refuses_a_run_with_no_pml"] = _verdict(
        family, fields, None, grid, (), False, "no active PML")

    # A CONDUCTIVITY IS A PER-TARGET QUESTION ON THIS SEAM, not a per-engine one, and
    # the two answers differ. ``covers_real_pml_curl`` charges a sigma to the curl that
    # WRITES the carrying component (the 2026-08-15 change), so an engine that carries
    # one somewhere is admitted here and one whose ``step_D`` target carries it is
    # refused -- which is the board's own ``(cuda_constitutive/ordinary ->
    # cuda_conductive/conductive)`` cell, a DIFFERENT cell this product must not take.
    # Both halves of that split are measured; asking only the boolean would have
    # scored the admission as a defect and the refusal as untested.
    class _EngineWideConductivity:
        def __getattr__(self, item):
            return getattr(fields, item)

        has_conductivity = True

    checks["admits_an_engine_that_carries_a_conductivity_elsewhere"] = _verdict(
        family, _EngineWideConductivity(), pml, grid, (), True)

    conductive_fields, conductive_pml, conductive_grid = plain()
    # THE ENGINE'S OWN SETTER, and the D side specifically: MEEP's set_conductivity
    # redirects a medium's loss to D and B and never to E or H (fields.py:222,
    # structure.cpp:377-379), and it is the D one that reaches ``step_D``'s targets.
    conductive_shape = tuple(int(n) for n in conductive_grid.shape)
    conductive_fields.set_d_conductivity(
        cp.full(conductive_shape, np.float32(0.3), cp.float32))
    checks["refuses_a_conductivity_on_the_step_D_target"] = _verdict(
        family, conductive_fields, conductive_pml, conductive_grid, (), False,
        "carries a conductivity")

    class _Bfast:
        def __getattr__(self, item):
            return getattr(grid, item)

        bfast_active = True

    checks["refuses_bfast"] = _verdict(family, fields, pml, _Bfast(), (), False,
                                       "BFAST")

    class _Beta:
        def __getattr__(self, item):
            return getattr(grid, item)

        beta = 0.3

    checks["refuses_special_kz"] = _verdict(family, fields, pml, _Beta(), (), False,
                                            "beta")

    class _Cylindrical:
        def __getattr__(self, item):
            return getattr(grid, item)

        def is_cylindrical(self):
            return True

        cylindrical = True

    checks["refuses_a_cylindrical_grid"] = _verdict(
        family, fields, pml, _Cylindrical(), (), False)

    # 4. THE ROTATION'S OWN REQUIREMENT.
    class _NoHz:
        def __getattr__(self, item):
            return getattr(fields, item)

        Hz = None

    # AND THE REFUSAL IS NAMED BY THE CONSTITUTIVE HALF, not by this product's own
    # rotation clause -- measured, and recorded rather than tidied. The certified
    # constitutive predicate already checks its side's targets, auxiliaries and
    # sources, which is exactly the six volumes this weld rotates, so it answers
    # first. This product's clause is therefore fail-closed BELT AND BRACES with no
    # live path, and saying that is more useful than a check that pretends otherwise.
    entry = _verdict(family, _NoHz(), pml, grid, (), False)
    entry["named_by"] = ("the constitutive half"
                         if entry["reason"].startswith("constitutive half")
                         else "this product's rotation clause")
    entry["names_the_volume"] = "Hz" in entry["reason"]
    entry["passed"] = bool(entry["passed"] and entry["names_the_volume"])
    checks["refuses_an_unallocated_rotated_volume"] = entry

    # 5. THE NONLINEAR WIDENING, ADMITTED. Reached through the spine arm; the ordinary
    #    constitutive predicate refuses it by name, and both facts are recorded so the
    #    disjunction is a lookup rather than a widening.
    from meep_gpu.cuda_kernels.coverage import (  # noqa: PLC0415
        covers_real_pml_constitutive)
    from meep_gpu.cuda_kernels.nonlinear_constitutive import (  # noqa: PLC0415
        covers_real_pml_nonlinear_constitutive)

    nl_fields, nl_pml, nl_grid = plain()
    shape = tuple(int(n) for n in nl_grid.shape)
    nl_fields.set_nonlinearity(chi2={"Ez": cp.full(shape, np.float32(0.1),
                                                   cp.float32)},
                               chi3={"Ez": cp.full(shape, np.float32(0.2),
                                                   cp.float32)}) \
        if hasattr(nl_fields, "set_nonlinearity") else None
    if not (getattr(nl_fields, "_chi2_components", None)
            or getattr(nl_fields, "_chi3_components", None)):
        # NO SETTER ON THIS ENGINE: plant the two maps the predicates actually read,
        # by their own names, rather than skipping the case.
        nl_fields._chi2_components = {"Ez": cp.full(shape, np.float32(0.1),  # noqa: SLF001
                                                    cp.float32)}
        nl_fields._chi3_components = {"Ez": cp.full(shape, np.float32(0.2),  # noqa: SLF001
                                                    cp.float32)}
    ordinary = covers_real_pml_constitutive(nl_fields, nl_pml, nl_grid, "H")
    spine = covers_real_pml_nonlinear_constitutive(nl_fields, nl_pml, nl_grid, "H")
    entry = _verdict(family, nl_fields, nl_pml, nl_grid, (), True)
    entry.update({
        "the_ordinary_constitutive_predicate_refuses_it": not ordinary[0],
        "ordinary_reason": ordinary[1],
        "the_nonlinear_spine_arm_admits_it": bool(spine[0]),
        "spine_reason": spine[1],
        "the_two_are_disjoint": bool(ordinary[0]) != bool(spine[0]),
    })
    entry["passed"] = bool(entry["passed"] and entry["the_two_are_disjoint"])
    checks["admits_a_nonlinear_run_through_the_spine_arm"] = entry

    return {"checks": checks,
            "passed": all(entry["passed"] for entry in checks.values()),
            "denominator": len(checks)}


def _verdict(family, fields, pml, grid, sources, expected: bool,
             names: Optional[str] = None) -> Dict[str, Any]:
    covered, reason = family.covers_fused_hd_pair(fields, pml, grid, sources)
    entry = {"covered": bool(covered), "expected": expected, "reason": reason}
    entry["passed"] = bool(covered) == expected
    if names is not None:
        entry["names_the_reason"] = names in reason
        entry["passed"] = entry["passed"] and entry["names_the_reason"]
    return entry


# ---------------------------------------------------------------------------
# Host leg: the arbitration
# ---------------------------------------------------------------------------

def leg_arbitration(spec: Mapping[str, Any]) -> Dict[str, Any]:
    """The COMPOSER, asked. Who holds ``update_H``, and why this product does not.

    Driven through the shipped ``arms.plan_step(..., fuse=True)`` on a real device
    engine, so the answer is what the composer builds rather than what a table says.
    """
    from meep_gpu.cuda_kernels import arms, fused_pairs  # noqa: PLC0415

    family = _family()
    fields, grid, pml, dtdx = build(spec, "uniform", case_rng("arbitration"))
    plan = arms.plan_step(fields=fields, pml=pml, grid=grid, sources=(), fuse=True)
    selected = dict(getattr(plan, "selected", {}) or {})
    reasons = {name: list(value) for name, value
               in (getattr(plan, "reasons", {}) or {}).items()}
    mine = [text for key, texts in reasons.items() if family.FAMILY in key
            for text in texts]
    record = {
        "spec": spec["label"],
        "selected": selected,
        "update_H_holder": selected.get("update_H"),
        "step_D_holder": selected.get("step_D"),
        "this_product_is_refused": bool(mine),
        "refusal_reasons": mine,
        "refusal_names_installable":
            any("INSTALLABLE = False" in text for text in mine),
        "seam_row": list(fused_pairs.FUSED_PAIR_SEAMS.get("update_H", ())),
        "seam_row_routes_to_the_withdraw_hoist":
            fused_pairs.FUSED_PAIR_SEAMS.get("update_H", (None, None))[1]
            == withdraw_hoist.SEAM,
        "arm_row": list(fused_pairs.FUSED_PAIR_ARMS.get(family.FAMILY, ())),
        "extra_arm_rows": [list(row) for row in
                           fused_pairs.FUSED_PAIR_EXTRA_ARMS.get(family.FAMILY, ())],
        "product_row_present": family.FAMILY in fused_pairs.FUSED_PRODUCTS,
        "declared_uninstallable": fused_pairs._declared_uninstallable(  # noqa: SLF001
            family.FAMILY, fused_pairs.FUSED_PRODUCTS[family.FAMILY])
            if family.FAMILY in fused_pairs.FUSED_PRODUCTS else None,
        "installable": bool(family.INSTALLABLE),
        "what_a_release_does_not_license": family.WHAT_A_RELEASE_DOES_NOT_LICENSE,
    }
    # THE HOLDER IS A PLAN LABEL, not a family key -- ``_install_fused_pair`` writes
    # ``getattr(pair, "label", family)`` into ``selected`` -- so the claim is made
    # STRUCTURALLY rather than against a spelling this file invented: ONE label holds
    # both slots of the B->H seam, and it is not this product's.
    holder = selected.get("update_H")
    record["the_released_b_to_h_pair_keeps_update_H"] = bool(
        holder is not None and selected.get("step_B") == holder
        and family.FAMILY not in str(holder))
    record["the_released_d_to_e_pair_keeps_step_D"] = bool(
        selected.get("step_D") is not None
        and selected.get("step_D") == selected.get("update_E")
        and family.FAMILY not in str(selected.get("step_D")))
    record["this_product_holds_no_slot"] = not any(
        family.FAMILY in str(value) for value in selected.values())
    record["passed"] = bool(
        record["product_row_present"] and record["arm_row"]
        and record["this_product_is_refused"]
        and record["seam_row_routes_to_the_withdraw_hoist"]
        and not record["installable"]
        and record["declared_uninstallable"]
        and record["this_product_holds_no_slot"]
        and record["the_released_b_to_h_pair_keeps_update_H"]
        and record["the_released_d_to_e_pair_keeps_step_D"])
    return record


# ---------------------------------------------------------------------------
# Device leg: the purity / race ledger
# ---------------------------------------------------------------------------

def leg_purity_ledger(specs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Of the curl's valid foreign taps, how many land on a cell ``update_H`` MOVED?

    THIS IS THE MEASUREMENT THAT MAKES THE IDENTITY RESULT MEAN ANYTHING. The weld
    recomputes every foreign tap from PRE-LAUNCH state instead of loading the word an
    in-place weld would have raced on. If those two values were the same almost
    everywhere, the identity legs would be consistent with an in-place weld too and
    would license nothing about the design.

    Measured on the ARRAY PATH, not through the kernel: ``update_H`` is run on a copy
    of the seeded engine and the stepped ``H`` compared to the pre-launch ``H`` at
    exactly the cells the certified ``shift_dn`` resolves for each of the six taps,
    per component and per axis, counting a tap as DIFFERING when the two float32 words
    differ. The ghost arm (an exact ``0.0f``) is excluded from the denominator by name
    and reported separately: it reads no memory in either arrangement and can never
    race.
    """
    from meep_gpu.cuda_kernels import coverage as cuda_coverage  # noqa: PLC0415

    #: (target component, tap component, axis) for the six shifted reads, read off the
    #: certified step_D body rather than tabulated: Dx reads Hz on y and Hy on z, Dy
    #: reads Hx on z and Hz on x, Dz reads Hy on x and Hx on y.
    taps: Tuple[Tuple[int, int, int], ...] = (
        (0, 2, 1), (0, 1, 2), (1, 0, 2), (1, 2, 0), (2, 1, 0), (2, 0, 1))
    rows: List[Dict[str, Any]] = []
    for spec in specs:
        fields, grid, pml, _dtdx = build(spec, "uniform", case_rng(f"purity/{spec['label']}"))
        before = {name: to_host(getattr(fields, name)).copy()
                  for name in ("Hx", "Hy", "Hz")}
        stepping.update_H(fields, pml)
        after = {name: to_host(getattr(fields, name)).copy()
                 for name in ("Hx", "Hy", "Hz")}
        codes, refusal = cuda_coverage.real_curl_boundary_codes(grid)
        if refusal is not None:
            rows.append({"spec": spec["label"], "refused": refusal})
            continue
        codes = tuple(int(code) for code in codes)
        shape = tuple(int(n) for n in grid.shape)
        total = ghost = differing_taps = 0
        for _target, component, axis in taps:
            name = ("Hx", "Hy", "Hz")[component]
            n = shape[axis]
            index = np.arange(n)
            # ``shift_dn``'s own branch, per coordinate on this axis.
            source = np.where(index > 0, index - 1,
                              n - 1 if codes[axis] == 0 else -1)
            valid = source >= 0
            plane_cells = int(np.prod(shape) // n)
            total += int(valid.sum()) * plane_cells
            ghost += int((~valid).sum()) * plane_cells
            moved = np.take(after[name] != before[name], source[valid], axis=axis)
            differing_taps += int(moved.sum())
        rows.append({
            "spec": spec["label"], "shape": list(shape), "boundary_codes": list(codes),
            "taps_that_read_memory": total,
            "taps_that_read_the_exact_zero_ghost": ghost,
            "taps_whose_recomputed_value_differs": differing_taps,
            "fraction": round(differing_taps / total, 6) if total else None,
        })
    measured = [row for row in rows if "taps_that_read_memory" in row]
    denominator = sum(row["taps_that_read_memory"] for row in measured)
    numerator = sum(row["taps_whose_recomputed_value_differs"] for row in measured)
    return {
        "rows": rows, "fixtures": len(measured),
        "taps_that_read_memory": denominator,
        "taps_whose_recomputed_value_differs": numerator,
        "fraction": round(numerator / denominator, 6) if denominator else None,
        "what_this_licenses": (
            "the recompute is LOAD-BEARING wherever this fraction is high: every such "
            "tap is a cell an in-place weld would have read at a schedule-decided "
            "time, so the identity legs are evidence about THIS design and not a "
            "result an in-place weld would have shared. A fraction near zero would "
            "mean the opposite and the identity would license nothing"),
        "passed": bool(denominator > 0 and numerator == denominator),
    }


# ---------------------------------------------------------------------------
# Device leg: launch structure
# ---------------------------------------------------------------------------

def leg_launch_structure(spec: Mapping[str, Any], steps: int) -> Dict[str, Any]:
    """Launches per step at the seam and over the whole step, for all five arrangements.

    TWO INDEPENDENT COUNTERS: each arrangement's own tally and a wrapper around the
    shipped compile memo, which sees every launch through every shipped route.

    THE HONEST NUMBER IS REPORTED WHOLE. Over ``step_B`` .. ``update_E`` the
    composition installed today is TWO launches (the two released pairs) and the weld
    plus its neighbours' singles is THREE. The weld saves one launch against the
    unfused four and costs one against the composition that ships, which is the
    arbitration ruling as a measurement.
    """
    out: Dict[str, Any] = {"spec": spec["label"], "steps": steps, "modes": {}}
    for mode in MODES + ("weld_composed",):
        fields, grid, pml, dtdx = build(spec, "uniform", case_rng("launch"))
        arrangement = make_arrangement(mode, fields, grid, pml, dtdx)
        arrangement.step()      # warm the memo: an uncounted first compile
        cp.cuda.runtime.deviceSynchronize()
        arrangement.launches = 0
        with MemoLaunchCounter() as counter:
            for _ in range(steps):
                arrangement.step()
            cp.cuda.runtime.deviceSynchronize()
            named, total = counter.named(), counter.total
        out["modes"][mode] = {
            "arrangement_launches_per_step": arrangement.launches / steps,
            "memo_launches_per_step": total / steps,
            "memo_named": named,
            "counters_agree": arrangement.launches == total,
        }
    isolated = out["modes"]["weld"]["memo_launches_per_step"]
    composed = out["modes"]["weld_composed"]["memo_launches_per_step"]
    today = out["modes"]["composition_today"]["memo_launches_per_step"]
    unfused = out["modes"]["unfused"]["memo_launches_per_step"]
    singles = out["modes"]["singles"]["memo_launches_per_step"]
    out["seam_launches"] = {
        "weld": 1.0, "certified_singles": singles,
        "note": ("the H_to_D seam ALONE, with every other pass on the array path: one "
                 "launch welded against the two the certified singles take. This is "
                 "the only number the weld improves, and it is a seam number rather "
                 "than a step number")}
    out["whole_step"] = {
        "weld_isolated": isolated, "weld_composed": composed,
        "composition_today": today, "unfused": unfused,
        "weld_composed_minus_composition_today": round(composed - today, 6),
        "weld_composed_minus_unfused": round(composed - unfused, 6),
        "the_weld_costs_a_launch_against_the_composition_that_ships": composed > today,
        "the_weld_saves_a_launch_against_the_unfused_slots": composed < unfused,
        "what_this_says": (
            "over step_B .. update_E the composition that SHIPS is two launches (the "
            "two released pairs) and the composition with this product installed is "
            "three (step_B single, the weld, update_E single). The weld saves one "
            "launch against the four unfused slots and COSTS one against the "
            "composition that ships, which is the arbitration ruling reproduced as a "
            "measurement rather than re-litigated. No timing is claimed either way"),
    }
    out["passed"] = bool(
        all(entry["counters_agree"] for entry in out["modes"].values())
        and out["seam_launches"]["certified_singles"] == 2.0
        and out["whole_step"]["the_weld_saves_a_launch_against_the_unfused_slots"]
        and out["whole_step"]["the_weld_costs_a_launch_against_the_composition_that_ships"])
    return out


# ---------------------------------------------------------------------------
# Device leg: the sync channel
# ---------------------------------------------------------------------------

class Shim:
    """The driver's dispatch consult, implemented over a mapping of slot -> callable.

    ONE METHOD over the names it carries, which is the shipped protocol
    (``FastPathPlan.dispatch``): a name it does not carry answers False, which is the
    array path. ``answers_the_sync_consult`` is the leg's own switch -- the hazard arm
    answers the ``update_H_synchronize`` name and the guarded arm declines it by the
    containment rule.
    """

    __slots__ = ("plans", "answers_the_sync_consult", "calls")

    def __init__(self, plans: Mapping[str, Callable[[], None]], *,
                 answers_the_sync_consult: bool = False) -> None:
        self.plans = dict(plans)
        self.answers_the_sync_consult = bool(answers_the_sync_consult)
        self.calls: Dict[str, int] = {}

    def dispatch(self, name: str, fields: Any) -> bool:  # noqa: ARG002
        if name == SYNC_UPDATE_H_PASS:
            if not self.answers_the_sync_consult:
                return False
            name = SYNC_PASS_OWNERS[SYNC_UPDATE_H_PASS]
        run = self.plans.get(name)
        if run is None:
            return False
        self.calls[name] = self.calls.get(name, 0) + 1
        run()
        return True


#: THE SEED IS SCALED BY 2^80 ON THE DRIVER-LEVEL LEGS, AND THE EXPONENT IS A
#: MEASUREMENT RATHER THAN A TUNING. Those legs seed a source-free-decaying state and
#: run it inside an absorber, so the state DECAYS toward the float32 denormal band --
#: and from there no byte claim can be made for a reason that has nothing to do with
#: any kernel. The solver is LINEAR in the field state and every coefficient it
#: multiplies by is field-independent, so scaling every stored volume by a POWER OF TWO
#: shifts each float32 exponent by n and leaves every mantissa and every rounding
#: decision untouched: the same arithmetic is measured, further from the band. This is
#: the sibling Metal gate's own constant and its own argument.
SEED_SCALE_BITS = 80


def _build_driver(spec: Mapping[str, Any], *, integrated: bool = False,
                  seed: int = SEED, scale_bits: int = SEED_SCALE_BITS):
    """One seeded ``FdtdDriver`` on the device, with one electric source.

    A DRIVER, not a bare ``Fields``: the legs that use it are claims about a COMPLETE
    driver step -- the withdraw loop, the injection slot, the fill consults, the wall
    clears and the sync channel are ``FdtdDriver.step``'s and
    ``synchronize_magnetic_fields``'s, and a hand-written walk over the live pass list
    would be a second model of what a step is.
    """
    from meep_gpu.driver import FdtdDriver  # noqa: PLC0415

    boundaries = ({axis: kind for axis, kind in zip("xyz", spec["boundaries"])}
                  if spec["boundaries"] else None)
    driver = FdtdDriver(cell_size=spec["cell"], resolution=10.0, courant=0.35,
                        force_complex_fields=False, boundaries=boundaries,
                        symmetry=[{"direction": axis, "phase": phase}
                                  for axis, phase in spec["symmetry"]],
                        dimensions=3, prefer_gpu=True, gpu_id=0)
    driver.setup_pml(2)
    shape = tuple(int(n) for n in driver.grid.shape)
    driver.fields.set_epsilon_volumes(
        {name: cp.full(shape, np.float32(value), cp.float32)
         for name, value in (("Ex", 2.0), ("Ey", 2.5), ("Ez", 3.0))},
        {name: cp.full(shape, np.float32(1.0 / value), cp.float32)
         for name, value in (("Ex", 2.0), ("Ey", 2.5), ("Ez", 3.0))})
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
    state = {"tables": family.fused_hd_pair_tables(pml),
             "codes": family.fused_hd_pair_codes(grid),
             "scratch": family.fused_hd_pair_scratch(fields),
             "launches": 0, "withdrawn": 0}

    def run() -> None:
        if hoisted:
            state["withdrawn"] += withdraw_hoist.hoist(
                fields, driver._sources, span=family.REPLACES,  # noqa: SLF001
                placement=placement)
        family.launch_fused_hd_pair(fields, state["scratch"], state["tables"],
                                    state["codes"], dtdx)
        state["launches"] += 1
        state["scratch"] = family.rotate_into_fields(fields, state["scratch"])

    return {"update_H": run, "step_D": lambda: None}, state


def _driver_state(driver) -> Dict[str, np.ndarray]:
    return {name: to_host(array).copy()
            for name in STATE_NAMES
            if (array := getattr(driver.fields, name, None)) is not None}


def leg_sync(spec: Mapping[str, Any], steps: int) -> Dict[str, Any]:
    """The magnetic half-step's channel, with the product FORCE-INSTALLED.

    ``synchronize_magnetic_fields`` repeats the magnetic half of ``step`` and then
    UNDOES it, but its backup is ``_SYNC_FIELDS`` + ``_SYNC_AUXILIARY`` -- magnetic
    names only. ``D``, ``fu_D`` and ``f_cond_D`` are in NEITHER, so a product spanning
    ``update_H`` and ``step_D`` that answered the ``update_H_synchronize`` consult
    would advance the electric state inside a window nothing can undo, on every flux
    and energy call, and the run would carry it to the end.

    THIS LEG IS THE CITATION FOR THE COMPOSER REFUSAL and it has its own null control:
    the hazard arm MUST diverge in D/fu_D against a run that never synchronizes, and
    the guarded arm -- which declines the consult by the containment rule and lets the
    array ``update_H`` run inside the half-step -- MUST NOT.
    """
    facts: Dict[str, Any] = {"spec": spec["label"], "steps": steps,
                             "sync_steps": list(SYNC_STEPS),
                             "sync_fields": list(getattr(
                                 __import__("meep_gpu.driver", fromlist=["x"])
                                 .FdtdDriver, "_SYNC_FIELDS")),
                             "sync_auxiliary": list(getattr(
                                 __import__("meep_gpu.driver", fromlist=["x"])
                                 .FdtdDriver, "_SYNC_AUXILIARY"))}
    facts["D_is_in_neither_backup_list"] = not any(
        name.startswith(("D", "fu_D", "f_cond_D"))
        for name in facts["sync_fields"] + facts["sync_auxiliary"])

    arms_out: Dict[str, Any] = {}
    baselines: Dict[str, Dict[str, np.ndarray]] = {}
    for arm, answers, synchronizes in (("no_sync", False, False),
                                       ("guarded", False, True),
                                       ("hazard", True, True)):
        driver = _build_driver(spec)
        plans, state = _weld_plan_for(driver)
        driver._fast_path = Shim(plans, answers_the_sync_consult=answers)  # noqa: SLF001
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
    electric = tuple(f"{stem}{axis}" for stem in ("D", "fu_D") for axis in "xyz")
    for arm in ("guarded", "hazard"):
        moved = compare(baselines["no_sync"], baselines[arm])
        arms_out[arm]["differing_volumes"] = moved
        arms_out[arm]["differing_electric_volumes"] = {
            name: n for name, n in moved.items() if name in electric}
    facts["arms"] = arms_out
    facts["the_guarded_arm_is_identical"] = not arms_out["guarded"]["differing_volumes"]
    facts["the_hazard_arm_diverges_in_D"] = bool(
        arms_out["hazard"]["differing_electric_volumes"])
    facts["what_this_cites"] = (
        "a product spanning update_H and step_D advances D and fu_D inside "
        "synchronize_magnetic_fields, whose backup list holds neither, so the "
        "restore cannot reach them. The containment rule in FastPathPlan.dispatch -- "
        "answer update_H_synchronize only where the plan at update_H stays inside "
        "SYNC_PATH_SLOTS -- is what stops it, and this leg is the measurement behind "
        "that rule for THIS product")
    facts["passed"] = bool(facts["D_is_in_neither_backup_list"]
                           and facts["the_guarded_arm_is_identical"]
                           and facts["the_hazard_arm_diverges_in_D"])
    return facts


# ---------------------------------------------------------------------------
# Device leg: the withdraw
# ---------------------------------------------------------------------------

def leg_withdraw(spec: Mapping[str, Any], steps: int) -> Dict[str, Any]:
    """The seam's one pass, on a device, with both of the campaign's null controls.

    The configuration this predicate REFUSES today -- a standing integrated electric
    withdraw -- driven anyway with the hoist wired, so what the refusal costs is
    measured rather than assumed. Three arrangements, each stepped IN LOCKSTEP with its
    own array-path reference and compared per COMPLETE step:

    ``hoisted``          the withdraw performed immediately before the launch, the
                         driver's own loop then no-oping. MUST be identical.
    ``not_hoisted``      the same launch with no hoist: the curl reads a D still
                         holding the previous step's standing dipole. MUST diverge.
    ``after_step_D``     the campaign's own null control, at the placement
                         ``withdraw_hoist`` refuses BY NAME. MUST diverge.

    THE REFERENCE IS PER ARM AND STEPPED BESIDE IT. A reference run to completion and
    compared against a subject mid-run diverges at step 1 for every arm, which says
    nothing -- that was this leg's first shape and it is why the two engines are now
    advanced together.
    """
    family = _family()
    # THE SEED IS NOT SCALED HERE, AND THAT IS THE LEG'S OWN MEASUREMENT RATHER THAN
    # AN OVERSIGHT. Every other driver-level leg lifts the state clear of the denormal
    # band with a power-of-two scale (:data:`SEED_SCALE_BITS`); this one compares a
    # subtraction of the SOURCE's standing dipole, whose magnitude is the source
    # amplitude. Against a state scaled by 2^80 that subtraction is below the float32
    # resolution of every word it touches and is swallowed whole -- measured
    # 2026-09-06, when all three arms came back identical and the two required null
    # controls silently stopped controlling anything. The fixture therefore runs at
    # the state's own amplitude and the budget is short enough that it stays clear of
    # the band.
    probe_driver = _build_driver(spec, integrated=True, scale_bits=0)
    sources = tuple(probe_driver._sources)  # noqa: SLF001
    standing = withdraw_hoist.standing_withdraws(sources)
    covered, reason = family.covers_fused_hd_pair(
        probe_driver.fields, probe_driver.pml, probe_driver.grid, sources)
    hoistable, hoist_reasons = withdraw_hoist.hoistable(
        probe_driver.fields, sources, span=family.REPLACES)

    arms_out: Dict[str, Any] = {}
    for arm in ("hoisted", "not_hoisted", "after_step_D"):
        reference = _build_driver(spec, integrated=True, scale_bits=0)
        driver = _build_driver(spec, integrated=True, scale_bits=0)
        if arm == "after_step_D":
            plans, state = _weld_plan_for(driver)
            electric = tuple(driver._sources)  # noqa: SLF001
            inner = plans["update_H"]

            def _after(_inner=inner, _fields=driver.fields, _electric=electric):
                _inner()
                for source in _electric:
                    getattr(source, "withdraw", lambda *_a: None)(_fields)

            plans = {"update_H": _after, "step_D": lambda: None}
        else:
            plans, state = _weld_plan_for(driver, hoisted=(arm == "hoisted"))
        driver._fast_path = Shim(plans)  # noqa: SLF001
        driver._fast_path_stale = False  # noqa: SLF001
        first: Optional[Dict[str, Any]] = None
        compared = 0
        for step in range(steps):
            reference.step()
            driver.step()
            cp.cuda.runtime.deviceSynchronize()
            moved = compare(_driver_state(reference), _driver_state(driver))
            compared = step + 1
            if moved:
                first = {"step": step + 1, "volumes": moved}
                break
        arms_out[arm] = {"first_divergence": first, "identical": first is None,
                         "steps_compared": compared,
                         "withdrawn_on_the_last_step": state.get("withdrawn"),
                         "launches": state.get("launches")}
    record = {
        "spec": spec["label"], "steps": steps,
        "predicate_refuses_this_row": not covered, "predicate_reason": reason,
        "predicate_names_the_withdraw": "standing integrated" in reason,
        "standing_withdraws": len(standing),
        "hoistable": bool(hoistable), "hoistable_reasons": list(hoist_reasons),
        "arms": arms_out,
        "campaign": dict(withdraw_hoist.MEASUREMENT),
        "what_this_licenses": (
            "the LeadingWithdrawPlan placement on this device against the array path, "
            "with both of the array-path campaign's null controls diverging. It does "
            "NOT flip HOISTS_THE_WITHDRAW: the product still refuses these rows by "
            "name, and the flag moves only together with INSTALLABLE and the wiring"),
    }
    record["passed"] = bool(
        record["predicate_refuses_this_row"] and record["predicate_names_the_withdraw"]
        and record["hoistable"] and standing
        and arms_out["hoisted"]["identical"]
        and not arms_out["not_hoisted"]["identical"]
        and not arms_out["after_step_D"]["identical"])
    return record


# ---------------------------------------------------------------------------
# The mutations
# ---------------------------------------------------------------------------

#: A SENTINEL rather than a literal anchor: the near-face masks are twelve separate
#: lines, so this mutation is applied by :func:`_drop_near_face_masks` with its own
#: count assertion instead of by a single ``needle``.
_DROP_NEAR_FACE_MASKS = "<the twelve near-face curl mask lines>"


def _drop_near_face_masks(source: str) -> str:
    """Remove every ``BC_METALLIC``/``BC_MIRROR_PERIODIC`` near-face clear.

    THE COUNT IS ASSERTED. Six metallic and six mirror-periodic ``== 0`` clears is
    what the certified ``step_D`` writes -- two per component, on the component's two
    shift-0 axes -- and a mutation that removed a different number would be a defect
    this gate cannot name.
    """
    kept, dropped = [], 0
    for line in source.splitlines(keepends=True):
        stripped = line.strip()
        if (stripped.startswith(("if (bc_x == BC_METALLIC", "if (bc_y == BC_METALLIC",
                                 "if (bc_z == BC_METALLIC",
                                 "if (bc_x == BC_MIRROR_PERIODIC",
                                 "if (bc_y == BC_MIRROR_PERIODIC",
                                 "if (bc_z == BC_MIRROR_PERIODIC"))
                and "== 0)" in stripped and stripped.endswith("curl = 0.0f;")):
            dropped += 1
            continue
        kept.append(line)
    if dropped != 12:
        raise SystemExit(
            f"the certified step_D body carries {dropped} near-face curl mask clears, "
            f"not 12; this mutation would not have landed as declared and would have "
            f"scored against a body it does not describe")
    return "".join(kept)


#: ``{name: {"old", "new", "why", "live_on"}}`` -- device-source rewrites, each of
#: which MUST make a scored fixture diverge. ``live_on`` names the fixtures whose
#: machinery the defect actually disables; an arm scored where its machinery is inert
#: comes back UNCAUGHT and looks like a hole.
DEVICE_MUTATIONS: Dict[str, Dict[str, Any]] = {
    "foreign_tap_reads_stale_H": {
        "old": "    if (ia > 0) return resolve_H(comp, idx - stride, weld);",
        "new": ("    if (ia > 0) return (comp == 0 ? weld.Hx[idx - stride]\n"
                "                        : comp == 1 ? weld.Hy[idx - stride]\n"
                "                        : weld.Hz[idx - stride]);"),
        "why": ("the foreign tap loads the PRE-LAUNCH magnetic field instead of "
                "recomputing update_H there -- which is what an unfused step_D would "
                "read and is exactly one sub-step behind"),
        "live_on": "every fixture",
    },
    "foreign_tap_reads_B": {
        "old": "    if (ia > 0) return resolve_H(comp, idx - stride, weld);",
        "new": ("    if (ia > 0) return (comp == 0 ? weld.Bx[idx - stride]\n"
                "                        : comp == 1 ? weld.By[idx - stride]\n"
                "                        : weld.Bz[idx - stride]);"),
        "why": ("the tap reads the constitutive SOURCE rather than its result. Under "
                "mu = 1 outside the absorber H and B are close, so this is the "
                "plausible wrong answer rather than an obvious one"),
        "live_on": "every fixture",
    },
    "own_cell_reads_stale_H": {
        "old": "        float f1 = own_h[2];",
        "new": "        float f1 = Hz[idx];",
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
    },
    "foreign_history_reads_the_post_store_value": {
        "old": "    float prev = fw[idx];",
        "new": "    float prev = src;",
        "why": ("THE DETERMINISTIC TWIN of the in-place store: the recurrence reads "
                "the value the in-place arrangement would have handed it, with no "
                "race to depend on. This is the arithmetic consequence the racy "
                "mutation can only sometimes produce"),
        "live_on": "every fixture with an active absorber (kms != 0)",
    },
    # PREDICTED NULL, ARMED ANYWAY, AND ATTRIBUTED RATHER THAN ASSERTED. The certified
    # step_D masks the very curl component a near-face tap feeds, at the very plane
    # where the ghost is served (``_mask_non_owned_cells``; step_curl_kernels.py's
    # note 2 and the fold block below it, measured by
    # ``gate_cuda_folded_curl.py`` / ``results/cuda_folded_curl_2026-08-19/``), so NO
    # near-face ghost value reaches an output word of this kernel and this defect
    # cannot diverge. It is kept armed because it must still COMPILE, and
    # :func:`_ghost_attribution` is what turns "predicted null" into a measurement:
    # with the near-face masks dropped the SAME edit is required to change the answer.
    "ghost_replaced_by_the_wrapped_cell": {
        "old": ("    return (bc == BC_PERIODIC) ? resolve_H(comp, idx + (na - 1) * "
                "stride, weld) : 0.0f;"),
        "new": "    return resolve_H(comp, idx + (na - 1) * stride, weld);",
        "why": ("the metallic and mirror-periodic ghost -- an exact 0.0f -- is "
                "replaced by the constitutive recomputed at the WRAPPED cell. This is "
                "the boundary rule the whole lift exists to preserve"),
        "live_on": "walls_all, wall_x_only, wall_z_only, fold_y_metallic_termination",
        "predicted_null": (
            "the certified step_D zeroes the curl of the component this tap feeds at "
            "exactly the plane the near-face ghost is served on "
            "(stepping._mask_non_owned_cells, transcribed in step_curl_kernels.py's "
            "notes 2 and 3 and measured by gate_cuda_folded_curl.py / "
            "results/cuda_folded_curl_2026-08-19/), so no near-face ghost VALUE is "
            "observable in this kernel's output and this edit cannot diverge. The "
            "ghost_attribution sub-leg drops those masks and requires the same edit "
            "to change the answer, which is what makes the null a measurement"),
    },
    # THE MASK ITSELF, which is what makes the entry above null. It MUST be caught, or
    # the null has no explanation.
    "drop_the_near_face_curl_masks": {
        "old": _DROP_NEAR_FACE_MASKS,
        "new": None,
        "why": ("the near-face ownership masks -- the BC_METALLIC and "
                "BC_MIRROR_PERIODIC `== 0` clears -- are removed, so the curl at a "
                "wall plane is assembled from a ghost the cell does not own. This is "
                "the pass that makes ghost_replaced_by_the_wrapped_cell null, and a "
                "null whose cause is not itself measured is an assumption"),
        "live_on": "walls_all, fold_y_odd_phase, fold_y_metallic_termination",
    },
    "accumulations_regrouped": {
        "old": ("    float a = f[idx] + kps * src;"),
        "new": ("    float a = f[idx];\n    src = src;   // regrouped below"),
        "why": ("the two SEPARATE accumulations become one expression "
                "`f + (kps*src - kms*prev)`, which is a different float32 number "
                "because addition is not associative (constitutive_kernels.py note 1)"),
        "live_on": "every fixture with an active absorber",
        "also": [("    return a - kms * prev;",
                  "    return a + (kps * src - kms * prev);")],
    },
    "coefficient_index_is_the_threads_own": {
        "old": "    int k = idx % nz;\n    int j = (idx / nz) % ny;\n    int i = idx / (ny * nz);\n\n    // Hx <- Bx",
        "new": "    int k = 0;\n    int j = 0;\n    int i = 0;\n\n    // Hx <- Bx",
        "why": ("the recomputed cell's absorber coefficients are taken from the "
                "origin instead of from the recomputed cell -- the half-cell class of "
                "error, converged and smooth"),
        "live_on": "every fixture with an active absorber",
    },
}

#: The one armed edit required NOT to diverge. It replaces the own-cell registers with
#: reloads of the scratch words this thread has just stored -- the same value by a
#: different route -- so a byte-neutral control that DID diverge would mean the
#: harness, not the weld, decides the answer.
BYTE_NEUTRAL: Dict[str, str] = {
    "old": "        float f1 = own_h[2];",
    "new": "        float f1 = Hz_out[idx];",
}

#: HOST mutations: defects in the launcher and the plan rather than in the text.
HOST_MUTATIONS: Tuple[str, ...] = ("rotation_skipped", "scratch_aliased_to_storage",
                                   "swap_constitutive_sublattice")


def _mutated_source(name: str) -> str:
    family = _family()
    spec = DEVICE_MUTATIONS[name]
    if spec["old"] is _DROP_NEAR_FACE_MASKS:
        return _drop_near_face_masks(family.kernel_source())
    source = needle(family.kernel_source(), spec["old"], spec["new"])
    for old, new in spec.get("also", ()):
        source = needle(source, old, new)
    return source


def _ghost_attribution(spec: Mapping[str, Any], steps: int) -> Dict[str, Any]:
    """Why ``ghost_replaced_by_the_wrapped_cell`` is null, MEASURED rather than argued.

    Two mutated kernels are driven on a WALLED fixture: one with the near-face masks
    dropped, and one with those masks dropped AND the ghost replaced by the wrapped
    cell. If the exact ``0.0f`` were not being preserved, the two would agree; that
    they DIFFER is what attributes the null to the mask rather than to a lift that
    quietly stopped serving a ghost.
    """
    import cupy as cupy_module  # noqa: PLC0415

    family = _family()
    ghost = DEVICE_MUTATIONS["ghost_replaced_by_the_wrapped_cell"]
    masks_only = _drop_near_face_masks(family.kernel_source())
    both = needle(masks_only, ghost["old"], ghost["new"])
    states = {}
    for label, source in (("masks_dropped", masks_only),
                          ("masks_dropped_and_ghost_replaced", both)):
        kernel = cupy_module.RawKernel(source, family.KERNEL_NAME,
                                       options=family._COMPILE_OPTIONS)  # noqa: SLF001
        fields, grid, pml, dtdx = build(spec, "uniform", case_rng("ghost"))
        arrangement = make_arrangement("weld", fields, grid, pml, dtdx, kernel=kernel)
        for _ in range(steps):
            arrangement.step()
        cp.cuda.runtime.deviceSynchronize()
        states[label] = frozen(fields)
    moved = compare(states["masks_dropped"],
                    states["masks_dropped_and_ghost_replaced"])
    return {
        "spec": spec["label"], "steps": steps, "differing_volumes": moved,
        "the_ghost_is_observable_once_the_masks_are_dropped": bool(moved),
        "what_this_attributes": (
            "the exact 0.0f ghost IS being served by the welded shift; what makes the "
            "ghost mutation null on the shipped kernel is the certified near-face "
            "ownership mask, not a lift that stopped preserving the boundary rule"),
        "passed": bool(moved),
    }


def leg_mutation(specs: Sequence[Mapping[str, Any]], steps: int) -> Dict[str, Any]:
    """Every armed defect, each required to DIVERGE on at least one scored fixture."""
    import cupy as cupy_module  # noqa: PLC0415

    family = _family()
    results: List[Dict[str, Any]] = []
    for name in DEVICE_MUTATIONS:
        source = _mutated_source(name)
        try:
            kernel = cupy_module.RawKernel(source, family.KERNEL_NAME,
                                           options=family._COMPILE_OPTIONS)  # noqa: SLF001
            kernel.compile()
            compiled, compile_error = True, None
        except Exception as error:  # noqa: BLE001
            kernel, compiled, compile_error = None, False, repr(error)
        caught_on: List[str] = []
        for spec in specs:
            if not compiled:
                break
            record = drive(spec["label"], spec, "uniform", min(steps, 12),
                           modes=("array", "weld"), kernel=kernel)
            if not record["identical"]["weld"]:
                caught_on.append(spec["label"])
        entry = {"name": name, "kind": "device", "compiled": compiled,
                 "compile_error": compile_error,
                 "why": DEVICE_MUTATIONS[name]["why"],
                 "live_on": DEVICE_MUTATIONS[name]["live_on"],
                 "caught_on": caught_on, "caught": bool(caught_on)}
        entry["passed"] = bool(compiled and caught_on)
        if DEVICE_MUTATIONS[name].get("predicted_null"):
            # DECLARED AHEAD OF THE RUN, and recorded either way. A defect predicted
            # null that is CAUGHT is a finding about the prediction; one that is not
            # is the prediction holding, and the reason travels with it.
            entry["predicted_null"] = DEVICE_MUTATIONS[name]["predicted_null"]
            entry["prediction_held"] = not caught_on
            entry["passed"] = bool(compiled)
        if not entry["passed"] and name == "f_w_H_written_in_place":
            # PREDICTED NULL, RECORDED WITH ITS REASON rather than dropped. This defect
            # is a RACE: whether a foreign recompute reads a cell another block has
            # already written is a schedule, and a launch whose blocks happen to run in
            # index order can produce the shipped answer. Its DETERMINISTIC TWIN --
            # foreign_history_reads_the_post_store_value -- is armed for exactly this
            # reason and carries the arithmetic consequence with no schedule in it.
            entry["predicted_null"] = True
            entry["reason"] = (
                "schedule-dependent by construction; the deterministic twin "
                "foreign_history_reads_the_post_store_value carries the same "
                "arithmetic consequence and must be CAUGHT")
            entry["passed"] = compiled
        results.append(entry)

    # HOST MUTATIONS.
    for name in HOST_MUTATIONS:
        spec = specs[0]
        try:
            if name == "rotation_skipped":
                record = drive(spec["label"], spec, "uniform", min(steps, 12),
                               modes=("array", "weld"), rotate=False)
                caught = not record["identical"]["weld"]
                detail = {"first_divergence": record["first_divergence"]["weld"]}
            elif name == "scratch_aliased_to_storage":
                fields, grid, pml, _dtdx = build(spec, "uniform", case_rng("alias"))
                tables = family.fused_hd_pair_tables(pml)
                aliased = {vol: getattr(fields, vol) for vol in family.SCRATCH_VOLUMES}
                try:
                    family.assert_scratch_is_disjoint(fields, aliased, tables)
                    caught, detail = False, {"refused": None}
                except ValueError as error:
                    caught, detail = True, {"refused": str(error)[:400]}
            else:   # swap_constitutive_sublattice
                fields, grid, pml, _dtdx = build(spec, "uniform", case_rng("swap"))
                from meep_gpu.cuda_kernels import (  # noqa: PLC0415
                    constitutive_kernels, step_curl_kernels)
                half = constitutive_kernels.real_constitutive_tables(pml, True)
                curl = step_curl_kernels.real_pml_curl_tables(pml, False)
                mispaired = {"kms": {axis: curl[f"kms_{axis}"] for axis in "xyz"},
                             "sinv": {axis: curl[f"sinv_{axis}"] for axis in "xyz"},
                             "kps": {axis: half[f"kps_{axis}"] for axis in "xyz"}}
                reference, _g, ref_pml, _d = build(spec, "uniform", case_rng("swap"))
                dtdx = float(grid.dt / grid.dx)
                scratch = family.fused_hd_pair_scratch(fields)
                for _ in range(min(steps, 8)):
                    for pass_name in STEP_PASSES:
                        run_pass(pass_name, reference, ref_pml)
                    for pass_name in STEP_PASSES:
                        if pass_name == "step_D":
                            continue
                        if pass_name != "update_H":
                            run_pass(pass_name, fields, pml)
                            continue
                        family.launch_fused_hd_pair(
                            fields, scratch, mispaired,
                            family.fused_hd_pair_codes(grid), dtdx)
                        scratch = family.rotate_into_fields(fields, scratch)
                cp.cuda.runtime.deviceSynchronize()
                moved = compare(frozen(reference), frozen(fields))
                caught, detail = bool(moved), {"differing_volumes": moved}
            results.append({"name": name, "kind": "host", "compiled": True,
                            "caught": caught, "passed": caught, **detail})
        except Exception as error:  # noqa: BLE001
            results.append({"name": name, "kind": "host", "compiled": False,
                            "caught": False, "passed": False,
                            "error": repr(error)[:400]})
    attribution = _ghost_attribution(specs[1] if len(specs) > 1 else specs[0],
                                     min(steps, 8))
    return {"mutations": results, "denominator": len(results),
            "caught": sum(1 for entry in results if entry["caught"]),
            "predicted_null": [entry["name"] for entry in results
                               if entry.get("predicted_null")],
            "ghost_attribution": attribution,
            "passed": (all(entry["passed"] for entry in results)
                       and attribution["passed"])}


def leg_byte_neutral(spec: Mapping[str, Any], steps: int) -> Dict[str, Any]:
    """The one armed edit required NOT to diverge."""
    import cupy as cupy_module  # noqa: PLC0415

    family = _family()
    source = needle(family.kernel_source(), BYTE_NEUTRAL["old"], BYTE_NEUTRAL["new"])
    kernel = cupy_module.RawKernel(source, family.KERNEL_NAME,
                                   options=family._COMPILE_OPTIONS)  # noqa: SLF001
    record = drive(spec["label"], spec, "uniform", min(steps, 12),
                   modes=("array", "weld"), kernel=kernel)
    return {"spec": spec["label"], "identical": record["identical"]["weld"],
            "first_divergence": record["first_divergence"]["weld"],
            "why": ("the own-cell register is replaced by a RELOAD of the scratch word "
                    "this thread has just stored -- the same value by a different "
                    "route. A float32 stored to global memory and loaded back is the "
                    "identity on the bits, so an edit that diverged here would mean "
                    "the harness rather than the weld decides the answer"),
            "passed": bool(record["identical"]["weld"])}


def leg_disarm(spec: Mapping[str, Any], steps: int) -> Dict[str, Any]:
    """The identical harness on the SHIPPED bytes, required not to diverge."""
    record = drive(spec["label"], spec, "uniform", min(steps, 12),
                   modes=("array", "weld"))
    return {"spec": spec["label"], "identical": record["identical"]["weld"],
            "first_divergence": record["first_divergence"]["weld"],
            "passed": bool(record["identical"]["weld"])}


# ---------------------------------------------------------------------------
# Device leg: the compiler was exercised, under THIS policy, on an empty cache
# ---------------------------------------------------------------------------

def _cupy_cache_snapshot() -> Dict[str, Any]:
    """``CUPY_CACHE_DIR`` and how many entries it holds RIGHT NOW. Never raises."""
    directory = os.environ.get("CUPY_CACHE_DIR", "")
    try:
        entries = len(os.listdir(directory)) if directory and os.path.isdir(directory) else 0
        error = None
    except Exception as exc:  # noqa: BLE001 - an unreadable cache is a recorded fact
        entries, error = None, repr(exc)
    return {"dir": directory, "entries": entries, "preexisting": bool(entries),
            "error": error}


def leg_compiler(policy: Optional[str], at_install: Mapping[str, Any]) -> Dict[str, Any]:
    """Did THIS process compile the kernels it ran, and did the policy reach the compiler?

    THE ZERO THAT WAS NOT A MEASUREMENT. The 2026-09-06 release campaign
    (``results/cuda_fused_hd_pair_2026-09-06``) recorded ``nvrtc_calls 0`` and
    ``ftz_removed 0`` on both legs and ``cache_preexisting: true`` on both: the strip
    counters are read into the artifact at INSTALL time, before the first kernel is
    built, so those zeros could not distinguish a run that compiled everything from a
    run that compiled nothing -- and the caches did hold binaries from earlier passes.
    A "both policies" claim whose keep leg may have executed binaries some other
    process compiled is not a both-policies claim, and this leg is what makes the
    difference a released fact rather than a reading of a directory's mtimes.

    FOUR CLAUSES, each measured after every in-process leg has run:

    1. **THE COMPILER WAS EXERCISED.** The NVRTC binary observer (installed before the
       policy, so it sits INNERMOST and sees the option tuple NVRTC was really given)
       recorded at least one compile.
    2. **EVERY BINARY THIS PROCESS RAN, IT COMPILED.** The cache directory held ZERO
       entries when the policy was installed. CuPy keys its disk cache above the strip
       seam, so a warm directory can serve the other policy's bytes with no NVRTC
       call to notice; an empty one cannot serve anything.
    3. **THE POLICY REACHED THE COMPILER.** Under ``keep`` the strip counter removed
       ``-ftz=true`` at least once, the count of calls it saw equals the observer's,
       and NO observed option tuple still carried ``-ftz=true``. Under ``flush`` the
       strip is not installed, so its counters read zero by construction, and EVERY
       observed tuple carried ``-ftz=true`` -- CuPy's native behaviour, which is the
       flush policy.
    4. **THE FAMILY'S OWN SOURCE WENT THROUGH NVRTC**, recorded rather than required:
       the emitted device string's digest appears among the observed sources. It is
       informative because ``compile_cache`` may hand NVRTC a wrapped string; clause 2
       already establishes that every executed binary was compiled here.
    """
    report = probe.nvrtc_binary_report()
    # THE LIVE COUNTERS SIT ONE LEVEL DOWN. ``probe.subnormal_policy_stamp`` wraps
    # ``subnormal_policy.policy_stamp()`` under ``"stamp"`` beside the environment it
    # was read in; ``nvrtc_calls`` and ``ftz_removed`` live in that inner record and
    # are summed from the strip's per-entry-point counters at call time, so read
    # here they are this process's whole run. Read off the outer record they are
    # ``None`` -- which the first run of this leg (2026-09-07, campaign
    # cuda_regate_2026-09-06_p9) reported as a failed clause 3 beside an observer
    # that had seen 52 compiles with no ``-ftz=true`` reaching NVRTC.
    outer = probe.subnormal_policy_stamp(_REPO_API)
    stamp = dict(outer.get("stamp") or {})
    stamp.setdefault("policy", outer.get("policy"))
    now = _cupy_cache_snapshot()
    family = _family()
    try:
        family_source = family.source_digest()
    except Exception as exc:  # noqa: BLE001 - recorded; clause 4 is informative
        family_source = f"unavailable: {exc!r}"
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
    out = {
        "policy_requested": policy,
        "policy_stamp": stamp.get("policy"),
        "counters_after_every_in_process_leg": counters,
        "observed": {key: report.get(key) for key in
                     ("nvrtc_calls_observed", "distinct_binaries", "distinct_sources",
                      "any_ftz_true_reached_nvrtc", "all_ftz_true_reached_nvrtc")},
        "observations": report.get("observations"),
        "cache": {"dir": at_install.get("dir"),
                  "entries_at_install": at_install.get("entries"),
                  "preexisting_at_install": at_install.get("preexisting"),
                  "entries_after_every_in_process_leg": now["entries"]},
        "family_source_sha256": family_source,
        "family_source_observed_at_nvrtc": any(
            entry.get("source_sha256") == family_source
            for entry in (report.get("observations") or [])),
        "checks": checks,
        "what_this_licenses": (
            "that the binaries this process executed were compiled by this process, "
            "under the installed policy, with the policy's option treatment observed "
            "at the NVRTC seam. It says nothing about the lift children, which run in "
            "their own interpreters against the same directory and are therefore "
            "served the binaries this process compiled or compile their own under "
            "the same installed policy"),
        "passed": all(checks.values()),
    }
    return out


# ---------------------------------------------------------------------------
# The corpus lift -- the census driver's battery hook, and the parent that spawns it
# ---------------------------------------------------------------------------

class _HostGrid:
    xp = np


def runtime_reasons() -> List[str]:
    """Why THIS process cannot evaluate this battery at all, or ``[]``.

    Named by the census's own contract: a row measured in an interpreter with no CuPy
    would score as a gap the engine created rather than as one the harness did.
    """
    if cp is None:
        return ["cupy is not importable in this interpreter"]
    try:
        cp.cuda.runtime.getDeviceCount()
    except Exception as error:  # noqa: BLE001
        return [f"no CUDA device reachable from this interpreter: {error!r}"]
    return []


def _child_progress(message: str) -> None:
    path = os.environ.get("MEEP_GPU_HD_GATE_PROGRESS")
    label = os.environ.get("MEEP_GPU_HD_GATE_LABEL", "?")
    line = f"[{time.strftime('%H:%M:%S')}] {label} {message}"
    print(line, flush=True)
    if path:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()


def evaluate(driver: Any, probe_module: Any) -> Dict[str, Any]:  # noqa: ARG001
    """The census driver's battery hook: the four-reference identity on ONE lifted row."""
    from meep_gpu.cuda_kernels import arms  # noqa: PLC0415

    family = _family()
    steps = int(os.environ.get("MEEP_GPU_HD_GATE_STEPS", STEPS))
    max_cells = os.environ.get("MEEP_GPU_HD_GATE_MAX_CELLS")
    sources = tuple(getattr(driver, "_sources", ()) or ())
    block: Dict[str, Any] = {
        "family": family.FAMILY, "steps_requested": steps, "n_sources": len(sources),
        "sources": [{"type": type(s).__name__,
                     "field_type": str(getattr(s, "field_type", "")),
                     "is_integrated": bool(getattr(s, "is_integrated", False)),
                     "withdraw_does_work": bool(
                         withdraw_hoist._withdraw_does_work(s))}  # noqa: SLF001
                    for s in sources]}
    covered, reason = family.covers_fused_hd_pair(
        driver.fields, driver.pml, driver.grid, sources)
    block["predicate_admits"] = bool(covered)
    block["predicate_reason"] = reason
    try:
        plan = arms.plan_step(fields=driver.fields, pml=driver.pml, grid=driver.grid,
                              sources=sources, fuse=True)
        block["composer_selected"] = dict(getattr(plan, "selected", {}) or {})
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
        return {"cuda_fused_hd_pair_gate": block}
    if max_cells and cells > int(max_cells):
        block.update({"driven": False,
                      "why_not_driven": (f"{cells} cells exceeds "
                                         f"MEEP_GPU_HD_GATE_MAX_CELLS={max_cells}; "
                                         f"refused rather than run partially")})
        return {"cuda_fused_hd_pair_gate": block}
    block["driven"] = True
    block.update(_drive_lifted_row(driver, steps))
    _child_progress(f"passed={block.get('passed')} words={block.get('words_compared')}")
    return {"cuda_fused_hd_pair_gate": block}


def _stored_volumes(fields: Any) -> Dict[str, Any]:
    """Every stored volume this engine allocates, LIVE (not copied), by name.

    DERIVED from ``Fields.reset``'s own text, the sibling Metal gate's rule: a volume
    added to the engine joins the comparison without an edit here, and a list here
    could go stale while looking complete. Private scratch is dropped by the
    leading-underscore rule -- each such buffer is assigned WHOLE before anything reads
    it, so no value in it survives to be read by the next step.
    """
    global _RESET_VOLUMES  # noqa: PLW0603 - read once from disk, then held
    if _RESET_VOLUMES is None:
        import re  # noqa: PLC0415

        text = (Path(_REPO_API) / "meep_gpu" / "fields.py").read_text(encoding="utf-8")
        match = re.search(r"\n    def reset\(self\).*?(?=\n    def )", text, re.S)
        if match is None:
            raise SystemExit("cannot find Fields.reset in fields.py; refusing to "
                             "guess the list of stored volumes")
        names = set(re.findall(r"self\.([A-Za-z_][A-Za-z0-9_]*)", match.group(0)))
        _RESET_VOLUMES = tuple(sorted(name for name in names - {"polarizations"}
                                      if not name.startswith("_")))
    out: Dict[str, Any] = {}
    for name in _RESET_VOLUMES:
        array = getattr(fields, name, None)
        if array is not None:
            out[name] = array
    for index, state in enumerate(getattr(fields, "polarizations", ()) or ()):
        for component, array in getattr(state, "P", {}).items():
            out[f"P[{index}].{component}"] = array
        for component, array in getattr(state, "P_prev", {}).items():
            out[f"P_prev[{index}].{component}"] = array
    return out


_RESET_VOLUMES: Optional[Tuple[str, ...]] = None


def _capture(driver: Any) -> Dict[str, Any]:
    """The whole of the state one complete step advances."""
    return {"volumes": {name: to_host(array).copy()
                        for name, array in _stored_volumes(driver.fields).items()},
            "dipoles": [getattr(source, "_applied_dipole", None)  # noqa: SLF001
                        for source in getattr(driver, "_sources", ())],
            # ``driver.time`` is a PROPERTY over ``step_count`` (driver.py:4480-4481)
            # and is restored with it; a snapshot that carried a second copy would be
            # a second source of truth for the same fact.
            "step_count": int(getattr(driver, "step_count", 0))}


def _restore(driver: Any, snapshot: Mapping[str, Any]) -> None:
    """Put a captured state back IN PLACE, so the two arrangements start level."""
    for name, array in _stored_volumes(driver.fields).items():
        saved = snapshot["volumes"].get(name)
        if saved is None:
            array.fill(0)
            continue
        array[...] = cp.asarray(np.ascontiguousarray(saved)) if isinstance(
            array, cp.ndarray) else np.ascontiguousarray(saved)
    for source, dipole in zip(getattr(driver, "_sources", ()), snapshot["dipoles"]):
        if dipole is not None:
            source._applied_dipole = dipole  # noqa: SLF001
    driver.step_count = snapshot["step_count"]


def _subnormal_words(host: np.ndarray) -> int:
    word = np.frombuffer(np.ascontiguousarray(host, dtype=np.float32).tobytes(),
                         dtype=np.uint32)
    return int(np.count_nonzero(((word & 0x7F800000) == 0) & ((word & 0x7FFFFF) != 0)))


def _drive_lifted_row(driver: Any, steps: int) -> Dict[str, Any]:
    """One lifted corpus row, driven through the driver's own consult order.

    ONE DRIVER, CAPTURED AND RESTORED, rather than two engines: a lifted row carries its
    own sources, monitors and material, and a second copy of it is a second lift with
    its own state. Per complete step the state is captured, each dispatched arrangement
    runs one ``FdtdDriver.step`` from that state and is captured and rolled back, the
    ARRAY path then runs the same step, and the run continues from the ARRAY result --
    so the reference is never a subject's own output and the three arrangements are
    independent one-step experiments from a common state.

    THREE ARRANGEMENTS, NOT TWO, AND THE THIRD IS WHAT MAKES A DIVERGENCE
    ATTRIBUTABLE. Measured 2026-09-06 on ``examples:diffracted_planewave.py``
    (``probe_cuda_hd_lift_autopsy.py``): at the first step the row's state reaches the
    float32 denormal band, the weld and the CERTIFIED SINGLES are byte-identical to
    each other at every one of the 601 differing words and BOTH differ from the array
    path -- the array path holding an exact +0.0 where both dispatched arrangements hold
    a subnormal near 3.9e-39. A leg that compared only the weld against the array path
    scored that as a divergence of the weld, which is not what it is. So:

    * ``weld vs the certified singles`` is the WELD's claim. A difference here fails the
      row, and it is the only thing that can: a defect in this weld cannot appear in the
      certified singles.
    * ``the dispatched arrangements vs the array path`` is a claim about this BACKEND on
      this row. Where the two dispatched arrangements agree with each other and differ
      from the array path, the step is recorded under
      ``backend_disagreement_steps`` WITH its subnormal census and is not scored
      against the weld.

    THE FLUSH PRECONDITION IS ASKED BEFORE THE STEP IS SCORED, and that ordering is the
    2026-09-06 repair. A lifted row starts at the engine's zeros and is driven by its own
    sources, so it enters the denormal band on its own schedule; the census must therefore
    run over ALL THREE post-step states -- the case above has ZERO subnormal words in the
    array state and 334 in the dispatched ones -- and it must gate the comparison for the
    step that produced them, not the next one.
    """
    family = _family()
    weld_plans, state = _weld_plan_for(driver)
    weld_shim = Shim(weld_plans)
    singles_shim: Optional[Shim] = None
    singles_error: Optional[str] = None
    try:
        singles = _certified_single_launchers(
            driver.fields, driver.grid, driver.pml,
            float(driver.grid.dt / driver.grid.dx))
        singles_shim = Shim({"update_H": singles["update_H"],
                             "step_D": singles["step_D"]})
    except Exception as error:  # noqa: BLE001
        singles_error = repr(error)[:400]

    # THE MONITORS ARE DETACHED FOR THE COMPARISON, and that is a statement about what
    # is being measured rather than a convenience. Each complete step is run up to three
    # times from one captured state, and a DFT or flux monitor would accumulate all of
    # them, leaving the row's own monitor output describing a run that never happened.
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
        before = _capture(driver)
        per_step_words = sum(words(value).size for value in before["volumes"].values())
        if step == 0:
            # THE REPLAY CONTROL, AND IT IS THE PRECONDITION OF EVERY COMPARISON BELOW.
            # This leg runs one complete driver step SEVERAL TIMES from one captured
            # state and compares the results; that is meaningful only if the step is a
            # FUNCTION of the state it starts from. It is not on every corpus row: a
            # CustomSource drawing a fresh np.random.randn() per evaluation gives each
            # replay a different current, so no two arrangements consume the same
            # source and byte identity across them is UNDEFINED for that row -- not
            # false, undefined.
            #
            # SO IT IS MEASURED RATHER THAN NAMED. The array path is run TWICE from the
            # same captured state and required to reproduce itself; a row that does not
            # is refused BY NAME with the volumes that moved, counted out of the
            # denominator, and never scored against the weld. This also proves, per
            # row, that the capture/restore covers everything a step depends on --
            # which is the other thing every comparison below rests on.
            try:
                driver._fast_path = None  # noqa: SLF001
                driver._fast_path_stale = False  # noqa: SLF001
                driver.step()
                cp.cuda.runtime.deviceSynchronize()
                replay_a = _capture(driver)
                _restore(driver, before)
                driver.step()
                cp.cuda.runtime.deviceSynchronize()
                replay_b = _capture(driver)
                _restore(driver, before)
            except Exception as exc:  # noqa: BLE001
                error = repr(exc)[:400]
                break
            moved = compare(replay_a["volumes"], replay_b["volumes"])
            if moved:
                not_replayable = {
                    "volumes": moved,
                    "reason": ("one complete driver step, run twice from ONE captured "
                               "state on the ARRAY PATH, did not reproduce itself. The "
                               "step is not a function of the state it starts from on "
                               "this row -- a source drawing fresh randomness per "
                               "evaluation is the shape that does this -- so no two "
                               "arrangements consume the same current and byte "
                               "identity ACROSS arrangements is undefined here. The "
                               "row is refused by name rather than scored"),
                }
                break
        try:
            driver._fast_path_stale = False  # noqa: SLF001
            driver._fast_path = weld_shim  # noqa: SLF001
            driver.step()
            cp.cuda.runtime.deviceSynchronize()
            weld_state = _capture(driver)
            _restore(driver, before)
            singles_state = None
            if singles_shim is not None:
                driver._fast_path = singles_shim  # noqa: SLF001
                driver.step()
                cp.cuda.runtime.deviceSynchronize()
                singles_state = _capture(driver)
                _restore(driver, before)
            driver._fast_path = None  # noqa: SLF001
            driver.step()
            cp.cuda.runtime.deviceSynchronize()
            array_state = _capture(driver)
        except Exception as exc:  # noqa: BLE001
            error = repr(exc)[:400]
            break
        # THE PRECONDITION, OVER ALL THREE STATES, BEFORE THIS STEP IS SCORED.
        census = {
            "array_path": sum(_subnormal_words(v)
                              for v in array_state["volumes"].values()),
            "weld": sum(_subnormal_words(v) for v in weld_state["volumes"].values()),
            "certified_singles": (None if singles_state is None else
                                  sum(_subnormal_words(v)
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
                     "volumes": against_singles,
                     "subnormal_census": census}
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


# ---------------------------------------------------------------------------
# The per-row child: the corpus lift, ON THE DEVICE
# ---------------------------------------------------------------------------
#
# WHY THIS GATE SPAWNS ITS OWN CHILD RATHER THAN THE CENSUS DRIVER'S.
# ``measure_predicate_coverage``'s children lift every row with
# ``meep_gpu.lift_simulation(sim, prefer_gpu=False)`` -- a NUMPY engine -- because a
# census answers PREDICATES, and the CUDA battery answers them through a shim that
# renames the array module (``cuda_predicate_battery._GridWithCupyBackend``). That is
# exactly right for a coverage number and useless for a byte claim: this leg has to
# LAUNCH the kernel, and a kernel cannot be launched against NumPy storage. So the
# lift machinery is imported and reused -- ``sweep_corpus_lift_parity.capture_simulation``
# for an example script and ``survey_meep_tests``'s child namespace for a test module,
# the same two the census uses -- and only the backend argument differs.


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
        block = record.get("cuda_fused_hd_pair_gate") or {}
        mark(f"measured={record.get('measured')} admits={block.get('predicate_admits')} "
             f"passed={block.get('passed')} steps={block.get('steps_compared')}")
    return finish()


def _load_jsonl(path: Path) -> List[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def lift_basis(results: Path) -> Tuple[List[dict], Dict[str, Any]]:
    """The rows the standing seam record puts in this product's two cells.

    DERIVED from the seam record's own per-backend arm selection -- the arms the census
    chose for each row -- never from a list here.
    """
    seam = results / SEAM_RECORD
    if not seam.is_dir():
        raise SystemExit(f"the lift leg needs {seam}")
    # THE MODULE A TESTS ROW LIVES IN IS THE CENSUS'S FACT, NOT THE SEAM RECORD'S. The
    # seam record carries the row's LABEL and the arms each backend selected; which
    # MEEP test module has to be imported and replayed to reproduce it is recorded by
    # the census this cell was cut over, and joining on it here is what keeps the
    # lift's basis derived rather than transcribed. A tests row with no module in
    # either record is REFUSED BY NAME below, never guessed at.
    census = results / CENSUS
    modules: Dict[str, str] = {}
    #: THE CASE ID TO REPLAY, WHICH IS NOT ALWAYS THE ONE THE LABEL CARRIES. A
    #: parametrised MEEP test's case name is produced by the `parameterized` package,
    #: and that package is not installed on the validation host -- the census ships a
    #: SHIM for it (`parity/meep_gpu/shim`) whose expansion names the cases
    #: `<method>__idx<N>` where the real package names them `<method>_<args>`. The
    #: census's own `match_param_rows.py` already joined the two, so the mapping is
    #: READ from `tests_param_matched.jsonl` (`row` is the real name, `case` the
    #: shim's) rather than reconstructed here.
    #:
    #: WITHOUT THIS THE CHILD MEASURES NOTHING AND SAYS SO. Measured 2026-09-06:
    #: `tests:TestEigCoeffs.test_binary_grating_oblique_0_0_0` came back with no gate
    #: block at all, because the child replayed the whole module and found no case by
    #: that name -- the shim had enumerated it as `..._oblique__idx0`. One row, and a
    #: row the leg claimed to drive.
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
    rows: List[dict] = []
    for record in _load_jsonl(seam / "h_to_d_seam.jsonl"):
        cuda = (record.get("arms") or {}).get("cuda") or {}
        cell = (cuda.get("update_H"), cuda.get("step_D"))
        if cell not in (CELL_ARMS, EXTRA_CELL_ARMS):
            continue
        rows.append({
            "label": record["label"], "leg": record["leg"], "row": record["row"],
            "module": record.get("module") or modules.get(record["label"]),
            "replay_case": replay_case.get(record["label"], record["row"]),
            "interpreter": sys.executable,
            "cell": list(cell),
            "withdraw_in_seam": bool((record.get("h_to_d_seam") or {})
                                     .get("withdraw_in_seam")),
            "would_be_bucket": cuda.get("would_be_bucket"),
        })
    facts = {
        "seam_record": SEAM_RECORD, "census": CENSUS,
        "primary_cell": list(CELL_ARMS), "extra_cell": list(EXTRA_CELL_ARMS),
        "rows_in_the_primary_cell": sum(1 for row in rows
                                        if row["cell"] == list(CELL_ARMS)),
        "rows_in_the_extra_cell": sum(1 for row in rows
                                      if row["cell"] == list(EXTRA_CELL_ARMS)),
        "rows_total": len(rows),
        "rows_with_a_standing_withdraw": sorted(row["label"] for row in rows
                                                if row["withdraw_in_seam"]),
        "module_join": {"source": f"{CENSUS}/tests*.jsonl",
                        "cases_resolved": len(modules),
                        "parametrised_cases_replayed_under_the_shim_name": {
                            label: case for label, case in replay_case.items()
                            if label in {row["label"] for row in rows}},
                        "tests_rows_without_a_module": sorted(
                            row["label"] for row in rows
                            if row["leg"] != "examples" and not row["module"])},
    }
    return rows, facts


def leg_lift(out_dir: Path, steps: int, max_cells: Optional[int], timeout: float,
             resume: bool, only: Optional[Sequence[str]]) -> Dict[str, Any]:
    """Every corpus row in the cell, re-lifted in its own interpreter and driven."""
    import measure_predicate_coverage as census  # noqa: PLC0415

    # ABSOLUTE, BEFORE ANYTHING IS BUILT FROM IT: the child runs with ``cwd`` set to
    # the lift workdir, so a relative artifact or progress path handed to it resolves
    # somewhere that does not exist -- and the child then dies AFTER the lift, with the
    # parent recording a right-shaped row that measured nothing.
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
    probe_path = Path(_REPO_API) / census.PROBE
    environment = dict(os.environ)
    environment.update({"KMP_DUPLICATE_LIB_OK": "TRUE", "MPLBACKEND": "Agg",
                        "PYTHONPATH": _REPO_API,
                        "MEEP_GPU_HD_GATE_STEPS": str(steps),
                        "MEEP_GPU_HD_GATE_PROGRESS": str(progress_log)})
    if max_cells is not None:
        environment["MEEP_GPU_HD_GATE_MAX_CELLS"] = str(max_cells)
    # THE CHILDREN'S WORKING DIRECTORY IS NOT EVIDENCE AND DOES NOT LIVE IN THE
    # ARTIFACT. A corpus script writes whatever it likes there -- one of them drops a
    # PNG -- and a file inside the artifact that the manifest rule cannot hash is a
    # file the record cannot pin. Same discipline as the compile cache: scratch goes
    # outside the directory a record digests.
    work = Path(tempfile.mkdtemp(prefix="cuda_hd_lift_"))
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
        environment["MEEP_GPU_HD_GATE_LABEL"] = row["label"]
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
        block = record.get("cuda_fused_hd_pair_gate") or {}
        log(f"lift {index}/{len(rows)} {row['label']}: "
            f"measured={record.get('measured')} "
            f"admits={block.get('predicate_admits')} driven={block.get('driven')} "
            f"passed={block.get('passed')} steps={block.get('steps_compared')} "
            f"words={block.get('words_compared')} ({time.time() - started:.1f} s)")
    # A JSON ARRAY, NOT JSONL, AND THE SUFFIX IS THE POINT. ``certification.json``'s
    # manifest rule hashes every ``*.json`` under an artifact directory and nothing
    # else, so evidence written under another suffix is evidence the record cannot
    # pin -- and ``test_certification_metadata`` refuses exactly that, in its own
    # words: "if one of them is evidence the rule is wrong, not the file".
    (lift_dir / "rows.json").write_text(
        json.dumps(measured, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8")

    blocks = {r["label"]: (r.get("cuda_fused_hd_pair_gate") or {}) for r in measured}
    admitted = sorted(label for label, b in blocks.items() if b.get("predicate_admits"))
    refused = sorted(label for label, b in blocks.items()
                     if b.get("predicate_admits") is False)
    refused_by_the_withdraw = sorted(
        label for label in refused
        if "standing integrated" in str(blocks[label].get("predicate_reason", "")))
    # A ROW THIS HOST COULD NOT LIFT IS NOT A ROW THE ENGINE REFUSED, and the two are
    # separated BY NAME rather than summed. The corpus's lift records were cut across
    # three interpreters on the authoring machine -- one of them carrying gdspy, one
    # carrying a sigma build -- and the validation host has ONE. A row whose script
    # imports a package that is not here dies in the capture, before any predicate is
    # asked; scoring that as a gap the weld created is the defect this track has
    # already had once (a census that scored rows measured where torch was absent as
    # coverage gaps). Each is named with the import error the child reported, counted
    # out of the leg's denominator, and reported -- and anything unmeasured for
    # ANOTHER reason still fails the leg, because that is a hole.
    #: What an ABSENT EXTERNAL DEPENDENCY looks like in a child's own words. Measured
    #: on this corpus rather than guessed: three rows name a missing Python package
    #: (`PyMieScatt`, `gdspy`) and one names a MEEP build feature this host's MEEP was
    #: not compiled with (`libGDSII`). All four die in the CAPTURE, before any
    #: predicate is asked. A row that fails for any other reason is still a hole.
    _ABSENT_DEPENDENCY = ("ModuleNotFoundError", "ImportError",
                          "must be configured/compiled with")

    def _unlifted_reason(record: Mapping[str, Any]) -> Optional[str]:
        if record.get("has_simulation"):
            # THE CAPTURE SUCCEEDED, so whatever failed afterwards is this engine's
            # or this gate's, and an import error inside it is not a host fact.
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
    # NOT A FAILURE AND NOT SWEPT UP: a row whose own step is not a function of the
    # state it starts from, measured by the leg's own replay control, is named with the
    # volumes that moved and counted out of the denominator.
    unreplayable = {label: blocks[label]["not_replayable"]["volumes"]
                    for label in driven if blocks[label].get("not_replayable")}
    passed_rows = sorted(label for label in driven if blocks[label].get("passed"))
    # THE ROWS THE PREDICATE MUST REFUSE BY NAME, restricted to the rows it was ASKED
    # about. A row this host could not lift was never put to the predicate at all, so
    # expecting a refusal from it would fail the leg for a package that is not here --
    # the same confusion the unlifted split exists to remove, one level up. Measured
    # 2026-09-06: `examples:mie_scattering.py` is BOTH a standing-withdraw row and a
    # row this host cannot lift (PyMieScatt), and without this the leg failed on it.
    expected_refused = sorted(set(facts["rows_with_a_standing_withdraw"])
                              - set(unlifted))
    if only:
        expected_refused = sorted(set(expected_refused) & set(blocks))
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
        # NOT A FAILURE, AND NOT SWEPT UP EITHER. On these rows the weld and the
        # certified singles agreed word for word and BOTH left the array path; that is
        # a finding about this backend's agreement with the array path on those rows,
        # it is named here with its step count, and the per-row payload carries the
        # subnormal census that goes with it.
        "backend_disagreement_rows": backend_rows,
        "steps_scored": sum(int(blocks[label].get("steps_scored") or 0)
                            for label in driven),
        "below_the_floor": below_floor, "errored": errored,
        "not_replayable_rows": unreplayable,
        "complete_driver_steps": sum(int(blocks[label].get("steps_compared") or 0)
                                     for label in driven),
        "words_compared": sum(int(blocks[label].get("words_compared") or 0)
                              for label in driven),
        "passed": bool(measured and not diverged and not errored and not unmeasured
                       and refused == expected_refused),
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
    parser.add_argument("--lift-only", default=None,
                        help="comma-separated row labels")
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
        # THE CHILD TAKES NO ``--out``: it writes ONE row's payload, not an artifact.
        # Required-ness is enforced here rather than by argparse so the two modes can
        # share one parser without the child having to pass a path it never uses.
        return _lift_child(args.lift_child, args.lift_child_target,
                           json.loads(args.lift_child_cases), args.lift_child_out,
                           args.lift_steps, args.lift_child_progress)

    if not args.out:
        parser.error("--out is required unless --lift-child is given")
    started = time.perf_counter()
    legs = tuple(args.legs.split(",")) if args.legs else ALL_LEGS
    results: Dict[str, Any] = {
        "gate": Path(__file__).name,
        "family": "cuda_fused_hd_pair",
        "seam": withdraw_hoist.SEAM,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "argv": list(argv if argv is not None else sys.argv[1:]),
        "legs_requested": list(legs),
        "steps": args.steps,
        "what_a_release_does_not_license":
            None,
    }
    if cp is None or args.no_device:
        results["device_mode"] = False
        results["status"] = "refused: no CuPy" if cp is None else "structural only"
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
    # THE CACHE AS THIS PROCESS FOUND IT, before its first compile. CuPy keys its disk
    # cache ABOVE the seam the keep policy's strip installs at, so a directory that
    # already held binaries could serve this run bytes some earlier process compiled
    # under a policy nothing here can read back; the ``compiler`` leg refuses to
    # release on a cache that was not empty here, and this is the number it reads.
    results["cupy_cache"] = _cupy_cache_snapshot()
    # THE MACHINE, NAMED BY THE PROCESS THAT RAN ON IT. A certification block names a
    # host, and a host typed from memory into a writer is a hypothesis -- the same
    # argument this track already makes for the device. `probe.device_info()` reports
    # the GPU and not the box, so the box is stamped here.
    results["host"] = platform.node()
    results["environment"] = probe.device_info()
    results["subnormal_policy_stamp"] = probe.subnormal_policy_stamp(_REPO_API)

    family = _family()
    results["what_a_release_does_not_license"] = family.WHAT_A_RELEASE_DOES_NOT_LICENSE
    results["source_sha256"] = family.source_digest()
    save(results, args.out)

    specs = list(SPECS) if args.product == "full" else [
        spec for spec in SPECS if spec["label"] in REDUCED_LABELS]
    by_label = {spec["label"]: spec for spec in SPECS}
    mutation_specs = [by_label[label] for label in MUTATION_SPEC_LABELS]

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
    run_leg("arbitration", lambda: leg_arbitration(by_label["periodic"]))

    if "product" in legs:
        cases: List[Dict[str, Any]] = []
        for spec in specs:
            for value_class in VALUE_CLASSES:
                record = drive(spec["label"], spec, value_class, args.steps,
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
        nonlinear = [r for r in cases if r.get("nonlinear")]
        results["product"] = {
            "cases": cases, "denominator": len(cases),
            "complete_driver_steps": sum(r["steps_compared"] for r in cases),
            "words_compared": sum(r["words_compared"] for r in cases),
            "the_band_class_really_contains_subnormals":
                bool(band) and all(r["operand_census"]["subnormals"] > 0
                                   for r in band),
            "the_uniform_class_contains_none":
                bool(uniform) and all(r["operand_census"]["subnormals"] == 0
                                      for r in uniform),
            # THE NONLINEAR CELL IS A FLOOR, NOT A BONUS: a product leg that scored no
            # chi2/chi3 case cannot release, so the two seam-instances the extra-arm
            # row credits can never again rest on the module identity alone.
            "nonlinear_cases": len(nonlinear),
            "nonlinear_fixtures": sorted({r["spec"] for r in nonlinear}),
            "every_uniform_nonlinear_case_moved_under_chi": all(
                r["nonlinear"]["chi_moved_words_in_one_array_step"] > 0
                for r in nonlinear if r["value_class"] == "uniform"),
            "every_nonlinear_case_was_admitted_through_the_extra_arm_row": all(
                r["nonlinear"]["admitted_through_the_extra_arm_row"] for r in nonlinear),
            "the_nonlinear_cell_was_driven": bool(nonlinear) and all(
                r["nonlinear"]["passed"] for r in nonlinear),
            "passed": (bool(cases) and all(r["passed"] for r in cases)
                       and bool(nonlinear)
                       and all(r["nonlinear"]["passed"] for r in nonlinear)),
        }
        save(results, args.out)

    run_leg("purity_ledger", lambda: leg_purity_ledger(specs))

    if "block_sizes" in legs:
        sweep: List[Dict[str, Any]] = []
        for spec in specs:
            for threads in BLOCK_SIZES:
                record = drive(spec["label"], spec, "uniform", min(args.steps, 12),
                               modes=("array", "weld"), threads=threads)
                record["threads"] = threads
                sweep.append(record)
                log(f"[block_sizes] {spec['label']:28s} b{threads:<5d} "
                    f"{'IDENTICAL' if record['identical']['weld'] else 'DIVERGED'}")
            results["block_sizes"] = {"cases": sweep}
            save(results, args.out)
        results["block_sizes"] = {
            "cases": sweep, "denominator": len(sweep),
            "block_sizes": list(BLOCK_SIZES),
            "passed": bool(sweep) and all(r["identical"]["weld"] for r in sweep)}
        save(results, args.out)

    run_leg("launch_structure",
            lambda: leg_launch_structure(by_label["periodic"], min(args.steps, 12)))
    run_leg("sync", lambda: leg_sync(by_label["periodic"], min(args.steps, 12)))
    run_leg("withdraw", lambda: leg_withdraw(by_label["periodic"], min(args.steps, 12)))
    run_leg("byte_neutral",
            lambda: leg_byte_neutral(by_label["walls_all"], args.steps))
    run_leg("mutation", lambda: leg_mutation(mutation_specs, args.steps))
    run_leg("disarm", lambda: leg_disarm(by_label["walls_all"], args.steps))
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
        "released": bool(results["verdict"]["passed"] and not results["verdict"]["legs_missing"]),
        "reasons": [name for name, value in clauses.items() if not value]
                   + [f"leg did not run: {name}"
                      for name in results["verdict"]["legs_missing"]],
        "what_it_licenses": (
            "one CUDA kernel measured byte-identical, per COMPLETE DRIVER STEP and as "
            "uint32 words over every stored volume, to four independent arrangements "
            "of the update_H -> step_D seam, at every block size, on the fixtures and "
            "corpus rows this record names. It licenses NO throughput claim, NO "
            "dispatch claim and NO composition claim: the family declares INSTALLABLE "
            "= False and the composer refuses to install it on every configuration, "
            "so a credited seam-instance is PREDICATE ADMISSION and the product "
            "executes nowhere"),
    }
    results["elapsed_s"] = round(time.perf_counter() - started, 1)
    save(results, args.out)
    log("=" * 78)
    for clause, value in clauses.items():
        log(f"  {'PASS' if value else 'FAIL'}  {clause}")
    log(f"VERDICT: {'PASS' if results['verdict']['passed'] else 'FAIL'}  "
        f"({results['elapsed_s']} s)")
    return 0 if results["verdict"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
