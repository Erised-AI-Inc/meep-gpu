"""LONG HORIZON and DETERMINISM stress for the shipped CUDA kernel families.

WHAT THIS IS FOR, AND WHY IT IS NOT A GATE
==========================================

The 26 gates under ``parity/meep_gpu/`` each establish, per family, byte
identity against ``stepping``'s array path as uint32 words, at ONE launch and at
``MULTI_STEP_BUDGET = 60``, under both float32 subnormal policies, over uniform
and subnormal-band value classes and Courants 0.5 and 0.35, with host and source
mutations scored CAUGHT or NULL CONFIRMED.

Two things that record does NOT establish are what this file measures, and both
are quoted from the tree's own evidence rather than invented here:

A. LONG HORIZON. Every gate stops at 60 launches; a real MEEP run is thousands
   of steps. ``meep_gpu/triton_kernels/no_pml.py:92`` records what happens past
   60 on a sibling track -- ``2d_plain`` at resolution 40, every state array
   compared bytewise after every one of 6000 whole ``FdtdDriver.step`` calls:
   "identical through step 66, FIRST DIFFERENT AT STEP 67", 28 floats of
   2,457,600, in Bx/By/Dz, the kernel writing ``0x004fbb9e`` where the array path
   writes ``0x00000000``. Its module constant says the same in one field:
   ``no_pml.py:241``, ``"first_divergent_step": 67``. Sixty is not enough, and
   that is measured on this project's own hardware, not argued.

D. DETERMINISM. Every gate leg is a single comparison against an oracle, so
   nothing has ever run the same case twice and asked whether the kernel agrees
   with ITSELF. That matters concretely:
   ``meep_gpu/triton_kernels/folded_fused_magnetic_pair.py:359-361`` documents the
   ownership mask as being about RACE-FREEDOM rather than about the final value --
   "the masks are what make the carry race-free -- a lane that merely discarded
   the value would still have READ a word another lane writes". A race that
   happens to resolve the same way is invisible to a single comparison against an
   oracle; two runs disagreeing is the only instrument that can see it.

So this file is a STRESS HARNESS, not a gate. It releases nothing, changes no
predicate and raises no cap. Its verdict field is ``findings``, not ``released``.

=============================================================================
WHAT IT MEASURES
=============================================================================

LEG "long": two fixtures built from the SAME digest seed step in lockstep -- one
driven by ``stepping``, one by the kernel -- for up to ``MAX_STEPS`` steps, with
every compared array read as uint32 words at each of :data:`CHECKPOINTS`. Per
checkpoint the row carries HOW MANY words differ and in WHICH array, plus the
index and the two words at the first differing position, so a divergence is
diagnosable from the artifact without a re-run. ``--bisect`` then re-runs the
pair to find the EXACT first divergent step inside the bracket the checkpoints
leave, which is what turns "somewhere between 60 and 100" into "67".

LEG "determinism": the SAME case is built and run N times (default
:data:`DETERMINISM_REPEATS`) and every repeat is compared against the FIRST, not
against the oracle. The oracle is not involved: this leg asks only whether the
kernel is a function of its inputs.

TWO LADDERS, AND THE SPARSE ONE HAS A MEASURED BLIND SPOT. By default the long
leg compares at the checkpoints only; ``--compare-every-step`` compares at every
step. They are not two settings of one instrument, they answer different
questions, and the difference was measured here rather than assumed: a one-ULP
perturbation planted at step 67 HEALS within six steps in four of the seven
families, because these recurrences are absorbing and a single ULP is rounded
away by the next ``* sinv``. A checkpoint ladder at (60, 100) is blind to that,
correctly reports IDENTICAL, and would report the same words for a kernel that
had no defect at all. The per-family survival is in
:func:`plant_late_persistent_perturbation`'s docstring; a run that used the
sparse ladder carries ``compare_every_step: false`` in every case so the reading
cannot be mistaken for the dense one.

=============================================================================
WHAT THE NumPy LEG ESTABLISHED, 2026-08-20 (and what it did not)
=============================================================================

Run on a laptop, ``--product full``, 26 labels x 2 courants, 5000 steps per long
case and 20 x 250 per determinism case:

* CLEAN: 52/52 long cases byte-identical at EVERY STEP to 5000 under
  ``--compare-every-step`` -- 260,000 step-wise word comparisons, not 468 sampled
  ones -- with 468 scored checkpoints and ZERO vacuous; every pair
  ``starts_identical``; the smallest per-checkpoint ``oracle_moved_fraction``
  anywhere was 0.4341 and the subnormal band was reached naturally (435 words at
  one checkpoint), so no case passed by not moving. 52/52 determinism cases with
  all 20 repeats in agreement over 4,940 repeat comparisons, and no case skipped
  for build nondeterminism. The 60-launch result the gates certified reproduces
  under this driver and holds eighty-three times further out ON THE
  TRANSCRIPTIONS. Whole campaign: 138 s on a laptop.
* ``late_persistent_perturbation`` (one field word, one ULP, from step 67 on):
  CAUGHT 26/26 on the sparse ladder and 26/26 dense; ``--bisect`` returned 67 in
  every case; and the harness's own "would a 60-step budget have seen it" field
  answered NO 26/26. That is the leg shown to find the exact shape this file
  exists for.
* ``late_word_perturbation`` (the same, once): CAUGHT 26/26 dense, first
  divergent step 67 everywhere; MISSED by the sparse ladder on the absorbing
  families. That is the blind spot above, measured.
* ``subnormal_flush_in_aux``: bit in 11/26 cases, first at steps 39, 40, 64, 66,
  269, 272 and 278 depending on family. ``conductive_B``'s at STEP 269 is the
  strongest number here -- the mechanism the sibling track attributes its real
  divergence to, arriving more than four times past the budget every certified
  gate stops at.
* ``order_dependent_write``: CAUGHT 26/26 by leg D, and carrying no expectation
  on leg A. ON SEVEN OF THE 26 CASES ONLY SOME REPEATS DISAGREED -- 7, 8, 10, 11
  and 12 of 19 -- a nondeterminism that resolves the same way much of the time,
  which is exactly why leg D's finding is a LOWER bound on nondeterminism and why
  20 repeats is a starting point rather than a derived number. On
  ``pml_curl_D/fold_Y_metallic`` a run of four repeats would have had a better
  than even chance of reporting agreement.
* ``deterministic_wrong_word``: CAUGHT 26/26 on leg A, NULL CONFIRMED 0/26 on
  leg D. Without this pairing, "leg D found nothing" would be indistinguishable
  from "leg D is not looking".

None of that is evidence about the shipped device bytes. See the closing section.

=============================================================================
WHAT WOULD MAKE EITHER LEG VACUOUS, AND THE FLOORS THAT REFUSE IT
=============================================================================

* ``starts_identical`` -- the two fixtures must be byte-identical on every
  compared array BEFORE step 1. Two builds that differ are two different
  problems, and their divergence at step 500 would be a fact about the fixture.
* ``moved_since_previous_checkpoint`` -- the fraction of compared words the
  ORACLE changed since the previous checkpoint, recorded PER CHECKPOINT and not
  once at the end. A zero-initialised constitutive leaves every word +0.0
  forever, so a deliberately wrong reference still reports IDENTICAL; half of one
  earlier gate's cases could not fail for exactly that reason. A checkpoint that
  moved nothing is marked vacuous and excluded from the verdict.
* ``finite_fraction`` -- once a recurrence has gone to NaN everywhere, both paths
  agree trivially and "identical" means nothing. Recorded per checkpoint.
* ``determinism_moved`` -- a determinism leg on a run that changed no state is
  20 repeats of "+0.0 == +0.0". Repeat 0 must have moved state by each
  checkpoint or the checkpoint is vacuous.
* THE PLANTED DEFECTS. A stress leg nobody has seen fail measures nothing; see
  :data:`PLANTED_DEFECTS`, and ``--plant`` to arm one. Two of them MUST be caught
  by one leg and MUST NOT be caught by the other, which is the only thing that
  separates a working comparator from one that has degenerated into failing
  everything.

THE DRIVE IS SINUSOIDAL, AND THAT IS A CORRECTION, NOT A PREFERENCE. Every gate
moves its sources between launches by ``_ADVANCE = 0.97`` so the auxiliary does
not settle. Over 60 launches that is right; over 5000 it is not.
:func:`measure_geometric_drive_death` measures where it stops being a drive, in
this interpreter, and the artifact carries the numbers: measured 2026-08-20 on
this laptop, the scale leaves the normal range at STEP 2868 and becomes a FIXED
POINT at STEP 3297 -- round-to-nearest pins the smallest subnormals, so it never
reaches zero, it simply stops changing. Either way the last third of a 5000-step
leg would run with
sources 38 orders of magnitude below the state, which is precisely the vacuity
this file exists not to have. :func:`drive_scale` is a bounded sinusoid instead,
applied as the SAME float32 scalar on both paths, so it cancels out of every
comparison and cannot be the reason two paths agree or differ.

=============================================================================
RUNNING IT
=============================================================================

Laptop, NumPy backend (no CUDA). The transcriptions are IMPORTED from the gates
that own them -- this file spells no kernel arithmetic of its own -- so what the
NumPy leg settles is whether the DRIVER can find a divergence, localize it and
fail. It compiles nothing and certifies nothing::

    PYTHONPATH=<repo> python -u \\
        parity/meep_gpu/stress_cuda_long_horizon.py --backend numpy \\
        --leg both --product reduced --out /tmp/stress_local

Device (ONE verified-empty GPU). Same file, same case table, same floors; the
only change is which of each family's two runners is bound -- the flag is the
whole difference. The cache dir MUST carry the policy token, because CuPy's disk
cache key is computed above the strip seam and a shared directory serves the
other policy's binaries (``probe:4358-4361``)::

    export TRITON_LIBCUDA_PATH="$HOME/triton_libcuda_stub"
    export LD_LIBRARY_PATH="$TRITON_LIBCUDA_PATH:$LD_LIBRARY_PATH"
    OUT=$HOME/stress_long_horizon_2026-08-20
    CUDA_VISIBLE_DEVICES=<n> CUPY_CACHE_DIR=$OUT/cupy_cache/ftz_stripped \\
    PYTHONPATH=<repo> \\
        python -u parity/meep_gpu/stress_cuda_long_horizon.py \\
        --backend cuda --leg both --product full --courants 0.35,0.5 \\
        --max-steps 5000 --repeats 20 --determinism-steps 250 \\
        --compare-every-step --bisect \\
        --subnormal-policy keep --out $OUT/keep

then the same again with ``--subnormal-policy flush`` into ``$OUT/flush`` and its
own ``CUPY_CACHE_DIR``.

COST, PROJECTED FROM A MEASURED DEVICE NUMBER rather than guessed. The certified
``cuda_cylindrical_constitutive_2026-08-20`` artifact ran on the GPU host's RTX A6000
at a median 0.0373 s per case over 96 cases, and each of those cases is 122
steps (1 + 60 oracle, 1 + 60 kernel) on grids of this size class -- 0.31 ms per
step pair, sync included. At that rate the command above is roughly 25-30 minutes
per policy, so about an hour of one GPU for both. See ``--product reduced`` for
the form that fits one GPU in well under an hour.

One flushed line per checkpoint (the progress-reporting rule). Rows are APPENDED to
``checkpoints.jsonl`` as they land and the full artifact ``stress.json`` is
rewritten atomically after every case, so an interrupted run keeps everything up
to the failure; re-running the same command SKIPS cases whose terminal row is
already in the ledger.

WHAT THIS FILE DOES NOT CLAIM
=============================

* Nothing here is a release clause. No predicate is read as a gate, no cap is
  raised, no ``released`` field is emitted.
* SCALE is a different dimension and is not measured here: these fixtures are the
  gates' own small ones, and the int32 index-range clause in the predicates is
  nowhere near its bound on any of them.
* The NumPy backend is not the shipped bytes. It cannot see a contraction the
  NVRTC guard exists for, it does not exercise the subnormal policy, and -- for
  leg D specifically -- SINGLE-THREADED NumPy HAS NO RACES. The determinism leg's
  NumPy run establishes that the comparator detects a value that is not a
  function of its inputs; it cannot establish that a device kernel has none.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

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

log = probe.log
to_host = probe.to_host

SEED = 20260820

#: Where the words are read and compared. 60 is in the ladder ON PURPOSE: it is
#: the budget every certified gate stops at (``MULTI_STEP_BUDGET`` in all 26), so
#: a row at 60 is the standing result reproduced under this driver and every row
#: past it is new. The ladder is geometric past 100 because a divergence that
#: exists is overwhelmingly likely to be a first-occurrence, and bracketing it to
#: within a factor of ~2.5 is enough for ``--bisect`` to finish it cheaply.
CHECKPOINTS: Tuple[int, ...] = (1, 10, 60, 100, 250, 500, 1000, 2500, 5000)

#: The number every gate in this tree stops at. Carried as a named constant so
#: the "a 60-step budget would have missed this" reading is computed from the
#: rows rather than asserted.
GATE_BUDGET = 60

#: Repeats in the determinism leg. Twenty is a starting point, not a derived
#: number: a race that resolves the same way 19 times in 20 is not distinguished
#: from no race, and this leg's finding is a LOWER bound on nondeterminism.
DETERMINISM_REPEATS = 20

#: Steps per determinism repeat, and where those repeats are compared. Shorter
#: than the long leg because the cost is R times the steps, and because a race is
#: a per-launch property: if it never fires in 250 launches x 20 repeats it is
#: not something 5000 launches of one repeat would have shown either.
DETERMINISM_STEPS = 250
DETERMINISM_CHECKPOINTS: Tuple[int, ...] = (1, 10, 60, 100, 250)

#: The sinusoidal drive's period, in steps. Chosen coprime to every checkpoint in
#: :data:`CHECKPOINTS` so no checkpoint lands on the same phase as another and a
#: "identical at 250 and 500" reading cannot be an artifact of both being sampled
#: at ``sin = 0``.
DRIVE_PERIOD = 37

#: The step at which :data:`PLANTED_DEFECTS`'s ``late_word_perturbation`` fires.
#: 67 is not arbitrary -- it is the step at which the sibling track's measured
#: divergence first appeared (``triton_kernels/no_pml.py:241``), so a harness
#: shown to catch a defect planted there is a harness shown to catch the one
#: divergence past 60 this project has actually seen.
LATE_DEFECT_STEP = 67


def drive_scale(step: int) -> np.float32:
    """The bounded float32 scalar this step's sources are set to.

    A SINUSOID, NOT A DECAY. See the module header: the gates' ``0.97`` per launch
    underflows to zero in float32 well inside a 5000-step leg, and
    :func:`measure_geometric_drive_death` measures where. The same float32 value
    multiplies the same base array on both paths, so it cancels out of every
    comparison and cannot be the reason two paths agree or differ.
    """
    return np.float32(math.sin(2.0 * math.pi * step / DRIVE_PERIOD))


def measure_geometric_drive_death(factor: float = 0.97) -> Dict[str, Any]:
    """Where does the gates' geometric drive stop being a drive?

    MEASURED IN THIS INTERPRETER, not quoted, and the first version of this
    function got the mechanism WRONG in a way worth leaving recorded: it looked
    for the step at which ``0.97**n`` reaches float32 +0.0, and there is no such
    step. Round-to-nearest makes the smallest subnormal a FIXED POINT --
    ``1.4e-45 * 0.97 = 1.358e-45`` is nearer to ``1.4e-45`` than to zero, so it
    rounds back to itself and the loop ran to its cap. The claim "it underflows to
    zero" was false; the substantive claim is not, and these are the two numbers
    that carry it:

    * ``steps_to_subnormal`` -- when the scale leaves the normal range. Past that
      the source is smaller than the smallest normal float while the state is
      O(1), so it contributes nothing any accumulation can round to.
    * ``steps_to_fixed_point`` -- when the scale stops CHANGING at all. Past that
      the sources are literally constant and every remaining launch is the same
      launch, which is the vacuity a multi-step leg exists not to have.

    Both are compared against ``max(CHECKPOINTS)``, because the reading that
    matters is whether they fall inside this file's horizon.
    """
    tiny_normal = np.float32(np.finfo(np.float32).tiny)
    value = np.float32(1.0)
    step = 0
    subnormal_at: Optional[int] = None
    fixed_at: Optional[int] = None
    while step < 100000:
        nxt = np.float32(value * np.float32(factor))
        step += 1
        if subnormal_at is None and nxt != np.float32(0.0) and abs(nxt) < tiny_normal:
            subnormal_at = step
        if nxt == value:
            fixed_at = step
            break
        value = nxt
    horizon = max(CHECKPOINTS)
    return {
        "factor": float(factor),
        "steps_to_subnormal": subnormal_at,
        "steps_to_fixed_point": fixed_at,
        "final_scale": float(value),
        "max_checkpoint": horizon,
        "dead_before_last_checkpoint": bool(
            subnormal_at is not None and subnormal_at < horizon),
        "why_this_file_does_not_use_it": (
            "past steps_to_subnormal the gates' 0.97-per-launch drive is 38 "
            "orders of magnitude below the state it is supposed to move, and "
            "past steps_to_fixed_point it does not change at all; a 5000-step "
            "leg driven that way spends its tail as a slow single-step leg"),
    }


# ---------------------------------------------------------------------------
# The NumPy-behind-CuPy's-name shim
# ---------------------------------------------------------------------------

class _NumpyWearingCupysName:
    """NumPy behind CuPy's ``__name__``.

    The predicates' first question is whether the backend is CuPy at all, and
    that is the one thing about the device library a laptop cannot supply.
    Everything else a fixture exercises -- the fold, the stored extent, the
    coefficient vector lengths, the dtype and contiguity -- is a real object
    either way. Same shim all 26 gates use, so the legs are commensurable.
    """

    __name__ = "cupy"

    def __getattr__(self, item):
        return getattr(np, item)


def backend_xp(backend: str):
    return cp if backend == "cuda" else _NumpyWearingCupysName()


def case_rng(*parts: Any) -> np.random.Generator:
    """SEEDED FROM A DIGEST, NEVER FROM ``hash()``.

    Python salts ``hash()`` of a tuple containing strings with ``PYTHONHASHSEED``
    -- measured on this tree, one key gave four distinct seeds in four
    interpreters -- so a hash-seeded case draws a different fixture every process
    and a failing case cannot be replayed from its own record. Every fixture in
    this file, including the SECOND fixture of a lockstep pair and every repeat of
    the determinism leg, is drawn from this one function so that "the same case"
    is a statement about bytes.
    """
    key = "|".join(str(part) for part in parts)
    return np.random.default_rng(
        SEED + int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "big"))


# ---------------------------------------------------------------------------
# The fixture: what a family hands the driver
# ---------------------------------------------------------------------------

class Fixture:
    """One built problem plus the two ways to step it.

    ``get`` is a CALLABLE rather than a captured array because the ADE family
    rotates its buffers: ``PolarizationState.update`` rebinds ``state.P`` and
    ``state.P_prev`` after every write (dispersion.py:689-691), so a fixture that
    captured the arrays at build time would compare the wrong buffers from step
    two -- exactly the class of long-horizon defect this file exists to find, and
    no reason to have it in the instrument.
    """

    __slots__ = ("family", "label", "backend", "fields", "layer", "grid",
                 "outputs", "aux", "get", "step_oracle", "step_kernel", "drive",
                 "facts")

    def __init__(self, family: str, label: str, backend: str, fields, layer, grid,
                 outputs: Sequence[str], aux: Sequence[str],
                 get: Callable[[str], Any],
                 step_oracle: Callable[[], None],
                 step_kernel: Callable[[], None],
                 drive: Callable[[int], None],
                 facts: Optional[Dict[str, Any]] = None) -> None:
        self.family = family
        self.label = label
        self.backend = backend
        self.fields = fields
        self.layer = layer
        self.grid = grid
        self.outputs = tuple(outputs)
        self.aux = tuple(aux)
        self.get = get
        self.step_oracle = step_oracle
        self.step_kernel = step_kernel
        self.drive = drive
        self.facts = facts or {}

    def read(self) -> Dict[str, np.ndarray]:
        """Host copies of every compared array, as they are RIGHT NOW."""
        return {name: np.ascontiguousarray(to_host(self.get(name))).copy()
                for name in self.outputs}


def structure_facts(grid) -> Dict[str, Any]:
    return {
        "shape": [int(n) for n in grid.shape],
        "cells": int(np.prod([int(n) for n in grid.shape])),
        "mirrored": [bool(grid.is_mirrored(a)) for a in range(3)],
        "metallic": [bool(grid.is_metallic(a)) for a in range(3)],
    }


def make_drive(fields, source_names: Sequence[str]) -> Callable[[int], None]:
    """Set this step's sources to ``base * drive_scale(step)``.

    THE BASE IS CAPTURED ONCE. A drive that read the CURRENT source and scaled it
    would be geometric again by another name, and would decay to zero for the same
    reason.
    """
    xp = fields.grid.xp
    base = {name: np.ascontiguousarray(to_host(getattr(fields, name))).copy()
            for name in source_names}

    def drive(step: int) -> None:
        scale = drive_scale(step)
        for name, values in base.items():
            getattr(fields, name)[...] = xp.asarray((values * scale).astype(np.float32))

    return drive


# ---------------------------------------------------------------------------
# The families
# ---------------------------------------------------------------------------
#
# EVERY ONE OF THESE IMPORTS ITS ARITHMETIC FROM THE GATE THAT OWNS IT. Not one
# line of kernel transcription is spelled here. That is the same rule
# gate_cuda_folded_constitutive states about its source mutations -- "borrowed
# from the shared probe so this gate cannot own a second spelling of a defect the
# certified record was cut against" -- applied to the transcription itself: a
# second copy of ``run_kernel_numpy`` in this file would drift from the gate's,
# and a stress run against a drifted transcription reports a fact about the copy.
#
# The families chosen are the ones with ACCUMULATING AUXILIARY STATE, because
# those are the places 5000 steps differ from 60:
#
#   pml_curl_{B,D}         the PML split field ``fu``          (gate_cuda_folded_curl)
#   pml_constitutive_{H,E} the constitutive auxiliary ``f_w``  (gate_cuda_folded_constitutive)
#   conductive_B           ``fu`` AND the conductive history ``f_cond`` (gate_cuda_conductive)
#   dispersive_pml_E       ``f_w`` under a pole chain          (gate_cuda_dispersive, arm 1)
#   ade_update_P           the P / P_prev / scratch ROTATION   (gate_cuda_dispersive, arm 3)
#   magnetic_pair          step_B + the driver's fold repairs + update_H, composed
#
# ``magnetic_pair`` is the composition the fused magnetic-pair kernels fuse
# (driver.py:3282-3289: ``step_B`` -> ``fill_symmetry_bc_B`` -> ``zero_metal_B``
# -> ``fill_folded_far_ghosts_B`` -> ``update_H``). It is carried for leg D
# because that is where the ownership/carry question lives -- and it is COMPOSED
# FROM THE TWO IMPORTED TRANSCRIPTIONS, so it does not reproduce the fusion's own
# carry. See ``not_established``: this file does not run a fused kernel.


def _family_pml_curl(sub_step: str):
    import gate_cuda_folded_curl as gate  # noqa: PLC0415

    specs = {spec["label"]: spec for spec in gate.FOLD_SPECS}

    def build(backend: str, label: str, courant: float) -> Fixture:
        spec = specs[label]
        xp = backend_xp(backend)
        rng = case_rng("pml_curl", sub_step, label, courant)
        fields, layer, grid = gate.build(xp, spec, courant)
        gate.seed_state(fields, grid, sub_step, "uniform", rng)
        dtdx = float(grid.dt / grid.dx)
        codes, kinds = gate.boundary_codes_for(grid, gate.LICENSED_SUBSTITUTION)
        tables = gate.tables_for(sub_step, layer)
        runner = (gate.run_kernel_cuda if backend == "cuda"
                  else gate.run_kernel_numpy)
        oracle = stepping.step_B if sub_step == "step_B" else stepping.step_D
        arrays = gate.SUB_STEP_ARRAYS[sub_step]
        return Fixture(
            family=f"pml_curl_{sub_step[-1]}", label=label, backend=backend,
            fields=fields, layer=layer, grid=grid,
            outputs=gate.outputs(sub_step), aux=tuple(arrays["aux"]),
            get=lambda name: getattr(fields, name),
            step_oracle=lambda: oracle(fields, layer),
            step_kernel=lambda: runner(sub_step, fields, tables, codes, dtdx),
            drive=make_drive(fields, arrays["sources"]),
            facts={"structure": structure_facts(grid), "dtdx": dtdx,
                   "boundary_codes": [int(c) for c in codes],
                   "resolved_kinds": list(kinds), "courant": courant,
                   "gate_module": gate.__name__})

    return build, tuple(specs)


def _family_pml_constitutive(side: str):
    import gate_cuda_folded_constitutive as gate  # noqa: PLC0415

    specs = {spec["label"]: spec for spec in gate.FOLD_SPECS}

    def build(backend: str, label: str, courant: float) -> Fixture:
        spec = specs[label]
        xp = backend_xp(backend)
        rng = case_rng("pml_constitutive", side, label, courant)
        fields, layer, grid = gate.build(xp, spec, courant)
        if side == "E":
            gate.install_epsilon(fields, grid, rng)
        gate.seed_state(fields, grid, side, "uniform", rng)
        tables = gate.tables_for(side, layer)
        runner = (gate.run_kernel_cuda if backend == "cuda"
                  else gate.run_kernel_numpy)
        oracle = stepping.update_H if side == "H" else stepping.update_E
        arrays = gate.SIDE_ARRAYS[side]
        return Fixture(
            family=f"pml_constitutive_{side}", label=label, backend=backend,
            fields=fields, layer=layer, grid=grid,
            outputs=gate.outputs(side), aux=tuple(arrays["aux"]),
            get=lambda name: getattr(fields, name),
            step_oracle=lambda: oracle(fields, layer),
            step_kernel=lambda: runner(side, fields, tables),
            drive=make_drive(fields, arrays["sources"]),
            facts={"structure": structure_facts(grid), "courant": courant,
                   "gate_module": gate.__name__})

    return build, tuple(specs)


def _family_conductive():
    """The three-history tail: ``fu`` and ``f_cond`` accumulating together.

    Held to ``step_B`` with an ACTIVE layer and every component conductive, which
    is the only arm on which both auxiliaries are live at once -- the arm the
    gate's own case product reaches on ``3d_pml_all_axes`` and the one whose
    long-horizon behaviour nothing has looked at.
    """
    import gate_cuda_conductive as gate  # noqa: PLC0415

    active = {spec["label"]: spec for spec in gate.SPECS
              if gate.spec_is_active(spec)}
    mask = (True, True, True)
    sub_step = "step_B"

    def build(backend: str, label: str, courant: float) -> Fixture:
        spec = active[label]
        xp = backend_xp(backend)
        rng = case_rng("conductive", label, courant)
        arm = gate.arm_of(spec, sub_step, "stored")
        kind = gate.layer_kind(spec)
        fields, layer, grid = gate.build(xp, spec, sub_step, mask, courant,
                                         "stored", "shipped", rng)
        names = gate.state_names(sub_step, arm, kind, mask)
        gate.seed_state(fields, grid, names, "uniform", rng)
        dtdx = float(grid.dt / grid.dx)
        codes, resolved, kinds, why = gate.boundary_codes_for(grid, "resolved")
        if why is not None:
            raise ValueError(f"{label}: {why}")
        runner = (gate.run_kernel_cuda if backend == "cuda"
                  else gate.run_kernel_numpy)
        outputs = gate.outputs(sub_step, arm, kind, mask)
        aux = tuple(n for n in outputs if n.startswith(("fu_", "f_cond_")))
        return Fixture(
            family="conductive_B", label=label, backend=backend,
            fields=fields, layer=layer, grid=grid,
            outputs=outputs, aux=aux,
            get=lambda name: getattr(fields, name),
            step_oracle=lambda: stepping.step_B(fields, layer),
            step_kernel=lambda: runner(spec, sub_step, arm, fields, layer,
                                       codes, dtdx),
            drive=make_drive(fields, gate.arm_arrays(sub_step, arm, kind)["sources"]),
            facts={"structure": structure_facts(grid), "dtdx": dtdx,
                   "arm": arm, "layer": kind, "courant": courant,
                   "boundary_codes": [int(c) for c in codes],
                   "resolved_kinds": list(kinds), "gate_module": gate.__name__})

    return build, tuple(active)


def _family_dispersive_pml_e(arity: Tuple[int, int, int] = (3, 2, 1)):
    """``f_w`` under a pole chain: the constitutive auxiliary plus a subtraction.

    THE ARITY IS UNEQUAL PER COMPONENT on purpose -- ``(3, 2, 1)`` -- because a
    chain-length defect is invisible when every component carries the same number
    of poles, which is the gate's own reason for sweeping arities at all.
    """
    import gate_cuda_dispersive as gate  # noqa: PLC0415

    specs = {spec["label"]: spec for spec in gate.SPECS if spec["pml"]}

    def build(backend: str, label: str, courant: float) -> Fixture:
        spec = specs[label]
        xp = backend_xp(backend)
        rng = case_rng("dispersive_pml_E", label, courant, arity)
        fields, layer, grid = gate.build(xp, spec, courant, arity, rng)
        gate.seed_state(fields, grid, "arm1", "uniform", arity, rng)
        tables = gate.tables_for(layer)
        runner = (gate.run_kernel_cuda if backend == "cuda"
                  else gate.run_kernel_numpy)

        def step_kernel() -> None:
            # THE PLAN IS RE-RESOLVED EVERY STEP, which is what a dispatch does.
            # A plan hoisted out of the loop is the ``arm3_stale_pole_plan``
            # hazard the sibling gate arms by name; carrying it here by accident
            # would make this leg measure the harness.
            runner("arm1", fields, layer, tables, gate.plan_for(fields))

        return Fixture(
            family="dispersive_pml_E", label=label, backend=backend,
            fields=fields, layer=layer, grid=grid,
            outputs=("Ex", "Ey", "Ez", "f_w_Ex", "f_w_Ey", "f_w_Ez"),
            aux=("f_w_Ex", "f_w_Ey", "f_w_Ez"),
            get=lambda name: getattr(fields, name),
            step_oracle=lambda: stepping.update_E(fields, layer),
            step_kernel=step_kernel,
            drive=make_drive(fields, ("Dx", "Dy", "Dz")),
            facts={"structure": structure_facts(grid), "arity": list(arity),
                   "courant": courant, "poles": len(fields.polarizations),
                   "gate_module": gate.__name__})

    return build, tuple(specs)


def _family_ade_update_p(arity: Tuple[int, int, int] = (2, 2, 2)):
    """The BUFFER ROTATION. ``P`` / ``P_prev`` / ``_scratch`` permute every step.

    This is the family whose long-horizon behaviour is least like its single
    launch: the arrays being compared are not the arrays that were compared last
    step, and a rotation defect is exact for the first component of the first
    launch and wrong from the second onwards (gate_cuda_dispersive:929-934).
    Sixty launches see that; whether five thousand see anything ELSE is the
    question, and nothing has asked it.
    """
    import gate_cuda_dispersive as gate  # noqa: PLC0415

    specs = {spec["label"]: spec for spec in gate.SPECS if not spec["pml"]}

    def build(backend: str, label: str, courant: float) -> Fixture:
        spec = specs[label]
        xp = backend_xp(backend)
        rng = case_rng("ade_update_P", label, courant, arity)
        fields, layer, grid = gate.build(xp, spec, courant, arity, rng)
        gate.seed_state(fields, grid, "arm3", "uniform", arity, rng)
        names: List[str] = []
        for index, state in enumerate(fields.polarizations):
            for component in state.driven():
                names.append(f"P{index}_{component}")
                names.append(f"Pprev{index}_{component}")

        def get(name: str):
            head, component = name.rsplit("_", 1)
            if head.startswith("Pprev"):
                return fields.polarizations[int(head[5:])].P_prev[component]
            return fields.polarizations[int(head[1:])].P[component]

        def step_kernel() -> None:
            if backend == "cuda":
                gate.run_arm3_cuda(fields, None)
            else:
                gate.run_arm3_numpy(fields, None)

        return Fixture(
            family="ade_update_P", label=label, backend=backend,
            fields=fields, layer=layer, grid=grid,
            outputs=tuple(names), aux=tuple(names),
            get=get,
            step_oracle=lambda: stepping.update_P(fields, layer),
            step_kernel=step_kernel,
            drive=make_drive(fields, ("Ex", "Ey", "Ez")),
            facts={"structure": structure_facts(grid), "arity": list(arity),
                   "courant": courant, "poles": len(fields.polarizations),
                   "rotating_buffers": len(names),
                   "gate_module": gate.__name__})

    return build, tuple(specs)


def _family_magnetic_pair():
    """``step_B`` -> the driver's three fold repairs -> ``update_H``, composed.

    THE ORACLE IS THE DRIVER'S OWN SEQUENCE, transcribed from driver.py:3282-3289
    with the source injections removed (there are no sources in this fixture)::

        step_B(fields, pml)
        fill_symmetry_bc_B(fields)
        zero_metal_B(fields)
        fill_folded_far_ghosts_B(fields)
        update_H(fields, pml)

    THE KERNEL LEG IS THE SAME SEQUENCE WITH THE TWO CURL/CONSTITUTIVE STEPS
    REPLACED by the imported transcriptions, and the three repairs left as the
    ENGINE's own functions on both legs -- they are not kernels, and a copy of
    them here would be a third spelling to drift.

    WHAT THIS IS AND IS NOT. It is the pair the fused magnetic-pair kernels fuse,
    run for thousands of steps and repeated for leg D. It is NOT the fused kernel:
    the fusion's own near-fill carry and its ownership mask
    (folded_fused_magnetic_pair.py:324-368) live INSIDE one launch and this
    composition has a repair pass between two launches instead. So a race in that
    carry cannot appear here, and ``not_established`` says so.
    """
    import gate_cuda_folded_curl as curl_gate  # noqa: PLC0415
    import gate_cuda_folded_constitutive as const_gate  # noqa: PLC0415

    specs = {spec["label"]: spec for spec in curl_gate.FOLD_SPECS}

    def build(backend: str, label: str, courant: float) -> Fixture:
        spec = specs[label]
        xp = backend_xp(backend)
        rng = case_rng("magnetic_pair", label, courant)
        # ONE fixture, seeded for BOTH halves: step_B writes B/fu_B from E, and
        # update_H writes H/f_w_H from B. Seeding the curl's names and then the
        # constitutive's leaves every array the pair touches drawn.
        fields, layer, grid = curl_gate.build(xp, spec, courant)
        curl_gate.seed_state(fields, grid, "step_B", "uniform", rng)
        const_gate.seed_state(fields, grid, "H", "uniform", rng)
        dtdx = float(grid.dt / grid.dx)
        codes, kinds = curl_gate.boundary_codes_for(
            grid, curl_gate.LICENSED_SUBSTITUTION)
        curl_tables = curl_gate.tables_for("step_B", layer)
        const_tables = const_gate.tables_for("H", layer)
        curl_run = (curl_gate.run_kernel_cuda if backend == "cuda"
                    else curl_gate.run_kernel_numpy)
        const_run = (const_gate.run_kernel_cuda if backend == "cuda"
                     else const_gate.run_kernel_numpy)

        def repairs() -> None:
            stepping.fill_symmetry_bc_B(fields)
            stepping.zero_metal_B(fields)
            stepping.fill_folded_far_ghosts_B(fields)

        def step_oracle() -> None:
            stepping.step_B(fields, layer)
            repairs()
            stepping.update_H(fields, layer)

        def step_kernel() -> None:
            curl_run("step_B", fields, curl_tables, codes, dtdx)
            repairs()
            const_run("H", fields, const_tables)

        outputs = tuple(curl_gate.outputs("step_B")) + tuple(const_gate.outputs("H"))
        return Fixture(
            family="magnetic_pair", label=label, backend=backend,
            fields=fields, layer=layer, grid=grid,
            outputs=outputs,
            aux=("fu_Bx", "fu_By", "fu_Bz", "f_w_Hx", "f_w_Hy", "f_w_Hz"),
            get=lambda name: getattr(fields, name),
            step_oracle=step_oracle, step_kernel=step_kernel,
            drive=make_drive(fields, ("Ex", "Ey", "Ez")),
            facts={"structure": structure_facts(grid), "dtdx": dtdx,
                   "boundary_codes": [int(c) for c in codes],
                   "resolved_kinds": list(kinds), "courant": courant,
                   "composition": ["step_B", "fill_symmetry_bc_B", "zero_metal_B",
                                   "fill_folded_far_ghosts_B", "update_H"],
                   "gate_modules": [curl_gate.__name__, const_gate.__name__]})

    return build, tuple(specs)


#: The case tables, per family. ``full`` is what a device campaign runs; ``reduced``
#: is what the laptop leg runs; ``smoke`` is one case, for a merge-bar check that
#: the driver still imports and steps.
def family_registry() -> Dict[str, Dict[str, Any]]:
    """Built lazily: importing eight gate modules costs a second and the merge
    bar should not pay it to collect a test that never calls this."""
    registry: Dict[str, Dict[str, Any]] = {}

    def add(name: str, builder, labels: Sequence[str],
            full: Sequence[str], reduced: Sequence[str]) -> None:
        missing = [label for label in tuple(full) + tuple(reduced)
                   if label not in labels]
        if missing:
            raise KeyError(f"{name}: {missing} not in {sorted(labels)}")
        registry[name] = {"build": builder, "labels": tuple(labels),
                          "full": tuple(full), "reduced": tuple(reduced)}

    build, labels = _family_pml_curl("step_B")
    add("pml_curl_B", build, labels,
        full=("unfolded_periodic", "unfolded_metallic", "fold_Y_periodic",
              "fold_Y_metallic", "fold_XY_mixed"),
        reduced=("unfolded_periodic", "fold_Y_periodic"))
    build, labels = _family_pml_curl("step_D")
    add("pml_curl_D", build, labels,
        full=("unfolded_periodic", "fold_Y_metallic", "fold_XY_mixed"),
        reduced=("fold_Y_metallic",))
    build, labels = _family_pml_constitutive("H")
    add("pml_constitutive_H", build, labels,
        full=("unfolded_periodic", "fold_Y_periodic", "fold_XYZ_metallic"),
        reduced=("fold_Y_periodic",))
    build, labels = _family_pml_constitutive("E")
    add("pml_constitutive_E", build, labels,
        full=("unfolded_metallic", "fold_Y_metallic", "fold_XYZ_periodic"),
        reduced=("fold_Y_metallic",))
    build, labels = _family_conductive()
    add("conductive_B", build, labels,
        full=("3d_pml_all_axes", "3d_pml_two_axes", "3d_pml_fold_Y"),
        reduced=("3d_pml_all_axes",))
    build, labels = _family_dispersive_pml_e()
    add("dispersive_pml_E", build, labels,
        full=("pml_2d_periodic", "pml_2d_folded_metallic", "pml_3d_metallic"),
        reduced=("pml_2d_periodic",))
    build, labels = _family_ade_update_p()
    add("ade_update_P", build, labels,
        full=("no_pml_all_periodic", "no_pml_1d_metallic_z", "no_pml_folded"),
        reduced=("no_pml_all_periodic",))
    build, labels = _family_magnetic_pair()
    add("magnetic_pair", build, labels,
        full=("unfolded_periodic", "fold_Y_metallic", "fold_XY_mixed"),
        reduced=("fold_Y_metallic",))
    return registry


#: 0.5 is exactly representable in float32 and 0.35 is not. The gates score both
#: because only the second can distinguish a contracted expression from an
#: uncontracted one; the long leg carries the inexact one by default because a
#: 5000-step run at two courants is twice the cost for a second draw of the same
#: question, and ``--courants`` re-opens it.
COURANTS: Tuple[float, ...] = (0.35, 0.5)
INEXACT_COURANT = 0.35


# ---------------------------------------------------------------------------
# The planted defects
# ---------------------------------------------------------------------------
#
# A STRESS LEG NOBODY HAS SEEN FAIL MEASURES NOTHING. Each of these wraps a
# fixture's ``step_kernel`` -- the ORACLE leg is never touched, so a planted
# defect is a defect in the thing under test and not in the reference.
#
# Two of them are scored the other way round on the two legs, and that pairing is
# the point: a battery on which every leg must fire scores identically whether
# the comparator works or has degenerated into failing everything.

def _flip_low_bit(fixture: Fixture, name: str, index: int = 0) -> None:
    array = fixture.get(name)
    host = np.ascontiguousarray(to_host(array)).copy()
    words = host.reshape(-1).view(np.uint32)
    words[index] ^= np.uint32(1)
    fixture.get(name)[...] = fixture.grid.xp.asarray(host)


def plant_late_word_perturbation(fixture: Fixture) -> Callable[[int], None]:
    """ONE WORD, ONE ULP, AT STEP 67 -- and nothing before it.

    This is the shape of the only past-60 divergence this project has measured
    (``triton_kernels/no_pml.py:241``: ``"first_divergent_step": 67``). It is
    invisible to every certified gate by construction, because every certified
    gate stops at 60.

    It is planted in the AUXILIARY rather than the field: the auxiliary feeds the
    next step's field through ``- fprev``, so a one-ULP change there propagates,
    whereas a one-ULP change in a damped field can round back to the same word and
    the instrument would report a fact about rounding.
    """
    inner = fixture.step_kernel
    target = fixture.aux[0] if fixture.aux else fixture.outputs[0]

    def stepped(step: int) -> None:
        inner()
        if step == LATE_DEFECT_STEP:
            _flip_low_bit(fixture, target)

    return stepped


def plant_late_persistent_perturbation(fixture: Fixture) -> Callable[[int], None]:
    """ONE FIELD WORD, ONE ULP, AT STEP 67 **AND EVERY STEP AFTER**.

    THIS EXISTS BECAUSE THE ONE-SHOT VERSION IS A TRANSIENT, and that was
    MEASURED on this harness before this defect was written rather than reasoned
    about. Flipping one auxiliary word once at step 67 and then comparing every
    step to 200 on the NumPy backend::

        pml_curl_B/unfolded_periodic      2 words at 68, gone by 72
        conductive_B/3d_pml_all_axes      1 word  at 68, gone by 69
        dispersive_pml_E/pml_2d_periodic  1 word  at 68, gone by 74
        pml_constitutive_{H,E}            gone at 68 -- ``fw[idx] = src`` overwrites
                                          the auxiliary WHOLESALE every step, so a
                                          perturbation there lives exactly one launch
        pml_curl_D, magnetic_pair         1-2 words, still present at 200

    These recurrences are absorbing: a single ULP is rounded away by the next
    ``* sinv``. So a checkpoint ladder at (60, 100) IS BLIND to a divergence that
    appears at 67 and heals by 72 -- which is a property of the instrument, is
    reported in ``sparse_ladder_blind_to_transients``, and is why
    ``--compare-every-step`` exists.

    The real divergence this file is built to find is not a transient: the sibling
    record says its spatial spread SATURATES from step 1500 at a median 49.09% of
    stored floats (``no_pml.py:104-107``). This defect has that shape -- it arms at
    67 and stays armed -- and it is planted in the FIELD, which every family
    read-modify-writes, rather than in the auxiliary, which two of them overwrite.
    """
    inner = fixture.step_kernel
    target = fixture.outputs[0]

    def stepped(step: int) -> None:
        inner()
        if step >= LATE_DEFECT_STEP:
            _flip_low_bit(fixture, target)

    return stepped


def plant_subnormal_flush_in_aux(fixture: Fixture) -> Callable[[int], None]:
    """Flush subnormal AUXILIARY words to +0.0 after every kernel step.

    THE PHYSICALLY MOTIVATED LATE DEFECT. The sibling track attributes its
    step-67 divergence to exactly this mechanism -- ``no_pml.py:101``: "The seed
    is CuPy's flush-to-zero meeting ``dtdx * (a difference)`` at the wavefront" --
    and a uniform-seeded fixture has no subnormal word at step 1, so WHEN this
    first bites is a measurement rather than a prediction. The artifact carries
    the first checkpoint at which it does.
    """
    inner = fixture.step_kernel
    names = fixture.aux or fixture.outputs
    tiny = np.float32(np.finfo(np.float32).tiny)

    def stepped(step: int) -> None:
        inner()
        for name in names:
            array = fixture.get(name)
            host = np.ascontiguousarray(to_host(array)).copy()
            subnormal = np.abs(host) < tiny
            if bool(subnormal.any()):
                host[subnormal] = np.float32(0.0)
                fixture.get(name)[...] = fixture.grid.xp.asarray(host)

    return stepped


def plant_order_dependent_write(fixture: Fixture) -> Callable[[int], None]:
    """A CARRY WHOSE DIRECTION IS NOT A FUNCTION OF THE INPUTS.

    ``folded_fused_magnetic_pair.py:359-361`` records the ownership mask as being
    about race-freedom rather than the final value: without it, the lane holding
    stored cell 0 and the lane holding stored cell 2 both write cell 0, and which
    write lands is decided by the scheduler. This plants that structure: cell 0
    and cell 2 of the first auxiliary exchange values in an order chosen from
    ``os.urandom`` -- a source outside the seeded fixture entirely, which is the
    defining property of the defect (a value that is not a function of the
    inputs), and one no oracle comparison can see because both orderings are
    plausible answers.

    IT EMULATES A RACE; IT IS NOT ONE. Single-threaded NumPy has no races. What
    this establishes is that leg D's comparator fires on run-to-run variation --
    the property that makes it the only leg that COULD see the device's.
    """
    inner = fixture.step_kernel
    name = fixture.aux[0] if fixture.aux else fixture.outputs[0]

    def stepped(step: int) -> None:
        inner()
        array = fixture.get(name)
        host = np.ascontiguousarray(to_host(array)).copy()
        flat = host.reshape(-1)
        if flat.size < 3:
            return
        if os.urandom(1)[0] & 1:
            flat[0] = flat[2]
        else:
            flat[2] = flat[0]
        fixture.get(name)[...] = fixture.grid.xp.asarray(host)

    return stepped


def plant_deterministic_wrong_word(fixture: Fixture) -> Callable[[int], None]:
    """WRONG, BUT THE SAME WRONG EVERY RUN. The determinism leg MUST NOT see it.

    One auxiliary word is negated after every kernel step. Leg A catches that at
    checkpoint 1; leg D must come back IDENTICAL, because every repeat is wrong in
    the same way. Without this control, "leg D found nothing" is consistent with
    a comparator that is not looking, and the two readings would be indistinguishable.
    """
    inner = fixture.step_kernel
    name = fixture.aux[0] if fixture.aux else fixture.outputs[0]

    def stepped(step: int) -> None:
        inner()
        array = fixture.get(name)
        host = np.ascontiguousarray(to_host(array)).copy()
        flat = host.reshape(-1)
        flat[0] = np.float32(-flat[0])
        fixture.get(name)[...] = fixture.grid.xp.asarray(host)

    return stepped


#: name -> (installer, must be caught by leg A, must be caught by leg D, why)
PLANTED_DEFECTS: Dict[str, Dict[str, Any]] = {
    "late_word_perturbation": {
        "install": plant_late_word_perturbation,
        # NO MUST-CATCH. Measured on this harness: in four of seven families the
        # one-shot flip is rounded away within six steps, so a ladder sampling at
        # 60 and 100 correctly reports IDENTICAL and an expectation of CAUGHT here
        # would be an expectation the arithmetic refuses. What this defect
        # measures is the ladder's blindness to a TRANSIENT, which is a real
        # property of the instrument and is reported as one.
        "long_must_catch": None, "determinism_must_catch": False,
        "dense_must_catch": True,
        "why": ("one auxiliary word, one ULP, at step 67 -- the step the sibling "
                "track's measured divergence first appeared at, and a step every "
                "certified gate stops seven launches short of. A TRANSIENT in the "
                "absorbing families: it must be caught under --compare-every-step "
                "and it may legitimately be missed by the sparse ladder"),
    },
    "late_persistent_perturbation": {
        "install": plant_late_persistent_perturbation,
        "long_must_catch": True, "determinism_must_catch": False,
        "dense_must_catch": True,
        "why": ("one FIELD word, one ULP, at step 67 and every step after: the "
                "shape the sibling track's real divergence has (its spread "
                "saturates rather than healing, no_pml.py:104-107). This is the "
                "defect the SPARSE ladder must catch, in every family"),
    },
    "subnormal_flush_in_aux": {
        "install": plant_subnormal_flush_in_aux,
        "long_must_catch": None, "determinism_must_catch": False,
        "why": ("subnormal auxiliary words flushed to +0.0 every step: the "
                "mechanism no_pml.py attributes its step-67 divergence to. WHEN "
                "it first bites is measured, not asserted, so it carries no "
                "must-catch expectation -- a run where no auxiliary word ever "
                "reaches the subnormal band is a NULL and the record says which"),
    },
    "order_dependent_write": {
        "install": plant_order_dependent_write,
        "long_must_catch": None, "determinism_must_catch": True,
        "why": ("a carry whose direction comes from os.urandom: a value that is "
                "not a function of the inputs. Leg D must catch it. Leg A MAY "
                "catch it and MAY not -- one of the two orderings can agree with "
                "the oracle -- so it carries no expectation there"),
    },
    "deterministic_wrong_word": {
        "install": plant_deterministic_wrong_word,
        "long_must_catch": True, "determinism_must_catch": False,
        "why": ("wrong, identically, every run: the NULL control that separates a "
                "determinism leg that is looking from one that fails everything"),
    },
}


# ---------------------------------------------------------------------------
# The comparison
# ---------------------------------------------------------------------------

def word_diff(reference: np.ndarray, got: np.ndarray) -> Dict[str, Any]:
    """Exact uint32-word comparison, WITH the first differing index.

    Never ``allclose``: ``-0.0 == 0.0`` and ``NaN != NaN`` both lie, and a leg
    that ran into the subnormal band deliberately puts signed zeros in the
    operands. The first index and the two words at it are what make a divergence
    diagnosable from the artifact without a re-run, which is the whole value of
    knowing WHERE.
    """
    a = np.ascontiguousarray(reference, dtype=np.float32).ravel().view(np.uint32)
    b = np.ascontiguousarray(got, dtype=np.float32).ravel().view(np.uint32)
    unequal = a != b
    differing = int(np.count_nonzero(unequal))
    out: Dict[str, Any] = {"differing_words": differing, "total_words": int(a.size)}
    if differing:
        first = int(np.flatnonzero(unequal)[0])
        out["first_index"] = first
        out["first_reference_word"] = f"0x{int(a[first]):08x}"
        out["first_kernel_word"] = f"0x{int(b[first]):08x}"
        ulp = np.abs(probe._ordered_key(a.view(np.float32))
                     - probe._ordered_key(b.view(np.float32)))
        out["max_ulp"] = int(ulp.max())
    return out


def compare_states(reference: Dict[str, np.ndarray],
                   got: Dict[str, np.ndarray]) -> Dict[str, Any]:
    per_array = {name: word_diff(reference[name], got[name]) for name in reference}
    differing = {name: row for name, row in per_array.items()
                 if row["differing_words"]}
    return {
        "identical": not differing,
        "differing_words": sum(row["differing_words"] for row in per_array.values()),
        "total_words": sum(row["total_words"] for row in per_array.values()),
        # WHICH ARRAY, not merely how many. "28 floats in Bx/By/Dz" is what made
        # the sibling track's step-67 record actionable; "not identical" would not
        # have been.
        "differing_arrays": sorted(differing),
        "per_array": per_array,
    }


def moved_fraction(before: Dict[str, np.ndarray],
                   after: Dict[str, np.ndarray]) -> float:
    moved = total = 0
    for name in before:
        a = np.ascontiguousarray(before[name], dtype=np.float32).ravel().view(np.uint32)
        b = np.ascontiguousarray(after[name], dtype=np.float32).ravel().view(np.uint32)
        moved += int(np.count_nonzero(a != b))
        total += int(a.size)
    return moved / total if total else 0.0


def finite_fraction(state: Dict[str, np.ndarray]) -> float:
    finite = total = 0
    for values in state.values():
        flat = np.ascontiguousarray(values, dtype=np.float32).ravel()
        finite += int(np.count_nonzero(np.isfinite(flat)))
        total += int(flat.size)
    return finite / total if total else 0.0


def subnormal_words(state: Dict[str, np.ndarray]) -> int:
    """How many compared words are in the subnormal band RIGHT NOW.

    Recorded per checkpoint because it is the operand class the one measured
    past-60 divergence in this project is attributed to, and because a
    ``subnormal_flush_in_aux`` leg that never saw one is a NULL rather than a pass.
    """
    tiny = np.float32(np.finfo(np.float32).tiny)
    count = 0
    for values in state.values():
        flat = np.ascontiguousarray(values, dtype=np.float32).ravel()
        count += int(np.count_nonzero((flat != 0.0) & (np.abs(flat) < tiny)))
    return count


# ---------------------------------------------------------------------------
# LEG A: the long horizon
# ---------------------------------------------------------------------------

def run_long_horizon_case(registry, family: str, backend: str, label: str,
                          courant: float, checkpoints: Sequence[int],
                          plant: Optional[str],
                          emit: Callable[[Dict[str, Any]], None],
                          dense: bool = False) -> Dict[str, Any]:
    """One family, one label: lockstep to the last checkpoint, comparing as we go.

    WHY TWO FIXTURES AND NOT ONE RESTORED. Comparing at checkpoint N by restoring
    a frozen state and re-running the oracle N times would be O(N^2) and would put
    twelve million array-path steps in a 5000-step leg. Two fixtures stepping in
    lockstep is O(N). The price is that the pair must be shown to START identical,
    which is what the ``starts_identical`` floor below does -- and it is a real
    floor, not a formality: a family whose build drew from ``rng`` in a different
    order on the second call would fail it here rather than reporting a fixture
    difference as a kernel divergence at step 500.
    """
    started = time.time()
    builder = registry[family]["build"]
    oracle_fx = builder(backend, label, courant)
    kernel_fx = builder(backend, label, courant)
    kernel_step: Callable[..., None] = kernel_fx.step_kernel
    if plant:
        kernel_step = PLANTED_DEFECTS[plant]["install"](kernel_fx)

    case: Dict[str, Any] = {
        "leg": "long_horizon", "family": family, "label": label,
        "backend": backend, "courant": courant, "plant": plant,
        "checkpoints": list(checkpoints),
        "compared_arrays": list(oracle_fx.outputs),
        "auxiliaries": list(oracle_fx.aux),
        "facts": oracle_fx.facts,
        "drive": {"kind": "sinusoid", "period_steps": DRIVE_PERIOD},
        "compare_every_step": bool(dense),
        "gate_budget_this_leg_exceeds": GATE_BUDGET,
        "rows": [],
    }

    # FLOOR 0. Two builds that are not byte-identical are two problems.
    start_oracle = oracle_fx.read()
    start_kernel = kernel_fx.read()
    start = compare_states(start_oracle, start_kernel)
    case["starts_identical"] = start["identical"]
    if not start["identical"]:
        case["skipped"] = ("the two fixtures were not byte-identical before step 1; "
                           "their divergence later would be a fact about the "
                           "fixture, not about the kernel")
        case["start_comparison"] = {k: v for k, v in start.items() if k != "per_array"}
        case["seconds"] = time.time() - started
        return case

    previous = start_oracle
    previous_step = 0
    first_divergent: Optional[int] = None
    last_clean = 0
    total = max(checkpoints)
    # DENSE MODE: every step compared, so a TRANSIENT that heals between two
    # checkpoints is seen. It is off by default because on a device it forces a
    # host read and a synchronise per step -- the sibling track paid exactly that
    # cost for its 6000-step run and it is what let it say "first different at
    # step 67" rather than "different somewhere in a bracket". These fields are
    # what the sparse ladder cannot supply.
    dense_first: Optional[int] = None
    dense_divergent_steps = 0
    dense_intervals: List[List[int]] = []
    open_interval: Optional[List[int]] = None
    checkpoint_set = set(checkpoints)

    for step in range(1, total + 1):
        oracle_fx.step_oracle()
        oracle_fx.drive(step)
        if plant:
            kernel_step(step)
        else:
            kernel_step()
        kernel_fx.drive(step)

        if dense and step not in checkpoint_set:
            if compare_states(oracle_fx.read(), kernel_fx.read())["identical"]:
                if open_interval is not None:
                    dense_intervals.append([open_interval[0], step - 1])
                    open_interval = None
            else:
                dense_divergent_steps += 1
                if dense_first is None:
                    dense_first = step
                if open_interval is None:
                    open_interval = [step, step]
            continue
        if step not in checkpoint_set:
            continue

        reference = oracle_fx.read()
        got = kernel_fx.read()
        comparison = compare_states(reference, got)
        moved = moved_fraction(previous, reference)
        row = {
            "leg": "long_horizon", "family": family, "label": label,
            "backend": backend, "courant": courant, "plant": plant,
            "step": step,
            "identical": comparison["identical"],
            "differing_words": comparison["differing_words"],
            "total_words": comparison["total_words"],
            "differing_arrays": comparison["differing_arrays"],
            "per_array": {name: entry
                          for name, entry in comparison["per_array"].items()
                          if entry["differing_words"]},
            # NON-VACUITY, PER CHECKPOINT. Not once at the end: a run that stopped
            # moving at step 200 reports IDENTICAL at 5000 for a reason that has
            # nothing to do with the kernel.
            "oracle_moved_since_step": previous_step,
            "oracle_moved_fraction": moved,
            "oracle_finite_fraction": finite_fraction(reference),
            "oracle_subnormal_words": subnormal_words(reference),
            "vacuous": moved == 0.0,
            "elapsed_seconds": time.time() - started,
        }
        # THE WINDOW ADVANCES. Left un-advanced, every checkpoint would report
        # movement since STEP 0 -- a number that only grows and can never refuse a
        # run that stopped moving at step 200, which is exactly the vacuity this
        # floor exists to catch.
        previous = reference
        previous_step = step
        case["rows"].append(row)
        emit(row)
        if comparison["identical"]:
            last_clean = step
            if dense and open_interval is not None:
                dense_intervals.append([open_interval[0], step - 1])
                open_interval = None
        else:
            if first_divergent is None:
                first_divergent = step
            if dense:
                dense_divergent_steps += 1
                if dense_first is None:
                    dense_first = step
                if open_interval is None:
                    open_interval = [step, step]
        log(f"[long] {family}/{label} c={courant}"
            f"{' plant=' + plant if plant else ''} step {step}/{total} "
            f"{'IDENTICAL' if comparison['identical'] else 'DIVERGED'} "
            f"words={comparison['differing_words']}/{comparison['total_words']} "
            f"arrays={','.join(comparison['differing_arrays']) or '-'} "
            f"moved={moved:.4f} finite={row['oracle_finite_fraction']:.3f} "
            f"subn={row['oracle_subnormal_words']} "
            f"({row['elapsed_seconds']:.1f} s)")

    case["first_divergent_checkpoint"] = first_divergent
    case["last_clean_checkpoint"] = last_clean
    if dense:
        if open_interval is not None:
            dense_intervals.append([open_interval[0], total])
        # A HEALED interval is one that closed before the run ended. Those are the
        # ones a sparse ladder can miss entirely, and counting them is how the
        # instrument reports its own blind spot instead of inheriting it silently.
        healed = [pair for pair in dense_intervals if pair[1] < total]
        case["dense"] = {
            "steps_compared": total,
            "first_divergent_step": dense_first,
            "divergent_steps": dense_divergent_steps,
            "divergent_intervals": dense_intervals,
            "healed_intervals": healed,
            "sparse_ladder_blind_to_transients": bool(
                healed and first_divergent is None),
        }
    scored = [row for row in case["rows"] if not row["vacuous"]]
    case["scored_checkpoints"] = len(scored)
    case["vacuous_checkpoints"] = len(case["rows"]) - len(scored)
    # THE READING THE 60-STEP BUDGET WOULD HAVE GIVEN, computed from the rows
    # rather than asserted. This is the number that says whether this leg found
    # anything the certified record could not have.
    within = [row for row in case["rows"] if row["step"] <= GATE_BUDGET]
    case["would_a_60_step_budget_have_seen_it"] = (
        None if first_divergent is None
        else bool(any(not row["identical"] for row in within)))
    if dense:
        # THE SAME QUESTION ASKED OF THE DENSE SCAN, and it is a different
        # question: a divergence that appears at 67 and heals by 72 is invisible
        # to the checkpoint ladder AND to any 60-launch budget, so the sparse
        # field above says None where this one says False.
        case["dense"]["would_a_60_step_budget_have_seen_it"] = (
            None if dense_first is None else bool(dense_first <= GATE_BUDGET))
    case["seconds"] = time.time() - started
    return case


def bisect_first_divergent_step(registry, family: str, backend: str, label: str,
                                courant: float, low: int, high: int,
                                plant: Optional[str]) -> Dict[str, Any]:
    """The EXACT first divergent step inside ``(low, high]``, by re-running the pair.

    TWO ASSUMPTIONS, BOTH CARRIED INTO THE ARTIFACT AS FIELDS RATHER THAN LEFT IN
    A COMMENT, because one of them is MEASURED FALSE on some cases:

    ``assumes_step_reproducibility`` -- a bisection re-runs, so it assumes
    stepping to m twice lands on the same words. On the NumPy backend that holds
    by construction. On a device it is exactly what LEG D measures, so a bisect
    number from a device run is void unless leg D came back clean for the same
    family; that dependence is the reason the two legs share this file.

    ``assumes_monotone_divergence`` -- a bisection finds the first m where the
    state DIFFERS assuming that once it differs it keeps differing. Measured
    2026-08-20 on this harness: it does not always. A one-ULP perturbation
    planted at step 67 flickers -- the damped recurrences round it away and the
    plant re-introduces it -- so on some cases this returns 81, 86 or 100 where
    the divergence FIRST appeared at 67. ``--compare-every-step`` is the
    instrument that gets that right, and when both are present
    ``dense.first_divergent_step`` is the authoritative number and this one is
    the bracket refinement. ``agrees_with_dense`` records whether they matched.
    """
    started = time.time()
    builder = registry[family]["build"]
    probes = 0

    def diverged_by(steps: int) -> bool:
        nonlocal probes
        probes += 1
        oracle_fx = builder(backend, label, courant)
        kernel_fx = builder(backend, label, courant)
        kernel_step: Callable[..., None] = kernel_fx.step_kernel
        if plant:
            kernel_step = PLANTED_DEFECTS[plant]["install"](kernel_fx)
        for step in range(1, steps + 1):
            oracle_fx.step_oracle()
            oracle_fx.drive(step)
            if plant:
                kernel_step(step)
            else:
                kernel_step()
            kernel_fx.drive(step)
        return not compare_states(oracle_fx.read(), kernel_fx.read())["identical"]

    lo, hi = low, high            # diverged_by(lo) is False, diverged_by(hi) True
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if diverged_by(mid):
            hi = mid
        else:
            lo = mid
        log(f"[bisect] {family}/{label} bracket ({lo}, {hi}] after {probes} probes")
    return {"first_divergent_step": hi, "bracket": [low, high], "probes": probes,
            "assumes_step_reproducibility": True,
            "assumes_monotone_divergence": True,
            "seconds": time.time() - started}


# ---------------------------------------------------------------------------
# LEG D: determinism
# ---------------------------------------------------------------------------

def run_determinism_case(registry, family: str, backend: str, label: str,
                         courant: float, repeats: int, steps: int,
                         checkpoints: Sequence[int], plant: Optional[str],
                         emit: Callable[[Dict[str, Any]], None]) -> Dict[str, Any]:
    """The same case, N times, every repeat compared against the FIRST.

    THE ORACLE IS NOT INVOLVED. This leg does not ask whether the kernel is right;
    it asks whether the kernel is a FUNCTION -- whether the same inputs give the
    same words. That is the only question a single comparison against a reference
    cannot answer, and the only one that can see a race whose two resolutions are
    both plausible answers.
    """
    started = time.time()
    builder = registry[family]["build"]
    baseline: Dict[int, Dict[str, np.ndarray]] = {}
    baseline_moved: Dict[int, float] = {}
    case: Dict[str, Any] = {
        "leg": "determinism", "family": family, "label": label,
        "backend": backend, "courant": courant, "plant": plant,
        "repeats": repeats, "steps": steps, "checkpoints": list(checkpoints),
        "rows": [],
    }
    first_divergent: Optional[int] = None
    divergent_repeats: set = set()

    # FLOOR: ONE REPEAT COMPARES NOTHING. "All repeats agree" with a single
    # repeat is the same vacuous pass as an oracle comparison against itself, and
    # it would be reported in the same words as a real twenty-repeat agreement.
    if repeats < 2:
        case["skipped"] = (f"--repeats {repeats}: a determinism leg needs at "
                           f"least two runs to compare; one repeat agrees with "
                           f"itself by construction")
        case["seconds"] = time.time() - started
        return case

    for repeat in range(repeats):
        fixture = builder(backend, label, courant)
        kernel_step: Callable[..., None] = fixture.step_kernel
        if plant:
            kernel_step = PLANTED_DEFECTS[plant]["install"](fixture)
        # THE BUILD IS PART OF "THE SAME CASE", so it is checked too. Without
        # this, a fixture whose construction was itself nondeterministic -- a
        # differently-ordered rng draw, an uninitialised allocation -- would make
        # every repeat disagree, and leg D would report a FIXTURE defect in the
        # words it reserves for a kernel race. The two are separated here, before
        # a single step is taken.
        this_start = fixture.read()
        if repeat == 0:
            case["compared_arrays"] = list(fixture.outputs)
            case["auxiliaries"] = list(fixture.aux)
            case["facts"] = fixture.facts
            start = this_start
        else:
            build_match = compare_states(start, this_start)
            if not build_match["identical"]:
                case["skipped"] = (
                    f"repeat {repeat} did not BUILD to the same bytes as repeat 0 "
                    f"({build_match['differing_words']} words differ in "
                    f"{build_match['differing_arrays']}); the fixture is "
                    f"nondeterministic and any later disagreement would be its, "
                    f"not the kernel's")
                case["build_nondeterminism"] = {
                    k: v for k, v in build_match.items() if k != "per_array"}
                case["seconds"] = time.time() - started
                return case
        for step in range(1, steps + 1):
            if plant:
                kernel_step(step)
            else:
                kernel_step()
            fixture.drive(step)
            if step not in checkpoints:
                continue
            state = fixture.read()
            if repeat == 0:
                baseline[step] = state
                # NON-VACUITY: repeat 0 must have MOVED state by this checkpoint,
                # or the twenty repeats agree because nothing happened.
                baseline_moved[step] = moved_fraction(start, state)
                row = {
                    "leg": "determinism", "family": family, "label": label,
                    "backend": backend, "courant": courant, "plant": plant,
                    "repeat": 0, "step": step, "identical": True,
                    "differing_words": 0,
                    "kernel_moved_fraction": baseline_moved[step],
                    "kernel_finite_fraction": finite_fraction(state),
                    "kernel_subnormal_words": subnormal_words(state),
                    "vacuous": baseline_moved[step] == 0.0,
                    "elapsed_seconds": time.time() - started,
                }
            else:
                comparison = compare_states(baseline[step], state)
                row = {
                    "leg": "determinism", "family": family, "label": label,
                    "backend": backend, "courant": courant, "plant": plant,
                    "repeat": repeat, "step": step,
                    "identical": comparison["identical"],
                    "differing_words": comparison["differing_words"],
                    "total_words": comparison["total_words"],
                    "differing_arrays": comparison["differing_arrays"],
                    "per_array": {n: r for n, r in comparison["per_array"].items()
                                  if r["differing_words"]},
                    "kernel_moved_fraction": baseline_moved[step],
                    "vacuous": baseline_moved[step] == 0.0,
                    "elapsed_seconds": time.time() - started,
                }
                if not comparison["identical"]:
                    divergent_repeats.add(repeat)
                    if first_divergent is None or step < first_divergent:
                        first_divergent = step
            case["rows"].append(row)
            emit(row)
        log(f"[determinism] {family}/{label} c={courant}"
            f"{' plant=' + plant if plant else ''} repeat {repeat + 1}/{repeats} "
            f"{'AGREES' if repeat not in divergent_repeats else 'DIFFERS'} "
            f"moved={baseline_moved.get(max(checkpoints), 0.0):.4f} "
            f"({time.time() - started:.1f} s)")

    case["first_divergent_checkpoint"] = first_divergent
    case["divergent_repeats"] = sorted(divergent_repeats)
    case["agreeing_repeats"] = repeats - 1 - len(divergent_repeats)
    case["all_repeats_agree"] = not divergent_repeats
    scored = [step for step, moved in baseline_moved.items() if moved > 0.0]
    case["scored_checkpoints"] = len(scored)
    case["vacuous_checkpoints"] = len(baseline_moved) - len(scored)
    if not scored:
        # TWENTY REPEATS OF "+0.0 == +0.0". A case whose kernel moved no state is
        # not a case on which twenty agreeing runs mean anything, and reporting it
        # as agreement is the same vacuity leg A's oracle_moved floor refuses.
        case["skipped"] = ("the kernel changed no state word at any checkpoint; "
                           "twenty repeats agree because nothing happened")
    case["seconds"] = time.time() - started
    return case


# ---------------------------------------------------------------------------
# The ledger: partial results as they land, and resumption
# ---------------------------------------------------------------------------

def case_key(leg: str, family: str, label: str, courant: float,
             plant: Optional[str]) -> str:
    return f"{leg}|{family}|{label}|{courant}|{plant or 'none'}"


class Ledger:
    """Append-only JSONL of rows, plus a set of case keys already finished.

    The progress-reporting rule in its strongest form for this file: a stress run is long by
    definition and one whose only signal is an exit code cannot be told from a
    hang. Every checkpoint is a line on disk the moment it lands, and a
    ``terminal`` line closes a case. Re-running the same command reads the
    terminal lines and SKIPS those cases, so an interrupted eight-hour run
    resumes at the case boundary rather than from the top.

    THE GRANULARITY IS THE CASE, NOT THE CHECKPOINT, and that is deliberate:
    resuming mid-case would mean serialising and restoring the fixture state,
    which is a second way to build a fixture and therefore a second thing that can
    differ from the first. The rows of an interrupted case are kept for diagnosis
    and the case is re-run.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        self.done: set = set()
        if os.path.exists(path):
            with open(path) as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # a torn last line from a kill; not fatal
                    if row.get("terminal"):
                        self.done.add(row["case_key"])
        self.handle = open(path, "a", buffering=1)  # line buffered

    def append(self, row: Dict[str, Any]) -> None:
        self.handle.write(json.dumps(row, default=str) + "\n")
        self.handle.flush()
        os.fsync(self.handle.fileno())

    def close_case(self, key: str, summary: Dict[str, Any]) -> None:
        self.append({"terminal": True, "case_key": key, "summary": summary})
        self.done.add(key)

    def close(self) -> None:
        self.handle.close()


def read_previous_artifact(artifact: str) -> Dict[str, Any]:
    """The last process's ``stress.json``, read BEFORE this one overwrites it.

    Order matters and it bit once: ``main`` saves the fresh header early (so an
    interrupted run still has an artifact naming its argv and environment), which
    would destroy exactly the file :func:`carry_forward` needs. It is read here,
    at the top, and held.
    """
    if not os.path.exists(artifact):
        return {}
    try:
        with open(artifact) as handle:
            return json.load(handle)
    except (json.JSONDecodeError, OSError):
        return {}


def carry_forward(previous: Dict[str, Any], ledger: Ledger,
                  leg: str) -> List[Dict[str, Any]]:
    """Cases a PREVIOUS process of this run already finished, read back for the artifact.

    WITHOUT THIS, A RESUMED RUN'S ARTIFACT DESCRIBES ONLY THE LAST PROCESS. The
    ledger keeps every row -- that is what makes the run resumable at all -- but
    ``stress.json`` is the deliverable a reader opens, and one that silently lost
    the first five of nine cases would report "4 cases scored" for a nine-case
    campaign. That is a record describing a run that did not happen, which is the
    same failure ``gate_provenance`` refuses on digests.

    Only cases whose TERMINAL row is in the ledger are carried: a case that was
    interrupted mid-flight is re-run, and its partial rows stay in the ledger for
    diagnosis.
    """
    key_leg = "long" if leg == "long_horizon" else "determinism"
    return [case for case in previous.get(leg, [])
            if case_key(key_leg, case["family"], case["label"], case["courant"],
                        case.get("plant")) in ledger.done]


def save(results: Dict[str, Any], out_path: str) -> None:
    """Atomic rewrite, with the bytes THIS process imported recorded first."""
    gate_provenance.stamp(results)
    tmp = out_path + ".tmp"
    with open(tmp, "w") as handle:
        json.dump(results, handle, indent=2, sort_keys=False, default=str)
    os.replace(tmp, out_path)


# ---------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------

def plan_cases(registry, families: Sequence[str], product: str,
               courants: Sequence[float]) -> List[Tuple[str, str, float]]:
    out: List[Tuple[str, str, float]] = []
    for family in families:
        entry = registry[family]
        if product == "smoke":
            labels = entry["reduced"][:1]
        elif product == "reduced":
            labels = entry["reduced"]
        else:
            labels = entry["full"]
        for label in labels:
            for courant in courants:
                out.append((family, label, courant))
    return out


def summarize(results: Dict[str, Any]) -> Dict[str, Any]:
    """FINDINGS, not a release. This file licenses nothing."""
    findings: List[str] = []
    notes: List[str] = []

    long_cases = results.get("long_horizon", [])
    det_cases = results.get("determinism", [])

    scored_long = [c for c in long_cases if not c.get("skipped")]
    diverged = [c for c in scored_long if c.get("first_divergent_checkpoint")]
    past_60 = [c for c in diverged
               if c["would_a_60_step_budget_have_seen_it"] is False]
    for case in diverged:
        arrays = sorted({name for row in case["rows"]
                         for name in row.get("differing_arrays", [])})
        findings.append(
            f"LONG HORIZON: {case['family']}/{case['label']} c={case['courant']}"
            f"{' plant=' + case['plant'] if case.get('plant') else ''} first "
            f"differs at checkpoint {case['first_divergent_checkpoint']} "
            f"(last clean {case['last_clean_checkpoint']}) in {arrays}"
            + (f"; exact first divergent step "
               f"{case['bisect']['first_divergent_step']}"
               if case.get("bisect") else "")
            + ("; A 60-STEP BUDGET WOULD HAVE MISSED IT"
               if case["would_a_60_step_budget_have_seen_it"] is False else ""))

    # DENSE-ONLY DIVERGENCES ARE FINDINGS TOO, and they would otherwise fall
    # through: a transient that appears and heals between two checkpoints leaves
    # ``first_divergent_checkpoint`` None, so the loop above says nothing about
    # the very case the dense scan exists to find.
    dense_only = [c for c in scored_long
                  if c.get("first_divergent_checkpoint") is None
                  and c.get("dense", {}).get("first_divergent_step") is not None]
    for case in dense_only:
        dense = case["dense"]
        findings.append(
            f"LONG HORIZON (DENSE ONLY -- the checkpoint ladder saw nothing): "
            f"{case['family']}/{case['label']} c={case['courant']}"
            f"{' plant=' + case['plant'] if case.get('plant') else ''} first "
            f"differs at STEP {dense['first_divergent_step']}, divergent on "
            f"{dense['divergent_steps']} of {dense['steps_compared']} steps, "
            f"healed intervals {dense['healed_intervals']}"
            + ("; A 60-STEP BUDGET WOULD HAVE MISSED IT"
               if dense.get("would_a_60_step_budget_have_seen_it") is False else ""))

    vacuous_long = sum(c.get("vacuous_checkpoints", 0) for c in scored_long)
    if vacuous_long:
        notes.append(f"{vacuous_long} long-horizon checkpoints were VACUOUS (the "
                     f"array path moved no word since the previous checkpoint) "
                     f"and are excluded from every reading above")

    scored_det = [c for c in det_cases if not c.get("skipped")]
    for case in scored_det:
        if case["all_repeats_agree"]:
            continue
        findings.append(
            f"DETERMINISM: {case['family']}/{case['label']} c={case['courant']}"
            f"{' plant=' + case['plant'] if case.get('plant') else ''} "
            f"{len(case['divergent_repeats'])} of {case['repeats'] - 1} repeats "
            f"disagree with repeat 0, first at checkpoint "
            f"{case['first_divergent_checkpoint']}")

    # THE PLANTED-DEFECT SCORECARD. A leg nobody has seen fail measures nothing,
    # and a leg that fails on everything measures nothing either. Each armed
    # defect is scored against BOTH expectations.
    scorecard: Dict[str, Any] = {}
    for name, spec in PLANTED_DEFECTS.items():
        armed_long = [c for c in scored_long if c.get("plant") == name]
        armed_det = [c for c in scored_det if c.get("plant") == name]
        row: Dict[str, Any] = {"why": spec["why"]}
        # THE TWO LADDERS ARE SCORED SEPARATELY, because they answer different
        # questions. A defect the sparse ladder misses and the dense one catches
        # is a fact about the LADDER; scoring them together would report it as a
        # fact about the comparator.
        sparse = [c for c in armed_long if not c.get("compare_every_step")]
        dense = [c for c in armed_long if c.get("compare_every_step")]
        if sparse:
            caught = sum(1 for c in sparse
                         if c.get("first_divergent_checkpoint") is not None)
            row["long_horizon"] = {
                "ladder": "sparse (checkpoints only)",
                "ran": len(sparse), "caught": caught,
                "must_catch": spec["long_must_catch"],
                "verdict": _verdict(caught, len(sparse), spec["long_must_catch"]),
            }
        if dense:
            caught = sum(1 for c in dense
                         if c.get("dense", {}).get("first_divergent_step") is not None)
            must = spec.get("dense_must_catch", spec["long_must_catch"])
            row["long_horizon_dense"] = {
                "ladder": "dense (every step)",
                "ran": len(dense), "caught": caught,
                "must_catch": must,
                "verdict": _verdict(caught, len(dense), must),
                "first_divergent_steps": sorted(
                    {c["dense"]["first_divergent_step"] for c in dense
                     if c["dense"]["first_divergent_step"] is not None}),
                "healed_intervals": [c["dense"]["healed_intervals"] for c in dense],
            }
        if armed_det:
            caught = sum(1 for c in armed_det if not c["all_repeats_agree"])
            row["determinism"] = {
                "ran": len(armed_det), "caught": caught,
                "must_catch": spec["determinism_must_catch"],
                "verdict": _verdict(caught, len(armed_det),
                                    spec["determinism_must_catch"]),
            }
        if row.keys() - {"why"}:
            scorecard[name] = row

    for name, row in scorecard.items():
        for leg in ("long_horizon", "long_horizon_dense", "determinism"):
            if leg in row and row[leg]["verdict"] not in (
                    "CAUGHT", "NULL CONFIRMED", "MEASURED"):
                findings.append(f"HARNESS: planted defect {name} is "
                                f"{row[leg]['verdict']} on leg {leg} "
                                f"({row[leg]['caught']}/{row[leg]['ran']})")

    return {
        "findings": findings,
        "notes": notes,
        "long_horizon_cases_scored": len(scored_long),
        "long_horizon_cases_diverged": len(diverged),
        "long_horizon_cases_diverged_dense_only": len(dense_only),
        "long_horizon_divergences_a_60_step_budget_would_miss": len(past_60),
        "long_horizon_dense_steps_compared": sum(
            c["dense"]["steps_compared"] for c in scored_long if c.get("dense")),
        "determinism_cases_scored": len(scored_det),
        "determinism_cases_with_disagreeing_repeats":
            sum(1 for c in scored_det if not c["all_repeats_agree"]),
        "planted_defects": scorecard,
        "max_checkpoint_reached": max(
            [row["step"] for case in scored_long for row in case["rows"]] or [0]),
        "claim": ("this run stepped the listed families past the 60-launch budget "
                  "every certified gate stops at, and repeated the listed families "
                  "to ask whether the kernel agrees with itself. It is a "
                  "measurement, not a release: no predicate is read as a gate and "
                  "no cap is raised"),
        "does_not_claim": _does_not_claim(results),
    }


def _verdict(caught: int, ran: int, must_catch: Optional[bool]) -> str:
    if not ran:
        return "NO LEGS"
    if must_catch is None:
        return "MEASURED"
    if must_catch:
        return "CAUGHT" if caught == ran else "PARTIAL" if caught else "UNCAUGHT"
    return "NULL CONFIRMED" if caught == 0 else "NULL VIOLATED"


def _does_not_claim(results: Dict[str, Any]) -> List[str]:
    out = [
        "nothing here is a release clause: no predicate is changed, no cap is "
        "raised, and there is no 'released' field",
        "SCALE is a different dimension. Every fixture here is one of the gates' "
        "own small ones; the largest is a few tens of thousands of cells against "
        "a corpus that reaches 11,245,000, and the int32 index-range clause in "
        "the predicates is nowhere near its bound on any of them",
        "the fused magnetic pair's OWN carry is not exercised. "
        "``magnetic_pair`` composes two separate launches with the engine's fold "
        "repairs between them (driver.py:3282-3289); the fusion's near-fill and "
        "its ownership mask live inside ONE launch "
        "(folded_fused_magnetic_pair.py:324-368) and no leg here runs that kernel",
        "BREADTH is a different dimension: this file drives the families with "
        "accumulating auxiliary state, not all 22, and not the off-diagonal row "
        "masks or the NAIVE expansion arm",
    ]
    if results.get("backend") == "numpy":
        out += [
            "the NumPy backend compiles nothing and certifies nothing. It is not "
            "the shipped bytes, it cannot see a contraction the NVRTC guard "
            "exists for, and it does not exercise either subnormal policy",
            "SINGLE-THREADED NumPy HAS NO RACES. The determinism leg's NumPy run "
            "establishes that the comparator fires on a value that is not a "
            "function of the inputs; it establishes nothing about whether a "
            "device kernel has such a value",
        ]
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backend", choices=("cuda", "numpy"), default="cuda")
    parser.add_argument("--legs", "--leg", dest="leg", choices=("long", "determinism", "both"),
                        default="both")
    parser.add_argument("--product", choices=("full", "reduced", "smoke"),
                        default="full")
    parser.add_argument("--families", default=None,
                        help="comma-separated subset; default is every family")
    parser.add_argument("--courants", default=None,
                        help=f"comma-separated; default {INEXACT_COURANT}")
    parser.add_argument("--steps", "--max-steps", dest="max_steps", type=int, default=max(CHECKPOINTS))
    parser.add_argument("--repeats", type=int, default=DETERMINISM_REPEATS)
    parser.add_argument("--determinism-steps", type=int, default=DETERMINISM_STEPS)
    parser.add_argument("--defect", "--plant", dest="plant", default=None,
                        choices=tuple(PLANTED_DEFECTS),
                        help="arm one planted defect; the scorecard scores it")
    parser.add_argument("--compare-every-step", action="store_true",
                        help=("compare at EVERY step, not only at the checkpoints. "
                              "Sees a transient that heals between two checkpoints; "
                              "costs a host read and a device synchronise per step"))
    parser.add_argument("--bisect", action="store_true",
                        help=("after a divergent checkpoint, re-run the pair to "
                              "find the EXACT first divergent step"))
    parser.add_argument("--subnormal-policy", choices=("keep", "flush"), default=None)
    # MEASURED ON DEVICE 2026-08-21, not a style choice: without this the "flush"
    # policy is UNATTAINABLE for the HOST executor ("MEEP is not imported") and
    # the run refuses at launch. I judged this flag a "gate concept" during the
    # coherence pass and left it off three of the four harnesses; the device
    # disproved that in the first minute. Any harness that INSTALLS a policy
    # needs it, because resolving the host half of that policy reads what MEEP
    # left the process doing.
    parser.add_argument("--import-meep-for-host-policy",
                        action="store_true")
    parser.add_argument("--out", required=True,
                        help="directory: stress.json + checkpoints.jsonl")
    args = parser.parse_args(argv)

    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    artifact = os.path.join(out_dir, "stress.json")
    # READ BEFORE THE FIRST save() OVERWRITES IT. See read_previous_artifact.
    previous_artifact = read_previous_artifact(artifact)
    ledger = Ledger(os.path.join(out_dir, "checkpoints.jsonl"))

    courants = ([float(v) for v in args.courants.split(",")] if args.courants
                else [INEXACT_COURANT])
    checkpoints = tuple(c for c in CHECKPOINTS if c <= args.max_steps)
    det_checkpoints = tuple(c for c in DETERMINISM_CHECKPOINTS
                            if c <= args.determinism_steps)

    results: Dict[str, Any] = {
        "harness": "stress_cuda_long_horizon",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": os.uname().nodename,
        "argv": list(argv if argv is not None else sys.argv[1:]),
        "backend": args.backend,
        "product": args.product,
        "questions": [
            "A: past the 60-launch budget every certified gate stops at, does any "
            "family's kernel path leave the array path -- and at which step, in "
            "which array, in how many words?",
            "D: run the same case twenty times, does the kernel agree with "
            "ITSELF? (the only leg that can see a race whose two resolutions are "
            "both plausible answers)",
        ],
        # ``gate_provenance.stamp`` will write ``canonical_verdict`` and find no
        # recognised verdict shape here. THAT IS CORRECT AND DELIBERATE: this
        # harness has no ``released``, ``passed`` or ``status`` field because it
        # releases nothing. A reader who sees UNREADABLE there should read
        # ``summary.findings``, not conclude the run failed.
        "verdict_shape": ("none: this is a measurement harness, not a gate. The "
                          "outcome is summary.findings; canonical_verdict is "
                          "expected to read UNREADABLE"),
        "gate_budget_every_certified_gate_stops_at": GATE_BUDGET,
        "sibling_track_measured_first_divergent_step": {
            "value": 67,
            "source": "meep_gpu/triton_kernels/no_pml.py:241",
            "what": ("2d_plain at resolution 40 on an RTX A6000, every state "
                     "array compared bytewise after every one of 6000 whole "
                     "FdtdDriver.step calls: identical through 66, 28 floats of "
                     "2,457,600 different at 67, in Bx/By/Dz"),
        },
        "geometric_drive_death": measure_geometric_drive_death(),
        "checkpoints": list(checkpoints),
        "determinism_checkpoints": list(det_checkpoints),
        "plant": args.plant,
        "planted_defect_catalog": {name: spec["why"]
                                   for name, spec in PLANTED_DEFECTS.items()},
    }

    if args.backend == "cuda":
        if cp is None:
            log("[fatal] --backend cuda but CuPy did not import")
            results["status"] = "refused: no CuPy"
            save(results, artifact)
            return 2
        # THE OBSERVER GOES IN BEFORE THE POLICY, and the order is the design
        # (probe:4208-4211): under 'keep' the policy's strip wraps this, so it
        # records the option tuple NVRTC was really given, post-strip. Installed
        # afterwards it would sit outside the strip and record the PRE-strip
        # tuple -- the one thing that would make the two policy legs look alike.
        results["nvrtc_observer"] = probe.install_nvrtc_binary_observer()
        if args.subnormal_policy:
            # BEFORE the install, as stress_cuda_scale.py does it: resolving the
            # host half of a policy reads what MEEP left the process doing, so the
            # import has to have happened first.
            if getattr(args, "import_meep_for_host_policy", False):
                results["meep_host_import"] = probe.import_meep_for_host_policy()
            results["subnormal_policy_install"] = \
                probe.install_subnormal_policy_for_run(args.subnormal_policy, _REPO_API)
        results["environment"] = probe.device_info()
        results["subnormal_policy_stamp"] = probe.subnormal_policy_stamp(_REPO_API)
    else:
        results["environment"] = {
            "python": sys.version.split()[0], "numpy_version": np.__version__,
            "note": "NumPy backend: compiles nothing, certifies nothing"}
    save(results, artifact)

    log("[setup] building the family registry (imports the gate modules that own "
        "each transcription)")
    registry = family_registry()
    families = ([name.strip() for name in args.families.split(",")]
                if args.families else list(registry))
    unknown = [name for name in families if name not in registry]
    if unknown:
        log(f"[fatal] unknown families {unknown}; known: {sorted(registry)}")
        results["status"] = f"refused: unknown families {unknown}"
        save(results, artifact)
        return 2
    results["families"] = families
    plan = plan_cases(registry, families, args.product, courants)
    results["planned_cases"] = len(plan)
    log(f"[setup] {len(plan)} case(s) per leg over families {families}")

    if args.leg in ("long", "both"):
        cases: List[Dict[str, Any]] = carry_forward(
            previous_artifact, ledger, "long_horizon")
        results["long_horizon"] = cases
        if cases:
            log(f"[resume] carried {len(cases)} finished long-horizon case(s) "
                f"forward from the previous artifact")
        for index, (family, label, courant) in enumerate(plan, start=1):
            key = case_key("long", family, label, courant, args.plant)
            if key in ledger.done:
                log(f"[long] case {index}/{len(plan)} {key} ALREADY DONE, skipping")
                continue
            log(f"[long] case {index}/{len(plan)} {family}/{label} c={courant} "
                f"to step {max(checkpoints)}")
            case = run_long_horizon_case(registry, family, args.backend, label,
                                         courant, checkpoints, args.plant,
                                         ledger.append, dense=args.compare_every_step)
            if case.get("dense", {}).get("first_divergent_step"):
                log(f"[long] {family}/{label} DENSE first divergent step "
                    f"{case['dense']['first_divergent_step']}, "
                    f"{case['dense']['divergent_steps']} of "
                    f"{case['dense']['steps_compared']} steps divergent, "
                    f"healed intervals {case['dense']['healed_intervals']}")
            if args.bisect and case.get("first_divergent_checkpoint"):
                case["bisect"] = bisect_first_divergent_step(
                    registry, family, args.backend, label, courant,
                    case["last_clean_checkpoint"],
                    case["first_divergent_checkpoint"], args.plant)
                # WHEN BOTH INSTRUMENTS RAN, THEY ARE CHECKED AGAINST EACH OTHER.
                # A bisect that disagrees with a dense scan has met a
                # NON-MONOTONE divergence, and the dense number is the right one;
                # recording the disagreement is how the artifact says which.
                dense_first = case.get("dense", {}).get("first_divergent_step")
                if dense_first is not None:
                    case["bisect"]["dense_first_divergent_step"] = dense_first
                    case["bisect"]["agrees_with_dense"] = bool(
                        dense_first == case["bisect"]["first_divergent_step"])
                log(f"[bisect] {family}/{label} first divergent step "
                    f"{case['bisect']['first_divergent_step']}"
                    + ("" if dense_first is None else
                       f" (dense says {dense_first}"
                       f"{'' if dense_first == case['bisect']['first_divergent_step'] else ', NON-MONOTONE: the dense number is authoritative'})"))
            cases.append(case)
            ledger.close_case(key, {
                "first_divergent_checkpoint": case.get("first_divergent_checkpoint"),
                "skipped": case.get("skipped"),
                "seconds": case.get("seconds")})
            save(results, artifact)

    if args.leg in ("determinism", "both"):
        cases = carry_forward(previous_artifact, ledger, "determinism")
        results["determinism"] = cases
        if cases:
            log(f"[resume] carried {len(cases)} finished determinism case(s) "
                f"forward from the previous artifact")
        for index, (family, label, courant) in enumerate(plan, start=1):
            key = case_key("determinism", family, label, courant, args.plant)
            if key in ledger.done:
                log(f"[determinism] case {index}/{len(plan)} {key} ALREADY DONE, "
                    f"skipping")
                continue
            log(f"[determinism] case {index}/{len(plan)} {family}/{label} "
                f"c={courant} {args.repeats} repeats x "
                f"{args.determinism_steps} steps")
            case = run_determinism_case(registry, family, args.backend, label,
                                        courant, args.repeats,
                                        args.determinism_steps, det_checkpoints,
                                        args.plant, ledger.append)
            cases.append(case)
            ledger.close_case(key, {
                "all_repeats_agree": case.get("all_repeats_agree"),
                "divergent_repeats": case.get("divergent_repeats"),
                "seconds": case.get("seconds")})
            save(results, artifact)

    if args.backend == "cuda":
        # WHETHER THE TWO POLICY LEGS COMPILED TWO BINARIES OR ONE SERVED TWICE.
        # Both policies compile the same SOURCE, so a compile log agrees by
        # construction and settles nothing; only the compiler's own output does.
        results["nvrtc_binary_report"] = probe.nvrtc_binary_report()

    results["summary"] = summarize(results)
    results["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save(results, artifact)
    ledger.close()

    verdict = results["summary"]
    log(f"[summary] long cases scored={verdict['long_horizon_cases_scored']} "
        f"diverged={verdict['long_horizon_cases_diverged']} "
        f"(a 60-step budget would miss "
        f"{verdict['long_horizon_divergences_a_60_step_budget_would_miss']}) | "
        f"dense-only={verdict['long_horizon_cases_diverged_dense_only']} | "
        f"determinism cases scored={verdict['determinism_cases_scored']} "
        f"disagreeing={verdict['determinism_cases_with_disagreeing_repeats']}")
    for finding in verdict["findings"]:
        log(f"[summary]   - {finding}")
    for note in verdict["notes"]:
        log(f"[summary]   . {note}")
    # EXIT 0 EVEN WITH FINDINGS. This is a measurement harness, not a gate: a
    # divergence found past step 60 is the RESULT, not a failure of the run, and a
    # non-zero exit would make a successful campaign look like a crashed one to
    # whatever scheduler is watching. A non-zero exit is reserved for a harness
    # that could not measure -- a refused backend, an unknown family, or a planted
    # defect whose verdict says the comparator is not working.
    broken = [f for f in verdict["findings"] if f.startswith("HARNESS:")]
    return 3 if broken else 0


if __name__ == "__main__":
    raise SystemExit(main())
