#!/usr/bin/env python3
"""Native MPS byte gate for the Metal H->D weld: ``update_H`` welded into ``step_D``.

THE FOURTH SEAM'S FIRST DEVICE GATE ON THIS BACKEND, and the first Metal fused-pair
gate whose subject is a product the composer must NOT install. That shapes every leg:
the arithmetic is certified the way the sibling gates certify theirs, and the
arbitration -- who holds ``update_H`` -- is MEASURED here as a fact about what the
shipped composer builds, never asserted from a docstring.

WHAT IS BEING CERTIFIED. ``meep_gpu/metal_kernels/fused_hd_pair.py`` computes the
``update_H`` constitutive into launch-local SCRATCH, takes its own cell's magnetic
field from registers, RECOMPUTES every one of the curl half's six foreign taps from
pre-launch state, steps ``D``/``fu_D`` in place, and rotates the ``H``/``f_w_H``
bindings afterwards. The claim is per COMPLETE DRIVER STEP -- ``FdtdDriver.step``'s
own consult order, sources withdrawn and injected where the driver does it, fills and
wall clears where the driver runs them -- over a stated budget of steps, as uint32
WORDS over EVERY stored volume the engine allocates (primaries, split-field PML
auxiliaries, both ``f_w`` histories; never ``allclose``, because ``-0.0 == 0.0``
lies), against FOUR reference engines from one seed:

  1. the ARRAY PATH -- ``stepping``'s passes under the driver's own loop;
  2. the CERTIFIED SINGLES -- ``update_H`` then ``step_D`` as two dispatched plans;
  3. the COMPOSITION THE COMPOSER INSTALLS ON THESE ROWS TODAY -- ``plan_step(fuse=
     True)``'s own plans, which on every row this product reaches puts the released
     B->H fused magnetic pair in ``step_B``/``update_H`` and the certified ``step_D``
     (bare, or inside the released D->E pair) after it. This is the reference the
     arbitration finding rests on, and its slot table is recorded per case;
  4. the same slots DISPATCHED UNFUSED -- ``plan_step(fuse=False)``, one single per
     slot.

All four must agree with the weld word for word. What differs is the LAUNCH COUNT,
which ``launch_structure`` reports twice over (the plans' own counters and an
independent wrapper around every compiled function) at the seam AND over the whole
step -- where the honest number is that the weld does NOT reduce the step's launches
against the composition installed today. That is the arbitration ruling reproduced
as a measurement rather than re-litigated.

THE SUBNORMAL POLICY IS A PRECONDITION, NOT A DEFAULT. The MPS executor flushes
denormals natively and exposes no lever, so ``flush`` is the one policy it honours;
a run whose resolved policy is ``keep`` is REFUSED BY NAME (``coverage.py`` clause 4)
and recorded as refused rather than run. Every product row censuses the oracle's own
state for subnormal words and reports it.

AND THE FLUSH PRECONDITION HAS TWO HALVES SINCE 2026-09-06, because one of them was
blind. The stored-state census sees a subnormal only once one is WRITTEN, and the IEEE
underflow flag beside it fires only when a result is tiny AND INEXACT -- so an
intermediate that is tiny and EXACT, which is what the difference of two nearby normals
into the subnormal range is, is invisible to both. That is what put
``examples:gaussian-beam.py`` and ``tests:TestEigenmodeSource.test_waveguide_flux`` 1-2
units in the last place away from the array path on EVERY dispatched arrangement, the
certified singles included: the curl's own ``shifted_first - first`` at two
mirror-symmetric absorber cells is ``0x80485000``, a subnormal, and flushing just that
one intermediate reproduces the device's words bit for bit. :class:`IntermediateCensus`
censuses the reference's own arithmetic per step, MODELS each helper's chain in float32
and asserts its model reproduces what the helper produced, and stops the walk on the
same rule the stored census stops it on. Measured over six corpus rows, it fires on the
same step the stored census does on five and one step earlier on the sixth.

TWO OTHER DISPOSITIONS A LIFTED ROW CAN TAKE, both measured rather than named. A row
whose SOURCE WAVEFORM IS NOT A FUNCTION OF TIME hands each of the five arrangements a
different current, so byte identity across them is undefined by construction rather
than false: :func:`waveform_determinism` asks the engine's own accessor twice at the
same instant over the whole budget and refuses such a row inside the cell's
denominator. And ``Fields``'s PRIVATE SCRATCH is not state -- assigned whole before
anything reads it -- so it leaves the comparison on the leading-underscore rule twelve
sibling probes already apply, with the premise driven by ``private_scratch``.

LEGS
  host legs (no device launch; the predicate's backend clause still needs MPS)
  driver_order       REPLACES is exactly the driver's two adjacent consults, the only
                     statement between them is the electric withdraw loop, and the
                     sync channel's containment rule excludes step_D -- all read off
                     the tree, never spelled
  transcription      both halves are the certified emitters' own bytes: the curl tail
                     differs in EXACTLY the nine redirected magnetic loads, the
                     constitutive tail in EXACTLY the declared lift edits, the
                     signature is the platform ceiling as an equality, every mutation
                     needle resolves exactly once, the text is ASCII
  refusal            by name: an undeclared source list; a standing INTEGRATED
                     electric withdraw (the reason the two corpus rows are refused);
                     a non-integrated electric source NOT refused; a magnetic source
                     NOT refused; a folded grid naming the second product; a
                     cylindrical grid; an inactive absorber naming the constitutive
                     half; a resolved keep policy
  arbitration        the composer, asked: on a row both incumbents reach the released
                     B->H pair holds step_B AND update_H, this product is refused by
                     name, launches over step_B..update_E are 2, and the H_to_D seam
                     row routes to the withdraw hoist
  purity_ledger      per fixture, on the array path: of the curl's valid foreign
                     taps, how many land on a cell update_H MOVED -- the count of taps
                     whose recomputed value differs from what an in-place weld would
                     have raced on. Near zero would mean the race legs license nothing
  private_scratch    Fields' private buffers, POISONED before a complete step on a
                     dispersive driver: no stored word may move and no poison may
                     survive, with the same poison in Dz as the null control that must
                     fire. This is the premise the comparison's exclusion rests on
  compile legs (compile only)
  binding_ceiling    35 separate bindings must FAIL, 34 (kms unshared) must FAIL, 32
                     (one more pointer) must FAIL, the shipped 31 must COMPILE on all
                     sixteen specialisations
  mutants_compile    every armed shader defect and the byte-neutral control COMPILE,
                     so a mutant reported caught was a launched kernel
  device legs
  product            the four-reference identity on the eight boundary triples,
                     complete driver steps, movement floor, subnormal census
  lift               every corpus row the standing census puts in the (update_H
                     ordinary -> step_D PML) cell, re-lifted in its own interpreter
                     through the census driver: the predicate's verdict per row (47
                     admitted, 2 refused by name expected), the four-reference
                     identity per admitted row over the same budget, the composer's
                     own slot table per row, and each row's own disposition -- driven,
                     refused by the predicate, or undefined by construction
  launch_structure   launches per step at the seam and over the whole step, for the
                     weld and all four references, two independent counters
  rotation_guard     the guard that decides whether a comparison is reading buffers
                     anybody launched against: a MIRRORED volume pointed at a twin
                     must be named, the engine's own polarization rotation must not be
  sync               the product FORCE-installed; flux_in_box called mid-run so
                     synchronize_magnetic_fields consults update_H_synchronize. An
                     arm whose plan answers that consult MUST diverge in D/fu_D (the
                     hazard); the arm that declines by the containment rule MUST NOT
  withdraw           an integrated electric source in the seam (the configuration the
                     predicate refuses today): the hoisted arrangement identical, the
                     un-hoisted launch and the array path's after_step_D both diverge
                     -- the device counterpart of results/h_to_d_withdraw_order
  byte_neutral       the own-cell register reads replaced by reloads of the scratch
                     just stored: the one armed edit required NOT to diverge
  mutation           the armed shader and host defects, each of which MUST diverge;
                     predicted-null entries recorded with their reason, never dropped
  disarm             the identical harness, shipped bytes, must not diverge

Rule 7: one flushed line per case; every row appended and fsynced as it lands; the
lift leg writes one JSON per corpus row as it lands and a progress log the child
appends to per step.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# THE POLICY, SET BEFORE ANY meep_gpu MODULE IS REACHED. `flush` is the only value
# the MPS executor can honour; a resolved `keep` refuses every predicate by name.
os.environ.setdefault("MEEP_GPU_SUBNORMAL_POLICY", "flush")
HERE = Path(__file__).resolve().parent
# BY NAME, never by parents[N]: a moved harness resolving a wrong root measures nothing.
API_ROOT = next(parent for parent in HERE.parents
                if (parent / "meep_gpu" / "metal_kernels").is_dir())
for _path in (str(API_ROOT), str(HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np  # noqa: E402

import h_to_d_seam  # noqa: E402
import metal_composition_matrix as matrix  # noqa: E402

from meep_gpu import stepping, withdraw_hoist  # noqa: E402
from meep_gpu.fastpath import SYNC_PASS_OWNERS, SYNC_PATH_SLOTS, SYNC_UPDATE_H_PASS  # noqa: E402
from meep_gpu.metal_kernels import (  # noqa: E402
    fused_hd_pair as family, fused_magnetic_pair as magnetic,
    launch as metal_launch, shaders, subnormal,
)
from meep_gpu.metal_kernels.device import (  # noqa: E402
    MAX_BUFFER_BINDINGS, Residency, compile_source, metal_frontend_version,
)

# ---------------------------------------------------------------------------
# The battery contract, so the census driver can lift a corpus row into this file
# ---------------------------------------------------------------------------

#: ``measure_predicate_coverage`` digests this package into every lifted row's
#: ``subject_manifest_sha256``. Declared, as every battery declares it.
SUBJECT_PACKAGE = "metal_kernels"

#: The budget every case runs. SIXTY, not the sibling gates' twelve: the state this
#: weld carries between steps (the rotated ``H``/``f_w_H`` pair AND the in-place
#: ``fu_D``) compounds across steps, and the withdraw campaign this gate is the device
#: counterpart of ran sixty. The comparison is per COMPLETE STEP.
STEPS = 60

#: The steps at which the sync leg calls the flux accessor. Two, so the hazard is
#: measured to persist and not to be a one-step transient.
SYNC_STEPS: Tuple[int, ...] = (3, 7)

#: HOW MANY CLEAN COMPLETE STEPS A LIFTED CORPUS ROW MUST REACH TO COUNT.
#:
#: The synthetic fixture's amplitude is this gate's to choose, so it is placed clear
#: of the denormal band for the whole budget (:data:`SEED_SCALE_BITS`). A lifted row's
#: state is the ROW'S: it starts at the engine's zeros and is driven by its own
#: sources, so its wavefront's leading cells are arbitrarily small and it enters the
#: band on its own schedule -- measured 2026-09-05 over four rows at step 20, 24, 34
#: and 55 of 60. Past that step MPS flushes where NumPy does not and NO byte claim can
#: be made, so the lifted budget is every step the precondition holds, capped at the
#: requested one, and this is the floor below which a row measured too little to be
#: worth a verdict. It is a FLOOR and not a target: the record carries each row's own
#: step count and the words it compared.
LIFT_CLEAN_STEP_FLOOR = 8

#: The boundary triples, as ``metal_composition_matrix.cart`` keywords: all eight
#: specialisations, because the ghost rule is exactly what specialisation changes
#: and the recompute's guard is what a redirect is most likely to eat. The names are
#: the probe's, so the two records read side by side.
CASES: Tuple[Tuple[str, Dict[str, Any]], ...] = tuple(
    (f"bc_{'m' if x else 'p'}{'m' if y else 'p'}{'m' if z else 'p'}",
     {"boundaries": {axis: "metallic" for axis, walled in zip("xyz", (x, y, z))
                     if walled}})
    for x in (0, 1) for y in (0, 1) for z in (0, 1))

#: The case the mutations are armed on: every axis walled, so every line the shipped
#: kernel can emit is present and the metallic ghost rule is live on every axis.
MUTATION_CASE = "bc_mmm"

#: THE SECOND MUTATION CASE, and it is not a duplicate of the first. On a metallic
#: axis ``stepping._mask_non_owned_cells`` zeroes the very curl component the shifted
#: tap feeds, at the very plane where the ghost is served, so no metallic BOUNDARY
#: VALUE is observable in this kernel's output. On an all-periodic grid that mask does
#: not fire and cell 0's curl is a live stepped value assembled from the WRAPPED row,
#: which is the only specialisation on which the boundary rule can be measured at all.
PERIODIC_MUTATION_CASE = "bc_ppp"

def codes_of(name: str) -> Tuple[int, ...]:
    """The per-axis PERIODIC/METALLIC triple one case compiles to, off its keywords.

    Read from the case table rather than written twice, so a mutation armed on a
    specialisation is armed on the specialisation the driver for that case builds.
    """
    walls = dict(CASES)[name].get("boundaries") or {}
    return tuple(1 if walls.get(axis) == "metallic" else 0 for axis in "xyz")


#: The two specialisations the mutation legs compile against.
MUTATION_CODES: Tuple[int, ...] = codes_of(MUTATION_CASE)
PERIODIC_CODES: Tuple[int, ...] = codes_of(PERIODIC_MUTATION_CASE)

#: The synthetic fixture's geometry -- the composition matrix's own ``cart`` row, so
#: the fixture certified here is the fixture the two halves were certified on.
CELL: Tuple[float, float, float] = (2.0, 2.1, 1.2)
RESOLUTION = 10.0
COURANT = 0.35
PML_CELLS = 2
#: Per-component permittivity, ``metal_composition_matrix._epsilon``'s own values.
EPSILON: Dict[str, float] = {"Ex": 2.0, "Ey": 2.5, "Ez": 3.0}

#: THE SEED IS SCALED BY 2^80, AND THE EXPONENT IS THE MEASUREMENT.
#:
#: This gate's budget is SIXTY complete steps and its fixture is a seeded,
#: source-free run inside an absorber, so the state DECAYS. Measured 2026-09-05 on
#: ``bc_mmm`` at the composition matrix's own amplitude (0.37): the array path's
#: ``fu_Bz`` enters the float32 DENORMAL band at step **39 of 60**, and from there the
#: byte claim is not merely unproven, it is unprovable -- MPS flushes denormals
#: natively and NumPy does not, so the two paths differ by exactly one word for a
#: reason that has nothing to do with any kernel. (That is what the first full run of
#: this gate reported: one differing word in ``fu_Bz`` at step 38-40 on seven of the
#: eight cases, in every arrangement that dispatches ``step_B`` and in none that
#: leaves it on the array path.)
#:
#: A POWER OF TWO IS WHY THIS IS NOT A TUNED FIXTURE. The solver is linear in the
#: field state and every coefficient it multiplies by -- ``kms``, ``sinv``, ``kps``,
#: ``dtdx``, the inverse permittivity -- is field-independent, so scaling every stored
#: volume by 2^n shifts each float32 EXPONENT by n and leaves every MANTISSA and every
#: rounding decision untouched. The same arithmetic is measured, further from the
#: band. ``leg_seed_scale`` drives that equality rather than asserting it.
#:
#: WHY 80. Measured over the whole budget on ``bc_mmm``: 2^0 enters the band at step
#: 39, 2^40 at step 53, 2^80 never, and 2^80 peaks at |x| = 3.3e26 -- twelve decimal
#: orders under float32's finite range, so nothing overflows either. 80 is the first
#: multiple of 40 that clears the budget with that much room on both sides.
SEED_SCALE_BITS = 80

#: The reference engines, in the order the record reports them. ``weld`` is the
#: subject; the other four are the references named in the module docstring.
MODES: Tuple[str, ...] = ("array", "singles", "composition_today", "unfused",
                          "weld", "weld_seam_only")

LEG_GROUPS: Dict[str, Tuple[str, ...]] = {
    "host": ("driver_order", "transcription", "refusal", "arbitration",
             "seed_scale", "purity_ledger", "private_scratch"),
    "compile": ("binding_ceiling", "mutants_compile"),
    "device": ("product", "lift", "launch_structure", "rotation_guard", "sync",
               "withdraw", "byte_neutral", "mutation", "disarm"),
}
ALL_LEGS: Tuple[str, ...] = tuple(leg for group in LEG_GROUPS.values() for leg in group)

#: The census this gate lifts its corpus rows from, and the seam record whose
#: ``withdraw_in_seam`` flag names the rows the predicate must refuse.
CENSUS = "metal_coverage_2026-09-04_m0complex"
SEAM_RECORD = "h_to_d_seam_2026-09-04"
CELL_ARMS: Tuple[str, str] = ("ordinary", "PML")

#: The runner's "cannot certify on this host" code: a partial run (not every leg
#: requested) exits with it so ``release.released`` is False and nothing can mint it.
EXIT_INCOMPLETE = 75


def log(message: str) -> None:
    print(message, flush=True)


# ---------------------------------------------------------------------------
# Words, volumes, comparison
# ---------------------------------------------------------------------------

def words(array: Any) -> np.ndarray:
    """One array as uint32 WORDS. Byte compares, never allclose."""
    contiguous = np.ascontiguousarray(array)
    if contiguous.dtype == np.complex64:
        return contiguous.reshape(-1).view(np.uint32)
    return np.frombuffer(contiguous.astype(np.float32, copy=False).tobytes(),
                         dtype=np.uint32)


def differing(left: Any, right: Any) -> int:
    a, b = words(left), words(right)
    if a.shape != b.shape:
        return max(a.size, b.size)
    return int(np.count_nonzero(a != b))


_RESET_VOLUMES: Optional[Tuple[str, ...]] = None

#: PRIVATE SCRATCH ON ``Fields``, WHICH IS NOT STATE, AND THE RULE IS THE LEADING
#: UNDERSCORE rather than this tuple -- which is recorded so the artifact can name
#: what the rule excluded on the fixture it ran.
#:
#: ``_fmp_scratch`` and ``_fmp_scratch_by_component`` are the buffers
#: ``Fields.displacement_minus_polarization`` (fields.py:1099-1105) and
#: ``displacement_minus_polarization_volumes`` (:1130-1136) allocate on first use.
#: Each is assigned WHOLE -- ``scratch[...] = displacement`` -- before anything reads
#: it, so no value in it survives to be read by the next step, and WHICH ROUTE
#: allocated it differs by route while the answer does not. Twelve sibling Triton
#: probes and ``gate_dispatch_end_to_end.SCRATCH_NAMES`` already exclude it; this gate
#: compared it, and on ``examples:stochastic_emitter_line.py`` and
#: ``examples:stochastic_emitter_reciprocity.py`` that was the ONLY thing it found --
#: four and three words, in a buffer no route reads, on rows whose whole engine state
#: was byte-identical on every arrangement.
#:
#: THE PREMISE IS NOT ASSERTED HERE, IT IS MEASURED: ``leg_private_scratch`` poisons
#: the buffer before a complete step on a dispersive driver and requires every stored
#: volume to come back byte-identical, with the poison's disappearance as the
#: non-vacuity witness that the buffer was exercised at all.
PRIVATE_SCRATCH: Tuple[str, ...] = ("_fmp_scratch", "_fmp_scratch_by_component")


def reset_declared_volumes() -> Tuple[str, ...]:
    """Every array ``Fields.reset`` zeroes, read out of ``fields.py``'s own text.

    DERIVED, NOT TRANSCRIBED: a volume added to the engine joins the comparison
    without an edit here, and a list here could go stale while looking complete.
    Private scratch is dropped by the leading-underscore rule (:data:`PRIVATE_SCRATCH`),
    which is why a name is excluded by an argument rather than by being forgotten.
    """
    global _RESET_VOLUMES  # noqa: PLW0603 - read once from disk, then held
    if _RESET_VOLUMES is not None:
        return _RESET_VOLUMES
    text = (API_ROOT / "meep_gpu" / "fields.py").read_text(encoding="utf-8")
    match = re.search(r"\n    def reset\(self\).*?(?=\n    def )", text, re.S)
    if match is None:
        raise SystemExit("cannot find Fields.reset in fields.py; refusing to guess "
                         "the list of stored volumes")
    names = set(re.findall(r"self\.([A-Za-z_][A-Za-z0-9_]*)", match.group(0)))
    _RESET_VOLUMES = tuple(sorted(name for name in names - {"polarizations"}
                                  if not name.startswith("_")))
    return _RESET_VOLUMES


def stored_volumes(fields: Any) -> Dict[str, Any]:
    """Every stored volume this engine allocates, LIVE (not copied), by name."""
    out: Dict[str, Any] = {}
    for name in reset_declared_volumes():
        array = getattr(fields, name, None)
        if array is not None:
            out[name] = array
    for index, state in enumerate(getattr(fields, "polarizations", ()) or ()):
        for component, array in getattr(state, "P", {}).items():
            out[f"P[{index}].{component}"] = array
        for component, array in getattr(state, "P_prev", {}).items():
            out[f"P_prev[{index}].{component}"] = array
    return out


def capture(driver: Any) -> Dict[str, Any]:
    """The whole of the state one complete step advances."""
    return {
        "volumes": {name: np.array(array, copy=True)
                    for name, array in stored_volumes(driver.fields).items()},
        "dipoles": [getattr(source, "_applied_dipole", None)
                    for source in getattr(driver, "_sources", ())],
        "step_count": int(driver.step_count),
    }


def restore(driver: Any, snapshot: Mapping[str, Any]) -> List[str]:
    """Put a captured state back IN PLACE. Returns volumes the seed did not carry."""
    lazily: List[str] = []
    for name, array in stored_volumes(driver.fields).items():
        saved = snapshot["volumes"].get(name)
        if saved is None:
            array.fill(0)
            lazily.append(name)
        else:
            array[...] = saved
    for source, dipole in zip(getattr(driver, "_sources", ()), snapshot["dipoles"]):
        if dipole is not None:
            source._applied_dipole = dipole  # noqa: SLF001 - the offset IS the state
    driver.step_count = snapshot["step_count"]
    return lazily


def compare_snapshots(reference: Mapping[str, Any],
                      other: Mapping[str, Any]) -> Dict[str, int]:
    a, b = reference["volumes"], other["volumes"]
    assert set(a) == set(b), sorted(set(a) ^ set(b))
    return {name: n for name in sorted(a) if (n := differing(a[name], b[name]))}


#: How many differing words of each volume the autopsy transcribes. A divergence
#: this gate cares about is a handful of words; a wholesale one is characterised
#: by its counts, which are recorded for every differing word regardless.
AUTOPSY_WORDS = 12


def _classify(word: int) -> str:
    """``zero`` / ``subnormal`` / ``normal`` -- read off the BITS, never by value."""
    if not word & 0x7FFFFFFF:
        return "zero"
    if not word & 0x7F800000:
        return "subnormal"
    return "normal"


def autopsy(reference: Mapping[str, Any], other: Mapping[str, Any],
            difference: Mapping[str, int]) -> Dict[str, Any]:
    """The BIT PATTERNS behind a divergence, so a row can be ATTRIBUTED.

    A count of differing words says a run disagreed; it does not say whether the
    two sides differ by one unit in the last place, by a flush of a subnormal to
    zero, or by a wholesale wrong answer -- and those three findings have three
    different subjects. This transcribes each differing word from BOTH sides with
    its class, so the record carries the evidence rather than a later reading of
    it. Purely additive: nothing here decides a verdict.
    """
    out: Dict[str, Any] = {}
    for name in sorted(difference):
        left, right = words(reference["volumes"][name]), words(other["volumes"][name])
        if left.shape != right.shape:
            out[name] = {"shape_differs": [int(left.size), int(right.size)]}
            continue
        where = np.flatnonzero(left != right)
        classes: Dict[str, int] = {}
        for a_word, b_word in zip(left[where].tolist(), right[where].tolist()):
            key = f"{_classify(a_word)}->{_classify(b_word)}"
            classes[key] = classes.get(key, 0) + 1
        transcript = []
        for index in where[:AUTOPSY_WORDS].tolist():
            a_word, b_word = int(left[index]), int(right[index])
            a_value = float(np.frombuffer(np.uint32(a_word).tobytes(),
                                          dtype=np.float32)[0])
            b_value = float(np.frombuffer(np.uint32(b_word).tobytes(),
                                          dtype=np.float32)[0])
            transcript.append({
                "index": int(index),
                "reference_word": f"0x{a_word:08x}", "other_word": f"0x{b_word:08x}",
                "reference": a_value, "other": b_value,
                "reference_class": _classify(a_word), "other_class": _classify(b_word),
                # THE SIGNED-MAGNITUDE DISTANCE, which is the ulp distance for two
                # words of the same sign and is reported as-is otherwise.
                "word_distance": abs(a_word - b_word),
                "same_sign": (a_word >> 31) == (b_word >> 31)})
        out[name] = {
            "differing_words": int(where.size),
            "words_in_volume": int(left.size),
            "reference_subnormals": int(np.count_nonzero(
                ((left & np.uint32(0x7F800000)) == 0) & ((left & np.uint32(0x007FFFFF)) != 0))),
            "other_subnormals": int(np.count_nonzero(
                ((right & np.uint32(0x7F800000)) == 0) & ((right & np.uint32(0x007FFFFF)) != 0))),
            "classes": dict(sorted(classes.items())),
            "transcript": transcript}
    return out


# ---------------------------------------------------------------------------
# The synthetic fixture: a REAL driver, the composition matrix's own geometry
# ---------------------------------------------------------------------------

def build_driver(keywords: Mapping[str, Any], seed: int,
                 sources: Sequence[Mapping[str, Any]] = (),
                 scale_bits: int = SEED_SCALE_BITS) -> Any:
    """One seeded ``FdtdDriver`` on the ``cart`` row's geometry and material.

    A DRIVER, not a bare ``Fields``: every leg here is a claim about a COMPLETE
    driver step -- the withdraw loop, the injection slot, the fill consults, the
    wall clears and the sync channel are ``FdtdDriver.step``'s and
    ``synchronize_magnetic_fields``'s, and a hand-written walk over the live pass
    list would be a second model of what a step is.
    """
    from meep_gpu.driver import FdtdDriver  # noqa: PLC0415

    driver = FdtdDriver(cell_size=CELL, resolution=RESOLUTION, courant=COURANT,
                        force_complex_fields=False,
                        boundaries=dict(keywords.get("boundaries") or {}) or None,
                        dimensions=3)
    driver.setup_pml(int(keywords.get("pml", PML_CELLS)))
    shape = tuple(driver.grid.shape)
    driver.fields.set_epsilon_volumes(
        {name: np.full(shape, np.float32(value), np.float32)
         for name, value in EPSILON.items()},
        {name: np.full(shape, np.float32(1.0 / value), np.float32)
         for name, value in EPSILON.items()})
    for source in sources:
        driver.add_source(dict(source))
    rng = np.random.default_rng(seed)
    scale = np.float32(2.0) ** int(scale_bits)
    for array in stored_volumes(driver.fields).values():
        array[...] = (0.37 * rng.standard_normal(array.shape)).astype(array.dtype)
        array *= scale
    driver.invalidate_fast_path()
    pin_array_path(driver)
    return driver


def pin_array_path(driver: Any) -> None:
    """Freeze this driver's fast path as the pure array path (or as a shim)."""
    driver._fast_path = None  # noqa: SLF001 - the engine's own freeze slot
    driver._fast_path_stale = False  # noqa: SLF001


def install(driver: Any, shim: Optional["Shim"]) -> None:
    driver._fast_path = shim  # noqa: SLF001
    driver._fast_path_stale = False  # noqa: SLF001


# ---------------------------------------------------------------------------
# The dispatch shim -- the engine's own seam, answered by this gate's plans
# ---------------------------------------------------------------------------

class ABSORBED:
    """The marker for a slot a fused launch already performed: answer True, run nothing."""


class SyncedPlan:
    """A device plan bracketed by the residency's host->device / device->host copies.

    THE BRACKET SITS IMMEDIATELY AROUND THE LAUNCH and nowhere else, because two
    wrappers in this composition do HOST work before their launch --
    ``LeadingRepairPlan`` saves the deposit points and ``LeadingWithdrawPlan``
    performs the seam's electric withdraw -- and the launch must read the host state
    AFTER that work. A shim that synchronised around the whole slot would hand the
    fused launch a ``D`` still holding the previous step's standing dipole, which is
    the exact defect the hoist exists to prevent, and report success.
    """

    __slots__ = ("inner", "residency")

    def __init__(self, inner: Any, residency: Residency) -> None:
        self.inner = inner
        self.residency = residency

    def run(self, *args: Any, **kwargs: Any) -> None:
        self.residency.sync_in()
        self.inner.run(*args, **kwargs)
        self.residency.sync_out()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


def synced(plan: Any, residency: Residency) -> Any:
    """Wrap the DEVICE half of whatever sits in a slot; leave host-only work alone.

    The three shapes a slot can hold after the composer runs: a bare device plan
    (wrapped), a wrapper whose ``inner`` launches (its ``inner`` is wrapped, so the
    host work it does first stays first), and a host-only or no-op plan
    (``NoopPlan``, ``TrailingRepairPlan``: left alone -- the next launch's
    ``sync_in`` carries their host writes to the device).
    """
    if plan is None or plan is ABSORBED:
        return plan
    inner = getattr(plan, "inner", None)
    if inner is not None and getattr(plan, "absorbed_by", None) is inner:
        plan.inner = SyncedPlan(inner, residency)
        return plan
    if getattr(plan, "absorbed_by", None) is not None:
        return plan  # NoopPlan / TrailingRepairPlan: the launch is elsewhere
    if not getattr(plan, "performs_device_work", True):
        return plan
    return SyncedPlan(plan, residency)


def declaring(plan: Any) -> Any:
    """The plan whose counters describe the device work in a slot."""
    if isinstance(plan, SyncedPlan):
        plan = plan.inner
    inner = getattr(plan, "absorbed_by", None)
    if inner is not None:
        return declaring(inner)
    return plan


class Shim:
    """A ``FastPathPlan``-shaped object that runs this gate's plans inside the driver.

    THE SEAM IS THE ENGINE'S OWN: ``driver.step`` reads ``self._fast_path`` once per
    step and consults ``fast.dispatch(slot, fields)``; ``synchronize_magnetic_fields``
    consults it again under the by-name channel ``update_H_synchronize``. This object
    answers both. The shipped planner is untouched.

    THE SYNC CHANNEL IS ANSWERED BY THE CONTAINMENT RULE, READ OFF ``fastpath``: a
    plan may run inside the magnetic half-step only when every slot it spans is one
    the half-step itself runs (``SYNC_PATH_SLOTS``). A weld spanning ``step_D`` is
    refused by its span. ``sync_hazard=True`` is the ARMED arm of the sync leg: the
    plan answers the consult anyway, which is what a product that did not decline
    would do, and the leg requires the divergence in ``D``.

    EVERY ANSWER IS COUNTED. A shim that quietly declined everything would still
    produce a green byte comparison, because the oracle IS the array path.
    """

    __slots__ = ("fields", "plans", "dispatched", "declined", "absorbed",
                 "sync_refusals", "sync_answered", "sync_hazard")

    def __init__(self, fields: Any, plans: Mapping[str, Any],
                 sync_hazard: bool = False) -> None:
        self.fields = fields
        self.plans = {name: plan for name, plan in plans.items() if plan is not None}
        self.dispatched: Dict[str, int] = {}
        self.declined: Dict[str, int] = {}
        self.absorbed: Dict[str, int] = {}
        self.sync_refusals = 0
        self.sync_answered = 0
        self.sync_hazard = sync_hazard

    def span_of(self, slot: str) -> Tuple[str, ...]:
        plan = self.plans.get(slot)
        if plan is None or plan is ABSORBED:
            return (slot,)
        return tuple(getattr(declaring(plan), "replaces_sub_steps", None) or (slot,))

    def dispatch(self, slot: str, fields: Any) -> bool:
        owner = SYNC_PASS_OWNERS.get(slot)
        if owner is not None:
            if owner not in self.plans:
                self.declined[slot] = self.declined.get(slot, 0) + 1
                return False
            outside = tuple(name for name in self.span_of(owner)
                            if name not in SYNC_PATH_SLOTS)
            if outside and not self.sync_hazard:
                self.sync_refusals += 1
                return False
            self.sync_answered += 1
            slot = owner
        plan = self.plans.get(slot)
        if plan is None or fields is not self.fields:
            self.declined[slot] = self.declined.get(slot, 0) + 1
            return False
        if plan is ABSORBED:
            self.absorbed[slot] = self.absorbed.get(slot, 0) + 1
            return True
        plan.run()
        self.dispatched[slot] = self.dispatched.get(slot, 0) + 1
        return True

    @property
    def launches(self) -> int:
        """Launches the PLANS counted, a different witness from the shim's tallies."""
        seen: List[int] = []
        total = 0
        for plan in self.plans.values():
            if plan is ABSORBED:
                continue
            owner = declaring(plan)
            if id(owner) in seen:
                continue
            seen.append(id(owner))
            total += int(getattr(owner, "launches", 0))
        return total


class CountingFunction:
    """An independent launch counter: wraps a compiled function and counts calls."""

    __slots__ = ("function", "calls")

    def __init__(self, function: Any) -> None:
        self.function = function
        self.calls = 0

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        return self.function(*args, **kwargs)


def count_functions(plan: Any) -> List[CountingFunction]:
    """Replace every compiled function the plan holds by a counting wrapper."""
    owner = declaring(plan)
    functions = getattr(owner, "_functions", None)
    if not isinstance(functions, dict):
        return []
    wrapped = {mode: CountingFunction(function) for mode, function in functions.items()}
    owner._functions = wrapped  # noqa: SLF001 - the sibling gates' mutation seam
    return list(wrapped.values())


# ---------------------------------------------------------------------------
# The five engines, on ONE driver, in lockstep from one seed
# ---------------------------------------------------------------------------

def settle_rotation(plan: Any, fields: Any, originals: Mapping[str, Any]) -> int:
    """After a step that rotated, put the ENGINE's references back on the seed arrays.

    THE WELD SWAPS ``fields.Hx`` AND ITS TWIN AFTER EVERY LAUNCH, and that is the
    product's own certified choreography. But this gate drives FIVE arrangements on
    ONE driver, and the other four bound their device mirrors to the seed arrays by
    identity; a rotation left standing at the end of a step would leave them writing
    an array the engine no longer names. So once the step is complete the advanced
    values are copied into the seed array, the reference is put back, and the twin
    resumes its role as scratch. The rotation INSIDE the step -- the thing the null
    control catches -- is untouched. Returns how many volumes were settled.
    """
    settled = 0
    for name in getattr(plan, "rotated_names", ()):
        current = getattr(fields, name)
        original = originals[name]
        if current is not original:
            original[...] = current
            setattr(fields, name, original)
            plan.rotated[name] = current
            settled += 1
    return settled


def rotating_owners(shim: Optional[Shim]) -> Tuple[Any, ...]:
    """Every distinct plan in this shim that rotates one of the ENGINE's volumes.

    DERIVED FROM THE SHIM, never listed by the caller, and that is the repair for a
    measured defect. This gate used to hand ``rotating`` in by hand, and only the
    weld arrangement handed anything: the WELD is not the only rotating product a
    step can carry. ``plan_step`` installs the off-diagonal fused electric pair on
    any row with an off-diagonal epsilon, and that product is a SCRATCH-OUTPUT weld
    whose ``rotated_names`` are ``D``/``fu_D`` -- so ``composition_today`` left
    ``fields.D*`` naming the pair's twins, and every arrangement stepped after it
    wrote the original arrays through mirrors bound by identity while ``capture``
    read the orphaned twins. Measured 2026-09-05 on ``examples:cavity_arrayslice.py``:
    ``unfused``, ``weld`` and ``weld_seam_only`` each came back 4 words short in
    ``D``/``fu_D`` -- the source deposit's whole contribution -- and the same weld
    driven with ``composition_today`` absent was byte-identical. A list a caller
    maintains cannot see a product the composer chose; the shim can.
    """
    if shim is None:
        return ()
    owners: List[Any] = []
    seen: List[int] = []
    for plan in shim.plans.values():
        if plan is ABSORBED:
            continue
        owner = declaring(plan)
        if id(owner) in seen or not getattr(owner, "rotated_names", ()):
            continue
        seen.append(id(owner))
        owners.append(owner)
    return tuple(owners)


def rotating_originals(fields: Any, owners: Sequence[Any]) -> Dict[str, Any]:
    """The array the engine names for each rotating volume, BEFORE any launch."""
    return {name: getattr(fields, name)
            for owner in owners for name in owner.rotated_names}


def orphaned_mirrors(arrangements: Mapping[str, "Arrangement"],
                     fields: Any) -> List[str]:
    """Every MIRRORED volume the engine has stopped naming, across all arrangements.

    THE HAZARD IS AN ORPHANED MIRROR, NOT A RENAME. A device mirror is registered
    once per name against the host array it shadows (``device.Residency.mirror``) and
    syncs THAT array for the life of the residency, so a rotation left standing at the
    end of a step leaves a later arrangement launching against a buffer nobody reads
    -- which manufactures a divergence and, just as badly, an agreement between two
    arrangements that both wrote an orphan. Measured 2026-09-05 on
    ``examples:cavity_arrayslice.py``, where ``composition_today``'s off-diagonal
    scratch-output pair left ``fields.D*`` naming its twins.
    This asks the mirrors themselves: for every name any arrangement's residency
    binds, is the array it binds still the one the engine names?

    THAT IS A DIFFERENT QUESTION FROM "did any name's array object change", which is
    what this gate scored until 2026-09-06 and which the ENGINE'S OWN RECURRENCE
    answers yes to. ``PolarizationState.step`` (dispersion.py:679-691) rotates three
    buffers per driven component every step BY DESIGN -- this step's result is written
    into the scratch, the scratch takes the array that held ``P_prev``, and the names
    shift down -- and ``capture``/``restore`` follow those names, so the comparison is
    exact through it. Measured 2026-09-06 on ``examples:stochastic_emitter_line.py``:
    all six arrangements rotate those 36 volumes, and NO arrangement's residency
    mirrors a single one of them (0 of 24, 105, 105, 111 and 30 mirror names), so the
    orphan the id check was reporting could not exist. Both facts are recorded: the
    engine's rename under ``engine_volumes_left_rotated``, the hazard here.
    """
    seen: List[str] = []
    for arrangement in arrangements.values():
        residency = arrangement.residency
        if residency is None:
            continue
        for name in residency.names:
            bound = residency.host(name)
            current = getattr(fields, name, None)
            if bound is None or current is None:
                continue  # plan-owned scratch, or a mirror that is not a Fields volume
            if current is not bound and name not in seen:
                seen.append(name)
    return sorted(seen)


class Arrangement:
    """One engine: a shim (or the array path), its residency, and its bookkeeping."""

    __slots__ = ("name", "shim", "residency", "counters", "rotating", "originals",
                 "selected", "reasons", "on_step_done")

    def __init__(self, name: str, shim: Optional[Shim], residency: Optional[Residency],
                 selected: Optional[Mapping[str, str]] = None,
                 reasons: Optional[Mapping[str, Any]] = None) -> None:
        self.name = name
        self.shim = shim
        self.residency = residency
        self.counters: List[CountingFunction] = []
        if shim is not None:
            # ONCE PER DECLARING OWNER, never once per slot. A fused pair occupies
            # BOTH slots of its seam, so walking slots wraps that owner's function
            # table twice -- the second wrapper counting the first -- and the
            # independent counter then reports exactly twice the launches the plans
            # report, which reads as the two witnesses disagreeing. Measured
            # 2026-09-05 on the ``composition_today`` arrangement: 12 against 6.
            wrapped: List[int] = []
            for plan in shim.plans.values():
                if plan is ABSORBED:
                    continue
                owner = declaring(plan)
                if id(owner) in wrapped:
                    continue
                wrapped.append(id(owner))
                self.counters.extend(count_functions(plan))
        self.rotating = rotating_owners(shim)
        self.originals = ({} if shim is None
                          else rotating_originals(shim.fields, self.rotating))
        self.selected = dict(selected or {})
        self.reasons = dict(reasons or {})
        self.on_step_done: Optional[Callable[[], None]] = None

    def settle(self, fields: Any) -> int:
        return sum(settle_rotation(plan, fields, self.originals) for plan in self.rotating)

    def launches(self) -> Dict[str, int]:
        return {"plans": 0 if self.shim is None else self.shim.launches,
                "functions": sum(counter.calls for counter in self.counters)}


def arrangement_singles(driver: Any) -> Arrangement:
    """Reference 2: the two certified singles dispatched at the seam's two slots."""
    residency = Residency()
    fields, pml = driver.fields, driver.pml
    constitutive = metal_launch.plan_constitutive(fields, pml, "H", residency)
    curl = metal_launch.plan_pml_curl(fields, pml, "step_D", residency)
    if constitutive is None or curl is None:
        raise RuntimeError("a certified single was refused on a fixture this gate "
                           "expects it to admit")
    shim = Shim(fields, {"update_H": synced(constitutive, residency),
                         "step_D": synced(curl, residency)})
    return Arrangement("singles", shim, residency,
                       selected={"update_H": "ordinary", "step_D": "PML"})


def _composed(driver: Any, fuse: bool,
              residency: Optional[Residency] = None) -> Tuple[Any, Residency]:
    """``plan_step`` on this driver, on ``residency`` or on a fresh one.

    ONE ARRANGEMENT IS ONE RESIDENCY, and that is not bookkeeping. The registry's
    whole purpose is that every plan touching a volume binds the SAME device tensor
    (device.py:157-164): ``step_B`` writes ``Bx`` where ``update_H`` reads it, and
    this weld writes ``D`` in place where ``update_E`` reads it. An arrangement whose
    slots were built on two registries would launch against two different device
    buffers for one host array and the copies back would overwrite each other -- a
    silent wrong answer, measured here on 2026-09-05: every word of every volume
    diverged at step 1 while the seam's own arithmetic was byte-exact.
    """
    residency = Residency() if residency is None else residency
    plan = metal_launch.plan_step(driver.fields, driver.pml, residency=residency,
                                  sources=tuple(getattr(driver, "_sources", ())),
                                  fuse=fuse)
    return plan, residency


def arrangement_composition(driver: Any, fuse: bool) -> Arrangement:
    """References 3 (``fuse=True``, the composer's own installation) and 4 (unfused).

    THE SHIPPED COMPOSER BUILDS THE PLANS, not this file: reaching for the family
    builders would measure the kernels and skip the composition, and the
    composition is what the arbitration finding is about. Its slot table and its
    refusal of THIS product are recorded on the arrangement.
    """
    plan, residency = _composed(driver, fuse)
    plans = {slot: synced(plan.plans[slot], residency)
             for slot in metal_launch.STEP_ORDER if slot in plan.plans}
    return Arrangement("composition_today" if fuse else "unfused",
                       Shim(driver.fields, plans), residency,
                       selected=plan.selected,
                       reasons={key: list(value) for key, value in plan.reasons.items()
                                if key.startswith("fused_pair")})


def build_weld(driver: Any, residency: Residency,
               functions: Optional[Mapping[str, Any]] = None,
               sources: Any = ()) -> Any:
    plan = family.plan_metal_fused_hd_pair(driver.fields, driver.pml, sources=sources,
                                           residency=residency, functions=functions)
    if plan is None:
        reasons = family.metal_fused_hd_pair_coverage(driver.fields, driver.pml,
                                                      sources, residency).reasons
        raise RuntimeError("the fused H/D pair was refused: " + "; ".join(reasons))
    return plan


def arrangement_weld(driver: Any, functions: Optional[Mapping[str, Any]] = None,
                     launcher: Optional[Callable[[Any, Residency, Any], Any]] = None,
                     sync_hazard: bool = False, hoist: bool = False,
                     rest_unfused: bool = True, name: str = "weld") -> Arrangement:
    """The SUBJECT: the weld force-installed at ``update_H``, ``step_D`` absorbed.

    FORCE-INSTALLED, because the composer refuses this product by name (no absorb
    row, ``INSTALLABLE = False``) and this gate must measure it anyway. Every other
    slot carries the unfused single ``plan_step(fuse=False)`` selects for it, so the
    only difference between this arrangement and reference 4 is the seam.

    ``launcher`` is the HOST mutation seam: it receives the built plan and returns
    the object the slot runs (a rotation-skipping wrapper, an aliasing launcher).
    ``functions`` is the SHADER mutation seam, handed to the family's own builder.
    ``hoist`` wraps the launch in the ``LeadingWithdrawPlan`` the installer would --
    the withdraw leg's arrangement.
    """
    residency = Residency()
    sources = tuple(getattr(driver, "_sources", ()))
    plan = build_weld(driver, residency, functions=functions, sources=())
    occupant: Any = (plan if launcher is None
                     else launcher(plan, residency, driver.pml))
    occupant = SyncedPlan(occupant, residency)
    if hoist:
        occupant = withdraw_hoist.LeadingWithdrawPlan(
            occupant, driver.fields, sources, span=family.REPLACES)
    plans: Dict[str, Any] = {}
    if rest_unfused:
        # ON THIS ARRANGEMENT'S OWN RESIDENCY -- see :func:`_composed`. The weld
        # writes D and fu_D IN PLACE and ``update_E`` reads them; a second registry
        # here would give that read a different device buffer.
        base, _base_residency = _composed(driver, fuse=False, residency=residency)
        for slot in metal_launch.STEP_ORDER:
            if slot in base.plans and slot not in family.REPLACES:
                plans[slot] = synced(base.plans[slot], residency)
    plans["update_H"] = occupant
    plans["step_D"] = ABSORBED
    shim = Shim(driver.fields, plans, sync_hazard=sync_hazard)
    return Arrangement(name, shim, residency,
                       selected={"update_H": "fused H/D pair (forced)",
                                 "step_D": "fused H/D pair (forced)"})


# ---------------------------------------------------------------------------
# The flush precondition's second half: the reference's own INTERMEDIATES
# ---------------------------------------------------------------------------

def subnormal_words(array: Any) -> int:
    """How many float32 words of ``array`` are in the denormal band, off the BITS.

    A VIEW, NOT A COPY, and on this leg that is a working-set decision rather than a
    micro-optimisation: the census runs on every intermediate of every helper call of
    every reference step, and the corpus carries an 11.2-million-cell row whose six
    arrangement snapshots already dominate this process's memory. Reading the words
    through ``view`` costs nothing; reading them through ``tobytes`` costs a full copy
    of every intermediate, which is how a measured row becomes a killed child.
    """
    magnitude = np.ascontiguousarray(array, dtype=np.float32).view(np.uint32)
    magnitude = magnitude & np.uint32(0x7FFFFFFF)
    return int(np.count_nonzero((magnitude != 0)
                                & (magnitude < np.uint32(0x00800000))))


class IntermediateCensus:
    """Every float32 intermediate the reference step forms, censused for the band.

    WHY THIS EXISTS, MEASURED RATHER THAN ARGUED. This gate's standing precondition
    censuses the STORED volumes after each reference step, on the rule that MPS
    flushes the float32 denormal band natively and NumPy does not, so past the first
    banded step no byte claim can be made. That census can only see a subnormal once
    one is WRITTEN TO A STORED VOLUME -- and the flush that decides a comparison need
    not leave one there.

    On ``examples:gaussian-beam.py`` at step 50 and
    ``tests:TestEigenmodeSource.test_waveguide_flux`` at step 30 the array path and
    EVERY dispatched arrangement -- the certified singles included -- disagreed by 1
    to 2 units in the last place at magnitude ~1e-32, in mirror-symmetric cells inside
    the absorber, with a stored-state census of 0 on every compared step and the
    reference's IEEE underflow flag never raised over the whole walk. Bisected
    2026-09-06 on ``gaussian-beam.py``: ``step_D`` alone reproduces it and ``update_H``
    alone is clean, and at cell (52, 101, 0) and its mirror (648, 101, 0) the curl's
    own operand difference ``shifted_first - first`` is ``-6.64085431592474e-39``
    (word ``0x80485000``), a float32 SUBNORMAL. Flushing that one intermediate to zero
    and running the rest of the certified arithmetic in float32 in the shipped order
    reproduces the device's words bit for bit -- curl ``0xa8044ba`` -> ``0xa8044bc``,
    the recurrence's inner term ``0x8a81fe31`` -> ``0x8a81fe33``, and ``Dz``/``fu_Dz``
    ``0x8a8014f5`` -> ``0x8a8014f7``, which IS what the device wrote.

    So it is the SAME platform fact the precondition already refuses on, one level
    below where it was looking, and the IEEE flag cannot see it either: 754 signals
    underflow only when a result is tiny AND INEXACT, and the difference of two nearby
    normals into the subnormal range is EXACT.

    THE ARITHMETIC IS NOT TRANSCRIBED ON TRUST. Each wrapper recomputes the helper's
    own chain in float32 from the operands and compares its result to what the helper
    actually produced; a mismatch RAISES, so a census that stopped modelling the
    engine fails loudly instead of quietly censusing the wrong expression.

    WHAT IT COVERS is the three helpers the ``(ordinary -> PML)`` cell's arithmetic
    runs through. The two conductive paths are COUNTED and not modelled, and a step
    that reached one reports ``covered`` False rather than a census it did not take.
    """

    #: Helpers whose intermediates are modelled, and the two that are only counted.
    MODELLED: Tuple[str, ...] = ("_curl_from_operands", "_apply_pml_update",
                                 "_apply_constitutive_pml")
    COUNTED_ONLY: Tuple[str, ...] = ("_apply_conductive_update",
                                     "_apply_conductive_pml_update")

    def __init__(self) -> None:
        self.subnormals: Dict[str, int] = {}
        self.calls = 0
        self.uncovered_calls = 0
        self._originals: Dict[str, Any] = {}

    # -- the modelled chains -------------------------------------------------
    def _note(self, name: str, array: Any) -> Any:
        count = subnormal_words(array)
        if count:
            self.subnormals[name] = self.subnormals.get(name, 0) + count
        return array

    @property
    def total(self) -> int:
        return int(sum(self.subnormals.values()))

    def __enter__(self) -> "IntermediateCensus":
        from meep_gpu import stepping  # noqa: PLC0415

        f32 = np.float32
        for name in self.MODELLED + self.COUNTED_ONLY:
            self._originals[name] = getattr(stepping, name)

        original_curl = self._originals["_curl_from_operands"]
        original_pml = self._originals["_apply_pml_update"]
        original_constitutive = self._originals["_apply_constitutive_pml"]

        def curl(operands: Any, dtdx: Any, scratch: Any = None) -> Any:
            self.calls += 1
            first = np.array(operands.first, copy=True)
            shifted_first = np.array(operands.shifted_first, copy=True)
            second = np.array(operands.second, copy=True)
            shifted_second = np.array(operands.shifted_second, copy=True)
            out = original_curl(operands, dtdx, scratch)
            if out.dtype != np.float32:
                return out
            # EACH INTERMEDIATE IS RELEASED AS SOON AS THE NEXT ONE HOLDS ITS VALUE.
            # Holding the whole chain at once triples this process's peak on the
            # corpus's largest row for no measurement.
            head = self._note("curl:shifted_first-first",
                              (shifted_first - first).astype(f32))
            del first, shifted_first
            tail = self._note("curl:second-shifted_second",
                              (second - shifted_second).astype(f32))
            del second, shifted_second
            total = self._note("curl:sum", (head + tail).astype(f32))
            del head, tail
            scaled = self._note("curl:dtdx*sum", (f32(dtdx) * total).astype(f32))
            del total
            _agrees("_curl_from_operands", scaled, out)
            return out

        def pml(field: Any, curl_value: Any, kms: Any, sinv: Any, kms_u: Any,
                sinv_u: Any, fu: Any, scratch: Any = None) -> Any:
            self.calls += 1
            field_in = np.array(field, copy=True)
            fu_in = np.array(fu, copy=True)
            curl_in = np.array(curl_value, copy=True)
            result = original_pml(field, curl_value, kms, sinv, kms_u, sinv_u, fu,
                                  scratch=scratch)
            if field_in.dtype != np.float32:
                return result
            product = self._note("pml:fu*kms", (fu_in * kms).astype(f32))
            inner = self._note("pml:fu*kms-curl", (product - curl_in).astype(f32))
            del product, curl_in
            fu_new = self._note("pml:(fu*kms-curl)*sinv", (inner * sinv).astype(f32))
            del inner
            scaled = self._note("pml:field*kms_u", (field_in * kms_u).astype(f32))
            del field_in
            summed = self._note("pml:field*kms_u+fu", (scaled + fu_new).astype(f32))
            del scaled
            outer = self._note("pml:+fu-fu_prev", (summed - fu_in).astype(f32))
            del summed, fu_in
            field_new = self._note("pml:*sinv_u", (outer * sinv_u).astype(f32))
            del outer
            _agrees("_apply_pml_update:fu", fu_new, fu)
            _agrees("_apply_pml_update:field", field_new, field)
            return result

        def constitutive(field: Any, source: Any, kps: Any, kms: Any, fw: Any,
                         scratch: Any = None) -> Any:
            self.calls += 1
            field_in = np.array(field, copy=True)
            source_in = np.array(source, copy=True)
            fw_in = np.array(fw, copy=True)
            result = original_constitutive(field, source, kps, kms, fw, scratch=scratch)
            if field_in.dtype != np.float32:
                return result
            product = self._note("constitutive:kps*fw", (kps * source_in).astype(f32))
            del source_in
            summed = self._note("constitutive:field+kps*fw",
                                (field_in + product).astype(f32))
            del field_in, product
            previous = self._note("constitutive:kms*fw_prev", (kms * fw_in).astype(f32))
            del fw_in
            field_new = self._note("constitutive:-kms*fw_prev",
                                   (summed - previous).astype(f32))
            del summed, previous
            _agrees("_apply_constitutive_pml:field", field_new, field)
            return result

        def counted(name: str, original: Any) -> Any:
            def wrapper(*arguments: Any, **keywords: Any) -> Any:
                self.calls += 1
                self.uncovered_calls += 1
                return original(*arguments, **keywords)
            wrapper.__name__ = f"counted_{name}"
            return wrapper

        stepping._curl_from_operands = curl  # noqa: SLF001 - the oracle's own seam
        stepping._apply_pml_update = pml  # noqa: SLF001
        stepping._apply_constitutive_pml = constitutive  # noqa: SLF001
        for name in self.COUNTED_ONLY:
            setattr(stepping, name, counted(name, self._originals[name]))
        return self

    def __exit__(self, *_exception: Any) -> None:
        from meep_gpu import stepping  # noqa: PLC0415

        for name, original in self._originals.items():
            setattr(stepping, name, original)

    def report(self) -> Dict[str, Any]:
        return {"subnormal_intermediates": self.total,
                "by_expression": dict(sorted(self.subnormals.items())),
                "helper_calls": self.calls,
                "calls_this_census_does_not_model": self.uncovered_calls,
                "covered": self.uncovered_calls == 0}


def _agrees(what: str, recomputed: Any, actual: Any) -> None:
    """The census's model of a helper must reproduce that helper's own bytes."""
    if differing(recomputed, actual):
        raise AssertionError(
            f"the intermediate census no longer models {what}: its float32 "
            f"recomputation differs from what the engine produced, so its subnormal "
            f"count is a census of the wrong expression. Fix the model, never the "
            f"comparison")


def drive(driver: Any, arrangements: Mapping[str, Arrangement], steps: int,
          reference: str = "array", secondary: Optional[str] = None,
          per_step_hook: Optional[Callable[[int, str, Any], Any]] = None,
          progress: Optional[Callable[[str], None]] = None,
          stop_on_divergence: bool = True) -> Dict[str, Any]:
    """Step every arrangement in lockstep from one seed and compare per complete step.

    ONE driver, restored to each arrangement's own state before its step, captured
    after it -- the withdraw campaign's proven mechanism, extended with the settled
    rotation. ``per_step_hook(step, name, driver)`` runs after each arrangement's
    step (the sync leg's accessor call); its return value is recorded.
    """
    seed = capture(driver)
    # THE ARRAY OBJECT THE ENGINE NAMES FOR EACH STORED VOLUME, BEFORE ANY LAUNCH.
    # Every arrangement's device mirrors are bound to these by identity, and
    # ``capture`` reads whatever ``fields`` names at the end of a step, so a rotation
    # a step leaves standing silently decouples the two: the launch writes the array
    # the mirror holds and the comparison reads the twin. That is not a hypothesis --
    # it is the 2026-09-05 ``cavity_arrayslice`` defect, and it degrades a divergence
    # AND an agreement, so it is measured on every arrangement of every step rather
    # than argued.
    identities = {name: id(array)
                  for name, array in stored_volumes(driver.fields).items()}
    left_rotated: Dict[str, List[str]] = {name: [] for name in arrangements}
    orphaned: Dict[str, List[str]] = {name: [] for name in arrangements}
    states = {name: seed for name in arrangements}
    per_step: Dict[str, List[Dict[str, Any]]] = {name: [] for name in arrangements
                                                  if name != reference}
    autopsies: Dict[str, Dict[str, Any]] = {}
    reference_underflow: List[Dict[str, Any]] = []
    intermediate_census: List[Dict[str, Any]] = []
    hooks: Dict[str, List[Any]] = {name: [] for name in arrangements}
    lazily: List[str] = []
    census_per_step: List[Dict[str, int]] = []
    steps_done = 0
    # STEPPED IS NOT COMPARED. The banded step is taken -- every arrangement advances
    # through it -- and then refused as a comparison, so the launch counters answer to
    # this number and the byte claim answers to ``steps_done``. Scoring the counters
    # against the compared count would report the weld as having missed a launch it
    # in fact performed.
    steps_stepped = 0
    error: Optional[str] = None
    started = time.time()
    try:
        for step in range(1, steps + 1):
            for name, arrangement in arrangements.items():
                lazily.extend(restore(driver, states[name]))
                install(driver, arrangement.shim)
                if name == reference:
                    # THE ORACLE'S ARITHMETIC, NOT ONLY ITS STATE. The stored-state
                    # census below sees a subnormal only once one is WRITTEN; a
                    # product that underflows and is then added to a normal word
                    # leaves no subnormal behind, and the two executors still part
                    # company there because one keeps it and the other does not. The
                    # IEEE underflow flag is raised by the reference's own ufuncs, so
                    # this is a measurement of the oracle, not of a device.
                    underflowed = False

                    def _flag(kind: str, _bits: int) -> None:  # noqa: ANN001
                        nonlocal underflowed
                        if kind == "underflow":
                            underflowed = True

                    # AND THE INTERMEDIATES, WHICH NEITHER OF THE OTHER TWO SEES.
                    # See :class:`IntermediateCensus`: an intermediate that is tiny
                    # and EXACT raises no IEEE flag and leaves no subnormal in a
                    # stored volume, and it is what put two corpus rows 1-2 ULP apart.
                    with IntermediateCensus() as intermediates:
                        with np.errstate(under="call", over="ignore", divide="ignore",
                                         invalid="ignore", call=_flag):
                            driver.step()
                    intermediate_census.append({"step": step, **intermediates.report()})
                    reference_underflow.append({"step": step,
                                                "reference_underflowed": underflowed})
                else:
                    driver.step()
                if per_step_hook is not None:
                    hooks[name].append(per_step_hook(step, name, driver))
                arrangement.settle(driver.fields)
                moved = [volume for volume, array
                         in stored_volumes(driver.fields).items()
                         if volume in identities and identities[volume] is not None
                         and identities[volume] != id(array)]
                if moved:
                    left_rotated[name] = sorted(set(left_rotated[name]) | set(moved))
                orphans = orphaned_mirrors(arrangements, driver.fields)
                if orphans:
                    orphaned[name] = sorted(set(orphaned[name]) | set(orphans))
                states[name] = capture(driver)
            steps_stepped = step
            # THE PRECONDITION IS TAKEN PER STEP, BEFORE THE COMPARISON, AND THE
            # ORDER IS THE WHOLE POINT. Byte identity on this backend is claimed
            # subject to the oracle's state staying out of the float32 denormal band
            # -- MPS flushes natively and NumPy does not. A census read after the
            # comparison would still RECORD the banded step's comparison, and that
            # step is exactly the one the two paths are expected to disagree on, so a
            # run cut short by the precondition would report a divergence it was never
            # entitled to look for. The banded step is counted and named; it is not
            # compared, and nothing past it is measurable.
            #
            # THE BAND IS ENTERED IN TWO PLACES AND BOTH STOP THE WALK. The stored
            # census is the state the step left behind; the intermediate census is the
            # arithmetic that produced it. A flushed intermediate moves a stored word
            # by one or two units in the last place while leaving nothing subnormal
            # anywhere and raising no IEEE flag -- measured, see
            # :class:`IntermediateCensus` -- so a precondition that read only the
            # stored words would go on comparing a step neither executor can be held
            # to. Which half fired is recorded, because they are different facts.
            census = int(sum(subnormal.census(value)
                             for value in states[reference]["volumes"].values()))
            formed = (intermediate_census[-1] if intermediate_census
                      else {"subnormal_intermediates": 0, "covered": True})
            census_per_step.append({"step": step, "reference_subnormals": census,
                                    "reference_subnormal_intermediates":
                                        int(formed["subnormal_intermediates"])})
            if census or formed["subnormal_intermediates"]:
                if progress is not None:
                    progress(f"step {step}/{steps} NOT COMPARED: the oracle entered "
                             f"the denormal band (stored {census} words, formed "
                             f"{formed['subnormal_intermediates']} words in "
                             f"{sorted(formed.get('by_expression') or ())}); the walk "
                             f"stops ({time.time() - started:.1f} s)")
                break
            for name in per_step:
                difference = compare_snapshots(states[reference], states[name])
                row: Dict[str, Any] = {
                    "step": step,
                    "differing_words": int(sum(difference.values())),
                    "differing_volumes": difference}
                # AND AGAINST ONE OTHER ARRANGEMENT, WHICH SEPARATES TWO FINDINGS THE
                # ARRAY-PATH COMPARISON ALONE CONFLATES. Every dispatched arrangement
                # shares this gate's non-seam plans, so a defect in a REFERENCE shows
                # up on the weld's row too and reads as "the weld disagrees". The
                # weld against the certified singles is device-to-device under one
                # policy, so it answers the question this gate is actually about:
                # does the fused launch compute what the two singles compute?
                if secondary is not None and secondary in states and name != secondary:
                    against = compare_snapshots(states[secondary], states[name])
                    row[f"differing_words_vs_{secondary}"] = int(sum(against.values()))
                    row[f"differing_volumes_vs_{secondary}"] = against
                # THE FIRST DIVERGENCE IS TRANSCRIBED, ONCE PER ARRANGEMENT. A count
                # cannot tell a flushed subnormal from a wrong answer, and this gate
                # has already had to attribute one divergence to the executor's flush
                # policy rather than to a weld. Recording the words at the moment they
                # first differ costs one pass over the differing indices and removes
                # the argument.
                if difference and name not in autopsies:
                    autopsies[name] = {
                        "step": step,
                        "against": reference,
                        "volumes": autopsy(states[reference], states[name], difference)}
                    if secondary is not None and name != secondary and against:
                        autopsies[name]["volumes_vs_secondary"] = autopsy(
                            states[secondary], states[name], against)
                per_step[name].append(row)
            steps_done = step
            if progress is not None:
                progress(f"step {step}/{steps} "
                         + " ".join(f"{name}={rows[-1]['differing_words']}"
                                    for name, rows in per_step.items())
                         + f" subnormals=0 ({time.time() - started:.1f} s)")
            if stop_on_divergence and any(rows[-1]["differing_words"]
                                          for rows in per_step.values()):
                break
    except Exception as exc:  # noqa: BLE001 - a row that cannot be stepped is named
        error = f"{type(exc).__name__}: {exc}"[:600]
    finally:
        pin_array_path(driver)
    reference_state = states[reference]
    moved = {name: differing(seed["volumes"][name], reference_state["volumes"][name])
             for name in seed["volumes"]}
    banded = next((row["step"] for row in census_per_step
                   if row["reference_subnormals"]
                   or row["reference_subnormal_intermediates"]), None)
    legs: Dict[str, Any] = {}
    for name, rows in per_step.items():
        first = next((row["step"] for row in rows if row["differing_words"]), None)
        legs[name] = {
            # "NO STEP THAT WAS COMPARED DISAGREED", and the BUDGET is a separate
            # question the caller settles. A walk stops for two different reasons --
            # a divergence, or the oracle entering the denormal band, past which
            # nothing is measurable -- and folding the step count in here would make
            # a run that was cut short by the precondition indistinguishable from one
            # that disagreed. ``steps_compared`` and ``first_banded_step`` are beside
            # this so the caller can require the full budget (the synthetic fixture,
            # which is scaled to clear it) or a floor (a lifted corpus row, whose own
            # state the gate does not choose).
            "identical": bool(rows and all(not row["differing_words"]
                                           for row in rows)),
            "identical_vs_secondary": (
                None if secondary is None or name == secondary
                else bool(rows and all(not row.get(f"differing_words_vs_{secondary}")
                                       for row in rows))),
            "steps_compared": len(rows),
            "first_divergence": first,
            "differing_words_final": rows[-1]["differing_words"] if rows else None,
            "differing_volumes_final": rows[-1]["differing_volumes"] if rows else {},
            "launches": arrangements[name].launches(),
            "dispatched": dict(arrangements[name].shim.dispatched)
            if arrangements[name].shim else {},
            "absorbed": dict(arrangements[name].shim.absorbed)
            if arrangements[name].shim else {},
            "sync_refusals": (arrangements[name].shim.sync_refusals
                              if arrangements[name].shim else 0),
            "sync_answered": (arrangements[name].shim.sync_answered
                              if arrangements[name].shim else 0),
            "selected": arrangements[name].selected,
            "rotating_plans": len(arrangements[name].rotating),
            "engine_volumes_left_rotated": left_rotated[name],
            "hook": hooks[name],
            "first_divergence_autopsy": autopsies.get(name),
            "per_step": rows,
        }
    return {
        "steps_requested": steps,
        "steps_compared": steps_done,
        "steps_stepped": steps_stepped,
        "step_error": error,
        "volumes_compared": sorted(seed["volumes"]),
        "words_per_step": int(sum(words(value).size for value in seed["volumes"].values())),
        "words_compared": int(sum(words(value).size for value in seed["volumes"].values())
                              * steps_done * len(per_step)),
        "lazily_allocated_volumes": sorted(set(lazily)),
        "moved_from_seed": {name: n for name, n in sorted(moved.items()) if n},
        "arrays_that_never_moved": sorted(name for name, n in moved.items() if not n),
        "reference_subnormals_per_step": census_per_step,
        "reference_underflow_per_step": reference_underflow,
        "first_step_the_reference_underflowed": next(
            (row["step"] for row in reference_underflow
             if row["reference_underflowed"]), None),
        "reference_subnormals": (census_per_step[-1]["reference_subnormals"]
                                 if census_per_step else 0),
        "reference_intermediates_per_step": intermediate_census,
        "first_step_an_intermediate_entered_the_band": next(
            (row["step"] for row in intermediate_census
             if row["subnormal_intermediates"]), None),
        "which_half_of_the_precondition_fired": (
            None if banded is None else sorted(
                half for half, fired in (
                    ("stored state", any(row["reference_subnormals"]
                                         for row in census_per_step)),
                    ("intermediates", any(row["reference_subnormal_intermediates"]
                                          for row in census_per_step)))
                if fired)),
        "the_intermediate_census_modelled_every_helper_the_step_ran": all(
            row["covered"] for row in intermediate_census) if intermediate_census else None,
        "helper_calls_the_intermediate_census_does_not_model": int(sum(
            row["calls_this_census_does_not_model"] for row in intermediate_census)),
        "first_banded_step": banded,
        "precondition_clean": banded is None,
        "reference_hook": hooks[reference],
        "engine_volumes_left_rotated": {name: volumes for name, volumes
                                        in left_rotated.items() if volumes},
        "mirrored_volumes_left_orphaned": {name: volumes for name, volumes
                                           in orphaned.items() if volumes},
        # THE VERDICT IS THE ORPHAN CHECK, NOT THE RENAME. See :func:`orphaned_mirrors`
        # for why, and for the measurement that separated the two.
        "every_arrangement_gave_the_engine_its_volumes_back": not any(
            orphaned.values()),
        "rotating_plans_per_arrangement": {name: len(arrangement.rotating)
                                           for name, arrangement
                                           in arrangements.items()},
        "seconds": round(time.time() - started, 2),
        "arrangements": legs,
    }


def in_place_words(row: Mapping[str, Any]) -> int:
    return int(sum(count for name, count in row["differing_volumes"].items()
                   if name.startswith(("Dx", "Dy", "Dz", "fu_D"))))


# ---------------------------------------------------------------------------
# The armed defects
# ---------------------------------------------------------------------------

def needle(source: str, old: str, new: str, count: int = 1) -> str:
    """Replace exactly ``count`` occurrences and REFUSE a no-op or ambiguous edit.

    A mutation that changed nothing would launch the shipped kernel and report the
    defect as uncaught, which is the one failure a mutation leg cannot see from its
    own result.
    """
    hits = source.count(old)
    if hits != count:
        raise AssertionError(f"mutation needle matches {hits} times, expected {count}: "
                             f"{old!r}")
    mutated = source.replace(old, new)
    if mutated == source:
        raise AssertionError(f"mutation needle changed nothing: {old!r}")
    return mutated


#: A tap's call, built the way the family builds it, so the needles cannot drift.
def _tap_call(coords: str, target: int) -> str:
    return f"h_cell({coords}, {family.H_CELL_TAIL_ARGS}).a{target}"


_A_Y = f"    float a_y = vy ? {_tap_call('i, sj, k', 0)} : 0.0f;\n"
_ACCUMULATION_0 = ("        a0 = a0 + kp_0 * src0;\n"
                   "        a0 = a0 - km_0 * prev0;\n")


def shader_mutations(codes: Sequence[int]) -> Dict[str, Tuple[str, str]]:
    """Every armed source defect as ``name -> (source, why it must fire)``.

    Every needle is anchored on text only the mutated line carries and is verified
    to resolve EXACTLY ONCE before it is applied -- an absent or doubled needle
    raises here rather than arming nothing. The host suite re-asserts the anchors
    on every test run.
    """
    base = family.fused_hd_pair_source(codes)
    out: Dict[str, Tuple[str, str]] = {}

    def arm(name: str, source: str, why: str) -> None:
        out[name] = (source, why)

    # THE LEGS THIS FAMILY EXISTS FOR: the foreign tap.
    arm("foreign_tap_reads_stale_H",
        needle(base, _A_Y, "    float a_y = vy ? hi0[oy] : 0.0f;\n"),
        "the tap reads the PRE-LAUNCH H at the neighbour instead of recomputing "
        "update_H there; fires wherever update_H moves that cell (the purity ledger)")
    arm("foreign_tap_reads_B",
        needle(base, _A_Y, "    float a_y = vy ? b0[oy] : 0.0f;\n"),
        "the tap reads the flux density where the curl needs the stepped H")
    arm("foreign_tap_reads_the_scratch_output",
        needle(base, _A_Y, "    float a_y = vy ? ho0[oy] : 0.0f;\n"),
        "THE PLANTED RACE: the tap reads another thread's store, which holds either "
        "the neighbour's new H (if that thread ran first) or two-launches-old scratch")
    arm("halo_recompute_dropped",
        needle(base, _A_Y, "    float a_y = 0.0f;\n"),
        "one shifted magnetic load replaced by the ghost value everywhere")
    arm("halo_moved_to_the_forward_neighbour",
        needle(base, "    int si = i - 1, sj = j - 1, sk = k - 1;\n",
               "    int si = i - 1, sj = j + 1, sk = k - 1;\n"),
        "step_D is the BACKWARD curl; one axis differenced forward is step_B's stencil")
    arm("own_cell_reads_stale_H_not_the_register",
        needle(base, "    float a   = own.a0;\n", "    float a   = hi0[ii];\n"),
        "the curl takes the pre-launch H at its own cell: the seam undone")
    # THE CONSTITUTIVE HALF, inside h_cell (8-space indent: the lifted body).
    arm("constitutive_accumulations_regrouped",
        needle(base, _ACCUMULATION_0,
               "        a0 = a0 + (kp_0 * src0 - km_0 * prev0);\n"),
        "((f + kps*src) - kms*prev) regrouped to f + (kps*src - kms*prev) is a "
        "different float32 number")
    arm("constitutive_accumulations_reversed",
        needle(base, _ACCUMULATION_0,
               "        a0 = a0 - km_0 * prev0;\n        a0 = a0 + kp_0 * src0;\n"),
        "the two accumulations in the other order")
    arm("prev_reads_the_value_the_store_writes",
        needle(base, "        float prev0 = wi0[ii];\n",
               "        float prev0 = b0[ii];\n"),
        "the split-field history read AFTER the store that overwrote it with B; "
        "fires inside the PML only, where kms != 0")
    arm("kps_kms_swapped",
        needle(base, "        float kp_0 = kp0[i], km_0 = kmx[i];\n",
               "        float kp_0 = kmx[i], km_0 = kp0[i];\n"),
        "the two absorber coefficients exchanged on one component")
    arm("constitutive_coefficient_index_moved_off_the_own_axis",
        needle(base, "        float kp_0 = kp0[i], km_0 = kmx[i];\n",
               "        float kp_0 = kp0[j], km_0 = kmx[j];\n"),
        "THE CONTROL FOR THE PREDICTED NULL BELOW: the dsigw index is the component's "
        "OWN axis; moved to another axis it is a silent half-cell error and MUST fire")
    arm("h_store_dropped",
        needle(base, "    ho0[ii] = own.a0; ho1[ii] = own.a1; ho2[ii] = own.a2;\n",
               "    ho1[ii] = own.a1; ho2[ii] = own.a2;\n"),
        "one magnetic component never reaches the scratch; the next step reads stale")
    arm("fw_store_dropped",
        needle(base, "    wo0[ii] = own.src0; wo1[ii] = own.src1; wo2[ii] = own.src2;\n",
               "    wo1[ii] = own.src1; wo2[ii] = own.src2;\n"),
        "one split-field history never advances")
    # THE CURL HALF.
    arm("curl_parens_flattened",
        needle(base, "dtdx * ((c_y - c) + (b - b_z))", "dtdx * (c_y - c + b - b_z)"),
        "shaders.py rule 2: reassociation is a different float32 number")
    arm("ownership_mask_dropped",
        needle(base, "    curl0 = at_y ? 0.0f : curl0;\n", ""),
        "the non-owned plane keeps a curl the array path masks")
    arm("recurrence_axis_pair_swapped",
        needle(base, "    float n0 = ((p0 * km_y) - curl0) * si_y;\n",
               "    float n0 = ((p0 * km_z) - curl0) * si_z;\n"),
        "vec.hpp's cycle_direction: target 0 takes (y, z)")
    arm("fu_store_dropped",
        needle(base, "    u0[ii] = n0; u1[ii] = n1; u2[ii] = n2;\n",
               "    u1[ii] = n1; u2[ii] = n2;\n"),
        "the split-field auxiliary never advances on one component")
    arm("flux_store_dropped",
        needle(base, "    f0[ii] = v0; f1[ii] = v1; f2[ii] = v2;\n",
               "    f1[ii] = v1; f2[ii] = v2;\n"),
        "one displacement component never advances")
    arm("contraction_pragma_removed",
        needle(base, shaders.contraction_pragma(shaders.CONTRACT_OFF) + "\n", "\n"),
        "without the pragma the toolchain may contract a multiply-add into an fma; "
        "shaders.py carries the pragma because that moved bytes")
    return out


def periodic_wrap_mutations(codes: Sequence[int]) -> Dict[str, Tuple[str, str]]:
    """The boundary defects only an ALL-PERIODIC specialisation can carry.

    WHY THIS SET EXISTS AT ALL, and it is a measurement rather than a preference.
    The metallic ghost's VALUE is unobservable through ``step_D``'s output: every
    shifted tap along an axis feeds the ONE curl component
    ``stepping._mask_non_owned_cells`` zeroes at index 0 of that same axis, and on a
    metallic axis that mask fires (stepping.py:1940-1948, ``grid.is_metallic``). So a
    mutant that replaces the exact ``0.0f`` past a metallic wall changes only words
    the mask overwrites -- measured, not argued: it was armed on ``bc_mmm``, launched,
    and came back byte-identical over the whole budget.

    On a PERIODIC axis that same mask does NOT fire, so cell 0's curl is a live
    stepped value assembled from the WRAPPED row -- and that is where the boundary
    specialisation is observable. Arming the wrap here is what keeps "the ghost rule
    is the emitter's own" a measurement on the boundary plane rather than only on the
    interior, which is all ``halo_recompute_dropped`` reaches.
    """
    base = family.fused_hd_pair_source(codes)
    return {
        "periodic_wrap_reads_the_own_row": (
            needle(base, "    sj = (sj < 0) ? (nyi - 1) : sj;\n",
                   "    sj = (sj < 0) ? 0 : sj;\n"),
            "the backward y tap at j = 0 wraps to the LAST stored row; reading row 0 "
            "instead is the periodic rule dropped, and on a periodic axis the "
            "ownership mask does not cover that plane"),
        "periodic_wrap_reads_one_row_short": (
            needle(base, "    sj = (sj < 0) ? (nyi - 1) : sj;\n",
                   "    sj = (sj < 0) ? (nyi - 2) : sj;\n"),
            "the wrap lands one row short of the last: a plausible off-by-one that "
            "keeps the branch and moves the cell"),
    }


def inert_mutations(codes: Sequence[int]) -> Dict[str, Tuple[str, str, str]]:
    """Armed defects required NOT to fire, each with its reason and its control.

    A defect that cannot be observed is a fact about the kernel, and the honest place
    to record it is a ROW THAT RAN rather than a sentence in a table. These mutants
    are compiled, installed and launched exactly like the firing ones; the leg scores
    them on AGREEING, and requires the named control -- which reaches the same rule
    where it IS observable -- to have fired.
    """
    base = family.fused_hd_pair_source(codes)
    return {
        "metallic_ghost_replaced_by_the_own_cells_constitutive": (
            needle(base, _A_Y,
                   f"    float a_y = vy ? {_tap_call('i, sj, k', 0)} : own.a0;\n"),
            "INERT BY CONSTRUCTION, and it is a property of the ARRAY PATH rather "
            "than of this weld. `a_y` feeds `curl2` and nothing else, and "
            "`stepping._mask_non_owned_cells` (stepping.py:1940-1948) zeroes curl2 "
            "wherever j == 0 on a metallic y axis -- which is exactly the plane where "
            "`vy` is false and the exact 0.0f past the wall is served. So no edit to "
            "a metallic ghost VALUE can reach this kernel's output; every shifted tap "
            "has the same shape. The same boundary rule IS observable on a PERIODIC "
            "axis, where that mask does not fire, and that is the control",
            "periodic_wrap_reads_the_own_row"),
    }


#: Defects that CANNOT be armed as a text edit at all, recorded with the reason
#: rather than dropped, each paired with the armed control that fires.
PREDICTED_NULL: Tuple[Dict[str, str], ...] = (
    {"name": "foreign_tap_coefficient_index_shifted_along_the_shifted_axis",
     "reason": ("inert by construction and NOT ARMABLE as a shader edit: h_cell "
                "indexes kps/kms at the CELL'S OWN coordinate on the component's own "
                "axis (kp0[i] for component 0), and a curl never differences a "
                "component along its own axis, so a tap shifted along y or z reads "
                "the same coefficient row as the own cell -- there is no text whose "
                "edit shifts the coefficient without also moving the cell. Asserted "
                "structurally by test_metal_fused_hd_pair::"
                "test_a_curl_never_differences_a_component_along_its_own_axis"),
     "control": "constitutive_coefficient_index_moved_off_the_own_axis"},
)


def byte_neutral_source(codes: Sequence[int]) -> str:
    """The own-cell register reads replaced by RELOADS of the scratch just stored.

    Not a defect: ``ho0[ii] = own.a0`` executes three lines above, so the reload is
    the same float32 word. This is the weld's central claim written as a program --
    "the fusion removes a round trip and changes no arithmetic" -- and the leg
    requires it NOT to diverge.
    """
    source = family.fused_hd_pair_source(codes)
    for target, var in enumerate(("a", "b", "c")):
        source = needle(source, f"    float {var}   = own.a{target};\n",
                        f"    float {var}   = ho{target}[ii];\n")
    return source


#: Where each binding group starts in ``plan.static_args``: flux(3) + displacement(3)
#: + auxiliary(3), then the six curl coefficients, then the three kps.
CURL_COEFFICIENT_STATIC_SLOTS = tuple(range(9, 15))
CONSTITUTIVE_COEFFICIENT_STATIC_SLOTS = tuple(range(15, 18))


class RotationSkipped:
    """HOST DEFECT: launch, then put the engine's references back where they were.

    The engine keeps naming the PRE-launch buffers, so the constitutive half's output
    is invisible under the engine's own name (diverges over every volume at step 1)
    and the SECOND launch reads an H nobody advanced (diverges over the in-place
    displacement at step 2). Both numbers are recorded because they are two claims.
    """

    # ``launches`` and ``absorbed_by`` are PROPERTIES forwarding to the wrapped plan
    # and must not be slots.
    __slots__ = ("inner", "fields", "launches_per_run", "replaces_sub_steps",
                 "rotated_names", "rotated", "_functions")

    def __init__(self, plan: Any) -> None:
        self.inner = plan
        self.fields = plan.fields
        self.rotated_names = plan.rotated_names
        self.rotated = plan.rotated
        self.replaces_sub_steps = plan.replaces_sub_steps
        self.launches_per_run = plan.launches_per_run
        self._functions = plan._functions  # noqa: SLF001

    @property
    def absorbed_by(self) -> Any:
        """The plan that did the DEVICE work, for :func:`declaring`.

        Without it the counters resolve to this wrapper: ``count_functions`` would
        replace THIS object's ``_functions`` while the launch reads the wrapped
        plan's, so the independent function counter would report zero for the very
        launch it exists to witness -- and the two-counter agreement, which is what
        stands between a mutation leg and a hollow pass, would fail for a harness
        reason on every host mutation.
        """
        return self.inner

    @property
    def launches(self) -> int:
        return self.inner.launches

    def run(self) -> None:
        held = {name: getattr(self.fields, name) for name in self.rotated_names}
        self.inner.run()
        for name, value in held.items():
            self.inner.rotated[name] = getattr(self.fields, name)
            setattr(self.fields, name, value)


class AliasedLaunch:
    """HOST DEFECT: bind the named rotating volumes IN PLACE (scratch := live buffer).

    The kernel's own argument order, resolved the plan's own way, with the write slot
    of each aliased name pointed at its read slot. Aliased names are not rotated (the
    write landed in place); the rest rotate exactly as the plan rotates them. With
    ``f_w_H`` aliased this is "f_w_H written in place": a foreign recompute at a
    neighbour whose thread already stored reads B where it needs B_prev.
    """

    __slots__ = ("inner", "alias", "fields", "launches_per_run", "replaces_sub_steps",
                 "rotated_names", "rotated", "_functions")

    def __init__(self, plan: Any, alias: Sequence[str]) -> None:
        self.inner = plan
        self.alias = tuple(alias)
        self.fields = plan.fields
        self.rotated_names = plan.rotated_names
        self.rotated = plan.rotated
        self.replaces_sub_steps = plan.replaces_sub_steps
        self.launches_per_run = plan.launches_per_run
        self._functions = plan._functions  # noqa: SLF001
        missing = sorted(set(self.alias) - set(self.rotated_names))
        if missing:
            raise ValueError(f"{missing} are not rotating volumes of this plan")

    @property
    def absorbed_by(self) -> Any:
        """The plan that did the device work -- see :class:`RotationSkipped`."""
        return self.inner

    @property
    def launches(self) -> int:
        return self.inner.launches

    def run(self) -> None:
        plan = self.inner
        writes, reads = plan._resolve()  # noqa: SLF001 - the plan's own binding
        for index, name in enumerate(plan.rotated_names):
            if name in self.alias:
                writes[index] = reads[index]
        function = plan._functions[shaders.CONTRACT_OFF]  # noqa: SLF001
        plan.launches += 1
        function(*(list(writes) + list(reads) + list(plan.static_args)))
        for name in plan.rotated_names:
            if name in self.alias:
                continue
            current = getattr(plan.fields, name)
            setattr(plan.fields, name, plan.rotated[name])
            plan.rotated[name] = current


def rebind_static(plan: Any, slots: Sequence[int], tensors: Sequence[Any]) -> None:
    static = list(plan.static_args)
    for slot, tensor in zip(slots, tensors):
        static[slot] = tensor
    plan.static_args = tuple(static)


# THE LAUNCHER SEAM'S SIGNATURE IS ``(plan, residency, pml)`` AND THE THIRD ARGUMENT
# IS NOT A CONVENIENCE. Two of these defects rebind an absorber coefficient group and
# so need the PML the plan was built on. That used to be looked up in a module dict
# keyed by ``id(plan)`` which was filled AFTER the launcher had already run, so the
# lookup resolved only when CPython happened to reuse a freed plan's id -- which is to
# say it resolved to a PML belonging to a DIFFERENT driver whenever it resolved at
# all. Measured 2026-09-05: it passed six-step smoke runs that way and RAISED in the
# full campaign, where the id was not reused. Passing the object removes both the
# stale bind and the raise.


def curl_takes_the_half_integer_lattice(plan: Any, residency: Residency,
                                        pml: Any) -> Any:
    """HOST DEFECT: the shared kms and the curl's sinv from the HALF-INTEGER set.

    ``step_D`` reads integer positions (``SUB_STEPS['step_D']['suffix'] == ''``).
    The kernel cannot tell: a half-cell error in the absorber profile, not a crash.
    Because the kms group is SHARED, this also feeds the constitutive half the wrong
    lattice -- which is exactly why sharing is load-bearing.
    """
    rebind_static(plan, CURL_COEFFICIENT_STATIC_SLOTS,
                  [residency.mirror(f"mutation:{stem}_{axis}_h",
                                    getattr(pml, f"{stem}_{axis}_h"), constant=True)
                   for axis in "xyz" for stem in ("kms", "sinv")])
    return plan


def constitutive_takes_the_half_integer_lattice(plan: Any, residency: Residency,
                                                pml: Any) -> Any:
    """HOST DEFECT: bind ``kps_*_h`` -- ``update_E``'s lattice -- to the H half."""
    rebind_static(plan, CONSTITUTIVE_COEFFICIENT_STATIC_SLOTS,
                  [residency.mirror(f"mutation:kps_{axis}_h",
                                    getattr(pml, f"kps_{axis}_h"), constant=True)
                   for axis in "xyz"])
    return plan


HOST_MUTATIONS: Dict[str, Tuple[Callable[[Any, Residency, Any], Any], str, str]] = {
    "f_w_H_written_in_place": (
        lambda plan, residency, pml: AliasedLaunch(plan,
                                                   ("f_w_Hx", "f_w_Hy", "f_w_Hz")),
        "all",
        "the split-field history stored in place: a foreign recompute at a neighbour "
        "whose thread already stored reads B where it needs B_prev"),
    "rotation_skipped": (
        lambda plan, residency, pml: RotationSkipped(plan), "all",
        "the launcher's post-launch rotation dropped; all volumes at step 1, the "
        "in-place displacement at step 2"),
    "curl_takes_the_half_integer_lattice": (
        curl_takes_the_half_integer_lattice, "all",
        "the shared kms and sinv bound from the half-integer set"),
    "constitutive_takes_the_half_integer_lattice": (
        constitutive_takes_the_half_integer_lattice, "all",
        "kps bound from update_E's lattice"),
}


# ---------------------------------------------------------------------------
# Host legs
# ---------------------------------------------------------------------------

def leg_driver_order() -> Dict[str, Any]:
    """REPLACES is the driver's two ADJACENT consults and nothing else sits between.

    Read off ``driver.py`` by the seam module's own locator, which RAISES unless the
    only statement between the two consults is the electric withdraw loop. The sync
    channel's facts are read off ``fastpath`` the same way: the by-name consult maps
    to ``update_H`` and its containment set excludes ``step_D``, which is what makes
    the shim's default refusal a derived rule and not this file's opinion.
    """
    fact = h_to_d_seam.driver_seam_fact()
    span_ok = tuple(family.REPLACES) == tuple(h_to_d_seam.HALVES) == \
        tuple(withdraw_hoist.SEAM_SPAN)
    sync_owner = SYNC_PASS_OWNERS.get(SYNC_UPDATE_H_PASS)
    outside = tuple(name for name in family.REPLACES if name not in SYNC_PATH_SLOTS)
    return {
        "passed": bool(span_ok and sync_owner == "update_H" and outside == ("step_D",)
                       and family.SEAM == withdraw_hoist.SEAM
                       and metal_launch.FUSED_PAIR_SEAMS.get("update_H")
                       == ("step_D", withdraw_hoist.SEAM)),
        "driver_seam": fact,
        "replaces": list(family.REPLACES),
        "seam_halves": list(h_to_d_seam.HALVES),
        "withdraw_span": list(withdraw_hoist.SEAM_SPAN),
        "sync_consult": SYNC_UPDATE_H_PASS,
        "sync_owner_slot": sync_owner,
        "sync_path_slots": list(SYNC_PATH_SLOTS),
        "slots_of_this_span_outside_the_sync_path": list(outside),
        "seam_row": list(metal_launch.FUSED_PAIR_SEAMS.get("update_H") or ()),
    }


def _tail_after_decode(source: str) -> str:
    tail = source.split(family.DECODE_END, 1)[1]
    return tail[: -len("}\n")] if tail.endswith("}\n") else tail


def leg_transcription(codes: Sequence[int]) -> Dict[str, Any]:
    """Both halves are the CERTIFIED emitters' own bytes, differing exactly where declared."""
    source = family.fused_hd_pair_source(codes)
    certified_curl = _tail_after_decode(
        shaders.curl_source(codes, bool(metal_launch.SUB_STEPS["step_D"]["backward"]),
                            shaders.CONTRACT_OFF)).splitlines()
    welded_curl = family.welded_curl_tail(codes).splitlines()
    removed = [line for line in certified_curl if line not in welded_curl]
    added = [line for line in welded_curl if line not in certified_curl]
    expected_moves = len(family.OWN_LOAD_EDITS) + len(family.HALO_TAPS)

    certified_constitutive = _tail_after_decode(
        shaders.constitutive_source("H", shaders.CONTRACT_OFF)).splitlines()
    lifted = family.certified_constitutive_tail().splitlines()
    constitutive_removed = [line for line in certified_constitutive if line not in lifted]
    constitutive_added = [line for line in lifted if line not in certified_constitutive]
    accumulations_verbatim = all(
        f"        a{t} = a{t} + kp_{t} * src{t};\n        a{t} = a{t} - km_{t} * prev{t};"
        in source for t in range(3))
    # EVERY NEEDLE IN THE FILE, RESOLVED. Each builder raises unless its needle
    # matched exactly once, so reaching the end of each call IS the assertion; the
    # names are recorded so a needle that stopped being armed is visible in the
    # record rather than only in a traceback.
    anchors: Dict[str, bool] = {}
    for name in shader_mutations(codes):
        anchors[name] = True
    for name in inert_mutations(codes):
        anchors[name] = True
    for name in periodic_wrap_mutations(PERIODIC_CODES):
        anchors[name] = True
    try:
        byte_neutral_source(codes)
        anchors["byte_neutral_control"] = True
    except AssertionError:
        anchors["byte_neutral_control"] = False
    # THE CONSTITUTIVE LIFT, COUNTED BOTH WAYS, AGAINST THE DECLARED TABLE. Every
    # entry of `CONSTITUTIVE_LIFT_EDITS` is a per-COMPONENT edit and there are three
    # components, so the certified body must lose exactly three lines per declared
    # edit and gain three per declared edit that REWRITES rather than deletes.
    #
    # ONE ENTRY IS EXCLUDED BY NAME, WITH ITS REASON: the decode edit
    # (``int k = ii % nzi; ...``) describes text ABOVE the anchor both sides are cut
    # at, so it can appear in neither list. Excluding it by shape rather than by
    # position keeps the arithmetic tied to the table.
    visible = [edit for edit in family.CONSTITUTIVE_LIFT_EDITS
               if not edit["line"].startswith("    int k")]
    dropped = [edit for edit in visible if edit["became"].startswith("(removed")]
    rewrites = [edit for edit in visible if edit not in dropped]
    return {
        "passed": bool(
            len(removed) == len(added) == expected_moves
            and family.welded_curl_tail(codes) in source
            and family.h_cell_function() in source
            and accumulations_verbatim
            and len(constitutive_removed) == 3 * len(visible)
            and len(constitutive_added) == 3 * len(rewrites)
            and not any(f"{stem}{t}[" in family.certified_constitutive_tail()
                        for stem in ("f", "w", "g") for t in range(3))
            and all(anchors.values())
            and family.shipped_signature_bindings() == family.PACKED_BINDINGS
            == MAX_BUFFER_BINDINGS
            and source.isascii()
            and source.count(shaders.contraction_pragma(shaders.CONTRACT_OFF)) == 1),
        "constitutive_edits_visible_below_the_decode_anchor": len(visible),
        "constitutive_edits_that_delete_a_store": len(dropped),
        "constitutive_edits_that_rewrite_a_line": len(rewrites),
        "constitutive_lines_removed_expected": 3 * len(visible),
        "constitutive_lines_added_expected": 3 * len(rewrites),
        "curl_lines_removed": removed,
        "curl_lines_added": added,
        "curl_moves_expected": expected_moves,
        "constitutive_lines_removed": constitutive_removed,
        "constitutive_lines_added": constitutive_added,
        "constitutive_lift_edits_declared": len(family.CONSTITUTIVE_LIFT_EDITS),
        "accumulations_verbatim": accumulations_verbatim,
        "welded_curl_tail_in_source": family.welded_curl_tail(codes) in source,
        "h_cell_in_source": family.h_cell_function() in source,
        "mutation_anchors_resolve_once": anchors,
        "signature_bindings_counted": family.shipped_signature_bindings(),
        "packed_bindings_declared": family.PACKED_BINDINGS,
        "ceiling": MAX_BUFFER_BINDINGS,
        "ascii": source.isascii(),
        "contraction_pragmas": source.count(shaders.contraction_pragma(shaders.CONTRACT_OFF)),
    }


def _volume_source(fields: Any, component: str, integrated: bool) -> Any:
    from meep_gpu.sources import GaussianEnvelope, VolumeSource  # noqa: PLC0415

    return VolumeSource(grid=fields.grid, component=component,
                        center=(0.15, -0.1, 0.05), size=(0.0, 0.0, 0.0),
                        envelope=GaussianEnvelope(frequency=1.0, fwidth=0.5,
                                                  is_integrated=integrated))


@contextlib.contextmanager
def policy(value: str):
    previous = os.environ.get("MEEP_GPU_SUBNORMAL_POLICY")
    os.environ["MEEP_GPU_SUBNORMAL_POLICY"] = value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("MEEP_GPU_SUBNORMAL_POLICY", None)
        else:
            os.environ["MEEP_GPU_SUBNORMAL_POLICY"] = previous


def leg_refusal() -> Dict[str, Any]:
    """Each refusal by NAME, and the two admissions that are not this seam's business."""
    fields, pml = matrix.cart(boundaries={"x": "metallic"})
    residency = Residency()
    cover = family.metal_fused_hd_pair_coverage

    admitted = cover(fields, pml, (), residency)
    undeclared = cover(fields, pml, None, residency)
    integrated = cover(fields, pml, (_volume_source(fields, "Ez", True),), residency)
    plain_electric = cover(fields, pml, (_volume_source(fields, "Ez", False),), residency)
    magnetic_source = cover(fields, pml, (_volume_source(fields, "Hy", True),), residency)
    folded_fields, folded_pml = matrix.folded(boundaries={"y": "metallic"}, depth=1.2)
    folded = cover(folded_fields, folded_pml, (), Residency())
    cyl_fields, cyl_pml = matrix.cylindrical(m=0, complex_storage=False)
    cylindrical = cover(cyl_fields, cyl_pml, (), Residency())
    inactive_fields, inactive_pml = matrix.cart(pml=0)
    inactive = cover(inactive_fields, inactive_pml, (), Residency())
    with policy("keep"):
        keep = cover(fields, pml, (), residency)
        keep_report = subnormal.mps_policy_report()

    def named(verdict: Any, *needles: str) -> List[str]:
        return [reason for reason in verdict.reasons
                if all(text in reason for text in needles)]

    checks = {
        "admitted_on_the_cell": admitted.covered,
        "undeclared_refused_by_name": bool(not undeclared.covered
                                           and named(undeclared, "was not declared")),
        "integrated_electric_withdraw_refused_by_name": bool(
            not integrated.covered
            and named(integrated, "standing integrated", "HOISTS_THE_WITHDRAW = False")),
        "plain_electric_source_not_refused": plain_electric.covered,
        "magnetic_source_not_refused": magnetic_source.covered,
        "folded_refused_naming_the_second_product": bool(
            not folded.covered and named(folded, "folded", "own product")),
        "cylindrical_refused": bool(not cylindrical.covered
                                    and named(cylindrical, "cylindrical")),
        "inactive_absorber_refused_naming_the_constitutive_half": bool(
            not inactive.covered and named(inactive, "constitutive half:")),
        "keep_policy_refused_by_name": bool(not keep.covered
                                            and named(keep, "subnormal policy is 'keep'")
                                            and not keep_report["admitted"]),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "reasons": {
            "undeclared": list(undeclared.reasons),
            "integrated_electric": list(integrated.reasons),
            "plain_electric": list(plain_electric.reasons),
            "magnetic": list(magnetic_source.reasons),
            "folded": list(folded.reasons),
            "cylindrical": list(cylindrical.reasons),
            "inactive_absorber": list(inactive.reasons),
            "keep_policy": list(keep.reasons),
        },
        "keep_policy_report": keep_report,
    }


@contextlib.contextmanager
def _installable(value: bool):
    """Hold ``family.INSTALLABLE`` at ``value`` for the block, then put it back.

    The second brake cannot be measured any other way: the flag refuses first, so a
    run with it standing never reaches ``_pair_may_absorb`` and a leg that only
    watched the shipped configuration would report ONE brake as if it were two.
    """
    previous = family.INSTALLABLE
    family.INSTALLABLE = value
    try:
        yield
    finally:
        family.INSTALLABLE = previous


def _composer_outcome(keywords: Mapping[str, Any]) -> Dict[str, Any]:
    """``plan_step(fuse=True)`` on one fixture: who holds what, and what was refused."""
    fields, pml = matrix.cart(**dict(keywords))
    plan = metal_launch.plan_step(fields, pml, residency=Residency(), sources=(),
                                  fuse=True)
    owners: Dict[str, Any] = {}
    for slot in ("step_B", "update_H", "step_D", "update_E"):
        occupant = plan.plans.get(slot)
        owners[slot] = None if occupant is None else declaring(occupant)
    distinct: List[Any] = []
    for owner in owners.values():
        if owner is not None and not any(owner is seen for seen in distinct):
            distinct.append(owner)
    return {
        "selected": dict(plan.selected),
        "owners": owners,
        "launches": sum(int(getattr(owner, "launches_per_run", 1))
                        for owner in distinct),
        "refusal": list(plan.reasons.get(f"fused_pair_{family.FAMILY}", ())),
    }


def leg_arbitration(keywords: Mapping[str, Any]) -> Dict[str, Any]:
    """The composer, asked -- outcomes, not mechanism, and BOTH brakes driven.

    What this leg pins is what the rule must PRODUCE and not how: on a row both
    incumbents reach, the released B->H pair holds ``step_B`` and ``update_H``, this
    product is refused BY NAME, its label appears in no slot, and launches over
    ``step_B..update_E`` are two.

    THE ABSORB ROW IS REQUIRED PRESENT, NOT ABSENT, AND THAT IS THE STRONGER CLAIM.
    Until 2026-09-05 this product had no ``FUSED_PAIR_ARMS`` row, so the seam loop
    refused it before ever asking anything about the run -- a refusal that says
    nothing about arbitration. The row landed, so the loop now ASKS this product and
    the refusal has to survive being asked. A leg that still required the row absent
    would be measuring the tree of the day before and would pass on a product the
    composer never considered.

    AND THE SECOND BRAKE IS DRIVEN RATHER THAN CITED. Shipped, the reason recorded is
    the product's own ``INSTALLABLE = False`` -- a fact about the product, reported on
    every configuration, which is the FIRST brake and the only one a shipped run can
    reach. With the flag held out of the way the composer must STILL refuse, this
    time naming the arm that already holds ``update_H`` (``_pair_may_absorb`` reading
    the live ``selected`` after the released B->H pair installed first), and the
    installed composition must not move by one slot. Two independent brakes, measured
    as two.
    """
    shipped = _composer_outcome(keywords)
    magnetic_owner = shipped["owners"]["step_B"]
    magnetic_holds = (magnetic_owner is not None
                      and magnetic_owner is shipped["owners"]["update_H"]
                      and tuple(getattr(magnetic_owner, "replaces_sub_steps", ()))
                      == tuple(magnetic.REPLACES))
    label_absent = (family.FAMILY not in " ".join(shipped["selected"].values())
                    and not any("H/D" in value
                                for value in shipped["selected"].values()))
    flag_named = bool(shipped["refusal"]) and shipped["refusal"][0].startswith(
        f"{family.FAMILY} declares INSTALLABLE = False: ")
    with _installable(True):
        unflagged = _composer_outcome(keywords)
    slot_named = bool(unflagged["refusal"]) and unflagged["refusal"][0].startswith(
        f"update_H was selected by the "
        f"{shipped['selected'].get('update_H')!r} arm")
    composition_unmoved = unflagged["selected"] == shipped["selected"]
    absorb_row = metal_launch.FUSED_PAIR_ARMS.get(family.FAMILY)
    return {
        "passed": bool(magnetic_holds and flag_named and label_absent
                       and shipped["launches"] == 2 and not family.INSTALLABLE
                       and absorb_row == CELL_ARMS
                       and slot_named and composition_unmoved
                       and unflagged["launches"] == 2),
        "selected": shipped["selected"],
        "released_magnetic_pair_holds_step_B_and_update_H": magnetic_holds,
        "launches_over_step_B_to_update_E": shipped["launches"],
        "this_product_refused_by_name": shipped["refusal"],
        "first_brake_is_the_products_own_flag": flag_named,
        "this_product_label_absent_from_selected": label_absent,
        "installable_declared": family.INSTALLABLE,
        "absorb_row": list(absorb_row) if absorb_row else None,
        "absorb_row_is_the_cells_two_arms": absorb_row == CELL_ARMS,
        "with_the_flag_held_out_of_the_way": {
            "selected": unflagged["selected"],
            "refused_by_name": unflagged["refusal"],
            "second_brake_names_the_released_slot": slot_named,
            "composition_unmoved": composition_unmoved,
            "launches_over_step_B_to_update_E": unflagged["launches"],
        },
        "seam_row": list(metal_launch.FUSED_PAIR_SEAMS.get("update_H") or ()),
    }


def purity_ledger(fields: Any, pml: Any) -> Dict[str, Any]:
    """Of the curl's valid foreign taps, how many land on a cell ``update_H`` MOVED.

    Every foreign tap the curl half makes reads a neighbour whose stepped value an
    in-place weld would have overwritten at a schedule-decided moment. The tap is
    RACED on only where the neighbour's new H differs from its old one, so this
    count is exactly the number of taps whose recomputed value differs from what
    the in-place arrangement might have read. Measured on the ARRAY PATH: a
    property of the physics, not of the kernel. Near zero would mean the race
    controls and the stale-H mutation have nothing to catch.
    """
    from meep_gpu.stepping import _boundary_kinds  # noqa: PLC0415

    codes = tuple(1 if kind == "metallic" else 0 for kind in _boundary_kinds(fields.grid, pml))
    before = {name: np.array(getattr(fields, name), copy=True)
              for name in ("Hx", "Hy", "Hz")}
    stepping.update_H(fields, pml)
    moved = {name: words(before[name]) != words(getattr(fields, name))
             for name in before}
    shape = tuple(fields.grid.shape)
    per_tap: Dict[str, Dict[str, int]] = {}
    total_valid = 0
    total_raced = 0
    for var, target in family.HALO_TAPS:
        axis = "xyz".index(var[-1])
        component = ("Hx", "Hy", "Hz")[target]
        flags = moved[component].reshape(shape)
        # The tap reads the BACKWARD neighbour along `axis`: periodic wraps, metallic
        # serves an exact zero at index 0 and is not a read at all.
        shifted = np.roll(flags, 1, axis=axis)
        valid = np.ones(shape, dtype=bool)
        if codes[axis]:
            index = [slice(None)] * 3
            index[axis] = 0
            valid[tuple(index)] = False
        raced = int(np.count_nonzero(shifted & valid))
        count = int(np.count_nonzero(valid))
        per_tap[var] = {"valid_taps": count, "taps_on_a_moved_cell": raced}
        total_valid += count
        total_raced += raced
    cells = int(np.prod(shape))
    return {
        "codes": list(codes),
        "cells": cells,
        "cells_moved_by_update_H": {name: int(np.count_nonzero(flag))
                                    for name, flag in moved.items()},
        "per_tap": per_tap,
        "valid_foreign_taps": total_valid,
        "foreign_taps_whose_recompute_differs_from_the_in_place_read": total_raced,
        "fraction": round(total_raced / total_valid, 6) if total_valid else 0.0,
    }


def leg_seed_scale() -> Dict[str, Any]:
    """The 2^80 seed scale changes no mantissa, and the fixture then clears the budget.

    THREE CLAIMS, ALL MEASURED ON THE ARRAY PATH, so none of them is about a device.

    1. AT THE COMPOSITION MATRIX'S OWN AMPLITUDE THE FIXTURE ENTERS THE BAND inside
       the budget. If it did not, the scale would be an unexplained decoration, so the
       leg requires the unscaled run to fail the precondition and records the step.
    2. AT 2^80 IT DOES NOT, for every step of the budget, and the peak magnitude is
       recorded so "further from the band" cannot quietly mean "near the other end of
       the range".
    3. THE TWO RUNS ARE THE SAME ARITHMETIC. Every step, over every stored volume, the
       scaled state must equal the unscaled state times 2^80 BIT FOR BIT -- compared as
       uint32 words -- for as long as the unscaled run stays out of the band. That is
       the claim the scale rests on, and it is a comparison rather than an argument
       about linearity. Past the unscaled run's first banded step the equality is not
       expected and is not asserted: there the unscaled word is the flushed one and the
       scaled word is the arithmetic the fixture would have had.
    """
    scale = np.float32(2.0) ** SEED_SCALE_BITS
    rows: Dict[str, Any] = {}
    for name in (MUTATION_CASE, PERIODIC_MUTATION_CASE):
        keywords = dict(CASES)[name]
        plain = build_driver(keywords, 90100, scale_bits=0)
        scaled = build_driver(keywords, 90100)
        plain_band: Optional[int] = None
        scaled_band: Optional[int] = None
        mismatch: Optional[Dict[str, Any]] = None
        peak = 0.0
        for step in range(1, STEPS + 1):
            plain.step()
            scaled.step()
            a = stored_volumes(plain.fields)
            b = stored_volumes(scaled.fields)
            if plain_band is None and sum(subnormal.census(v) for v in a.values()):
                plain_band = step
            if scaled_band is None and sum(subnormal.census(v) for v in b.values()):
                scaled_band = step
            peak = max(peak, max(float(np.max(np.abs(v))) for v in b.values()))
            if plain_band is None and mismatch is None:
                bad = {key: differing(np.asarray(a[key]) * scale, b[key])
                       for key in sorted(a)}
                bad = {key: count for key, count in bad.items() if count}
                if bad:
                    mismatch = {"step": step, "volumes": bad}
        rows[name] = {
            "unscaled_first_banded_step": plain_band,
            "scaled_first_banded_step": scaled_band,
            "scaled_peak_magnitude": peak,
            "float32_max": float(np.finfo(np.float32).max),
            "words_per_step": int(sum(words(v).size
                                      for v in stored_volumes(scaled.fields).values())),
            "steps_the_equality_was_checked_over": (plain_band - 1 if plain_band
                                                    else STEPS),
            "first_word_that_is_not_the_scaled_twin": mismatch,
        }
        log(f"    seed_scale {name} unscaled_banded_at={plain_band} "
            f"scaled_banded_at={scaled_band} peak={peak:.3e} mismatch={mismatch}")
    # NECESSITY IS A CLAIM ABOUT THE FIXTURE SET, NOT ABOUT EVERY CASE IN IT. Only a
    # run that decays into the band needs the scale, and the two cases here decay at
    # different rates -- the walled one bands inside the budget and the all-periodic
    # one does not. Requiring it of every case would fail for the case that needed the
    # scale least; requiring it of NONE would let the scale become decoration. So the
    # floor is that at least one case witnesses the need, and the equality and the
    # cleanliness are required of all of them.
    needed = sorted(name for name, row in rows.items()
                    if row["unscaled_first_banded_step"] is not None)
    return {
        "passed": bool(rows and needed
                       and all(row["scaled_first_banded_step"] is None
                               and row["first_word_that_is_not_the_scaled_twin"] is None
                               and row["scaled_peak_magnitude"] < row["float32_max"]
                               and row["steps_the_equality_was_checked_over"] >= 2
                               for row in rows.values())),
        "seed_scale_bits": SEED_SCALE_BITS,
        "steps": STEPS,
        "cases_where_the_unscaled_fixture_enters_the_band": needed,
        "cases": rows,
        "what_this_licenses": (
            "that this gate's synthetic fixture is the composition matrix's own state "
            "with every exponent shifted by a constant, and that the shifted fixture "
            "stays out of the float32 denormal band for the whole 60-step budget. It "
            "licenses nothing about a device and nothing about any other amplitude"),
    }


def leg_purity_ledger() -> Dict[str, Any]:
    rows = {}
    for name, keywords in CASES:
        fields, pml = matrix.cart(**dict(keywords))
        rows[name] = purity_ledger(fields, pml)
    fractions = [row["fraction"] for row in rows.values()]
    floor = min(fractions) if fractions else 0.0
    return {
        "passed": bool(rows and floor >= 0.99),
        "floor_fraction": floor,
        "what_this_licenses": (
            f"on every fixture at least {floor:.4f} of the curl's valid foreign taps "
            f"land on a cell update_H moved, so the race null control, the planted "
            f"race and the stale-H mutation each have that many taps to catch; a "
            f"near-zero number here would have made a green identity worthless"),
        "cases": rows,
    }


#: The poison a private buffer is filled with before the step that must ignore it. A
#: NEGATIVE, physically absurd word, so a buffer that survived into an answer would
#: move that answer wholesale rather than by a rounding.
SCRATCH_POISON = np.float32(-1.0e30)


def leg_private_scratch() -> Dict[str, Any]:
    """The premise under :data:`PRIVATE_SCRATCH`, MEASURED on a dispersive driver.

    ``stored_volumes`` drops every leading-underscore name on the rule that
    ``Fields``'s private scratch is not state. That is a claim about the engine, and
    a claim is what an exclusion is worth, so it is driven rather than asserted:

    * a driver carrying a live susceptibility is stepped once with the buffers as the
      engine left them, and once with every one of them filled with
      :data:`SCRATCH_POISON` beforehand;
    * every STORED volume must come back byte-identical between the two, which is the
      exclusion's premise -- nothing reads the buffer before writing it;
    * and the poison must be GONE afterwards, which is the non-vacuity witness. A
      fixture where the buffer is never touched would satisfy the first requirement
      by never exercising anything, and this leg would then be measuring nothing.

    The susceptibility is what makes the buffer live at all:
    ``displacement_minus_polarization`` returns ``D`` itself and allocates nothing
    when no polarization drives the component (fields.py:1096-1098).
    """
    from meep_gpu.dispersion import PolarizationState, Susceptibility  # noqa: PLC0415

    def dispersive_driver(seed: int) -> Any:
        driver = build_driver(dict(CASES)[MUTATION_CASE], seed)
        sigma = {name: (0.25 if name == "Ez" else 0.0) for name in ("Ex", "Ey", "Ez")}
        driver.fields.polarizations.append(
            PolarizationState(Susceptibility(1.0, 0.1, "lorentzian"), sigma,
                              driver.fields.grid, np.float32))
        rng = np.random.default_rng(seed + 1)
        scale = np.float32(2.0) ** SEED_SCALE_BITS
        for name, array in stored_volumes(driver.fields).items():
            if name.startswith(("P[", "P_prev[")):
                array[...] = (0.37 * rng.standard_normal(array.shape)).astype(array.dtype)
                array *= scale
        driver.invalidate_fast_path()
        pin_array_path(driver)
        return driver

    def private_buffers(fields: Any) -> Dict[str, Any]:
        found: Dict[str, Any] = {}
        buffer = getattr(fields, "_fmp_scratch", None)
        if buffer is not None:
            found["_fmp_scratch"] = buffer
        for component, array in (getattr(fields, "_fmp_scratch_by_component", None)
                                 or {}).items():
            if array is not None:
                found[f"_fmp_scratch_by_component[{component}]"] = array
        return found

    clean = dispersive_driver(98100)
    clean.step()
    allocated = sorted(private_buffers(clean.fields))
    after_clean = capture(clean)

    poisoned = dispersive_driver(98100)
    poisoned.step()          # the first step is what ALLOCATES the buffers
    poisoned_names = sorted(private_buffers(poisoned.fields))
    for array in private_buffers(poisoned.fields).values():
        array.fill(SCRATCH_POISON)
    # ...and the second step is the one that must ignore what they hold.
    clean.step()
    poisoned.step()
    survived = {name: int(np.count_nonzero(np.asarray(array) == SCRATCH_POISON))
                for name, array in private_buffers(poisoned.fields).items()}
    difference = compare_snapshots(capture(clean), capture(poisoned))

    # THE NULL CONTROL: THE SAME POISON, IN A VOLUME THAT IS STATE. Without it a green
    # result above is equally consistent with a poison this comparison cannot see at
    # all -- a wrong dtype, a buffer the fill missed, a snapshot taken too early. The
    # control puts the identical word into ``Dz`` at the identical moment and requires
    # the step to come out different.
    control = dispersive_driver(98100)
    control.step()
    control.fields.Dz.fill(SCRATCH_POISON)
    control.step()
    control_reference = dispersive_driver(98100)
    control_reference.step()
    control_reference.step()
    control_difference = compare_snapshots(capture(control_reference), capture(control))
    return {
        "passed": bool(allocated and allocated == poisoned_names and not difference
                       and not any(survived.values())
                       and control_difference
                       and "_fmp_scratch" not in after_clean["volumes"]),
        "null_control_poisons_a_stored_volume_instead": {
            "volume": "Dz",
            "differing_stored_words": int(sum(control_difference.values())),
            "differing_volumes": control_difference,
            "fired": bool(control_difference)},
        "private_buffers_the_fixture_allocated": allocated,
        "excluded_by_the_leading_underscore_rule": list(PRIVATE_SCRATCH),
        "stored_volumes_after_a_step_carry_no_private_name": sorted(
            name for name in after_clean["volumes"] if name.startswith("_")),
        "differing_stored_words_with_the_buffers_poisoned": difference,
        "poison_words_still_standing_after_the_step": survived,
        "poison": float(SCRATCH_POISON),
        "what_this_licenses": (
            "dropping Fields' private scratch from the comparison: on a driver whose "
            "susceptibility makes those buffers live, a complete step writes every one "
            "of them in full before reading it, so poisoning them changes no stored "
            "word and the poison does not survive. It licenses nothing about a volume "
            "whose name does not start with an underscore"),
    }


# ---------------------------------------------------------------------------
# Compile legs
# ---------------------------------------------------------------------------

def _compiles(source: str) -> Tuple[bool, str]:
    try:
        compile_source(source)
        return True, ""
    except Exception as exc:  # noqa: BLE001 - the failure IS the measurement
        message = str(exc)
        return False, message.splitlines()[0] if message else type(exc).__name__


def leg_rotation_guard(driver: Any) -> Dict[str, Any]:
    """:func:`orphaned_mirrors` must FIRE on an orphan and stay silent on a rename.

    The guard that stands between this gate and a comparison of buffers nobody reads
    is only worth what it catches, and the 2026-09-06 repair narrowed it from "a name
    changed" to "a MIRRORED name changed" -- so both halves are driven here rather
    than argued:

    * ARMED: after a step, ``fields.Dz`` is pointed at a twin of the array every
      arrangement's residency mirrors. That is the 2026-09-05 ``cavity_arrayslice``
      shape exactly, and the guard must name ``Dz``.
    * INERT: the engine's own polarization rotation is performed and the guard must
      stay silent, because no residency mirrors those buffers -- which is itself
      asserted here, so a future residency that DOES mirror them turns this row red
      instead of turning the guard vacuous.
    """
    arrangements = all_arrangements(driver)
    fields = driver.fields
    before = orphaned_mirrors(arrangements, fields)
    mirrored = sorted({name for arrangement in arrangements.values()
                       if arrangement.residency is not None
                       for name in arrangement.residency.names})
    polarization_mirrors = [name for name in mirrored
                            if name.startswith(("P[", "P_prev[", "P.", "P_prev."))]

    original = fields.Dz
    twin = np.array(original, copy=True)
    fields.Dz = twin
    armed = orphaned_mirrors(arrangements, fields)
    fields.Dz = original
    after = orphaned_mirrors(arrangements, fields)

    renamed: List[str] = []
    for state in getattr(fields, "polarizations", ()) or ():
        renamed.extend(sorted(state.P))
    return {
        "passed": bool(not before and armed == ["Dz"] and not after
                       and "Dz" in mirrored and not polarization_mirrors),
        "mirror_names_across_every_arrangement": len(mirrored),
        "armed_control_named": armed,
        "silent_before_the_rebind": before,
        "silent_after_the_rebind_is_undone": after,
        "Dz_is_mirrored": "Dz" in mirrored,
        "polarization_volumes_any_residency_mirrors": polarization_mirrors,
        "polarization_components_this_fixture_drives": renamed,
        "what_this_licenses": (
            "that the rotation guard names a mirrored volume the engine has stopped "
            "naming, and that it does not name the engine's own polarization buffers "
            "-- which no residency here mirrors. It licenses nothing about a rotation "
            "INSIDE a launch, which the rotation_skipped host mutation measures"),
    }


def leg_binding_ceiling() -> Dict[str, Any]:
    """35, 34 and 32 bindings must FAIL; the shipped 31 must COMPILE everywhere."""
    refuted = {}
    for label, builder, count in (
            ("separate_scalars", family.refuted_separate_scalar_source,
             family.SEPARATE_SCALAR_BINDINGS),
            ("unshared_kms", family.refuted_unshared_kms_source,
             family.UNSHARED_KMS_BINDINGS),
            ("one_more_pointer", family.refuted_one_more_pointer_source,
             family.ONE_MORE_POINTER_BINDINGS)):
        compiled, error = _compiles(builder())
        refuted[label] = {"bindings": count, "compiled": compiled, "error": error,
                          "refused_for_the_right_reason":
                              (not compiled and "out of bounds" in error
                               and "buffer" in error)}
    shipped = {}
    for codes in ((x, y, z) for x in (0, 1) for y in (0, 1) for z in (0, 1)):
        for mode in shaders.CONTRACT_MODES:
            compiled, error = _compiles(family.fused_hd_pair_source(codes, mode))
            shipped[f"{codes}:{mode}"] = {"compiled": compiled, "error": error}
    return {
        "passed": bool(all(row["refused_for_the_right_reason"] for row in refuted.values())
                       and all(row["compiled"] for row in shipped.values())
                       and family.PACKED_BINDINGS == MAX_BUFFER_BINDINGS),
        "ceiling": MAX_BUFFER_BINDINGS,
        "packed_bindings": family.PACKED_BINDINGS,
        "refuted": refuted,
        "shipped_specialisations_compiled": sum(1 for r in shipped.values() if r["compiled"]),
        "shipped_specialisations": len(shipped),
        "shipped": shipped,
    }


def leg_mutants_compile(codes: Sequence[int]) -> Dict[str, Any]:
    """Every armed defect COMPILES, so a mutant reported caught was a launched kernel.

    All three sets, on the specialisation each is armed against: a mutant that failed
    to compile would be installed as nothing, the slot would fall to the shipped
    kernel and the leg below would report the defect as UNCAUGHT for a reason that
    has nothing to do with the comparison.
    """
    rows = {}
    for name, (source, why) in shader_mutations(codes).items():
        compiled, error = _compiles(source)
        rows[name] = {"compiled": compiled, "error": error, "why_it_must_fire": why,
                      "case": MUTATION_CASE, "scored_on": "diverging"}
    for name, (source, reason, control) in inert_mutations(codes).items():
        compiled, error = _compiles(source)
        rows[name] = {"compiled": compiled, "error": error, "why_it_must_fire": reason,
                      "case": MUTATION_CASE, "scored_on": "agreeing",
                      "control": control}
    for name, (source, why) in periodic_wrap_mutations(PERIODIC_CODES).items():
        compiled, error = _compiles(source)
        rows[name] = {"compiled": compiled, "error": error, "why_it_must_fire": why,
                      "case": PERIODIC_MUTATION_CASE, "scored_on": "diverging"}
    compiled, error = _compiles(byte_neutral_source(codes))
    rows["byte_neutral_control"] = {"compiled": compiled, "error": error,
                                    "case": MUTATION_CASE, "scored_on": "agreeing",
                                    "why_it_must_fire": "it must NOT: scored on agreeing"}
    return {"passed": all(row["compiled"] for row in rows.values()),
            "armed": len(rows) - 1, "mutants": rows,
            "predicted_null": [dict(row) for row in PREDICTED_NULL]}


# ---------------------------------------------------------------------------
# Device legs
# ---------------------------------------------------------------------------

def all_arrangements(driver: Any, functions: Optional[Mapping[str, Any]] = None,
                     launcher: Optional[Callable[[Any, Residency, Any], Any]] = None,
                     ) -> Dict[str, Arrangement]:
    """The array path, the three references and the subject TWICE, on one driver.

    ``weld`` is the subject in a fully dispatched composition: the seam is the fused
    launch and every other slot carries the unfused single ``plan_step(fuse=False)``
    selects. ``weld_seam_only`` is the same launch with every other slot LEFT ON THE
    ARRAY PATH, which is the ``singles`` composition with the two halves welded.

    THE SECOND ONE EXISTS BECAUSE THE FIRST SHARES ITS NON-SEAM PLANS WITH A
    REFERENCE. Measured 2026-09-05 on ``examples:cavity_arrayslice.py``: the
    ``unfused`` composition disagreed with the array path at step 1 by 16 words in
    ``D``/``fu_D`` while ``singles`` and ``composition_today`` did not -- and because
    the weld arrangement borrows exactly those plans, its row disagreed too and read
    as a weld defect. Isolating the seam is the only way that row can be attributed,
    and attribution is the difference between a finding and a red.
    """
    return {
        "array": Arrangement("array", None, None),
        "singles": arrangement_singles(driver),
        "composition_today": arrangement_composition(driver, fuse=True),
        "unfused": arrangement_composition(driver, fuse=False),
        "weld": arrangement_weld(driver, functions=functions, launcher=launcher),
        "weld_seam_only": arrangement_weld(driver, functions=functions,
                                           launcher=launcher, rest_unfused=False,
                                           name="weld_seam_only"),
    }


def run_product(driver: Any, steps: int, progress: Optional[Callable[[str], None]] = None,
                movement_floor: str = "all",
                functions: Optional[Mapping[str, Any]] = None,
                launcher: Optional[Callable[[Any, Residency, Any], Any]] = None,
                stop_on_divergence: bool = True,
                require_full_budget: bool = True,
                clean_floor: int = 0) -> Dict[str, Any]:
    """The four-reference identity on one driver, per complete step.

    ``require_full_budget`` is True for the SYNTHETIC fixture, whose amplitude this
    gate chooses (:data:`SEED_SCALE_BITS`) precisely so the flush precondition holds
    for the whole budget. It is False for a LIFTED CORPUS ROW, whose state is the
    row's own: a run driven from the engine's zeros has a wavefront whose leading
    cells are arbitrarily small, so it enters the float32 denormal band on its own
    schedule and no byte claim can be made past that step. There the budget is every
    step the precondition holds, up to ``steps``, with ``clean_floor`` as the floor
    below which the row measured too little to count.
    """
    arrangements = all_arrangements(driver, functions=functions, launcher=launcher)
    result = drive(driver, arrangements, steps, progress=progress,
                   secondary="singles", stop_on_divergence=stop_on_divergence)
    legs = result["arrangements"]
    identical = all(legs[name]["identical"] for name in legs)
    compared = int(result["steps_compared"])
    budget_ok = (compared == steps if require_full_budget
                 else compared >= max(clean_floor, 1))
    launched = all(legs[name]["launches"]["plans"] > 0
                   and legs[name]["launches"]["plans"] == legs[name]["launches"]["functions"]
                   for name in legs)
    weld_ok = all(
        legs[name]["dispatched"].get("update_H", 0) == result["steps_stepped"]
        and legs[name]["absorbed"].get("step_D", 0) == result["steps_stepped"]
        for name in ("weld", "weld_seam_only"))
    # THE FLOOR, AND WHICH ONE DEPENDS ON WHERE THE STATE CAME FROM.
    #
    # ``all`` is the SYNTHETIC fixture's floor: every stored volume is seeded with
    # physical-band noise, so every one of them must move or the comparison is a
    # no-op agreeing with a no-op.
    #
    # ``seam`` is the LIFTED CORPUS ROW's, and the difference is physics rather than
    # leniency. A lifted row starts at the engine's own zeros and is driven by its
    # own sources, so a component the row's polarization never excites is identically
    # zero for the whole run -- a 2-D TM row moves Ez, Hx and Hy and nothing else, and
    # demanding all six would fail every such row for being what it is. What must not
    # be allowed to pass is a row where the seam wrote NOTHING, so the floor is that
    # the volumes this weld's launch produces moved somewhere, and the per-volume
    # split is recorded so a reader can see which components the row carries.
    seam_outputs = tuple(name for name in
                         [f"{stem}{axis}" for stem in ("D", "fu_D", "H", "f_w_H")
                          for axis in "xyz"]
                         if name in result["volumes_compared"])
    seam_moved = {name: int(result["moved_from_seed"].get(name, 0))
                  for name in seam_outputs}
    if movement_floor == "all":
        floor = not result["arrays_that_never_moved"]
    else:
        floor = bool(seam_outputs) and sum(seam_moved.values()) > 0
    # A ROTATION LEFT STANDING VOIDS EVERY COMPARISON AFTER IT, IN BOTH DIRECTIONS.
    # The arrangements' mirrors are bound to the engine's arrays by identity and
    # ``capture`` reads the names, so an orphaned volume can manufacture a divergence
    # (measured) and can equally manufacture an agreement between two arrangements
    # that both wrote an array nobody read. Scored, never inferred from the byte
    # result.
    settled = bool(result["every_arrangement_gave_the_engine_its_volumes_back"])
    # THE PRECONDITION HAS TO HAVE BEEN TAKEN, NOT ONLY TO HAVE HELD. A step whose
    # arithmetic ran through a helper the intermediate census does not model was
    # compared without half of its precondition, and a byte result taken there is a
    # claim the gate cannot support -- so it fails rather than passing quietly.
    census_covered = result["the_intermediate_census_modelled_every_helper_the_step_ran"]
    result.update({
        "passed": bool(identical and launched and weld_ok and floor and budget_ok
                       and settled and result["step_error"] is None
                       and census_covered is not False
                       and (result["precondition_clean"] or not require_full_budget)),
        "bit_identical": identical,
        "budget_met": budget_ok,
        "require_full_budget": require_full_budget,
        "clean_floor": clean_floor,
        "every_arrangement_launched_and_the_two_counters_agree": launched,
        "weld_launched_once_per_step_and_absorbed_step_D": weld_ok,
        "movement_floor": movement_floor,
        "movement_floor_met": floor,
        "seam_output_volumes": list(seam_outputs),
        "seam_output_words_moved": seam_moved,
        "seam_output_volumes_that_moved": sorted(name for name, n in seam_moved.items()
                                                 if n),
        "seam_output_volumes_that_never_moved": sorted(
            name for name, n in seam_moved.items() if not n),
        "weld_agrees_with_the_certified_singles":
            legs["weld"]["identical_vs_secondary"],
        "the_seam_alone_agrees_with_the_array_path":
            legs["weld_seam_only"]["identical"],
        "the_seam_alone_agrees_with_the_certified_singles":
            legs["weld_seam_only"]["identical_vs_secondary"],
        "references_that_disagree_with_the_array_path": sorted(
            name for name in ("singles", "composition_today", "unfused")
            if not legs[name]["identical"]),
        "composition_today_selected": legs["composition_today"]["selected"],
        "composition_today_refuses_this_product": legs["composition_today"].get("selected")
        and arrangements["composition_today"].reasons.get(f"fused_pair_{family.FAMILY}"),
    })
    return result


def leg_launch_structure(driver: Any, steps: int) -> Dict[str, Any]:
    """Launches per step, at the seam and over the whole step, two counters each."""
    arrangements = all_arrangements(driver)
    result = drive(driver, arrangements, steps, stop_on_divergence=False)
    rows = {}
    for name, arrangement in arrangements.items():
        shim = arrangement.shim
        counts = arrangement.launches()
        seam = 0
        if shim is not None:
            for slot in family.REPLACES:
                plan = shim.plans.get(slot)
                if plan is not None and plan is not ABSORBED:
                    seam += int(getattr(declaring(plan), "launches_per_run", 1))
        rows[name] = {
            "launches_per_step_whole_step_plans":
                counts["plans"] / max(result["steps_stepped"], 1),
            "launches_per_step_whole_step_functions":
                counts["functions"] / max(result["steps_stepped"], 1),
            "launches_per_step_at_the_seam": seam,
            "dispatched": {} if shim is None else dict(shim.dispatched),
            "absorbed": {} if shim is None else dict(shim.absorbed),
            "selected": arrangement.selected,
        }
    weld = rows["weld"]
    singles = rows["singles"]
    today = rows["composition_today"]
    step_saving = (today["launches_per_step_whole_step_plans"]
                   - weld["launches_per_step_whole_step_plans"])
    return {
        "passed": bool(weld["launches_per_step_at_the_seam"] == 1
                       and singles["launches_per_step_at_the_seam"] == 2
                       and all(row["launches_per_step_whole_step_plans"]
                               == row["launches_per_step_whole_step_functions"]
                               for row in rows.values())
                       and result["step_error"] is None),
        "arrangements": rows,
        "seam_reduction_against_the_singles":
            singles["launches_per_step_at_the_seam"] - weld["launches_per_step_at_the_seam"],
        "whole_step_reduction_against_the_composition_installed_today": step_saving,
        "what_this_says": (
            f"the weld saves exactly one launch against the two singles at the seam and "
            f"{step_saving:+g} against the composition the composer installs today over "
            f"the whole step: a product spanning update_H/step_D takes one slot from the "
            f"released B->H pair, so at the step level it does not reduce launches. That "
            f"is the arbitration measurement, reproduced here rather than argued"),
    }


def leg_sync(driver: Any, steps: int) -> Dict[str, Any]:
    """FORCE-INSTALLED, with the flux accessor fired mid-run: the hazard, measured.

    ``synchronize_magnetic_fields`` backs up and restores the MAGNETIC names only,
    then consults ``update_H_synchronize``. A product spanning ``update_H``/``step_D``
    that ANSWERED that consult would advance ``D``/``fu_D`` inside a half-step nothing
    undoes, on every flux and energy call, and the run would carry it to the end.
    The armed arm does exactly that and MUST diverge in ``D``; the control declines
    by the containment rule read off ``fastpath`` and MUST NOT.
    """
    hazard = arrangement_weld(driver, sync_hazard=True)
    declining = arrangement_weld(driver, sync_hazard=False)
    arrangements = {"array": Arrangement("array", None, None),
                    "weld_declining": declining, "weld_hazard": hazard}

    def hook(step: int, name: str, engine: Any) -> Any:
        if step in SYNC_STEPS:
            return {"step": step, "flux_z": float(engine.flux_in_box(2))}
        return None

    result = drive(driver, arrangements, steps, per_step_hook=hook,
                   stop_on_divergence=False)
    legs = result["arrangements"]
    hazard_rows = legs["weld_hazard"]["per_step"]
    first = legs["weld_hazard"]["first_divergence"]
    diverged_in_D = bool(first is not None and any(
        in_place_words(row) for row in hazard_rows if row["differing_words"]))
    return {
        "passed": bool(legs["weld_declining"]["identical"]
                       and legs["weld_declining"]["sync_refusals"] == len(SYNC_STEPS)
                       and legs["weld_hazard"]["sync_answered"] == len(SYNC_STEPS)
                       and first is not None and first == min(SYNC_STEPS)
                       and diverged_in_D and result["step_error"] is None),
        "sync_steps": list(SYNC_STEPS),
        "declining_identical": legs["weld_declining"]["identical"],
        "declining_sync_refusals": legs["weld_declining"]["sync_refusals"],
        "hazard_sync_answered": legs["weld_hazard"]["sync_answered"],
        "hazard_first_divergence": first,
        "hazard_diverges_in_D_or_fu_D": diverged_in_D,
        "hazard_differing_volumes_at_first_divergence": next(
            (row["differing_volumes"] for row in hazard_rows if row["differing_words"]), {}),
        "flux_values": {name: legs[name]["hook"] for name in legs},
        "array_flux_values": result["reference_hook"],
        "citation": (
            "a product installed at update_H that also advances D corrupts D on every "
            "flux_in_box and field_energy_in_box call unless it declines the "
            "update_H_synchronize consult; the composer must refuse (or the product "
            "decline) by the containment rule, and this row is the measurement"),
        "steps_compared": result["steps_compared"],
        "step_error": result["step_error"],
    }


#: THE SOURCE IS SCALED WITH THE SEED, and leaving it unscaled DISARMS this leg.
#:
#: Measured 2026-09-05, and it is the sharpest interaction in this file. The seeded
#: state carries :data:`SEED_SCALE_BITS`; an O(1) injected dipole added to a 2^80 field
#: rounds away entirely, so dropping the withdraw or moving it after ``step_D`` changes
#: no word at all -- BOTH null controls of this leg came back not diverging, which is a
#: leg that proves nothing while reporting a green hoisted arrangement. Scaling the
#: amplitude by the same power of two restores the whole run as the exact 2^80 twin of
#: the unscaled one: 0 differing words over every step the unscaled run stays out of
#: the band, both controls diverging at step 2 again, and the budget clear of the band
#: (peak |x| = 3.2e26).
INTEGRATED_SOURCE: Dict[str, Any] = {
    "component": "Ez", "center": (0.15, -0.1, 0.05), "size": (0.0, 0.0, 0.0),
    "frequency": 1.0, "source_type": "gaussian", "fwidth": 0.5, "is_integrated": True,
    "amplitude": 2.0 ** SEED_SCALE_BITS,
}


def leg_withdraw(keywords: Mapping[str, Any], seed: int, steps: int) -> Dict[str, Any]:
    """An integrated electric source in the seam: hoisted, un-hoisted, after_step_D.

    THE PREDICATE REFUSES THIS CONFIGURATION TODAY, by name, because the product
    declares ``HOISTS_THE_WITHDRAW = False``; that refusal is recorded here. What this
    leg then measures is the WIRING the installer would give a product that declared
    True -- ``LeadingWithdrawPlan`` around the launch -- against the array path, with
    the two null controls the campaign measured on the array path reproduced on the
    device: the un-hoisted launch (the withdraw runs after it) and the array path with
    the withdraw moved after ``step_D``. Both must diverge at step 2; the hoisted
    arrangement must be identical. This licenses the wiring, not the flag.
    """
    driver = build_driver(keywords, seed, sources=(INTEGRATED_SOURCE,))
    sources = tuple(driver._sources)  # noqa: SLF001
    refusal = family.metal_fused_hd_pair_coverage(driver.fields, driver.pml, sources,
                                                  Residency())
    hoistable, hoist_reasons = withdraw_hoist.hoistable(driver.fields, sources)
    standing = withdraw_hoist.standing_withdraws(sources)

    class AfterStepD:
        """The array path with the withdraw moved to AFTER the curl, before injection."""

        __slots__ = ("fields", "declined", "dispatched", "absorbed", "sync_refusals",
                     "sync_answered", "plans")

        def __init__(self, fields: Any) -> None:
            self.fields = fields
            self.declined: Dict[str, int] = {}
            self.dispatched: Dict[str, int] = {}
            self.absorbed: Dict[str, int] = {}
            self.sync_refusals = 0
            self.sync_answered = 0
            self.plans: Dict[str, Any] = {}

        def dispatch(self, slot: str, fields: Any) -> bool:
            if slot != "step_D":
                self.declined[slot] = self.declined.get(slot, 0) + 1
                return False
            stepping.step_D(fields, driver.pml)
            for _index, source in standing:
                type(source).withdraw(source, fields)
            self.dispatched[slot] = self.dispatched.get(slot, 0) + 1
            return True

        @property
        def launches(self) -> int:
            return 0

    suppressed = {id(source): source for _index, source in standing}

    def suppress() -> None:
        for source in suppressed.values():
            source.withdraw = lambda fields: None  # the driver's in-seam call, made a no-op

    def release() -> None:
        for source in suppressed.values():
            if "withdraw" in vars(source):
                del source.withdraw

    after = Arrangement("after_step_D", AfterStepD(driver.fields), None)  # type: ignore[arg-type]
    hoisted = arrangement_weld(driver, hoist=True)
    not_hoisted = arrangement_weld(driver, hoist=False)
    arrangements = {"array": Arrangement("array", None, None), "after_step_D": after,
                    "weld_hoisted": hoisted, "weld_not_hoisted": not_hoisted}

    original_step = driver.step

    def stepped() -> None:
        # The suppression applies to the after_step_D arrangement only.
        if driver._fast_path is after.shim:  # noqa: SLF001
            suppress()
            try:
                original_step()
            finally:
                release()
        else:
            original_step()

    driver.step = stepped  # type: ignore[method-assign]
    try:
        result = drive(driver, arrangements, steps, stop_on_divergence=False)
    finally:
        driver.step = original_step  # type: ignore[method-assign]
    legs = result["arrangements"]
    leading = hoisted.shim.plans["update_H"]
    return {
        "passed": bool(not refusal.covered
                       and any("standing integrated" in r for r in refusal.reasons)
                       and hoistable and standing
                       and legs["weld_hoisted"]["identical"]
                       and legs["weld_not_hoisted"]["first_divergence"] == 2
                       and legs["after_step_D"]["first_divergence"] == 2
                       and getattr(leading, "withdrawn", 0) >= 1
                       and result["step_error"] is None),
        "predicate_refuses_this_row_by_name": list(refusal.reasons),
        "hoistable": hoistable,
        "hoistable_reasons": list(hoist_reasons),
        "standing_withdraws": len(standing),
        "hoisted_withdraws_on_last_step": getattr(leading, "withdrawn", None),
        "hoisted_identical": legs["weld_hoisted"]["identical"],
        "not_hoisted_first_divergence": legs["weld_not_hoisted"]["first_divergence"],
        "after_step_D_first_divergence": legs["after_step_D"]["first_divergence"],
        "campaign": dict(withdraw_hoist.MEASUREMENT),
        "what_this_licenses": (
            "the LeadingWithdrawPlan wiring at the BEFORE placement on this device, "
            "against the array path, with both null controls diverging where the array-"
            "path campaign measured them. It does not flip HOISTS_THE_WITHDRAW: the "
            "product still refuses this row by name, and the flag moves only with a "
            "FUSED_PAIR_ARMS row and INSTALLABLE"),
        "steps_compared": result["steps_compared"],
        "step_error": result["step_error"],
        "per_step": {name: legs[name]["per_step"] for name in legs},
    }


# ---------------------------------------------------------------------------
# The corpus lift -- the census driver's battery hook, and the parent that spawns it
# ---------------------------------------------------------------------------

class _HostGrid:
    xp = np


def runtime_reasons() -> List[str]:
    """Why THIS process cannot evaluate this battery at all, or ``[]``."""
    from meep_gpu.metal_kernels import coverage  # noqa: PLC0415
    return list(coverage._metal_backend_reasons(_HostGrid()))  # noqa: SLF001


def _child_progress(message: str) -> None:
    path = os.environ.get("MEEP_GPU_HD_GATE_PROGRESS")
    label = os.environ.get("MEEP_GPU_HD_GATE_LABEL", "?")
    line = f"[{time.strftime('%H:%M:%S')}] {label} {message}"
    print(line, flush=True)
    if path:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()


def waveform_determinism(driver: Any, steps: int) -> Dict[str, Any]:
    """Is every source's waveform a FUNCTION OF TIME? Measured, never assumed.

    THE COMPARISON THIS GATE MAKES PRESUMES IT. ``drive`` steps five arrangements
    from ONE captured seed, one after another, and holds them to the same bytes.
    That is a claim about the arrangements only where the row's own inputs are the
    same for each of them -- and a source whose waveform is not a function of ``t``
    hands arrangement two a different current from arrangement one, so the five walks
    are five different problems and byte identity across them is undefined BY
    CONSTRUCTION rather than false.

    ``examples:stochastic_emitter.py:80`` is the corpus row that is: ten dipoles on
    ``mp.CustomSource(src_func=lambda t: np.random.randn())`` with no seeding, and
    ``CustomEnvelope.current`` calls that function LIVE at every evaluation
    (sources.py:1559-1562 through ``_custom_dipole`` :584-595), so each arrangement
    consumes draws the one before it did not. (Its two similarly named neighbours,
    ``stochastic_emitter_line.py`` and ``stochastic_emitter_reciprocity.py``, drive
    ``GaussianSource`` and are not affected -- which is exactly why the disposition is
    a measurement and not a list of file names.)

    THE PROBE IS THE ENGINE'S OWN ACCESSOR, twice at the same argument, over the
    times this row would be driven at. ``current`` is pure on every deterministic
    envelope -- it is a closed-form waveform (sources.py:1448-1449 and each
    subclass's override) -- so a row that passes is untouched by having been asked.
    ``_custom_dipole`` gates on a time window, so the sweep runs the whole budget
    rather than one instant: a waveform that is silent at ``t = 0`` cannot answer for
    the walk.
    """
    dt = float(driver.grid.dt)
    start = float(getattr(driver, "time", 0.0))
    rows: List[Dict[str, Any]] = []
    for index, source in enumerate(getattr(driver, "_sources", ()) or ()):
        disagreements: List[Dict[str, Any]] = []
        for step in range(steps + 1):
            moment = start + step * dt
            for accessor in ("current", "dipole"):
                call = getattr(source, accessor, None)
                if not callable(call):
                    continue
                try:
                    first = call(moment, dt) if accessor == "current" else call(moment)
                    second = call(moment, dt) if accessor == "current" else call(moment)
                except Exception as exc:  # noqa: BLE001 - an accessor that raises is not a verdict
                    disagreements.append({"step": step, "accessor": accessor,
                                          "error": f"{type(exc).__name__}: {exc}"[:200]})
                    continue
                if complex(first) != complex(second):
                    disagreements.append({"step": step, "accessor": accessor,
                                          "time": moment,
                                          "first": str(first), "second": str(second)})
            if disagreements:
                break
        rows.append({"index": index, "type": type(source).__name__,
                     "envelope": type(getattr(source, "envelope", source)).__name__,
                     "deterministic": not disagreements,
                     "disagreements": disagreements[:4]})
    undefined = [row for row in rows if not row["deterministic"]]
    return {
        "deterministic": not undefined,
        "sources": rows,
        "times_probed_per_source": steps + 1,
        "reason": ("" if not undefined else
                   "; ".join(f"source {row['index']} ({row['type']}, envelope "
                             f"{row['envelope']}) returned two different values from "
                             f"the same accessor at the same time -- its waveform is "
                             f"not a function of t, so the five arrangements are five "
                             f"different problems and byte identity across them is "
                             f"undefined by construction" for row in undefined)),
    }


def evaluate(driver: Any, probe: Any) -> Dict[str, Any]:  # noqa: ARG001 - the census probe is unused
    """The census driver's battery hook: the four-reference identity on ONE lifted row."""
    steps = int(os.environ.get("MEEP_GPU_HD_GATE_STEPS", STEPS))
    max_cells = os.environ.get("MEEP_GPU_HD_GATE_MAX_CELLS")
    sources = tuple(getattr(driver, "_sources", ()) or ())
    block: Dict[str, Any] = {"family": family.FAMILY, "steps_requested": steps,
                             "n_sources": len(sources),
                             "sources": [{"type": type(s).__name__,
                                          "field_type": str(getattr(s, "field_type", "")),
                                          "is_integrated": bool(getattr(s, "is_integrated", False)),
                                          "withdraw_does_work": bool(
                                              withdraw_hoist._withdraw_does_work(s))}  # noqa: SLF001
                                         for s in sources]}
    pin_array_path(driver)
    verdict = family.metal_fused_hd_pair_coverage(driver.fields, driver.pml, sources,
                                                  Residency())
    block["predicate_admits"] = bool(verdict.covered)
    block["predicate_reasons"] = list(verdict.reasons)
    composed, _residency = _composed(driver, fuse=True)
    block["composer_selected"] = dict(composed.selected)
    block["composer_refuses_this_product"] = list(
        composed.reasons.get(f"fused_pair_{family.FAMILY}", ()))
    cells = int(np.prod(driver.grid.shape))
    block["grid_cells"] = cells
    if not verdict.covered:
        block["driven"] = False
        block["why_not_driven"] = "the predicate refused this row"
        return {"metal_fused_hd_pair_gate": block}
    if max_cells and cells > int(max_cells):
        block["driven"] = False
        block["why_not_driven"] = (f"{cells} cells exceeds MEEP_GPU_HD_GATE_MAX_CELLS="
                                   f"{max_cells}; refused rather than run partially")
        return {"metal_fused_hd_pair_gate": block}
    # THE ROW'S INPUTS MUST BE THE SAME FOR EVERY ARRANGEMENT, AND THAT IS MEASURED.
    # See :func:`waveform_determinism`. A row that fails it is refused BY NAME with
    # the values that refused it, stays in the cell's denominator, and is never
    # scored as a divergence -- the comparison it would fail is undefined for it.
    determinism = waveform_determinism(driver, steps)
    block["waveform_determinism"] = determinism
    if not determinism["deterministic"]:
        block["driven"] = False
        block["undefined_by_construction"] = True
        block["why_not_driven"] = determinism["reason"]
        _child_progress(f"REFUSED as undefined: {determinism['reason'][:160]}")
        return {"metal_fused_hd_pair_gate": block}
    block["driven"] = True
    # THE FLOOR IS THE LEG'S, NOT THE ROW'S. A row answers for what it measured --
    # byte identity over every step it compared, the launch counters, the movement --
    # and whether that was ENOUGH is a judgement about the cell, made once, where the
    # rows below the floor can be named together rather than each reported as a
    # failure of the weld.
    block.update(run_product(driver, steps, progress=_child_progress,
                             movement_floor="seam", require_full_budget=False,
                             clean_floor=1))
    _child_progress(f"passed={block['passed']} words={block['words_compared']} "
                    f"({block['seconds']} s)")
    return {"metal_fused_hd_pair_gate": block}


def _load_jsonl(path: Path) -> List[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def lift_basis(results: Path) -> Tuple[List[dict], Dict[str, Any]]:
    """The rows the standing census puts in this product's cell, and the seam facts.

    DERIVED from the census's own ``plan_step.selected`` -- the arms the composer
    chose for each row -- never from a list here. The seam record supplies each
    row's interpreter and module (how it was lifted) and its ``withdraw_in_seam``
    flag, which names the rows the predicate must refuse.
    """
    census = results / CENSUS
    seam = results / SEAM_RECORD
    if not census.is_dir() or not seam.is_dir():
        raise SystemExit(f"the lift leg needs {census} and {seam}")
    seam_rows = {row["label"]: row for row in _load_jsonl(seam / "h_to_d_seam.jsonl")}
    rows: List[dict] = []
    for leg in ("examples", "tests", "tests_param_matched"):
        path = census / f"{leg}.jsonl"
        if not path.is_file():
            continue
        for record in _load_jsonl(path):
            selected = ((record.get("plan_step") or {}).get("selected") or {})
            if (selected.get("update_H"), selected.get("step_D")) != CELL_ARMS:
                continue
            label = f"{record.get('leg', leg)}:{record['row']}"
            seam_row = seam_rows.get(label, {})
            rows.append({
                "label": label, "leg": record.get("leg", leg), "row": record["row"],
                "module": record.get("module") or seam_row.get("module"),
                "interpreter": (record.get("interpreter") or record.get("python")
                                or seam_row.get("interpreter") or sys.executable),
                "grid_cells": record.get("grid_cells"),
                "grid_shape": record.get("grid_shape"),
                "withdraw_in_seam": bool((seam_row.get("h_to_d_seam") or {})
                                         .get("withdraw_in_seam")),
                "census_selected": selected,
            })
    unique = {row["label"]: row for row in rows}
    facts = {"census": CENSUS, "seam_record": SEAM_RECORD, "cell_arms": list(CELL_ARMS),
             "rows_in_cell": len(unique),
             "rows_with_a_standing_withdraw": sorted(
                 label for label, row in unique.items() if row["withdraw_in_seam"])}
    return list(unique.values()), facts


def leg_lift(out_dir: Path, steps: int, max_cells: Optional[int], timeout: float,
             resume: bool, only: Optional[Sequence[str]]) -> Dict[str, Any]:
    """Every corpus row in the cell, re-lifted in its own interpreter and driven.

    ``gate_provenance`` is imported HERE and not only in ``main``: the placeholder this
    leg writes for a child that produced nothing is stamped like any other payload, and
    a name bound only in ``main``'s scope turned that one path into a ``NameError``
    -- so the first child this leg ever lost took the whole leg down with it instead of
    being recorded as the unmeasured row it was. Measured 2026-09-06 on
    ``examples:grating2d_triangular_lattice.py``, the corpus's 11.2-million-cell row.
    """
    import gate_provenance  # noqa: PLC0415
    import measure_predicate_coverage as census  # noqa: PLC0415

    # ABSOLUTE, BEFORE ANYTHING IS BUILT FROM IT. The child runs with ``cwd`` set to
    # the lift workdir, so a relative artifact or progress path handed to it resolves
    # somewhere that does not exist -- and the child then dies AFTER the lift, with
    # the parent recording a right-shaped row that measured nothing. Measured
    # 2026-09-05: 23 of 23 rows came back ``measured: false`` from a
    # FileNotFoundError on the progress log, on a run whose only difference from a
    # green one was that ``--out`` was given relative.
    out_dir = Path(out_dir).resolve()
    rows, facts = lift_basis(HERE / "results")
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
                "reason": (f"the MEEP corpus is not at {examples_dir} / {tests_dir}; set "
                           f"MEEP_GPU_CORPUS_ROOT to the checkout the census was cut "
                           f"over. Refused rather than measured on nothing")}
    probe_path = API_ROOT / census.PROBE
    environment = dict(os.environ)
    environment.update({"KMP_DUPLICATE_LIB_OK": "TRUE", "MPLBACKEND": "Agg",
                        "PYTHONPATH": str(API_ROOT),
                        "MEEP_GPU_SUBNORMAL_POLICY": "flush",
                        "MEEP_GPU_HD_GATE_STEPS": str(steps),
                        "MEEP_GPU_HD_GATE_PROGRESS": str(progress_log)})
    if max_cells is not None:
        environment["MEEP_GPU_HD_GATE_MAX_CELLS"] = str(max_cells)
    work = lift_dir / "workdir"
    work.mkdir(exist_ok=True)
    for entry in examples_dir.iterdir():
        if entry.suffix in (".py", ".ipynb"):
            continue
        link = work / entry.name
        if not link.exists():
            with contextlib.suppress(OSError):
                link.symlink_to(entry)
    # THE CENSUS'S OWN PATH, FOR THE MODULES THAT NEED IT. Six MEEP test modules
    # decorate with ``parameterized``, which is not installed here; the census runs
    # them with ``parity/meep_gpu/shim`` ahead of the API root
    # (``measure_predicate_coverage.py``'s ``tests_param`` leg) and lifted them. This
    # leg did not, so every row from one of those modules died on the IMPORT and came
    # back ``measured: false`` -- a row scored as unmeasurable by the harness rather
    # than by the engine. Measured 2026-09-05:
    # ``tests:TestModeDecomposition.test_oblique_waveguide_backward_mode`` was the
    # cell's one unmeasured row for exactly this reason, and with the shim on the
    # path it drives clean. The shim is added ONLY for the modules whose text asks
    # for it, on the census's own test, so no other row's imports change.
    shim_path = Path(census.__file__).resolve().parent / "shim"
    needs_shim: set = set()
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
        record_path = per_row / (row["label"].replace(":", "__").replace(".", "_") + ".json")
        environment["MEEP_GPU_HD_GATE_LABEL"] = row["label"]
        environment["PYTHONPATH"] = (f"{shim_path}{os.pathsep}{API_ROOT}"
                                     if row.get("module") in needs_shim
                                     else str(API_ROOT))
        started = time.time()
        if not (resume and record_path.exists()):
            if row["leg"] == "examples":
                command = [row["interpreter"], "-u", str(Path(census.__file__).resolve()),
                           "--leg", "examples", "--child-script",
                           str(examples_dir / row["row"]),
                           "--out-json", str(record_path), "--probe", str(probe_path),
                           "--progress-log", str(progress_log),
                           "--battery", Path(__file__).stem]
            else:
                if not row["module"]:
                    log(f"lift {index}/{len(rows)} {row['label']}: REFUSED, no module "
                        f"recorded for this tests row")
                    measured.append({**row, "measured": False,
                                     "note": "no module recorded"})
                    continue
                command = [row["interpreter"], "-u", str(Path(census.__file__).resolve()),
                           "--leg", "tests", "--child-module",
                           str(tests_dir / row["module"]),
                           "--child-cases", json.dumps([row["row"]]),
                           "--out-json", str(record_path), "--probe", str(probe_path),
                           "--progress-log", str(progress_log),
                           "--battery", Path(__file__).stem]
            log(f"lift {index}/{len(rows)} {row['label']} start (cells "
                f"{row.get('grid_cells')}, {Path(row['interpreter']).parent.parent.name})")
            stderr_text = ""
            note = "child died"
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
                # STAMPED LIKE ANY OTHER PAYLOAD. This is the record a reader gets
                # when a child produced nothing, which is exactly the case where
                # "which tree was this?" is the first question -- the 2026-09-05
                # unmeasured row was a MISSING IMPORT in the parent's environment, and
                # an unstamped placeholder is how that reads as an engine refusal.
                # `test_gate_provenance` requires every payload write to be preceded
                # by a stamp, not merely the artifact's.
                died = {"row": row["row"], "measured": False, "note": note,
                        "stderr_tail": stderr_text[-1500:]}
                gate_provenance.stamp(died)
                record_path.write_text(json.dumps(died, default=str), encoding="utf-8")
        payload = json.loads(record_path.read_text(encoding="utf-8"))
        candidates = payload if isinstance(payload, list) else [payload]
        record = next((r for r in candidates if r.get("row") == row["row"]),
                      candidates[0] if candidates else {})
        record.update({key: value for key, value in row.items() if key != "label"})
        record["label"] = row["label"]
        measured.append(record)
        block = record.get("metal_fused_hd_pair_gate") or {}
        log(f"lift {index}/{len(rows)} {row['label']}: measured={record.get('measured')} "
            f"admits={block.get('predicate_admits')} driven={block.get('driven')} "
            f"passed={block.get('passed')} steps={block.get('steps_compared')} "
            f"words={block.get('words_compared')} ({time.time() - started:.1f} s)")
    with (lift_dir / "rows.jsonl").open("w", encoding="utf-8") as handle:
        for record in measured:
            handle.write(json.dumps(record, default=str) + "\n")

    blocks = {r["label"]: (r.get("metal_fused_hd_pair_gate") or {}) for r in measured}
    admitted = sorted(label for label, b in blocks.items() if b.get("predicate_admits"))
    refused = sorted(label for label, b in blocks.items()
                     if b.get("predicate_admits") is False)
    refused_by_the_withdraw = sorted(
        label for label in refused
        if any("standing integrated" in r for r in blocks[label].get("predicate_reasons", ())))
    # THE DENOMINATOR IS THE CELL'S, EXCEPT ON A DELIBERATE SUBSET. A ``--lift-only``
    # run measures the rows it names and must be scoreable on them; the FULL run is
    # the one whose pass condition carries the cell's own 49 and both refusals, and
    # ``rows_in_cell`` is reported either way so a subset can never read as the cell.
    expected_refused = sorted(facts["rows_with_a_standing_withdraw"])
    if only:
        expected_refused = sorted(set(expected_refused) & set(blocks))
    unmeasured = sorted(label for label, b in blocks.items() if "predicate_admits" not in b)
    driven = sorted(label for label in admitted if blocks[label].get("driven"))
    passed_rows = sorted(label for label in driven if blocks[label].get("passed"))
    # THE THIRD DISPOSITION, AND IT IS A MEASUREMENT RATHER THAN AN EXEMPTION. A row
    # whose source waveform is not a function of time hands each arrangement a
    # different current, so the comparison this leg makes is undefined for it -- not
    # failed. It stays in ``admitted`` and therefore in the cell's denominator, it is
    # named here with the values that refused it, and it is subtracted from what
    # ``driven`` is required to be. ``waveform_determinism`` is what says so.
    undefined = {label: blocks[label].get("why_not_driven")
                 for label in admitted
                 if blocks[label].get("undefined_by_construction")}
    not_driven = {label: blocks[label].get("why_not_driven")
                  for label in admitted
                  if not blocks[label].get("driven") and label not in undefined}
    composer_refuses = all(blocks[label].get("composer_refuses_this_product")
                           for label in admitted)
    # THE THREE OUTCOMES A DRIVEN ROW CAN HAVE, AND ONLY ONE OF THEM IS A FAILURE.
    #
    #   diverged          the weld or a reference disagreed on a step the flush
    #                     precondition held for. That is what this gate exists to
    #                     find, and it fails the leg by name.
    #   below_the_floor   byte-identical over every step it compared, but the ROW's
    #                     own state entered the denormal band before
    #                     LIFT_CLEAN_STEP_FLOOR complete steps. Nothing about the weld
    #                     is wrong there; the fixture cannot support the claim, and
    #                     the gate does not choose that fixture. Named, with its first
    #                     banded step, and counted -- never quietly dropped and never
    #                     scored as a defect.
    #   cleared_the_floor byte-identical over at least the floor.
    diverged = sorted(label for label in driven
                      if not blocks[label].get("bit_identical"))
    # WHICH SIDE OF THE COMPARISON MOVED. A row where the weld still agrees with the
    # certified singles but some arrangement disagrees with the array path is a
    # finding about a REFERENCE composition on that row, not about this weld, and the
    # record has to say so -- it still fails the leg, because a gate whose references
    # disagree has certified nothing, but it fails naming the right subject.
    weld_disagrees = sorted(label for label in diverged
                            if blocks[label].get("weld_agrees_with_the_certified_singles")
                            is False)
    reference_disagrees = {
        label: blocks[label].get("references_that_disagree_with_the_array_path")
        for label in diverged
        if blocks[label].get("references_that_disagree_with_the_array_path")}
    seam_alone_disagrees = sorted(
        label for label in diverged
        if blocks[label].get("the_seam_alone_agrees_with_the_array_path") is False)
    below = {label: {"steps_compared": blocks[label].get("steps_compared"),
                     "first_banded_step": blocks[label].get("first_banded_step")}
             for label in driven
             if blocks[label].get("bit_identical")
             and int(blocks[label].get("steps_compared") or 0) < LIFT_CLEAN_STEP_FLOOR}
    cleared = sorted(label for label in driven
                     if label not in below and label not in diverged)
    # NON-VACUITY, STATED: a cell in which most rows cannot carry even the floor is a
    # cell this leg has not really measured, whatever its zero divergences say. Half
    # is the bar, and the count is reported either way.
    majority_cleared = len(cleared) * 2 >= len(driven) if driven else False
    # AND THE UNDEFINED ROWS ARE BOUNDED. A disposition that can grow without limit is
    # an escape hatch; this one may take at most a tenth of the cell, and the leg fails
    # if it grows past that rather than reporting a shrinking measurement as a pass.
    undefined_within_bounds = len(undefined) * 10 <= facts["rows_in_cell"]
    return {
        "passed": bool(measured and not unmeasured
                       and (not only and len(admitted) + len(refused) == facts["rows_in_cell"]
                            or bool(only))
                       and refused == refused_by_the_withdraw == expected_refused
                       and not not_driven
                       and not diverged
                       and undefined_within_bounds
                       and passed_rows == driven
                       and sorted(set(driven) | set(undefined)) == admitted
                       and majority_cleared
                       and composer_refuses),
        "rows_undefined_by_construction": undefined,
        "undefined_rows_are_within_a_tenth_of_the_cell": undefined_within_bounds,
        "rows_that_diverged": diverged,
        "rows_where_the_weld_disagrees_with_the_certified_singles": weld_disagrees,
        "rows_where_the_SEAM_ALONE_disagrees_with_the_array_path": seam_alone_disagrees,
        "rows_where_a_reference_disagrees_with_the_array_path": reference_disagrees,
        "rows_below_the_clean_step_floor": below,
        "rows_that_cleared_the_clean_step_floor": cleared,
        "majority_cleared_the_floor": majority_cleared,
        "facts": facts,
        "rows_in_cell": facts["rows_in_cell"],
        "rows_measured": len(measured),
        "rows_unmeasured": unmeasured,
        "admitted": len(admitted),
        "admitted_rows": admitted,
        "refused": refused,
        "refused_by_the_standing_withdraw": refused_by_the_withdraw,
        "refused_expected_from_the_seam_record": expected_refused,
        "driven": len(driven),
        "undefined": len(undefined),
        "not_driven": not_driven,
        "rows_bit_identical": len(passed_rows),
        "clean_step_floor": LIFT_CLEAN_STEP_FLOOR,
        "first_banded_step_half_per_row": {
            label: blocks[label].get("which_half_of_the_precondition_fired")
            for label in driven},
        "steps_compared_per_row": {label: blocks[label].get("steps_compared")
                                   for label in driven},
        "first_banded_step_per_row": {label: blocks[label].get("first_banded_step")
                                      for label in driven},
        "complete_driver_steps_compared": int(sum(
            blocks[label].get("steps_compared") or 0 for label in driven)),
        "what_the_budget_is_on_a_lifted_row": (
            f"every complete driver step the flush precondition holds for, capped at "
            f"the requested budget and floored at {LIFT_CLEAN_STEP_FLOOR}. A lifted "
            f"row starts at the engine's zeros and is driven by its own sources, so "
            f"its own state enters the float32 denormal band on its own schedule; "
            f"past that step MPS flushes where NumPy does not and no byte claim can "
            f"be made. Each row's step count and first banded step are recorded"),
        "composer_refuses_this_product_on_every_admitted_row": composer_refuses,
        "modules_lifted_with_the_parameterized_shim": sorted(needs_shim),
        "rows_lifted_with_the_parameterized_shim": sorted(
            row["label"] for row in rows if row.get("module") in needs_shim),
        "words_compared": int(sum(blocks[label].get("words_compared", 0) for label in driven)),
        "steps": steps,
        "rows_jsonl": str(lift_dir / "rows.jsonl"),
    }


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------

def emit(handle: Any, row: Dict[str, Any]) -> None:
    handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
    handle.flush()
    os.fsync(handle.fileno())
    log(f"{row['index']}/{row['total']} [{row['leg']}] {row['label']}: "
        f"passed={row.get('passed')} "
        f"first_divergence={row.get('first_divergence', row.get('hazard_first_divergence', '-'))}")


def parse_legs(value: str) -> Tuple[str, ...]:
    if value in ("all", ""):
        return ALL_LEGS
    wanted: List[str] = []
    for token in value.split(","):
        token = token.strip()
        if token in LEG_GROUPS:
            wanted.extend(LEG_GROUPS[token])
        elif token in ALL_LEGS:
            wanted.append(token)
        else:
            raise SystemExit(f"unknown leg {token!r}; legs are {ALL_LEGS} or groups "
                             f"{tuple(LEG_GROUPS)}")
    return tuple(dict.fromkeys(wanted))


def _mutation_row(name: str, result: Mapping[str, Any], why: str,
                  host_defect: bool = False) -> Dict[str, Any]:
    weld = result["arrangements"]["weld"]
    first = weld["first_divergence"]
    in_place_first = next((row["step"] for row in weld["per_step"] if in_place_words(row)),
                          None)
    # THE PRECONDITION IS PART OF "CAUGHT". A walk that stopped because the oracle
    # entered the denormal band has fewer rows than the budget, which makes
    # ``identical`` False for every arrangement -- so a banded run would report every
    # mutant as caught and the shipped kernel as diverging. The census is asserted
    # here rather than left to the product leg, because a mutation row that read
    # "caught" for that reason is the vacuous pass this leg exists to prevent.
    caught = not weld["identical"] and result["precondition_clean"]
    return {"caught": caught, "passed": bool(caught and weld["launches"]["plans"]),
            "precondition_clean": result["precondition_clean"],
            "first_banded_step": result["first_banded_step"],
            "host_defect": host_defect, "why_it_must_fire": why,
            "first_divergence": first, "first_divergence_in_place": in_place_first,
            "differing_volumes": weld["differing_volumes_final"],
            "launches": weld["launches"],
            # NO REFERENCE DIVERGED IN THE STEPS THAT RAN. Not ``identical``, which
            # additionally requires the full budget: a caught mutant stops the walk
            # at its first divergence, so ``identical`` would read False for every
            # reference on every caught row and look like a three-way disagreement.
            "references_still_agree": all(
                result["arrangements"][name]["first_divergence"] is None
                for name in ("singles", "composition_today", "unfused"))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True,
                        help="the artifact: results/metal_fused_hd_pair_<stamp>/gate.json")
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--legs", default="all",
                        help="comma list of legs or groups (host, compile, device); "
                             "anything short of all exits 75 (cannot certify)")
    parser.add_argument("--lift-max-cells", type=int, default=None,
                        help="refuse (by name) to drive corpus rows above this many cells")
    parser.add_argument("--lift-timeout", type=float, default=10800.0)
    parser.add_argument("--lift-resume", action="store_true")
    parser.add_argument("--lift-only", action="append", default=None)
    parser.add_argument("--progress-log", type=Path, default=None)
    args = parser.parse_args()
    args.out = args.out.resolve()
    if args.progress_log is not None:
        args.progress_log = args.progress_log.resolve()
    if args.steps < 1:
        raise SystemExit("--steps must be positive")
    legs = parse_legs(args.legs)

    progress_path = args.progress_log

    def progress(message: str) -> None:
        log(message)
        if progress_path:
            with open(progress_path, "a", encoding="utf-8") as handle:
                handle.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")

    report = subnormal.mps_policy_report()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    jsonl = args.out.with_suffix(".jsonl")
    started = time.perf_counter()
    rows: List[Dict[str, Any]] = []
    cases = dict(CASES)
    codes = MUTATION_CODES
    assert all(codes), (f"{MUTATION_CASE} does not wall every axis; the wall "
                        f"mutations would be armed on lines the shipped kernel "
                        f"does not emit")
    assert not any(PERIODIC_CODES), (
        f"{PERIODIC_MUTATION_CASE} walls an axis; the wrap mutations would be armed "
        f"on a specialisation that clamps instead of wrapping")
    needs_device = any(leg in LEG_GROUPS["device"] or leg in LEG_GROUPS["compile"]
                       for leg in legs)
    device_ok = True
    if needs_device:
        try:
            import torch  # noqa: PLC0415
            device_ok = bool(torch.backends.mps.is_available())
        except Exception:  # noqa: BLE001
            device_ok = False
    mutants = shader_mutations(codes)
    periodic_mutants = periodic_wrap_mutations(PERIODIC_CODES)
    inert = inert_mutations(codes)
    total = (len([leg for leg in legs if leg not in ("product", "mutation", "arbitration")])
             + (len(CASES) if "product" in legs else 0)
             + (len(mutants) + len(periodic_mutants) + len(HOST_MUTATIONS)
                + len(inert) + len(PREDICTED_NULL) if "mutation" in legs else 0)
             + (2 if "arbitration" in legs else 0))
    index = 0

    def add(leg: str, label: str, payload: Mapping[str, Any]) -> None:
        nonlocal index
        index += 1
        row = {"index": index, "total": total, "leg": leg, "label": label, **payload}
        rows.append(row)
        emit(handle, row)

    with jsonl.open("w", encoding="utf-8") as handle:
        if not report["admitted"]:
            # REFUSED BY NAME, NEVER RUN: a keep-resolved process cannot certify bytes
            # under a policy the device was never in.
            add("policy", "resolved_subnormal_policy_refused",
                {"passed": False, "refused": True, "report": report})
            legs = tuple(leg for leg in legs if leg in LEG_GROUPS["host"])
        if needs_device and not device_ok:
            add("device", "mps_unavailable",
                {"passed": False, "refused": True,
                 "reason": "MPS is not available; the compile and device legs cannot run"})
            legs = tuple(leg for leg in legs if leg in LEG_GROUPS["host"])

        if "driver_order" in legs:
            add("driver_order", "replaces_is_the_drivers_two_adjacent_consults",
                leg_driver_order())
        if "transcription" in legs:
            add("transcription", "both_halves_are_the_certified_bytes",
                leg_transcription(codes))
        if "refusal" in legs:
            add("refusal", "by_name", leg_refusal())
        if "arbitration" in legs:
            for name in ("bc_ppp", MUTATION_CASE):
                add("arbitration", f"{name}:the_released_pair_keeps_update_H",
                    leg_arbitration(cases[name]))
        if "seed_scale" in legs:
            add("seed_scale", "the_power_of_two_seed_changes_no_mantissa",
                leg_seed_scale())
        if "purity_ledger" in legs:
            add("purity_ledger", "foreign_taps_on_moved_cells", leg_purity_ledger())
        if "private_scratch" in legs:
            add("private_scratch", "poisoned_private_buffers_change_no_stored_word",
                leg_private_scratch())
        if "binding_ceiling" in legs:
            add("binding_ceiling", "thirty_pointers_plus_one_packed_struct_is_the_ceiling",
                leg_binding_ceiling())
        if "mutants_compile" in legs:
            add("mutants_compile", "every_armed_defect_compiles", leg_mutants_compile(codes))

        if "product" in legs:
            for offset, (name, keywords) in enumerate(CASES):
                driver = build_driver(keywords, 90100 + offset)
                result = run_product(driver, args.steps,
                                     progress=lambda m, n=name: progress(f"product {n} {m}"))
                result["case"] = name
                result["keywords"] = dict(keywords)
                result["first_divergence"] = min(
                    (a["first_divergence"] for a in result["arrangements"].values()
                     if a["first_divergence"] is not None), default=None)
                add("product", name, result)
        if "launch_structure" in legs:
            add("launch_structure", f"{MUTATION_CASE}:launches_at_the_seam_and_over_the_step",
                leg_launch_structure(build_driver(cases[MUTATION_CASE], 91000), args.steps))
        if "rotation_guard" in legs:
            add("rotation_guard", f"{MUTATION_CASE}:an_orphaned_mirror_is_named",
                leg_rotation_guard(build_driver(cases[MUTATION_CASE], 91500)))
        if "sync" in legs:
            add("sync", f"{MUTATION_CASE}:flux_in_box_mid_run",
                leg_sync(build_driver(cases[MUTATION_CASE], 92000), max(SYNC_STEPS) + 3))
        if "withdraw" in legs:
            add("withdraw", f"{MUTATION_CASE}:integrated_electric_source_in_the_seam",
                leg_withdraw(cases[MUTATION_CASE], 93000, args.steps))
        if "byte_neutral" in legs:
            neutral = {shaders.CONTRACT_OFF:
                       compile_source(byte_neutral_source(codes)).fused_hd_pair_step}
            result = run_product(build_driver(cases[MUTATION_CASE], 94000), args.steps,
                                 functions=neutral)
            weld = result["arrangements"]["weld"]
            add("byte_neutral", "register_replaced_by_a_reload_of_the_same_word",
                {"passed": bool(result["passed"]), "diverged": not weld["identical"],
                 "differing_words": weld["differing_words_final"],
                 "launches": weld["launches"]})
        if "mutation" in legs:
            for name, (source, why) in mutants.items():
                functions = {shaders.CONTRACT_OFF: compile_source(source).fused_hd_pair_step}
                result = run_product(build_driver(cases[MUTATION_CASE], 95000), args.steps,
                                     functions=functions)
                add("mutation", name, _mutation_row(name, result, why))
            # THE PERIODIC SET, ON THE PERIODIC CASE. Armed against a different
            # specialisation because it is a different specialisation's line: the
            # metallic kernel clamps where this one wraps.
            for name, (source, why) in periodic_mutants.items():
                functions = {shaders.CONTRACT_OFF: compile_source(source).fused_hd_pair_step}
                result = run_product(build_driver(cases[PERIODIC_MUTATION_CASE], 95500),
                                     args.steps, functions=functions)
                row = _mutation_row(name, result, why)
                row["case"] = PERIODIC_MUTATION_CASE
                add("mutation", name, row)
            # THE HOST DEFECTS, WITHOUT THE EARLY STOP. ``rotation_skipped`` makes TWO
            # claims at two different steps -- every volume at 1, the in-place
            # displacement at 2 -- and a walk that stopped at the first divergence
            # could never reach the second one, so it would report the harder half as
            # unmeasured while calling the mutation caught.
            for name, (launcher, _scope, why) in HOST_MUTATIONS.items():
                result = run_product(build_driver(cases[MUTATION_CASE], 96000), args.steps,
                                     launcher=launcher, stop_on_divergence=False)
                row = _mutation_row(name, result, why, host_defect=True)
                if name == "rotation_skipped":
                    row["passed"] = bool(row["passed"] and row["first_divergence"] == 1
                                         and row["first_divergence_in_place"] == 2)
                    row["expected"] = {"first_divergence": 1, "first_divergence_in_place": 2}
                add("mutation", name, row)
            # THE ARMED DEFECTS REQUIRED NOT TO FIRE. Scored on AGREEING, with the
            # named control's own outcome carried on the row: an inert mutant whose
            # control did not fire is a hole rather than a fact about the kernel.
            for name, (source, reason, control) in inert.items():
                functions = {shaders.CONTRACT_OFF: compile_source(source).fused_hd_pair_step}
                result = run_product(build_driver(cases[MUTATION_CASE], 95800), args.steps,
                                     functions=functions)
                weld = result["arrangements"]["weld"]
                control_caught = next((r.get("caught") for r in rows
                                       if r["leg"] == "mutation"
                                       and r["label"] == control), None)
                add("mutation", name,
                    {"passed": bool(weld["identical"] and weld["launches"]["plans"]
                                    and weld["launches"]["plans"]
                                    == weld["launches"]["functions"]
                                    and control_caught is True),
                     "armed": True, "scored_on": "agreeing", "caught": False,
                     "inert_by_construction": True, "reason": reason,
                     "control": control, "control_caught": control_caught,
                     "differing_words": weld["differing_words_final"],
                     "launches": weld["launches"],
                     "references_still_agree": all(
                         result["arrangements"][other]["first_divergence"] is None
                         for other in ("singles", "composition_today", "unfused"))})
            for entry in PREDICTED_NULL:
                control_caught = next((r.get("caught") for r in rows
                                       if r["leg"] == "mutation"
                                       and r["label"] == entry["control"]), None)
                add("mutation", entry["name"],
                    {"passed": bool(control_caught is True), "predicted_null": True,
                     "armed": False, "scored_on": "the control fires",
                     "reason": entry["reason"], "control": entry["control"],
                     "control_caught": control_caught})
        if "disarm" in legs:
            result = run_product(build_driver(cases[MUTATION_CASE], 97000), args.steps)
            weld = result["arrangements"]["weld"]
            add("disarm", "shipped_bytes_on_the_mutation_case",
                {"passed": bool(result["passed"]), "differing_words": weld["differing_words_final"],
                 "launches": weld["launches"],
                 "arrays_that_never_moved": result["arrays_that_never_moved"]})
        if "lift" in legs:
            add("lift", "every_corpus_row_in_the_cell",
                leg_lift(args.out.parent, args.steps, args.lift_max_cells,
                         args.lift_timeout, args.lift_resume, args.lift_only))

    ran = tuple(dict.fromkeys(row["leg"] for row in rows))
    complete = all(leg in ran for leg in ALL_LEGS)
    all_passed = bool(rows) and all(row["passed"] for row in rows)
    if complete:
        verdict = "PASS" if all_passed else "FAIL"
    else:
        verdict = ("INCOMPLETE: not every leg ran -- "
                   f"{'all requested legs passed' if all_passed else 'a requested leg FAILED'}; "
                   f"missing {sorted(set(ALL_LEGS) - set(ran))}")
    import torch  # noqa: PLC0415

    import gate_provenance  # noqa: PLC0415

    result = {
        "verdict": verdict,
        "complete": complete,
        "legs_requested": list(legs),
        "legs_run": list(ran),
        "legs_missing": sorted(set(ALL_LEGS) - set(ran)),
        "elapsed_seconds": time.perf_counter() - started,
        "steps": args.steps,
        "rows": rows,
        "counts": {"product_cases": len(CASES) if "product" in ran else 0,
                   "shader_mutations": len(mutants) if "mutation" in ran else 0,
                   "periodic_shader_mutations": (len(periodic_mutants)
                                                 if "mutation" in ran else 0),
                   "host_mutations": len(HOST_MUTATIONS) if "mutation" in ran else 0,
                   "inert_mutations_scored_on_agreeing": (len(inert)
                                                          if "mutation" in ran else 0),
                   "predicted_null": len(PREDICTED_NULL) if "mutation" in ran else 0,
                   "byte_neutral_controls": 1 if "byte_neutral" in ran else 0,
                   "sync_steps": len(SYNC_STEPS) if "sync" in ran else 0,
                   "lift_rows_admitted": next((r.get("admitted") for r in rows
                                               if r["leg"] == "lift"), None),
                   "lift_rows_driven": next((r.get("driven") for r in rows
                                             if r["leg"] == "lift"), None),
                   "lift_rows_undefined_by_construction": next(
                       (r.get("undefined") for r in rows if r["leg"] == "lift"), None),
                   "lift_complete_driver_steps": next(
                       (r.get("complete_driver_steps_compared") for r in rows
                        if r["leg"] == "lift"), None),
                   "lift_words_compared": next(
                       (r.get("words_compared") for r in rows
                        if r["leg"] == "lift"), None)},
        "mutation_case": MUTATION_CASE,
        "periodic_mutation_case": PERIODIC_MUTATION_CASE,
        "mutation_specialisation": {"codes": list(codes),
                                    "periodic_codes": list(PERIODIC_CODES)},
        "references": {
            "1": "the array path under the driver's own loop",
            "2": "the certified singles update_H then step_D, dispatched",
            "3": "plan_step(fuse=True): the composition the composer installs today",
            "4": "plan_step(fuse=False): the same slots dispatched unfused",
        },
        "signature": {"pointers": family.PACKED_BINDINGS - 1,
                      "packed_bindings": family.PACKED_BINDINGS,
                      "ceiling": MAX_BUFFER_BINDINGS,
                      "headroom": MAX_BUFFER_BINDINGS - family.PACKED_BINDINGS},
        "product": {"family": family.FAMILY, "slot": family.SLOT,
                    "replaces": list(family.REPLACES), "seam": family.SEAM,
                    "installable": family.INSTALLABLE,
                    "hoists_the_withdraw": family.HOISTS_THE_WITHDRAW,
                    "carries_deposit_repair": family.CARRIES_DEPOSIT_REPAIR,
                    "absorb_row": metal_launch.FUSED_PAIR_ARMS.get(family.FAMILY)},
        "corpus": {"census": CENSUS, "seam_record": SEAM_RECORD, "cell_arms": list(CELL_ARMS)},
        "subnormal_policy": report,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "metal_frontend": metal_frontend_version(),
        "jsonl": str(jsonl),
        "what_this_does_not_license": (
            "installing the product (the composer refuses it by name and the arbitration "
            "leg pins that), flipping HOISTS_THE_WITHDRAW (the withdraw leg licenses the "
            "wiring, not the flag), any throughput claim (launch counts are not time), "
            "the (folded -> folded) cell, cylindrical storage, the nonlinear widening "
            "(admitted by the predicate, driven by no leg here), a keep-resolved run, "
            "or any step past a row's own first banded step -- the byte claim is over "
            "the steps BOTH halves of the flush precondition held for, and each row's "
            "count and first banded step are recorded beside it"),
        "source_sha256": {
            "family": hashlib.sha256(Path(family.__file__).read_bytes()).hexdigest(),
            "gate": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
    }
    gate_provenance.stamp(result)  # bytes THIS process imported; see gate_provenance
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n",
                        encoding="utf-8")
    log(f"VERDICT {verdict} in {result['elapsed_seconds']:.2f}s; artifact {args.out}")
    if not all_passed:
        return 1
    return 0 if complete else EXIT_INCOMPLETE


if __name__ == "__main__":
    from metal_gate_runner import run_current_measurement

    raise SystemExit(run_current_measurement(__file__, sys.argv[1:]))
