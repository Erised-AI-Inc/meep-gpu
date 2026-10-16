#!/usr/bin/env python
"""DEVICE byte gate for the Triton COMPLEX H->D weld: complex ``update_H`` welded
into complex ``step_D``.

THE COMPLEX TWIN OF ``gate_triton_fused_hd_pair.py``, on the
``(update_H complex -> step_D complex PML)`` H->D cell -- 17 corpus instances, all
``buildable_not_built``, none carrying a standing in-seam withdraw. Read that gate
first: the seam, the four-reference identity, the arrangement machinery, the word
comparison and the lift child are ITS, imported here rather than re-spelled, and its
bytes are bound in :data:`SOURCES`. This file states only what the complex arm adds.

WHAT IS BEING CERTIFIED. ``meep_gpu/triton_kernels/complex_fused_hd_pair.py`` computes
the complex ``update_H`` constitutive into launch-local SCRATCH on both float32 word
planes, takes its own cell's magnetic field from registers, RECOMPUTES every one of the
curl half's six foreign taps from pre-launch state, leaves the Bloch phase block
standing on the loaded register, steps ``D``/``fu_D`` in place, and rotates the
``H``/``f_w_H`` references afterwards. The claim is per COMPLETE DRIVER STEP, as uint32
WORDS over every stored volume the engine allocates (never ``allclose``, because
``-0.0 == 0.0`` lies).

=============================================================================
WHAT THE COMPLEX ARM ADDS TO THE REAL GATE'S LEGS
=============================================================================

**BOTH POLICIES REACH THE COMPOSER.** No complex predicate carries a policy clause, so
under either policy the predicate answers, the composer selects the complex arms, and
all six reference arrangements are built. ``composer_reachable_under_this_policy`` in
the record says so.

**EVERY PRODUCT ON COMPLEX STORAGE IS FORMED PART BY PART.** The certified complex
bodies (``complex_fields``) take the root kernel's forms on each stored part: MEEP's
left-to-right curl stencil, the chunk-keyed split-field recurrence, the one-add
constitutive accumulation, every PML coefficient a real product per part. No complex
multiply and no multiply-add is left in this kernel, so the previous arithmetic
mutations (the folded zero cross terms, the unary negation of an FMA addend, the two
multiply orientations) have nothing to plant into. What is armed instead is the form
contract (KERNELS.md 7.1 inversions and the 7.2 battery, on both halves) and the Bloch
connection.

**A COMPLEX-BY-REAL DIVIDE DOES NOT OCCUR IN THIS KERNEL, AND THAT IS MEASURED THREE
WAYS RATHER THAN CITED.** ``divide_free_evidence`` counts zero true divisions over the
parse tree of all three device functions, the laptop suite plants a divide and a
``div_rn`` and requires both counts to move, and this gate arms
``m_divide_by_the_reciprocal_of_sinv`` -- one ``sinv`` load respelled as a divide by
its reciprocal -- and requires the divergence on the planted tiny-normal case (under
``flush`` the policy's own PTX audit refuses the planted ``div.full.f32``).

**THE ROW-ZERO TINY-NORMAL CASE STAYS PLANTED.** ``planted_row0_tiny`` seeds row 0 of
every stored volume with tiny normals of both signs beside exact zeros, so the product
leg carries the signed-zero class through both halves on every policy.

**THE BLOCH CONNECTION IS ITS OWN SET OF MUTATIONS.** The certified curl connects the
UNSHIFTED partners on plane 0 of each partner's axis (MEEP's not-owned word,
``complex_fields._connect``, selected by ``tl.where``); this weld redirects the loads
and leaves that block standing. ``m_connection_applied_to_every_lane``,
``m_connection_dropped_on_x`` and ``m_connection_on_the_old_wrapped_plane`` (the
inversion of the previous wrapped-lane rotation) are scored on a PHASED fixture,
because on an unphased one the ``CN*`` constexpr compiles the block away; the host
mutation ``m_connection_direction_swapped`` binds step_B's ``conj(eikna)`` where step_D
reads ``eikna``.

**MEEP'S START OF H IS MEASURED, NOT ASSUMED.** Every fixture is seeded MID-RUN
(``Fields.h_separated`` True) except ``bc_mmm_first_update``, the run's first
``update_H``: every arrangement starts H and its history from B at step 1 (the weld
through its ``start_h`` scalar), and ``b11_start_of_H_ignored`` is scored there.

LEGS
  host legs (no device launch)
  driver_order       REPLACES is exactly the driver's two adjacent consults, the only
                     statement between them is the magnetic f_minus_p restore, and the
                     sync channel's containment rule excludes step_D
  transcription      both halves are the certified COMPLEX kernels' own text: the
                     constitutive tail differs in EXACTLY the declared lift edits, the
                     curl tail in EXACTLY the twelve redirected magnetic word loads,
                     every tap's cell and guard is PARSED from the emitter's own lines,
                     no g/e pointer survives, the divide count is zero, the helper set
                     is the connection alone, and every mutation needle resolves once
  refusal            by name: an undeclared source list; a standing INTEGRATED electric
                     withdraw; a non-integrated electric source NOT refused; a magnetic
                     source NOT refused; a folded grid naming the folded-complex cells;
                     a cylindrical grid; REAL storage
  arbitration        the composer, asked, with this product's row INJECTED IN PROCESS
  purity_ledger      per fixture, on the array path: of the curl's valid foreign taps,
                     how many land on a cell complex update_H MOVED
  ghost_observability the metallic ghost's unobservability, DERIVED from the certified
                     text, which is what makes one ghost mutation a recorded null
  device legs
  product            the identity on every boundary triple, three phased fixtures,
                     the planted tiny-normal case and the run's first update_H,
                     complete driver steps
  lift               every corpus row the standing census puts in the cell, re-lifted
                     in its own interpreter which INSTALLS THE SAME SUBNORMAL POLICY
  launch_structure   launches per step at the seam and over the whole step, two
                     independent counters
  sync               the product FORCE-installed; a plan spanning step_D must DECLINE
  withdraw           the seam's one pass (hd_seam_restore.seam_restore_leg): integrated
                     electric AND magnetic sources, the weld identical to the array
                     path, the restore moved before the launch and dropped both diverge
  seed_scale         the 2^80 seed scale is a change of exponent and nothing else
  byte_neutral       the own-cell register pair replaced by a reload of the scratch
  mutation           the armed kernel and host defects, each with its declared outcome
  disarm             the identical harness, shipped bytes, must not diverge

Progress reporting: one flushed line per case; every row appended and fsynced as it lands.

    CUDA_VISIBLE_DEVICES=<n> TRITON_LIBCUDA_PATH=$HOME/triton_libcuda_stub \\
      LD_LIBRARY_PATH=$HOME/triton_libcuda_stub \\
      CUPY_CACHE_DIR=<fresh>/cupy_cache/ftz_stripped TRITON_CACHE_DIR=<fresh>/triton \\
      MEEP_GPU_CORPUS_ROOT=<meep>/python PYTHONPATH=. \\
      python -u parity/meep_gpu/gate_triton_complex_fused_hd_pair.py \\
      --subnormal-policy keep --out <fresh>/keep/gate.json

    # laptop: the host legs only
    PYTHONPATH=. python -u \\
      parity/meep_gpu/gate_triton_complex_fused_hd_pair.py --no-device \\
      --out <fresh>/no_device.json
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
# BY NAME, never by parents[N]: a moved harness resolving a wrong root measures
# nothing, and the parent then writes a full-length, right-shaped artifact.
API_ROOT = next(parent for parent in HERE.parents
                if (parent / "meep_gpu" / "triton_kernels").is_dir())
for _path in (str(API_ROOT), str(HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import gate_provenance  # noqa: E402
import gate_triton_fused_hd_pair as kit  # noqa: E402
import h_to_d_seam  # noqa: E402
import hd_seam_restore  # noqa: E402

from meep_gpu import stepping, withdraw_hoist  # noqa: E402
from meep_gpu.fastpath import (  # noqa: E402
    SYNC_PASS_OWNERS, SYNC_PATH_SLOTS, SYNC_UPDATE_H_PASS,
)
from meep_gpu.triton_kernels import complex_fields  # noqa: E402
from meep_gpu.triton_kernels import complex_fused_hd_pair as family  # noqa: E402
from meep_gpu.triton_kernels import coverage as tcoverage  # noqa: E402
from meep_gpu.triton_kernels import launch as triton_launch  # noqa: E402
from meep_gpu.triton_kernels.offdiag_scratch_weld import twin_table  # noqa: E402

GATE = "triton_complex_fused_hd_pair"

#: Every module whose bytes this gate's verdict depends on. The REAL gate is bound
#: because this file imports its arrangement machinery, its word comparison and its
#: lift child: a change there changes what this verdict measured.
SOURCES: Tuple[str, ...] = (
    "meep_gpu/triton_kernels/complex_fused_hd_pair.py",
    "meep_gpu/triton_kernels/fused_hd_pair.py",
    "meep_gpu/triton_kernels/offdiag_scratch_weld.py",
    "meep_gpu/triton_kernels/complex_fields.py",
    "meep_gpu/triton_kernels/kernels.py",
    "meep_gpu/triton_kernels/forms.py",
    "meep_gpu/triton_kernels/coverage.py",
    "meep_gpu/triton_kernels/launch.py",
    "meep_gpu/stepping.py",
    "meep_gpu/withdraw_hoist.py",
    "meep_gpu/subnormal_policy.py",
    # THE DRIVER IS BOUND ON PURPOSE. This weld's whole `REPLACES` claim is about WHICH
    # passes run between `update_H` and `step_D`. A driver that reordered its seam would
    # leave every clause here reading correctly while the bytes described a step nothing
    # performs.
    "meep_gpu/driver.py",
    "meep_gpu/fields.py",
    "meep_gpu/fastpath.py",
    "parity/meep_gpu/gate_triton_fused_hd_pair.py",
    "parity/meep_gpu/gate_triton_complex_fused_hd_pair.py",
)

#: The budget every synthetic case runs, and it is the real gate's own: the state this
#: weld carries between steps (the rotated H/f_w_H pair AND the in-place fu_D) compounds
#: across steps, and the comparison is per COMPLETE STEP.
STEPS = kit.STEPS

#: The steps at which the sync leg calls the flux accessor.
SYNC_STEPS = kit.SYNC_STEPS

#: How many clean complete steps a lifted corpus row must reach to count.
LIFT_CLEAN_STEP_FLOOR = kit.LIFT_CLEAN_STEP_FLOOR

#: The synthetic fixture's geometry and material -- the real gate's, so the two records
#: are read side by side and the only difference is the storage and the phase.
CELL = kit.CELL
RESOLUTION = kit.RESOLUTION
COURANT = kit.COURANT
PML_CELLS = kit.PML_CELLS
EPSILON = kit.EPSILON
SEED_SCALE_BITS = kit.SEED_SCALE_BITS

#: THE FIXTURES. Eight boundary triples (the ghost rule is exactly what specialisation
#: changes and the recompute's guard is what a redirect is most likely to eat), THREE
#: PHASED cases (an unphased axis compiles the whole Bloch block away, so a phase
#: mutation scored anywhere else is a disarmed leg wearing a pass), and one PLANTED
#: tiny-normal case (the class where a wrong complex multiply is byte-visible, measured
#: unreachable by a random battery on the Dcyl round).
CASES: Tuple[Tuple[str, Dict[str, Any]], ...] = tuple(
    (f"bc_{'m' if x else 'p'}{'m' if y else 'p'}{'m' if z else 'p'}",
     {"boundaries": {axis: "metallic" for axis, walled in zip("xyz", (x, y, z))
                     if walled}})
    for x in (0, 1) for y in (0, 1) for z in (0, 1)) + (
    ("bloch_x", {"k_point": (0.37, 0.0, 0.0)}),
    ("bloch_xy", {"k_point": (0.37, -0.21, 0.0)}),
    # THE BRILLOUIN EDGE on z, where the wrap factor is EXACTLY -1+0j: the one phase
    # whose imaginary part is a hard zero, so a conjugation defect cannot hide in it
    # and a zero-cross-term defect has nowhere else to show.
    ("bloch_xyz_edge", {"k_point": (0.37, -0.21, 0.5)}),
    # SEEDED AT 2^0, NOT 2^80, AND THAT IS WHAT MAKES THE PLANT LIVE. Everywhere else
    # this gate scales the seed by 2^80 to keep the state clear of the denormal band;
    # here the whole point is that a PRODUCT underflows, and row 0 at 1e-38 beside a
    # bulk at 1e24 would be swamped by the first curl. At unit scale row 0's own
    # constitutive stays in the tiny-normal class for the step the arithmetic
    # mutations are scored on.
    ("planted_row0_tiny", {"boundaries": {"x": "metallic"}, "plant": "row0_tiny",
                           "scale_bits": 0, "amplitude": 0.37}),
    # THE RUN'S FIRST update_H on the all-walled triple: unseparated, so every
    # arrangement performs MEEP's start of H (stepping.separate_h_from_b) at step 1 --
    # the array path in update_H, the weld through its start_h scalar.
    ("bc_mmm_first_update",
     {"boundaries": {"x": "metallic", "y": "metallic", "z": "metallic"},
      "h_started": False}),
)

#: The case MEEP's start of H is scored on.
FIRST_UPDATE_CASE = "bc_mmm_first_update"

#: The case the structural mutations are armed on: every axis walled, so every line the
#: shipped kernel can emit is present and the metallic ghost rule is live on every axis.
MUTATION_CASE = "bc_mmm"

#: THE SECOND MUTATION CASE. On a metallic axis ``stepping._mask_non_owned_cells``
#: zeroes the very curl component the shifted tap feeds, at the very plane where the
#: ghost is served, so no metallic BOUNDARY VALUE is observable there. On an
#: all-periodic grid that mask does not fire and cell 0's curl is a live stepped value
#: assembled from the WRAPPED row.
PERIODIC_MUTATION_CASE = "bc_ppp"

#: THE PHASED MUTATION CASE. ``CNX``/``CNY``/``CNZ`` are constexprs: on an unphased
#: fixture the connection block is not compiled at all and a connection mutation is a
#: needle in dead text.
PHASED_MUTATION_CASE = "bloch_xyz_edge"

#: THE ARITHMETIC MUTATION CASE. Row 0 carries tiny normals of both signs beside exact
#: zeros: the class in which a respelled coefficient product rounds differently.
ARITHMETIC_MUTATION_CASE = "planted_row0_tiny"

#: The reference engines, in the order the record reports them. ``weld_seam_only`` is
#: the DISCRIMINATING arrangement: every other arrangement that dispatches ``update_E``
#: carries that sub-step's own agreement with the array path into the comparison.
REFERENCE_MODES: Tuple[str, ...] = ("array", "singles", "composition_today", "unfused",
                                    "weld", "weld_seam_only")

LEG_GROUPS: Dict[str, Tuple[str, ...]] = {
    # THE SPLIT IS THE PREDICATE'S, NOT A CONVENIENCE. `coverage._grid_reasons` clause 1
    # refuses any grid whose xp is not CuPy, so `refusal` and `arbitration` -- which
    # LAUNCH nothing -- still need a device to build a fixture the predicate can admit.
    "host": ("driver_order", "transcription", "purity_ledger", "ghost_observability"),
    "device": ("refusal", "arbitration", "product", "seed_scale", "launch_structure",
               "sync", "withdraw", "byte_neutral", "mutation", "disarm", "lift"),
}
ALL_LEGS: Tuple[str, ...] = tuple(leg for group in LEG_GROUPS.values() for leg in group)

#: The census this gate lifts its corpus rows from, and the seam record whose
#: ``withdraw_in_seam`` flag names the rows the predicate must refuse.
CENSUS = "predicate_coverage_triton_2026-09-04_cylm0"
SEAM_RECORD = "h_to_d_seam_2026-09-04"
CELL_ARMS: Tuple[str, str] = family.ARMS  # ("complex", "complex PML")

#: The runner's "cannot certify on this host" code.
EXIT_INCOMPLETE = kit.EXIT_INCOMPLETE

log = kit.log

# The generic machinery, imported rather than re-spelled. Each of these is complex64
# aware already -- `native_words` views a complex64 volume as uint32 words directly.
ABSORBED = kit.ABSORBED
Shim = kit.Shim
CountingKernel = kit.CountingKernel
count_kernel = kit.count_kernel
declaring = kit.declaring
drive = kit.drive
capture = kit.capture
restore = kit.restore
pin_array_path = kit.pin_array_path
install = kit.install
stored_volumes = kit.stored_volumes
words = kit.words
differing = kit.differing
compare_snapshots = kit.compare_snapshots
subnormal_census = kit.subnormal_census
_host = kit._host
_module_of = kit._module_of


# ---------------------------------------------------------------------------
# The synthetic fixture: a REAL driver, complex storage
# ---------------------------------------------------------------------------

#: The planted class: tiny NORMALS just above the float32 subnormal boundary, of both
#: signs, beside exact zeros of both signs. Written into row 0 of every stored volume.
#: MEASURED NECESSARY on the Dcyl round: with a random battery alone the naive multiply
#: arrangement moved 0 of 14,387,104 words, and this is the class it moves.
TINY_NORMAL = np.float32(1.4e-38)


def _plant_row0_tiny(fields: Any) -> Dict[str, int]:
    """Seed row 0 of every stored volume with the class the multiply defects show in."""
    planted: Dict[str, int] = {}
    pattern = np.array(
        [TINY_NORMAL, -TINY_NORMAL, np.float32(0.0), np.float32(-0.0),
         TINY_NORMAL * np.float32(2.0), -TINY_NORMAL * np.float32(3.0)],
        dtype=np.float32)
    for name, array in stored_volumes(fields).items():
        xp = _module_of(array)
        host = _host(array)
        if host.ndim != 3 or host.shape[0] < 1:
            continue
        row = host[0]
        flat = row.reshape(-1)
        if host.dtype == np.complex64:
            view = flat.view(np.float32)
        else:
            view = flat
        tiled = np.resize(pattern, view.size).astype(np.float32)
        view[...] = tiled
        host[0] = row
        array[...] = xp.asarray(host)
        planted[name] = int(view.size)
    return planted


def build_driver(keywords: Mapping[str, Any], seed: int,
                 sources: Sequence[Mapping[str, Any]] = (),
                 scale_bits: int = SEED_SCALE_BITS,
                 amplitude: float = 0.37,
                 prefer_gpu: bool = True) -> Any:
    """One seeded ``FdtdDriver`` on this gate's geometry and material, COMPLEX storage.

    A DRIVER, not a bare ``Fields``: every leg here is a claim about a COMPLETE driver
    step -- the withdraw loop, the injection slot, the fill consults, the wall clears
    and the sync channel are ``FdtdDriver.step``'s, and a hand-written walk over the
    live pass list would be a second model of what a step is.
    """
    from meep_gpu.driver import FdtdDriver  # noqa: PLC0415

    driver = FdtdDriver(cell_size=CELL, resolution=RESOLUTION, courant=COURANT,
                        force_complex_fields=True,
                        k_point=tuple(keywords.get("k_point") or (0.0, 0.0, 0.0)),
                        boundaries=dict(keywords.get("boundaries") or {}) or None,
                        dimensions=3, prefer_gpu=prefer_gpu, gpu_id=0)
    driver.setup_pml(int(keywords.get("pml", PML_CELLS)))
    shape = tuple(driver.grid.shape)
    xp = driver.xp
    driver.fields.set_epsilon_volumes(
        {name: xp.asarray(np.full(shape, np.float32(value), np.float32))
         for name, value in EPSILON.items()},
        {name: xp.asarray(np.full(shape, np.float32(1.0 / value), np.float32))
         for name, value in EPSILON.items()})
    for source in sources:
        driver.add_source(dict(source))
    rng = np.random.default_rng(seed)
    # THE CASE MAY OWN ITS OWN SCALE, and an explicit argument still wins: the
    # seed_scale leg drives one fixture at three scales and must not be overridden by
    # a case keyword it did not set.
    if "scale_bits" in keywords and scale_bits == SEED_SCALE_BITS:
        scale_bits = int(keywords["scale_bits"])
    if "amplitude" in keywords and amplitude == 0.37:
        amplitude = float(keywords["amplitude"])
    scale = np.float32(2.0) ** int(scale_bits)
    for array in stored_volumes(driver.fields).values():
        real = (float(amplitude)
                * rng.standard_normal(array.shape)).astype(np.float32) * scale
        if str(array.dtype) == "complex64":
            imaginary = (float(amplitude)
                         * rng.standard_normal(array.shape)).astype(np.float32) * scale
            host = (real + 1j * imaginary).astype(np.complex64)
        else:
            host = real.astype(np.float32)
        array[...] = _module_of(array).asarray(host)
    if keywords.get("plant") == "row0_tiny":
        driver._planted = _plant_row0_tiny(driver.fields)  # noqa: SLF001
    # A SEEDED STATE IS A MID-RUN ONE unless the case says otherwise: H was started
    # from B long ago, so no arrangement starts it again. The first-update case keeps
    # the flag False, and the real gate's capture/restore carry it, so every
    # arrangement's step 1 is the run's first update_H.
    driver.fields.h_separated = bool(keywords.get("h_started", True))
    driver.invalidate_fast_path()
    pin_array_path(driver)
    return driver


# ---------------------------------------------------------------------------
# The engines
# ---------------------------------------------------------------------------

def _default_kernel_for(owner: Any) -> Any:
    """The kernel a plan launches when its ``_kernel`` override is unset.

    THIS FILE'S OWN, not the real gate's: the independent launch counter must be able
    to resolve a complex plan, and a plan it cannot resolve is reported as UNCOUNTED
    rather than counted as zero.
    """
    from meep_gpu.triton_kernels import kernels as tkernels  # noqa: PLC0415

    name = type(owner).__name__
    if name == "ComplexFusedHdPairPlan":
        return family.fused_complex_constitutive_curl_H_to_D_kernel()
    if name == "ComplexPmlCurlPlan":
        return complex_fields.bloch_pml_curl_step
    if name == "ComplexConstitutivePlan":
        return complex_fields.bloch_constitutive_step
    # THE COMPOSER'S OWN COMPLEX FUSED PAIRS. On a complex row `plan_step(fuse=True)`
    # installs `fused pair B (complex)` and `fused pair D (complex)`, so the
    # composition reference carries two plan classes this file must resolve or the
    # independent kernel counter reports them UNCOUNTED and the two witnesses stop
    # being comparable -- MEASURED on the first device smoke of this gate, where
    # `launch_structure` failed on exactly that and nothing else.
    if name == "ComplexFusedMagneticPairPlan":
        from meep_gpu.triton_kernels import (  # noqa: PLC0415
            complex_fused_magnetic_pair as cfm,
        )

        return cfm.complex_fused_curl_constitutive_B_kernel()
    if name == "ComplexFusedElectricPairPlan":
        from meep_gpu.triton_kernels import (  # noqa: PLC0415
            complex_fused_electric_pair as cfe,
        )

        return cfe.complex_fused_curl_constitutive_D_kernel()
    if name == "PmlCurlPlan":
        return tkernels.pml_curl_step
    if name == "ConstitutivePlan":
        return tkernels.constitutive_step
    if name == "FusedPairPlan":
        return (tkernels.fused_curl_constitutive_B
                if getattr(owner, "pair", "B") == "B"
                else tkernels.fused_curl_constitutive_D)
    return None


class Arrangement(kit.Arrangement):
    """The real gate's arrangement with THIS backend's kernel resolver."""

    def count_launches(self) -> None:
        if self.shim is None:
            return
        wrapped: List[int] = []
        for slot, plan in self.shim.plans.items():
            if plan is ABSORBED:
                continue
            for entry in (plan if isinstance(plan, (list, tuple)) else [plan]):
                owner = declaring(entry)
                if id(owner) in wrapped:
                    continue
                wrapped.append(id(owner))
                counter = count_kernel(owner, _default_kernel_for(owner))
                if counter is not None:
                    self.counters.append(counter)
                else:
                    self.uncounted.append(f"{slot}:{type(owner).__name__}")


def _from_arrays_inputs(driver: Any) -> Dict[str, Any]:
    """The bare-array bindings the forced route needs, off the engine's own objects."""
    fields, pml, grid = driver.fields, driver.pml, driver.fields.grid
    arrays = {name: getattr(fields, name)
              for name in (family.ROTATED + family.IN_PLACE
                           + family.CONSTITUTIVE_SOURCES)}
    for name, twin in twin_table(fields, family.ROTATED).items():
        arrays["scratch_" + name] = twin
    kinds = stepping._boundary_kinds(grid, pml)  # noqa: SLF001
    return {
        "arrays": arrays,
        "curl_flat": {f"{stem}_{axis}": getattr(pml, f"{stem}_{axis}")
                      for axis in "xyz" for stem in ("kms", "sinv")},
        "constitutive_flat": {f"{stem}_{axis}": getattr(pml, f"{stem}_{axis}")
                              for axis in "xyz" for stem in ("kps", "kms")},
        "dtdx": grid.dt / grid.dx,
        "codes": [1 if kind == "metallic" else 0 for kind in kinds],
        "phases": complex_fields.bloch_phase_table(grid, kinds),
    }


def build_weld(driver: Any, kernel: Any = None, sources: Any = (),
               forced: bool = False) -> Any:
    """The product's plan, through the predicate where the policy allows it.

    ``forced`` bypasses the predicate and binds the weld from arrays. Which route a
    case took rides in the record.
    """
    if not forced:
        plan = family.plan_complex_fused_hd_pair(driver.fields, driver.pml,
                                                 sources=sources, kernel=kernel)
        if plan is not None:
            return plan
        verdict = family.complex_fused_hd_pair_coverage(driver.fields, driver.pml,
                                                        sources)
        if not all("standing integrated" in reason for reason in verdict.reasons):
            raise RuntimeError("the complex H/D pair was refused on a fixture this "
                               "gate expects it to admit: "
                               + "; ".join(verdict.reasons))
    # The same plan the engine route builds, so the same chunk record it binds.
    from gate_chunk_records import layer_record  # noqa: PLC0415

    inputs = _from_arrays_inputs(driver)
    plan = family.plan_complex_fused_hd_pair_from_arrays(
        inputs["arrays"], {**inputs["curl_flat"],
                           **{f"kps_{axis}": inputs["constitutive_flat"][f"kps_{axis}"]
                              for axis in "xyz"}},
        inputs["dtdx"], inputs["codes"], inputs["phases"], driver.fields, kernel=kernel,
        chunk_bounds=layer_record(driver.fields.grid, driver.pml))
    # The from-arrays route starts no H (a harness state has already started); this
    # gate's fixtures say whether theirs has, so the route takes the engine route's
    # start-of-H state and a first-update case starts H here as it does there.
    plan._h_state = driver.fields  # noqa: SLF001
    return plan


def build_singles(driver: Any, forced: bool = False) -> Tuple[Any, Any]:
    """The two certified complex halves, at the seam's two slots."""
    fields, pml = driver.fields, driver.pml
    if not forced:
        constitutive = complex_fields.plan_complex_constitutive(fields, pml, "H")
        curl = complex_fields.plan_complex_pml_curl(fields, pml, "step_D")
        if constitutive is not None and curl is not None:
            return constitutive, curl
    from gate_chunk_records import layer_record  # noqa: PLC0415

    inputs = _from_arrays_inputs(driver)
    record = layer_record(fields.grid, pml)
    side = tcoverage.CONSTITUTIVE_SIDES["H"]
    h_arrays = {name: getattr(fields, name)
                for name in (tuple(side["targets"]) + tuple(side["aux"])
                             + tuple(side["sources"]))}
    constitutive = complex_fields.plan_complex_constitutive_from_arrays(
        "H", h_arrays, inputs["constitutive_flat"], chunk_bounds=record)
    # As for the weld: the engine route's start-of-H state (the plan's own slot).
    constitutive._h_state = fields  # noqa: SLF001
    spec = triton_launch.SUB_STEPS["step_D"]
    d_arrays = {name: getattr(fields, name)
                for name in tuple(spec["targets"]) + tuple(spec["sources"])}
    d_arrays.update({"fu_" + name: getattr(fields, "fu_" + name)
                     for name in spec["targets"]})
    curl = complex_fields.plan_complex_pml_curl_from_arrays(
        "step_D", d_arrays, inputs["curl_flat"], inputs["codes"], inputs["phases"],
        inputs["dtdx"], chunk_bounds=record)
    return constitutive, curl


def arrangement_singles(driver: Any, forced: bool = False) -> Arrangement:
    """Reference 2: the two certified complex singles dispatched at the seam's slots."""
    constitutive, curl = build_singles(driver, forced=forced)
    if constitutive is None or curl is None:
        raise RuntimeError("a certified complex single was refused on a fixture this "
                           "gate expects it to admit")
    shim = Shim(driver.fields, {"update_H": constitutive, "step_D": curl})
    return Arrangement("singles", shim,
                       selected={"update_H": CELL_ARMS[0], "step_D": CELL_ARMS[1]})


def _composed(driver: Any, fuse: bool) -> Any:
    """``plan_step`` on this driver -- the SHIPPED composer builds the plans."""
    return triton_launch.plan_step(
        driver.fields, driver.pml,
        sources=tuple(getattr(driver, "_sources", ())), fuse=fuse)


def arrangement_composition(driver: Any, fuse: bool) -> Optional[Arrangement]:
    """References 3 and 4, or ``None`` where the composer reaches this seam with nothing.

    NONE IS A MEASUREMENT, not a fallback: under ``flush`` the complex arms are refused
    by the certification clause, so the composer puts the array path on both of this
    seam's slots and there is no composition to compare against. A reference that
    dispatched nothing would compare the array path with itself and report a green zero.
    """
    plan = _composed(driver, fuse)
    plans = {slot: plan.plans[slot] for slot in triton_launch.STEP_ORDER
             if slot in plan.plans}
    if plan.polarization_plans:
        plans["update_P"] = list(plan.polarization_plans)
    if not any(slot in plans for slot in family.REPLACES):
        return None
    return Arrangement("composition_today" if fuse else "unfused",
                       Shim(driver.fields, plans),
                       selected=plan.selected,
                       reasons={key: list(value)
                                for key, value in plan.reasons.items()
                                if "fused" in key or "complex" in key})


def arrangement_weld(driver: Any, kernel: Any = None,
                     launcher: Optional[Callable[[Any], Any]] = None,
                     sync_hazard: bool = False, hoist: bool = False,
                     rest_unfused: bool = True, name: str = "weld",
                     forced: bool = False) -> Arrangement:
    """The SUBJECT: the weld force-installed at ``update_H``, ``step_D`` absorbed.

    FORCE-INSTALLED, because the composer refuses this product (``INSTALLABLE`` False,
    and no ``CERTIFIED_FUSED_PRODUCTS`` row on a backend whose label table lives in a
    file this round does not own) and this gate must measure it anyway.
    """
    sources = tuple(getattr(driver, "_sources", ()))
    plan = build_weld(driver, kernel=kernel, sources=(), forced=forced)
    occupant: Any = plan if launcher is None else launcher(plan)
    if hoist:
        occupant = withdraw_hoist.LeadingWithdrawPlan(
            occupant, driver.fields, sources, span=family.REPLACES)
    plans: Dict[str, Any] = {}
    if rest_unfused:
        base = _composed(driver, fuse=False)
        for slot in triton_launch.STEP_ORDER:
            if slot in base.plans and slot not in family.REPLACES:
                plans[slot] = base.plans[slot]
        if base.polarization_plans:
            plans["update_P"] = list(base.polarization_plans)
    plans["update_H"] = occupant
    plans["step_D"] = ABSORBED
    shim = Shim(driver.fields, plans, sync_hazard=sync_hazard)
    return Arrangement(name, shim,
                       selected={"update_H": f"{family.FAMILY} (forced)",
                                 "step_D": f"{family.FAMILY} (forced)"})


def all_arrangements(driver: Any, kernel: Any = None,
                     launcher: Optional[Callable[[Any], Any]] = None,
                     count: bool = False,
                     forced: bool = False) -> Tuple[Dict[str, Arrangement], List[str]]:
    """The engines, built on one driver, and the references this policy cannot build."""
    built: Dict[str, Arrangement] = {
        "array": Arrangement("array", None),
        "singles": arrangement_singles(driver, forced=forced),
    }
    absent: List[str] = []
    for label, fuse in (("composition_today", True), ("unfused", False)):
        composed = arrangement_composition(driver, fuse=fuse)
        if composed is None:
            absent.append(label)
        else:
            built[label] = composed
    built["weld"] = arrangement_weld(driver, kernel=kernel, launcher=launcher,
                                     rest_unfused=not absent, forced=forced)
    # THE SEAM ALONE: this weld at its two slots, every other slot on the ARRAY PATH.
    # It is what isolates a finding about the seam from a finding about a neighbouring
    # sub-step's own agreement with the array path.
    built["weld_seam_only"] = arrangement_weld(
        driver, kernel=kernel, launcher=launcher, rest_unfused=False,
        name="weld_seam_only", forced=forced)
    if count:
        for arrangement in built.values():
            arrangement.count_launches()
    return built, absent


def run_product(driver: Any, steps: int,
                progress: Optional[Callable[[str], None]] = None,
                require_full_budget: bool = True,
                clean_floor: int = 1,
                forced: bool = False) -> Dict[str, Any]:
    """The multi-reference identity on ONE driver. The core measurement."""
    started = time.time()
    arrangements, absent = all_arrangements(driver, count=True, forced=forced)
    pairs = [("weld_seam_only", "singles")]
    if "unfused" in arrangements:
        pairs.append(("weld", "unfused"))
    if "composition_today" in arrangements:
        pairs.append(("weld", "composition_today"))
    result = drive(driver, arrangements, steps, progress=progress,
                   stop_when_banded=False, stop_on_divergence=False,
                   pairs=tuple(pairs))
    references = [name for name in arrangements if name != "array"]

    def agrees(name: str) -> bool:
        return all(not row.get(name, {}).get("differing_words")
                   for row in result["per_step"])

    per_reference = {name: agrees(name) for name in references}
    disagree_with_array = sorted(name for name in references if not per_reference[name])

    def pair_agrees(key: str) -> bool:
        return all(not row.get("pairs", {}).get(key, {}).get("differing_words")
                   for row in result["per_step"])

    pair_agreement = {f"{a}|{b}": pair_agrees(f"{a}|{b}") for a, b in pairs}
    # THE SEAM CLAIM, stated as the conjunction it is. The first clause is the sharpest
    # statement of this product's own claim: ONE launch produces, word for word, what
    # the TWO certified complex launches it replaces produce, on the same driver, the
    # same seed and the same complete steps.
    seam_claim = {
        "the_weld_at_its_seam_equals_the_two_certified_singles": pair_agreement[
            "weld_seam_only|singles"],
    }
    if "weld|unfused" in pair_agreement:
        seam_claim["the_weld_equals_the_unfused_composition"] = pair_agreement[
            "weld|unfused"]
    if "weld|composition_today" in pair_agreement:
        seam_claim["the_weld_equals_the_composition_installed_today"] = pair_agreement[
            "weld|composition_today"]
    result.update({
        "passed": bool(all(seam_claim.values())
                       and result["reference_moved_words_step_1"] > 0
                       and result["steps_compared"] >= clean_floor
                       and (not require_full_budget
                            or result["steps_compared"] == steps)
                       and result["bit_identical"]),
        "seam_claim": seam_claim,
        "arrangements": sorted(arrangements),
        "references_this_policy_cannot_build": absent,
        "composer_reachable_under_this_policy": not absent,
        "built_from_arrays": bool(forced),
        "agreement_with_the_array_path": per_reference,
        "pairwise_agreement": pair_agreement,
        "references_that_disagree_with_the_array_path": disagree_with_array,
        "the_seam_alone_agrees_with_the_array_path": per_reference.get(
            "weld_seam_only"),
        "weld_agrees_with_the_certified_singles": bool(
            per_reference.get("weld_seam_only")
            and pair_agreement["weld_seam_only|singles"]),
        "the_certified_singles_agree_with_the_array_path": bool(
            per_reference.get("singles")),
        "what_a_shared_divergence_means": (
            "a dispatched arrangement that disagrees with the CuPy array path is named "
            "with its autopsy in references_that_disagree_with_the_array_path and the "
            "row FAILS. The pairwise comparisons stay in the record so a failure can be "
            "ATTRIBUTED -- to this seam (weld_seam_only differs from singles) or to a "
            "sub-step this product does not own (weld equals unfused and both differ "
            "from the array path) -- but attribution is not a pass"),
        "what_absent_references_mean": (
            "where the composer selects nothing complex on this seam, composition_today "
            "and unfused would compare the ARRAY PATH WITH ITSELF and report a green "
            "zero. They are named as absent and the seam claim rests on the certified "
            "singles, built from arrays"),
        "seconds": round(time.time() - started, 2),
    })
    return result


# ---------------------------------------------------------------------------
# HOST LEG: the driver's own order
# ---------------------------------------------------------------------------

def leg_driver_order() -> Dict[str, Any]:
    """``REPLACES`` is the driver's two adjacent consults, and nothing else is between."""
    findings: List[str] = []
    text = (API_ROOT / "meep_gpu" / "driver.py").read_text(encoding="utf-8")
    body = text.split("    def step(self", 1)[1].split("\n    def ", 1)[0]
    lines = [line.strip() for line in body.splitlines()]
    consults = [line for line in lines if 'fast.dispatch("' in line]
    order = [line.split('fast.dispatch("', 1)[1].split('"', 1)[0] for line in consults]
    try:
        first = order.index("update_H")
    except ValueError:
        findings.append("driver.step no longer consults update_H")
        first = -1
    if first >= 0 and order[first + 1:first + 2] != ["step_D"]:
        findings.append(f"the consult after update_H is {order[first + 1:first + 2]}, "
                        f"not ['step_D']")
    between = body.split('fast.dispatch("update_H"', 1)[1]
    between = between.split('fast.dispatch("step_D"', 1)[0]
    statements = [line.strip() for line in between.splitlines()
                  if line.strip() and not line.strip().startswith("#")]
    statements = [line for line in statements if "update_H(self.fields" not in line]
    # The magnetic f_minus_p restore, which writes B alone and which step_D does not
    # read (withdraw_hoist.SEAM_PASS); an integrated source stands no withdraw here.
    # statements[0] and [-1] are the tails of the two consult lines themselves.
    if tuple(statements[1:-1]) != h_to_d_seam.EXPECTED_BETWEEN[1:]:
        findings.append(f"between the two consults stands {statements[1:-1]}, not the magnetic "
                        f"f_minus_p restore alone")
    if any("withdraw" in line for line in statements):
        findings.append("a withdraw stands between the two consults")
    if any("inject" in line for line in statements):
        findings.append("an injection stands between the two consults; "
                        "CARRIES_DEPOSIT_REPAIR = False is then false about the driver")
    if family.REPLACES != withdraw_hoist.SEAM_SPAN:
        findings.append(f"REPLACES {family.REPLACES} is not withdraw_hoist.SEAM_SPAN")
    if family.SEAM != h_to_d_seam.SEAM:
        findings.append(f"the product's SEAM {family.SEAM!r} is not the boards' "
                        f"{h_to_d_seam.SEAM!r}")
    if "step_D" in SYNC_PATH_SLOTS:
        findings.append("step_D is inside SYNC_PATH_SLOTS; the containment rule would "
                        "then admit this weld into the magnetic half-step")
    if SYNC_PASS_OWNERS.get(SYNC_UPDATE_H_PASS) != "update_H":
        findings.append("the sync channel no longer owns update_H by name")
    if family.CARRIES_DEPOSIT_REPAIR:
        findings.append("CARRIES_DEPOSIT_REPAIR is True and nothing is injected here")
    if family.HOISTS_THE_WITHDRAW:
        findings.append("HOISTS_THE_WITHDRAW is True while INSTALLABLE is False")
    if family.INSTALLABLE:
        findings.append("INSTALLABLE is True; this round's verdict is that it must not")
    return {"passed": not findings, "findings": findings,
            "consult_order": order,
            "statements_between_the_two_consults": statements,
            "replaces": list(family.REPLACES),
            "sync_path_slots": list(SYNC_PATH_SLOTS),
            "installable": family.INSTALLABLE,
            "installable_reason": family.INSTALLABLE_REASON}


# ---------------------------------------------------------------------------
# HOST LEG: the transcription
# ---------------------------------------------------------------------------

def leg_transcription() -> Dict[str, Any]:
    """Both halves are the certified COMPLEX kernels' own text, plus the declared edits."""
    findings: List[str] = []
    measures: Dict[str, Any] = {}
    certified_constitutive = family.certified_constitutive_tail()
    lifted_constitutive = family.lifted_constitutive_tail()
    measures["constitutive_chars"] = len(certified_constitutive)
    if certified_constitutive != lifted_constitutive:
        findings.append(
            f"the weld's _h_cell_complex is NOT bloch_constitutive_step's own body "
            f"with the declared lift edits: {len(certified_constitutive)} chars "
            f"against {len(lifted_constitutive)}")
    certified_curl = family.certified_curl_tail()
    lifted_curl = family.lifted_curl_tail()
    measures["curl_chars"] = len(certified_curl)
    if certified_curl != lifted_curl:
        findings.append(
            f"the weld's curl half is NOT bloch_pml_curl_step's own body with the "
            f"twelve redirected magnetic word loads: {len(certified_curl)} chars "
            f"against {len(lifted_curl)}")
    # NON-VACUITY: an anchor pair that crossed would compare two short strings.
    for label, text in (("constitutive", certified_constitutive),
                        ("curl", certified_curl)):
        statements = [line for line in text.splitlines()
                      if line.strip() and not line.strip().startswith("#")]
        measures[f"{label}_statements"] = len(statements)
        if len(statements) < 30:
            findings.append(f"the lifted {label} body is only {len(statements)} "
                            f"statements; the anchors have crossed")
    raw_curl = family._real._cut(  # noqa: SLF001
        family.function_source("bloch_pml_curl_step"), family.DECODE_END)
    taps = family.halo_taps(raw_curl)
    offsets = family.offset_coordinates(raw_curl)
    measures["halo_taps"] = {name: {plane: list(value)
                                    for plane, value in planes.items()}
                             for name, planes in sorted(taps.items())}
    measures["offset_coordinates"] = {name: list(value)
                                      for name, value in sorted(offsets.items())}
    measures["curl_lift_edits"] = len(family.curl_lift_edits())
    if measures["curl_lift_edits"] != 12:
        findings.append(f"the curl half takes {measures['curl_lift_edits']} edits, not "
                        f"the six own-cell word loads plus six shifted pairs")
    measures["constitutive_lift_edits"] = len(family.CONSTITUTIVE_LIFT_EDITS)
    for edit in family.CONSTITUTIVE_LIFT_EDITS:
        if not edit.get("why"):
            findings.append(f"the lift edit {edit.get('line')!r} carries no reason")
    for stem in ("g0", "g1", "g2", "e0", "e1", "e2"):
        if f"{stem} +" in lifted_curl or f"{stem} +" in lifted_constitutive:
            findings.append(f"{stem} survives in the welded text; in this signature "
                            f"that pointer does not exist")
    # THE ARITHMETIC FACTS, as properties of the emitted text.
    measures["divide_free_evidence"] = family.divide_free_evidence()
    if not measures["divide_free_evidence"]["divide_free"]:
        findings.append(
            "the welded text DIVIDES; every PML reciprocal is precomputed on the host "
            "and a division here is either a wrong spelling or a fact this gate's "
            "DIVIDE_FACTS no longer describe")
    helpers = _multiply_helpers_called()
    measures["multiply_helpers_called"] = sorted(helpers)
    if helpers != set(family.MULTIPLY_HELPERS) or helpers != {"_connect"}:
        findings.append(
            f"the welded text reaches {sorted(helpers)}, not the declared "
            f"{sorted(family.MULTIPLY_HELPERS)} (the connection alone); every other "
            f"product on complex storage is formed part by part, and a helper the "
            f"per-part contract does not name is a product nothing measured")
    # ASCII EVERYWHERE THIS FILE WROTE, and the exception is the lift itself.
    # `complex_fields.py`'s certified bodies carry two em-dashes inside COMMENTS the
    # lift reproduces character for character, so a blanket `isascii()` on the family
    # module fails on text this product is required not to touch. The clause is
    # therefore the precise one: every non-ASCII LINE in the module must be a line of
    # the lifted certified text, and the module's own prose must be ASCII so a
    # record's digest is over what a reader sees.
    source = (API_ROOT / "meep_gpu" / "triton_kernels"
              / "complex_fused_hd_pair.py").read_text(encoding="utf-8")
    lifted_lines = {line.strip() for line in
                    (lifted_curl + lifted_constitutive).splitlines() if line.strip()}
    outside = sorted({line.strip() for line in source.splitlines()
                      if not line.isascii() and line.strip() not in lifted_lines})
    measures["non_ascii_lines_in_the_module"] = sorted(
        {line.strip() for line in source.splitlines() if not line.isascii()})
    measures["non_ascii_lines_outside_the_lift"] = outside
    if outside:
        findings.append(
            f"the family module carries non-ASCII text this product WROTE rather than "
            f"lifted: {outside}. A record's digest is over what a reader sees")
    # ...and the lifted non-ASCII lines are the CERTIFIED module's own, so a lift that
    # invented one would fail here rather than pass as "inherited".
    certified_source = (API_ROOT / "meep_gpu" / "triton_kernels"
                        / "complex_fields.py").read_text(encoding="utf-8")
    certified_lines = {line.strip() for line in certified_source.splitlines()}
    invented = sorted(line for line in measures["non_ascii_lines_in_the_module"]
                      if line not in certified_lines)
    measures["non_ascii_lines_the_certified_module_does_not_carry"] = invented
    if invented:
        findings.append(f"non-ASCII lines this module claims to have lifted are not in "
                        f"complex_fields.py: {invented}")
    # EVERY MUTATION NEEDLE RESOLVES EXACTLY ONCE, checked before any device time.
    unresolved = []
    for tag, spec in MUTATIONS.items():
        if spec["target"] != "kernel":
            continue
        hits = source.count(spec["old"])
        if hits != 1:
            unresolved.append(f"{tag}: {hits} matches")
    hits = source.count(BYTE_NEUTRAL["old"])
    if hits != 1:
        unresolved.append(f"{BYTE_NEUTRAL['tag']}: {hits} matches")
    if unresolved:
        findings.append("mutation needles that do not resolve exactly once: "
                        + "; ".join(unresolved))
    measures["mutation_needles"] = len([1 for s in MUTATIONS.values()
                                        if s["target"] == "kernel"])
    try:
        measures["sub_lattice_suffixes"] = list(
            family._sub_lattice_suffixes())  # noqa: SLF001
    except AssertionError as error:
        findings.append(str(error))
    return {"passed": not findings, "findings": findings, "measures": measures}


def _multiply_helpers_called() -> set:
    """Which complex-product helpers the shipped device text reaches. Parsed, not read."""
    import ast  # noqa: PLC0415

    called = set()
    for name in family.DEVICE_FUNCTIONS:
        tree = ast.parse(family.function_source(name, Path(family.__file__)))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            leaf = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            if leaf and (leaf.startswith("_mul_") or leaf.startswith("_rotate_")
                         or leaf.startswith("_div_") or leaf == "_connect"):
                called.add(leaf)
    return called


# ---------------------------------------------------------------------------
# HOST/DEVICE LEG: the refusals, by name
# ---------------------------------------------------------------------------

class _GridView:
    """The REAL grid with one attribute overridden, for a refusal stand-in."""

    def __init__(self, grid: Any, **overrides: Any) -> None:
        object.__setattr__(self, "_grid", grid)
        object.__setattr__(self, "_overrides", dict(overrides))

    def __getattr__(self, name: str) -> Any:
        overrides = object.__getattribute__(self, "_overrides")
        if name in overrides:
            return overrides[name]
        return getattr(object.__getattribute__(self, "_grid"), name)


class _FieldsView:
    """A ``Fields`` with one attribute overridden, for a refusal stand-in."""

    def __init__(self, fields: Any, grid: Any) -> None:
        self._fields = fields
        self.grid = grid

    def __getattr__(self, name: str) -> Any:
        return getattr(self._fields, name)


class _RealStorageView:
    """A ``Fields`` reporting float32 volumes -- the storage this product does not carry."""

    def __init__(self, fields: Any) -> None:
        self._fields = fields
        module = _module_of(fields.Hx)
        self._real = {name: module.zeros(fields.Hx.shape, dtype=module.float32)
                      for name in (family.ROTATED + family.IN_PLACE
                                   + family.CONSTITUTIVE_SOURCES)}

    def __getattr__(self, name: str) -> Any:
        if name in self._real:
            return self._real[name]
        return getattr(self._fields, name)


def _source(component: str = "Ez", integrated: bool = True,
            field_type: str = "D", points: int = 3) -> Any:
    """A source class carrying an integrated withdraw of its own, which the predicate's
    withdraw clause refuses. No engine source carries one since an integrated source
    entered through ``f_minus_p`` (``withdraw_hoist``)."""

    class _Stand:
        def withdraw(self, fields):  # never called: a predicate reads a source
            raise AssertionError("a coverage predicate called a source's withdraw")

    stand = _Stand()
    stand.component = component
    stand.field_type = field_type
    stand.is_integrated = integrated
    stand._n_source_points = points
    stand._applied_dipole = 0j
    return stand


def leg_refusal() -> Dict[str, Any]:
    """Every configuration outside the cell is refused BY NAME, with its reason."""
    findings: List[str] = []
    rows: List[Dict[str, Any]] = []

    def check(name: str, fields: Any, pml: Any, sources: Any, must_refuse: bool,
              needle: Optional[str] = None) -> None:
        verdict = family.complex_fused_hd_pair_coverage(fields, pml, sources)
        reasons = list(verdict.reasons)
        rows.append({"case": name, "covered": verdict.covered, "reasons": reasons})
        if must_refuse and verdict.covered:
            findings.append(f"{name}: admitted, and must be refused")
        if not must_refuse and not verdict.covered:
            findings.append(f"{name}: refused, and must be admitted ({reasons})")
        if needle is not None and not any(needle in reason for reason in reasons):
            findings.append(f"{name}: no refusal names {needle!r}; got {reasons}")

    driver = build_driver(dict(CASES[0][1]), seed=1)
    fields, pml = driver.fields, driver.pml
    # 1. The source set is not declared: ignorance is never an empty set.
    check("undeclared_source_list", fields, pml, None, True, "was not declared")
    # 2. No sources at all: admitted.
    check("no_sources", fields, pml, (), False)
    # 3. A standing INTEGRATED electric withdraw.
    check("integrated_electric_withdraw", fields, pml, (_source(),), True,
          "standing integrated")
    # 4. A NON-integrated electric source is NOT this seam's business.
    check("non_integrated_electric_source", fields, pml,
          (_source(integrated=False),), False)
    # 5. A MAGNETIC source is withdrawn one seam earlier.
    check("integrated_magnetic_source", fields, pml,
          (_source(component="Hz", field_type="B"),), False)
    # 6. A folded grid names the folded-complex cells this product does not widen into.
    folded = _FieldsView(fields, _GridView(fields.grid,
                                           is_mirrored=lambda axis: axis == 1))
    check("folded_grid", folded, pml, (), True, "`folded complex` arms")
    # 7. A cylindrical grid.
    cylindrical = _FieldsView(fields, _GridView(fields.grid, cylindrical=True))
    check("cylindrical_grid", cylindrical, pml, (), True, "cylindrical")
    # 8. REAL storage -- the sibling product's cell, refused by both complex halves.
    check("real_storage", _RealStorageView(fields), pml, (), True, "complex")
    # 9. An inactive absorber, named by the CONSTITUTIVE half -- the driver's order.
    class _Inactive:
        is_active = False

    verdict = family.complex_fused_hd_pair_coverage(fields, _Inactive(), ())
    rows.append({"case": "inactive_absorber", "covered": verdict.covered,
                 "reasons": list(verdict.reasons)})
    if verdict.covered:
        findings.append("inactive_absorber: admitted, and must be refused")
    elif not verdict.reasons[0].startswith("complex constitutive half"):
        findings.append(
            f"inactive_absorber: the FIRST refusal is {verdict.reasons[0]!r}; the "
            f"driver reaches update_H first, so the constitutive half must speak first")
    driver.close()
    return {"passed": not findings, "findings": findings, "cases": rows}


# ---------------------------------------------------------------------------
# DEVICE LEG: the arbitration
# ---------------------------------------------------------------------------

def _inject_product_row() -> Callable[[], None]:
    """Put this product's row into the composer's tables IN PROCESS, and undo it."""
    products = triton_launch.CERTIFIED_FUSED_PRODUCTS
    arms = triton_launch.CERTIFIED_FUSED_PAIR_ARMS
    name = family.FAMILY
    products[name] = {"curl_slot": "update_H", "module": "complex_fused_hd_pair",
                      "coverage": "complex_fused_hd_pair_coverage",
                      "builder": "plan_complex_fused_hd_pair",
                      "label": "complex fused pair H->D"}
    arms[name] = family.ARMS
    original_modules = triton_launch._certified_fused_product_modules  # noqa: SLF001

    def modules():
        table = original_modules()
        table["complex_fused_hd_pair"] = family
        return table

    triton_launch._certified_fused_product_modules = modules  # noqa: SLF001

    def undo() -> None:
        products.pop(name, None)
        arms.pop(name, None)
        triton_launch._certified_fused_product_modules = original_modules  # noqa: SLF001

    return undo


def leg_arbitration() -> Dict[str, Any]:
    """Who holds ``update_H`` on a row this product's predicate admits.

    WRITTEN FOR THE UNWIRED TREE, and the positional facts are RECORDED rather than
    asserted: which arm the composer puts on ``update_H`` today, and whether this
    product's row appears in any table, are exactly what wiring inverts. What is
    ASSERTED is substantive and survives wiring: the product is refused BY NAME, the
    refusal names ``INSTALLABLE = False``, the shipped selection is UNCHANGED by the
    injection, and the released neighbours keep their slots.
    """
    findings: List[str] = []
    driver = build_driver(dict(CASES[0][1]), seed=3)
    fields, pml = driver.fields, driver.pml
    admitted = family.complex_fused_hd_pair_coverage(fields, pml, ()).covered

    shipped = triton_launch.plan_step(fields, pml, sources=(), fuse=True)
    shipped_selected = dict(shipped.selected)
    holder = shipped_selected.get("update_H")
    already_carried = family.FAMILY in triton_launch.CERTIFIED_FUSED_PRODUCTS

    undo = _inject_product_row()
    try:
        injected = triton_launch.plan_step(fields, pml, sources=(), fuse=True)
        injected_selected = dict(injected.selected)
        refusals = [reason for key, value in injected.reasons.items()
                    if family.FAMILY in key for reason in value]
        absorb_refusal = triton_launch._pair_may_absorb(  # noqa: SLF001
            injected_selected, "update_H", "step_D", family.ARMS)
        uninstallable = triton_launch._declared_uninstallable(  # noqa: SLF001
            {"complex_fused_hd_pair": family}, family.FAMILY,
            triton_launch.CERTIFIED_FUSED_PRODUCTS[family.FAMILY])
    finally:
        undo()

    # ---- ASSERTED: the substantive claims, which wiring does not invert -------------
    if not uninstallable:
        findings.append("_declared_uninstallable does not refuse this product BY NAME")
    elif "INSTALLABLE = False" not in uninstallable:
        findings.append(f"the flag refusal does not name the flag: {uninstallable!r}")
    if injected_selected != shipped_selected:
        moved = {slot: (shipped_selected.get(slot), injected_selected.get(slot))
                 for slot in set(shipped_selected) | set(injected_selected)
                 if shipped_selected.get(slot) != injected_selected.get(slot)}
        findings.append(
            f"offering this product to the composer CHANGED the selection: {moved}. "
            f"An uninstallable product must leave every slot exactly where it was, "
            f"including the released neighbours'")
    if not refusals:
        findings.append("the composer records no refusal naming this product")
    # THE SEAM ROW EXISTS AND IS LAST, and its pair name is the withdraw hoist's.
    seams = triton_launch.CERTIFIED_FUSED_PAIR_SEAMS
    if seams.get("update_H") != ("step_D", withdraw_hoist.SEAM):
        findings.append(f"the composer's H->D seam row is {seams.get('update_H')!r}")
    if list(seams)[-1] != "update_H":
        findings.append("the H->D seam row is not the last row of the seam table")
    driver.close()
    # ---- RECORDED: the positional facts wiring inverts ------------------------------
    return {"passed": not findings, "findings": findings,
            "predicate_admits_the_fixture": admitted,
            "shipped_selected": shipped_selected,
            "selected_with_the_row_injected": injected_selected,
            "the_selection_is_unchanged_by_the_offer": (
                injected_selected == shipped_selected),
            "update_H_is_held_by": holder,
            "a_neighbouring_pair_holds_update_H_today": bool(
                holder and holder == shipped_selected.get("step_B")
                and "fused pair" in str(holder)),
            # RECORDED, NOT ASSERTED, and deliberately: whether the shipped tables
            # already carry this product is exactly what a wiring patch inverts. On
            # the UNWIRED tree the injection is a counterfactual; on a wired one it
            # rewrites the row the tree already has and the leg still measures the
            # same substantive claims, because a routed product's refusal is what the
            # composer records for it either way. A gate that failed here would fail
            # the day the product is routed, which is the day it should not.
            "the_shipped_tables_already_carry_this_product": already_carried,
            "what_the_injection_is": (
                "a COUNTERFACTUAL while this product is unrouted -- the composer is "
                "offered a product it does not carry -- and a REWRITE of its own row "
                "once a wiring patch routes it. Either way the offer must move no "
                "slot and the refusal must name the flag, which is what this leg "
                "asserts" if not already_carried else
                "a REWRITE of the row the shipped tables already carry: this product "
                "is now routed, so the offer is no longer a counterfactual and the "
                "leg measures the routed composer's own refusal"),
            "composer_refusals": refusals,
            "pair_may_absorb_refusal": absorb_refusal,
            "declared_uninstallable_refusal": uninstallable,
            "what_is_recorded_rather_than_asserted": (
                "WHO holds update_H today, and whether _pair_may_absorb refuses this "
                "product, are POSITIONAL facts about an unwired tree: on a complex row "
                "the released B->H pair may or may not install, and wiring inverts "
                "exactly those. What this leg ASSERTS "
                "is substantive: the product is refused by name, the refusal names the "
                "flag, and the offer moves NO slot -- the released neighbours keep "
                "theirs")}


# ---------------------------------------------------------------------------
# HOST LEG: the purity / race ledger
# ---------------------------------------------------------------------------

def purity_ledger(fields: Any, pml: Any) -> Dict[str, Any]:
    """How many of the curl's valid foreign taps land on a cell complex ``update_H`` MOVED.

    THIS IS THE LEDGER THIS SHAPE UNIQUELY OWES. The weld's whole claim is that a
    foreign tap RECOMPUTES a value an in-place arrangement would have raced on. If
    ``update_H`` moved nothing at the cells the curl taps, the recompute and the stale
    read would agree by accident and the identity result would license nothing about
    the hazard. Counted on the ARRAY PATH, over BOTH word planes.
    """
    before = {name: _host(getattr(fields, name)).copy() for name in family.ROTATED}
    stepping.update_H(fields, pml)
    after = {name: _host(getattr(fields, name)).copy() for name in family.ROTATED}
    shape = tuple(int(n) for n in fields.grid.shape)
    moved = {}
    for name in family.ROTATED:
        # A COMPLEX CELL COUNTS AS MOVED IF EITHER WORD PLANE MOVED, which is what a
        # foreign tap reads: the pair, not a word.
        left, right = words(before[name]), words(after[name])
        per_word = (left != right)
        if before[name].dtype == np.complex64:
            per_word = per_word.reshape(-1, 2).any(axis=1)
        moved[name] = per_word.reshape(shape)
    nx, ny, nz = shape
    kinds = tcoverage._boundary_kinds(fields.grid, pml)  # noqa: SLF001
    metallic = tuple(kind == "metallic" for kind in kinds)
    index = np.indices((nx, ny, nz))
    i, j, k = index[0], index[1], index[2]
    shifted = {"x": i - 1, "y": j - 1, "z": k - 1}
    extents = {"x": nx, "y": ny, "z": nz}
    valid = {}
    for axis, walled in zip("xyz", metallic):
        s = shifted[axis]
        if walled:
            valid[axis] = (s >= 0) & (s < extents[axis])
            shifted[axis] = np.clip(s, 0, extents[axis] - 1)
        else:
            valid[axis] = np.ones_like(s, dtype=bool)
            shifted[axis] = np.where(s < 0, extents[axis] - 1, s)
    taps = {
        "a_y": ("Hx", (i, shifted["y"], k), valid["y"]),
        "a_z": ("Hx", (i, j, shifted["z"]), valid["z"]),
        "b_x": ("Hy", (shifted["x"], j, k), valid["x"]),
        "b_z": ("Hy", (i, j, shifted["z"]), valid["z"]),
        "c_x": ("Hz", (shifted["x"], j, k), valid["x"]),
        "c_y": ("Hz", (i, shifted["y"], k), valid["y"]),
    }
    per_tap: Dict[str, Dict[str, int]] = {}
    total_valid = 0
    total_moved = 0
    for name, (volume, coordinates, guard) in taps.items():
        landed = moved[volume][coordinates]
        n_valid = int(np.count_nonzero(guard))
        n_moved = int(np.count_nonzero(landed & guard))
        per_tap[name] = {"volume": volume, "valid_taps": n_valid,
                         "taps_on_a_moved_cell": n_moved}
        total_valid += n_valid
        total_moved += n_moved
    history_moved = int(sum(int(np.count_nonzero(moved[f"f_w_H{axis}"]))
                            for axis in "xyz"))
    return {
        "cells": nx * ny * nz,
        "per_tap": per_tap,
        "valid_foreign_taps": total_valid,
        "foreign_taps_on_a_cell_update_H_moved": total_moved,
        "fraction_of_valid_taps_that_would_have_raced": (
            round(total_moved / total_valid, 6) if total_valid else 0.0),
        "split_field_history_cells_moved": history_moved,
        "what_this_licenses": (
            "the count of foreign taps whose RECOMPUTED value differs from what an "
            "in-place arrangement would have read at the moment the neighbour had not "
            "yet been written. A near-zero count would mean the byte-identity result "
            "says nothing about the hazard. It does NOT measure a race: nothing here "
            "launches, and the in-place arrangement's actual read order is "
            "schedule-dependent and is not reproduced"),
    }


def leg_purity_ledger() -> Dict[str, Any]:
    """The ledger on every fixture, with its non-vacuity bar."""
    findings: List[str] = []
    rows: Dict[str, Any] = {}
    for name, keywords in CASES:
        driver = build_driver(dict(keywords), seed=11, prefer_gpu=False)
        rows[name] = purity_ledger(driver.fields, driver.pml)
        driver.close()
        ledger = rows[name]
        if ledger["foreign_taps_on_a_cell_update_H_moved"] == 0:
            findings.append(
                f"{name}: NO foreign tap lands on a cell update_H moved, so the "
                f"identity result licenses nothing about the read-after-write hazard "
                f"on this fixture")
        if ledger["split_field_history_cells_moved"] == 0:
            findings.append(
                f"{name}: update_H moves no f_w_H cell, so the second premise of the "
                f"hazard (a racing neighbour reading B where it needs B_prev) is not "
                f"live here")
    return {"passed": not findings, "findings": findings, "fixtures": rows}


# ---------------------------------------------------------------------------
# HOST LEG: is the metallic ghost observable at all?
# ---------------------------------------------------------------------------

CURL_EXPRESSIONS = ("curl0_re", "curl1_re", "curl2_re")


def ghost_observability() -> Dict[str, Any]:
    """Does any metallic BOUNDARY VALUE reach complex ``step_D``'s output?

    THE ANSWER IS NO, AND IT IS DERIVED. Each shifted tap feeds one curl component
    (read off the three ``curlN_re = curl_stencil(...)`` calls, which is where the
    complex emitter assembles the stencil on each part) and its ghost fires on the
    plane of the axis it shifts
    along; the ownership mask's own block says which (component, plane) pairs are
    zeroed. On the D side the two sets coincide exactly, which is why an edit to what
    the metallic ghost SERVES moves no byte.
    """
    tail = family._real._cut(  # noqa: SLF001
        family.function_source("bloch_pml_curl_step"), family.DECODE_END)
    taps = family.halo_taps(tail)
    offsets = family.offset_coordinates(tail)
    shifted = {"si": "x", "sj": "y", "sk": "z"}
    axis_of_offset = {}
    for name, coordinates in offsets.items():
        moved = [shifted[value] for value in coordinates if value in shifted]
        if len(moved) != 1:
            raise AssertionError(
                f"the certified curl's offset {name} shifts {moved}, not one axis")
        axis_of_offset[name] = moved[0]
    feeds: Dict[str, str] = {}
    for line in tail.splitlines():
        stripped = line.strip()
        for index, term in enumerate(CURL_EXPRESSIONS):
            if not stripped.startswith(f"{term} = curl_stencil("):
                continue
            for register in taps:
                if re.search(rf"\b{re.escape(register)}_re\b", stripped):
                    feeds[register] = f"curl{index}"
    if set(feeds) != set(taps):
        raise AssertionError(
            f"the certified complex curl's terms consume {sorted(feeds)}, not the six "
            f"shifted taps {sorted(taps)}")
    body = tail.split("at_x, at_y, at_z = i == 0", 1)[1].split("\nelse:", 1)[0]
    masked = set(re.findall(
        r"if BC([XYZ]) == METALLIC:\n\s*(curl\d)_re = tl\.where\(at_([xyz])", body))
    rows: Dict[str, Any] = {}
    for register, planes in taps.items():
        _component, offset, _mask = planes["re"]
        axis = axis_of_offset[offset]
        curl = feeds[register]
        rows[register] = {
            "feeds": curl, "ghost_fires_on_plane": axis,
            "masked_there": any(code.lower() == axis and component == curl
                                and plane == axis
                                for code, component, plane in masked)}
    return {"taps": rows,
            "ownership_mask": sorted(masked),
            "every_tap_is_masked_where_its_ghost_fires": all(
                row["masked_there"] for row in rows.values())}


def leg_ghost_observability() -> Dict[str, Any]:
    """The coincidence, asserted; and what it costs the mutation set, stated."""
    findings: List[str] = []
    result = ghost_observability()
    if not result["every_tap_is_masked_where_its_ghost_fires"]:
        findings.append(
            "at least one shifted tap feeds a curl component the ownership mask does "
            "NOT zero at the plane its metallic ghost fires on, so a metallic boundary "
            "value IS observable in step_D's output -- and "
            "m_ghost_is_the_clamped_cells_constitutive is then an ARMED mutation that "
            "must be caught rather than a structural null")
    if len(result["ownership_mask"]) != 6:
        findings.append(f"the BACKWARD ownership mask has "
                        f"{len(result['ownership_mask'])} clauses, not the six the D "
                        f"targets' two zero shifts each give")
    return {"passed": not findings, "findings": findings, **result,
            "what_this_licenses": (
                "that no edit to what the METALLIC ghost serves can move a byte of "
                "complex step_D's output, which is why this gate's ghost mutation is a "
                "recorded null and its PERIODIC sibling is the armed one. It licenses "
                "nothing about the periodic wrap, which is observable, nor about the "
                "BLOCH rotation applied to the wrapped lane, which has its own two "
                "mutations on a phased fixture")}


# ---------------------------------------------------------------------------
# The mutations
# ---------------------------------------------------------------------------

#: The outcome token for a mutant the SUBNORMAL POLICY refused to build. It is a
#: measured state, not an error: under ``flush`` the policy's PTX audit rejects a
#: kernel whose arithmetic disagrees with the policy in force, and a mutation that
#: plants exactly such an instruction is refused before it can run.
PTX_AUDIT_REFUSAL = "REFUSED BY THE PTX AUDIT"

#: What that refusal must SAY before this gate accepts it. A refusal is scored as
#: the declared outcome only when the policy machinery raised it (recognised by the
#: exception class, never by a word) AND its text carries every needle here: the
#: audit's own sentence, and the instruction the mutation planted. Without the
#: second needle any policy refusal at all -- an unattainable executor, a cache-token
#: violation -- would score this row passed, which is a different measurement.
DIVIDE_PTX_REFUSAL_NEEDLES: Tuple[str, ...] = (
    "subnormal policy is not uniform in this kernel's PTX",
    "div.full.f32",
)

#: The shipped forms import the divide mutation SHADOWS. Replacing it with a local
#: ``triton.jit`` definition is what lets a defect in a shared form reach the launch
#: through the shipped planner's own ``kernel=`` door, without touching ``forms.py``,
#: which every kernel of this table imports.
FORMS_IMPORT = ("    from .forms import (axis_present, constitutive_form, curl_form,  "
                "# noqa: PLC0415\n"
                "                        curl_stencil, record_in)\n")
FORMS_IMPORT_WITH_A_DIVIDE = (
    "    from .forms import (axis_present, constitutive_form,  # noqa: PLC0415\n"
    "                        curl_stencil, record_in)\n"
    "\n"
    "    @triton.jit\n"
    "    def curl_form(f, fu, curl, kms, sinv, kms_u, sinv_u, dsigu):\n"
    "        n = ((fu * kms) - curl) * sinv\n"
    "        recurrence = (((f * kms_u) + n) - fu) / (1.0 / sinv_u)\n"
    "        dsig_only = ((f * kms) - curl) * sinv\n"
    "        return tl.where(dsigu, recurrence, dsig_only), n\n")

#: The scratch history store of component 0, as the weld spells it: B on the dsigw
#: cells, the pre-launch word pair everywhere else (the auxiliary contract on a twin
#: the rotation publishes). Spelled once because three rows rewrite it.
_FW_SCRATCH_STORE_0 = (
    "        tl.store(wo0 + 2 * idx, tl.where(dw0, src0_re, tl.load(\n"
    "            wi0 + 2 * idx, mask=live & (dw0 == 0), other=0.0)), mask=live)\n")

#: ``tag -> {target, old, new, expected, case, why}``. ``case`` is the fixture the
#: mutation is SCORED ON, and it is not decoration: a mutation scored where it cannot
#: fire is a disarmed leg wearing a pass.
MUTATIONS: Dict[str, Dict[str, Any]] = {
    # ---- the shape ------------------------------------------------------------------
    "m_foreign_tap_reads_the_stale_value": {
        "target": "kernel", "expected": "CAUGHT", "case": MUTATION_CASE,
        "old": ("        v_re = o0_re\n"
                "        v_im = o0_im\n"
                "        if COMP == 1:\n"),
        "new": ("        _sidx = i * (ny * nz) + j * nz + k\n"
                "        v_re = tl.load(hi0 + 2 * _sidx, mask=valid, other=0.0)\n"
                "        v_im = tl.load(hi0 + 2 * _sidx + 1, mask=valid, other=0.0)\n"
                "        if COMP == 1:\n"),
        "why": "the foreign tap's component 0 reads the PRE-LAUNCH H instead of "
               "recomputing it -- exactly the value an in-place weld would have read "
               "at a neighbour the launch had not yet written. This is the defect the "
               "whole design removes, so it MUST be caught; a null here would mean the "
               "recompute is decorative on this fixture and the purity ledger is the "
               "leg that would have said so first",
    },
    "m_foreign_tap_reads_B": {
        "target": "kernel", "expected": "CAUGHT", "case": MUTATION_CASE,
        "old": ("        (o0_re, o0_im, o1_re, o1_im, o2_re, o2_im,\n"
                "         _s0_re, _s0_im, _s1_re, _s1_im, _s2_re, _s2_im,\n"),
        "new": ("        (_o0_re, _o0_im, _o1_re, _o1_im, _o2_re, _o2_im,\n"
                "         o0_re, o0_im, o1_re, o1_im, o2_re, o2_im,\n"),
        "why": "the tap returns the flux density the constitutive read rather than the "
               "stepped magnetic field -- the mis-selection a fifteen-value return makes "
               "easy, and one the curl would consume without complaint",
    },
    "m_f_w_H_written_in_place": {
        "target": "kernel", "expected": "CAUGHT", "case": MUTATION_CASE,
        "old": "        tl.store(wo0 + 2 * idx, tl.where(dw0, src0_re, tl.load(\n",
        "new": "        tl.store(wi0 + 2 * idx, tl.where(dw0, src0_re, tl.load(\n",
        "why": "the split-field history is written into the PRE-LAUNCH buffer, so a "
               "foreign recompute of a neighbour reads B where it needs B_prev -- the "
               "second, easily missed half of the hazard -- and the rotation then "
               "publishes an unwritten scratch",
    },
    "m_own_cell_takes_the_foreign_recompute": {
        "target": "kernel", "expected": "CAUGHT", "expected_override": "NULL",
        "case": MUTATION_CASE,
        "old": ("        a_re = own0_re\n"
                "        a_im = own0_im\n"),
        "new": ("        a_re, a_im = _h_tap_complex(0, i, j, k, live,\n"
                "                             hi0, hi1, hi2, wi0, wi1, wi2, b0, b1, "
                "b2,\n"
                "                             kp0, kp1, kp2, kmx, kmy, kmz, cb, "
                "start_h, ny, nz)\n"),
        "why": "PREDICTED NULL and recorded as one. `_h_tap_complex` at the program's "
               "own cell with `valid=live` is `_h_cell_complex` at that cell closed on "
               "the same guard, which is what `own0_re/own0_im` already are -- so this "
               "must NOT diverge, and that is the measurement showing the own-cell "
               "register pair and the recompute are ONE code path rather than two that "
               "happen to agree",
    },
    # ---- the forms, inverted to the previous package grouping (KERNELS.md 7.1) --------
    "a3_previous_constitutive_grouping": {
        "target": "kernel", "expected": "CAUGHT", "case": MUTATION_CASE,
        "old": ("        out0_re = constitutive_form(a_re, prev_re, src_re, kp_0, km_0, "
                "dw0)\n"),
        "new": ("        out0_re = tl.where(dw0, (a_re + kp_0 * src_re) - km_0 * prev_re, "
                "src_re)\n"),
        "why": "INVERTED (KERNELS.md 7.1): Hx's real part takes the PREVIOUS package "
               "grouping, the two accumulations `((f + kps*src) - kms*prev)`, in place "
               "of MEEP's one add of the difference `f + ((kps*src) - (kms*prev))` that "
               "forms.constitutive_form spells. Different float32 numbers on the dsigw "
               "cells, at the own cell and at every foreign recompute alike",
    },
    "a1_pairwise_stencil": {
        "target": "kernel", "expected": "CAUGHT", "case": MUTATION_CASE,
        "old": ("        curl0_re = curl_stencil(c_y_re, c0_re, b0_re, b_z_re, dtdx, py, "
                "pz)\n"),
        "new": "        curl0_re = dtdx * ((c_y_re - c0_re) + (b0_re - b_z_re))\n",
        "why": "INVERTED (KERNELS.md 7.1): Dx's real curl takes the PREVIOUS package "
               "grouping, the pairwise `(g1s - g1) + (g2 - g2s)`, in place of MEEP's "
               "left-to-right `((g1s - g1) + g2) - g2s` that forms.curl_stencil spells; "
               "`(a + b) - c` and `a + (b - c)` round differently in float32",
    },
    # ---- the geometry, unmoved -------------------------------------------------------
    "m_ghost_is_the_clamped_cells_constitutive": {
        "target": "kernel", "expected": "CAUGHT", "expected_override": "NULL",
        "case": MUTATION_CASE,
        "old": "        b_x_re, b_x_im = _h_tap_complex(1, si, j, k, vx,\n",
        "new": ("        b_x_re, b_x_im = _h_tap_complex(1, tl.maximum(si, 0), j, k, "
                "live,\n"),
        "why": "past a metallic wall the certified load serves an EXACT (+0.0, +0.0); "
               "this serves the ghost cell's own constitutive instead, clamped into "
               "range. DECLARED CAUGHT, MEASURED NULL, AND THE NULL IS STRUCTURAL -- "
               "on the D side every shifted tap feeds a curl component "
               "`_mask_non_owned_cells` zeroes at EXACTLY the plane where the metallic "
               "ghost fires, which `leg_ghost_observability` DERIVES from the certified "
               "text. `m_ownership_mask_dropped_on_x` is the discriminating control",
    },
    "m_ownership_mask_dropped_on_x": {
        "target": "kernel", "expected": "CAUGHT", "case": MUTATION_CASE,
        "old": ("            if BCX == METALLIC:\n"
                "                curl2_re = tl.where(at_x, 0.0, curl2_re)\n"
                "                curl2_im = tl.where(at_x, 0.0, curl2_im)\n"),
        "new": ("            if BCX == METALLIC:\n"
                "                curl2_re = curl2_re\n"
                "                curl2_im = curl2_im\n"),
        "why": "the ownership mask stops zeroing curl2 on the metallic x wall. THE "
               "DISCRIMINATING CONTROL for the structural null above: if this were also "
               "null, the tap feeding curl2 would be dead and the ghost's "
               "unobservability would say nothing",
    },
    "m_ownership_mask_dropped_on_the_imaginary_plane": {
        "target": "kernel", "expected": "CAUGHT", "case": MUTATION_CASE,
        "old": "                curl0_im = tl.where(at_y, 0.0, curl0_im)\n",
        "new": "                curl0_im = curl0_im\n",
        "why": "THE COMPLEX-SPECIFIC HAZARD. The array path assigns COMPLEX zero to a "
               "non-owned cell; a transcription that cleared only the real plane is "
               "wrong exactly where the imaginary part is live at a wall, and nowhere "
               "else -- invisible to any real-field test",
    },
    "m_periodic_wrap_reads_the_near_row": {
        "target": "kernel", "expected": "CAUGHT", "case": PERIODIC_MUTATION_CASE,
        "old": "            si = tl.where(si < 0, nx - 1, tl.where(si == nx, 0, si))\n",
        "new": "            si = tl.where(si < 0, 0, tl.where(si == nx, 0, si))\n",
        "why": "the periodic wrap reads row 0 instead of the far row. SCORED ON THE "
               "ALL-PERIODIC FIXTURE and nowhere else: on a metallic axis this branch "
               "does not compile at all",
    },
    "m_word_index_not_doubled": {
        "target": "kernel", "expected": "CAUGHT", "case": MUTATION_CASE,
        "old": ("        prev_re = tl.load(wi1 + 2 * idx, mask=live & dw1, "
                "other=0.0)\n"),
        "new": "        prev_re = tl.load(wi1 + idx, mask=live & dw1, other=0.0)\n",
        "why": "the complex word addressing undone on Hy's pre-launch history: "
               "`wi1 + idx` reads the wrong plane of the wrong cell on every dsigw "
               "cell, at the own cell and at every foreign recompute",
    },
    # ---- the Bloch connection, on a phased fixture -----------------------------------
    "m_connection_applied_to_every_lane": {
        "target": "kernel", "expected": "CAUGHT", "case": PHASED_MUTATION_CASE,
        "old": ("            c1_re = tl.where(at_x, r_re, c_re)\n"
                "            c1_im = tl.where(at_x, r_im, c_im)\n"),
        "new": ("            c1_re = r_re\n"
                "            c1_im = r_im\n"),
        "why": "the x connection reaches EVERY lane of target 1's g2 instead of plane "
               "0 of the unshifted operand. The certified curl applies it through "
               "`tl.where`, a bitwise select, so a cell off plane 0 keeps its loaded "
               "words; this weld redirects the LOADS and must leave that select "
               "standing. SCORED ON A PHASED FIXTURE: `CNX` is a constexpr",
    },
    "m_connection_dropped_on_x": {
        "target": "kernel", "expected": "CAUGHT", "case": PHASED_MUTATION_CASE,
        "old": "        if CNX:\n",
        "new": "        if 0:\n",
        "why": "the x connection thrown away: the Bloch run becomes a plain periodic "
               "one on x. Invisible at k = 0 by construction, which is why it is scored "
               "on the phased fixture",
    },
    "m_connection_on_the_old_wrapped_plane": {
        "target": "kernel", "expected": "CAUGHT", "case": PHASED_MUTATION_CASE,
        "old": ("            c1_re = tl.where(at_x, r_re, c_re)\n"
                "            c1_im = tl.where(at_x, r_im, c_im)\n"
                "            r_re, r_im = _connect(b_re, b_im, cxr, cxi, CNX)\n"
                "            b2_re = tl.where(at_x, r_re, b_re)\n"
                "            b2_im = tl.where(at_x, r_im, b_im)\n"),
        "new": ("            c1_re = tl.where(i == nx - 1, r_re, c_re)\n"
                "            c1_im = tl.where(i == nx - 1, r_im, c_im)\n"
                "            r_re, r_im = _connect(b_re, b_im, cxr, cxi, CNX)\n"
                "            b2_re = tl.where(i == nx - 1, r_re, b_re)\n"
                "            b2_im = tl.where(i == nx - 1, r_im, b_im)\n"),
        "why": "INVERTED (Appendix A): the x connection on the LAST plane, where the "
               "rolled operand used to carry the phase (the previous wrapped-lane "
               "rotation), instead of the unshifted operand's plane 0 that MEEP's "
               "not-owned word lands on",
    },
    "m_connection_direction_swapped": {
        "target": "host", "expected": "CAUGHT", "case": PHASED_MUTATION_CASE,
        "why": "the plan binds step_B's connection, `conj(eikna)` below the window, "
               "where step_D's backward differences read `eikna` above it "
               "(complex_fields.connection_arguments, backward=True). It compiles, "
               "launches and converges, and it is the wrong not-owned word on every "
               "phased axis. Driven by rebinding the plan's kinds and parts to the "
               "forward classification",
    },
    "m_divide_by_the_reciprocal_of_sinv": {
        "target": "kernel", "expected": "CAUGHT", "case": ARITHMETIC_MUTATION_CASE,
        "old": FORMS_IMPORT, "new": FORMS_IMPORT_WITH_A_DIVIDE,
        "why": "THE DIVIDE FACT, RE-MEASURED ON THIS KERNEL. Every PML reciprocal is "
               "precomputed on the HOST into sinv_* and both halves MULTIPLY by it; "
               "this mutant shadows the shared forms import with a curl_form whose "
               "recurrence divides by the reciprocal of sinv_u instead of multiplying "
               "by it -- the same mathematics, and wrong wherever the quotient rounds "
               "differently from the product. WHERE IT IS SCORED IS MEASURED, NOT "
               "ASSUMED: Triton's `/` is div.full.f32, and on the device the quotient "
               "x / (1 / s) equals x * s bitwise for two of this geometry's four PML "
               "reciprocals (0.43956688 and 0.63830394) while it moves 81-97 % of "
               "words for the other two (0.30612803 and 0.87591523); a correctly "
               "rounded host moves 28-44 % for all four. On bc_mmm the recurrence "
               "site moves no word on the device (NULL over 6 steps with the planted "
               "div.full.f32 compiled; the same respelling at the two sinv sites moves "
               "514 words at step 1 with or without the recurrence site), so the row "
               "is scored on the planted tiny-normal case, where the recurrence's "
               "divide moves 2,656 words at step 1 on the device (one RTX A6000, "
               "Triton 3.1.0, keep, 2026-10-10). UNDER `flush` THE MUTANT NEVER "
               "RUNS, and that is a stronger answer than a divergence: the same "
               "`div.full.f32` is what the flush policy's own PTX audit refuses at "
               "compile time. The refusal is accepted only when the policy machinery "
               "itself raises it AND its text carries every needle below",
        "expected_by_policy": {"keep": "CAUGHT", "flush": PTX_AUDIT_REFUSAL},
        "refusal_needles_by_policy": {"flush": DIVIDE_PTX_REFUSAL_NEEDLES},
    },
    # ---- the chunk forms (KERNELS.md 7.2): every select keyed on the record ----------
    "b1_whole_grid_recurrence": {
        "target": "kernel", "expected": "CAUGHT", "case": MUTATION_CASE,
        "old": "        du0 = record_in(cb, lattice + 8, k)\n",
        "new": "        du0 = idx >= 0\n",
        "why": "the select is removed for Dx: the split-field recurrence runs at EVERY "
               "cell on both parts, as the 0.9.2 kernels ran it, and fu_Dx is read and "
               "written everywhere. Off the dsigu chunk MEEP steps the dsig-only (or "
               "plain) form on the field alone, so the words differ and the auxiliary "
               "moves",
    },
    "b2_coefficient_predicate": {
        "target": "kernel", "expected": "CAUGHT", "case": MUTATION_CASE,
        "old": "        du0 = record_in(cb, lattice + 8, k)\n",
        "new": "        du0 = km_z != 1.0\n",
        "why": "Dx's form is keyed on coefficient identity (`kms != 1`) instead of chunk "
               "membership. A cell inside the dsigu chunk whose sigma is zero (the "
               "layer's inner edge) then takes the dsig-only form where MEEP steps the "
               "recurrence",
    },
    "b3_membership_one_row_off": {
        "target": "kernel", "expected": "CAUGHT", "case": MUTATION_CASE,
        "old": "        du0 = record_in(cb, lattice + 8, k)\n",
        "new": "        du0 = record_in(cb, lattice + 8, k + 1)\n",
        "why": "Dx's membership is read one row off along its dsigu axis, so the "
               "boundary plane of each chunk run changes form",
    },
    "b4_curl_lattice_swapped": {
        "target": "kernel", "expected": "CAUGHT", "case": MUTATION_CASE,
        "old": ("        if BACKWARD:\n"
                "            lattice = 0\n"
                "        else:\n"
                "            lattice = RECORD_HALF_INTEGER\n"),
        "new": ("        if BACKWARD:\n"
                "            lattice = RECORD_HALF_INTEGER\n"
                "        else:\n"
                "            lattice = 0\n"),
        "why": "the D curl reads the record's HALF-INTEGER lines instead of its integer "
               "ones; the two runs differ by one cell at every layer edge",
    },
    "b5_dsig_dsigu_swapped": {
        "target": "kernel", "expected": "CAUGHT", "case": MUTATION_CASE,
        "old": "        du0 = record_in(cb, lattice + 8, k)\n",
        "new": "        du0 = record_in(cb, lattice + 4, j)\n",
        "why": "Dx's recurrence is keyed on its dsig axis (y) instead of its dsigu axis "
               "(z): a cell in a y layer but no z layer takes the recurrence where MEEP "
               "steps the dsig-only form, and the reverse",
    },
    "b6_fu_store_unmasked": {
        "target": "kernel", "expected": "CAUGHT", "case": MUTATION_CASE,
        "old": "        tl.store(u0 + 2 * idx, n0_re, mask=live & du0)\n",
        "new": "        tl.store(u0 + 2 * idx, n0_re, mask=live)\n",
        "why": "fu_Dx's real word is stored on a plain cell, which the auxiliary "
               "contract (C2) leaves untouched; every fixture volume is seeded at every "
               "cell, so the stray word is observable",
    },
    "b7_fw_scratch_stored_unmasked": {
        "target": "kernel", "expected": "CAUGHT", "case": MUTATION_CASE,
        "old": _FW_SCRATCH_STORE_0,
        "new": "        tl.store(wo0 + 2 * idx, src0_re, mask=live)\n",
        "why": "the scratch f_w_Hx real word takes B at every cell, as the 0.9.2 "
               "kernels wrote it, instead of B on the dsigw cells and the pre-launch "
               "word everywhere else; after the rotation the engine's history has moved "
               "on cells the auxiliary contract leaves at rest",
    },
    "b8_fw_scratch_stored_on_the_layer_only": {
        "target": "kernel", "expected": "CAUGHT", "case": MUTATION_CASE,
        "old": _FW_SCRATCH_STORE_0,
        "new": "        tl.store(wo0 + 2 * idx, src0_re, mask=live & dw0)\n",
        "why": "the certified single's masked history store copied literally onto the "
               "scratch: the twin is rotated into the engine, so every plain cell of "
               "f_w_Hx is published as the twin's stale word instead of the kept word. "
               "The trap a scratch weld sets for the auxiliary contract",
    },
    "b9_constitutive_membership_one_row_off": {
        "target": "kernel", "expected": "CAUGHT", "case": MUTATION_CASE,
        "old": "        dw0 = record_in(cb, lattice + 0, i)\n",
        "new": "        dw0 = record_in(cb, lattice + 0, i + 1)\n",
        "why": "Hx's dsigw membership is read one row off along x, at the own cell and "
               "at every foreign recompute alike, so the boundary plane of each run "
               "takes the other constitutive form",
    },
    "b10_constitutive_lattice_swapped": {
        "target": "kernel", "expected": "CAUGHT", "case": MUTATION_CASE,
        "old": ("        lattice = 0\n"
                "        dw0 = record_in(cb, lattice + 0, i)\n"),
        "new": ("        lattice = RECORD_HALF_INTEGER\n"
                "        dw0 = record_in(cb, lattice + 0, i)\n"),
        "why": "update_H reads the record's HALF-INTEGER dsigw lines instead of its "
               "integer ones; the runs differ by one cell at every layer edge",
    },
    "b11_start_of_H_ignored": {
        "target": "kernel", "expected": "CAUGHT", "case": FIRST_UPDATE_CASE,
        "old": "        start = start_h != 0\n",
        "new": "        start = start_h != start_h\n",
        "why": "MEEP's start of H is dropped: the run's first update_H accumulates onto "
               "the stored H and history instead of starting both as copies of B on the "
               "dsigw cells (stepping.separate_h_from_b). Scored on the first-update "
               "case, the one fixture that starts unseparated, so step 1 diverges",
    },
    "b12_presence_ignored": {
        "target": "kernel", "expected": "CAUGHT", "case": MUTATION_CASE,
        "old": "        presence = tl.load(cb + RECORD_PRESENCE)\n",
        "new": "        presence = tl.load(cb + RECORD_PRESENCE) * 0\n",
        "why": "the record's partner-presence word read as no partner on any axis: "
               "every curl takes the exact +0 MEEP reserves for a target with neither "
               "partner, where both partners are live on this 3-D fixture",
    },
    "n1_fu_load_unmasked": {
        "target": "kernel", "expected": "CAUGHT", "expected_override": "NULL",
        "case": MUTATION_CASE,
        "old": "        p_re = tl.load(u0 + 2 * idx, mask=live & du0, other=0.0)\n",
        "new": "        p_re = tl.load(u0 + 2 * idx, mask=live, other=0.0)\n",
        "why": "PREDICTED NULL (KERNELS.md 7.2, the unconditional auxiliary load): fu_Dx "
               "is read on every cell, but off the dsigu chunk the selected form never "
               "consumes it and its store stays masked, so no word moves. The guard is "
               "pinned by the lift instead (the load is the certified masked line)",
    },
    "n2_fw_load_unmasked": {
        "target": "kernel", "expected": "CAUGHT", "expected_override": "NULL",
        "case": MUTATION_CASE,
        "old": ("        prev_re = tl.load(wi0 + 2 * idx, mask=live & dw0, other=0.0)   "
                "# BEFORE the store.\n"),
        "new": ("        prev_re = tl.load(wi0 + 2 * idx, mask=live, other=0.0)   "
                "# BEFORE the store.\n"),
        "why": "PREDICTED NULL (KERNELS.md 7.2, the unconditional auxiliary load): the "
               "history is read on every cell, but off the dsigw chunk "
               "constitutive_form returns the source and never consumes it, so no word "
               "moves. The guard is pinned by the lift instead",
    },
    # ---- the host mutations ----------------------------------------------------------
    "m_rotation_skipped": {
        "target": "host", "expected": "CAUGHT", "case": MUTATION_CASE,
        "why": "the plan launches and does NOT move the engine's references onto the "
               "freshly written scratch, so the run publishes the pre-launch magnetic "
               "field. The choreography is half the product",
    },
    "m_constitutive_takes_the_half_integer_lattice": {
        "target": "host", "expected": "CAUGHT", "case": MUTATION_CASE,
        "why": "the constitutive half is bound the `_h` PML sub-lattice instead of the "
               "integer one. It compiles, launches and converges; what it produces is a "
               "half-cell-wrong absorber profile, which is why this is a gate leg and "
               "not a comment",
    },
    "m_withdraw_after_step_D": {
        "target": "host", "expected": "CAUGHT", "case": MUTATION_CASE,
        "why": "the seam's one pass on the wrong side of the launch. Since entry 25 that "
               "pass is the magnetic f_minus_p restore (no electric withdraw stands), "
               "and the withdraw leg drives it moved before the launch, and dropped, on "
               "a fixture carrying integrated electric and magnetic sources; both must "
               "diverge (hd_seam_restore.SEAM_NULL_CONTROLS)",
    },
}

#: The one edit required NOT to diverge, and it is the leg that makes the register pair
#: a measurement rather than a tautology.
BYTE_NEUTRAL = {
    "tag": "m_reload_own_H_from_the_scratch",
    "old": ("        a_re = own0_re\n"
            "        a_im = own0_im\n"
            "        b_re = own1_re\n"
            "        b_im = own1_im\n"
            "        c_re = own2_re\n"
            "        c_im = own2_im\n"),
    "new": ("        a_re = tl.load(ho0 + 2 * idx, mask=live, other=0.0)\n"
            "        a_im = tl.load(ho0 + 2 * idx + 1, mask=live, other=0.0)\n"
            "        b_re = tl.load(ho1 + 2 * idx, mask=live, other=0.0)\n"
            "        b_im = tl.load(ho1 + 2 * idx + 1, mask=live, other=0.0)\n"
            "        c_re = tl.load(ho2 + 2 * idx, mask=live, other=0.0)\n"
            "        c_im = tl.load(ho2 + 2 * idx + 1, mask=live, other=0.0)\n"),
    "why": "the curl half re-loads its own cell's magnetic word pair from the scratch "
           "this program just wrote, instead of using the registers. PREDICTED NULL: "
           "one program's store and load of one address are ordered, so this is inert "
           "-- and confirming it is what shows the correctness comes from the FOREIGN "
           "taps and not from the register",
}


def mutated_family(tag: str, old: str, new: str) -> Any:
    """Import a COPY of the family module with one source edit applied.

    The mutation reaches the launch through the shipped planner's ``kernel=`` door,
    which is the door the product itself documents; a harness that launched the shipped
    kernel instead would report a pass for a defect it never introduced.
    """
    import linecache  # noqa: PLC0415
    import types  # noqa: PLC0415

    source_path = (API_ROOT / "meep_gpu" / "triton_kernels"
                   / "complex_fused_hd_pair.py")
    text = source_path.read_text(encoding="utf-8")
    hits = text.count(old)
    if hits != 1:
        raise AssertionError(
            f"{tag}: the mutation needle matches {hits} times, not once; a mutation "
            f"that matched nothing is a disarmed leg and one that matched twice is two "
            f"defects")
    mutated_text = text.replace(old, new, 1)
    if mutated_text == text:
        raise AssertionError(f"{tag}: the mutation changed nothing")
    name = f"meep_gpu.triton_kernels.complex_fused_hd_pair__mut_{tag}"
    filename = str(source_path) + f"#{tag}"
    # TRITON READS THE SOURCE BACK. `JITFunction.__init__` calls
    # `inspect.getsourcelines(fn)`, and a function created by `exec` has no file for
    # `inspect` to find. Registering the mutated text in `linecache` under this module's
    # own fake filename is what makes the mutant compile, and it also keeps Triton's JIT
    # cache key distinct: the key is over the SOURCE, so a warm cache cannot serve the
    # shipped binary for a mutant.
    linecache.cache[filename] = (len(mutated_text), None,
                                 mutated_text.splitlines(True), filename)
    module = types.ModuleType(name)
    module.__file__ = filename
    module.__package__ = "meep_gpu.triton_kernels"
    sys.modules[name] = module
    code = compile(mutated_text, filename, "exec")
    exec(code, module.__dict__)  # noqa: S102 - a deliberate mutation copy
    return module


class RotationSkipped:
    """The weld's launch WITHOUT the reference rotation -- a host mutation."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.absorbed_by = inner

    def run(self, guard: Optional[bool] = None) -> None:
        writes, reads = self.inner._resolve()  # noqa: SLF001
        self.inner._launch(writes, reads, guard)  # noqa: SLF001
        self.inner.launches += 1

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


def half_integer_rebind(pml: Any) -> Callable[[Any], Any]:
    """Rebind the plan's coefficient group to the ``_h`` sub-lattice -- a host mutation."""

    def wrap(plan: Any) -> Any:
        from meep_gpu.triton_kernels.launch import CupyPointer, _flat  # noqa: PLC0415

        plan._curl_coeff = tuple(  # noqa: SLF001
            CupyPointer(_flat(getattr(pml, f"{stem}_{axis}_h")))
            for axis in "xyz" for stem in ("kms", "sinv"))
        plan._kps = tuple(  # noqa: SLF001
            CupyPointer(_flat(getattr(pml, f"kps_{axis}_h"))) for axis in "xyz")
        return plan

    return wrap


def connection_direction_swapped(driver: Any) -> Callable[[Any], Any]:
    """Rebind the plan's connection to the FORWARD (step_B) classification."""

    def wrap(plan: Any) -> Any:
        kinds = stepping._boundary_kinds(driver.fields.grid, driver.pml)  # noqa: SLF001
        phases = complex_fields.bloch_phase_table(driver.fields.grid, kinds)
        connections, values = complex_fields.connection_arguments(
            phases, backward=False)
        plan.connections = tuple(int(kind) for kind in connections)
        plan.connection_values = tuple(float(value) for value in values)
        return plan

    return wrap


# ---------------------------------------------------------------------------
# DEVICE LEGS
# ---------------------------------------------------------------------------

def leg_product(steps: int, seed: int,
                progress: Optional[Callable[[str], None]] = None) -> List[Dict[str, Any]]:
    """The multi-reference identity on every fixture."""
    rows: List[Dict[str, Any]] = []
    for name, keywords in CASES:
        driver = build_driver(dict(keywords), seed=seed)
        try:
            result = run_product(driver, steps, progress=(
                (lambda message, case=name: progress(f"{case} {message}"))
                if progress else None))
            result["planted_words"] = getattr(driver, "_planted", None)
        finally:
            driver.close()
        result.update({"leg": "product", "case": name,
                       "boundaries": dict(keywords.get("boundaries") or {}),
                       "k_point": list(keywords.get("k_point") or (0.0, 0.0, 0.0))})
        rows.append(result)
    return rows


def leg_seed_scale(steps: int, seed: int) -> Dict[str, Any]:
    """The 2^80 seed scale is a change of exponent and nothing else. DRIVEN."""
    findings: List[str] = []
    rows: Dict[str, Any] = {}
    for bits in (0, 40, SEED_SCALE_BITS):
        driver = build_driver(dict(CASES[MUTATION_INDEX][1]), seed=seed,
                              scale_bits=bits)
        try:
            result = run_product(driver, steps, require_full_budget=True)
        finally:
            driver.close()
        rows[f"2^{bits}"] = {
            "bit_identical": result["bit_identical"],
            "first_divergence_step": result["first_divergence_step"],
            "steps_compared": result["steps_compared"],
            "reference_subnormal_words": result["reference_subnormal_words"]}
    if not rows[f"2^{SEED_SCALE_BITS}"]["bit_identical"]:
        findings.append(f"the shipped scale 2^{SEED_SCALE_BITS} is not identical")
    if rows[f"2^{SEED_SCALE_BITS}"]["reference_subnormal_words"]:
        findings.append(
            f"the shipped scale still leaves stored words in the denormal band: "
            f"{rows[f'2^{SEED_SCALE_BITS}']['reference_subnormal_words']}")
    return {"passed": not findings, "findings": findings, "scales": rows}


MUTATION_INDEX = [name for name, _ in CASES].index(MUTATION_CASE)
PERIODIC_INDEX = [name for name, _ in CASES].index(PERIODIC_MUTATION_CASE)
PHASED_INDEX = [name for name, _ in CASES].index(PHASED_MUTATION_CASE)


def _weld_step_launches(arrangements: Mapping[str, Arrangement]) -> int:
    shim = arrangements["weld"].shim
    owners: List[int] = []
    for plan in (shim.plans.values() if shim else ()):
        if plan is ABSORBED:
            continue
        for entry in (plan if isinstance(plan, (list, tuple)) else [plan]):
            owner = declaring(entry)
            if id(owner) not in owners:
                owners.append(id(owner))
    return len(owners)


def leg_launch_structure(steps: int, seed: int) -> Dict[str, Any]:
    """Launches per step at the seam AND over the whole step, two counters."""
    findings: List[str] = []
    driver = build_driver(dict(CASES[MUTATION_INDEX][1]), seed=seed)
    try:
        arrangements, absent = all_arrangements(driver, count=True)
        result = drive(driver, arrangements, min(steps, 8))
        launches = result["launches"]
        seam_launches: Dict[str, int] = {}
        for name, arrangement in arrangements.items():
            if arrangement.shim is None:
                seam_launches[name] = 0
                continue
            owners: List[int] = []
            total = 0
            for slot in family.REPLACES:
                plan = arrangement.shim.plans.get(slot)
                if plan is None or plan is ABSORBED:
                    continue
                owner = declaring(plan)
                if id(owner) in owners:
                    continue
                owners.append(id(owner))
                total += 1
            seam_launches[name] = total
        weld_step_launches = _weld_step_launches(arrangements)
    finally:
        driver.close()
    steps_run = result["steps_compared"]
    for name, counts in launches.items():
        if name == "array":
            continue
        if counts["uncounted_plans"]:
            findings.append(
                f"{name}: the kernel counter cannot see {counts['uncounted_plans']}, "
                f"so the two witnesses are not comparable on this fixture")
        elif counts["plans"] != counts["kernels"]:
            findings.append(
                f"{name}: the plans counted {counts['plans']} launches and the "
                f"independent kernel counter {counts['kernels']}")
    if seam_launches.get("weld") != 1:
        findings.append(f"the weld occupies {seam_launches.get('weld')} launches at the "
                        f"seam, not one")
    if seam_launches.get("singles") != 2:
        findings.append(f"the certified singles occupy {seam_launches.get('singles')} "
                        f"launches at the seam, not two")
    expected = steps_run * weld_step_launches
    if launches["weld"]["kernels"] != expected:
        findings.append(
            f"the weld arrangement launched {launches['weld']['kernels']} kernels over "
            f"{steps_run} steps; its {weld_step_launches} distinct plans launching once "
            f"each per step is {expected}")
    saved_whole_step = None
    if "composition_today" in launches and steps_run:
        saved_whole_step = ((launches["composition_today"]["kernels"]
                             - launches["weld"]["kernels"]) / steps_run)
    return {
        "passed": not findings, "findings": findings,
        "steps": steps_run,
        "references_this_policy_cannot_build": absent,
        "launches_over_the_run": launches,
        "launches_per_step": {name: {key: round(value / steps_run, 4)
                                     for key, value in counts.items()
                                     if isinstance(value, int)}
                              for name, counts in launches.items()} if steps_run else {},
        "seam_launches_per_step": seam_launches,
        "what_the_weld_saves_at_the_seam": (
            seam_launches.get("singles", 0) - seam_launches.get("weld", 0)),
        "what_the_weld_saves_over_the_whole_step_against_the_composition_today":
            saved_whole_step,
        "the_honest_reading": (
            "the weld saves ONE launch against the two certified complex singles at "
            "its own seam. What it saves over the whole step depends on what the "
            "composer installs on this row, which is measured here and, per corpus "
            "row, in the lift leg's arbitration table -- never assumed. No timing "
            "exists for this shape and none is licensed by this count"),
    }


def leg_sync(steps: int, seed: int) -> Dict[str, Any]:
    """The sync channel: a plan spanning ``step_D`` must DECLINE the half-step."""
    findings: List[str] = []
    out: Dict[str, Any] = {}

    def hook(step: int, name: str, driver: Any) -> Any:
        if step not in SYNC_STEPS:
            return None
        driver.synchronize_magnetic_fields()
        driver.restore_magnetic_fields()
        return {"step": step, "synchronized": True}

    for armed in (False, True):
        driver = build_driver(dict(CASES[MUTATION_INDEX][1]), seed=seed)
        try:
            arrangements = {
                "array": Arrangement("array", None),
                "weld": arrangement_weld(
                    driver, sync_hazard=armed,
                    name="weld_sync_armed" if armed else "weld"),
            }
            result = drive(driver, arrangements, min(steps, 10), per_step_hook=hook,
                           stop_on_divergence=False)
            shim = arrangements["weld"].shim
            out["armed" if armed else "declining"] = {
                "bit_identical": result["bit_identical"],
                "first_divergence_step": result["first_divergence_step"],
                "sync_refusals": shim.sync_refusals,
                "sync_answered": shim.sync_answered,
                "differing_volumes_at_the_end": (
                    result["per_step"][-1].get("weld_sync_armed")
                    or result["per_step"][-1].get("weld", {})).get(
                        "differing_volumes", {}),
                "steps_compared": result["steps_compared"]}
        finally:
            driver.close()
    declining, armed_row = out["declining"], out["armed"]
    if not declining["bit_identical"]:
        findings.append("the DECLINING arm is not identical to the array path")
    if declining["sync_refusals"] == 0:
        findings.append("the declining arm never refused the sync consult; the leg is "
                        "disarmed -- the accessor is not reaching the channel")
    if armed_row["bit_identical"]:
        findings.append("the ARMED arm did NOT diverge; the hazard this refusal exists "
                        "to prevent is not observable on this fixture")
    else:
        electric = {name for name in armed_row["differing_volumes_at_the_end"]
                    if name.startswith(("D", "fu_D", "f_cond_D"))}
        if not electric:
            findings.append(
                f"the armed arm diverged but not in D/fu_D/f_cond_D "
                f"({sorted(armed_row['differing_volumes_at_the_end'])})")
        armed_row["electric_volumes_that_diverged"] = sorted(electric)
    return {"passed": not findings, "findings": findings, "arms": out,
            "sync_steps": list(SYNC_STEPS)}


def leg_withdraw(steps: int, seed: int) -> Dict[str, Any]:
    """The seam's one pass on a device: :func:`hd_seam_restore.seam_restore_leg`.

    Since entry 25 what stands between the two consults is the magnetic ``f_minus_p``
    restore, not an electric withdraw. The complex driver carries an integrated electric
    AND an integrated magnetic source; the weld arrangement is force-installed, stepped
    in lockstep with its own array-path reference (its rotation settled after every
    step, as :func:`drive` settles it) and required identical, and the weld with the
    restore moved before the launch and with no magnetic ``f_minus_p`` must each
    diverge. THE SEED IS NOT SCALED HERE (2^0): the ``f_minus_p`` words are the sources'
    amplitude, which a 2^80 state would swallow.

    Both policies reach the composer (entry 89 deleted the complex licence), so the
    predicate must ADMIT the integrated sources under either, and never refuse by a
    withdraw.
    """
    keywords = dict(CASES[MUTATION_INDEX][1])
    sources = (hd_seam_restore.INTEGRATED_ELECTRIC_SOURCE,
               hd_seam_restore.INTEGRATED_MAGNETIC_SOURCE)
    built: List[Any] = []
    welds: Dict[int, Arrangement] = {}

    def build() -> Any:
        driver = build_driver(keywords, seed=seed, sources=sources, scale_bits=0)
        built.append(driver)
        return driver

    def install_weld(driver: Any) -> Callable[[], int]:
        arrangement = arrangement_weld(driver)
        welds[id(driver)] = arrangement
        install(driver, arrangement.shim)
        # The weld's own launches: the update_H slot it holds (the shim counts every
        # slot's plan, and the other slots carry the unfused singles).
        return lambda: int(arrangement.shim.dispatched.get("update_H", 0))

    def coverage(driver: Any) -> Tuple[bool, str]:
        verdict = family.complex_fused_hd_pair_coverage(
            driver.fields, driver.pml, tuple(getattr(driver, "_sources", ())))
        return bool(verdict.covered), "; ".join(verdict.reasons)

    try:
        record = hd_seam_restore.seam_restore_leg(
            build=build, install_weld=install_weld, coverage=coverage,
            state_of=lambda driver: capture(driver)["volumes"],
            compare=lambda left, right: compare_snapshots({"volumes": left},
                                                          {"volumes": right}),
            synchronize=lambda: None, steps=steps, spec_label=MUTATION_CASE,
            settle=lambda driver: welds[id(driver)].settle(driver.fields))
    finally:
        for driver in built:
            with contextlib.suppress(Exception):
                driver.close()
    return record


def _mutation_case_driver(tag: str, seed: int) -> Tuple[Any, str]:
    case = MUTATIONS[tag]["case"] if tag in MUTATIONS else MUTATION_CASE
    keywords = dict(dict(CASES)[case])
    return build_driver(keywords, seed=seed), case


def leg_byte_neutral(steps: int, seed: int) -> Dict[str, Any]:
    """The one armed edit required NOT to diverge."""
    findings: List[str] = []
    driver, case = _mutation_case_driver(BYTE_NEUTRAL["tag"], seed)
    try:
        module = mutated_family(BYTE_NEUTRAL["tag"], BYTE_NEUTRAL["old"],
                                BYTE_NEUTRAL["new"])
        kernel = module.fused_complex_constitutive_curl_H_to_D
        arrangements = {"array": Arrangement("array", None),
                        "weld": arrangement_weld(driver, kernel=kernel)}
        result = drive(driver, arrangements, steps, stop_on_divergence=False)
    finally:
        driver.close()
    if not result["bit_identical"]:
        findings.append(
            f"the byte-neutral control DIVERGED at step "
            f"{result['first_divergence_step']}; the register pair is then not the "
            f"identity of the store it replaces and this gate's mutation legs are "
            f"measuring something other than what they claim")
    return {"passed": not findings, "findings": findings, "case": case,
            "tag": BYTE_NEUTRAL["tag"], "why": BYTE_NEUTRAL["why"],
            "bit_identical": result["bit_identical"],
            "steps_compared": result["steps_compared"]}


def leg_disarm(steps: int, seed: int) -> List[Dict[str, Any]]:
    """The identical harness with the SHIPPED bytes, on every mutation fixture.

    ONE ROW PER FIXTURE THE MUTATIONS ARE SCORED ON, not one for the default: a
    mutation scored on the planted or the phased case is only attributable if the
    shipped bytes are identical THERE.
    """
    rows: List[Dict[str, Any]] = []
    for case in dict.fromkeys(
            (MUTATION_CASE, PERIODIC_MUTATION_CASE, PHASED_MUTATION_CASE,
             ARITHMETIC_MUTATION_CASE)):
        findings: List[str] = []
        driver = build_driver(dict(dict(CASES)[case]), seed=20260907)
        try:
            arrangements = {"array": Arrangement("array", None),
                            "weld": arrangement_weld(driver, kernel=None)}
            result = drive(driver, arrangements, steps, stop_on_divergence=False)
        finally:
            driver.close()
        if not result["bit_identical"]:
            findings.append(
                f"the DISARMED harness diverged at step "
                f"{result['first_divergence_step']} on {case}; every CAUGHT verdict "
                f"scored there is unattributable")
        rows.append({"leg": "disarm", "passed": not findings, "findings": findings,
                     "case": case, "bit_identical": result["bit_identical"],
                     "steps_compared": result["steps_compared"]})
    return rows


def policy_refusal_types() -> Tuple[type, ...]:
    """The exception classes the subnormal-policy machinery raises to REFUSE.

    Read off :mod:`meep_gpu.subnormal_policy` rather than spelled here or matched as
    text, so a class renamed there stops being recognised at the seam that
    recognises it. An import that fails recognises NOTHING, which is the strict
    answer: a refusal that cannot be attributed to the policy is an error.
    """
    try:
        from meep_gpu import subnormal_policy  # noqa: PLC0415
    except Exception:  # noqa: BLE001 - no policy module here recognises no refusal
        return ()
    return tuple(
        cls for cls in (getattr(subnormal_policy, name, None)
                        for name in ("SubnormalPolicyUnattainable",
                                     "SubnormalPolicyLocked"))
        if isinstance(cls, type) and issubclass(cls, BaseException))


def leg_mutation(steps: int, seed: int, policy: Optional[str]) -> List[Dict[str, Any]]:
    """Every armed defect, each with its declared outcome and its reason.

    A MUTANT THE POLICY REFUSES TO BUILD IS A RESULT, BUT ONLY A DECLARED ONE. The
    default for a build failure is ``REFUSED TO BUILD``, which fails: a mutation that
    cannot be compiled has measured nothing about the weld. One row declares
    otherwise — ``m_divide_by_the_reciprocal_of_sinv`` under ``flush``, whose planted
    ``div.full.f32`` is the very instruction that policy's PTX audit exists to
    refuse — and that declaration is honoured only when the exception came from the
    policy machinery (:func:`policy_refusal_types`, an isinstance test, never a
    word) and its text carries every needle the row declares. The needles are
    asserted, not logged: a refusal that does not say what it was supposed to say is
    a different refusal and still fails.
    """
    rows: List[Dict[str, Any]] = []
    refusal_types = policy_refusal_types()
    for tag, spec in MUTATIONS.items():
        expected = spec.get("expected_override", spec["expected"])
        by_policy = spec.get("expected_by_policy") or {}
        if policy in by_policy:
            expected = by_policy[policy]
        needles = tuple((spec.get("refusal_needles_by_policy") or {}).get(policy, ()))
        case = spec["case"]
        driver = build_driver(dict(dict(CASES)[case]), seed=seed)
        row: Dict[str, Any] = {"leg": "mutation", "mutation": tag, "case": case,
                               "target": spec["target"], "expected": expected,
                               "why": spec["why"]}
        try:
            kernel = None
            launcher = None
            if spec["target"] == "kernel":
                module = mutated_family(tag, spec["old"], spec["new"])
                kernel = module.fused_complex_constitutive_curl_H_to_D
            elif tag == "m_rotation_skipped":
                launcher = RotationSkipped
            elif tag == "m_constitutive_takes_the_half_integer_lattice":
                launcher = half_integer_rebind(driver.pml)
            elif tag == "m_connection_direction_swapped":
                launcher = connection_direction_swapped(driver)
            elif tag == "m_withdraw_after_step_D":
                row.update({"outcome": "MEASURED IN THE withdraw LEG", "passed": True,
                            "note": "driven there, on a fixture carrying an integrated "
                                    "electric source, with the hoisted arrangement as "
                                    "its positive control"})
                rows.append(row)
                continue
            arrangements = {"array": Arrangement("array", None),
                            "weld": arrangement_weld(driver, kernel=kernel,
                                                     launcher=launcher)}
            result = drive(driver, arrangements, steps, stop_on_divergence=True)
            outcome = "NULL" if result["bit_identical"] else "CAUGHT"
            row.update({"outcome": outcome,
                        "first_divergence_step": result["first_divergence_step"],
                        "steps_compared": result["steps_compared"],
                        "differing_words_at_the_first_divergence": (
                            result["per_step"][-1].get("weld", {}).get(
                                "differing_words") if result["per_step"] else 0),
                        "passed": (outcome == expected or expected == "RECORDED")})
        except Exception as error:  # noqa: BLE001 - a mutant that cannot build
            text = f"{type(error).__name__}: {error}"
            by_the_policy = bool(refusal_types) and isinstance(error, refusal_types)
            missing = [needle for needle in needles if needle not in text]
            accepted = (expected == PTX_AUDIT_REFUSAL and by_the_policy
                        and bool(needles) and not missing)
            # HOISTED OUT OF THE f-STRING BELOW, and the reason is the interpreter
            # this gate runs on rather than taste: a conditional expression spanning
            # LINES inside an f-string replacement field is PEP 701, which lands in
            # CPython 3.12. The device host's campaign interpreter is 3.10.20, where
            # the same text is a SyntaxError at import -- so the gate died in 0 s
            # before measuring anything, on both policies, while parsing cleanly on
            # a 3.12 laptop. A gate that cannot be parsed by the interpreter that
            # runs it is not a stricter gate, it is an unrunnable one.
            how_it_failed = ("a policy refusal" if by_the_policy else
                             "an exception the policy machinery did not raise")
            row.update({
                "outcome": PTX_AUDIT_REFUSAL if accepted else "REFUSED TO BUILD",
                "passed": accepted,
                "error": text[:600],
                # WHY THIS ROW SCORED WHAT IT SCORED, kept in the record so an
                # accepted refusal can be read back without re-running the gate.
                "refused_by": type(error).__name__ if by_the_policy else None,
                "refusal_needles_required": list(needles),
                "refusal_needles_missing": missing,
                "findings": ([] if accepted else
                             [f"{tag}: expected {expected}, and the build failed "
                              f"with {how_it_failed}"
                              + (f" missing {missing}" if missing else "")]),
            })
        finally:
            driver.close()
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# The corpus lift
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path) -> List[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def lift_basis(results: Path) -> Tuple[List[dict], Dict[str, Any]]:
    """The rows the standing census puts in this product's cell, and the seam facts.

    THE ARMS COME FROM THE SEAM RECORD. The census's own ``plan_step.selected`` is empty
    on every row -- it was cut on a NumPy host, where ``coverage._grid_reasons`` clause 1
    refuses every arm -- so a join on it would silently return an EMPTY cell and this
    leg would report a full-length pass over nothing.
    """
    census = results / CENSUS
    seam = results / SEAM_RECORD
    if not census.is_dir() or not seam.is_dir():
        raise SystemExit(f"the lift leg needs {census} and {seam}")
    seam_rows = {row["label"]: row for row in _load_jsonl(seam / "h_to_d_seam.jsonl")}
    facts_by_label: Dict[str, dict] = {}
    for leg in ("examples", "tests", "tests_param_matched"):
        path = census / f"{leg}.jsonl"
        if not path.is_file():
            continue
        for record in _load_jsonl(path):
            facts_by_label[f"{record.get('leg', leg)}:{record['row']}"] = record
    rows: List[dict] = []
    for label, seam_row in seam_rows.items():
        arms = (seam_row.get("arms") or {}).get("triton") or {}
        if (arms.get("update_H"), arms.get("step_D")) != CELL_ARMS:
            continue
        record = facts_by_label.get(label, {})
        rows.append({
            "label": label, "leg": seam_row.get("leg"), "row": seam_row["row"],
            "module": record.get("module"),
            "grid_cells": record.get("grid_cells"),
            "grid_shape": record.get("grid_shape"),
            "withdraw_in_seam": bool((seam_row.get("h_to_d_seam") or {})
                                     .get("withdraw_in_seam")),
            "arms": {"update_H": arms.get("update_H"), "step_D": arms.get("step_D")},
        })
    unique = {row["label"]: row for row in rows}
    facts = {"census": CENSUS, "seam_record": SEAM_RECORD, "cell_arms": list(CELL_ARMS),
             "rows_in_cell": len(unique),
             "rows_with_a_standing_withdraw": sorted(
                 label for label, row in unique.items() if row["withdraw_in_seam"])}
    return list(unique.values()), facts


def _child_progress(message: str) -> None:
    path = os.environ.get("MEEP_GPU_CHD_GATE_PROGRESS")
    label = os.environ.get("MEEP_GPU_CHD_GATE_LABEL", "?")
    line = f"[{time.strftime('%H:%M:%S')}] {label} {message}"
    print(line, flush=True)
    if path:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()


def evaluate_row(driver: Any, steps: int, max_cells: Optional[int]) -> Dict[str, Any]:
    """The multi-reference identity on ONE lifted row."""
    sources = tuple(getattr(driver, "_sources", ()) or ())
    block: Dict[str, Any] = {
        "family": family.FAMILY, "steps_requested": steps,
        "n_sources": len(sources),
        "sources": [{"type": type(s).__name__,
                     "field_type": str(getattr(s, "field_type", "")),
                     "is_integrated": bool(getattr(s, "is_integrated", False)),
                     "withdraw_does_work": bool(
                         withdraw_hoist._withdraw_does_work(s))}  # noqa: SLF001
                    for s in sources]}
    pin_array_path(driver)
    verdict = family.complex_fused_hd_pair_coverage(driver.fields, driver.pml, sources)
    block["predicate_admits"] = bool(verdict.covered)
    block["predicate_reasons"] = list(verdict.reasons)
    composed = _composed(driver, fuse=True)
    block["composer_selected"] = dict(composed.selected)
    cells = int(np.prod(driver.grid.shape))
    block["grid_cells"] = cells
    if not verdict.covered:
        block.update(driven=False, why_not_driven="the predicate refused this row")
        return block
    if max_cells and cells > int(max_cells):
        block.update(driven=False,
                     why_not_driven=(f"{cells} cells exceeds the leg's cap "
                                     f"{max_cells}; refused rather than run partially"))
        return block
    # THE DETERMINISM CONTROL, and it is a MEASUREMENT rather than a list of names. This
    # comparison steps ONE driver through several arrangements in lockstep from one
    # captured seed; a row whose current is drawn fresh on every evaluation gives each
    # arrangement a DIFFERENT source, and byte identity across them is undefined rather
    # than false. Its second arm also wipes every private volume before every step, so
    # an identical result licenses leaving that scratch out of the comparison.
    seed = capture(driver)
    control = drive(driver, {"array": Arrangement("array", None),
                             "array_repeat": Arrangement("array_repeat", None,
                                                         wipe_private=True)},
                    min(steps, 4), stop_on_divergence=True)
    restore(driver, seed)
    pin_array_path(driver)
    block["array_path_repeats_itself"] = bool(control["bit_identical"])
    block["the_private_scratch_carries_nothing_across_a_step"] = bool(
        control["bit_identical"])
    block["determinism_control_steps"] = control["steps_compared"]
    if not control["bit_identical"]:
        block.update(
            driven=False,
            why_not_driven=(
                f"NOT DETERMINISTIC (or the private scratch is load-bearing): two "
                f"array-path runs from one captured seed, one of them wiping every "
                f"private volume before each step, disagree at step "
                f"{control['first_divergence_step']} of {control['steps_compared']}. "
                f"Refused by measurement rather than scored"))
        return block
    block["driven"] = True
    # THE DRIVER'S OWN CHUNK DIVISION for every plan built on this lifted row, the one the
    # array path steps with (hd_seam_restore.plans_bind_the_driver_chunk_layout).
    with hd_seam_restore.plans_bind_the_driver_chunk_layout(driver):
        block.update(run_product(driver, steps, progress=_child_progress,
                                 require_full_budget=True, clean_floor=1))
    return block


def lift_child(leg: str, script: Optional[str], module_path: Optional[str],
               case: Optional[str], out_json: str, steps: int,
               max_cells: Optional[int], policy: Optional[str] = None) -> int:
    """One corpus row: capture the simulation, lift it TO THE DEVICE, drive it.

    THE POLICY IS INSTALLED HERE, IN THIS PROCESS, BEFORE THE FIRST DEVICE COMPILE. The
    parent's install lives on ``CUDABackend`` and on CuPy's compiler front ends in the
    PARENT's memory; a child inherits the environment -- the cache directories -- and
    nothing else. A child that ran native CuPy (``-ftz=true``) against native Triton (no
    ``.ftz``) and was stamped with the parent's policy is the 2026-09-06 defect
    ``results/triton_hd_lift_policy_2026-09-06`` is the autopsy of.
    """
    record: Dict[str, Any] = {"leg": leg, "row": case or Path(script or "").name}
    started = time.time()
    try:
        import cupy  # noqa: PLC0415
        import meep_gpu  # noqa: PLC0415
        from meep_gpu import backends, subnormal_policy  # noqa: PLC0415

        if leg == "examples":
            from parity.meep_gpu.sweep_corpus_lift_parity import (  # noqa: PLC0415
                capture_simulation,
            )

            captured, sim, restore_sim = capture_simulation(script)
            record.update({key: value for key, value in captured.items()
                           if key not in ("row",)})
            restore_sim()
        else:
            from parity.meep_gpu import survey_meep_tests as harness  # noqa: PLC0415

            namespace = harness.build_child_namespace()
            module = namespace["_import_module"](module_path)
            sim = None
            for class_name, method_name in namespace["_enumerate_cases"](module):
                captured, sim_candidate, restore_case, _case = namespace["_run_case"](
                    module, module_path, class_name, method_name, 900.0)
                restore_case()
                if f"{class_name}.{method_name}" == case:
                    record.update({key: value for key, value in captured.items()
                                   if key not in ("row",)})
                    sim = sim_candidate
                    break
        if sim is None:
            # A ROW THIS HOST'S MEEP CANNOT BUILD IS NAMED, NOT SCORED.
            record.update(measured=False, unliftable_on_this_host=True,
                          note="no mp.Simulation to lift on this host's MEEP build")
            _write_child(out_json, record)
            return 0
        if policy is not None:
            subnormal_policy.install_subnormal_policy(policy, cupy=cupy, strict=True)
        record["host_flushing_at_lift"] = bool(backends.subnormals_flushed())
        _child_progress("lifting to the device")
        driver = meep_gpu.lift_simulation(sim, prefer_gpu=True, gpu_id=0)
        record["lift_s"] = round(time.time() - started, 2)
        record["grid_shape"] = [int(v) for v in driver.shape]
        record["grid_cells"] = int(math.prod(int(v) for v in driver.shape))
        try:
            record["triton_complex_fused_hd_pair_gate"] = evaluate_row(
                driver, steps, max_cells)
            record["measured"] = True
        finally:
            with contextlib.suppress(Exception):
                driver.close()
        record["subnormal_policy"] = subnormal_policy.policy_stamp()
    except BaseException as error:  # noqa: BLE001 - a refusal is the result
        import traceback  # noqa: PLC0415

        record.update(measured=False,
                      child_error=f"{type(error).__name__}: {error}"[:800],
                      child_traceback=traceback.format_exc()[-2500:])
    _write_child(out_json, record)
    return 0


def _write_child(out_json: str, record: Dict[str, Any]) -> None:
    gate_provenance.stamp(record)
    Path(out_json).write_text(json.dumps(record, default=str), encoding="utf-8")
    _child_progress(f"measured={record.get('measured')}")


def leg_lift(out_dir: Path, steps: int, max_cells: Optional[int], timeout: float,
             resume: bool, only: Optional[Sequence[str]],
             interpreter: str, policy: Optional[str] = None) -> Dict[str, Any]:
    """Every corpus row in the cell, re-lifted in its own interpreter and driven."""
    import measure_predicate_coverage as census  # noqa: PLC0415

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
                "reason": (f"the MEEP corpus is not at {examples_dir} / {tests_dir}; "
                           f"set MEEP_GPU_CORPUS_ROOT to the checkout the census was "
                           f"cut over. Refused rather than measured on nothing")}
    environment = dict(os.environ)
    environment.update({"KMP_DUPLICATE_LIB_OK": "TRUE", "MPLBACKEND": "Agg",
                        "PYTHONPATH": str(API_ROOT),
                        "MEEP_GPU_CHD_GATE_PROGRESS": str(progress_log)})
    work = lift_dir / "workdir"
    work.mkdir(exist_ok=True)
    for entry in examples_dir.iterdir():
        if entry.suffix in (".py", ".ipynb"):
            continue
        link = work / entry.name
        if not link.exists():
            with contextlib.suppress(OSError):
                link.symlink_to(entry)
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
        record_path = per_row / (row["label"].replace(":", "__").replace(".", "_")
                                 + ".json")
        environment["MEEP_GPU_CHD_GATE_LABEL"] = row["label"]
        environment["PYTHONPATH"] = (f"{shim_path}{os.pathsep}{API_ROOT}"
                                     if row.get("module") in needs_shim
                                     else str(API_ROOT))
        started = time.time()
        if not (resume and record_path.exists()):
            command = [interpreter, "-u", str(Path(__file__).resolve()),
                       "--lift-child", "--lift-child-leg", row["leg"],
                       "--lift-child-out", str(record_path),
                       "--steps", str(steps)]
            if policy is not None:
                command += ["--subnormal-policy", policy]
            if max_cells is not None:
                command += ["--lift-max-cells", str(max_cells)]
            if row["leg"] == "examples":
                command += ["--lift-child-script", str(examples_dir / row["row"])]
            else:
                if not row["module"]:
                    measured.append({**row, "measured": False,
                                     "note": "no module recorded"})
                    log(f"lift {index}/{len(rows)} {row['label']}: REFUSED, no module")
                    continue
                command += ["--lift-child-module", str(tests_dir / row["module"]),
                            "--lift-child-case", row["row"]]
            log(f"lift {index}/{len(rows)} {row['label']} start "
                f"(cells {row.get('grid_cells')})")
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
                died = {"row": row["row"], "measured": False, "note": note,
                        "stderr_tail": stderr_text[-1500:]}
                gate_provenance.stamp(died)
                record_path.write_text(json.dumps(died, default=str), encoding="utf-8")
        record = json.loads(record_path.read_text(encoding="utf-8"))
        record.update({key: value for key, value in row.items() if key != "label"})
        record["label"] = row["label"]
        measured.append(record)
        block = record.get("triton_complex_fused_hd_pair_gate") or {}
        log(f"lift {index}/{len(rows)} {row['label']}: "
            f"measured={record.get('measured')} "
            f"admits={block.get('predicate_admits')} driven={block.get('driven')} "
            f"passed={block.get('passed')} steps={block.get('steps_compared')} "
            f"words={block.get('words_compared')} ({time.time() - started:.1f} s)")
    with (lift_dir / "rows.jsonl").open("w", encoding="utf-8") as handle:
        for record in measured:
            handle.write(json.dumps(record, default=str) + "\n")

    blocks = {r["label"]: (r.get("triton_complex_fused_hd_pair_gate") or {})
              for r in measured}
    unliftable = {record["label"]: str(record.get("error") or record.get("note") or "")
                  for record in measured if record.get("unliftable_on_this_host")}
    # A ROW THE ENGINE'S OWN LIFTER REFUSES BY NAME (MeepSimulationNotLiftable) is
    # nobody's to step either: named and counted out with the rows above.
    refused_by_the_lifter = {record["label"]: reason for record in measured
                             if record["label"] not in unliftable
                             and not (record.get("triton_complex_fused_hd_pair_gate") or {})
                             and (reason := hd_seam_restore.lifter_refusal(record))}
    unliftable.update({label: f"refused by the engine's lifter: {reason}"
                       for label, reason in refused_by_the_lifter.items()})
    blocks = {label: block for label, block in blocks.items()
              if label not in unliftable}
    admitted = sorted(label for label, b in blocks.items() if b.get("predicate_admits"))
    refused = sorted(label for label, b in blocks.items()
                     if b.get("predicate_admits") is False)
    # THE WITHDRAW REFUSALS, DERIVED FROM EACH ROW'S OWN SOURCES (an electric source
    # whose withdraw does work; none since entry 25), never from the predicate's answer.
    accounting = hd_seam_restore.lift_withdraw_accounting(
        blocks, facts["rows_with_a_standing_withdraw"])
    refused_by_the_withdraw = accounting["refused_by_a_standing_withdraw"]
    expected_refused = accounting["expected_refused_from_the_row_sources"]
    unmeasured = sorted(label for label, b in blocks.items()
                        if "predicate_admits" not in b)
    driven = sorted(label for label in admitted if blocks[label].get("driven"))
    passed_rows = sorted(label for label in driven if blocks[label].get("passed"))
    not_driven = {label: blocks[label].get("why_not_driven")
                  for label in admitted if not blocks[label].get("driven")}
    non_deterministic = {label: blocks[label].get("why_not_driven")
                         for label in admitted
                         if blocks[label].get("array_path_repeats_itself") is False}
    not_driven = {label: reason for label, reason in not_driven.items()
                  if label not in non_deterministic}
    diverged = sorted(label for label in driven
                      if not all((blocks[label].get("seam_claim") or {}).values()))
    diverged_against_the_array_path = sorted(
        label for label in driven if not blocks[label].get("bit_identical"))

    def _pair_holds(table: Mapping[str, Any], first: str, second: str) -> bool:
        held = table.get(first)
        return bool(held) and held == table.get(second) and "fused pair" in str(held)

    arbitration: Dict[str, Any] = {"loss": [], "tie": [], "gain": [], "unknown": []}
    for label in driven:
        table = blocks[label].get("composer_selected") or {}
        if not table:
            arbitration["unknown"].append(label)
            continue
        neighbours = (_pair_holds(table, "step_B", "update_H")
                      + _pair_holds(table, "step_D", "update_E"))
        arbitration[("gain", "tie", "loss")[neighbours]].append(label)
    singles_disagree = sorted(
        label for label in driven
        if blocks[label].get("the_certified_singles_agree_with_the_array_path") is False)
    weld_disagrees = sorted(
        label for label in diverged
        if blocks[label].get("weld_agrees_with_the_certified_singles") is False)
    reference_disagrees = {
        label: blocks[label].get("references_that_disagree_with_the_array_path")
        for label in driven
        if blocks[label].get("references_that_disagree_with_the_array_path")}
    seam_alone_disagrees = sorted(
        label for label in driven
        if blocks[label].get("the_seam_alone_agrees_with_the_array_path") is False)
    below = {label: {"steps_compared": blocks[label].get("steps_compared"),
                     "first_banded_step": blocks[label].get("first_banded_step")}
             for label in driven
             if blocks[label].get("bit_identical")
             and int(blocks[label].get("steps_compared") or 0) < LIFT_CLEAN_STEP_FLOOR}
    cleared = sorted(label for label in driven
                     if label not in below and label not in diverged)
    majority_cleared = len(cleared) * 2 >= len(driven) if driven else False
    expected_stamp = ({"keep": "ieee_keep_ftz_stripped",
                       "flush": "meep_x86_flush"}.get(policy) if policy else None)
    stamps = {rec["label"]: (rec.get("subnormal_policy") or {})
              for rec in measured if rec["label"] in driven}
    child_policy_mismatches = {
        label: {"policy": stamp.get("policy"), "installed": stamp.get("installed"),
                "unattained": stamp.get("unattained"), "expected": expected_stamp}
        for label, stamp in stamps.items()
        if policy is not None and (stamp.get("policy") != expected_stamp
                                   or not stamp.get("installed")
                                   or stamp.get("unattained"))}
    children_ran_under_the_policy = (policy is None or (
        bool(driven) and len(stamps) == len(driven) and not child_policy_mismatches))
    passed = bool(measured and not unmeasured
                  and (bool(only)
                       or len(admitted) + len(refused) + len(unliftable)
                       == facts["rows_in_cell"])
                  and len(driven) + len(non_deterministic) == len(admitted)
                  and refused == refused_by_the_withdraw == expected_refused
                  and accounting["passed"]
                  and not not_driven and not diverged
                  and passed_rows == driven
                  and set(driven) | set(non_deterministic) == set(admitted)
                  and majority_cleared
                  and children_ran_under_the_policy)
    return {
        "passed": passed,
        "facts": facts,
        "rows_in_cell": facts["rows_in_cell"],
        "rows_unliftable_on_this_host": unliftable,
        "rows_this_host_could_lift": len(blocks),
        "the_denominator": (
            f"{len(blocks)} of the cell's {facts['rows_in_cell']} rows; the other "
            f"{len(unliftable)} are NAMED in rows_unliftable_on_this_host with the "
            f"error MEEP raised before any Simulation existed on this build"),
        "rows_measured": len(measured),
        "rows_unmeasured": unmeasured,
        "admitted": len(admitted), "admitted_rows": admitted,
        "refused": refused,
        "refused_by_the_standing_withdraw": refused_by_the_withdraw,
        "refused_expected_from_the_row_sources": expected_refused,
        "refusal_accounting": accounting,
        "rows_refused_by_the_lifter": refused_by_the_lifter,
        "driven": len(driven), "not_driven": not_driven,
        "rows_refused_as_non_deterministic": non_deterministic,
        "rows_bit_identical": len(passed_rows),
        "rows_that_diverged": diverged,
        "rows_where_ANY_arrangement_disagrees_with_the_array_path":
            diverged_against_the_array_path,
        "rows_where_the_CERTIFIED_SINGLES_disagree_with_the_array_path":
            singles_disagree,
        "arbitration_over_the_driven_rows": {
            key: sorted(value) for key, value in arbitration.items()},
        "arbitration_counts": {key: len(value) for key, value in arbitration.items()},
        "what_the_arbitration_counts_mean": (
            "installing a TWO-SLOT H->D product on each driven row, priced off that "
            "row's OWN composer slot table: LOSS where both neighbouring pairs install "
            "(2 pairs -> 1), TIE where exactly one does, GAIN where NEITHER does "
            "(4 launches / 0 seams -> 3 / 1). It prices the COMPOSITION and says "
            "nothing about the arithmetic, which the seam claim measures; and it "
            "prices a two-slot span, so it says nothing about the four-slot "
            "step_B -> update_H -> step_D -> update_E weld, which is the only span "
            "that is strictly additive and is not built"),
        "child_subnormal_policy": {
            "requested": policy, "expected_stamp": expected_stamp,
            "every_driven_child_installed_and_attained_it": children_ran_under_the_policy,
            "mismatches": child_policy_mismatches,
            "stamps_per_row": stamps},
        "seam_claim_per_row": {label: blocks[label].get("seam_claim")
                               for label in driven},
        "rows_where_the_weld_disagrees_with_the_certified_singles": weld_disagrees,
        "rows_where_the_SEAM_ALONE_disagrees_with_the_array_path": seam_alone_disagrees,
        "rows_where_a_reference_disagrees_with_the_array_path": reference_disagrees,
        "first_banded_step_per_row": {label: blocks[label].get("first_banded_step")
                                      for label in driven},
        "rows_below_the_clean_step_floor": below,
        "rows_that_cleared_the_clean_step_floor": cleared,
        "majority_cleared_the_floor": majority_cleared,
        "clean_step_floor": LIFT_CLEAN_STEP_FLOOR,
        "complete_driver_steps_compared": int(sum(
            blocks[label].get("steps_compared") or 0 for label in driven)),
        "words_compared": int(sum(blocks[label].get("words_compared", 0)
                                  for label in driven)),
        "modules_lifted_with_the_parameterized_shim": sorted(needs_shim),
        "steps": steps,
        "rows_jsonl": str(lift_dir / "rows.jsonl"),
    }


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes() -> Dict[str, str]:
    return {name: sha256(API_ROOT / name) for name in SOURCES
            if (API_ROOT / name).is_file()}


def emit(handle: Any, row: Dict[str, Any]) -> None:
    handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def parse_legs(value: str) -> Tuple[str, ...]:
    if value in ("all", ""):
        return ALL_LEGS
    if value in LEG_GROUPS:
        return LEG_GROUPS[value]
    wanted = tuple(name.strip() for name in value.split(",") if name.strip())
    unknown = [name for name in wanted if name not in ALL_LEGS]
    if unknown:
        raise SystemExit(f"unknown leg(s) {unknown}; known: {ALL_LEGS}")
    return wanted


def verdict(rows: Sequence[Dict[str, Any]], legs: Sequence[str],
            policy_attained: bool) -> Dict[str, Any]:
    by_leg: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_leg.setdefault(row["leg"], []).append(row)
    clauses: Dict[str, Any] = {}
    for leg in legs:
        entries = by_leg.get(leg, [])
        clauses[f"{leg}_ran"] = bool(entries)
        clauses[f"{leg}_passes"] = bool(entries) and all(
            entry.get("passed") for entry in entries)
    if "product" in legs:
        clauses["product_ran_every_fixture"] = (
            {row["case"] for row in by_leg.get("product", ())}
            == {name for name, _ in CASES})
        clauses["every_product_case_moved_the_reference"] = all(
            row.get("reference_moved_words_step_1", 0) > 0
            for row in by_leg.get("product", ()))
        clauses["every_product_case_ran_the_full_budget"] = all(
            row.get("steps_compared") == row.get("steps_requested")
            for row in by_leg.get("product", ()))
        # ALL SIX REFERENCES under either policy, asserted rather than relaxed.
        expected = set(REFERENCE_MODES)
        clauses["every_product_case_carried_the_references_this_policy_allows"] = all(
            set(row.get("launches", {})) == expected
            for row in by_leg.get("product", ()))
        clauses["the_planted_tiny_normal_case_ran"] = any(
            row.get("case") == ARITHMETIC_MUTATION_CASE and row.get("planted_words")
            for row in by_leg.get("product", ()))
    if "mutation" in legs:
        armed = [row for row in by_leg.get("mutation", ())
                 if row.get("expected") == "CAUGHT"]
        clauses["every_armed_mutation_is_caught"] = bool(armed) and all(
            row.get("outcome") in ("CAUGHT", "MEASURED IN THE withdraw LEG")
            for row in armed)
        clauses["the_stale_read_plant_is_CAUGHT"] = any(
            row["mutation"] == "m_foreign_tap_reads_the_stale_value"
            and row.get("outcome") == "CAUGHT"
            for row in by_leg.get("mutation", ()))
        # THE DIVIDE SPELLING DOES NOT SURVIVE, AND WHICH WAY IT DIES IS THE POLICY'S
        # ANSWER, NOT A RELAXATION. Under `keep` the mutant compiles and the bytes
        # diverge: CAUGHT. Under `flush` the mutant never runs, because the
        # `div.full.f32` it plants is exactly what that policy's PTX audit refuses —
        # a stronger answer, from the shipped machinery, and the reason this leg
        # cost the flush release 3 of 42 clauses until it was declared. The refusal
        # is accepted only where the ROW recorded it as accepted, which is where the
        # exception came from the policy machinery AND carried its needles, so this
        # clause cannot be satisfied by any other build failure.
        clauses["the_divide_spelling_does_not_survive_this_policy"] = any(
            row["mutation"] == "m_divide_by_the_reciprocal_of_sinv"
            and row.get("outcome") in ("CAUGHT", PTX_AUDIT_REFUSAL)
            and row.get("passed")
            for row in by_leg.get("mutation", ()))
        clauses["every_predicted_null_is_recorded_with_its_reason"] = all(
            row.get("why") for row in by_leg.get("mutation", ())
            if row.get("expected") == "NULL")
    clauses["the_policy_installed_is_the_one_requested"] = bool(policy_attained)
    clauses["every_leg_requested_ran"] = set(legs) <= set(by_leg)
    clauses["the_whole_gate_ran"] = set(legs) == set(ALL_LEGS)
    return {"clauses": clauses, "released": all(clauses.values()),
            "legs_requested": list(legs),
            "legs_that_ran": sorted(by_leg)}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=None)
    parser.add_argument("--no-device", action="store_true")
    parser.add_argument("--legs", default="all")
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--subnormal-policy", default=None,
                        choices=(None, "keep", "flush"))
    parser.add_argument("--import-meep-for-host-policy", action="store_true")
    parser.add_argument("--lift-steps", type=int, default=None)
    parser.add_argument("--lift-max-cells", type=int, default=None)
    parser.add_argument("--lift-timeout", type=float, default=2400.0)
    parser.add_argument("--lift-resume", action="store_true")
    parser.add_argument("--lift-only", default=None)
    parser.add_argument("--lift-interpreter", default=sys.executable)
    parser.add_argument("--lift-child", action="store_true")
    parser.add_argument("--lift-child-leg", default="examples")
    parser.add_argument("--lift-child-script", default=None)
    parser.add_argument("--lift-child-module", default=None)
    parser.add_argument("--lift-child-case", default=None)
    parser.add_argument("--lift-child-out", default=None)
    args = parser.parse_args(argv)

    if args.lift_child:
        return lift_child(args.lift_child_leg, args.lift_child_script,
                          args.lift_child_module, args.lift_child_case,
                          args.lift_child_out, args.steps, args.lift_max_cells,
                          policy=args.subnormal_policy)

    legs = parse_legs(args.legs)
    if args.no_device:
        legs = tuple(leg for leg in legs if leg in LEG_GROUPS["host"])
    started = time.time()
    rows: List[Dict[str, Any]] = []
    payload: Dict[str, Any] = {
        "gate": GATE, "family": family.FAMILY,
        "seed": args.seed, "steps": args.steps,
        "legs_requested": list(legs),
        "numpy_version": np.__version__,
        "subnormal_policy_requested": args.subnormal_policy,
        "cases": [name for name, _ in CASES],
        "cell_arms": list(CELL_ARMS),
        "source_sha256": source_hashes(),
        "installable": family.INSTALLABLE,
        "installable_reason": family.INSTALLABLE_REASON,
        "divide_facts": [dict(fact) for fact in family.DIVIDE_FACTS],
        "what_served_means_on_a_board": (
            "PREDICATE ADMISSION, by the boards' own definition. A released product "
            "credits the seam-instances its predicate admits while executing NOWHERE: "
            "this product is in no fastpath.RELEASED_FUSED_ARMS envelope, fastpath "
            "never plans it, and the composer is not even offered it"),
        "the_composition_installed_reference_awaits_wiring": (
            "this product is UNREGISTERED: launch.CERTIFIED_FUSED_PRODUCTS carries no "
            "row for it, so nothing the composer builds can be compared against it as "
            "an INSTALLED product. The gate builds the plan itself, force-installs it "
            "at the seam's two slots, and drives the ARRAY PATH as the oracle; the "
            "composition references are the composer's own plans on the OTHER slots. "
            "What awaits wiring is a composition gate that reaches this product "
            "through plan_step's own installation"),
        "rows": rows,
    }
    policy_attained = True
    if not args.no_device:
        triton_launch.require_triton()
        import cupy  # noqa: PLC0415

        if args.import_meep_for_host_policy:
            import probe_fused_kernel_bit_identity as bit_identity  # noqa: PLC0415

            payload["meep_host_import"] = bit_identity.import_meep_for_host_policy()
        from meep_gpu import subnormal_policy  # noqa: PLC0415

        # BEFORE THE FIRST DEVICE COMPILE. strict=True: a process that asked to keep
        # and quietly did not is a process whose bytes mean nothing.
        subnormal_policy.install_subnormal_policy(args.subnormal_policy, cupy=cupy,
                                                  strict=True)
        payload["subnormal_policy"] = subnormal_policy.policy_stamp()
        policy_attained = bool(payload["subnormal_policy"].get("attained", True))
        import triton  # noqa: PLC0415

        payload.update({
            "cupy_version": cupy.__version__, "triton_version": triton.__version__,
            "device": str(cupy.cuda.runtime.getDeviceProperties(
                cupy.cuda.runtime.getDevice())["name"]),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cupy_cache_dir": os.environ.get("CUPY_CACHE_DIR"),
            "triton_cache_dir": os.environ.get("TRITON_CACHE_DIR"),
            "references_this_policy_can_build": list(REFERENCE_MODES),
        })
        log(f"subnormal policy installed: "
            f"{payload['subnormal_policy'].get('policy')!r} "
            f"(requested {args.subnormal_policy!r}, "
            f"CUPY_CACHE_DIR={os.environ.get('CUPY_CACHE_DIR')!r})")

    out_path = Path(args.out).resolve() if args.out else None
    stream = (out_path.with_suffix(".rows.jsonl").open("w", encoding="utf-8")
              if out_path else None)

    def record(leg: str, result: Any) -> None:
        entries = result if isinstance(result, list) else [result]
        for entry in entries:
            entry.setdefault("leg", leg)
            entry["elapsed_s"] = round(time.time() - started, 1)
            rows.append(entry)
            if stream is not None:
                emit(stream, entry)
            log(f"{leg:20s} {entry.get('case', entry.get('mutation', '')):38s} "
                f"{'PASS' if entry.get('passed') else 'FAIL'} "
                f"({entry['elapsed_s']} s)")
            for finding in entry.get("findings", []):
                log(f"    ! {finding}")

    try:
        if "driver_order" in legs:
            record("driver_order", leg_driver_order())
        if "transcription" in legs:
            record("transcription", leg_transcription())
        if "refusal" in legs:
            record("refusal", leg_refusal())
        if "arbitration" in legs:
            record("arbitration", leg_arbitration())
        if "purity_ledger" in legs:
            record("purity_ledger", leg_purity_ledger())
        if "ghost_observability" in legs:
            record("ghost_observability", leg_ghost_observability())
        if "product" in legs:
            record("product", leg_product(args.steps, args.seed,
                                          progress=lambda m: log(f"    {m}")))
        if "seed_scale" in legs:
            record("seed_scale", leg_seed_scale(args.steps, args.seed))
        if "launch_structure" in legs:
            record("launch_structure", leg_launch_structure(args.steps, args.seed))
        if "sync" in legs:
            record("sync", leg_sync(args.steps, args.seed))
        if "withdraw" in legs:
            record("withdraw", leg_withdraw(args.steps, args.seed))
        if "byte_neutral" in legs:
            record("byte_neutral", leg_byte_neutral(args.steps, args.seed))
        if "mutation" in legs:
            record("mutation", leg_mutation(args.steps, args.seed,
                                            args.subnormal_policy))
        if "disarm" in legs:
            record("disarm", leg_disarm(args.steps, args.seed))
        if "lift" in legs:
            record("lift", leg_lift(
                out_path.parent if out_path else Path.cwd(),
                args.lift_steps or args.steps, args.lift_max_cells,
                args.lift_timeout, args.lift_resume,
                (args.lift_only.split(",") if args.lift_only else None),
                args.lift_interpreter, policy=args.subnormal_policy))
    finally:
        if stream is not None:
            stream.close()

    payload["release"] = verdict(rows, legs, policy_attained)
    # THE SHAPE THE BOARD'S RELEASE BINDING READS. `build_triton_fusion_matrix`'s
    # GATE_BOUND route requires all three of `device_status == "RUN"`,
    # `release.released` and `passed`, so a `--no-device` or host-only run cannot be
    # read as a release.
    payload["device_status"] = ("RUN" if any(row["leg"] in LEG_GROUPS["device"]
                                             for row in rows) else "NOT_RUN")
    payload["passed"] = bool(payload["release"]["released"])
    payload["seconds"] = round(time.time() - started, 1)
    gate_provenance.stamp(payload)
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False,
                                       default=str), encoding="utf-8")
        log(f"wrote {out_path}")
    log(f"RELEASED={payload['release']['released']} ({payload['seconds']} s)")
    for name, value in sorted(payload["release"]["clauses"].items()):
        if not value:
            log(f"  clause FAILED: {name}")
    if not payload["release"]["released"]:
        return 1 if set(legs) == set(ALL_LEGS) else EXIT_INCOMPLETE
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
