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

**THE EXPANSION ARM IS A MEASURED PLATFORM FACT AND THE LICENCE IS POLICY-CONDITIONAL.**
Every complex arm in this package was certified under ``keep``
(``complex_fields.CERTIFIED_UNDER_SUBNORMAL_POLICY``). Under ``keep`` the predicate
answers, the composer selects the complex arms, and all six reference arrangements are
built. Under ``flush`` the predicate REFUSES by the certification clause, the composer
selects nothing complex on these rows, and the two composition references do not exist:
what the flush record measures is this product's arithmetic FROM ARRAYS with the arm
forced to the one the keep-cut probe licenses, against the array path and the two
certified singles built the same way. ``composer_reachable_under_this_policy`` in the
record says which reading applies, and the release clauses are policy-aware rather than
silently weaker.

**FOUR ARITHMETIC SPELLINGS ARE ARMED AS MUTATIONS RATHER THAN REDISCOVERED**, each
measured elsewhere in this package and each a spelling a careful reader would reach for:
the zero cross terms folded to a literal ``0.0``; unary ``-`` in place of ``* -1.0``
(Triton lowers ``-x`` as ``0.0 - x``, which loses an addend's zero sign); the two
multiply ORIENTATIONS swapped (which factor the FMA fuses differs between
``_mul_field_left`` and ``_mul_coefficient_left``); and the ``NAIVE`` expansion arm in
place of the licensed one. Each is planted by REDEFINING the helper inside a mutated
copy of the family module, so the defect reaches the launch through the shipped
planner's own ``kernel=`` door.

**A COMPLEX-BY-REAL DIVIDE DOES NOT OCCUR IN THIS KERNEL, AND THAT IS MEASURED THREE
WAYS RATHER THAN CITED.** The neighbouring Dcyl product had to settle one (CuPy's
``complex64 / float32`` is the SCALED complex-by-complex algorithm, not numpy's
reciprocal multiply; Triton's ``/`` is ``div.full.f32`` and a correctly rounded quotient
needs ``tl.math.div_rn``) and its verdict does not travel. Here: ``divide_free_evidence``
counts zero true divisions over the parse tree of all three device functions, the laptop
suite plants a divide and a ``div_rn`` and requires both counts to move, and this gate
arms ``m_divide_by_the_reciprocal_of_sinv`` -- one ``_mul_field_left(z, si)`` respelled
as a componentwise divide -- and requires the divergence. So the fact is re-measured on
this kernel.

**THE ROW-ZERO TINY-NORMAL CASE IS PLANTED, NOT HOPED FOR.** The Dcyl round measured
that the wrong multiply arrangement is byte-visible only where a product underflows or
is a signed zero, and that a random battery did not reach that class at all
(0 of 14,387,104 words with the naive spelling). ``planted_row0_tiny`` seeds row 0 of
every stored volume with tiny normals of both signs beside exact zeros, and the multiply
mutations are scored THERE as well as on the random battery.

**THE BLOCH PHASE IS ITS OWN PAIR OF MUTATIONS.** The certified curl rotates the LOADED
register on the wrapped lane only; this weld redirects the load and leaves that block
standing. ``m_phase_applied_inside_the_recompute`` moves the rotation into the tap
(which rotates every unwrapped lane too) and ``m_phase_not_conjugated`` drops the
``step_D`` conjugation; both are scored on a PHASED fixture, because on an unphased one
the ``PH*`` constexpr compiles the whole block away and neither edit can fire.

LEGS
  host legs (no device launch)
  driver_order       REPLACES is exactly the driver's two adjacent consults, the only
                     statement between them is the electric withdraw loop, and the
                     sync channel's containment rule excludes step_D
  transcription      both halves are the certified COMPLEX kernels' own text: the
                     constitutive tail differs in EXACTLY the declared lift edits, the
                     curl tail in EXACTLY the twelve redirected magnetic word loads,
                     every tap's cell and guard is PARSED from the emitter's own lines,
                     no g/e pointer survives, the divide count is zero, the multiply
                     helper set is the base three, and every mutation needle resolves
                     once
  refusal            by name: an undeclared source list; a standing INTEGRATED electric
                     withdraw; a non-integrated electric source NOT refused; a magnetic
                     source NOT refused; a folded grid naming the folded-complex cells;
                     a cylindrical grid; REAL storage; the expansion-probe clause
  arbitration        the composer, asked, with this product's row INJECTED IN PROCESS
  purity_ledger      per fixture, on the array path: of the curl's valid foreign taps,
                     how many land on a cell complex update_H MOVED
  ghost_observability the metallic ghost's unobservability, DERIVED from the certified
                     text, which is what makes one ghost mutation a recorded null
  device legs
  product            the identity on every boundary triple, three phased fixtures and
                     the planted tiny-normal case, complete driver steps
  lift               every corpus row the standing census puts in the cell, re-lifted
                     in its own interpreter which INSTALLS THE SAME SUBNORMAL POLICY
  launch_structure   launches per step at the seam and over the whole step, two
                     independent counters
  sync               the product FORCE-installed; a plan spanning step_D must DECLINE
  withdraw           an integrated electric source in the seam: hoisted identical, the
                     un-hoisted launch and the after_step_D placement both diverge
  seed_scale         the 2^80 seed scale is a change of exponent and nothing else
  byte_neutral       the own-cell register pair replaced by a reload of the scratch
  mutation           the armed kernel and host defects, each with its declared outcome
  disarm             the identical harness, shipped bytes, must not diverge

Rule 7: one flushed line per case; every row appended and fsynced as it lands.

    MEEP_GPU_COMPLEX_EXPANSION_PROBE=<keep-cut probe.json> \\
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
)

#: The case the structural mutations are armed on: every axis walled, so every line the
#: shipped kernel can emit is present and the metallic ghost rule is live on every axis.
MUTATION_CASE = "bc_mmm"

#: THE SECOND MUTATION CASE. On a metallic axis ``stepping._mask_non_owned_cells``
#: zeroes the very curl component the shifted tap feeds, at the very plane where the
#: ghost is served, so no metallic BOUNDARY VALUE is observable there. On an
#: all-periodic grid that mask does not fire and cell 0's curl is a live stepped value
#: assembled from the WRAPPED row.
PERIODIC_MUTATION_CASE = "bc_ppp"

#: THE PHASED MUTATION CASE. ``PHX``/``PHY``/``PHZ`` are constexprs: on an unphased
#: fixture the rotation block is not compiled at all and a phase mutation is a needle in
#: dead text.
PHASED_MUTATION_CASE = "bloch_xyz_edge"

#: THE ARITHMETIC MUTATION CASE. Row 0 carries tiny normals of both signs beside exact
#: zeros, which is the only class in which the wrong complex-multiply arrangement and
#: the folded zero cross terms are byte-visible.
ARITHMETIC_MUTATION_CASE = "planted_row0_tiny"

#: The reference engines, in the order the record reports them. ``weld_seam_only`` is
#: the DISCRIMINATING arrangement: every other arrangement that dispatches ``update_E``
#: carries that sub-step's own agreement with the array path into the comparison.
MODES_UNDER_KEEP: Tuple[str, ...] = ("array", "singles", "composition_today", "unfused",
                                     "weld", "weld_seam_only")

#: Under ``flush`` the predicate refuses by the certification clause and the composer
#: selects nothing complex, so the two composition references DO NOT EXIST. They are
#: named as absent rather than quietly dropped.
MODES_UNDER_FLUSH: Tuple[str, ...] = ("array", "singles", "weld", "weld_seam_only")

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
# The expansion arm
# ---------------------------------------------------------------------------

def _probe_record() -> Tuple[Optional[str], Any]:
    path = os.environ.get(complex_fields.PROBE_PATH_ENVIRONMENT)
    if not path:
        return None, None
    return path, complex_fields.load_expansion_probe(path)


def forced_expansion() -> Dict[str, Any]:
    """The arm the KEEP-CUT probe licenses, read without the policy clause.

    Used only where the policy in force refuses the licence outright -- under ``flush``,
    where the composer selects nothing complex on these rows. Binding it from arrays is
    what lets the flush record measure the product's ARITHMETIC at all; what it does not
    do is license the product under that policy, and the record says so.
    """
    path, record = _probe_record()
    if record is None:
        raise SystemExit(
            f"this gate needs {complex_fields.PROBE_PATH_ENVIRONMENT} to name a "
            f"keep-cut expansion probe artifact; none is readable at {path!r}")
    licence = complex_fields.expansion_license(record)
    if licence["expansion"] is None:
        raise SystemExit(f"the probe artifact licenses no arm: {licence['refusals']}")
    return {"probe": path, "expansion": int(licence["expansion"]),
            "arm": licence.get("arm"), "basis": licence.get("basis"),
            "what_forced_means": (
                "the arm the keep-cut probe licenses, bound FROM ARRAYS without the "
                "predicate, because under this policy the predicate refuses by the "
                "certification clause and the composer selects nothing complex")}


def resolved_expansion(forced: bool) -> int:
    if forced:
        return int(forced_expansion()["expansion"])
    code = complex_fields._resolve_expansion(None)  # noqa: SLF001
    if code is None:
        raise RuntimeError(
            "no EXPANSION arm is licensed under the policy in force: "
            + "; ".join(complex_fields._expansion_reasons(None)))  # noqa: SLF001
    return int(code)


def policy_admits_the_complex_arms() -> bool:
    """Does the policy IN FORCE license the complex tranche at all?

    Read off the certified module's own clause rather than off the
    ``--subnormal-policy`` flag: the flag is what was asked for and this is what the
    process attained.

    THE ARGUMENT IS THE POLICY, NOT ``None``, AND THAT COST A DEAD FLUSH LEG.
    ``expansion_certification_reasons(None)`` ABSTAINS by design -- its own docstring
    says artifact-reading tooling that binds no arm is asking nothing -- so passing
    ``None`` made this predicate answer True under EVERY policy. Under ``keep`` that
    was right by luck; under ``flush`` it sent ``build_weld`` down the predicate route,
    which refused by the certification clause and killed the leg at its first fixture
    (2026-09-07, the first flush launch of this gate). The policy in force is resolved
    the same way the certified module resolves it, so an installed policy outranks a
    declaration exactly as it does at every other seam.
    """
    return not complex_fields.expansion_certification_reasons(
        complex_fields._policy_in_force())  # noqa: SLF001


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

    ``forced`` bypasses the predicate and binds the keep-cut arm from arrays; the
    withdraw leg's fixture is refused BY DESIGN (a standing integrated electric source)
    and takes the same route so the hoist and its two null controls can be driven at
    all. Which route a case took rides in the record.
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
    inputs = _from_arrays_inputs(driver)
    return family.plan_complex_fused_hd_pair_from_arrays(
        inputs["arrays"], {**inputs["curl_flat"],
                           **{f"kps_{axis}": inputs["constitutive_flat"][f"kps_{axis}"]
                              for axis in "xyz"}},
        inputs["dtdx"], inputs["codes"], inputs["phases"],
        resolved_expansion(forced=forced), driver.fields, kernel=kernel)


def build_singles(driver: Any, forced: bool = False) -> Tuple[Any, Any]:
    """The two certified complex halves, at the seam's two slots."""
    fields, pml = driver.fields, driver.pml
    if not forced:
        constitutive = complex_fields.plan_complex_constitutive(fields, pml, "H")
        curl = complex_fields.plan_complex_pml_curl(fields, pml, "step_D")
        if constitutive is not None and curl is not None:
            return constitutive, curl
    inputs = _from_arrays_inputs(driver)
    expansion = resolved_expansion(forced=True)
    side = tcoverage.CONSTITUTIVE_SIDES["H"]
    h_arrays = {name: getattr(fields, name)
                for name in (tuple(side["targets"]) + tuple(side["aux"])
                             + tuple(side["sources"]))}
    constitutive = complex_fields.plan_complex_constitutive_from_arrays(
        "H", h_arrays, inputs["constitutive_flat"], expansion)
    spec = triton_launch.SUB_STEPS["step_D"]
    d_arrays = {name: getattr(fields, name)
                for name in tuple(spec["targets"]) + tuple(spec["sources"])}
    d_arrays.update({"fu_" + name: getattr(fields, "fu_" + name)
                     for name in spec["targets"]})
    curl = complex_fields.plan_complex_pml_curl_from_arrays(
        "step_D", d_arrays, inputs["curl_flat"], inputs["codes"], inputs["phases"],
        inputs["dtdx"], expansion)
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
        "forced_expansion_route": bool(forced),
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
            "under a policy that does not license the complex expansion arm the "
            "composer selects nothing complex on this seam, so composition_today and "
            "unfused would compare the ARRAY PATH WITH ITSELF and report a green zero. "
            "They are named as absent and the seam claim rests on the certified "
            "singles, which are built from arrays with the keep-cut arm forced"),
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
    if not any("withdraw" in line for line in statements):
        findings.append("no withdraw statement stands between the two consults")
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
    if helpers != set(family.MULTIPLY_HELPERS):
        findings.append(
            f"the welded text reaches {sorted(helpers)}, not the declared "
            f"{sorted(family.MULTIPLY_HELPERS)}; a fourth operand orientation needs "
            f"its own probe pattern and the licence would not have measured it")
    if family.PROBE_PATTERNS != complex_fields.PROBE_PATTERNS:
        findings.append("the product's pattern set is not the certified module's base "
                        "four")
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
                         or leaf.startswith("_div_")):
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
    """A stand-in the predicate's withdraw clause reads exactly as it reads a real one."""
    from meep_gpu import sources as sources_module  # noqa: PLC0415

    class _Stand:
        withdraw = sources_module.VolumeSource.withdraw

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
    admits = policy_admits_the_complex_arms()

    def check(name: str, fields: Any, pml: Any, sources: Any, must_refuse: bool,
              needle: Optional[str] = None) -> None:
        verdict = family.complex_fused_hd_pair_coverage(fields, pml, sources)
        reasons = list(verdict.reasons)
        rows.append({"case": name, "covered": verdict.covered, "reasons": reasons})
        if must_refuse and verdict.covered:
            findings.append(f"{name}: admitted, and must be refused")
        # UNDER A POLICY THAT REFUSES THE COMPLEX LICENCE every configuration is
        # refused by that clause, so an ADMIT expectation is not evaluable and is
        # recorded rather than scored. The refusals still are.
        if not must_refuse and not verdict.covered and admits:
            findings.append(f"{name}: refused, and must be admitted ({reasons})")
        if needle is not None and not any(needle in reason for reason in reasons):
            findings.append(f"{name}: no refusal names {needle!r}; got {reasons}")

    driver = build_driver(dict(CASES[0][1]), seed=1)
    fields, pml = driver.fields, driver.pml
    # 1. The source set is not declared: ignorance is never an empty set.
    check("undeclared_source_list", fields, pml, None, True, "was not declared")
    # 2. No sources at all: admitted (under a policy that licenses the arm).
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
    # 10. THE EXPANSION CLAUSE, driven with a probe this predicate must refuse.
    refused_probe = family.complex_fused_hd_pair_coverage(
        fields, pml, (), probe={"backend": "cupy", "patterns": {}})
    rows.append({"case": "non_discriminating_probe",
                 "covered": refused_probe.covered,
                 "reasons": list(refused_probe.reasons)})
    if refused_probe.covered:
        findings.append("a probe artifact that classifies no pattern was ADMITTED; the "
                        "EXPANSION constexpr is then a guess")
    driver.close()
    return {"passed": not findings, "findings": findings, "cases": rows,
            "the_policy_licenses_the_complex_arms": admits,
            "what_an_admit_expectation_means_here": (
                "under a policy that refuses the complex expansion licence every "
                "configuration is refused by that clause, so this leg scores the "
                "REFUSALS and records the admits without scoring them")}


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
                "under a policy that licenses the arm the released B->H pair may or may "
                "not install, and wiring inverts exactly those. What this leg ASSERTS "
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
CURL_TERMS = ("t0_re", "t1_re", "t2_re")


def ghost_observability() -> Dict[str, Any]:
    """Does any metallic BOUNDARY VALUE reach complex ``step_D``'s output?

    THE ANSWER IS NO, AND IT IS DERIVED. Each shifted tap feeds one curl component
    (read off the three ``tN_re = ((...))`` terms, which is where the complex emitter
    assembles the stencil) and its ghost fires on the plane of the axis it shifts
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
        for index, term in enumerate(CURL_TERMS):
            if not stripped.startswith(f"{term} = "):
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

#: The shipped import block the arithmetic mutations SHADOW. Replacing it with local
#: ``triton.jit`` definitions is what lets a defect in a helper this module imports
#: reach the launch through the shipped planner's own ``kernel=`` door -- without
#: touching the certified module, which every other product also imports.
HELPER_IMPORT = ("    from .complex_fields import (  # noqa: PLC0415\n"
                 "        _mul_coefficient_left, _mul_field_left, _rotate_field_left,\n"
                 "    )\n")


def _shadowed_helpers(*, coefficient_left: str, field_left: str,
                      rotate: str) -> str:
    """The import block replaced by local definitions with the given bodies."""
    return (
        "    from .complex_fields import (  # noqa: PLC0415\n"
        "        _mul_coefficient_left as _certified_coefficient_left,\n"
        "        _mul_field_left as _certified_field_left,\n"
        "        _rotate_field_left as _certified_rotate,\n"
        "    )\n"
        "\n"
        "    @triton.jit\n"
        "    def _mul_coefficient_left(c, z_re, z_im, EXPANSION: tl.constexpr):\n"
        f"{coefficient_left}"
        "        return out_re, out_im\n"
        "\n"
        "    @triton.jit\n"
        "    def _mul_field_left(z_re, z_im, c, EXPANSION: tl.constexpr):\n"
        f"{field_left}"
        "        return out_re, out_im\n"
        "\n"
        "    @triton.jit\n"
        "    def _rotate_field_left(g_re, g_im, p_re, p_im, "
        "EXPANSION: tl.constexpr):\n"
        f"{rotate}"
        "        return out_re, out_im\n")


#: The SHIPPED bodies, respelled as local definitions. Every arithmetic mutation is
#: this text with exactly one spelling changed, so the mutation is the spelling and not
#: the shadowing.
_SHIPPED_COEFFICIENT_LEFT = (
    "        if EXPANSION == 1:\n"
    "            out_re = tl.math.fma(c, z_re, (0.0 * z_im) * -1.0)\n"
    "            out_im = tl.math.fma(c, z_im, 0.0 * z_re)\n"
    "        else:\n"
    "            out_re = (c * z_re) - (0.0 * z_im)\n"
    "            out_im = (c * z_im) + (0.0 * z_re)\n")
_SHIPPED_FIELD_LEFT = (
    "        if EXPANSION == 1:\n"
    "            out_re = tl.math.fma(z_re, c, (z_im * 0.0) * -1.0)\n"
    "            out_im = tl.math.fma(z_re, 0.0, z_im * c)\n"
    "        else:\n"
    "            out_re = (z_re * c) - (z_im * 0.0)\n"
    "            out_im = (z_re * 0.0) + (z_im * c)\n")
_SHIPPED_ROTATE = (
    "        if EXPANSION == 1:\n"
    "            out_re = tl.math.fma(g_re, p_re, (g_im * p_im) * -1.0)\n"
    "            out_im = tl.math.fma(g_re, p_im, g_im * p_re)\n"
    "        else:\n"
    "            out_re = (g_re * p_re) - (g_im * p_im)\n"
    "            out_im = (g_re * p_im) + (g_im * p_re)\n")


def _helpers(**overrides: str) -> str:
    bodies = {"coefficient_left": _SHIPPED_COEFFICIENT_LEFT,
              "field_left": _SHIPPED_FIELD_LEFT, "rotate": _SHIPPED_ROTATE}
    bodies.update(overrides)
    return _shadowed_helpers(**bodies)


#: THE TWO ZERO-SIGN EDITS, SPELLED ONCE. Each is scored TWICE -- on the planted
#: row-0 fixture, where the class they move is reachable, and on the random battery,
#: where it is not -- and two spellings of one edit would let the pair drift into
#: measuring two different things.
MUTATIONS_ZERO_CROSS_OLD = HELPER_IMPORT
MUTATIONS_ZERO_CROSS_NEW = _helpers(
    coefficient_left=(
        "        if EXPANSION == 1:\n"
        "            out_re = tl.math.fma(c, z_re, 0.0)\n"
        "            out_im = tl.math.fma(c, z_im, 0.0)\n"
        "        else:\n"
        "            out_re = c * z_re\n"
        "            out_im = c * z_im\n"),
    field_left=(
        "        if EXPANSION == 1:\n"
        "            out_re = tl.math.fma(z_re, c, 0.0)\n"
        "            out_im = tl.math.fma(z_im, c, 0.0)\n"
        "        else:\n"
        "            out_re = z_re * c\n"
        "            out_im = z_im * c\n"))
MUTATIONS_UNARY_OLD = HELPER_IMPORT
MUTATIONS_UNARY_NEW = _helpers(
    coefficient_left=(
        "        if EXPANSION == 1:\n"
        "            out_re = tl.math.fma(c, z_re, -(0.0 * z_im))\n"
        "            out_im = tl.math.fma(c, z_im, 0.0 * z_re)\n"
        "        else:\n"
        "            out_re = (c * z_re) - (0.0 * z_im)\n"
        "            out_im = (c * z_im) + (0.0 * z_re)\n"),
    field_left=(
        "        if EXPANSION == 1:\n"
        "            out_re = tl.math.fma(z_re, c, -(z_im * 0.0))\n"
        "            out_im = tl.math.fma(z_re, 0.0, z_im * c)\n"
        "        else:\n"
        "            out_re = (z_re * c) - (z_im * 0.0)\n"
        "            out_im = (z_re * 0.0) + (z_im * c)\n"))

#: The outcome token for a mutant the SUBNORMAL POLICY refused to build. It is a
#: measured state, not an error: under ``flush`` the policy's PTX audit rejects a
#: kernel whose arithmetic disagrees with the policy in force, and a mutation that
#: plants exactly such an instruction is refused before it can run.
PTX_AUDIT_REFUSAL = "REFUSED BY THE PTX AUDIT"

#: What that refusal must SAY before this gate accepts it. A refusal is scored as
#: the declared outcome only when the policy machinery raised it (recognised by the
#: exception class, never by a word) AND its text carries every needle here: the
#: audit's own sentence, and the instruction the mutation planted. Without the
#: second needle any policy refusal at all — an unattainable executor, a cache-token
#: violation — would score this row passed, which is a different measurement.
DIVIDE_PTX_REFUSAL_NEEDLES: Tuple[str, ...] = (
    "subnormal policy is not uniform in this kernel's PTX",
    "div.full.f32",
)

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
                "         _s0_re, _s0_im, _s1_re, _s1_im, _s2_re, _s2_im) = "
                "_h_cell_complex(\n"),
        "new": ("        (_o0_re, _o0_im, _o1_re, _o1_im, _o2_re, _o2_im,\n"
                "         o0_re, o0_im, o1_re, o1_im, o2_re, o2_im) = "
                "_h_cell_complex(\n"),
        "why": "the tap returns the flux density the constitutive read rather than the "
               "stepped magnetic field -- the mis-selection a twelve-value return makes "
               "easy, and one the curl would consume without complaint",
    },
    "m_f_w_H_written_in_place": {
        "target": "kernel", "expected": "CAUGHT", "case": MUTATION_CASE,
        "old": "        tl.store(wo0 + 2 * idx, src0_re, mask=live)\n",
        "new": "        tl.store(wi0 + 2 * idx, src0_re, mask=live)\n",
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
                "                             kp0, kp1, kp2, kmx, kmy, kmz, ny, nz, "
                "EXPANSION)\n"),
        "why": "PREDICTED NULL and recorded as one. `_h_tap_complex` at the program's "
               "own cell with `valid=live` is `_h_cell_complex` at that cell closed on "
               "the same guard, which is what `own0_re/own0_im` already are -- so this "
               "must NOT diverge, and that is the measurement showing the own-cell "
               "register pair and the recompute are ONE code path rather than two that "
               "happen to agree",
    },
    # ---- the certified arithmetic, unmoved -------------------------------------------
    "m_regroup_the_two_accumulations": {
        "target": "kernel", "expected": "CAUGHT", "case": MUTATION_CASE,
        "old": ("        t_re, t_im = _mul_coefficient_left(kp_0, src_re, src_im, "
                "EXPANSION)\n"
                "        a_re = a_re + t_re\n"
                "        a_im = a_im + t_im\n"
                "        t_re, t_im = _mul_coefficient_left(km_0, prev_re, prev_im, "
                "EXPANSION)\n"
                "        a_re = a_re - t_re\n"
                "        a_im = a_im - t_im\n"),
        "new": ("        u_re, u_im = _mul_coefficient_left(kp_0, src_re, src_im, "
                "EXPANSION)\n"
                "        t_re, t_im = _mul_coefficient_left(km_0, prev_re, prev_im, "
                "EXPANSION)\n"
                "        a_re = a_re + (u_re - t_re)\n"
                "        a_im = a_im + (u_im - t_im)\n"),
        "why": "the two accumulations flattened into one. `((f + kps*src) - kms*prev)` "
               "and `f + (kps*src - kms*prev)` are different float32 numbers, and the "
               "certified body's docstring names the grouping -- separate, "
               "left-to-right, never flattened -- as one of the things that decide "
               "bit-identity",
    },
    "m_flatten_curl_parens": {
        "target": "kernel", "expected": "CAUGHT", "case": MUTATION_CASE,
        "old": "        t0_re = ((c_y_re - c_re) + (b_re - b_z_re))\n",
        "new": "        t0_re = (((c_y_re - c_re) + b_re) - b_z_re)\n",
        "why": "the curl's association is re-parsed on the real plane; `(a + b) - c` "
               "and `a + (b - c)` round differently in float32",
    },
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
    "m_periodic_wrap_reads_the_near_row": {
        "target": "kernel", "expected": "CAUGHT", "case": PERIODIC_MUTATION_CASE,
        "old": "            si = tl.where(si < 0, nx - 1, tl.where(si == nx, 0, si))\n",
        "new": "            si = tl.where(si < 0, 0, tl.where(si == nx, 0, si))\n",
        "why": "the periodic wrap reads row 0 instead of the far row. SCORED ON THE "
               "ALL-PERIODIC FIXTURE and nowhere else: on a metallic axis this branch "
               "does not compile at all",
    },
    # ---- the Bloch phase, on a phased fixture ----------------------------------------
    "m_phase_applied_to_every_lane": {
        "target": "kernel", "expected": "CAUGHT", "case": PHASED_MUTATION_CASE,
        "old": ("            b_x_re = tl.where(wx, rot_re, b_x_re)\n"
                "            b_x_im = tl.where(wx, rot_im, b_x_im)\n"),
        "new": ("            b_x_re = rot_re\n"
                "            b_x_im = rot_im\n"),
        "why": "the Bloch rotation reaches EVERY lane instead of the wrapped one. The "
               "certified curl applies the phase through `tl.where`, which is a "
               "bitwise select, so an unwrapped lane keeps the loaded words untouched; "
               "this weld redirects the LOAD and must leave that select standing. "
               "SCORED ON A PHASED FIXTURE and nowhere else: `PHX` is a constexpr, so "
               "on an unphased grid the whole block is compiled away and the needle "
               "sits in dead text. "
               "IT REPLACES A MEASURED-WEAK PLANT. The first cut of this leg rotated "
               "by the UNIT phase INSIDE `_h_tap_complex` and came back NULL, which is "
               "correct rather than a miss: `fma(v_re, 1.0, (v_im*0.0)*-1.0)` is "
               "`v_re` and `fma(v_re, 0.0, v_im*1.0)` is `v_im` on every word that is "
               "not a signed zero, so a unit rotation is the identity away from that "
               "class. That the rotation is OUTSIDE the recompute is pinned by the "
               "transcription leg's byte equality and by the laptop suite's scan of "
               "`_h_tap_complex`; what a DEVICE mutation can add is that the SELECT is "
               "load-bearing, which is this one",
    },
    "m_phase_not_conjugated": {
        "target": "host", "expected": "CAUGHT", "case": PHASED_MUTATION_CASE,
        "why": "the per-axis phase is passed WITHOUT the step_D conjugation -- the "
               "`backward=True` negation `complex_fields._phase_arguments` applies "
               "(S:1818-1822). It compiles, launches and converges, and it is the wrong "
               "wrap on every phased axis. Driven by rebinding the plan's phase values "
               "to the un-conjugated table",
    },
    # ---- the complex arithmetic, armed rather than rediscovered ----------------------
    "m_zero_cross_terms_folded": {
        "target": "kernel", "expected": "CAUGHT", "case": ARITHMETIC_MUTATION_CASE,
        "old": MUTATIONS_ZERO_CROSS_OLD, "new": MUTATIONS_ZERO_CROSS_NEW,
        "why": "the zero cross terms folded to a literal 0.0 -- the 'obvious' tidy-up. "
               "They carry the SIGN of the field's words into the result exactly as the "
               "full complex multiply does; MEASURED byte-wrong on signed zeros "
               "(complex_fields._mul_field_left). SCORED ON THE PLANTED ROW-0 FIXTURE, "
               "because that is the class in which it is visible at all",
    },
    "m_unary_negation_of_the_fma_addend": {
        "target": "kernel", "expected": "CAUGHT", "case": ARITHMETIC_MUTATION_CASE,
        "old": MUTATIONS_UNARY_OLD, "new": MUTATIONS_UNARY_NEW,
        "why": "`* -1.0` respelled as unary `-`. Triton lowers `-x` as `0.0 - x` "
               "(3.1.0, language/semantic.py:386-391), and under round-to-nearest "
               "`0.0 - (+0.0)` is `+0.0` -- the addend's zero SIGN is lost, while the "
               "array path's complex multiply negates the rounded cross product "
               "sign-exactly. MEASURED on device by the special_kz tranche; re-measured "
               "on THIS kernel here",
    },
    "m_orientation_swapped_in_the_constitutive": {
        "target": "kernel", "expected": "CAUGHT", "expected_override": "NULL",
        "case": ARITHMETIC_MUTATION_CASE,
        "old": ("        t_re, t_im = _mul_coefficient_left(kp_0, src_re, src_im, "
                "EXPANSION)\n"),
        "new": ("        t_re, t_im = _mul_field_left(src_re, src_im, kp_0, "
                "EXPANSION)\n"),
        "why": "the coefficient-LEFT product respelled as the field-LEFT one for the "
               "kps term. DECLARED CAUGHT, MEASURED NULL, AND THE NULL IS A RESULT: "
               "for a ZERO-IMAGINARY product the two orientations are the same bytes. "
               "Take FMA_V1: coefficient-left is `fma(c, z_re, (0.0*z_im)*-1.0)` / "
               "`fma(c, z_im, 0.0*z_re)` and field-left is "
               "`fma(z_re, c, (z_im*0.0)*-1.0)` / `fma(z_re, 0.0, z_im*c)`. The real "
               "planes are the same product with its factors commuted and the same "
               "signed-zero addend. On the imaginary plane one form fuses `c*z_im` and "
               "adds a zero term while the other fuses a zero product and adds "
               "`round(z_im*c)` -- and adding a zero to an already-rounded value is "
               "exact, so both are `round(c*z_im)`. Under NAIVE the two are the same "
               "two terms with the addition commuted. The orientation table is "
               "NORMATIVE for the product where the addend is NOT a zero term, and "
               "`m_rotation_operands_swapped` is the discriminating control that shows "
               "it is observable on this kernel at all",
    },
    "m_rotation_operands_swapped": {
        "target": "kernel", "expected": "CAUGHT", "case": PHASED_MUTATION_CASE,
        "old": ("            rot_re, rot_im = _rotate_field_left(b_x_re, b_x_im, pxr, "
                "pxi, EXPANSION)\n"),
        "new": ("            rot_re, rot_im = _rotate_field_left(pxr, pxi, b_x_re, "
                "b_x_im, EXPANSION)\n"),
        "why": "the Bloch rotation computed with the PHASE on the field side -- "
               "`p * g` where the array path computes `g * p` (S:1862). THE "
               "DISCRIMINATING CONTROL for the structural null above: here both "
               "operands are genuinely complex, so the FMA addend is a real product "
               "rather than a zero term and WHICH product is fused is byte-visible "
               "(`fma(p_re, g_im, p_im*g_re)` against `fma(g_re, p_im, g_im*p_re)`). "
               "It must be CAUGHT, and that is what shows the orientation table is "
               "load-bearing on this kernel rather than only in the certified "
               "module's docstring",
    },
    "m_divide_by_the_reciprocal_of_sinv": {
        "target": "kernel", "expected": "CAUGHT", "case": ARITHMETIC_MUTATION_CASE,
        "old": ("        v0_re, v0_im = _mul_field_left(r_re, r_im, si_z, "
                "EXPANSION)\n"),
        "new": ("        v0_re = r_re / (1.0 / si_z)\n"
                "        v0_im = r_im / (1.0 / si_z)\n"),
        "why": "THE DIVIDE FACT, RE-MEASURED ON THIS KERNEL rather than cited from the "
               "Dcyl round. Every PML reciprocal is precomputed on the HOST into sinv_* "
               "and both halves MULTIPLY by it; this respells one as a componentwise "
               "divide by the reciprocal, which is the same mathematics, is what the "
               "sibling family's comment would suggest, and is wrong twice over -- "
               "Triton's `/` is div.full.f32 (~2 ulp) and the componentwise form drops "
               "the zero cross terms. UNDER `flush` THE MUTANT NEVER RUNS, and that is "
               "a stronger answer than a divergence rather than a missing one: the "
               "same `div.full.f32` this mutation plants is what the flush policy's "
               "own PTX audit refuses at compile time, so the shipped machinery "
               "rejects the respelling before a single step is taken. Measured "
               "2026-09-07 on `results/triton_complex_fused_hd_pair_2026-09-07/flush` "
               "-- 24 of 1624 audited f32 instructions disagreed -- and it is the ONE "
               "row that separated that leg's 39 of 42 clauses from its keep leg's 42 "
               "of 42. The refusal is accepted only when the policy machinery itself "
               "raises it AND its text carries every needle below; anything else is "
               "still REFUSED TO BUILD and still fails",
        "expected_by_policy": {"keep": "CAUGHT", "flush": PTX_AUDIT_REFUSAL},
        "refusal_needles_by_policy": {"flush": DIVIDE_PTX_REFUSAL_NEEDLES},
    },
    # ---- what the PLANT buys, measured rather than asserted ---------------------------
    # The two zero-sign spellings above are scored on the planted row-0 fixture. These
    # two rows are the SAME EDITS on the RANDOM battery (`bc_mmm`, seeded at 2^80),
    # RECORDED rather than scored: if they are null there and caught on the plant, the
    # plant is what reaches the class, which is the Dcyl round's own finding repeated
    # on this kernel instead of cited from it.
    "m_zero_cross_terms_folded_on_the_random_battery": {
        "target": "kernel", "expected": "RECORDED", "case": MUTATION_CASE,
        "old": MUTATIONS_ZERO_CROSS_OLD, "new": MUTATIONS_ZERO_CROSS_NEW,
        "why": "the zero-cross-term fold on the RANDOM battery, recorded so the "
               "planted fixture's catch has a denominator: a null here and a catch "
               "there is the measurement that a random battery does not reach the "
               "signed-zero class at all",
    },
    "m_unary_negation_on_the_random_battery": {
        "target": "kernel", "expected": "RECORDED", "case": MUTATION_CASE,
        "old": MUTATIONS_UNARY_OLD, "new": MUTATIONS_UNARY_NEW,
        "why": "the unary-negation respelling on the RANDOM battery, recorded for the "
               "same reason: `0.0 - x` differs from `x * -1.0` only on a zero addend, "
               "so a null here is what makes the planted catch a statement about the "
               "class rather than about the fixture",
    },
    # ---- the host mutations ----------------------------------------------------------
    "m_naive_expansion_arm": {
        "target": "host", "expected": "CAUGHT", "case": PHASED_MUTATION_CASE,
        "why": "the EXPANSION constexpr forced to NAIVE -- every product rounded "
               "separately -- where the probe licenses FMA_V1. SCORED ON A PHASED "
               "FIXTURE, and that placement is a MEASURED finding rather than a "
               "convenience: every product this kernel performs OUTSIDE the Bloch "
               "rotation is a ZERO-IMAGINARY one, where the FMA's addend is a zero "
               "term and the fused and unfused spellings round identically. The arm is "
               "byte-visible on this seam ONLY through `_rotate_field_left`, where "
               "both operands are complex -- so on an unphased complex row (k = 0) "
               "this kernel's bytes do not depend on the arm at all. The first cut "
               "scored it on the planted row-0 fixture and measured NULL over six "
               "complete steps, which is what established the finding. Under `flush` "
               "the licence's own docstring records that flush destroys exactly the "
               "subnormal lanes that separate the two arms, so a NULL there is a "
               "finding about the policy and is RECORDED rather than scored",
        "expected_by_policy": {"keep": "CAUGHT", "flush": "RECORDED"},
    },
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
        "why": "the seam's electric withdraw is performed AFTER the launch rather than "
               "before it -- the array-path campaign's own `after_step_D` null control. "
               "Driven in the withdraw leg, on a fixture carrying an integrated "
               "electric source",
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


def unconjugated_phase(driver: Any) -> Callable[[Any], Any]:
    """Rebind the plan's phase values WITHOUT the ``step_D`` conjugation."""

    def wrap(plan: Any) -> Any:
        kinds = stepping._boundary_kinds(driver.fields.grid, driver.pml)  # noqa: SLF001
        phases = complex_fields.bloch_phase_table(driver.fields.grid, kinds)
        _flags, values = complex_fields._phase_arguments(  # noqa: SLF001
            phases, backward=False)
        plan.phase_values = tuple(float(value) for value in values)
        return plan

    return wrap


def naive_expansion(plan: Any) -> Any:
    """Force the ``NAIVE`` expansion arm -- a host mutation on the constexpr."""
    plan.expansion = 0
    return plan


# ---------------------------------------------------------------------------
# DEVICE LEGS
# ---------------------------------------------------------------------------

def _forced() -> bool:
    """Does this process have to bypass the predicate to build the product at all?"""
    return not policy_admits_the_complex_arms()


def leg_product(steps: int, seed: int,
                progress: Optional[Callable[[str], None]] = None) -> List[Dict[str, Any]]:
    """The multi-reference identity on every fixture."""
    rows: List[Dict[str, Any]] = []
    forced = _forced()
    for name, keywords in CASES:
        driver = build_driver(dict(keywords), seed=seed)
        try:
            result = run_product(driver, steps, forced=forced, progress=(
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
    forced = _forced()
    for bits in (0, 40, SEED_SCALE_BITS):
        driver = build_driver(dict(CASES[MUTATION_INDEX][1]), seed=seed,
                              scale_bits=bits)
        try:
            result = run_product(driver, steps, require_full_budget=True,
                                 forced=forced)
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
    forced = _forced()
    driver = build_driver(dict(CASES[MUTATION_INDEX][1]), seed=seed)
    try:
        arrangements, absent = all_arrangements(driver, count=True, forced=forced)
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
    forced = _forced()

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
                    driver, sync_hazard=armed, forced=forced,
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


def _volume_source(component: str = "Ez", integrated: bool = True) -> Dict[str, Any]:
    """One integrated electric source declaration, in the driver's OWN schema.

    CONTINUOUS AND STARTING AT ZERO, and both are measurements rather than taste: a
    gaussian's turn-on is hundreds of steps of dead time, so on the steps compared its
    ``_applied_dipole`` is zero and the withdraw this leg exists to move is a no-op.
    """
    return {"source_type": "continuous", "component": component, "frequency": 0.6,
            "start_time": 0.0, "amplitude": 1.0, "center": (0.0, 0.0, 0.0),
            "is_integrated": integrated}


class _WithdrawAfter:
    """The launch, then the withdraw -- the campaign's ``after_step_D`` null control."""

    def __init__(self, inner: Any, fields: Any, sources: Sequence[Any]) -> None:
        self.inner = inner
        self.absorbed_by = inner
        self._fields = fields
        self._sources = tuple(sources)

    def run(self, *args: Any, **kwargs: Any) -> None:
        self.inner.run(*args, **kwargs)
        for _index, source in withdraw_hoist.standing_withdraws(self._sources):
            source.withdraw(self._fields)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


def leg_withdraw(steps: int, seed: int) -> Dict[str, Any]:
    """The seam's one pass, on a device, with both of the campaign's null controls.

    The configuration this product's predicate REFUSES today -- a standing integrated
    electric withdraw -- driven anyway through the from-arrays route, with the hoist in
    place. Hoisted must be identical; un-hoisted and after_step_D must both diverge.
    """
    findings: List[str] = []
    out: Dict[str, Any] = {}
    keywords = dict(CASES[MUTATION_INDEX][1])
    sources = (_volume_source(),)
    standing = None
    dipoles: Dict[str, List[float]] = {}
    # THE FIELD STARTS AT EXACTLY ZERO ON THIS LEG. Everywhere else this gate seeds at
    # 2^80 to keep the state clear of the denormal band; here the ONLY thing that may
    # move a byte is the source, and a standing dipole of order 1 subtracted from a
    # field of order 1e26 is lost in float32 rounding.
    for mode in ("hoisted", "not_hoisted", "after_step_D"):
        driver = build_driver(keywords, seed=seed, sources=sources, amplitude=0.0)
        try:
            declared = tuple(getattr(driver, "_sources", ()))
            standing = len(withdraw_hoist.standing_withdraws(declared))
            verdict = family.complex_fused_hd_pair_coverage(driver.fields, driver.pml,
                                                            declared)
            if mode == "hoisted":
                out["predicate_refuses_this_configuration"] = not verdict.covered
                out["predicate_reasons"] = list(verdict.reasons)
                out["the_refusal_names_the_standing_withdraw"] = any(
                    "standing integrated" in reason for reason in verdict.reasons)
            launcher = None
            if mode == "after_step_D":
                def launcher(plan: Any) -> Any:
                    return _WithdrawAfter(plan, driver.fields, declared)
            arrangements = {
                "array": Arrangement("array", None),
                "weld": arrangement_weld(driver, launcher=launcher,
                                         hoist=(mode == "hoisted"),
                                         forced=True),
            }
            seen: List[float] = []

            def watch(step: int, name: str, run: Any,
                      seen: List[float] = seen) -> Any:
                seen.append(max(
                    (abs(complex(getattr(source, "_applied_dipole", 0j) or 0j))
                     for source in getattr(run, "_sources", ())), default=0.0))
                return None

            result = drive(driver, arrangements, steps, per_step_hook=watch,
                           stop_on_divergence=False)
            dipoles[mode] = [round(value, 12) for value in seen]
            out[mode] = {"bit_identical": result["bit_identical"],
                         "first_divergence_step": result["first_divergence_step"],
                         "steps_compared": result["steps_compared"],
                         "reference_moved_words_step_1":
                             result["reference_moved_words_step_1"]}
        finally:
            driver.close()
    out["standing_withdraws"] = standing
    out["standing_dipole_magnitude_per_step"] = dipoles
    if not standing:
        findings.append("no standing withdraw on the withdraw leg's fixture; the leg "
                        "is disarmed and measures nothing")
    if not any(value > 0.0 for values in dipoles.values() for value in values):
        findings.append(
            "the standing dipole is ZERO on every step compared, so the seam's "
            "withdraw does no work and neither null control could diverge")
    if not out["hoisted"]["reference_moved_words_step_1"]:
        findings.append("the array path moved no word on step 1: on a zero-seeded "
                        "fixture the source is the only driver")
    if not out.get("predicate_refuses_this_configuration"):
        findings.append("the predicate ADMITS a row with a standing integrated electric "
                        "withdraw while HOISTS_THE_WITHDRAW is False")
    elif not out.get("the_refusal_names_the_standing_withdraw"):
        findings.append("the predicate refuses this configuration for some OTHER "
                        "reason; the withdraw clause is then not what this leg drove")
    if not out["hoisted"]["bit_identical"]:
        findings.append("the HOISTED arrangement is not identical to the array path")
    for null in ("not_hoisted", "after_step_D"):
        if out[null]["bit_identical"]:
            findings.append(f"the {null} null control did NOT diverge")
    out["what_this_licenses"] = (
        "a DEVICE measurement of the BEFORE placement on this backend, on this "
        "fixture, for the ELECTRIC withdraw on complex storage. It does not license "
        "INSTALLABLE or HOISTS_THE_WITHDRAW: both would additionally need the "
        "composer's hoist branch to be reachable for this product, which it is not")
    return {"passed": not findings, "findings": findings, **out}


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
                        "weld": arrangement_weld(driver, kernel=kernel,
                                                 forced=_forced())}
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
    forced = _forced()
    for case in dict.fromkeys(
            (MUTATION_CASE, PERIODIC_MUTATION_CASE, PHASED_MUTATION_CASE,
             ARITHMETIC_MUTATION_CASE)):
        findings: List[str] = []
        driver = build_driver(dict(dict(CASES)[case]), seed=20260907)
        try:
            arrangements = {"array": Arrangement("array", None),
                            "weld": arrangement_weld(driver, kernel=None,
                                                     forced=forced)}
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
    forced = _forced()
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
            elif tag == "m_phase_not_conjugated":
                launcher = unconjugated_phase(driver)
            elif tag == "m_naive_expansion_arm":
                launcher = naive_expansion
            elif tag == "m_withdraw_after_step_D":
                row.update({"outcome": "MEASURED IN THE withdraw LEG", "passed": True,
                            "note": "driven there, on a fixture carrying an integrated "
                                    "electric source, with the hoisted arrangement as "
                                    "its positive control"})
                rows.append(row)
                continue
            arrangements = {"array": Arrangement("array", None),
                            "weld": arrangement_weld(driver, kernel=kernel,
                                                     launcher=launcher,
                                                     forced=forced)}
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
    block.update(run_product(driver, steps, progress=_child_progress,
                             require_full_budget=True, clean_floor=1,
                             forced=_forced()))
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
    blocks = {label: block for label, block in blocks.items()
              if label not in unliftable}
    admitted = sorted(label for label, b in blocks.items() if b.get("predicate_admits"))
    refused = sorted(label for label, b in blocks.items()
                     if b.get("predicate_admits") is False)
    refused_by_the_withdraw = sorted(
        label for label in refused
        if any("standing integrated" in reason
               for reason in blocks[label].get("predicate_reasons", ())))
    refused_by_the_policy = sorted(
        label for label in refused
        if any("subnormal" in reason or "certif" in reason
               for reason in blocks[label].get("predicate_reasons", ())))
    expected_refused = sorted(set(facts["rows_with_a_standing_withdraw"]) & set(blocks))
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
    # UNDER A POLICY THAT REFUSES THE COMPLEX LICENCE every row is refused by that
    # clause and NOTHING is driven. That is a correct answer, not a gap, and it is what
    # the leg asserts there: every liftable row refused, and refused BY THAT NAME.
    licensed = policy_admits_the_complex_arms()
    if licensed:
        passed = bool(measured and not unmeasured
                      and (bool(only)
                           or len(admitted) + len(refused) + len(unliftable)
                           == facts["rows_in_cell"])
                      and len(driven) + len(non_deterministic) == len(admitted)
                      and refused == refused_by_the_withdraw == expected_refused
                      and not not_driven and not diverged
                      and passed_rows == driven
                      and set(driven) | set(non_deterministic) == set(admitted)
                      and majority_cleared
                      and children_ran_under_the_policy)
    else:
        passed = bool(measured and not unmeasured and not admitted
                      and refused == refused_by_the_policy
                      and (bool(only)
                           or len(refused) + len(unliftable)
                           == facts["rows_in_cell"]))
    return {
        "passed": passed,
        "facts": facts,
        "the_policy_licenses_the_complex_arms": licensed,
        "what_this_leg_asserts_under_an_unlicensed_policy": (
            "every liftable row REFUSED, and refused by the certification clause -- "
            "not by the withdraw clause and not by silence. Nothing is driven, which "
            "is the correct answer under a policy no complex arm was certified for, "
            "and calling it a gap would publish a phantom one"),
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
        "refused_by_the_subnormal_certification_clause": refused_by_the_policy,
        "refused_expected_from_the_seam_record": expected_refused,
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
            policy_attained: bool, licensed: bool) -> Dict[str, Any]:
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
        # THE REFERENCE SET IS POLICY-DEPENDENT and is asserted as such rather than
        # relaxed: under a licensing policy all six must be present, and under an
        # unlicensed one exactly the two composition references must be NAMED absent.
        expected = set(MODES_UNDER_KEEP if licensed else MODES_UNDER_FLUSH)
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
    probe_path, probe_record = _probe_record()
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
        "expansion_probe_path": probe_path,
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
    licensed = False
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

        licensed = policy_admits_the_complex_arms()
        payload.update({
            "cupy_version": cupy.__version__, "triton_version": triton.__version__,
            "device": str(cupy.cuda.runtime.getDeviceProperties(
                cupy.cuda.runtime.getDevice())["name"]),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cupy_cache_dir": os.environ.get("CUPY_CACHE_DIR"),
            "triton_cache_dir": os.environ.get("TRITON_CACHE_DIR"),
            "the_policy_licenses_the_complex_arms": licensed,
            "expansion_licence": (complex_fields.expansion_license(probe_record)
                                  if probe_record is not None else None),
            "expansion_certification_refusals": list(
                complex_fields.expansion_certification_reasons(None)),
            "references_this_policy_can_build": list(
                MODES_UNDER_KEEP if licensed else MODES_UNDER_FLUSH),
            "what_an_unlicensed_policy_measures": (
                "under a policy the complex tranche was not certified for, the "
                "predicate refuses by the certification clause, the composer selects "
                "nothing complex on this seam, and the two composition references do "
                "not exist. What this record then measures is the product's ARITHMETIC "
                "from arrays with the EXPANSION arm forced to the one the keep-cut "
                "probe licenses, against the array path and the two certified singles "
                "built the same way. It does NOT license the product under that "
                "policy"),
            "forced_expansion": (forced_expansion() if not licensed else None),
        })
        log(f"subnormal policy installed: "
            f"{payload['subnormal_policy'].get('policy')!r} "
            f"(requested {args.subnormal_policy!r}, "
            f"CUPY_CACHE_DIR={os.environ.get('CUPY_CACHE_DIR')!r}); "
            f"complex arms licensed: {licensed}")

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

    payload["release"] = verdict(rows, legs, policy_attained, licensed)
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
