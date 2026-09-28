#!/usr/bin/env python
"""DEVICE byte gate for the Triton FOLDED H->D weld: folded ``update_H`` welded into
folded ``step_D``.

THE SECOND PRODUCT ON THE FOURTH SEAM, and the largest single cell this backend's
fusion board carries unbuilt: ``(update_H folded -> step_D folded PML)`` is 78
seam-instances of a 597 denominator on
``results/fusion_matrix_triton_2026-09-07_cyl`` -- 75 ``buildable_not_built`` and 3
``withdraw_seam``. Its sibling ``gate_triton_fused_hd_pair.py`` certifies the
unfolded cell; this file is that gate with the fold in it, and everything the fold
changes is a leg here rather than a sentence.

WHAT IS BEING CERTIFIED. ``meep_gpu/triton_kernels/folded_fused_hd_pair.py`` computes
the folded ``update_H`` constitutive into launch-local SCRATCH through the CERTIFIED
``fused_hd_pair._h_cell`` (imported, not copied), takes its own cell's magnetic field
from registers, RECOMPUTES every one of the folded curl's six foreign taps from
pre-launch state, steps ``D``/``fu_D`` in place with
``symmetry.pml_curl_step_folded``'s own body, and rotates the ``H``/``f_w_H``
references afterwards. The claim is per COMPLETE DRIVER STEP -- ``FdtdDriver.step``'s
own consult order, with the near symmetry fill, the wall clear and the folded FAR
ghost pass running exactly where the driver runs them -- over a stated budget of
steps, as uint32 WORDS over EVERY stored volume the engine allocates (primaries,
split-field PML auxiliaries, both ``f_w`` histories; never ``allclose``, because
``-0.0 == 0.0`` lies), against FOUR reference engines driven in lockstep from one
seed:

  1. the ARRAY PATH -- ``stepping``'s passes under the driver's own loop;
  2. the CERTIFIED SINGLES -- ``symmetry.plan_folded_constitutive(..., "H")`` then
     ``symmetry.plan_folded_pml_curl(..., "step_D")`` as two dispatched plans;
  3. the COMPOSITION THE COMPOSER INSTALLS ON THESE ROWS TODAY -- ``plan_step(fuse=
     True)``'s own plans, which on every row this product reaches puts the released
     ``fused pair B (folded)`` in ``step_B``/``update_H``;
  4. the same slots DISPATCHED UNFUSED -- ``plan_step(fuse=False)``.

WHY EVERY STORED VOLUME AND NOT JUST THE PRIMARIES. ``symmetry.py``'s own docstring
says the two ownership masks are INVISIBLE after a complete driver step, because the
fill passes overwrite exactly the planes they protect. THAT IS TRUE OF ``D`` AND
FALSE OF ``fu_D``, and it is measured rather than reasoned: on the Metal sibling
(``results/metal_folded_fused_hd_pair_2026-09-07_wired``) dropping the top-plane mask
moved 264 words and dropping the cell-0 mask 540, after complete steps, EVERY one of
them in the auxiliary and none in the displacement. ``curlN`` feeds both the stored
auxiliary ``nN`` and the stored displacement ``vN``; the fills and the wall clear
walk the D components and never the ``fu_*`` ones. So a whole-step comparison over
the primaries alone would certify a MASK-LESS folded curl as correct. Both mask drops
are armed here, SCORED at whole step, and their ATTRIBUTION is asserted: each must
move ``fu_D`` and must NOT move ``D``.

THE CODES ARE THIS FAMILY'S SINGLE POINT OF FAILURE. ``plan_fused_hd_pair`` maps
``_boundary_kinds`` to ``1 if metallic else 0``; on a folded axis that reports
``"mirror"``, which the expression sends to 0 = PERIODIC -- the ghost wraps to the far
plane and the cell-0 mask is never emitted. Both are valid codes, so nothing in the
kernel can catch it. ``codes_source`` arms exactly that mapping and requires it caught
on every folded fixture.

BOTH CANONICAL SUBNORMAL POLICIES. ``keep`` needs a ``CUPY_CACHE_DIR`` carrying the
``ftz_stripped`` token; ``flush`` needs ``--import-meep-for-host-policy``, because
``mp.set_zero_subnormals`` is the only exposure of this process's FTZ/DAZ bits the
package may use. ``strict=True``: a process that asked to keep and quietly did not is
a process whose bytes mean nothing.

LEGS
  host legs (no device launch)
  driver_order       REPLACES is exactly the driver's two adjacent consults, the only
                     statement between them is the electric withdraw loop -- on a fold
                     as much as off one, with all three B-side fills closing before
                     the first consult and all three D-side ones opening after the
                     second -- and the sync channel's containment rule excludes
                     step_D; all read off the tree, never spelled
  transcription      the curl half is ``symmetry.pml_curl_step_folded``'s own text
                     with EXACTLY the nine redirected magnetic reads, every tap's cell
                     and guard PARSED from the FOLDED emitter's own lines; the
                     constitutive half is IMPORTED from ``fused_hd_pair`` and this
                     module defines no copy of it; the fold's own three deltas (the
                     inverted ghost branch, the ``!= PERIODIC`` cell-0 mask and the
                     MIRROR_PERIODIC top-plane block) are present here and absent from
                     the plain kernel; no ``g``/``e`` pointer survives below the seam;
                     every mutation needle resolves once, in the file it names
  purity_ledger      per fixture, on the array path: of the folded curl's valid
                     foreign taps, how many land on a cell ``update_H`` MOVED -- and,
                     fold-specific, how many land on the near-ghost plane (stored cell
                     0 of a folded axis) and how many of THOSE moved. Near zero would
                     mean the race legs license nothing
  ghost_observability  whether any BOUNDARY VALUE the folded ghost serves reaches
                     ``step_D``'s output, derived from the folded kernel's own text:
                     each shifted tap feeds one curl component and its ghost fires on
                     the plane the cell-0 mask zeroes on every NON-PERIODIC axis, the
                     fold included. What that costs the mutation set is stated
  device legs
  refusal            by name: an undeclared source list; a standing INTEGRATED
                     electric withdraw (the 3 rows of 78 the board files under
                     withdraw_seam); a non-integrated electric source NOT refused; a
                     magnetic source NOT refused; an UNFOLDED grid, naming
                     ``fused_hd_pair``; a cylindrical grid; a conductivity naming the
                     curl half; an inactive absorber naming the constitutive half
                     FIRST; and the DISJOINTNESS of the two H->D products driven in
                     both directions on the same fixtures
  arbitration        the composer, asked: on a folded row this product's predicate
                     admits, the released ``fused pair B (folded)`` holds step_B AND
                     update_H, and with this product's row INJECTED IN PROCESS it is
                     refused twice over -- by ``_pair_may_absorb`` naming the incumbent
                     and by ``_declared_uninstallable`` naming the flag -- while the
                     incumbent KEEPS its slots
  product            the four-reference identity on the eight folded specialisations,
                     complete driver steps, movement floor, subnormal census
  seed_scale         the 2^80 seed scale is a change of exponent and nothing else
  launch_structure   launches per step at the seam and over the whole step, two
                     independent counters
  sync               the product FORCE-installed; a flux accessor called mid-run so
                     synchronize_magnetic_fields consults update_H_synchronize. An arm
                     that answers MUST diverge in D/fu_D; the arm that declines by the
                     containment rule MUST NOT
  withdraw           an integrated electric source in the seam (the configuration the
                     predicate refuses today): the hoisted arrangement identical, the
                     un-hoisted launch and the after_step_D placement both diverge
  byte_neutral       the own-cell register read replaced by a reload of the scratch
                     just stored: the one armed edit required NOT to diverge
  mutation           the armed kernel and host defects, each of which MUST diverge,
                     with the two mask drops' ATTRIBUTION asserted; predicted-null
                     entries recorded with their reason and with the paired control
                     that earns the null, never dropped
  disarm             the identical harness, shipped bytes, must not diverge
  lift               every corpus row the standing census puts in the (folded ->
                     folded PML) cell -- 78 of them, 3 expected refused by name --
                     re-lifted in its own interpreter WHICH INSTALLS THE SAME
                     SUBNORMAL POLICY BEFORE ITS FIRST DEVICE COMPILE, driven to
                     bit-identity over the full budget, with the composer's own slot
                     table per row

THE PRODUCT IS UNREGISTERED WHILE THIS GATE RUNS. No composer installs it -- there is
no ``CERTIFIED_FUSED_PRODUCTS`` row and no ``fastpath`` label -- so reference 3 is the
composition the composer builds WITHOUT it, and the weld is force-installed by this
gate's own shim. A composition-installed reference awaits wiring and is named as owed
rather than implied.

Rule 7: one flushed line per case; every row appended and fsynced as it lands; the
lift leg writes one JSON per corpus row as it lands and a progress log the child
appends to per step.

    CUDA_VISIBLE_DEVICES=<n> TRITON_LIBCUDA_PATH=$HOME/triton_libcuda_stub \\
      LIBRARY_PATH=$HOME/triton_libcuda_stub \\
      CUPY_CACHE_DIR=<fresh>/cupy_cache/ftz_stripped PYTHONPATH=. \\
      python -u parity/meep_gpu/gate_triton_folded_fused_hd_pair.py \\
      --subnormal-policy keep --out <fresh>/gate.json

    # laptop: the host legs only
    PYTHONPATH=. python -u \\
      parity/meep_gpu/gate_triton_folded_fused_hd_pair.py --no-device \\
      --out <fresh>/no_device.json
"""
# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import argparse
import ast
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
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

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
import h_to_d_seam  # noqa: E402

from meep_gpu import stepping, withdraw_hoist  # noqa: E402
from meep_gpu.fastpath import (  # noqa: E402
    SYNC_PASS_OWNERS, SYNC_PATH_SLOTS, SYNC_UPDATE_H_PASS,
)
from meep_gpu.triton_kernels import coverage as tcoverage  # noqa: E402
from meep_gpu.triton_kernels import (  # noqa: E402
    folded_fused_hd_pair as family,
)
from meep_gpu.triton_kernels import fused_hd_pair as plain  # noqa: E402
from meep_gpu.triton_kernels import launch as triton_launch  # noqa: E402
from meep_gpu.triton_kernels import symmetry as tsymmetry  # noqa: E402

GATE = "triton_folded_fused_hd_pair"

#: Every module whose bytes this gate's verdict depends on.
SOURCES: Tuple[str, ...] = (
    "meep_gpu/triton_kernels/folded_fused_hd_pair.py",
    # THE PLAIN PRODUCT IS BOUND BECAUSE ITS TEXT IS HALF OF THIS ONE: `_h_cell` and
    # `_h_tap` are IMPORTED from it and are this weld's whole constitutive arithmetic,
    # and `OWN_LOAD_EDITS`/`HALO_TAPS`/`offset_coordinates`/`halo_taps`/`needle` are
    # the lift machinery this module reuses rather than copies.
    "meep_gpu/triton_kernels/fused_hd_pair.py",
    # AND SYMMETRY, BECAUSE THE CURL HALF IS CUT OUT OF IT. `pml_curl_step_folded` is
    # the certified body this weld lifts, and `folded_axis_kinds` is the classifier
    # that decides which of the four ghost rules each axis takes -- the single point
    # of failure in this family.
    "meep_gpu/triton_kernels/symmetry.py",
    "meep_gpu/triton_kernels/offdiag_scratch_weld.py",
    "meep_gpu/triton_kernels/kernels.py",
    "meep_gpu/triton_kernels/coverage.py",
    "meep_gpu/triton_kernels/launch.py",
    "meep_gpu/stepping.py",
    "meep_gpu/withdraw_hoist.py",
    # THE DRIVER IS BOUND ON PURPOSE. This weld's whole `REPLACES` claim is about
    # WHICH passes run between `update_H` and `step_D`, and on a folded grid that
    # claim is larger than on an unfolded one: three B-side passes close before the
    # first consult and three D-side ones open after the second. A driver that
    # reordered its seam would leave every clause here reading correctly while the
    # bytes described a step nothing performs.
    "meep_gpu/driver.py",
    "meep_gpu/fields.py",
    "meep_gpu/fastpath.py",
    "parity/meep_gpu/gate_triton_folded_fused_hd_pair.py",
)

#: The budget every synthetic case runs. SIXTY, not the sibling gates' twelve: the
#: state this weld carries between steps (the rotated ``H``/``f_w_H`` pair AND the
#: in-place ``fu_D``) compounds across steps, and the withdraw campaign this gate is
#: the device counterpart of ran sixty. The comparison is per COMPLETE STEP.
STEPS = 60

#: The steps at which the sync leg calls the flux accessor. Two, so the hazard is
#: measured to persist and not to be a one-step transient.
SYNC_STEPS: Tuple[int, ...] = (3, 7)

#: HOW MANY CLEAN COMPLETE STEPS A LIFTED CORPUS ROW MUST REACH TO COUNT. The
#: synthetic fixture's amplitude is this gate's to choose; a lifted row's state is
#: the ROW'S and enters the denormal band on its own schedule. It is a FLOOR and not
#: a target: the record carries each row's own step count and the words it compared.
LIFT_CLEAN_STEP_FLOOR = 8

#: THE EIGHT FOLDED SPECIALISATIONS. The fold is expressed entirely through the
#: four-valued boundary constexprs and the stored extent, so what a fixture varies
#: is exactly what the kernel specialises on: which axes are folded, which
#: termination each fold carries (MIRROR_METALLIC stores no slot past MEEP's owned
#: window, MIRROR_PERIODIC does), whether a non-folded wall or a periodic axis is
#: live beside them, and the mirror PHASE -- which this product must be blind to.
CASES: Tuple[Tuple[str, Dict[str, Any]], ...] = (
    # (M, MM, P) -- one folded metallic axis, a live non-folded wall, a periodic axis.
    # The shape family the largest partition of the cell's rows carries, and the only
    # one of these eight on which the PERIODIC wrap is observable at all.
    ("fold_y_metallic", {"symmetry": (("Y", 1),),
                         "boundaries": {"x": "metallic", "y": "metallic"}}),
    # (M, MP, P) -- the folded PERIODIC termination: the stored array carries the slot
    # past MEEP's owned window, the driver's FAR ghost pass is live, and the
    # top-plane mask is emitted on y.
    ("fold_y_periodic", {"symmetry": (("Y", 1),),
                         "boundaries": {"x": "metallic", "y": "periodic"}}),
    # THE SAME GRID WITH THE ODD MIRROR PLANE. This product bakes no parity -- its
    # folded ghost is an exact 0.0 and the phase enters only the fill kernels, which
    # are outside REPLACES -- so the emitted source must be IDENTICAL to the case
    # above and the bytes must still equal the array path, which images with -1.
    ("fold_y_periodic_odd_phase", {"symmetry": (("Y", -1),),
                                   "boundaries": {"x": "metallic",
                                                  "y": "periodic"}}),
    # (MM, MM, P) -- two folded metallic axes, the cell's largest single shape.
    ("fold_xy_metallic", {"symmetry": (("X", 1), ("Y", 1)),
                          "boundaries": {"x": "metallic", "y": "metallic"}}),
    # (MP, MM, M) -- THE MUTATION CASE. Every line the shipped kernel can emit except
    # the periodic wrap: a top-plane mask on x, a folded metallic ghost on y, a plain
    # metallic wall on z, and the cell-0 mask live on all three.
    ("fold_xy_mixed_z_wall", {"symmetry": (("X", 1), ("Y", 1)),
                              "boundaries": {"x": "periodic", "y": "metallic",
                                             "z": "metallic"}}),
    # (M, MP, MP) -- two folded PERIODIC axes: two top-plane lines and a doubly
    # unowned corner.
    ("fold_yz_periodic", {"symmetry": (("Y", 1), ("Z", 1)),
                          "boundaries": {"x": "metallic", "y": "periodic",
                                         "z": "periodic"}}),
    # (M, MM, M) -- a fold with NO periodic axis at all: every axis masks at cell 0.
    ("fold_y_metallic_z_wall", {"symmetry": (("Y", 1),),
                                "boundaries": {"x": "metallic", "y": "metallic",
                                               "z": "metallic"}}),
    # (MM, MM, MM) -- three folded metallic axes, the 3-D corner of the cell.
    ("fold_xyz_metallic", {"symmetry": (("X", 1), ("Y", 1), ("Z", 1)),
                           "boundaries": {"x": "metallic", "y": "metallic",
                                          "z": "metallic"}}),
)
#: THE CASE THE MUTATIONS ARE ARMED ON: (MIRROR_PERIODIC, MIRROR_METALLIC,
#: METALLIC). Every line the shipped kernel can emit except the periodic wrap is
#: present -- a top-plane mask on x, a folded metallic ghost on y, a plain metallic
#: wall on z, and the widened cell-0 mask live on all three.
MUTATION_CASE = "fold_xy_mixed_z_wall"
#: THE SECOND MUTATION CASE, and it is not a duplicate of the first. The fold WIDENED
#: the cell-0 mask from ``== METALLIC`` to ``!= PERIODIC``, so on a folded or walled
#: axis it zeroes the very curl component the shifted tap feeds at the very plane
#: where the ghost is served and no boundary VALUE is observable there at all. Only a
#: genuinely PERIODIC axis leaves cell 0's curl a live stepped value assembled from
#: the wrapped row, which is the only specialisation on which the wrap is measurable.
PERIODIC_MUTATION_CASE = "fold_y_metallic"
#: The synthetic fixture's geometry and material.
CELL: Tuple[float, float, float] = (2.0, 2.1, 1.2)
RESOLUTION = 10.0
COURANT = 0.35
PML_CELLS = 2
EPSILON: Dict[str, float] = {"Ex": 2.0, "Ey": 2.5, "Ez": 3.0}

#: THE SEED IS SCALED BY 2^80, AND THE EXPONENT IS A MEASUREMENT RATHER THAN A TUNING.
#: The solver is linear in the field state and every coefficient it multiplies by is
#: field-independent, so scaling every stored volume by 2^n shifts each float32
#: EXPONENT by n and leaves every MANTISSA and every rounding decision untouched --
#: the same arithmetic, measured further from the denormal band, where a ``flush``
#: leg and a ``keep`` leg would otherwise differ for reasons that have nothing to do
#: with any kernel. ``leg_seed_scale`` DRIVES that equality rather than asserting it.
SEED_SCALE_BITS = 80

#: The reference engines, in the order the record reports them.
#: ``weld_seam_only`` is the DISCRIMINATING arrangement and is not decoration. Every
#: other arrangement that dispatches ``update_E`` carries that sub-step's own
#: agreement with the array path into this comparison; measured 2026-09-06 on
#: ``examples:bend-flux.py``, where ``unfused``, ``composition_today`` and ``weld``
#: all disagreed with the array path by the SAME 9 words of ``Ez`` (every one of them
#: ``zero -> subnormal``) while ``singles`` -- which dispatches only this seam's two
#: slots -- was identical. A comparison that could not separate the two would have
#: read another sub-step's finding as this weld's.
MODES: Tuple[str, ...] = ("array", "singles", "composition_today", "unfused",
                          "weld", "weld_seam_only")

#: THE SPLIT IS THE PREDICATE'S, NOT A CONVENIENCE. ``coverage._grid_reasons`` clause
#: 1 refuses any grid whose ``xp`` is not CuPy, so ``refusal`` and ``arbitration`` --
#: which LAUNCH nothing -- still need a device to build a fixture the predicate can
#: admit at all. They sit in the device group for that reason and not because they
#: dispatch: a laptop run that scored them would be scoring a backend refusal.
LEG_GROUPS: Dict[str, Tuple[str, ...]] = {
    "host": ("driver_order", "transcription", "purity_ledger",
             "ghost_observability"),
    "device": ("refusal", "arbitration", "product", "seed_scale",
               "launch_structure", "sync", "withdraw", "byte_neutral", "mutation",
               "disarm", "lift"),
}
ALL_LEGS: Tuple[str, ...] = tuple(leg for group in LEG_GROUPS.values() for leg in group)

#: The census this gate lifts its corpus rows from, and the seam record whose
#: ``withdraw_in_seam`` flag names the rows the predicate must refuse.
CENSUS = "predicate_coverage_triton_2026-09-04_cylm0"
SEAM_RECORD = "h_to_d_seam_2026-09-04"
CELL_ARMS: Tuple[str, str] = family.ARMS  # ("folded", "folded PML")

#: The runner's "cannot certify on this host" code: a partial run exits with it so
#: ``release.released`` is False and nothing can mint it.
EXIT_INCOMPLETE = 75


def log(message: str) -> None:
    print(message, flush=True)


# ---------------------------------------------------------------------------
# Words, volumes, comparison
# ---------------------------------------------------------------------------

def _module_of(array: Any) -> Any:
    """The array module that owns ``array`` -- CuPy on the device, NumPy on the host."""
    if type(array).__module__.split(".")[0] == "cupy":  # pragma: no cover - device
        import cupy  # noqa: PLC0415

        return cupy
    return np


def _host(array: Any) -> np.ndarray:
    """One device or host array as a contiguous NumPy array."""
    xp = _module_of(array)
    if xp is not np:  # pragma: no cover - device path
        return np.ascontiguousarray(xp.asnumpy(array))
    return np.ascontiguousarray(array)


def native_words(array: Any) -> Any:
    """One array as uint32 WORDS, WHERE IT LIVES. Byte compares, never allclose.

    THE COMPARISON STAYS ON THE DEVICE, and that is not only speed. Every snapshot
    this gate takes is of the whole stored state, five arrangements deep, once per
    complete step; brought to the host that is a gigabyte of copies per step on the
    corpus's largest rows and the lift leg cannot finish at all. The words are the
    same words either way -- a reinterpretation of the same bits, not a conversion --
    and :func:`words` is the host spelling the autopsy uses on the handful of volumes
    a divergence names.
    """
    xp = _module_of(array)
    contiguous = xp.ascontiguousarray(array)
    if contiguous.dtype == xp.complex64:
        return contiguous.reshape(-1).view(xp.uint32)
    return xp.ascontiguousarray(
        contiguous.astype(xp.float32, copy=False)).reshape(-1).view(xp.uint32)


def words(array: Any) -> np.ndarray:
    """One array as uint32 WORDS on the HOST -- the autopsy's and the census's route."""
    contiguous = _host(array)
    if contiguous.dtype == np.complex64:
        return contiguous.reshape(-1).view(np.uint32)
    return np.frombuffer(contiguous.astype(np.float32, copy=False).tobytes(),
                         dtype=np.uint32)


def differing(left: Any, right: Any) -> int:
    a, b = native_words(left), native_words(right)
    if a.shape != b.shape:
        return max(a.size, b.size)
    return int(_module_of(a).count_nonzero(a != b))


_RESET_VOLUMES: Optional[Tuple[str, ...]] = None


def reset_declared_volumes() -> Tuple[str, ...]:
    """Every array ``Fields.reset`` zeroes, read out of ``fields.py``'s own text.

    DERIVED, NOT TRANSCRIBED: a volume added to the engine joins the comparison
    without an edit here, and a list here could go stale while looking complete.
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
    _RESET_VOLUMES = tuple(sorted(names - {"polarizations",
                                           "_fmp_scratch_by_component"}))
    return _RESET_VOLUMES


def stored_volumes(fields: Any) -> Dict[str, Any]:
    """Every stored volume this engine allocates, LIVE (not copied), by name."""
    out: Dict[str, Any] = {}
    for name in reset_declared_volumes():
        array = getattr(fields, name, None)
        if array is not None:
            out[name] = array
    for component, array in (getattr(fields, "_fmp_scratch_by_component", None)
                             or {}).items():
        if array is not None:
            out[f"_fmp_scratch_by_component[{component}]"] = array
    for index, state in enumerate(getattr(fields, "polarizations", ()) or ()):
        for component, array in getattr(state, "P", {}).items():
            out[f"P[{index}].{component}"] = array
        for component, array in getattr(state, "P_prev", {}).items():
            out[f"P_prev[{index}].{component}"] = array
    return out


def capture(driver: Any) -> Dict[str, Any]:
    """The whole of the state one complete step advances, WHERE IT LIVES."""
    return {
        "volumes": {name: _module_of(array).array(array, copy=True)
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


#: THE PRIVATE-SCRATCH RULE IS THE LEADING UNDERSCORE, and it is this package's own
#: (``gate_triton_folded_dispersive_fused_pair.PRIVATE_SCRATCH`` and the test that
#: pins the rule rather than the name). ``Fields._fmp_scratch`` is allocated by
#: whichever route calls ``displacement_minus_polarization`` -- the array path always,
#: a fused route only where its own composition reaches it -- so its CONTENTS differ
#: by route while it carries no state any route reads across a step. It is out of the
#: byte comparison for that reason, it is REPORTED separately rather than dropped, and
#: this gate MEASURES the premise instead of inheriting it: the determinism control's
#: second arm wipes every private volume before every step and must still be
#: bit-identical to the arm that does not.
#:
#: A PUBLIC asymmetry is still a failure: the inventories are asserted equal first.
def _is_private(name: str) -> bool:
    return name.startswith("_")


def compare_snapshots(reference: Mapping[str, Any], other: Mapping[str, Any],
                      include_private: bool = False) -> Dict[str, int]:
    a, b = reference["volumes"], other["volumes"]
    assert set(a) == set(b), sorted(set(a) ^ set(b))
    return {name: n for name in sorted(a)
            if (include_private or not _is_private(name))
            and (n := differing(a[name], b[name]))}


#: How many differing words of each volume the autopsy transcribes.
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

    A count of differing words says a run disagreed; it does not say whether the two
    sides differ by one unit in the last place, by a flush of a subnormal to zero, or
    by a wholesale wrong answer -- and those three findings have three different
    subjects. Purely additive: nothing here decides a verdict.
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
                "word_distance": abs(a_word - b_word),
                "same_sign": (a_word >> 31) == (b_word >> 31)})
        out[name] = {
            "differing_words": int(where.size),
            "words_in_volume": int(left.size),
            "classes": dict(sorted(classes.items())),
            "transcript": transcript}
    return out


def subnormal_census(snapshot: Mapping[str, Any]) -> Dict[str, int]:
    """How many stored words are in the float32 denormal band, per volume.

    A byte claim made where the two sides' denormal handling differs is a claim about
    the platform and not about the kernel, so the census rides with every product row
    whether or not it is zero.
    """
    out: Dict[str, int] = {}
    for name, array in snapshot["volumes"].items():
        w = native_words(array)
        xp = _module_of(w)
        banded = int(xp.count_nonzero(
            ((w & xp.uint32(0x7F800000)) == 0) & ((w & xp.uint32(0x007FFFFF)) != 0)))
        if banded:
            out[name] = banded
    return out


# ---------------------------------------------------------------------------
# The synthetic fixture: a REAL driver
# ---------------------------------------------------------------------------

def build_driver(keywords: Mapping[str, Any], seed: int,
                 sources: Sequence[Mapping[str, Any]] = (),
                 scale_bits: int = SEED_SCALE_BITS,
                 amplitude: float = 0.37,
                 prefer_gpu: bool = True) -> Any:
    """One seeded FOLDED ``FdtdDriver`` on this gate's geometry and material.

    A DRIVER, not a bare ``Fields``: every leg here is a claim about a COMPLETE
    driver step -- and on a folded grid that is a larger claim than on an unfolded
    one. ``fill_symmetry_bc_B``, ``zero_metal_B`` and ``fill_folded_far_ghosts_B``
    run BEFORE the ``update_H`` consult and their D-side triple runs AFTER the
    ``step_D`` consult; the values every one of them writes are what this weld's
    foreign taps recompute from. A hand-written walk over the pass list would be a
    second model of what a folded step is.

    THE ABSORBER GOES ON THE FAR FACE OF A FOLDED AXIS ONLY. The near face of a
    folded axis is the MIRROR PLANE, not a boundary -- ``PML.__post_init__`` refuses
    a layer there, and ``symmetry.folded_axis_kinds`` refuses a grid whose layer it
    cannot resolve. So the thickness is spelled per face from the fixture's own
    mirror set rather than passed as a scalar.
    """
    from meep_gpu.driver import FdtdDriver  # noqa: PLC0415
    from meep_gpu.grid import Mirror  # noqa: PLC0415

    symmetry = tuple(keywords.get("symmetry") or ())
    driver = FdtdDriver(cell_size=CELL, resolution=RESOLUTION, courant=COURANT,
                        force_complex_fields=False,
                        symmetry=tuple(Mirror(axis, int(phase))
                                       for axis, phase in symmetry),
                        boundaries=dict(keywords.get("boundaries") or {}) or None,
                        dimensions=3, prefer_gpu=prefer_gpu, gpu_id=0)
    cells = float(keywords.get("pml", PML_CELLS))
    folded = {axis.upper() for axis, _phase in symmetry}
    driver.setup_pml({name: ((0.0, cells) if name.upper() in folded else cells)
                      for name in "xyz"})
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
    scale = np.float32(2.0) ** int(scale_bits)
    for array in stored_volumes(driver.fields).values():
        host = (float(amplitude)
                * rng.standard_normal(array.shape)).astype(np.float32) * scale
        array[...] = _module_of(array).asarray(host.astype(np.float32))
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


class CountingPlan:
    """One plan, with its ``run()`` calls counted -- the PLAN-side launch witness.

    ``absorbed_by`` points at the wrapped plan so that :func:`declaring` sees straight
    through it: every reader that asks a plan for its ``rotated_names``, its span or
    its kernel gets the plan, not the counter.
    """

    __slots__ = ("inner", "absorbed_by", "plan_runs")

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.absorbed_by = inner
        self.plan_runs = 0

    def run(self, *args: Any, **kwargs: Any) -> None:
        self.plan_runs += 1
        self.inner.run(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


def declaring(plan: Any) -> Any:
    """The plan whose counters describe the device work in a slot."""
    inner = getattr(plan, "absorbed_by", None)
    if inner is not None and inner is not plan:
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
                 "sync_refusals", "sync_answered", "sync_hazard", "counting_plans")

    def __init__(self, fields: Any, plans: Mapping[str, Any],
                 sync_hazard: bool = False) -> None:
        self.fields = fields
        self.plans = {name: plan for name, plan in plans.items() if plan is not None}
        # ONE COUNTER PER DECLARING OWNER, claimed by the FIRST slot that names it.
        # A fused pair sits in both slots of its seam -- the pair in the first, a
        # ``NoopPlan`` whose ``absorbed_by`` points back at it in the second -- so
        # wrapping every slot would report one launch as two and the two witnesses
        # would disagree by construction. Claiming by declaring owner also carries the
        # gate's own wrappers (the withdraw hoist, the rotation-skipping mutant),
        # whose ``absorbed_by`` is likewise the plan that launches.
        self.counting_plans: List["CountingPlan"] = []
        claimed: List[int] = []

        def wrap(plan: Any) -> Any:
            owner = declaring(plan)
            if id(owner) in claimed:
                return plan
            claimed.append(id(owner))
            wrapper = CountingPlan(plan)
            self.counting_plans.append(wrapper)
            return wrapper

        for slot, plan in list(self.plans.items()):
            if plan is ABSORBED:
                continue
            if isinstance(plan, (list, tuple)):
                self.plans[slot] = [wrap(entry) for entry in plan]
            else:
                self.plans[slot] = wrap(plan)
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
        owner = declaring(plan)
        return tuple(getattr(owner, "replaces_sub_steps", None)
                     or getattr(owner, "replaces", None) or (slot,))

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
        if slot == "update_P" and isinstance(plan, (list, tuple)):
            # THE ONE ASYMMETRIC SLOT, and this shim answers it the way
            # ``FastPathPlan.dispatch`` does (fastpath.py:1818-1827): it holds the
            # LIST of per-susceptibility plans, and each takes ``fields.drive_field``
            # -- MEEP's ``w``, not a stored-E reader. The two agree exactly outside
            # the absorber, so passing E instead is right on every no-PML case and
            # wrong only under PML. Measured 2026-09-06: without the argument every
            # corpus row with a live pole came back unmeasured on a `TypeError` from
            # this shim, which is a harness refusal wearing an engine's clothes.
            drive = fields.drive_field
            for entry in plan:
                entry.run(drive)
        else:
            plan.run()
        self.dispatched[slot] = self.dispatched.get(slot, 0) + 1
        return True

    @property
    def launches(self) -> int:
        """Launches the PLAN side counted -- ``run()`` calls on the plan that WORKS.

        THE WRAPPER GOES ROUND THE PLAN WHOSE ``declaring`` OWNER IS ITSELF, which is
        exactly the one slot of a fused pair that launches; the other slot holds a
        ``NoopPlan`` whose ``absorbed_by`` points back at it. Counting every slot
        instead would report a fused pair as two launches -- the two witnesses would
        then disagree by construction, which is how a real disagreement gets argued
        away. Measured 2026-09-06: the first version of this property read a
        ``launches`` attribute only :class:`.ScratchWeldPairPlan` defines, so the
        certified singles reported ZERO against a kernel counter's eight.
        """
        return sum(wrapper.plan_runs for wrapper in self.counting_plans)


class CountingKernel:
    """An independent launch counter: wraps a Triton kernel and counts launches.

    Triton kernels are subscripted with a grid before they are called, so the counter
    has to intercept ``__getitem__`` rather than ``__call__`` -- a wrapper that only
    forwarded the call would count nothing and report a green zero.
    """

    __slots__ = ("kernel", "calls")

    def __init__(self, kernel: Any) -> None:
        self.kernel = kernel
        self.calls = 0

    def __getitem__(self, grid: Any) -> Any:
        launcher = self.kernel[grid]

        def call(*args: Any, **kwargs: Any) -> Any:
            self.calls += 1
            return launcher(*args, **kwargs)

        return call

    def __getattr__(self, name: str) -> Any:
        return getattr(self.kernel, name)


def count_kernel(plan: Any, kernel: Any) -> Optional[CountingKernel]:
    """Force a plan to launch through a counting wrapper of ``kernel``, or decline.

    DECLINES RATHER THAN WRAPS NOTHING. A plan whose kernel this file cannot resolve
    -- the composer may install any arm on the slots this weld does not own -- would
    otherwise get a wrapper around ``None`` and die at its first launch inside
    ``__getitem__``. Measured 2026-09-06 on ``examples:cavity_arrayslice.py``, whose
    off-diagonal epsilon puts ``offdiag_update_e``'s arm on ``update_E``: the row came
    back unmeasured with a ``TypeError`` from the counter, not from the engine. The
    declined plans are NAMED by :meth:`Arrangement.count_launches` so the leg that
    compares the two witnesses knows which plans one of them cannot see.
    """
    owner = declaring(plan)
    if not hasattr(owner, "_kernel"):
        return None
    resolved = getattr(owner, "_kernel", None) or kernel
    if resolved is None:
        return None
    wrapper = CountingKernel(resolved)
    owner._kernel = wrapper  # noqa: SLF001 - the sibling gates' mutation seam
    return wrapper


# ---------------------------------------------------------------------------
# The five engines, on ONE driver, in lockstep from one seed
# ---------------------------------------------------------------------------

def settle_rotation(plan: Any, fields: Any, originals: Mapping[str, Any]) -> int:
    """After a step that rotated, put the ENGINE's references back on the seed arrays.

    THE WELD SWAPS ``fields.Hx`` AND ITS TWIN AFTER EVERY LAUNCH, and that is the
    product's own certified choreography. But this gate drives FIVE arrangements on
    ONE driver, and the other four bound their pointers to the seed arrays by
    identity; a rotation left standing at the end of a step would leave them writing
    an array the engine no longer names. So once the step is complete the advanced
    values are copied into the seed array, the reference is put back, and the twin
    resumes its role as scratch. The rotation INSIDE the step -- the thing the null
    control catches -- is untouched.
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

    DERIVED FROM THE SHIM, never listed by the caller: the WELD is not the only
    rotating product a step can carry, and a list a caller maintains cannot see a
    product the composer chose.
    """
    if shim is None:
        return ()
    owners: List[Any] = []
    seen: List[int] = []
    for plan in shim.plans.values():
        if plan is ABSORBED:
            continue
        for entry in (plan if isinstance(plan, (list, tuple)) else [plan]):
            owner = declaring(entry)
            if id(owner) in seen or not getattr(owner, "rotated_names", ()):
                continue
            seen.append(id(owner))
            owners.append(owner)
    return tuple(owners)


def rotating_originals(fields: Any, owners: Sequence[Any]) -> Dict[str, Any]:
    """The array the engine names for each rotating volume, BEFORE any launch."""
    return {name: getattr(fields, name)
            for owner in owners for name in owner.rotated_names}


class Arrangement:
    """One engine: a shim (or the array path) and its bookkeeping."""

    __slots__ = ("name", "shim", "counters", "uncounted", "rotating", "originals",
                 "selected", "reasons", "on_step_done", "wipe_private")

    def __init__(self, name: str, shim: Optional[Shim],
                 selected: Optional[Mapping[str, str]] = None,
                 reasons: Optional[Mapping[str, Any]] = None,
                 wipe_private: bool = False) -> None:
        self.name = name
        self.shim = shim
        #: Zero every private (leading-underscore) volume immediately before this
        #: arrangement's step -- the control that MEASURES the private-scratch rule.
        self.wipe_private = bool(wipe_private)
        self.counters: List[CountingKernel] = []
        #: Plans the kernel counter could not reach, by class name and slot. A leg
        #: comparing the two witnesses must know which plans one of them cannot see.
        self.uncounted: List[str] = []
        self.rotating = rotating_owners(shim)
        self.originals = ({} if shim is None
                          else rotating_originals(shim.fields, self.rotating))
        self.selected = dict(selected or {})
        self.reasons = dict(reasons or {})
        self.on_step_done: Optional[Callable[[], None]] = None

    def count_launches(self) -> None:
        """Wrap every plan's kernel ONCE PER DECLARING OWNER, never once per slot.

        A fused pair occupies BOTH slots of its seam, so walking slots would wrap that
        owner's kernel twice -- the second wrapper counting the first -- and the
        independent counter would then report exactly twice the launches the plans
        report, which reads as the two witnesses disagreeing.
        """
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

    def settle(self, fields: Any) -> int:
        return sum(settle_rotation(plan, fields, self.originals)
                   for plan in self.rotating)

    def launches(self) -> Dict[str, int]:
        return {"plans": 0 if self.shim is None else self.shim.launches,
                "kernels": sum(counter.calls for counter in self.counters),
                "uncounted_plans": list(self.uncounted)}


def _default_kernel_for(owner: Any) -> Any:
    """The kernel a plan launches when its ``_kernel`` override is unset.

    LARGER THAN THE PLAIN GATE'S TABLE, and it has to be: a folded ``plan_step``
    installs the folded curl, the folded fills and the two released folded pairs on
    the slots this weld does not own, and a plan the counter cannot resolve makes the
    two launch witnesses incomparable rather than disagreeing (``count_kernel``
    declines, ``leg_launch_structure`` fails by name).
    """
    from meep_gpu.triton_kernels import kernels as tkernels  # noqa: PLC0415

    name = type(owner).__name__
    if name == "FoldedFusedHdPairPlan":
        return family.fused_constitutive_curl_H_to_D_folded_kernel()
    if name == "FusedHdPairPlan":
        return plain.fused_constitutive_curl_H_to_D_kernel()
    if name == "FoldedPmlCurlPlan":
        return tsymmetry.pml_curl_step_folded
    if name == "MirrorGhostFillPlan":
        return tsymmetry.mirror_ghost_fill
    if name == "PmlCurlPlan":
        return tkernels.pml_curl_step
    if name == "ConstitutivePlan":
        return tkernels.constitutive_step
    if name == "FoldedFusedMagneticPairPlan":
        from meep_gpu.triton_kernels import (  # noqa: PLC0415
            folded_fused_magnetic_pair as magnetic,
        )

        return magnetic.folded_fused_curl_constitutive_B_kernel()
    if name == "FoldedFusedPairPlan":
        from meep_gpu.triton_kernels import (  # noqa: PLC0415
            folded_fused_pair as electric,
        )

        return electric.folded_fused_curl_constitutive_D_kernel()
    if name == "FusedPairPlan":
        return (tkernels.fused_curl_constitutive_B
                if getattr(owner, "pair", "B") == "B"
                else tkernels.fused_curl_constitutive_D)
    return None
def arrangement_singles(driver: Any) -> Arrangement:
    """Reference 2: the two CERTIFIED FOLDED singles at the seam's two slots.

    ``plan_folded_constitutive`` builds the certified ``ConstitutivePlan`` around the
    certified ``kernels.constitutive_step`` -- the fold reaches ``update_H`` through
    the stored extent and nothing else -- and ``plan_folded_pml_curl`` launches
    ``symmetry.pml_curl_step_folded``. Two launches on the seam; this weld is one.
    """
    fields, pml = driver.fields, driver.pml
    constitutive = tsymmetry.plan_folded_constitutive(fields, pml, "H")
    curl = tsymmetry.plan_folded_pml_curl(fields, pml, "step_D")
    if constitutive is None or curl is None:
        raise RuntimeError(
            "a certified FOLDED single was refused on a fixture this gate expects it "
            "to admit: constitutive="
            + "; ".join(tsymmetry.folded_constitutive_coverage(
                fields, pml, "H").reasons)
            + " | curl="
            + "; ".join(tsymmetry.folded_composition_curl_coverage(
                fields, pml, "step_D").reasons))
    shim = Shim(fields, {"update_H": constitutive, "step_D": curl})
    return Arrangement("singles", shim,
                       selected={"update_H": "folded", "step_D": "folded PML"})
def _composed(driver: Any, fuse: bool) -> Any:
    """``plan_step`` on this driver -- the SHIPPED composer builds the plans.

    Reaching for the family builders would measure the kernels and skip the
    composition, and the composition is what the arbitration finding is about.
    """
    return triton_launch.plan_step(
        driver.fields, driver.pml,
        sources=tuple(getattr(driver, "_sources", ())), fuse=fuse)


def arrangement_composition(driver: Any, fuse: bool) -> Arrangement:
    """References 3 (``fuse=True``, the composer's own installation) and 4 (unfused)."""
    plan = _composed(driver, fuse)
    plans = {slot: plan.plans[slot] for slot in triton_launch.STEP_ORDER
             if slot in plan.plans}
    if plan.polarization_plans:
        plans["update_P"] = list(plan.polarization_plans)
    return Arrangement("composition_today" if fuse else "unfused",
                       Shim(driver.fields, plans),
                       selected=plan.selected,
                       reasons={key: list(value)
                                for key, value in plan.reasons.items()
                                if "fused" in key})


def build_weld(driver: Any, kernel: Any = None, sources: Any = ()) -> Any:
    plan = family.plan_folded_fused_hd_pair(driver.fields, driver.pml,
                                            sources=sources, kernel=kernel)
    if plan is None:
        reasons = family.folded_fused_hd_pair_coverage(
            driver.fields, driver.pml, sources).reasons
        raise RuntimeError("the folded fused H/D pair was refused: "
                           + "; ".join(reasons))
    return plan
def arrangement_weld(driver: Any, kernel: Any = None,
                     launcher: Optional[Callable[[Any], Any]] = None,
                     sync_hazard: bool = False, hoist: bool = False,
                     rest_unfused: bool = True, name: str = "weld") -> Arrangement:
    """The SUBJECT: the weld force-installed at ``update_H``, ``step_D`` absorbed.

    FORCE-INSTALLED, because the composer refuses this product (``INSTALLABLE`` False,
    and no ``CERTIFIED_FUSED_PRODUCTS`` row on a backend whose label table lives in a
    file this round does not own) and this gate must measure it anyway. Every other
    slot carries the unfused single ``plan_step(fuse=False)`` selects for it, so the
    only difference between this arrangement and reference 4 is the seam.

    ``launcher`` is the HOST mutation seam: it receives the built plan and returns the
    object the slot runs (a rotation-skipping wrapper, a half-integer rebind).
    ``kernel`` is the KERNEL mutation seam, handed to the family's own builder.
    ``hoist`` wraps the launch in the ``LeadingWithdrawPlan`` the installer would.
    """
    sources = tuple(getattr(driver, "_sources", ()))
    plan = build_weld(driver, kernel=kernel, sources=())
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
                       selected={"update_H": "fused pair H->D (folded, forced)",
                                 "step_D": "fused pair H->D (folded, forced)"})


def drive(driver: Any, arrangements: Mapping[str, Arrangement], steps: int,
          reference: str = "array",
          per_step_hook: Optional[Callable[[int, str, Any], Any]] = None,
          progress: Optional[Callable[[str], None]] = None,
          stop_on_divergence: bool = True,
          stop_when_banded: bool = False,
          pairs: Sequence[Tuple[str, str]] = ()) -> Dict[str, Any]:
    """Step every arrangement in lockstep from one seed and compare per complete step.

    ONE driver, restored to each arrangement's own state before its step, captured
    after it -- the withdraw campaign's proven mechanism, extended with the settled
    rotation. ``per_step_hook(step, name, driver)`` runs after each arrangement's step
    (the sync leg's accessor call); its return value is recorded.
    """
    seed = capture(driver)
    # THE ARRAY OBJECT THE ENGINE NAMES FOR EACH STORED VOLUME, BEFORE ANY LAUNCH.
    # Every arrangement's plans hold pointers to these by identity, and ``capture``
    # reads whatever ``fields`` names at the end of a step, so a rotation a step
    # leaves standing silently decouples the two.
    state: Dict[str, Dict[str, Any]] = {
        name: {"volumes": {key: value.copy()
                           for key, value in seed["volumes"].items()},
               "dipoles": list(seed["dipoles"]),
               "step_count": seed["step_count"]}
        for name in arrangements}
    per_step: List[Dict[str, Any]] = []
    hooks: Dict[str, List[Any]] = {name: [] for name in arrangements}
    first_divergence: Optional[int] = None
    first_banded: Optional[int] = None
    moved_words = 0
    words_compared = 0
    for step in range(1, steps + 1):
        for name, arrangement in arrangements.items():
            restore(driver, state[name])
            if arrangement.wipe_private:
                for volume_name, array in stored_volumes(driver.fields).items():
                    if _is_private(volume_name):
                        array.fill(0)
            install(driver, arrangement.shim)
            driver.step()
            arrangement.settle(driver.fields)
            if per_step_hook is not None:
                hooks[name].append(per_step_hook(step, name, driver))
            state[name] = capture(driver)
        pin_array_path(driver)
        # THE PRECONDITION IS CHECKED BEFORE THE COMPARISON, never after: a step whose
        # reference state has entered the denormal band is not compared at all, so a
        # disagreement inside the band can never be counted as an agreement OR as a
        # divergence of this product.
        banded_now = subnormal_census(state[reference])
        if banded_now and first_banded is None:
            first_banded = step
        if stop_when_banded and banded_now:
            if progress is not None:
                progress(f"step {step}/{steps} STOP: the reference state entered the "
                         f"float32 denormal band; no byte claim is made past here")
            break
        row: Dict[str, Any] = {"step": step}
        if step == 1:
            moved_words = sum(differing(seed["volumes"][k],
                                        state[reference]["volumes"][k])
                              for k in seed["volumes"])
        for name in arrangements:
            if name == reference:
                continue
            difference = compare_snapshots(state[reference], state[name])
            private = {key: value for key, value in compare_snapshots(
                state[reference], state[name], include_private=True).items()
                if _is_private(key)}
            row[name] = {"differing_volumes": difference,
                         "differing_words": int(sum(difference.values())),
                         "private_volumes_that_differ": private}
            if difference and not row.get("autopsy"):
                row["autopsy"] = {name: autopsy(state[reference], state[name],
                                                difference)}
        # PAIRWISE, BETWEEN ARRANGEMENTS, and this is what separates a finding about
        # THIS weld from a finding about a sub-step it does not own. Two arrangements
        # that dispatch the same slots and differ only at this seam must agree WORD
        # FOR WORD whatever either of them does against the array path.
        for left, right in pairs:
            difference = compare_snapshots(state[left], state[right])
            row.setdefault("pairs", {})[f"{left}|{right}"] = {
                "differing_volumes": difference,
                "differing_words": int(sum(difference.values()))}
        words_compared += sum(
            int(np.prod(array.shape)) * (2 if array.dtype.name == "complex64" else 1)
            for array in state[reference]["volumes"].values()
        ) * max(len(arrangements) - 1, 1)
        per_step.append(row)
        diverged = any(isinstance(row.get(name), dict)
                       and row[name]["differing_words"] for name in arrangements
                       if name != reference)
        if diverged and first_divergence is None:
            first_divergence = step
        if progress is not None:
            progress(f"step {step}/{steps} "
                     + " ".join(f"{name}={row[name]['differing_words']}"
                                for name in arrangements if name != reference))
        if diverged and stop_on_divergence:
            break
    banded = subnormal_census(state[reference])
    return {
        "first_banded_step": first_banded,
        "steps_compared": len(per_step),
        "steps_requested": steps,
        "bit_identical": first_divergence is None,
        "first_divergence_step": first_divergence,
        "words_compared": int(words_compared),
        "reference_moved_words_step_1": int(moved_words),
        "per_step": per_step,
        "hooks": {name: value for name, value in hooks.items() if any(
            entry is not None for entry in value)},
        "reference_subnormal_words": banded,
        "launches": {name: arrangement.launches()
                     for name, arrangement in arrangements.items()},
        "selected": {name: arrangement.selected
                     for name, arrangement in arrangements.items()},
    }


def all_arrangements(driver: Any, kernel: Any = None,
                     launcher: Optional[Callable[[Any], Any]] = None,
                     count: bool = False) -> Dict[str, Arrangement]:
    """The five engines, built on one driver, in the record's own order."""
    built: Dict[str, Arrangement] = {
        "array": Arrangement("array", None),
        "singles": arrangement_singles(driver),
        "composition_today": arrangement_composition(driver, fuse=True),
        "unfused": arrangement_composition(driver, fuse=False),
        "weld": arrangement_weld(driver, kernel=kernel, launcher=launcher),
        # THE SEAM ALONE: this weld at its two slots, every other slot on the ARRAY
        # PATH. It is what isolates a finding about the seam from a finding about a
        # neighbouring sub-step's own agreement with the array path.
        "weld_seam_only": arrangement_weld(driver, kernel=kernel, launcher=launcher,
                                           rest_unfused=False,
                                           name="weld_seam_only"),
    }
    if count:
        for arrangement in built.values():
            arrangement.count_launches()
    return built


def run_product(driver: Any, steps: int,
                progress: Optional[Callable[[str], None]] = None,
                require_full_budget: bool = True,
                clean_floor: int = 1) -> Dict[str, Any]:
    """The four-reference identity on ONE driver. The core measurement.

    THE SAME BAR ON A LIFTED CORPUS ROW AS ON THE SYNTHETIC FIXTURE: the full budget,
    and every arrangement equal to the array path outright. An earlier revision bounded
    a lifted row's budget at the last step whose reference state carried no word in the
    float32 denormal band and let the seam claim alone carry the row, on the premise
    that "inside the band the CuPy array path and the Triton kernels are measured to
    disagree for a reason that has nothing to do with any weld". THAT PREMISE WAS
    MEASURED FALSE on 2026-09-06 (``results/triton_hd_lift_policy_2026-09-06``): the
    disagreement belonged to the lift leg's CHILD INTERPRETER, which ran with NO
    subnormal policy installed -- native CuPy (``-ftz=true``, flush) against native
    Triton (no ``.ftz`` on arithmetic, keep) -- so the array path flushed the curl
    product ``dtdx * stencil`` where its exact result was subnormal and the Triton
    singles kept it (``0x00000000`` vs ``0x8074a6ab`` on straight-waveguide, cell
    (33, 40, 0), the first step the fields reach 1e-38). With the policy installed in
    the child, the singles and the weld equal the array path on every one of 60 steps
    under BOTH policies, through the band under ``keep``. The band census still rides
    with every row (``first_banded_step``, ``reference_subnormal_words``) so a reader
    can see where the state is; it no longer shortens anything.
    """
    started = time.time()
    arrangements = all_arrangements(driver, count=True)
    #: THE PAIRS THAT CARRY THIS PRODUCT'S OWN CLAIM. ``weld`` and ``unfused``
    #: dispatch the SAME four slots and differ only at this seam;
    #: ``composition_today`` is what the composer installs on these rows. Either can
    #: disagree with the array path for a reason belonging to ``step_B`` or
    #: ``update_E`` -- measured 2026-09-06 on ``examples:gaussian-beam.py``, where all
    #: three disagreed by the SAME 24 words while ``singles`` and ``weld_seam_only``
    #: were identical -- and a comparison that only ever asked "does this equal the
    #: array path" would have read that as this weld's.
    PAIRS = (("weld_seam_only", "singles"), ("weld", "unfused"),
             ("weld", "composition_today"))
    result = drive(driver, arrangements, steps, progress=progress,
                   stop_when_banded=False, stop_on_divergence=False, pairs=PAIRS)
    references = [name for name in arrangements if name != "array"]

    def agrees(name: str) -> bool:
        return all(not row.get(name, {}).get("differing_words")
                   for row in result["per_step"])

    per_reference = {name: agrees(name) for name in references}
    # WHICH SIDE MOVED. A row where the weld still agrees with the certified singles
    # but some arrangement disagrees with the array path is a finding about THAT
    # arrangement, not about this weld, and the record has to say so.
    disagree_with_array = sorted(name for name in references if not per_reference[name])
    def pair_agrees(key: str) -> bool:
        return all(not row.get("pairs", {}).get(key, {}).get("differing_words")
                   for row in result["per_step"])

    pair_agreement = {f"{a}|{b}": pair_agrees(f"{a}|{b}") for a, b in PAIRS}
    # THE SEAM CLAIM, stated as the conjunction it is. Each clause is a comparison
    # this gate makes, and together they say: at ITS OWN two slots this weld is the
    # array path (`weld_seam_only`), the two certified halves it replaces are the
    # array path on this row (`singles`), and against the two whole-step compositions
    # that dispatch the same slots it is identical word for word.
    seam_claim = {
        # THE SHARPEST STATEMENT OF THIS PRODUCT'S OWN CLAIM: ONE launch produces,
        # word for word, what the TWO certified launches it replaces produce -- on the
        # same driver, the same seed and the same complete steps -- whatever either of
        # them does against the array path. Two entirely different kernels: the
        # singles are `constitutive_step` then `pml_curl_step`; this is one fused
        # launch with the foreign recompute and the reference rotation.
        "the_weld_at_its_seam_equals_the_two_certified_singles": pair_agreement[
            "weld_seam_only|singles"],
        "the_weld_equals_the_unfused_composition": pair_agreement["weld|unfused"],
        "the_weld_equals_the_composition_installed_today": pair_agreement[
            "weld|composition_today"],
    }
    result.update({
        "passed": bool(all(seam_claim.values())
                       and result["reference_moved_words_step_1"] > 0
                       and result["steps_compared"] >= clean_floor
                       and (not require_full_budget
                            or result["steps_compared"] == steps)
                       # EVERY ARRANGEMENT EQUALS THE ARRAY PATH OUTRIGHT, on the
                       # synthetic fixture and on a lifted corpus row alike.
                       and result["bit_identical"]),
        "seam_claim": seam_claim,
        "agreement_with_the_array_path": per_reference,
        "pairwise_agreement": pair_agreement,
        "references_that_disagree_with_the_array_path": disagree_with_array,
        "the_seam_alone_agrees_with_the_array_path": per_reference.get(
            "weld_seam_only"),
        "weld_agrees_with_the_certified_singles": (
            pair_agreement["weld|unfused"]
            and bool(per_reference.get("weld_seam_only"))),
        "the_certified_singles_agree_with_the_array_path": bool(
            per_reference.get("singles")),
        "what_a_shared_divergence_means": (
            "a dispatched arrangement that disagrees with the CuPy array path is "
            "named with its autopsy in references_that_disagree_with_the_array_path "
            "and the row FAILS: every arrangement must equal the array path over the "
            "full budget, on the synthetic fixture and on a lifted corpus row alike. "
            "The pairwise comparisons stay in the record so a failure can be "
            "ATTRIBUTED -- to this seam (weld_seam_only differs from singles) or to a "
            "sub-step this product does not own (weld equals unfused and both differ "
            "from the array path) -- but attribution is not a pass. The 2026-09-06 "
            "campaign's zero -> subnormal divergences on every driven row were the "
            "lift child running with no subnormal policy installed "
            "(results/triton_hd_lift_policy_2026-09-06), not a fact about any kernel"),
        "seconds": round(time.time() - started, 2),
    })
    return result


# ---------------------------------------------------------------------------
# HOST LEG: the driver's own order
# ---------------------------------------------------------------------------

def leg_driver_order() -> Dict[str, Any]:
    """``REPLACES`` is the driver's two adjacent consults, and nothing else is between.

    READ OFF THE TREE. A gate that spelled the seam would keep reading correctly after
    a driver move while the bytes described a step nothing performs.
    """
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
                        f"not ['step_D']; this product's REPLACES is not two adjacent "
                        f"consults")
    # WHAT SITS BETWEEN THEM, as statements rather than as a claim.
    between = body.split('fast.dispatch("update_H"', 1)[1]
    between = between.split('fast.dispatch("step_D"', 1)[0]
    statements = [line.strip() for line in between.splitlines()
                  if line.strip() and not line.strip().startswith("#")]
    # The update_H consult's own two lines close first.
    statements = [line for line in statements if "update_H(self.fields" not in line]
    expected_withdraw = any("withdraw" in line for line in statements)
    if not expected_withdraw:
        findings.append("no withdraw statement stands between the two consults; the "
                        "seam this product spans is not the seam withdraw_hoist "
                        "describes")
    if any("inject" in line for line in statements):
        findings.append("an injection stands between the two consults; "
                        "CARRIES_DEPOSIT_REPAIR = False is then false about the driver")
    if family.REPLACES != withdraw_hoist.SEAM_SPAN:
        findings.append(f"REPLACES {family.REPLACES} is not withdraw_hoist.SEAM_SPAN "
                        f"{withdraw_hoist.SEAM_SPAN}")
    if family.SEAM != h_to_d_seam.SEAM:
        findings.append(f"the product's SEAM {family.SEAM!r} is not the boards' "
                        f"{h_to_d_seam.SEAM!r}")
    # THE FOLD-SIDE PASSES, BY NAME: all three B-side ones close BEFORE the first
    # consult and all three D-side ones open AFTER the second, so the seam this
    # product spans is fill-FREE on a folded grid and nothing has to be carried
    # across the launch. Read off the statements, not asserted.
    for pass_name in ("fill_symmetry_bc_B", "fill_symmetry_bc_D", "zero_metal_B",
                      "zero_metal_D", "fill_folded_far_ghosts_B",
                      "fill_folded_far_ghosts_D"):
        if any(pass_name in line for line in statements):
            findings.append(
                f"{pass_name} runs BETWEEN the two consults; on a folded grid this "
                f"seam would then carry a fill and this product performs none")
    # THE SYNC CHANNEL'S CONTAINMENT RULE, read off fastpath rather than asserted.
    if "step_D" in SYNC_PATH_SLOTS:
        findings.append("step_D is inside SYNC_PATH_SLOTS; the containment rule would "
                        "then admit this weld into the magnetic half-step, where D is "
                        "in neither backup list")
    if SYNC_PASS_OWNERS.get(SYNC_UPDATE_H_PASS) != "update_H":
        findings.append("the sync channel no longer owns update_H by name")
    # The declared flags against the driver's own facts.
    if family.CARRIES_DEPOSIT_REPAIR:
        findings.append("CARRIES_DEPOSIT_REPAIR is True and nothing is injected in "
                        "this seam")
    if family.HOISTS_THE_WITHDRAW:
        findings.append("HOISTS_THE_WITHDRAW is True while INSTALLABLE is False; the "
                        "hoist branch is then unreachable and nothing performs the "
                        "withdraw before the launch")
    if family.INSTALLABLE:
        findings.append("INSTALLABLE is True; this round's arbitration verdict is that "
                        "it must not be")
    return {"passed": not findings, "findings": findings,
            "consult_order": order,
            "statements_between_the_two_consults": statements,
            "the_seam_is_fill_free_on_a_fold": True,
            "replaces": list(family.REPLACES),
            "sync_path_slots": list(SYNC_PATH_SLOTS),
            "installable": family.INSTALLABLE,
            "installable_reason": family.INSTALLABLE_REASON}


# ---------------------------------------------------------------------------
# HOST LEG: the transcription
# ---------------------------------------------------------------------------

def leg_transcription() -> Dict[str, Any]:
    """Both halves are the certified kernels' own text, plus the declared edits.

    THE CURL HALF IS CUT OUT OF ``symmetry.py`` AND THE CONSTITUTIVE HALF IS NOT CUT
    AT ALL: it is imported. So this leg asserts three things the plain gate does not
    have to -- that the two certified curl bodies are DIFFERENT text, that the fold's
    own three deltas are in ours and not in the plain one, and that this module
    defines no copy of ``_h_cell``/``_h_tap``.
    """
    findings: List[str] = []
    measures: Dict[str, Any] = {}

    # 1. THE CONSTITUTIVE HALF IS THE PLAIN PRODUCT'S, AND IS IMPORTED.
    certified_constitutive = plain.certified_constitutive_tail()
    lifted_constitutive = plain.lifted_constitutive_tail()
    measures["constitutive_chars"] = len(certified_constitutive)
    if certified_constitutive != lifted_constitutive:
        findings.append(
            f"fused_hd_pair._h_cell is NOT kernels.constitutive_step's own body with "
            f"the declared lift edits: {len(certified_constitutive)} chars against "
            f"{len(lifted_constitutive)}; this family imports that body, so the "
            f"drift is this family's too")
    module_text = (API_ROOT / "meep_gpu" / "triton_kernels"
                   / "folded_fused_hd_pair.py").read_text(encoding="utf-8")
    tree = ast.parse(module_text)
    defined = {node.name for node in ast.walk(tree)
               if isinstance(node, ast.FunctionDef)}
    for name in ("_h_cell", "_h_tap"):
        if name in defined:
            findings.append(f"this module DEFINES {name}; a second copy of the lifted "
                            f"constitutive body is a second place for it to drift")
    if "from .fused_hd_pair import _h_cell, _h_tap" not in module_text:
        findings.append("the constitutive device functions are not imported from the "
                        "plain product")
    measures["constitutive_is_imported_not_copied"] = (
        "_h_cell" not in defined and "_h_tap" not in defined)

    # 2. THE CURL HALF IS THE FOLDED BODY WITH THE NINE REDIRECTS.
    certified_curl = family.certified_curl_tail()
    lifted_curl = family.lifted_curl_tail()
    measures["curl_chars"] = len(certified_curl)
    if certified_curl != lifted_curl:
        findings.append(
            f"the weld's curl half is NOT symmetry.pml_curl_step_folded's own body "
            f"with the nine redirected magnetic reads: {len(certified_curl)} chars "
            f"against {len(lifted_curl)}")
    measures["curl_lift_edits"] = len(family.folded_curl_lift_edits())
    if measures["curl_lift_edits"] != 9:
        findings.append(f"the curl half takes {measures['curl_lift_edits']} edits, "
                        f"not the three own-cell reads plus six shifted taps")

    # 3. THE TWO CERTIFIED CURLS ARE DIFFERENT TEXT, AND THE FOLD'S DELTAS ARE OURS.
    folded_raw = family.raw_folded_curl_tail()
    plain_raw = plain._cut(plain._source_of("pml_curl_step"),  # noqa: SLF001
                           plain.DECODE_END)
    if folded_raw == plain_raw:
        findings.append("the folded and plain certified curl bodies are the SAME "
                        "text; this family would then be cutting the wrong kernel")
    deltas = {
        "inverted_ghost_branch": ("if BCX == PERIODIC:" in folded_raw
                                  and "if BCX == PERIODIC:" not in plain_raw),
        "cell_zero_mask_widened_to_not_periodic": ("!= PERIODIC" in folded_raw
                                                   and "!= PERIODIC" not in plain_raw),
        "top_plane_mask_on_mirror_periodic": (
            "MIRROR_PERIODIC" in folded_raw and "MIRROR_PERIODIC" not in plain_raw),
    }
    measures["the_folds_three_deltas"] = deltas
    for name, held in deltas.items():
        if not held:
            findings.append(f"the fold's delta {name!r} is not present in the folded "
                            f"body and absent from the plain one")
    # ...and each survives the lift, which is the half that matters.
    for needle_text in ("if BCX == PERIODIC:", "!= PERIODIC", "MIRROR_PERIODIC"):
        if needle_text not in lifted_curl:
            findings.append(f"{needle_text!r} did not survive into the welded text")
    measures["top_plane_mask_lines"] = lifted_curl.count("MIRROR_PERIODIC")
    if measures["top_plane_mask_lines"] != folded_raw.count("MIRROR_PERIODIC"):
        findings.append("the lift changed the number of top-plane mask clauses")

    # 4. NON-VACUITY: an anchor pair that crossed would compare two short strings.
    for label, text in (("constitutive", certified_constitutive),
                        ("curl", certified_curl)):
        statements = [line for line in text.splitlines()
                      if line.strip() and not line.strip().startswith("#")]
        measures[f"{label}_statements"] = len(statements)
        if len(statements) < 15:
            findings.append(f"the lifted {label} body is only {len(statements)} "
                            f"statements; the anchors have crossed")

    # 5. THE TAP TABLE IS PARSED FROM THE FOLDED EMITTER'S OWN LINES.
    taps = plain.halo_taps(folded_raw)
    offsets = plain.offset_coordinates(folded_raw)
    measures["halo_taps"] = {name: list(value) for name, value in sorted(taps.items())}
    measures["offset_coordinates"] = {name: list(value)
                                      for name, value in sorted(offsets.items())}
    if {name: value[0] for name, value in taps.items()} != dict(plain.HALO_TAPS):
        findings.append("the folded curl's shifted magnetic loads are not the six "
                        "declared taps; this family redirects one load per tap")
    for register, component in plain.HALO_TAPS:
        _component, offset, mask = taps[register]
        coordinates = ", ".join(offsets[offset])
        wanted = f"{register} = _h_tap({component}, {coordinates}, {mask},"
        if wanted not in lifted_curl:
            findings.append(f"the redirect for {register} does not land on the cell "
                            f"the folded emitter's own index line composed")

    # 6. No magnetic or inverse-epsilon pointer survives below the seam.
    for stem in ("g0", "g1", "g2", "e0", "e1", "e2"):
        if f"{stem} +" in lifted_curl or f"{stem} +" in lifted_constitutive:
            findings.append(f"{stem} survives in the welded text; in this signature "
                            f"that pointer does not exist")

    # 7. ASCII EVERYWHERE THIS MODULE WRITES ITS OWN PROSE, and NOT inside the lifted
    #    body -- because the lifted body is `symmetry.pml_curl_step_folded`'s own
    #    text, character for character, and that text carries four em dashes in its
    #    mask comments. Making the module ASCII would break the byte equality above,
    #    so the clause is the one that is actually true: every non-ASCII line in this
    #    file is a line of the certified body it lifts, and nothing else.
    certified_lines = {line.strip() for line in certified_curl.splitlines()}
    strays = [line.strip() for line in module_text.splitlines()
              if not line.isascii() and line.strip() not in certified_lines]
    measures["non_ascii_lines_in_the_module"] = len(
        [line for line in module_text.splitlines() if not line.isascii()])
    measures["non_ascii_lines_that_are_the_lifted_bodys_own"] = (
        measures["non_ascii_lines_in_the_module"] - len(strays))
    if strays:
        findings.append(
            f"the family module carries {len(strays)} non-ASCII line(s) that are NOT "
            f"the certified folded body's own: {strays[:3]}")

    # 8. EVERY MUTATION NEEDLE RESOLVES EXACTLY ONCE, IN THE FILE IT NAMES, checked
    #    before any device time. A needle that matched nothing is a DISARMED leg.
    texts = {"folded": module_text,
             "plain": (API_ROOT / "meep_gpu" / "triton_kernels"
                       / "fused_hd_pair.py").read_text(encoding="utf-8")}
    unresolved = []
    for tag, spec in MUTATIONS.items():
        if spec["target"] != "kernel":
            continue
        where = spec.get("module", "folded")
        # A MUTATION MAY CARRY SEVERAL EDITS -- the paired control that earns a null
        # is two -- and EVERY needle of EVERY one has to resolve exactly once.
        for old_text, _new_text in (spec.get("edits")
                                    or ((spec["old"], spec["new"]),)):
            hits = texts[where].count(old_text)
            if hits != 1:
                unresolved.append(f"{tag} in {where}: {hits} matches "
                                  f"for {old_text.strip()[:50]!r}")
    if texts[BYTE_NEUTRAL.get("module", "folded")].count(BYTE_NEUTRAL["old"]) != 1:
        unresolved.append("the byte-neutral needle")
    if unresolved:
        findings.append("mutation needles that do not resolve exactly once: "
                        + "; ".join(unresolved))
    measures["mutation_needles"] = len([1 for s in MUTATIONS.values()
                                        if s["target"] == "kernel"])

    # 9. The shared coefficient group's premise, driven rather than asserted.
    try:
        measures["sub_lattice_suffixes"] = list(
            plain._sub_lattice_suffixes())  # noqa: SLF001
    except AssertionError as error:
        findings.append(str(error))

    # 10. The four ghost codes are symmetry's own integers.
    measures["ghost_codes"] = [tsymmetry.CODE_PERIODIC, tsymmetry.CODE_METALLIC,
                               tsymmetry.CODE_MIRROR_METALLIC,
                               tsymmetry.CODE_MIRROR_PERIODIC]
    if measures["ghost_codes"] != [0, 1, 2, 3]:
        findings.append(f"the four ghost codes are {measures['ghost_codes']}")
    for name in ("CODE_PERIODIC", "CODE_METALLIC", "CODE_MIRROR_METALLIC",
                 "CODE_MIRROR_PERIODIC"):
        if f"_symmetry.{name}" not in module_text:
            findings.append(f"the kernel does not take {name} from symmetry")
    return {"passed": not findings, "findings": findings, "measures": measures}
# ---------------------------------------------------------------------------
# HOST LEG: the refusals, by name
# ---------------------------------------------------------------------------

class _GridView:
    """The REAL grid with one attribute overridden, for a refusal stand-in.

    A hand-built grid would be a second model of what a grid is, and a predicate that
    refused it would be refusing the model. Proxying the live object means every
    clause but the one under test reads exactly what it reads on a real run.
    """

    def __init__(self, grid: Any, **overrides: Any) -> None:
        object.__setattr__(self, "_grid", grid)
        object.__setattr__(self, "_overrides", dict(overrides))

    def __getattr__(self, name: str) -> Any:
        overrides = object.__getattribute__(self, "_overrides")
        if name in overrides:
            return overrides[name]
        return getattr(object.__getattribute__(self, "_grid"), name)


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
    """Every configuration outside the cell is refused BY NAME, with its reason.

    And the DISJOINTNESS is driven rather than argued: on each fixture exactly one of
    the two H->D products admits, so a composer offered both could not pick the wrong
    one by asking in the wrong order.
    """
    findings: List[str] = []
    rows: List[Dict[str, Any]] = []

    def check(name: str, fields: Any, pml: Any, sources: Any, must_refuse: bool,
              needle: Optional[str] = None) -> None:
        verdict = family.folded_fused_hd_pair_coverage(fields, pml, sources)
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
    # 3. A standing INTEGRATED electric withdraw -- the cell's three refused rows.
    check("integrated_electric_withdraw", fields, pml, (_source(),), True,
          "standing integrated")
    # 4. A NON-integrated electric source is NOT this seam's business.
    check("non_integrated_electric_source", fields, pml,
          (_source(integrated=False),), False)
    # 5. A MAGNETIC source is withdrawn one seam earlier.
    check("integrated_magnetic_source", fields, pml,
          (_source(component="Hz", field_type="B"),), False)
    # 6. A cylindrical grid.
    cylindrical = _FieldsView(fields, _GridView(fields.grid, cylindrical=True))
    check("cylindrical_grid", cylindrical, pml, (), True, "cylindrical")
    # 7. A conductivity on a step_D target names the CURL half.
    conductive = _ConductiveView(fields)
    check("conductive_step_D", conductive, pml, (), True, "folded curl half")
    # 8. A fold whose phase cannot be read is refused by the classifier, by name.
    unreadable = _FieldsView(fields, _GridView(fields.grid,
                                               mirror_phase=lambda axis: 0))
    check("unreadable_mirror_phase", unreadable, pml, (), True, "mirror phase")
    # 9. An inactive absorber, named by the CONSTITUTIVE half -- the driver's order.

    class _Inactive:
        is_active = False

    verdict = family.folded_fused_hd_pair_coverage(fields, _Inactive(), ())
    rows.append({"case": "inactive_absorber", "covered": verdict.covered,
                 "reasons": list(verdict.reasons)})
    if verdict.covered:
        findings.append("inactive_absorber: admitted, and must be refused")
    elif not verdict.reasons[0].startswith("folded constitutive half"):
        findings.append(
            f"inactive_absorber: the FIRST refusal is {verdict.reasons[0]!r}; the "
            f"driver reaches update_H first, so the constitutive half must speak "
            f"first")
    driver.close()

    # 10. AN UNFOLDED GRID -- a real one, not a proxy -- refused, naming the plain
    #     product; and the plain product ADMITS it. The inverse of clause 11.
    unfolded_driver = build_driver({"boundaries": {"x": "metallic", "y": "metallic"}},
                                   seed=1)
    try:
        check("unfolded_grid", unfolded_driver.fields, unfolded_driver.pml, (), True,
              "fused_hd_pair")
        rows[-1]["also_names"] = "no axis is folded"
        plain_on_unfolded = plain.fused_hd_pair_coverage(
            unfolded_driver.fields, unfolded_driver.pml, ())
        if not plain_on_unfolded.covered:
            findings.append(
                f"the PLAIN product refuses the unfolded fixture this one hands it: "
                f"{list(plain_on_unfolded.reasons)}")
    finally:
        unfolded_driver.close()

    # 11. DISJOINTNESS, DRIVEN ON EVERY FIXTURE: exactly one product admits each.
    disjoint: Dict[str, Any] = {}
    for name, keywords in CASES:
        case_driver = build_driver(dict(keywords), seed=1)
        try:
            folded_ok = family.folded_fused_hd_pair_coverage(
                case_driver.fields, case_driver.pml, ()).covered
            plain_ok = plain.fused_hd_pair_coverage(
                case_driver.fields, case_driver.pml, ()).covered
        finally:
            case_driver.close()
        disjoint[name] = {"folded_admits": folded_ok, "plain_admits": plain_ok}
        if not folded_ok:
            findings.append(f"{name}: this product refuses one of its own fixtures")
        if plain_ok:
            findings.append(f"{name}: the PLAIN product admits a folded fixture; the "
                            f"two are not disjoint and a composer could pick either")
    return {"passed": not findings, "findings": findings, "cases": rows,
            "disjointness_over_the_fixtures": disjoint,
            "what_the_disjointness_licenses": (
                "that the two H->D products partition their configurations by "
                "MEASUREMENT rather than by the order a composer asks them in: the "
                "fold is required here and refused there, on every fixture this gate "
                "drives, in both directions")}
class _FieldsView:
    """A ``Fields`` with one attribute overridden, for a refusal stand-in."""

    def __init__(self, fields: Any, grid: Any) -> None:
        self._fields = fields
        self.grid = grid

    def __getattr__(self, name: str) -> Any:
        return getattr(self._fields, name)


class _ConductiveView:
    """A ``Fields`` reporting a conductivity on the ``step_D`` targets.

    The curl half's own clause 8 (``pml_curl_coverage``) hands a conductive run to the
    conductive product; this stand-in drives that refusal rather than assuming it.
    """

    def __init__(self, fields: Any) -> None:
        self._fields = fields

    def condfac_for(self, target: str) -> Any:
        return object() if target in ("Dx", "Dy", "Dz") else None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._fields, name)


# ---------------------------------------------------------------------------
# HOST LEG: the arbitration
# ---------------------------------------------------------------------------

def _inject_product_row() -> Callable[[], None]:
    """Put this product's row into the composer's tables IN PROCESS, and undo it.

    THE COUNTERFACTUAL IS MEASURED RATHER THAN ARGUED. The shipped tables do not carry
    this product -- on this backend a composer label must also appear in
    ``fastpath.FUSED_ARM_CONSTITUENTS``, which is in a file this round does not own --
    so the arbitration cannot be observed from the tree as it stands. Injecting the
    row here drives the two brakes the composition would rely on, on the real
    composer, without editing anything.
    """
    products = triton_launch.CERTIFIED_FUSED_PRODUCTS
    arms = triton_launch.CERTIFIED_FUSED_PAIR_ARMS
    name = family.FAMILY
    products[name] = {"curl_slot": "update_H", "module": "folded_fused_hd_pair",
                      "coverage": "folded_fused_hd_pair_coverage",
                      "builder": "plan_folded_fused_hd_pair",
                      "label": "fused pair H->D (folded)"}
    arms[name] = family.ARMS
    original_modules = triton_launch._certified_fused_product_modules  # noqa: SLF001

    def modules():
        table = original_modules()
        table["folded_fused_hd_pair"] = family
        return table

    triton_launch._certified_fused_product_modules = modules  # noqa: SLF001

    def undo() -> None:
        products.pop(name, None)
        arms.pop(name, None)
        triton_launch._certified_fused_product_modules = original_modules  # noqa: SLF001

    return undo
def leg_arbitration() -> Dict[str, Any]:
    """Who holds ``update_H`` on a FOLDED row this product's predicate admits.

    Two brakes, driven separately on the real composer: ``_pair_may_absorb`` (the
    released ``fused pair B (folded)`` installs FIRST and holds the slot) and
    ``_declared_uninstallable`` (the flag). Neither is asserted from a docstring, and
    the incumbent is required to KEEP its slots with the row injected -- the clause
    that would catch this product taking a launch away from a released pair.
    """
    findings: List[str] = []
    per_case: Dict[str, Any] = {}
    injected_selected: Dict[str, Any] = {}
    absorb_refusal = None
    uninstallable = None
    refusals: List[str] = []
    for name, keywords in CASES:
        driver = build_driver(dict(keywords), seed=3)
        fields, pml = driver.fields, driver.pml
        try:
            admitted = family.folded_fused_hd_pair_coverage(fields, pml, ()).covered
            if not admitted:
                findings.append(f"{name}: the arbitration fixture is not one this "
                                f"product admits")
            shipped = triton_launch.plan_step(fields, pml, sources=(), fuse=True)
            shipped_selected = dict(shipped.selected)
            holder = shipped_selected.get("update_H")
            if holder is None or "fused pair" not in str(holder):
                findings.append(
                    f"{name}: update_H is held by {holder!r}, so the arbitration this "
                    f"product loses is not the one INSTALLABLE_REASON describes")
            if shipped_selected.get("step_B") != holder:
                findings.append(
                    f"{name}: step_B is held by {shipped_selected.get('step_B')!r} and "
                    f"update_H by {holder!r}; the folded B->H pair does not hold both")
            undo = _inject_product_row()
            try:
                injected = triton_launch.plan_step(fields, pml, sources=(), fuse=True)
                case_selected = dict(injected.selected)
                case_refusals = [reason for key, value in injected.reasons.items()
                                 if family.FAMILY in key for reason in value]
                case_absorb = triton_launch._pair_may_absorb(  # noqa: SLF001
                    case_selected, "update_H", "step_D", family.ARMS)
                case_uninstallable = triton_launch._declared_uninstallable(  # noqa: SLF001
                    {"folded_fused_hd_pair": family}, family.FAMILY,
                    triton_launch.CERTIFIED_FUSED_PRODUCTS[family.FAMILY])
            finally:
                undo()
            if case_selected.get("update_H") != holder:
                findings.append(
                    f"{name}: with the row injected update_H moved from {holder!r} to "
                    f"{case_selected.get('update_H')!r}; the incumbent must keep it")
            if case_selected.get("step_D") != shipped_selected.get("step_D"):
                findings.append(
                    f"{name}: step_D moved from {shipped_selected.get('step_D')!r} to "
                    f"{case_selected.get('step_D')!r} with the row injected")
            if not case_refusals:
                findings.append(f"{name}: the composer records no refusal naming this "
                                f"product")
            if case_absorb is None:
                findings.append(
                    f"{name}: _pair_may_absorb does NOT refuse this product against "
                    f"the live selected table")
            if case_uninstallable is None:
                findings.append(f"{name}: _declared_uninstallable does not refuse it")
            elif "INSTALLABLE = False" not in case_uninstallable:
                findings.append(f"{name}: the flag refusal does not name the flag: "
                                f"{case_uninstallable!r}")
            per_case[name] = {
                "predicate_admits": admitted,
                "shipped_selected": shipped_selected,
                "selected_with_the_row_injected": case_selected,
                "incumbent_kept_update_H": case_selected.get("update_H") == holder,
                "composer_refusals": case_refusals}
            injected_selected = case_selected
            refusals = case_refusals
            absorb_refusal = case_absorb
            uninstallable = case_uninstallable
        finally:
            driver.close()

    # THE SEAM ROW EXISTS AND IS LAST, and its pair name is the withdraw hoist's.
    seams = triton_launch.CERTIFIED_FUSED_PAIR_SEAMS
    if seams.get("update_H") != ("step_D", withdraw_hoist.SEAM):
        findings.append(f"the composer's H->D seam row is {seams.get('update_H')!r}")
    if list(seams)[-1] != "update_H":
        findings.append("the H->D seam row is not the last row of the seam table; "
                        "offered earlier it would find an ARM label on update_H")
    if family.FAMILY in triton_launch.CERTIFIED_FUSED_PRODUCTS:
        findings.append("the shipped tables carry this product; the injection above "
                        "was not a counterfactual")
    return {"passed": not findings, "findings": findings,
            "per_case": per_case,
            "selected_with_the_row_injected": injected_selected,
            "composer_refusals": refusals,
            "pair_may_absorb_refusal": absorb_refusal,
            "declared_uninstallable_refusal": uninstallable,
            "what_this_measures": (
                "the composition WITHOUT this product, and the two brakes that would "
                "hold it out if it were offered. It is not a composition-installed "
                "reference: no composer installs this product, so the byte comparison "
                "against an installed weld awaits wiring and is named as owed")}
# ---------------------------------------------------------------------------
# HOST LEG: the purity / race ledger
# ---------------------------------------------------------------------------

def purity_ledger(fields: Any, pml: Any) -> Dict[str, Any]:
    """How many of the FOLDED curl's valid foreign taps land on a cell ``update_H`` MOVED.

    THIS IS THE LEDGER THIS SHAPE UNIQUELY OWES, and the fold adds a second column to
    it. The weld's claim is that a foreign tap RECOMPUTES a value an in-place
    arrangement would have raced on; if ``update_H`` moved nothing at the cells the
    curl taps, the recompute and the stale read would agree by accident and the
    identity result would license nothing about the hazard.

    THE FOLD-SPECIFIC COLUMN is the NEAR-GHOST PLANE. Stored cell 0 of a folded axis
    is a real stored cell whose ``B`` the driver's near symmetry fill wrote before the
    seam opened, and it IS read -- by the thread at index 1. A tap landing there is
    only proof of a recompute if ``update_H`` moved that cell too, so both counts are
    reported and neither is inferred from the other.

    THE GUARD IS THE FOLDED KERNEL'S, not the plain one's: ``pml_curl_step_folded``
    wraps on PERIODIC and masks on EVERY other code, the two mirror codes included.
    Taking the plain kernel's `metallic` test here would count a folded axis as
    wrapping and the ledger would describe a kernel nobody launches.
    """
    codes, code_reasons = tsymmetry.folded_axis_kinds(fields.grid, pml)
    if codes is None:
        return {"codes": None, "reasons": list(code_reasons),
                "valid_foreign_taps": 0, "foreign_taps_on_a_cell_update_H_moved": 0,
                "split_field_history_words_moved": 0}
    before = {name: _host(getattr(fields, name)).copy()
              for name in family.ROTATED}
    stepping.update_H(fields, pml)
    after = {name: _host(getattr(fields, name)).copy() for name in family.ROTATED}
    moved = {name: (words(before[name]) != words(after[name])).reshape(
        tuple(fields.grid.shape)) for name in family.ROTATED}
    nx, ny, nz = (int(n) for n in fields.grid.shape)
    index = np.indices((nx, ny, nz))
    i, j, k = index[0], index[1], index[2]
    shifted = {"x": i - 1, "y": j - 1, "z": k - 1}
    extents = {"x": nx, "y": ny, "z": nz}
    folded_axis = {"x": codes[0] in (tsymmetry.CODE_MIRROR_METALLIC,
                                     tsymmetry.CODE_MIRROR_PERIODIC),
                   "y": codes[1] in (tsymmetry.CODE_MIRROR_METALLIC,
                                     tsymmetry.CODE_MIRROR_PERIODIC),
                   "z": codes[2] in (tsymmetry.CODE_MIRROR_METALLIC,
                                     tsymmetry.CODE_MIRROR_PERIODIC)}
    valid = {}
    lands_on_near_ghost = {}
    for axis, code in zip("xyz", codes):
        shift = shifted[axis]
        if code == tsymmetry.CODE_PERIODIC:
            valid[axis] = np.ones_like(shift, dtype=bool)
            shifted[axis] = np.where(shift < 0, extents[axis] - 1, shift)
        else:
            valid[axis] = (shift >= 0) & (shift < extents[axis])
            shifted[axis] = np.clip(shift, 0, extents[axis] - 1)
        lands_on_near_ghost[axis] = (shifted[axis] == 0) & valid[axis] \
            if folded_axis[axis] else np.zeros_like(shift, dtype=bool)
    # The six taps, named exactly as the certified folded curl names them.
    taps = {
        "a_y": ("Hx", (i, shifted["y"], k), valid["y"], lands_on_near_ghost["y"]),
        "a_z": ("Hx", (i, j, shifted["z"]), valid["z"], lands_on_near_ghost["z"]),
        "b_x": ("Hy", (shifted["x"], j, k), valid["x"], lands_on_near_ghost["x"]),
        "b_z": ("Hy", (i, j, shifted["z"]), valid["z"], lands_on_near_ghost["z"]),
        "c_x": ("Hz", (shifted["x"], j, k), valid["x"], lands_on_near_ghost["x"]),
        "c_y": ("Hz", (i, shifted["y"], k), valid["y"], lands_on_near_ghost["y"]),
    }
    per_tap: Dict[str, Dict[str, int]] = {}
    total_valid = 0
    total_moved = 0
    total_ghost = 0
    total_ghost_moved = 0
    for name, (volume, coordinates, guard, ghost) in taps.items():
        landed = moved[volume][coordinates]
        n_valid = int(np.count_nonzero(guard))
        n_moved = int(np.count_nonzero(landed & guard))
        n_ghost = int(np.count_nonzero(ghost))
        n_ghost_moved = int(np.count_nonzero(landed & ghost))
        per_tap[name] = {"volume": volume, "valid_taps": n_valid,
                         "taps_on_a_moved_cell": n_moved,
                         "taps_on_the_near_ghost_plane": n_ghost,
                         "taps_on_a_MOVED_near_ghost_cell": n_ghost_moved}
        total_valid += n_valid
        total_moved += n_moved
        total_ghost += n_ghost
        total_ghost_moved += n_ghost_moved
    history_moved = int(np.count_nonzero(moved["f_w_Hx"]) +
                        np.count_nonzero(moved["f_w_Hy"]) +
                        np.count_nonzero(moved["f_w_Hz"]))
    return {
        "codes": [int(code) for code in codes],
        "cells": nx * ny * nz,
        "per_tap": per_tap,
        "valid_foreign_taps": total_valid,
        "foreign_taps_on_a_cell_update_H_moved": total_moved,
        "foreign_taps_on_the_near_ghost_plane": total_ghost,
        "foreign_taps_on_a_MOVED_near_ghost_cell": total_ghost_moved,
        "fraction_of_valid_taps_that_would_have_raced": (
            round(total_moved / total_valid, 6) if total_valid else 0.0),
        "split_field_history_words_moved": history_moved,
        "what_this_licenses": (
            "the count of foreign taps whose RECOMPUTED value differs from what an "
            "in-place arrangement would have read at the moment the neighbour had not "
            "yet been written, and -- separately -- how many of those land on the "
            "near-ghost plane of a folded axis, the cells this product's own docstring "
            "argues are ordinary update_H outputs. A near-zero first count would mean "
            "the byte-identity result says nothing about the hazard. It does NOT "
            "measure a race: nothing here launches, and the in-place arrangement's "
            "actual read order is schedule-dependent and is not reproduced"),
    }
def leg_purity_ledger() -> Dict[str, Any]:
    """The ledger on every folded specialisation, with its non-vacuity bar."""
    findings: List[str] = []
    rows: Dict[str, Any] = {}
    ghost_fixtures = 0
    for name, keywords in CASES:
        driver = build_driver(dict(keywords), seed=11, prefer_gpu=False)
        rows[name] = purity_ledger(driver.fields, driver.pml)
        driver.close()
        ledger = rows[name]
        if ledger.get("codes") is None:
            findings.append(f"{name}: the folded axis codes did not resolve "
                            f"({ledger.get('reasons')})")
            continue
        if ledger["foreign_taps_on_a_cell_update_H_moved"] == 0:
            findings.append(
                f"{name}: NO foreign tap lands on a cell update_H moved, so the "
                f"identity result licenses nothing about the read-after-write hazard "
                f"on this fixture")
        if ledger["split_field_history_words_moved"] == 0:
            findings.append(
                f"{name}: update_H moves no f_w_H word, so the second premise of the "
                f"hazard (a racing neighbour reading B where it needs B_prev) is not "
                f"live here")
        if ledger["foreign_taps_on_a_MOVED_near_ghost_cell"] > 0:
            ghost_fixtures += 1
    if not ghost_fixtures:
        findings.append(
            "on NO fixture does a foreign tap land on a MOVED near-ghost cell of a "
            "folded axis; the fold-specific half of this ledger is then vacuous and "
            "the claim that the near-ghost plane is an ordinary update_H output is "
            "untested by any tap")
    return {"passed": not findings, "findings": findings, "fixtures": rows,
            "fixtures_whose_taps_reach_a_moved_near_ghost_cell": ghost_fixtures}
# ---------------------------------------------------------------------------
# HOST LEG: is the metallic ghost observable at all?
# ---------------------------------------------------------------------------

#: Which curl component each shifted magnetic tap feeds, and on which axis its ghost
#: fires. READ from the certified curl's own expressions by :func:`ghost_observability`
#: rather than tabulated, so the pairing is a property of the emitted text.
CURL_EXPRESSIONS = ("curl0", "curl1", "curl2")


def ghost_observability() -> Dict[str, Any]:
    """Does any BOUNDARY VALUE the FOLDED ghost serves reach ``step_D``'s output?

    THE ANSWER IS NO, AND IT IS DERIVED FROM THE FOLDED KERNEL'S OWN TEXT. Each
    shifted tap feeds one curl component (read off the three ``curlN = dtdx * (...)``
    expressions) and its ghost fires on the plane of the axis it shifts along (read
    off the offset lines); the folded cell-0 mask's own
    ``if BC? != PERIODIC: curlN = tl.where(at_?, 0.0, curlN)`` block says which
    (component, plane) pairs are zeroed -- and the fold widened that block from
    ``== METALLIC`` to ``!= PERIODIC``, so it now covers the mirror codes too. On the
    D side the two sets coincide exactly, which is why an edit to what the folded
    ghost SERVES moves no byte, and why the mask drops are the armed mutations and
    the ghost edit is the recorded null they earn.

    THE TOP-PLANE BLOCK IS PARSED SEPARATELY and is a different claim: it zeroes the
    LAST plane of a MIRROR_PERIODIC axis for each D target's own axis, which is the
    slot the driver's far fill writes and the curl must not.
    """
    tail = family.raw_folded_curl_tail()
    taps = plain.halo_taps(tail)
    offsets = plain.offset_coordinates(tail)
    shifted = {"si": "x", "sj": "y", "sk": "z"}
    axis_of_offset = {}
    for name, coordinates in offsets.items():
        moved = [shifted[value] for value in coordinates if value in shifted]
        if len(moved) != 1:
            raise AssertionError(
                f"the certified folded curl's offset {name} shifts {moved}, not one "
                f"axis")
        axis_of_offset[name] = moved[0]
    # Which curl each tap feeds, off the three curl expressions themselves.
    feeds: Dict[str, str] = {}
    for line in tail.splitlines():
        stripped = line.strip()
        for curl in CURL_EXPRESSIONS:
            if stripped.startswith(f"{curl} = dtdx * "):
                for register in taps:
                    if re.search(rf"\b{re.escape(register)}\b", stripped):
                        feeds[register] = curl
    if set(feeds) != set(taps):
        raise AssertionError(
            f"the certified folded curl's expressions consume {sorted(feeds)}, not "
            f"the six shifted taps {sorted(taps)}")
    # The CELL-0 mask's own (code, component, plane) triples, BACKWARD arm.
    body = tail.split("at_x, at_y, at_z = i == 0", 1)[1].split("\nelse:", 1)[0]
    masked = set(re.findall(
        r"if BC([XYZ]) != PERIODIC:\n\s*(curl\d) = tl\.where\(at_([xyz])", body))
    # The TOP-PLANE mask's own triples, BACKWARD arm -- a separate block.
    top_body = tail.split("last_x, last_y, last_z", 1)[1].split("\nelse:", 1)[0]
    top_masked = set(re.findall(
        r"if BC([XYZ]) == MIRROR_PERIODIC:\n\s*(curl\d) = tl\.where\(last_([xyz])",
        top_body))
    rows: Dict[str, Any] = {}
    for register, (_component, offset, _mask) in taps.items():
        axis = axis_of_offset[offset]
        curl = feeds[register]
        rows[register] = {
            "feeds": curl, "ghost_fires_on_plane": axis,
            "masked_there": any(code.lower() == axis and component == curl
                                and plane == axis
                                for code, component, plane in masked)}
    return {"taps": rows,
            "cell_zero_mask": sorted(masked),
            "top_plane_mask": sorted(top_masked),
            "every_tap_is_masked_where_its_ghost_fires": all(
                row["masked_there"] for row in rows.values())}
def leg_ghost_observability() -> Dict[str, Any]:
    """The coincidence, asserted; and what it costs the mutation set, stated."""
    findings: List[str] = []
    result = ghost_observability()
    if not result["every_tap_is_masked_where_its_ghost_fires"]:
        findings.append(
            "at least one shifted tap feeds a curl component the folded cell-0 mask "
            "does NOT zero at the plane its ghost fires on, so a boundary value IS "
            "observable in step_D's output -- and the ghost mutation is then an ARMED "
            "mutation that must be caught rather than a structural null")
    if len(result["cell_zero_mask"]) != 6:
        findings.append(f"the BACKWARD cell-0 mask has "
                        f"{len(result['cell_zero_mask'])} clauses, not the six the D "
                        f"targets' two zero shifts each give")
    if len(result["top_plane_mask"]) != 3:
        findings.append(f"the BACKWARD top-plane mask has "
                        f"{len(result['top_plane_mask'])} clauses, not the three the D "
                        f"targets' single shift-1 axis each gives")
    # THE TOP-PLANE MASK IS THE D FAMILY'S OWN AXIS, one clause per target.
    expected_top = {("X", "curl0", "x"), ("Y", "curl1", "y"), ("Z", "curl2", "z")}
    if set(result["top_plane_mask"]) != expected_top:
        findings.append(
            f"the top-plane mask's (axis, target, plane) triples are "
            f"{sorted(result['top_plane_mask'])}, not the D family's own-axis shift-1 "
            f"set {sorted(expected_top)}")
    return {"passed": not findings, "findings": findings, **result,
            "what_this_licenses": (
                "that no edit to what the folded GHOST serves can move a byte of "
                "step_D's output, which is why this gate's ghost mutation is a "
                "recorded null and the two MASK drops are the armed mutations that "
                "earn it. It licenses nothing about the periodic wrap, which is "
                "observable on a non-folded periodic axis and is armed separately, "
                "and nothing about the top-plane mask, whose own effect this gate "
                "measures in fu_D at whole step")}
# ---------------------------------------------------------------------------
# The mutations
# ---------------------------------------------------------------------------

#: ``tag -> {target, old, new, expected, case, why}``. ``target`` is ``kernel`` for a
#: source edit compiled through the family's ``kernel=`` door and ``host`` for a
#: launcher wrapper. ``case`` is the fixture the mutation is SCORED ON, and it is not
#: decoration: a mutation scored where it cannot fire is a disarmed leg wearing a pass.
MUTATIONS: Dict[str, Dict[str, Any]] = {
    # ---- the constitutive half, whose text lives in the PLAIN module -------------
    "m_foreign_tap_reads_the_stale_value": {
        "target": "kernel", "module": "plain", "expected": "CAUGHT",
        "case": MUTATION_CASE,
        "old": ("        value = a0\n"
                "        if COMP == 1:\n"
                "            value = a1\n"
                "        if COMP == 2:\n"
                "            value = a2\n"),
        "new": ("        _sidx = i * (ny * nz) + j * nz + k\n"
                "        value = tl.load(hi0 + _sidx, mask=valid, other=0.0)\n"
                "        if COMP == 1:\n"
                "            value = tl.load(hi1 + _sidx, mask=valid, other=0.0)\n"
                "        if COMP == 2:\n"
                "            value = tl.load(hi2 + _sidx, mask=valid, other=0.0)\n"),
        "why": "the foreign tap reads the PRE-LAUNCH H instead of recomputing it -- "
               "exactly the value an in-place weld would have read at a neighbour the "
               "launch had not yet written. This is the defect the whole design "
               "removes, so it MUST be caught; a null here would mean the recompute "
               "is decorative on this fixture and the purity ledger is the leg that "
               "would have said so first",
    },
    "m_foreign_tap_reads_B": {
        "target": "kernel", "module": "plain", "expected": "CAUGHT",
        "case": MUTATION_CASE,
        "old": ("        value = a0\n"
                "        if COMP == 1:\n"
                "            value = a1\n"
                "        if COMP == 2:\n"
                "            value = a2\n"
                "        return tl.where(valid, value, 0.0)\n"),
        "new": ("        value = s0\n"
                "        if COMP == 1:\n"
                "            value = s1\n"
                "        if COMP == 2:\n"
                "            value = s2\n"
                "        return tl.where(valid, value, 0.0)\n"),
        "why": "the tap returns the flux density the constitutive read rather than "
               "the stepped magnetic field -- the mis-selection a six-value return "
               "makes easy, and one the curl would consume without complaint",
    },
    "m_regroup_the_two_accumulations": {
        "target": "kernel", "module": "plain", "expected": "CAUGHT",
        "case": MUTATION_CASE,
        "old": ("        a0 = a0 + kp_0 * src0\n"
                "        a0 = a0 - km_0 * prev0\n"),
        "new": "        a0 = a0 + (kp_0 * src0 - km_0 * prev0)\n",
        "why": "the two accumulations are flattened into one. `((f + kps*src) - "
               "kms*prev)` and `f + (kps*src - kms*prev)` are different float32 "
               "numbers, and the certified body's docstring names the grouping as one "
               "of the three things that decide bit-identity",
    },
    # ---- the curl half and the weld, whose text lives in THIS family's module ----
    "m_f_w_H_written_in_place": {
        "target": "kernel", "module": "folded", "expected": "CAUGHT",
        "case": MUTATION_CASE,
        "old": "        tl.store(wo0 + idx, src0, mask=live)\n",
        "new": "        tl.store(wi0 + idx, src0, mask=live)\n",
        "why": "the split-field history is written into the PRE-LAUNCH buffer, so a "
               "foreign recompute of a neighbour reads B where it needs B_prev -- the "
               "second, easily missed half of the hazard -- and the rotation then "
               "publishes an unwritten scratch",
    },
    "m_flatten_curl_parens": {
        "target": "kernel", "module": "folded", "expected": "CAUGHT",
        "case": MUTATION_CASE,
        "old": "        curl0 = dtdx * ((c_y - c) + (b - b_z))\n",
        "new": "        curl0 = dtdx * (((c_y - c) + b) - b_z)\n",
        "why": "the curl's association is re-parsed; `(a + b) - c` and `a + (b - c)` "
               "round differently in float32, and this spelling is the one an earlier "
               "round measured actually mutating (its first attempt re-parsed to the "
               "original and reported a FALSE NULL)",
    },
    "m_drop_cell0_mask": {
        "target": "kernel", "module": "folded", "expected": "CAUGHT",
        "case": MUTATION_CASE, "attribution": {"must_move": "fu_D",
                                               "must_not_move": "D"},
        "old": ("            if BCY != PERIODIC:\n"
                "                curl0 = tl.where(at_y, 0.0, curl0)\n"
                "            if BCZ != PERIODIC:\n"
                "                curl0 = tl.where(at_z, 0.0, curl0)\n"),
        "new": ("            if BCY != PERIODIC:\n"
                "                curl0 = curl0\n"
                "            if BCZ != PERIODIC:\n"
                "                curl0 = curl0\n"),
        "why": "the cell-0 ownership mask (stepping._mask_non_owned_cells, widened by "
               "the fold from `== METALLIC` to `!= PERIODIC`) stops zeroing curl0 at "
               "the near plane of a folded or walled axis. THE CITATION THIS FAMILY "
               "INHERITED PREDICTED THIS NULL AT WHOLE STEP and it is not: on the "
               "Metal sibling it moved 540 words after complete driver steps, every "
               "one of them in fu_D and none in D, because the masks zero curlN, "
               "which feeds the stored auxiliary as well as the stored displacement, "
               "and the driver's fills and wall clear rewrite D at those planes and "
               "rewrite nothing of fu_D. Scored here, with the attribution asserted",
    },
    "m_drop_top_plane_mask": {
        "target": "kernel", "module": "folded", "expected": "CAUGHT",
        "case": MUTATION_CASE, "attribution": {"must_move": "fu_D",
                                               "must_not_move": "D"},
        "old": ("            if BCX == MIRROR_PERIODIC:\n"
                "                curl0 = tl.where(last_x, 0.0, curl0)\n"),
        "new": ("            if BCX == MIRROR_PERIODIC:\n"
                "                curl0 = curl0\n"),
        "why": "the fold's OWN mask stops zeroing the LAST plane of a MIRROR_PERIODIC "
               "axis for the D target whose Yee shift is 1 there -- the slot past "
               "MEEP's owned window that the driver's far fill writes and the curl "
               "must not. Same inherited null prediction, same refutation: 264 words "
               "on the Metal sibling, all in fu_D",
    },
    "m_top_plane_mask_on_a_metallic_fold": {
        "target": "kernel", "module": "folded", "expected": "CAUGHT",
        "case": MUTATION_CASE,
        "old": ("            if BCY == MIRROR_PERIODIC:\n"
                "                curl1 = tl.where(last_y, 0.0, curl1)\n"),
        "new": ("            if BCY != PERIODIC:\n"
                "                curl1 = tl.where(last_y, 0.0, curl1)\n"),
        "why": "the top-plane mask is emitted on a MIRROR_METALLIC axis as well. That "
               "axis stores no slot past MEEP's owned window, so its last plane IS "
               "stepped and this deletes a live cell -- the complement of dropping the "
               "mask, and the reason the MIRROR_METALLIC / MIRROR_PERIODIC split is "
               "derived from `_stored_past_owned` rather than from the declaration",
    },
    "m_periodic_wrap_reads_the_near_row": {
        "target": "kernel", "module": "folded", "expected": "CAUGHT",
        "case": PERIODIC_MUTATION_CASE,
        "old": "            sk = tl.where(sk < 0, nz - 1, tl.where(sk == nz, 0, sk))\n",
        "new": "            sk = tl.where(sk < 0, 0, tl.where(sk == nz, 0, sk))\n",
        "why": "the periodic wrap reads row 0 instead of the far row. SCORED ON THE "
               "FIXTURE WHOSE z AXIS IS PERIODIC and nowhere else: on every other code "
               "this branch does not compile, and the cell-0 mask -- which the fold "
               "widened to every non-periodic axis -- zeroes the very curl component "
               "the shifted tap feeds at the plane where the wrap fires",
    },
    "m_ghost_zero_replaced_by_the_mirror_source": {
        "target": "kernel", "module": "folded", "expected": "CAUGHT",
        "expected_override": "NULL", "case": MUTATION_CASE,
        "old": ("        b_x = _h_tap(1, si, j, k, vx, hi0, hi1, hi2, wi0, wi1, wi2, "
                "b0, b1, b2,\n"),
        "new": ("        b_x = _h_tap(1, tl.where(si < 0, 2, si), j, k, live, hi0, "
                "hi1, hi2, wi0, wi1, wi2, b0, b1, b2,\n"),
        "why": "past a folded face the certified kernel serves an EXACT 0.0; this "
               "serves the ARRAY PATH'S OWN mirror image instead -- the constitutive "
               "at stored cell MIRROR_SOURCE_INDEX = 2, which is `parity * H[2]` at "
               "parity +1. PREDICTED NULL AND THE NULL IS STRUCTURAL: the only "
               "consumer of that ghost is curl2 at i == 0, which the cell-0 mask "
               "zeroes on every non-periodic axis, the fold included. "
               "`m_ghost_mirror_source_with_the_cell0_mask_dropped` is the PAIRED "
               "CONTROL that earns the null: the same edit with the mask removed must "
               "fire, which is what shows the null is the mask's doing rather than a "
               "dead tap or a disarmed harness",
    },
    "m_ghost_mirror_source_with_the_cell0_mask_dropped": {
        "target": "kernel", "module": "folded", "expected": "CAUGHT",
        "case": MUTATION_CASE,
        "earns_the_null_of": "m_ghost_zero_replaced_by_the_mirror_source",
        "edits": (
            (("        b_x = _h_tap(1, si, j, k, vx, hi0, hi1, hi2, wi0, wi1, wi2, "
              "b0, b1, b2,\n"),
             ("        b_x = _h_tap(1, tl.where(si < 0, 2, si), j, k, live, hi0, "
              "hi1, hi2, wi0, wi1, wi2, b0, b1, b2,\n")),
            (("            if BCX != PERIODIC:\n"
              "                curl2 = tl.where(at_x, 0.0, curl2)\n"),
             ("            if BCX != PERIODIC:\n"
              "                curl2 = curl2\n")),
        ),
        "why": "THE PAIRED CONTROL. The same ghost edit with the cell-0 mask removed "
               "from curl2 on the x axis -- the exact plane the ghost feeds. It MUST "
               "fire; if it did not, the tap would be dead and the null above would "
               "say nothing about the mask",
    },
    "m_own_cell_takes_the_foreign_recompute": {
        "target": "kernel", "module": "folded", "expected": "CAUGHT",
        "expected_override": "NULL", "case": MUTATION_CASE,
        "old": "        a = own0\n",
        "new": ("        a = _h_tap(0, i, j, k, live, hi0, hi1, hi2, wi0, wi1, wi2, "
                "b0, b1, b2,\n"
                "                   kp0, kp1, kp2, kmx, kmy, kmz, ny, nz)\n"),
        "why": "PREDICTED NULL and recorded as one. `_h_tap` at the program's own cell "
               "with `valid=live` is `_h_cell` at that cell closed on the same guard, "
               "which is what `own0` already is -- so this must NOT diverge, and that "
               "is the measurement that shows the own-cell register and the recompute "
               "are ONE code path rather than two that happen to agree",
    },
    # ---- host mutations ---------------------------------------------------------
    "m_codes_from_boundary_kinds": {
        "target": "host", "expected": "CAUGHT", "case": "fold_y_periodic",
        "why": "the plan is built with the PLAIN product's triple -- "
               "`[1 if kind == 'metallic' else 0 for kind in _boundary_kinds(...)]` "
               "(fused_hd_pair.py's own expression) -- instead of "
               "`symmetry.folded_axis_kinds`'. On a folded axis `_boundary_kinds` "
               "reports 'mirror', which that expression maps to 0 = PERIODIC: the "
               "ghost wraps to the far plane, the cell-0 mask is never emitted and "
               "neither is the top-plane one. THE KERNEL CANNOT CATCH IT, because 0 is "
               "a valid code, so this is the family's single point of failure and it "
               "is a gate leg rather than a comment",
    },
    "m_mirror_codes_reduced_to_periodic": {
        "target": "host", "expected": "CAUGHT", "case": MUTATION_CASE,
        "why": "both mirror codes are mapped to PERIODIC while the metallic one is "
               "kept -- the narrower sibling of the mutation above, which isolates the "
               "fold from the wall: a reader who believed 'a folded axis is just a "
               "periodic one on a shorter array' would write exactly this",
    },
    "m_mirror_periodic_reported_as_mirror_metallic": {
        "target": "host", "expected": "CAUGHT", "case": MUTATION_CASE,
        "why": "the two folded terminations are swapped. It is the one classification "
               "error `folded_axis_kinds` cross-checks against `grid.is_metallic` for, "
               "and it costs exactly the top-plane mask: a MIRROR_PERIODIC axis stops "
               "protecting the slot the driver's far fill writes",
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
               "integer one. It compiles, launches and converges; what it produces is "
               "a half-cell-wrong absorber profile, which is why this is a gate leg "
               "and not a comment",
    },
    "m_withdraw_after_step_D": {
        "target": "host", "expected": "CAUGHT", "case": MUTATION_CASE,
        "why": "the seam's electric withdraw is performed AFTER the launch rather than "
               "before it -- the array-path campaign's own `after_step_D` null "
               "control. Driven on a fixture carrying an integrated electric source, "
               "in the withdraw leg",
    },
}

#: The one edit required NOT to diverge, and it is the leg that makes the register
#: read a measurement rather than a tautology.
BYTE_NEUTRAL = {
    "tag": "m_reload_own_H_from_the_scratch",
    "module": "folded",
    "old": ("        a = own0\n"
            "        b = own1\n"
            "        c = own2\n"),
    "new": ("        a = tl.load(ho0 + idx, mask=live, other=0.0)\n"
            "        b = tl.load(ho1 + idx, mask=live, other=0.0)\n"
            "        c = tl.load(ho2 + idx, mask=live, other=0.0)\n"),
    "why": "the curl half re-loads its own cell's magnetic field from the scratch this "
           "program just wrote, instead of using the register. PREDICTED NULL: one "
           "program's store and load of one address are ordered, so this is inert -- "
           "and confirming it is what shows the correctness comes from the FOREIGN "
           "taps and not from the register",
}


#: Inserted into every mutant's kernel body so its Triton cache key differs from the
#: shipped one's AND from every other mutant's. A comment is arithmetically inert; a
#: mutant served a warm binary of the shipped kernel would report every defect as
#: uncaught, which is the failure this line exists to prevent.
MUTANT_MARKER_ANCHOR = "        idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)\n"

#: Where each mutable module lives, by the key a mutation's ``module`` names.
MUTABLE_MODULES = {
    "folded": "meep_gpu/triton_kernels/folded_fused_hd_pair.py",
    "plain": "meep_gpu/triton_kernels/fused_hd_pair.py",
}


def _exec_module(name: str, filename: str, text: str) -> Any:
    """Register ``text`` under a fake filename and execute it as a module.

    TRITON READS THE SOURCE BACK. ``JITFunction.__init__`` calls
    ``inspect.getsourcelines(fn)``, and a function created by ``exec`` has no file for
    ``inspect`` to find. Registering the text in ``linecache`` under this module's own
    fake filename is what makes the mutant compile, and it also keeps Triton's JIT
    cache key distinct: the key is over the SOURCE, so a warm cache cannot serve the
    shipped binary for a mutant.
    """
    import linecache  # noqa: PLC0415
    import types  # noqa: PLC0415

    linecache.cache[filename] = (len(text), None, text.splitlines(True), filename)
    module = types.ModuleType(name)
    module.__file__ = filename
    module.__package__ = "meep_gpu.triton_kernels"
    sys.modules[name] = module
    exec(compile(text, filename, "exec"), module.__dict__)  # noqa: S102
    return module


def mutated_family(tag: str, edits: Sequence[Tuple[str, str]],
                   where: str = "folded") -> Any:
    """Import a COPY of this family with one or more source edits applied.

    TWO FILES, BECAUSE THIS PRODUCT'S TEXT LIVES IN TWO. The curl half and the weld
    are in ``folded_fused_hd_pair.py``; the constitutive half is
    ``fused_hd_pair._h_cell`` / ``_h_tap``, IMPORTED rather than copied. A mutation of
    the constitutive half therefore has to build a mutant of the PLAIN module first
    and then a copy of this family whose import line points at it -- otherwise the
    mutant would be built and the shipped device functions would still be called, and
    every constitutive defect would report as uncaught.

    The mutation reaches the launch through the shipped planner's ``kernel=`` door,
    which is the door the product itself documents.
    """
    if where not in MUTABLE_MODULES:
        raise AssertionError(f"{tag}: unknown mutation module {where!r}")
    folded_path = API_ROOT / MUTABLE_MODULES["folded"]
    folded_text = folded_path.read_text(encoding="utf-8")
    folded_name = f"meep_gpu.triton_kernels.folded_fused_hd_pair__mut_{tag}"
    folded_file = str(folded_path) + f"#{tag}"

    def apply(text: str, label: str) -> str:
        for old, new in edits:
            hits = text.count(old)
            if hits != 1:
                raise AssertionError(
                    f"{tag}: the mutation needle {old.strip()[:60]!r} matches {hits} "
                    f"times in {label}, not once; a mutation that matched nothing is "
                    f"a disarmed leg and one that matched twice is two defects")
            text = text.replace(old, new, 1)
        return text

    if where == "plain":
        plain_path = API_ROOT / MUTABLE_MODULES["plain"]
        plain_text = apply(plain_path.read_text(encoding="utf-8"), "the plain module")
        plain_name = f"meep_gpu.triton_kernels.fused_hd_pair__mut_{tag}"
        _exec_module(plain_name, str(plain_path) + f"#{tag}", plain_text)
        folded_text = folded_text.replace(
            "from .fused_hd_pair import _h_cell, _h_tap",
            f"from {plain_name} import _h_cell, _h_tap", 1)
        if plain_name not in folded_text:
            raise AssertionError(
                f"{tag}: the family's import of the certified device functions could "
                f"not be redirected at the mutant; the shipped ones would be launched")
    else:
        folded_text = apply(folded_text, "the family module")
    # THE MARKER, ALWAYS: a distinct kernel source per mutant, even when only the
    # imported plain module changed.
    folded_text = folded_text.replace(
        MUTANT_MARKER_ANCHOR, MUTANT_MARKER_ANCHOR + f"        # mutant: {tag}\n", 1)
    return _exec_module(folded_name, folded_file, folded_text)


def _rebound_codes(mapping: Callable[[Sequence[int]], Sequence[int]],
                   ) -> Callable[[Any], Any]:
    """Rebind a built plan's boundary codes -- the fold's host mutation seam.

    THE PLAN IS BUILT BY THE SHIPPED BUILDER FIRST, so what this replaces is exactly
    the triple ``folded_axis_kinds`` resolved, and nothing else about the plan.
    """

    def wrap(plan: Any) -> Any:
        rebound = tuple(int(code) for code in mapping(plan.bc))
        if rebound == tuple(plan.bc):
            # A NO-OP REBIND IS A DISARMED LEG WEARING A PASS. The mapping and the
            # fixture have to disagree, or the mutation launches the shipped codes and
            # reports the defect as uncaught.
            raise AssertionError(
                f"the code rebind is a no-op on this fixture: {rebound} is what "
                f"folded_axis_kinds already resolved")
        plan.bc = rebound
        return plan

    return wrap


def _boundary_kind_codes(driver: Any) -> Callable[[Any], Any]:
    """The PLAIN product's own 0/1 mapping, verbatim, as a host mutation."""
    kinds = tcoverage._boundary_kinds(driver.grid, driver.pml)  # noqa: SLF001
    codes = tuple(1 if kind == "metallic" else 0 for kind in kinds)
    return _rebound_codes(lambda _current: codes)
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


# ---------------------------------------------------------------------------
# DEVICE LEGS
# ---------------------------------------------------------------------------

def leg_product(steps: int, seed: int,
                progress: Optional[Callable[[str], None]] = None) -> List[Dict[str, Any]]:
    """The four-reference identity on the eight boundary triples."""
    rows: List[Dict[str, Any]] = []
    for name, keywords in CASES:
        driver = build_driver(dict(keywords), seed=seed)
        try:
            result = run_product(driver, steps, progress=(
                (lambda message, case=name: progress(f"{case} {message}"))
                if progress else None))
        finally:
            driver.close()
        result.update({"leg": "product", "case": name,
                       "boundaries": dict(keywords.get("boundaries") or {})})
        rows.append(result)
    return rows


def leg_seed_scale(steps: int, seed: int) -> Dict[str, Any]:
    """The 2^80 seed scale is a change of exponent and nothing else.

    DRIVEN, not asserted. Every case runs at three scales and the verdict must be the
    same at each; the record carries the denormal census at each so a reader can see
    why the shipped scale is the one it is.
    """
    findings: List[str] = []
    rows: Dict[str, Any] = {}
    for bits in (0, 40, SEED_SCALE_BITS):
        driver = build_driver(dict(CASES[-1][1]), seed=seed, scale_bits=bits)
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


def leg_launch_structure(steps: int, seed: int) -> Dict[str, Any]:
    """Launches per step at the seam AND over the whole step, two counters.

    THE HONEST NUMBER IS REPORTED WHETHER OR NOT IT FLATTERS THE WELD: against the
    composition the composer installs today the whole-step launch count does not fall.
    That is the arbitration ruling as a measurement.
    """
    findings: List[str] = []
    driver = build_driver(dict(CASES[MUTATION_INDEX][1]), seed=seed)
    try:
        arrangements = all_arrangements(driver, count=True)
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
    finally:
        driver.close()
    steps_run = result["steps_compared"]
    # THE FOLD MAKES ONE plan.run() MORE THAN ONE LAUNCH, and that is measured rather
    # than tolerated. `symmetry.MirrorGhostFillPlan.run` writes every folded axis's
    # ghost planes as one launch PER AXIS (symmetry.py:1089-1109), so on a grid with
    # two folded axes the two fill slots contribute two kernel launches each against
    # one plan run each. The two witnesses are still independent -- what changes is
    # the arithmetic that relates them, and it is DERIVED from each plan's own axis
    # table rather than from a constant.
    multi = multi_launch_plans(arrangements)
    for name, counts in launches.items():
        if name == "array":
            continue
        excess = sum(entry["launches_per_run"] - 1 for entry in multi.get(name, ()))
        if counts["uncounted_plans"]:
            findings.append(
                f"{name}: the kernel counter cannot see "
                f"{counts['uncounted_plans']}, so the two witnesses are not "
                f"comparable on this fixture -- this gate's own fixture must put "
                f"only plans it can resolve on the slots it does not own")
        elif counts["kernels"] != (counts["plans"] + excess * steps_run):
            findings.append(
                f"{name}: the plans counted {counts['plans']} launches and the "
                f"independent kernel counter {counts['kernels']}; the multi-launch "
                f"plans on this fixture account for {excess} extra launches per step "
                f"over {steps_run} steps ({[e['plan'] for e in multi.get(name, ())]})")
    if seam_launches.get("weld") != 1:
        findings.append(f"the weld occupies {seam_launches.get('weld')} launches at "
                        f"the seam, not one")
    if seam_launches.get("singles") != 2:
        findings.append(f"the certified singles occupy {seam_launches.get('singles')} "
                        f"launches at the seam, not two")
    # THE WELD'S OWN WHOLE-STEP COUNT, asserted rather than left to the table: one
    # launch per distinct plan per step, and the seam contributing exactly one of them.
    per_step_launches = _weld_step_launches(arrangements)
    expected = steps_run * per_step_launches
    if launches["weld"]["kernels"] != expected:
        findings.append(
            f"the weld arrangement launched {launches['weld']['kernels']} kernels over "
            f"{steps_run} steps; its plans launch {per_step_launches} kernels per step "
            f"({[e['plan'] + ':' + str(e['launches_per_run']) for e in multi.get('weld', ())]} "
            f"launch more than once), which is {expected}")
    return {
        "passed": not findings, "findings": findings,
        "steps": steps_run,
        "launches_over_the_run": launches,
        "launches_per_step": {name: {key: round(value / steps_run, 4)
                                     for key, value in counts.items()
                                     if isinstance(value, int)}
                              for name, counts in launches.items()} if steps_run else {},
        "seam_launches_per_step": seam_launches,
        "plans_whose_one_run_is_more_than_one_launch": multi,
        "what_the_multi_launch_table_is": (
            "symmetry.MirrorGhostFillPlan writes every folded axis's ghost planes as "
            "one launch PER AXIS inside a single run(), so on a folded grid the plan "
            "witness and the kernel witness are related by that axis count rather "
            "than by equality. The count is read off each plan's own axes table; a "
            "fixture with two folded axes contributes one extra launch per fill slot "
            "per step, and this gate asserts the relation rather than the equality"),
        "what_the_weld_saves_at_the_seam": (
            seam_launches.get("singles", 0) - seam_launches.get("weld", 0)),
        "what_the_weld_saves_over_the_whole_step_against_the_composition_today": (
            (launches["composition_today"]["kernels"] - launches["weld"]["kernels"])
            / steps_run if steps_run else 0),
        "the_honest_reading": (
            "the weld saves ONE launch against the two certified singles at its own "
            "seam and saves NOTHING -- it costs -- against the composition the "
            "composer installs today, because that composition already fuses "
            "step_B+update_H and this span would take update_H away from it. That is "
            "the arbitration verdict INSTALLABLE_REASON records, measured"),
    }


def _launches_per_run(owner: Any) -> int:
    """How many kernel launches ONE ``plan.run()`` performs on this fixture.

    ONE, except for ``symmetry.MirrorGhostFillPlan``, whose ``run`` loops over its own
    ``axes`` table and launches once per FOLDED AXIS (symmetry.py:1089-1109). Read off
    that table rather than tabulated here, so a grid with two folded axes and a grid
    with three are both accounted for by the same expression.
    """
    axes = getattr(owner, "axes", None)
    if axes is None:
        return 1
    return max(1, len(tuple(axes)))


def _distinct_owners(shim: Any) -> List[Any]:
    """Every plan that LAUNCHES in this shim, once each, in slot order."""
    owners: List[Any] = []
    seen: List[int] = []
    for plan in (shim.plans.values() if shim else ()):
        if plan is ABSORBED:
            continue
        for entry in (plan if isinstance(plan, (list, tuple)) else [plan]):
            owner = declaring(entry)
            if id(owner) not in seen:
                seen.append(id(owner))
                owners.append(owner)
    return owners


def multi_launch_plans(arrangements: Mapping[str, Arrangement]) -> Dict[str, Any]:
    """Per arrangement, the plans whose one ``run()`` is more than one launch."""
    out: Dict[str, Any] = {}
    for name, arrangement in arrangements.items():
        rows = [{"plan": type(owner).__name__,
                 "launches_per_run": _launches_per_run(owner)}
                for owner in _distinct_owners(arrangement.shim)
                if _launches_per_run(owner) > 1]
        if rows:
            out[name] = rows
    return out


def _weld_step_launches(arrangements: Mapping[str, Arrangement]) -> int:
    """KERNEL launches per step in the weld arrangement, not plan runs."""
    return sum(_launches_per_run(owner)
               for owner in _distinct_owners(arrangements["weld"].shim))


MUTATION_INDEX = [name for name, _ in CASES].index(MUTATION_CASE)
PERIODIC_INDEX = [name for name, _ in CASES].index(PERIODIC_MUTATION_CASE)


def leg_sync(steps: int, seed: int) -> Dict[str, Any]:
    """The sync channel: a plan spanning ``step_D`` must DECLINE the half-step.

    A REQUIRED LEG WITH ITS OWN NULL CONTROL. ``synchronize_magnetic_fields`` backs up
    ``_SYNC_FIELDS`` + ``_SYNC_AUXILIARY`` and restores them; ``D``, ``fu_D`` and
    ``f_cond_D`` are in NEITHER, so a weld that answered the consult would advance the
    electric state inside a half-step nothing can undo. The declining arm must be
    identical to the array path; the armed arm must diverge in D.
    """
    findings: List[str] = []
    out: Dict[str, Any] = {}

    def hook(step: int, name: str, driver: Any) -> Any:
        if step not in SYNC_STEPS:
            return None
        # Any accessor that synchronizes. `flux_in_box` is the one the driver's own
        # docstring names.
        driver.synchronize_magnetic_fields()
        driver.restore_magnetic_fields()
        return {"step": step, "synchronized": True}

    for armed in (False, True):
        driver = build_driver(dict(CASES[MUTATION_INDEX][1]), seed=seed)
        try:
            arrangements = {
                "array": Arrangement("array", None),
                "weld": arrangement_weld(driver, sync_hazard=armed,
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
        findings.append("the DECLINING arm is not identical to the array path; the "
                        "containment rule is not keeping this weld out of the "
                        "magnetic half-step")
    if declining["sync_refusals"] == 0:
        findings.append("the declining arm never refused the sync consult; the leg is "
                        "disarmed -- the accessor is not reaching the channel")
    if armed_row["bit_identical"]:
        findings.append("the ARMED arm did NOT diverge; the hazard this refusal "
                        "exists to prevent is not observable on this fixture, so the "
                        "refusal's citation is missing")
    else:
        electric = {name for name in armed_row["differing_volumes_at_the_end"]
                    if name.startswith(("D", "fu_D", "f_cond_D"))}
        if not electric:
            findings.append(
                f"the armed arm diverged but not in D/fu_D/f_cond_D "
                f"({sorted(armed_row['differing_volumes_at_the_end'])}); the citation "
                f"for the composer refusal is that the ELECTRIC state moves inside a "
                f"half-step nothing restores")
        armed_row["electric_volumes_that_diverged"] = sorted(electric)
    return {"passed": not findings, "findings": findings, "arms": out,
            "sync_steps": list(SYNC_STEPS)}


def _volume_source(component: str = "Ez", integrated: bool = True) -> Dict[str, Any]:
    """One integrated electric source declaration, in the driver's OWN schema.

    CONTINUOUS AND STARTING AT ZERO, and both are measurements rather than taste. A
    gaussian's turn-on is five widths of dead time (`cutoff` 5 over `fwidth` 0.4 is
    t0 = 12.5, some 350 steps at this Courant number), so on the first dozen steps its
    ``_applied_dipole`` is zero and the withdraw this leg exists to move is a no-op --
    which is exactly what the leg's first run reported, with both null controls coming
    back identical. ``add_source`` refuses an unknown key rather than ignoring it,
    which caught this declaration's other two spellings.
    """
    return {"source_type": "continuous", "component": component, "frequency": 0.6,
            "start_time": 0.0, "amplitude": 1.0, "center": (0.0, 0.0, 0.0),
            "is_integrated": integrated}


def leg_withdraw(steps: int, seed: int) -> Dict[str, Any]:
    """The seam's one pass, on a device, with both of the campaign's null controls.

    The configuration this product's predicate REFUSES today -- a standing integrated
    electric withdraw -- driven anyway, with the hoist in place. Hoisted must be
    identical; un-hoisted and after_step_D must both diverge. This is the device
    counterpart of ``results/h_to_d_withdraw_order_2026-09-04``, whose licence is the
    array path and the BEFORE placement only.
    """
    findings: List[str] = []
    out: Dict[str, Any] = {}
    keywords = dict(CASES[MUTATION_INDEX][1])
    sources = (_volume_source(),)
    standing = None
    dipoles: Dict[str, List[float]] = {}
    # THE FIELD STARTS AT EXACTLY ZERO ON THIS LEG, and that is the repair the first
    # run earned. Everywhere else this gate seeds at 2^80 to keep the state clear of
    # the denormal band; here the ONLY thing that may move a byte is the source, and a
    # standing dipole of order 1 subtracted from a field of order 1e26 is lost in
    # float32 rounding -- both null controls came back identical for that reason, on a
    # fixture whose withdraw was doing real work. A zero seed makes the pass this leg
    # measures the only thing in the comparison.
    for mode in ("hoisted", "not_hoisted", "after_step_D"):
        driver = build_driver(keywords, seed=seed, sources=sources, amplitude=0.0)
        try:
            declared = tuple(getattr(driver, "_sources", ()))
            standing = len(withdraw_hoist.standing_withdraws(declared))
            verdict = family.folded_fused_hd_pair_coverage(
                driver.fields, driver.pml, declared)
            if mode == "hoisted":
                out["predicate_refuses_this_configuration"] = not verdict.covered
                out["predicate_reasons"] = list(verdict.reasons)
            launcher = None
            if mode == "after_step_D":
                def launcher(plan: Any, fields=None) -> Any:  # noqa: ARG001
                    return _WithdrawAfter(plan, driver.fields, declared)
            arrangements = {
                "array": Arrangement("array", None),
                "weld": arrangement_weld(driver, launcher=launcher,
                                         hoist=(mode == "hoisted")),
            }
            seen: List[float] = []

            def watch(step: int, name: str, run: Any,
                      seen: List[float] = seen) -> Any:
                seen.append(max(
                    (abs(complex(getattr(source, "_applied_dipole", 0j) or 0j))
                     for source in getattr(run, "_sources", ())), default=0.0))
                return None

            # THE GATE'S STATED BUDGET, not twelve: the hoisted arm's identity and the
            # two null controls' divergence are only as strong as the steps compared.
            result = drive(driver, arrangements, steps,
                           per_step_hook=watch, stop_on_divergence=False)
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
    # THE LEG'S OWN ARMING CHECK. A configuration whose withdraw structurally
    # qualifies can still have NOTHING to withdraw on the steps compared -- the offset
    # is zero until the first injection and stays zero while the source's own envelope
    # is off. Both are true of a gaussian on step 1, and both nulls came back
    # identical the first time this leg ran.
    if not any(value > 0.0 for values in dipoles.values() for value in values):
        findings.append(
            "the standing dipole is ZERO on every step compared, so the seam's "
            "withdraw does no work and neither null control could diverge; this leg "
            "measured nothing")
    if not out["hoisted"]["reference_moved_words_step_1"]:
        findings.append("the array path moved no word on step 1: on a zero-seeded "
                        "fixture the source is the only driver, so this leg is not "
                        "driving one")
    if not out.get("predicate_refuses_this_configuration"):
        findings.append("the predicate ADMITS a row with a standing integrated "
                        "electric withdraw while HOISTS_THE_WITHDRAW is False")
    if not out["hoisted"]["bit_identical"]:
        findings.append("the HOISTED arrangement is not identical to the array path")
    for null in ("not_hoisted", "after_step_D"):
        if out[null]["bit_identical"]:
            findings.append(f"the {null} null control did NOT diverge")
    out["what_this_licenses"] = (
        "a DEVICE measurement of the BEFORE placement on this backend, on this "
        "fixture, for the ELECTRIC withdraw. It does not license INSTALLABLE or "
        "HOISTS_THE_WITHDRAW: both would additionally need the composer's hoist "
        "branch to be reachable for this product, which it is not")
    return {"passed": not findings, "findings": findings, **out}


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


def _mutation_case_driver(tag: str, seed: int) -> Any:
    case = MUTATIONS[tag]["case"] if tag in MUTATIONS else MUTATION_CASE
    keywords = dict(dict(CASES)[case])
    return build_driver(keywords, seed=seed), case


def leg_byte_neutral(steps: int, seed: int) -> Dict[str, Any]:
    """The one armed edit required NOT to diverge."""
    findings: List[str] = []
    driver, case = _mutation_case_driver(BYTE_NEUTRAL["tag"], seed)
    try:
        module = mutated_family(BYTE_NEUTRAL["tag"],
                                ((BYTE_NEUTRAL["old"], BYTE_NEUTRAL["new"]),),
                                BYTE_NEUTRAL.get("module", "folded"))
        kernel = module.fused_constitutive_curl_H_to_D_folded
        arrangements = {"array": Arrangement("array", None),
                        "weld": arrangement_weld(driver, kernel=kernel)}
        result = drive(driver, arrangements, steps, stop_on_divergence=False)
    finally:
        driver.close()
    if not result["bit_identical"]:
        findings.append(
            f"the byte-neutral control DIVERGED at step "
            f"{result['first_divergence_step']}; the register read is then not the "
            f"identity of the store it replaces and this gate's mutation legs are "
            f"measuring something other than what they claim")
    return {"passed": not findings, "findings": findings, "case": case,
            "tag": BYTE_NEUTRAL["tag"], "why": BYTE_NEUTRAL["why"],
            "bit_identical": result["bit_identical"],
            "steps_compared": result["steps_compared"]}
def leg_disarm(steps: int, seed: int) -> Dict[str, Any]:
    """The identical harness with the SHIPPED bytes: it must not diverge.

    A mutation leg that reported CAUGHT for every edit would be indistinguishable from
    a harness that diverges on its own.
    """
    findings: List[str] = []
    driver, case = _mutation_case_driver("m_flatten_curl_parens", seed)
    try:
        arrangements = {"array": Arrangement("array", None),
                        "weld": arrangement_weld(driver, kernel=None)}
        result = drive(driver, arrangements, steps, stop_on_divergence=False)
    finally:
        driver.close()
    if not result["bit_identical"]:
        findings.append(
            f"the DISARMED harness diverged at step {result['first_divergence_step']}; "
            f"every CAUGHT verdict in the mutation leg is then unattributable")
    return {"passed": not findings, "findings": findings, "case": case,
            "bit_identical": result["bit_identical"],
            "steps_compared": result["steps_compared"]}
def leg_mutation(steps: int, seed: int) -> List[Dict[str, Any]]:
    """Every armed defect, each with its declared outcome, its reason and -- where the
    defect is a mask -- the ATTRIBUTION of the words it moved.

    THE ATTRIBUTION IS THE FOLD'S OWN CLAUSE. ``symmetry.py``'s docstring says the two
    ownership masks are invisible after a complete driver step; the Metal sibling
    measured that FALSE for the auxiliary and true for the displacement. Each mask
    mutation therefore declares which volume family it must move and which it must
    not, and this leg fails a mask whose divergence lands anywhere else -- a mask that
    moved D would mean the fills stopped covering the planes it protects, and a mask
    that moved neither would mean this comparison is blind to it.
    """
    rows: List[Dict[str, Any]] = []
    for tag, spec in MUTATIONS.items():
        expected = spec.get("expected_override", spec["expected"])
        case = spec["case"]
        driver = build_driver(dict(dict(CASES)[case]), seed=seed)
        row: Dict[str, Any] = {"leg": "mutation", "mutation": tag, "case": case,
                               "target": spec["target"], "expected": expected,
                               "why": spec["why"]}
        if spec.get("module"):
            row["module"] = spec["module"]
        if spec.get("earns_the_null_of"):
            row["earns_the_null_of"] = spec["earns_the_null_of"]
        try:
            kernel = None
            launcher = None
            if spec["target"] == "kernel":
                edits = spec.get("edits") or ((spec["old"], spec["new"]),)
                module = mutated_family(tag, edits, spec.get("module", "folded"))
                kernel = module.fused_constitutive_curl_H_to_D_folded
            elif tag == "m_rotation_skipped":
                launcher = RotationSkipped
            elif tag == "m_constitutive_takes_the_half_integer_lattice":
                launcher = half_integer_rebind(driver.pml)
            elif tag == "m_codes_from_boundary_kinds":
                launcher = _boundary_kind_codes(driver)
            elif tag == "m_mirror_codes_reduced_to_periodic":
                launcher = _rebound_codes(lambda codes: tuple(
                    tsymmetry.CODE_PERIODIC
                    if code in (tsymmetry.CODE_MIRROR_METALLIC,
                                tsymmetry.CODE_MIRROR_PERIODIC) else code
                    for code in codes))
            elif tag == "m_mirror_periodic_reported_as_mirror_metallic":
                launcher = _rebound_codes(lambda codes: tuple(
                    {tsymmetry.CODE_MIRROR_PERIODIC: tsymmetry.CODE_MIRROR_METALLIC,
                     tsymmetry.CODE_MIRROR_METALLIC: tsymmetry.CODE_MIRROR_PERIODIC}
                    .get(code, code) for code in codes))
            elif tag == "m_withdraw_after_step_D":
                row.update({"outcome": "MEASURED IN THE withdraw LEG",
                            "passed": True,
                            "note": "driven there, on a fixture carrying an "
                                    "integrated electric source, with the hoisted "
                                    "arrangement as its positive control"})
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
                        "passed": outcome == expected})
            # WHICH VOLUMES MOVED, always recorded; asserted where the mutation
            # declares an attribution.
            volumes = {}
            for entry in result["per_step"]:
                for name, count in (entry.get("weld", {})
                                    .get("differing_volumes", {}) or {}).items():
                    volumes[name] = volumes.get(name, 0) + int(count)
            row["differing_volumes"] = volumes
            row["differing_words"] = int(sum(volumes.values()))
            attribution = spec.get("attribution")
            if attribution:
                moved = {name for name, count in volumes.items() if count}
                must_move = {name for name in moved
                             if name.startswith(attribution["must_move"])}
                must_not = {name for name in moved
                            if name.startswith(attribution["must_not_move"])
                            and not name.startswith(attribution["must_move"])}
                other = moved - must_move - must_not
                row["attribution"] = {
                    "declared": attribution,
                    "moved_in_the_required_family": sorted(must_move),
                    "moved_in_the_forbidden_family": sorted(must_not),
                    "moved_elsewhere": sorted(other),
                    "words_in_the_required_family": int(sum(
                        volumes[name] for name in must_move)),
                    "words_in_the_forbidden_family": int(sum(
                        volumes[name] for name in must_not)),
                    "held": bool(must_move) and not must_not}
                if not row["attribution"]["held"]:
                    row["passed"] = False
                    row["attribution_finding"] = (
                        f"{tag}: the mask moved {sorted(moved)}; it must move "
                        f"{attribution['must_move']}* and must not move "
                        f"{attribution['must_not_move']}*")
        except Exception as error:  # noqa: BLE001 - a mutant that cannot build
            row.update({"outcome": "REFUSED TO BUILD", "passed": False,
                        "error": f"{type(error).__name__}: {error}"[:600]})
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

    DERIVED from the census's own ``plan_step.selected`` -- the arms the composer chose
    for each row -- never from a list here. The seam record supplies each row's module
    and its ``withdraw_in_seam`` flag, which names the rows the predicate must refuse.
    """
    census = results / CENSUS
    seam = results / SEAM_RECORD
    if not census.is_dir() or not seam.is_dir():
        raise SystemExit(f"the lift leg needs {census} and {seam}")
    # THE ARMS COME FROM THE SEAM RECORD, and that is not a convenience. The census's
    # own `plan_step.selected` is empty on every row: it was cut on a NumPy host,
    # where `coverage._grid_reasons` clause 1 refuses every arm, so a join on it would
    # silently return an EMPTY cell and this leg would report a full-length pass over
    # nothing. `h_to_d_seam.jsonl`'s `arms.<backend>` is the board's own per-slot
    # verdict -- the one-admitter rule applied to the census's `slots[...].admitted`
    # -- and it is exactly what `h_to_d_seam.price` files the instances under.
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
    path = os.environ.get("MEEP_GPU_HD_GATE_PROGRESS")
    label = os.environ.get("MEEP_GPU_HD_GATE_LABEL", "?")
    line = f"[{time.strftime('%H:%M:%S')}] {label} {message}"
    print(line, flush=True)
    if path:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()


def lift_child(leg: str, script: Optional[str], module_path: Optional[str],
               case: Optional[str], out_json: str, steps: int,
               max_cells: Optional[int], policy: Optional[str] = None) -> int:
    """One corpus row: capture the simulation, lift it TO THE DEVICE, drive it.

    THE HARVESTERS ARE THE CENSUS'S OWN -- ``sweep_corpus_lift_parity.capture_simulation``
    for an example script and ``survey_meep_tests``' namespace for a test case -- so
    the row this drives is the row the census priced. What this child does NOT reuse is
    ``measure_predicate_coverage``'s own child, and the reason is one line of it: it
    lifts with ``prefer_gpu=False``, and every Triton plan binds a device pointer.

    THE POLICY IS INSTALLED HERE, IN THIS PROCESS, BEFORE THE FIRST DEVICE COMPILE. The
    parent's install lives on ``CUDABackend`` and on CuPy's compiler front ends in the
    PARENT's memory; a child inherits the environment -- the cache directories -- and
    nothing else. The 2026-09-06 campaign ran every corpus row without this: native
    CuPy (``-ftz=true`` on every NVRTC compile) against native Triton (no ``.ftz`` on
    arithmetic), stamped with the parent's policy, keyed WITHOUT the policy fold (the
    same cache-key hashes in both policies' directories), and every row's certified
    singles read as disagreeing with the array path by exactly one flushed subnormal
    (``results/triton_hd_lift_policy_2026-09-06`` is the autopsy). The child's own
    ``policy_stamp()`` -- installed, attained, and its compile counters -- rides in the
    row record, and ``leg_lift`` refuses a driven row whose stamp is not the parent's
    policy. MEEP is imported by the capture below, so the host half of a flush policy
    is reachable here without a separate import.
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
            # A ROW THIS HOST'S MEEP CANNOT BUILD IS NAMED, NOT SCORED. The census was
            # cut on a build with libGDSII; this one is not, and `get_GDSII_prisms`
            # raises before any Simulation exists. That is a fact about the HOST, not
            # about the engine or this weld, and scoring it as a gap is the phantom-
            # half defect the census battery's own runtime preflight exists to
            # prevent. The captured record's own `outcome`/`error` ride along.
            record.update(measured=False, unliftable_on_this_host=True,
                          note="no mp.Simulation to lift on this host's MEEP build")
            _write_child(out_json, record)
            return 0
        if policy is not None:
            # strict=True, as the parent: a child that asked to flush and quietly did
            # not is a child whose bytes mean nothing. MEEP is imported by the capture
            # above, which is what makes the host half reachable.
            subnormal_policy.install_subnormal_policy(policy, cupy=cupy, strict=True)
        record["host_flushing_at_lift"] = bool(backends.subnormals_flushed())
        _child_progress("lifting to the device")
        driver = meep_gpu.lift_simulation(sim, prefer_gpu=True, gpu_id=0)
        record["lift_s"] = round(time.time() - started, 2)
        record["grid_shape"] = [int(v) for v in driver.shape]
        record["grid_cells"] = int(math.prod(int(v) for v in driver.shape))
        try:
            record["triton_folded_fused_hd_pair_gate"] = evaluate_row(driver, steps, max_cells)
            record["measured"] = True
        finally:
            with contextlib.suppress(Exception):
                driver.close()
        # STAMPED AFTER THE DRIVE, so the counters say what this child compiled.
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


def evaluate_row(driver: Any, steps: int, max_cells: Optional[int]) -> Dict[str, Any]:
    """The four-reference identity on ONE lifted row."""
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
    verdict = family.folded_fused_hd_pair_coverage(driver.fields, driver.pml, sources)
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
    # THE DETERMINISM CONTROL, and it is a MEASUREMENT rather than a list of names.
    # This whole comparison steps ONE driver through several arrangements in lockstep
    # from one captured seed; a row whose current is drawn fresh on every evaluation
    # -- a CustomSource over `np.random.randn()`, which three corpus rows in this cell
    # are -- gives each arrangement a DIFFERENT source, and byte identity across them
    # is undefined rather than false. Two array-path runs from the same seed answer
    # that without anybody having to know which rows those are: identical means the
    # row is reproducible under this harness, and differing means no cross-arrangement
    # byte claim exists to make. Measured 2026-09-06: examples:stochastic_emitter.py
    # disagrees with ITSELF at step 1 on Dz/Ez/f_w_Ez, and every pair of arrangements
    # disagrees by the same order.
    # THE CONTROL LEAVES THE DRIVER FOUR STEPS IN, and the row is put back. ``drive``
    # returns with the last arrangement's state standing; without this restore the
    # measurement below would start at engine step 4 while reporting step 1, which is
    # what made the 2026-09-06 record's step 20 the probe's step 24 on the same row.
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
    block["what_the_determinism_control_measures"] = (
        "TWO premises in one control, and both are REQUIRED. The second arm zeroes "
        "every leading-underscore volume before every step, so an identical result "
        "says (a) this row's array path repeats itself from one captured seed and (b) "
        "the private scratch carries nothing across a step -- which is what licenses "
        "leaving it out of the byte comparison. A differing result refuses the row by "
        "measurement rather than scoring it")
    block["determinism_control_steps"] = control["steps_compared"]
    if not control["bit_identical"]:
        block.update(
            driven=False,
            why_not_driven=(
                f"NOT DETERMINISTIC (or the private scratch is load-bearing): "
                f"two array-path runs from one captured seed, one of them wiping "
                f"every private volume before each step, "
                f"disagree at step {control['first_divergence_step']} of "
                f"{control['steps_compared']} "
                f"({control['per_step'][-1]['array_repeat']['differing_words']} "
                f"words). This harness steps one driver through several arrangements "
                f"in lockstep, so a row whose current is drawn fresh per evaluation "
                f"gives each arrangement a different source and byte identity across "
                f"them is UNDEFINED, not false. Refused by measurement rather than "
                f"scored"))
        return block
    block["driven"] = True
    block.update(run_product(driver, steps, progress=_child_progress,
                             require_full_budget=True, clean_floor=1))
    return block


def leg_lift(out_dir: Path, steps: int, max_cells: Optional[int], timeout: float,
             resume: bool, only: Optional[Sequence[str]],
             interpreter: str, policy: Optional[str] = None) -> Dict[str, Any]:
    """Every corpus row in the cell, re-lifted in its own interpreter and driven.

    ``policy`` is the parent's requested subnormal policy, handed to every child, which
    installs it before its first device compile and stamps the row with the result. A
    driven row whose stamp does not name that policy, or whose executors did not attain
    it, fails this leg by name -- a resumed run over rows measured before the child
    installed anything is caught the same way, because those rows carry no stamp.
    """
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
                        "MEEP_GPU_HD_GATE_PROGRESS": str(progress_log)})
    work = lift_dir / "workdir"
    work.mkdir(exist_ok=True)
    for entry in examples_dir.iterdir():
        if entry.suffix in (".py", ".ipynb"):
            continue
        link = work / entry.name
        if not link.exists():
            with contextlib.suppress(OSError):
                link.symlink_to(entry)
    # THE CENSUS'S OWN SHIM PATH, for the modules whose text asks for it. Six MEEP
    # test modules decorate with `parameterized`, which is not installed; a row from
    # one of them dies on the IMPORT and comes back unmeasured -- a row scored as
    # unmeasurable by the harness rather than by the engine.
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
        environment["MEEP_GPU_HD_GATE_LABEL"] = row["label"]
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
                record_path.write_text(json.dumps(died, default=str),
                                       encoding="utf-8")
        record = json.loads(record_path.read_text(encoding="utf-8"))
        record.update({key: value for key, value in row.items() if key != "label"})
        record["label"] = row["label"]
        measured.append(record)
        block = record.get("triton_folded_fused_hd_pair_gate") or {}
        log(f"lift {index}/{len(rows)} {row['label']}: "
            f"measured={record.get('measured')} "
            f"admits={block.get('predicate_admits')} driven={block.get('driven')} "
            f"passed={block.get('passed')} steps={block.get('steps_compared')} "
            f"words={block.get('words_compared')} ({time.time() - started:.1f} s)")
    with (lift_dir / "rows.jsonl").open("w", encoding="utf-8") as handle:
        for record in measured:
            handle.write(json.dumps(record, default=str) + "\n")

    blocks = {r["label"]: (r.get("triton_folded_fused_hd_pair_gate") or {}) for r in measured}
    # THE DENOMINATOR IS THE CELL'S, MINUS THE ROWS THIS HOST'S MEEP CANNOT BUILD --
    # each NAMED with the error that establishes it, never a count. The census was cut
    # on a build with libGDSII and this one is not, so a handful of corpus rows raise
    # before any Simulation exists; a leg that scored them as failures of the weld
    # would be publishing a phantom gap.
    unliftable = {record["label"]: str(record.get("error") or record.get("note") or "")
                  for record in measured if record.get("unliftable_on_this_host")}
    liftable = {label: block for label, block in blocks.items()
                if label not in unliftable}
    blocks = liftable
    admitted = sorted(label for label, b in blocks.items() if b.get("predicate_admits"))
    refused = sorted(label for label, b in blocks.items()
                     if b.get("predicate_admits") is False)
    refused_by_the_withdraw = sorted(
        label for label in refused
        if any("standing integrated" in reason
               for reason in blocks[label].get("predicate_reasons", ())))
    # The rows the seam record says carry a standing withdraw, RESTRICTED to the ones
    # this host could lift at all: a row nobody could build cannot demonstrate a
    # refusal, and requiring it to would make an environment fact fail the weld.
    expected_refused = sorted(set(facts["rows_with_a_standing_withdraw"])
                              & set(blocks))
    if only:
        expected_refused = sorted(set(expected_refused) & set(blocks))
    unmeasured = sorted(label for label, b in blocks.items()
                        if "predicate_admits" not in b)
    driven = sorted(label for label in admitted if blocks[label].get("driven"))
    passed_rows = sorted(label for label in driven if blocks[label].get("passed"))
    not_driven = {label: blocks[label].get("why_not_driven")
                  for label in admitted if not blocks[label].get("driven")}
    # REFUSED BY MEASUREMENT, not by name: the row's own array path does not repeat
    # itself, so no cross-arrangement byte claim is defined for it.
    non_deterministic = {label: blocks[label].get("why_not_driven")
                         for label in admitted
                         if blocks[label].get("array_path_repeats_itself") is False}
    not_driven = {label: reason for label, reason in not_driven.items()
                  if label not in non_deterministic}
    # A ROW FAILS ON ITS SEAM CLAIM, not on a neighbouring sub-step's arithmetic.
    diverged = sorted(label for label in driven
                      if not all((blocks[label].get("seam_claim") or {}).values()))
    diverged_against_the_array_path = sorted(
        label for label in driven if not blocks[label].get("bit_identical"))
    # A STANDING PROPERTY OF THE TWO CERTIFIED ARMS UNDER THIS POLICY, not of this
    # weld: rows where `update_H` + `step_D` dispatched as the two separately
    # certified singles already disagree with the CuPy array path. Reported here
    # because this leg is where it is visible, and owned by nothing in this round.
    # THE ARBITRATION, MEASURED OVER THE DRIVEN CORPUS ROWS rather than argued in a
    # docstring. Over the driver's step_B - update_H - step_D - update_E slot path
    # launches are `4 - (installed pairs)`; a two-slot H->D product takes one slot
    # from EACH neighbour, so installing it is:
    #   LOSS  where BOTH neighbouring pairs install (2 pairs -> 1);
    #   TIE   where exactly one does (3 launches / 1 seam either way);
    #   GAIN  where NEITHER does (4 launches / 0 seams -> 3 / 1).
    # Read off each row's own `composer_selected` table -- what the SHIPPED composer
    # built for that row -- never off a list of names.
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
        if blocks[label].get("the_certified_singles_agree_with_the_array_path")
        is False)
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
    # THE CHILD'S OWN POLICY STAMP, per driven row. The parent's stamp says what the
    # parent installed; each child compiled and launched in its own process, and the
    # 2026-09-06 record was cut with every child running native CuPy against native
    # Triton under the parent's name. A row counts only if ITS process installed the
    # policy this leg was asked for and every executor attained it.
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
    child_compiles = {
        "rows_with_a_stamp": len(stamps),
        "nvrtc_calls": int(sum(int(s.get("nvrtc_calls") or 0) for s in stamps.values())),
        "ftz_removed": int(sum(int(s.get("ftz_removed") or 0) for s in stamps.values())),
        "triton_llir_compiles": int(sum(int((s.get("triton") or {}).get(
            "llir_compiles") or 0) for s in stamps.values())),
        "triton_ptx_audited": int(sum(int((s.get("triton") or {}).get(
            "ptx_audited") or 0) for s in stamps.values())),
        "triton_ptx_with_ftz": int(sum(int((s.get("triton") or {}).get(
            "ptx_with_ftz") or 0) for s in stamps.values())),
        "triton_ptx_violations": int(sum(int((s.get("triton") or {}).get(
            "ptx_violations") or 0) for s in stamps.values())),
        "rows_whose_host_flushed_at_lift": sorted(
            label for label in driven
            if next((r for r in measured if r["label"] == label), {}).get(
                "host_flushing_at_lift")),
    }
    children_ran_under_the_policy = (policy is None or (
        bool(driven) and len(stamps) == len(driven) and not child_policy_mismatches))
    return {
        "passed": bool(measured and not unmeasured
                       and (bool(only)
                            or len(admitted) + len(refused) + len(unliftable)
                            == facts["rows_in_cell"])
                       and len(driven) + len(non_deterministic) == len(admitted)
                       and refused == refused_by_the_withdraw == expected_refused
                       and not not_driven and not diverged
                       # EVERY DRIVEN ROW PASSED, and every ADMITTED row was either
                       # driven or refused BY MEASUREMENT. A SET equality, not a
                       # count: it closes the case of a row that is neither, which a
                       # count cannot see. The clause it replaces was
                       # `passed_rows == driven == admitted`, which predates the
                       # determinism control and contradicts it -- a row refused for
                       # non-determinism is admitted and not driven BY DESIGN, so the
                       # two clauses together were unsatisfiable whenever that control
                       # fired and the leg could not express its own verdict (measured
                       # 2026-09-06: 43 driven, 43 passed, 0 diverged, and FAIL).
                       and passed_rows == driven
                       and set(driven) | set(non_deterministic) == set(admitted)
                       and majority_cleared
                       and children_ran_under_the_policy),
        "facts": facts,
        "rows_in_cell": facts["rows_in_cell"],
        "rows_unliftable_on_this_host": unliftable,
        "rows_this_host_could_lift": len(blocks),
        "rows_admitted_and_driven": len(driven),
        "the_denominator": (
            f"{len(blocks)} of the cell's {facts['rows_in_cell']} rows; the other "
            f"{len(unliftable)} are NAMED in rows_unliftable_on_this_host with the "
            f"error MEEP raised before any Simulation existed on this build"),
        "rows_measured": len(measured),
        "rows_unmeasured": unmeasured,
        "admitted": len(admitted), "admitted_rows": admitted,
        "refused": refused,
        "refused_by_the_standing_withdraw": refused_by_the_withdraw,
        "refused_expected_from_the_seam_record": expected_refused,
        "driven": len(driven), "not_driven": not_driven,
        "rows_refused_as_non_deterministic": non_deterministic,
        "what_the_non_deterministic_refusal_is": (
            "a row whose OWN array path does not repeat itself from one captured "
            "seed. Established per row by a two-run control, never by a list of "
            "names; on such a row every arrangement consumes a different current and "
            "byte identity across arrangements is undefined rather than false"),
        "rows_bit_identical": len(passed_rows),
        "rows_that_diverged": diverged,
        "rows_where_ANY_arrangement_disagrees_with_the_array_path":
            diverged_against_the_array_path,
        "rows_where_the_CERTIFIED_SINGLES_disagree_with_the_array_path":
            singles_disagree,
        "arbitration_over_the_driven_rows": {
            key: sorted(value) for key, value in arbitration.items()},
        "arbitration_counts": {key: len(value)
                               for key, value in arbitration.items()},
        "what_the_arbitration_counts_mean": (
            "installing a TWO-SLOT H->D product on each driven row, priced off that "
            "row's OWN composer slot table: LOSS where both neighbouring pairs "
            "install (2 pairs -> 1), TIE where exactly one does, GAIN where NEITHER "
            "does (4 launches / 0 seams -> 3 / 1). It prices the COMPOSITION and says "
            "nothing about the arithmetic, which the seam claim measures; and it "
            "prices a two-slot span, so it says nothing about the four-slot "
            "step_B -> update_H -> step_D -> update_E weld, which is the only span "
            "that is strictly additive and is not built"),
        "what_the_certified_singles_row_list_is": (
            "rows on which the two certified arms, dispatched at the seam's slots in a "
            "child that INSTALLED this policy, disagree with the CuPy array path. Any "
            "entry fails the leg. The 43 entries the 2026-09-06 record carried were "
            "the child running with no policy at all "
            "(results/triton_hd_lift_policy_2026-09-06)"),
        "child_subnormal_policy": {
            "requested": policy, "expected_stamp": expected_stamp,
            "every_driven_child_installed_and_attained_it": children_ran_under_the_policy,
            "mismatches": child_policy_mismatches,
            "compiles_summed_over_the_children": child_compiles,
            "stamps_per_row": stamps},
        "seam_claim_per_row": {label: blocks[label].get("seam_claim")
                               for label in driven},
        "rows_where_the_weld_disagrees_with_the_certified_singles": weld_disagrees,
        "rows_where_the_SEAM_ALONE_disagrees_with_the_array_path": seam_alone_disagrees,
        "rows_where_a_reference_disagrees_with_the_array_path": reference_disagrees,
        "first_banded_step_per_row": {label: blocks[label].get("first_banded_step")
                                      for label in driven},
        "what_the_budget_is_on_a_lifted_row": (
            "the FULL requested budget of complete driver steps, from the engine's "
            "zeros, with every arrangement required to equal the array path on every "
            "step -- the synthetic fixture's bar. A lifted row is driven by its own "
            "sources and its state enters the float32 denormal band on its own "
            "schedule; the step it does so is RECORDED per row (first_banded_step) and "
            "no longer shortens the comparison, because with the policy installed in "
            "the child the array path and the kernels agree through the band under "
            f"both policies (2026-09-06). The floor of {LIFT_CLEAN_STEP_FLOOR} is kept "
            "as a second guard on the step count"),
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
        clauses["product_ran_every_boundary_triple"] = (
            {row["case"] for row in by_leg.get("product", ())}
            == {name for name, _ in CASES})
        clauses["every_product_case_moved_the_reference"] = all(
            row.get("reference_moved_words_step_1", 0) > 0
            for row in by_leg.get("product", ()))
        clauses["every_product_case_ran_the_full_budget"] = all(
            row.get("steps_compared") == row.get("steps_requested")
            for row in by_leg.get("product", ()))
        clauses["every_product_case_carried_four_references"] = all(
            set(row.get("launches", {})) == set(MODES)
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
        clauses["every_predicted_null_is_recorded_with_its_reason"] = all(
            row.get("why") for row in by_leg.get("mutation", ())
            if row.get("expected") == "NULL")
        # EVERY PREDICTED NULL IS EARNED BY A PAIRED CONTROL THAT FIRES. A null
        # nothing discriminates is indistinguishable from a disarmed leg.
        nulls = [row for row in by_leg.get("mutation", ())
                 if row.get("expected") == "NULL"]
        pairs = {row.get("earns_the_null_of"): row
                 for row in by_leg.get("mutation", ())
                 if row.get("earns_the_null_of")}
        clauses["every_measured_null_is_earned_by_a_control_that_fires"] = all(
            row["mutation"] not in pairs
            or pairs[row["mutation"]].get("outcome") == "CAUGHT"
            for row in nulls)
        # THE TWO MASK DROPS, WHOSE ATTRIBUTION IS THE FOLD'S OWN FINDING.
        masks = [row for row in by_leg.get("mutation", ())
                 if row.get("attribution")]
        clauses["both_mask_drops_are_armed"] = len(masks) == 2
        clauses["every_mask_drops_attribution_held"] = bool(masks) and all(
            row["attribution"].get("held") for row in masks)
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
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--subnormal-policy", default=None,
                        choices=(None, "keep", "flush"))
    parser.add_argument("--import-meep-for-host-policy", action="store_true")
    parser.add_argument("--lift-steps", type=int, default=None)
    parser.add_argument("--lift-max-cells", type=int, default=None)
    parser.add_argument("--lift-timeout", type=float, default=2400.0)
    parser.add_argument("--lift-resume", action="store_true")
    parser.add_argument("--lift-only", default=None)
    parser.add_argument("--lift-interpreter", default=sys.executable)
    # The child front door.
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
        "source_sha256": source_hashes(),
        "installable": family.INSTALLABLE,
        "installable_reason": family.INSTALLABLE_REASON,
        "what_served_means_on_a_board": (
            "PREDICATE ADMISSION, by the boards' own definition. A released product "
            "credits the seam-instances its predicate admits while executing NOWHERE: "
            "this product is in no fastpath.RELEASED_FUSED_ARMS envelope, fastpath "
            "never plans it, and the composer is not even offered it"),
        "the_composition_installed_reference_awaits_wiring": (
            "THE PRODUCT IS UNREGISTERED WHILE THIS GATE RUNS. It has no "
            "launch.CERTIFIED_FUSED_PRODUCTS row and no fastpath label, so no "
            "composer installs it and reference 3 (composition_today) is the "
            "composition the shipped composer builds WITHOUT it -- the released "
            "fused pair B (folded) holding step_B and update_H. The weld is "
            "force-installed by this gate's own shim at its two slots. A "
            "reference in which plan_step ITSELF installs this product is owed "
            "and is not measured here; the arbitration leg drives the two brakes "
            "that would hold it out, on the real composer, against an injected "
            "row, which is the strongest statement available before wiring"),
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
            "cupy_cache_dir": os.environ.get("CUPY_CACHE_DIR")})
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
            log(f"{leg:20s} {entry.get('case', entry.get('mutation', '')):34s} "
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
            record("mutation", leg_mutation(args.steps, args.seed))
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
    # `release.released` and `passed` -- deliberately, so a `--no-device` or host-only
    # run cannot be read as a release. `device_status` is derived from whether any
    # DEVICE leg actually ran, never from the flag that asked for one.
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
    log(f"RELEASED={payload['release']['released']} "
        f"({payload['seconds']} s)")
    for name, value in sorted(payload["release"]["clauses"].items()):
        if not value:
            log(f"  clause FAILED: {name}")
    if not payload["release"]["released"]:
        return 1 if set(legs) == set(ALL_LEGS) else EXIT_INCOMPLETE
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
