"""Byte-identity gate for the FOUR COMPLEX hand-CUDA fused products, on BOTH seams.

WHAT IS UNDER TEST. Four modules, each shipping ONE kernel that performs three driver
passes in one launch -- two on the MAGNETIC seam::

    step_B (driver.py:3291) -> zero_metal_B (:3295) -> update_H (:3298)

* ``cuda_kernels/complex_fused_magnetic_pair.py`` -- complex64 storage on a
  Cartesian grid (16 seam-instances of demand);
* ``cuda_kernels/cylindrical_fused_magnetic_pair.py`` -- complex64 storage on a
  Dcyl grid at |m| >= 1 (16), and the one Metal cannot bind at any sharing;

and two on the ELECTRIC seam::

    step_D (driver.py:3302) -> zero_metal_D (:3310) -> update_E (:3313)

* ``cuda_kernels/complex_fused_electric_pair.py`` -- the Cartesian D cell (16);
* ``cuda_kernels/cylindrical_fused_electric_pair.py`` -- the Dcyl D cell (16).

WHAT THE TWO ELECTRIC PRODUCTS ARE WORTH IS THE BRACKET, NOT THE LAUNCH COUNT. 31 of
those 32 seam-instances carry an ELECTRIC deposit between the halves, so the fusion
board priced both cells on the driver fact alone -- 1 and 0 reachable of 16 rows --
while both were POINTWISE-BUILDABLE. A fitness verdict buys nothing on a cell no
product occupies, because there is no product to ask about the repair.

ONE FILE FOR ALL FOUR, and that is a decision rather than convenience: they differ in
their fixtures, in which certified halves they splice and in WHICH SEAM they span, and
in NOTHING about how a fused seam is measured. A second copy of the harness would be a
second place for the launch-counting, the movement floor or the driver-order read to
drift, and the verdicts would stop being commensurable. ``--family`` picks the product;
every leg below runs identically on any of them.

=============================================================================
WHAT THE SEAM PARAMETRISATION HAD TO GET RIGHT, AND WHY EACH IS SILENT WHEN WRONG
=============================================================================

* **The passes BEFORE the seam.** Empty on the magnetic seam and the whole B/H half on
  the electric one (:func:`before_passes`). A D-side subject that launched without
  running it would be compared against an oracle half a step ahead of it.
* **The injection instant.** ``when`` on the magnetic seam and ``when + 0.5*dt`` on the
  electric one (driver.py:3303). A deposit leg using one for both compares two
  different runs and can pass only by cancelling its own error.
* **The wall table.** The B side's is the DIAGONAL and the D side's the OFF-DIAGONAL
  complement, so the deposit fixture's on-wall component and every wall mutation's
  needle are per seam.
* **The sub-lattice rename.** ``kms_int_*`` on the B seam and ``kms_half_*`` on the D
  seam -- mirror images -- so the device mutation that plants the half-cell error is
  two legs, not one.
* **The material.** ``update_E`` binds three inverse-permittivity pointers that
  ``update_H`` has no counterpart for, and they are the one group in those signatures
  that is deliberately NOT ``__restrict__``. On a vacuum fixture all three are one
  allocation of ones, so every mutation on that binding is bit-identical: the two
  electric families build with :func:`anisotropic_epsilon` instead.

=============================================================================
WHY THE COMPARISON IS PER COMPLETE DRIVER STEP
=============================================================================

The claim is about the SEAM between the halves -- the flux density never leaving a
register -- so a per-sub-step comparison could not see it at all. Two engines are
built from ONE seed and stepped side by side: the reference runs ``stepping``'s own
eleven-pass sequence, the subject runs the fused launch plus the six passes it does
not replace, and the two are compared as raw uint32 WORDS over all twenty-four stored
volumes after EVERY step. The D/E half runs on the array path on both sides, so a
corrupted B reaches E on the very next step; the first divergent step is reported
rather than a final pass/fail.

=============================================================================
THE EIGHT THINGS THIS GATE REFUSES TO LET PASS SILENTLY
=============================================================================

1. **A step this gate invented.** :func:`leg_driver_order` reads ``FdtdDriver.step``'s
   own source and extracts the ordered list of ``stepping`` calls it makes. The
   oracle walks THAT list, and the product's ``REPLACES`` is asserted to be a
   contiguous run of it.

2. **A silent fallback.** Bytes alone cannot prove the fused kernel ran: an engine
   that never launched it is byte-identical to the oracle BY CONSTRUCTION, because
   the oracle is the array path the same engine would otherwise take. So every case
   asserts EXACT launch counts from TWO INDEPENDENT COUNTERS -- a proxy installed
   over the shipped compile memo (``compile_cache._compiled_kernels``, the seam every
   ``_get_kernel`` route passes through) and the launcher's own returned report --
   and requires them to agree, to equal the step budget, and to name ONLY the fused
   kernel. A run that launched ``step_B_pml_complex_bloch`` beside the fused kernel
   is a run where the fusion added a launch instead of removing two.

3. **A vacuous comparison.** Two floors per volume. WHAT MUST MOVE: on the uniform
   class every one of the twenty-four stored volumes must differ from its seeded
   value. WHAT MUST NOT MOVE: the read-only volumes -- the PML coefficient vectors --
   must be bit-unchanged on the device side. Every argument in these signatures is
   ``__restrict__``, and a kernel that wrote through a ``const`` binding is UB NVRTC
   does not diagnose.

4. **An unexercised policy.** ``operand_census`` counts the subnormals and signed
   zeros the operands REALLY hold. The band class must contain subnormals and the
   uniform class must contain none -- a precondition that fires everywhere
   discriminates nothing, and one that never fires is a detector never shown to work.

5. **A guessed arm.** The complex multiply's orientation is a MEASURED platform fact
   and both arms compile. The licence is read from a probe artifact through
   ``complex_fields.expansion_license`` -- never re-derived here -- and the run is
   refused if the artifact was cut under a different float32 subnormal policy than
   the one this process installs. ``wrong_arm`` is a mutation leg, and it MUST be
   caught: an arm that made no difference would mean the licence was decorative.

6. **A mutation the harness cannot see.** Every leg is a real defect armed through
   the shipped launcher's own doors (its ``kernel``, ``tables``, ``walls``,
   ``boundary_codes`` arguments), and every one MUST be caught. A leg that is not
   caught is a defect this comparison is blind to, and is reported as such rather
   than dropped.

7. **A predicate that admits what the launch cannot serve.** :func:`leg_refusal`
   builds the configurations each module refuses BY NAME -- a real-storage run, a
   folded grid, a magnetic source with no publishable index, the wrong coordinate
   system -- and requires the refusal, with the reason recorded. Rule 3 of this
   round: a predicate admitting a row the launch cannot serve is a silent wrong
   answer, the one failure mode no bit-comparison catches.

8. **A deposit computed against a pre-injection field.** All four modules declare
   ``CARRIES_DEPOSIT_REPAIR``. :func:`leg_deposit` runs a real source IN THIS FAMILY'S
   SEAM through ``deposit_repair``'s two plans and compares against the driver's own
   order, with a NULL CONTROL that removes the repair and MUST diverge. The source's
   ``field_type`` is the seam's own letter: a magnetic source is not in the electric
   seam at all, so a leg on the wrong letter would run a CLEAN seam and report the
   null control's failure to diverge as a defect in the repair.

Usage, on the validation host (from ``parity/meep_gpu``)::

    PYTHONPATH=../.. CUDA_VISIBLE_DEVICES=<one empty index> \\
    CUPY_CACHE_DIR=<fresh, policy-token-carrying> \\
    python -u gate_cuda_fused_complex_pairs.py --family complex \\
        --subnormal-policy keep --expansion-probe <probe>.json \\
        --out <fresh dir>/gate.json

One flushed line per case (the progress-reporting rule); the artifact is rewritten after every
case, so an interrupted run keeps everything up to the failure. Correctness only: no
throughput or timing claim is made or possible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
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
except ImportError:  # laptop: only the source legs can run
    cp = None

import probe_fused_kernel_bit_identity as probe  # noqa: E402
import gate_provenance  # noqa: E402

from meep_gpu import deposit_repair, stepping  # noqa: E402
from meep_gpu.triton_kernels import complex_fields  # noqa: E402

log = probe.log
to_host = probe.to_host
operand_census = probe.operand_census
subnormal_band_hosts = probe.subnormal_band_hosts

#: Per-case RNG seed base, from a sha256 of the case's OWN label rather than
#: ``hash()``: ``PYTHONHASHSEED`` salts the hash of a string, so a hash-seeded case
#: could not be replayed from the record that names it.
SEED = 20260830

#: Complete DRIVER STEPS every product leg runs. Sixty, the budget every hand-CUDA
#: record on this track is cut at -- "identical for N steps" is a claim about N.
STEPS = 60

#: Every stored volume a complete step can touch, and every one is compared. The D/E
#: half is here even though neither fused product touches it, because a pair that
#: corrupted B reaches E through ``step_D`` on the very next step.
STATE_NAMES: Tuple[str, ...] = tuple(
    [f"{stem}{axis}" for stem in ("B", "D", "E", "H") for axis in "xyz"]
    + [f"fu_{stem}{axis}" for stem in ("B", "D") for axis in "xyz"]
    + [f"f_w_{stem}{axis}" for stem in ("E", "H") for axis in "xyz"])

#: The stored volumes a NO-ABSORBER family owns. THE TWELVE SPLIT-FIELD AUXILIARIES
#: ARE ABSENT RATHER THAN UNUSED: without a layer ``Fields`` allocates no ``fu`` and no
#: ``f_w`` at all, and both certified halves REFUSE a run where one exists anyway. A
#: comparison that asked for them would read ``None`` on every array and compare
#: nothing, which is the vacuity this gate exists to refuse.
#:
#: H IS ABSENT FOR THE SAME REASON AND IT IS NOT AN OVERSIGHT: without PML storage
#: ``Fields`` allocates no separate H at all and ``get_H`` returns the B arrays
#: themselves (fields.py:1164-1187, measured ``get_H('Hx') is fields.Bx`` on all six
#: corpus rows), so ``Hx`` is ``None`` on every engine this family builds. Naming it
#: would compare ``None`` against ``None`` on three of twelve arrays -- three
#: comparisons that can never fail, in the set that decides the verdict.
#:
#: THE POLE BUFFERS ARE ADDED PER FIXTURE, not here: how many there are is a property
#: of the spec. :func:`state_names_for` is what joins the two.
NO_PML_STATE_NAMES: Tuple[str, ...] = tuple(
    f"{stem}{axis}" for stem in ("B", "D", "E") for axis in "xyz")


def state_names_for(family: Dict[str, Any], fields: Any = None) -> Tuple[str, ...]:
    """Every array this family's comparison covers, for this engine.

    THE POLE BUFFERS ARE PART OF THE STATE ON A DISPERSIVE FAMILY and are read off the
    engine rather than declared: ``update_E`` subtracts them and ``update_P`` advances
    them inside the same driver step, so a weld that corrupted one would reach E on the
    NEXT step through a comparison that never looked. They are named by SLOT
    (``P[0][Ex]``) rather than by identity, because ``PolarizationState.update``
    rotates the array objects between ``P``, ``P_prev`` and ``_scratch`` -- a launcher
    that rotated wrongly would hold the right numbers in the wrong slots, and only a
    slot-named comparison catches that.
    """
    names = tuple(family.get("state_names", STATE_NAMES))
    if not family.get("pole_state") or fields is None:
        return names
    poles = tuple(
        f"P{attribute}[{index}][{component}]"
        for index, state in enumerate(getattr(fields, "polarizations", ()) or ())
        for component in sorted(state.driven())
        for attribute in ("", "_prev"))
    return names + poles


def _slot_array(fields: Any, name: str) -> Any:
    """One state slot's array, resolving the pole slots this gate names itself."""
    if not name.startswith("P"):
        return getattr(fields, name)
    attribute, rest = name.split("[", 1)
    index, component = rest[:-1].split("][")
    state = fields.polarizations[int(index)]
    return getattr(state, "P" if attribute == "P" else "P_prev")[component]

#: The volumes that must be BIT-UNCHANGED by a run: the PML coefficient vectors on
#: both Yee sub-lattices.
IMMUTABLE_PML: Tuple[str, ...] = tuple(
    f"{name}_{axis}{suffix}"
    for axis in "xyz" for name in ("kms", "sinv", "kps") for suffix in ("", "_h"))

#: The volumes the SUBNORMAL BAND class is held to, PER SEAM. Declared before the
#: first run. The B row is the original; the D row is its mirror image and names the
#: three groups the ELECTRIC seam's launch writes unconditionally.
BAND_MUST_MOVE_BY_SEAM: Dict[str, Tuple[str, ...]] = {
    "B": tuple([f"B{axis}" for axis in "xyz"] + [f"fu_B{axis}" for axis in "xyz"]
               + [f"f_w_H{axis}" for axis in "xyz"]),
    "D": tuple([f"D{axis}" for axis in "xyz"] + [f"fu_D{axis}" for axis in "xyz"]
               + [f"f_w_E{axis}" for axis in "xyz"]),
}


def band_must_move(family: Dict[str, Any]) -> Tuple[str, ...]:
    """The band class's movement floor for one family.

    DECLARED PER FAMILY WHERE THE SEAM'S ANSWER DOES NOT HOLD. The two seam rows above
    name the split-field auxiliary and the constitutive history, which a NO-ABSORBER
    family does not have -- so requiring them there would fail every case for an array
    that cannot exist rather than for a defect.
    """
    return tuple(family.get("band_must_move", BAND_MUST_MOVE_BY_SEAM[family["seam"]]))

MOVEMENT_FLOOR_NOTE = (
    "The uniform class is held to the full movement floor: all twenty-four stored "
    "volumes must differ from their seeded values. The subnormal-band class is held "
    "to BAND_MUST_MOVE_BY_SEAM[seam] only, and the reason is arithmetic and was "
    "written down before the first run: under 'flush' a state seeded entirely in the "
    "band drives constitutive_apply to f[idx] = (f[idx] + 0) - 0, since kps*src and "
    "kms*prev both flush, so the constitutive target is the identity and need not "
    "move. The flux density, its split-field auxiliary and the constitutive history "
    "are written unconditionally by the curl half and by the constitutive store, so "
    "they must move under either policy. Arrays that did not move are RECORDED for "
    "both classes either way.")

VALUE_CLASSES: Tuple[str, ...] = ("uniform", "subnormal_band")

#: NVRTC options the shipped modules compile with, restated so the guard control can
#: drop them. NOT imported from the module: a control that read the module's own
#: tuple would compile the same thing twice if that tuple were ever emptied.
GUARD_OPTIONS: Tuple[str, ...] = ("--fmad=false",)


# ---------------------------------------------------------------------------
# WHAT A COMPLETE STEP IS -- read off the driver, never modelled here
# ---------------------------------------------------------------------------

DRIVER_ORDER: Tuple[str, ...] = (
    "step_B", "fill_symmetry_bc_B", "zero_metal_B", "fill_folded_far_ghosts_B",
    "update_H",
    "step_D", "fill_symmetry_bc_D", "zero_metal_D", "fill_folded_far_ghosts_D",
    "update_E", "update_P",
)

#: The THREE passes each fused kernel replaces, PER SEAM, asserted equal to that
#: module's own ``REPLACES`` by :func:`leg_driver_order`. The two mirror fills are NOT
#: here: every product refuses a folded grid, so neither fill does anything on any
#: admitted configuration, and running them on the array path beside the kernel is
#: correct precisely because they are no-ops there.
FUSED_PASSES_BY_SEAM: Dict[str, Tuple[str, ...]] = {
    "B": ("step_B", "zero_metal_B", "update_H"),
    "D": ("step_D", "zero_metal_D", "update_E"),
}


def fused_passes(family: Dict[str, Any]) -> Tuple[str, ...]:
    """The driver passes one launch of this family performs, DECLARED BY THIS GATE.

    DECLARED HERE AND NOT READ FROM THE MODULE, which is the whole force of
    :func:`leg_driver_order`: that leg requires the module's own ``REPLACES`` to EQUAL
    this tuple, and a gate that derived it from the module would be comparing a value
    with itself.

    A FAMILY MAY DECLARE A SHORTER RUN THAN ITS SEAM'S. The no-absorber product carries
    no ``zero_metal_D`` -- it refuses every walled run instead -- so its two consults
    are contiguous with nothing between them, and requiring the seam's three passes
    would fail it for a pass it correctly does not perform.
    """
    return tuple(family.get("fused_passes", FUSED_PASSES_BY_SEAM[family["seam"]]))


def seam_span(family: Dict[str, Any]) -> Tuple[str, ...]:
    """The driver calls this family's seam spans, in driver order.

    DERIVED from :data:`DRIVER_ORDER` between the family's FIRST and LAST fused pass,
    so the span always contains every in-seam pass whether the product carries it or
    not -- which is what makes ``unfused_passes`` run the ones it does not.
    """
    passes = fused_passes(family)
    return DRIVER_ORDER[DRIVER_ORDER.index(passes[0]):
                        DRIVER_ORDER.index(passes[-1]) + 1]

_TAKES_PML = {"step_B", "update_H", "step_D", "update_E", "update_P"}


def run_pass(name: str, fields, pml) -> None:
    function = getattr(stepping, name)
    if name in _TAKES_PML:
        function(fields, pml)
    else:
        function(fields)


def array_step(fields, pml) -> None:
    """One COMPLETE driver step on the array path. The oracle."""
    for name in DRIVER_ORDER:
        run_pass(name, fields, pml)


def before_passes(family: Dict[str, Any], fields, pml) -> None:
    """Every driver pass that runs BEFORE this family's seam, in driver order.

    EMPTY ON THE MAGNETIC SEAM AND NOT ON THE ELECTRIC ONE, which is the one ordering
    fact a seam-parametrised harness has to get right: ``step_B`` and ``update_H``
    come first in ``FdtdDriver.step``, so a D-side subject that launched before
    running them would be comparing against an oracle a half-step ahead of it. On the
    B seam this is a no-op and the sequence is bit-for-bit the one this gate ran
    before it was parametrised.
    """
    span = seam_span(family)
    for name in DRIVER_ORDER[:DRIVER_ORDER.index(span[0])]:
        run_pass(name, fields, pml)


def unfused_passes(family: Dict[str, Any], fields, pml) -> None:
    """Every driver pass AFTER this family's seam that the fused launch did not do.

    THE TWO MIRROR FILLS RUN HERE, and that is not an oversight. They are between the
    two fused slots in the driver, so a launch spanning the seam must leave them
    correct -- and on every configuration these predicates admit they are no-ops,
    because all four refuse a folded grid. Running them anyway is what makes that a
    MEASUREMENT rather than a claim: if either ever wrote a word on an admitted row,
    the comparison would see it.
    """
    span = seam_span(family)
    fused = fused_passes(family)
    for name in DRIVER_ORDER[DRIVER_ORDER.index(span[0]):]:
        if name not in fused:
            run_pass(name, fields, pml)


def unfused_passes_after_seam(family: Dict[str, Any], fields, pml) -> None:
    """The passes AFTER a seam the separate route already ran whole."""
    span = seam_span(family)
    for name in DRIVER_ORDER[DRIVER_ORDER.index(span[-1]) + 1:]:
        run_pass(name, fields, pml)


def leg_driver_order(family: Dict[str, Any]) -> Dict[str, Any]:
    """``DRIVER_ORDER`` must be ``FdtdDriver.step``'s own call sequence.

    Read out of ``driver.py``'s text rather than by importing the driver: the question
    is what the SOURCE says, and a gate that imported the module to ask would be
    asking a different object than the one a reader checks.

    ``REPLACES`` is additionally required to be the FIRST and LAST of a CONTIGUOUS RUN
    of that order, with every pass between them accounted for. A product whose declared
    passes were not contiguous would be claiming to span a driver call it does not sit
    around.
    """
    path = os.path.join(_REPO_API, "meep_gpu", "driver.py")
    with open(path, "r", encoding="utf-8") as handle:
        text = handle.read()
    match = re.search(r"\n    def step\(self.*?(?=\n    def )", text, re.S)
    if match is None:
        return {"passed": False,
                "why": "FdtdDriver.step could not be located in driver.py"}
    body = match.group(0)
    names = "|".join(sorted(set(DRIVER_ORDER), key=len, reverse=True))
    found = tuple(re.findall(rf"\b({names})\(self\.fields", body))
    module = family["module"]
    seam = family["seam"]
    fused = fused_passes(family)
    span = seam_span(family)
    replaces = tuple(module.REPLACES)
    start = DRIVER_ORDER.index(replaces[0]) if replaces[0] in DRIVER_ORDER else -1
    # CONTIGUOUS MEANS "FIRST AND LAST OF A RUN OF DRIVER ORDER", and the run is the
    # one the GATE declares rather than the one the module does -- ``replaces ==
    # fused`` above is what ties the module to it. The span is the driver calls
    # between those two, which for a product carrying no in-seam pass is exactly the
    # two consults and nothing else.
    contiguous = (start >= 0
                  and DRIVER_ORDER[start] == span[0]
                  and DRIVER_ORDER[start + len(span) - 1] == span[-1])
    uncarried = [name for name in span if name not in replaces]
    return {
        "passed": bool(found == DRIVER_ORDER and replaces == fused and contiguous),
        "driver_file_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "driver_call_sequence": list(found),
        "gate_declared_order": list(DRIVER_ORDER),
        "seam": seam,
        "module_replaces": list(replaces),
        "gate_declared_fused_passes": list(fused),
        "driver_span_from_first_declared_pass": list(span),
        "passes_before_this_seam": list(
            DRIVER_ORDER[:DRIVER_ORDER.index(span[0])]),
        "passes_inside_the_seam_the_product_does_not_carry": uncarried,
        "why_those_are_safe": (
            f"fill_symmetry_bc_{seam} and fill_folded_far_ghosts_{seam} do nothing on "
            f"an unfolded grid, and every predicate here refuses a folded one -- twice "
            f"over, once in the curl half's own clause and once in the product's. "
            f"zero_metal_{seam} appears in this list ONLY for a product that refuses "
            f"every walled run, where it likewise writes nothing. They are still RUN "
            f"by the subject engine (unfused_passes), so a row on which any of them "
            f"wrote a word would diverge rather than pass."),
    }


# ---------------------------------------------------------------------------
# THE TWO FAMILIES
# ---------------------------------------------------------------------------
#
# Everything that differs between the two products lives in one dict each, and every
# leg below reads it. A leg that special-cased a family in its own body would be a
# place the two verdicts could stop meaning the same thing.

#: The Cartesian complex fixtures. THE EXTENTS ARE DELIBERATELY UNEQUAL: a cube lets
#: an index decomposition swap i and k and stay bit-identical, and this kernel
#: decomposes a flat thread index into i/j/k and indexes THREE coefficient vectors of
#: three different lengths with them.
#:
#: EVERY AXIS IS WALLED SOMEWHERE and one row is walled NOWHERE. The wall table this
#: product carries is the B DIAGONAL -- Bx on x, By on y, Bz on z -- so a row that
#: walled only one axis cannot tell the diagonal from the D-side off-diagonal
#: complement, and the all-periodic row proves the family is not wall-dependent.
#:
#: THE PHASED ROWS ARE THE OTHER HALF OF THIS FAMILY. A Bloch phase reaches only the
#: WRAPPED LANE of the curl, so a phased row exercises ``cshift_up``'s rotation that
#: no k = 0 row can; and a phase on one axis with walls on another is where a kernel
#: that applied the rotation to the wrong axis is separable from one that skipped it.
COMPLEX_SPECS: Tuple[Dict[str, Any], ...] = (
    {"label": "all_periodic", "boundaries": ("periodic", "periodic", "periodic"),
     "k_point": (0.0, 0.0, 0.0), "pml": 2, "cell": (9.0, 10.0, 11.0)},
    {"label": "wall_x", "boundaries": ("metallic", "periodic", "periodic"),
     "k_point": (0.0, 0.0, 0.0), "pml": 2, "cell": (9.0, 10.0, 11.0)},
    {"label": "wall_y", "boundaries": ("periodic", "metallic", "periodic"),
     "k_point": (0.0, 0.0, 0.0), "pml": 2, "cell": (9.0, 10.0, 11.0)},
    {"label": "wall_z", "boundaries": ("periodic", "periodic", "metallic"),
     "k_point": (0.0, 0.0, 0.0), "pml": 2, "cell": (9.0, 10.0, 11.0)},
    {"label": "wall_xyz", "boundaries": ("metallic", "metallic", "metallic"),
     "k_point": (0.0, 0.0, 0.0), "pml": 2, "cell": (9.0, 10.0, 11.0)},
    {"label": "bloch_x", "boundaries": ("periodic", "periodic", "periodic"),
     "k_point": (0.37, 0.0, 0.0), "pml": 2, "cell": (9.0, 10.0, 11.0)},
    {"label": "bloch_xyz", "boundaries": ("periodic", "periodic", "periodic"),
     "k_point": (0.37, -0.21, 0.5), "pml": 2, "cell": (9.0, 10.0, 11.0)},
    # A PHASE ON ONE AXIS AND WALLS ON THE OTHER TWO: the row where a rotation
    # applied to the wrong axis and one skipped altogether are separable.
    {"label": "bloch_y_wall_xz", "boundaries": ("metallic", "periodic", "metallic"),
     "k_point": (0.0, 0.42, 0.0), "pml": 2, "cell": (9.0, 10.0, 11.0)},
    # A THINNER LAYER on a different shape: the absorber profile changes, so a
    # coefficient index that happened to land inside a flat region of a 2-cell layer
    # has a different table to be wrong about.
    {"label": "wall_xy_thin_pml", "boundaries": ("metallic", "metallic", "periodic"),
     "k_point": (0.0, 0.0, 0.0), "pml": 1, "cell": (12.0, 9.0, 10.0)},
)

#: The Dcyl complex fixtures. |m| = 1 and |m| >= 2 are DIFFERENT KERNELS in effect --
#: ``m_class`` selects the axis increment on one and the six-volume axis zeroing on
#: the other -- so both classes are swept, at both terminations of the z axis, with
#: and without ``accurate_fields_near_cylorigin`` (which moves ``zero_rows``).
CYLINDRICAL_SPECS: Tuple[Dict[str, Any], ...] = (
    {"label": "m1_z_periodic", "m": 1, "shape": (9, 1, 11), "z_kind": "periodic",
     "accurate": False, "absorber": "thick"},
    {"label": "m1_z_metallic", "m": 1, "shape": (9, 1, 11), "z_kind": "metallic",
     "accurate": False, "absorber": "thick"},
    {"label": "m2_z_periodic", "m": 2, "shape": (9, 1, 11), "z_kind": "periodic",
     "accurate": False, "absorber": "thick"},
    {"label": "m2_z_metallic", "m": 2, "shape": (9, 1, 11), "z_kind": "metallic",
     "accurate": False, "absorber": "thick"},
    # ``accurate_fields_near_cylorigin`` DROPS THE ZERO ROWS near the axis, and Grid
    # refuses it above courant 1/(|m| + 0.5): without those rows the near-axis update
    # is unstable there, as a smooth growing mode rather than a crash. The bound is
    # the fixture's, not a tuning knob -- 0.4 at |m| = 2 and 0.28 at |m| = 3.
    {"label": "m2_accurate", "m": 2, "shape": (9, 1, 11), "z_kind": "metallic",
     "accurate": True, "absorber": "thick", "courant": 0.4},
    {"label": "m3_accurate", "m": 3, "shape": (11, 1, 9), "z_kind": "periodic",
     "accurate": True, "absorber": "thick", "courant": 0.28},
    {"label": "m_negative_2", "m": -2, "shape": (9, 1, 11), "z_kind": "metallic",
     "accurate": False, "absorber": "thick"},
    {"label": "m1_thin_absorber", "m": 1, "shape": (12, 1, 9), "z_kind": "metallic",
     "accurate": False, "absorber": "thin"},
    # m = 0 UNDER COMPLEX STORAGE (2026-09-04): the third ``m_class`` arm of the
    # certified curl, and the arm the corpus row ``examples:dipole_in_vacuum_cyl_
    # off_axis.py`` needs -- grid (150, 1, 300), m = 0, PML, z metallic, two ELECTRIC
    # sources in the D seam, which is why the m = 0 label rides in both cylindrical
    # families' deposit_labels. Both z terminations at the small shape, plus the
    # corpus row's own shape so the configuration the board prices is one the gate
    # stepped.
    {"label": "m0_z_metallic", "m": 0, "shape": (9, 1, 11), "z_kind": "metallic",
     "accurate": False, "absorber": "thick"},
    {"label": "m0_z_periodic", "m": 0, "shape": (9, 1, 11), "z_kind": "periodic",
     "accurate": False, "absorber": "thick"},
    {"label": "m0_corpus_shape", "m": 0, "shape": (150, 1, 300), "z_kind": "metallic",
     "accurate": False, "absorber": "thick"},
)


def case_rng(label: str) -> np.random.Generator:
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    return np.random.default_rng(SEED + int.from_bytes(digest[:4], "big"))


def _seed_one_complex(shape, value_class: str, rng, name: str = "seed") -> np.ndarray:
    """One complex64 volume in a value class, as a host array.

    THE TWO PLANES ARE ASSIGNED, NEVER COMBINED AS ``re + 1j*im``. Measured by a
    sibling tranche before it was reasoned about: ``1j * im`` carries a real part of
    ``0.0 * im``, so ``re + 1j*im`` computes ``(-0.0) + (+0.0) = +0.0`` and DESTROYS
    every negative zero in the plane the zero cross terms act on.
    """
    if value_class == "uniform":
        real = rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)
        imag = rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)
    elif value_class == "subnormal_band":
        real = subnormal_band_hosts((name,), tuple(shape), rng)[name]
        imag = subnormal_band_hosts((name,), tuple(shape), rng)[name]
    else:
        raise ValueError(f"value class {value_class!r} is not one of {VALUE_CLASSES}")
    host = np.empty(tuple(shape), dtype=np.complex64)
    host.real = np.asarray(real, dtype=np.float32)
    host.imag = np.asarray(imag, dtype=np.float32)
    return np.ascontiguousarray(host)


def _seed_complex_state(fields, grid, value_class: str, rng,
                        names: Sequence[str] = STATE_NAMES) -> Dict[str, np.ndarray]:
    """Seed every complex volume a complete step touches; return the host words.

    ``names`` IS THE FAMILY'S SET. A no-absorber engine allocates no ``fu`` and no
    ``f_w``, so seeding the default tuple there would raise on twelve arrays that do
    not exist.

    THE TWO PLANES ARE ASSIGNED, NEVER COMBINED AS ``re + 1j*im``. Measured by a
    sibling tranche before it was reasoned about: ``1j * im`` carries a real part of
    ``0.0 * im``, so ``re + 1j*im`` computes ``(-0.0) + (+0.0) = +0.0`` and DESTROYS
    every negative zero in the plane the zero cross terms act on.

    THE AUXILIARIES START NONZERO: a zero ``fu``/``f_w`` makes ``kms * prev`` exactly
    zero on the first launch whatever ``kms`` holds, so a mis-indexed coefficient
    would only show from launch two.
    """
    xp = grid.xp
    shape = tuple(int(n) for n in grid.shape)
    words: Dict[str, np.ndarray] = {}
    for name in names:
        host = _seed_one_complex(shape, value_class, rng, name)
        getattr(fields, name)[...] = xp.asarray(host)
        words[name] = host.view(np.float32).ravel().copy()
    return words


def anisotropic_epsilon(fields, grid, rng) -> Dict[str, Any]:
    """Install THREE DISTINCT inverse-permittivity volumes and report them.

    WHY THE ELECTRIC SEAM NEEDS THIS AND THE MAGNETIC ONE DOES NOT. ``update_E``
    multiplies the flux density by ``inverse_epsilon_for(component)`` -- three pointers
    THIS family's signature binds, and the one group in it that is NOT ``__restrict__``
    because an isotropic run legitimately hands one allocation three times. On a vacuum
    fixture all three arrays are that one allocation and every entry is 1.0, so binding
    them in the wrong order, or dropping the multiply outright, is BIT-IDENTICAL: every
    mutation on the binding would score UNCAUGHT while looking green, which is the
    vacuity this track exists to refuse. ``update_H`` has no such factor at all
    (mu = 1 is baked into the choice of source), so the B families need nothing here.

    DIAGONAL ANISOTROPY ONLY. An OFF-DIAGONAL chi1inv row makes ``update_E`` a stencil
    over the partner components' volumes, which the constitutive half refuses by name
    and the deposit repair refuses by name -- so a fixture carrying one would not be
    admitted at all. ``set_epsilon_volumes`` with three distinct arrays and no row is
    exactly the diagonal case both admit.

    THE VALUES COME OFF THE CASE'S OWN GENERATOR, after the field seeding, so the
    three engines a case builds from one label get the SAME material -- a per-build
    random draw would make the three subjects three different runs.
    """
    xp = grid.xp
    shape = tuple(int(n) for n in grid.shape)
    epsilon: Dict[str, Any] = {}
    inverse: Dict[str, Any] = {}
    for index, component in enumerate(("Ex", "Ey", "Ez")):
        # Distinct per component AND per cell: a per-component constant would leave a
        # SWAPPED binding visible but a rotated-index one inert.
        values = np.float32(1.25 + 0.5 * index) + np.float32(0.5) * rng.random(
            shape, dtype=np.float32)
        epsilon[component] = xp.asarray(values, dtype=xp.float32)
        inverse[component] = xp.asarray(
            np.float32(1.0) / values, dtype=xp.float32)
    fields.set_epsilon_volumes(epsilon, inverse)
    distinct = len({int(fields.inverse_epsilon_for(c).data.ptr)
                    for c in ("Ex", "Ey", "Ez")})
    if distinct != 3:
        raise SystemExit(
            f"the anisotropic fixture installed {distinct} distinct inverse-epsilon "
            f"allocations, not 3; every binding mutation on this seam would be inert")
    return {"distinct_inverse_epsilon_allocations": distinct,
            "component_means": [float(np.asarray(to_host(inverse[c])).mean())
                                for c in ("Ex", "Ey", "Ez")]}


def build_complex(spec: Dict[str, Any], value_class: str, rng):
    """One seeded Cartesian complex engine: ``(fields, grid, pml, host words)``."""
    from meep_gpu.fields import Fields  # noqa: PLC0415
    from meep_gpu.grid import Grid  # noqa: PLC0415
    from meep_gpu.pml import PML  # noqa: PLC0415

    grid = Grid(resolution=1.0, cell_size=tuple(spec["cell"]),
                boundaries=tuple(spec["boundaries"]),
                k_point=tuple(spec["k_point"]), xp=cp, courant=0.5)
    fields = Fields(grid=grid, force_complex_fields=True)
    fields.enable_field_storage()
    fields.enable_pml_storage()
    pml = PML(grid=grid, thickness=spec["pml"])
    host = _seed_complex_state(fields, grid, value_class, rng)
    return fields, grid, pml, host


def build_cylindrical(spec: Dict[str, Any], value_class: str, rng):
    """One seeded Dcyl complex engine: ``(fields, grid, pml, host words)``."""
    from meep_gpu.fields import Fields  # noqa: PLC0415
    from meep_gpu.grid import Grid  # noqa: PLC0415
    from meep_gpu.pml import PML  # noqa: PLC0415

    shape = tuple(int(n) for n in spec["shape"])
    grid = Grid(resolution=1.0, cell_size=(float(shape[0]), 0.0, float(shape[2])),
                cylindrical=True, m=int(spec["m"]),
                boundaries={"z": spec["z_kind"]},
                accurate_fields_near_cylorigin=bool(spec.get("accurate", False)),
                courant=float(spec.get("courant", 0.5)), xp=cp)
    if tuple(int(n) for n in grid.shape) != shape:
        raise SystemExit(f"grid built {tuple(grid.shape)}, spec asked for {shape}")
    fields = Fields(grid=grid, force_complex_fields=True)
    fields.enable_field_storage()
    fields.enable_pml_storage()
    if spec.get("absorber") == "thin":
        thickness = {"x": (0, 2), "z": 2}
    else:
        thickness = {"x": (0, max(2, shape[0] // 4)), "z": max(2, shape[2] // 4)}
    pml = PML(grid=grid, thickness=thickness)
    host = _seed_complex_state(fields, grid, value_class, rng)
    return fields, grid, pml, host


#: The complex NO-ABSORBER fixtures. THREE THINGS ARE SWEPT HERE THAT NO OTHER FAMILY
#: IN THIS FILE SWEEPS, and each is a compile-time or a bit-order axis rather than a
#: coefficient:
#:
#: * **THE CURL ARM.** This family's certified curl ships TWO kernels -- a plain tail
#:   and a conductive one -- and the weld is built on both. Every corpus row of its
#:   board cell takes the CONDUCTIVE one, so a sweep that only carried the plain arm
#:   would gate a product nothing uses; a sweep that only carried the conductive one
#:   would ship an ungated kernel. Both are here, on phased and unphased rows.
#: * **THE POLE COUNT.** 0 exercises ``update_E``'s pole-free store, which is a REAL
#:   WRITE (``E = D * inv_eps``) and not a no-op; 1 is the corpus's shape; 2 is what
#:   makes the LEFT-TO-RIGHT subtraction order observable at all -- with one pole every
#:   order agrees, and the pre-sum mutation would be inert on every fixture.
#: * **THE INERT LAYER.** ``pml.is_active`` is False for a zero-thickness PML, so
#:   ``stepping`` takes the same tail and the predicate admits it. Swept because "inert
#:   is the same as absent" is otherwise an inference about ``pml.py``.
#:
#: NO WALLED ROW APPEARS, AND THAT IS A REFUSAL RATHER THAN A GAP. The product refuses
#: every metallic run by name -- it carries no ``zero_metal_D`` -- so a walled fixture
#: is not a case this gate skipped but a configuration the predicate declines, and it
#: is exercised in :func:`leg_refusal` as a refusal instead. The cost of that decision
#: is recorded in the module's own ``REPLACES``: 0 corpus slots.
#:
#: THE EXTENTS ARE DELIBERATELY UNEQUAL for the reason the sibling specs give: a cube
#: lets an index decomposition swap i and k and stay bit-identical.
NO_PML_COMPLEX_SPECS: Tuple[Dict[str, Any], ...] = (
    {"label": "phased_conductive_2pole", "cell": (9.0, 10.0, 11.0), "dimensions": 3,
     "boundaries": ("periodic", "periodic", "periodic"),
     "k_point": (0.2, -0.35, 0.1), "conductive": True, "poles": 2,
     "absorber": "none"},
    {"label": "phased_plain_1pole", "cell": (9.0, 10.0, 11.0), "dimensions": 3,
     "boundaries": ("periodic", "periodic", "periodic"),
     "k_point": (0.2, -0.35, 0.1), "conductive": False, "poles": 1,
     "absorber": "none"},
    {"label": "unphased_conductive_1pole", "cell": (9.0, 10.0, 11.0),
     "dimensions": 3, "boundaries": ("periodic", "periodic", "periodic"),
     "k_point": (0.0, 0.0, 0.0), "conductive": True, "poles": 1,
     "absorber": "none"},
    {"label": "unphased_plain_0pole", "cell": (9.0, 10.0, 11.0), "dimensions": 3,
     "boundaries": ("periodic", "periodic", "periodic"),
     "k_point": (0.0, 0.0, 0.0), "conductive": False, "poles": 0,
     "absorber": "none"},
    {"label": "inert_layer_conductive_1pole", "cell": (11.0, 9.0, 10.0),
     "dimensions": 3, "boundaries": ("periodic", "periodic", "periodic"),
     "k_point": (0.3, 0.1, 0.0), "conductive": True, "poles": 1,
     "absorber": "inert_layer"},
    {"label": "two_d_phased_plain_1pole", "cell": (12.0, 14.0, 0.0),
     "dimensions": 2, "boundaries": ("periodic", "periodic", "periodic"),
     "k_point": (0.2, 0.4, 0.0), "conductive": False, "poles": 1,
     "absorber": "none"},
    {"label": "one_d_phased_conductive_1pole", "cell": (0.0, 0.0, 24.0),
     "dimensions": 1, "boundaries": ("periodic", "periodic", "periodic"),
     "k_point": (0.0, 0.0, 0.35), "conductive": True, "poles": 1,
     "absorber": "none"},
)


def build_no_pml_complex(spec: Dict[str, Any], value_class: str, rng):
    """One seeded complex NO-ABSORBER engine: ``(fields, grid, layer, host words)``.

    ``enable_pml_storage`` IS NEVER CALLED and that is load-bearing rather than an
    omission: it allocates the ``fu``/``f_w`` auxiliaries, and both certified halves
    REFUSE a run where one exists while the layer is inert. The comparison set
    (:data:`NO_PML_STATE_NAMES`) is the mirror of that fact.

    THE MATERIAL IS ANISOTROPIC AND THE CONDUCTIVITY IS PER COMPONENT, for the reasons
    :func:`anisotropic_epsilon` gives: against ones the inverse-permittivity multiply is
    invisible, against one shared volume a component-aliasing defect is invisible, and
    against a constant volume an index error is invisible. Here it is doubly
    load-bearing, because this launcher binds ``inv_eps`` a SECOND time in every unused
    pole slot.

    THE POLE HISTORIES ARE SEEDED NONZERO, including the scratch: it is ``update_P``'s
    OUTPUT and starts as whatever the previous rotation retired, so a kernel that
    failed to write some cells would agree with an oracle that also wrote nothing there.
    """
    from meep_gpu.dispersion import PolarizationState, Susceptibility  # noqa: PLC0415
    from meep_gpu.fields import Fields  # noqa: PLC0415
    from meep_gpu.grid import Grid  # noqa: PLC0415
    from meep_gpu.pml import PML  # noqa: PLC0415

    grid = Grid(resolution=1.0, cell_size=tuple(spec["cell"]),
                boundaries=tuple(spec["boundaries"]),
                dimensions=int(spec.get("dimensions", 3)),
                k_point=tuple(spec["k_point"]), xp=cp, courant=0.5)
    fields = Fields(grid=grid, force_complex_fields=True)
    fields.enable_field_storage()
    shape = tuple(int(n) for n in grid.shape)

    anisotropic_epsilon(fields, grid, rng)
    if spec["conductive"]:
        # A GRADED conductivity, PER COMPONENT, which is the mapping form
        # ``mp.Absorber`` produces (fields.py:755-759: MEEP point-samples each
        # component at its own Yee position). Under a UNIFORM sigma the three condfac
        # pointers are the same array and a component-aliasing defect is unobservable.
        for setter, names in ((fields.set_d_conductivity, ("Dx", "Dy", "Dz")),
                              (fields.set_b_conductivity, ("Bx", "By", "Bz"))):
            setter({name: cp.asarray(np.ascontiguousarray(
                rng.uniform(0.05, 0.4, size=shape).astype(np.float32)))
                for name in names})
    for index in range(int(spec["poles"])):
        sigma = {name: cp.asarray(np.ascontiguousarray(
            rng.uniform(0.1, 0.9, size=shape).astype(np.float32)))
            for name in ("Ex", "Ey", "Ez")}
        fields.polarizations.append(PolarizationState(
            Susceptibility(frequency=1.05 + 0.4 * index, gamma=0.05 + 0.02 * index),
            sigma, grid, fields._field_dtype()))  # noqa: SLF001 - the shipped dtype

    host = _seed_complex_state(fields, grid, value_class, rng,
                              names=NO_PML_STATE_NAMES)
    for state in fields.polarizations:
        for component in state.driven():
            for attribute in ("P", "P_prev"):
                target = getattr(state, attribute)[component]
                target[...] = cp.asarray(_seed_one_complex(shape, value_class, rng))
        state._scratch[...] = cp.asarray(  # noqa: SLF001 - the shipped rotation slot
            _seed_one_complex(shape, value_class, rng))
    layer = (PML(grid=grid, thickness=((0, 0), (0, 0), (0, 0)))
             if spec["absorber"] == "inert_layer" else None)
    return fields, grid, layer, host


def no_pml_read_only_material(fields) -> Dict[str, Any]:
    """The material volumes a no-absorber launch binds and must not write.

    THE REPLACEMENT FOR :data:`IMMUTABLE_PML` ON THIS FAMILY, and it is a replacement
    rather than an omission: with no layer the PML snapshot comes back EMPTY, and a
    read-only check over an empty set reports "nothing drifted" whatever the kernel
    did. Every array here is bound as a pointer to the fused launch -- three inverse
    permittivities (twice over, since the unused pole slots bind them again), six
    conductivity coefficients on the conductive arm, and every susceptibility sigma --
    and every one of them is read-only to it.
    """
    out: Dict[str, Any] = {}
    for component in ("Ex", "Ey", "Ez"):
        out[f"inv_eps.{component}"] = fields.inverse_epsilon_for(component)
    for component in ("Dx", "Dy", "Dz"):
        for stem, reader in (("condfac", fields.condfac_for),
                             ("condinv", fields.condinv_for)):
            value = reader(component)
            if value is not None:
                out[f"{stem}.{component}"] = value
    for index, state in enumerate(getattr(fields, "polarizations", ()) or ()):
        for component in sorted(state.driven()):
            sigma = (getattr(state, "sigma", {}) or {}).get(component)
            if sigma is not None and getattr(sigma, "shape", ()):
                out[f"sigma[{index}][{component}]"] = sigma
    return out


def _no_pml_complex_electric_family() -> Dict[str, Any]:
    """The complex NO-ABSORBER product on the ELECTRIC seam.

    THE SIMPLEST WELD ON THE BOARD and the only one here that refuses an absorber. It
    carries no coefficient vector (there is none without a layer), no split-field
    auxiliary, no sub-lattice pairing and no in-seam pass -- so ``resolve`` returns no
    ``tables`` and no ``walls``, and the legs that read those keys read them with
    ``.get``. What it DOES carry that no sibling does is the POLE BANK, and the
    ``curl_arm`` that decides which of two certified curls the weld is built on.
    """
    from meep_gpu.cuda_kernels import complex_no_pml_kernels as halves  # noqa: PLC0415
    from meep_gpu.cuda_kernels import (  # noqa: PLC0415
        no_pml_complex_fused_electric_pair as module,
    )

    def variant_of(spec: Dict[str, Any]) -> str:
        return "conductive" if spec["conductive"] else "plain"

    def resolve(fields, grid, pml, arm):
        from meep_gpu.cuda_kernels import complex_pml_kernels  # noqa: PLC0415

        # ``True`` IS THE BACKWARD FLAG, read off complex_emitter.KERNELS['step_D'][1]
        # the same way the certified launcher reads it. The forward table is a step_B
        # launch's and is a wrong phase here.
        flags, values = complex_pml_kernels.bloch_phase_arguments(grid, True)
        return {"boundary_codes": complex_pml_kernels.complex_boundary_codes(grid),
                "phase_flags": flags, "phase_values": values,
                # THE CURL ARM IS READ FROM THE FIELDS by the certified classifier, not
                # from the spec: the spec says what the fixture installed and the
                # classifier says what `_apply_curl` will do with it, and only the
                # second decides which kernel is correct.
                "curl_arm": module.no_pml_complex_curl_arm(fields, "step_D"),
                "dtdx": float(grid.dt / grid.dx), "arm": arm}

    def launch(fields, arguments, kernel=None):
        # THE POLE MUTATION IS RESOLVED HERE, PER LAUNCH, and never baked into the
        # resolved arguments -- for the reason the shipped launcher resolves the banks
        # per launch: ``PolarizationState.update`` rotates P/P_prev/scratch on every
        # ``update_P``, so a bank captured once would be a retired history from the
        # second step onward and the leg would be arming staleness instead of the
        # defect it declares.
        poles = None
        mutation = arguments.get("poles_mutation")
        if mutation is not None:
            banks = {component: tuple(state.P[component] for state in states)
                     for component, states
                     in halves.poles_per_component(fields).items()}
            if mutation == "rotate":
                order = ("Ey", "Ez", "Ex")
                poles = {component: banks[other]
                         for component, other in zip(("Ex", "Ey", "Ez"), order)}
            elif mutation == "zero":
                poles = {component: () for component in ("Ex", "Ey", "Ez")}
            else:
                raise ValueError(f"unknown pole mutation {mutation!r}")
        return module.launch_no_pml_complex_fused_electric_pair(
            fields, arguments["boundary_codes"], arguments["phase_flags"],
            arguments["phase_values"], arguments["dtdx"], arguments["curl_arm"],
            arguments["arm"], kernel, poles)

    def separate(fields, grid, pml, arm):
        """The certified route: curl, the three in-seam passes, constitutive.

        THE CURL LAUNCHER TAKES ``fields`` FIRST and its own arm classifier decides
        the tail; handing it ``grid`` is what makes the boundary codes, the BACKWARD
        phase table and ``dtdx`` derived by the same functions the predicate asks, so
        this route and the fused one cannot disagree about them by construction.
        """
        halves.step_complex_no_pml_curl(fields, "step_D", arm, grid=grid)
        stepping.fill_symmetry_bc_D(fields)
        stepping.zero_metal_D(fields)
        stepping.fill_folded_far_ghosts_D(fields)
        halves.update_E_complex_no_pml_stored(fields, arm)
        return halves  # imported so the reference is visible in a traceback

    return {
        "name": "no_pml_complex_electric",
        "shape": "cartesian",
        "seam": "D",
        "curl_sub_step": "step_D",
        "constitutive_side": "update_E",
        "seam_constitutive": "E",
        "module": module,
        "specs": NO_PML_COMPLEX_SPECS,
        "build": build_no_pml_complex,
        "covers": module.covers_no_pml_complex_fused_electric_pair,
        "resolve": resolve,
        "launch": launch,
        "separate": separate,
        "variants": ("plain", "conductive"),
        "variant_of": variant_of,
        "source": lambda arm, variant: (
            module.no_pml_complex_fused_electric_pair_source(
                arm, variant == "conductive")),
        "kernel_symbol": module.kernel_name,
        "certified_curl": lambda arm, variant: halves.kernel_source(
            "step_D_conductive" if variant == "conductive" else "step_D", arm),
        "certified_constitutive": lambda arm, variant: halves.kernel_source(
            "update_E", arm),
        "emitter_module": ("meep_gpu.cuda_kernels."
                           "no_pml_complex_fused_electric_pair"),
        "emitter_attribute": "no_pml_complex_fused_electric_pair_source",
        "disjoint": lambda fields, arguments: module.assert_disjoint_bindings(
            fields, arguments["curl_arm"]),
        # NO ``fu`` AND NO ``f_w``: without a layer Fields allocates neither, and both
        # halves refuse a run where one exists anyway.
        "state_names": NO_PML_STATE_NAMES,
        "pole_state": True,
        # THE CLAUSE IN THE OTHER DIRECTION, declared rather than inferred from the
        # family's name: this predicate REQUIRES an inert or absent layer where the
        # four split-field products require an active one, and `leg_refusal` builds
        # both directions off this flag.
        "requires_inert_layer": True,
        "band_must_move": tuple([f"D{axis}" for axis in "xyz"]
                                + [f"E{axis}" for axis in "xyz"]),
        "read_only_material": no_pml_read_only_material,
        # ``zero_metal_D`` IS NOT HERE. The product refuses every walled run, so the
        # two consults are contiguous with nothing between them -- and this tuple is
        # what `leg_driver_order` requires the module's own REPLACES to equal.
        "fused_passes": ("step_D", "update_E"),
        "mutation_labels": ("phased_conductive_2pole", "phased_plain_1pole",
                            "unphased_plain_0pole"),
        # NO DEPOSIT LEG. This product declares CARRIES_DEPOSIT_REPAIR False and
        # refuses every in-seam electric source, so there is no carried deposit to
        # compare; `leg_source_refusal` measures the refusal AND that it is not
        # decorative, which is what a deposit leg would otherwise have proved.
        "deposit_labels": (),
        "source_refusal_labels": ("phased_conductive_2pole",),
        "reduced_labels": ("phased_conductive_2pole", "phased_plain_1pole"),
    }


def _complex_family() -> Dict[str, Any]:
    # ONLY THE CuPy-FREE MODULES ARE IMPORTED HERE. The product module takes CuPy
    # defensively (its predicate and emitter run at the merge bar) and
    # ``in_seam_coverage`` never touches it, so the source legs -- driver order, the
    # lift -- run on a laptop. Every device-side import is inside a closure the
    # laptop path never calls.
    from meep_gpu.cuda_kernels import complex_fused_magnetic_pair as module  # noqa: PLC0415
    from meep_gpu.cuda_kernels import in_seam_coverage  # noqa: PLC0415

    def resolve(fields, grid, pml, arm):
        from meep_gpu.cuda_kernels import complex_pml_kernels  # noqa: PLC0415

        flags, values = complex_pml_kernels.bloch_phase_arguments(grid, False)
        return {"tables": module.complex_fused_magnetic_pair_tables(pml),
                "boundary_codes": complex_pml_kernels.complex_boundary_codes(grid),
                "phase_flags": flags, "phase_values": values,
                "walls": in_seam_coverage.zero_metal_axes(grid),
                "dtdx": float(grid.dt / grid.dx), "arm": arm}

    def launch(fields, arguments, kernel=None):
        return module.launch_complex_fused_magnetic_pair(
            fields, arguments["tables"], arguments["boundary_codes"],
            arguments["phase_flags"], arguments["phase_values"],
            arguments["walls"], arguments["dtdx"], arguments["arm"], kernel)

    def separate(fields, grid, pml, arm):
        """The certified route: curl, the wall pass where live, constitutive."""
        from meep_gpu.cuda_kernels import complex_pml_kernels  # noqa: PLC0415
        from meep_gpu.cuda_kernels import in_seam_passes  # noqa: PLC0415

        complex_pml_kernels.step_fused_pml_complex(
            "step_B", fields, arm, grid=grid, pml=pml)
        stepping.fill_symmetry_bc_B(fields)
        # The CERTIFIED in-seam kernel refuses complex64 storage by name, so the
        # separate route runs the ARRAY PATH pass here. That is what makes the
        # fused product's carry worth something rather than a re-spelling.
        stepping.zero_metal_B(fields)
        stepping.fill_folded_far_ghosts_B(fields)
        complex_pml_kernels.update_fused_pml_complex("H", fields, arm, pml=pml)
        return in_seam_passes  # imported so the reference is visible in a traceback

    return {
        "name": "complex",
        "shape": "cartesian",
        "seam": "B",
        "curl_sub_step": "step_B",
        "constitutive_side": "update_H",
        "seam_constitutive": "H",
        "module": module,
        "specs": COMPLEX_SPECS,
        "build": build_complex,
        "covers": module.covers_complex_fused_magnetic_pair,
        "resolve": resolve,
        "launch": launch,
        "separate": separate,
        "source": module.complex_fused_magnetic_pair_source,
        "emitter_module": "meep_gpu.cuda_kernels.complex_fused_magnetic_pair",
        "emitter_attribute": "complex_fused_magnetic_pair_source",
        "disjoint": lambda fields, arguments: module.assert_disjoint_bindings(
            fields, arguments["tables"]),
        "mutation_labels": ("wall_xyz", "all_periodic", "bloch_xyz",
                            "bloch_y_wall_xz"),
        "deposit_labels": ("wall_xyz", "all_periodic"),
        "reduced_labels": ("wall_xyz", "bloch_xyz"),
    }


def _complex_electric_family() -> Dict[str, Any]:
    """The Cartesian complex product on the ELECTRIC seam.

    THE SAME SHAPE AS ITS MAGNETIC TWIN and three arguments apart, each of which
    compiles, runs, and is a different engine: the BACKWARD Bloch table
    (``bloch_phase_arguments(grid, True)``, whose imaginary parts are the conjugates),
    the MIRROR-IMAGE sub-lattice pairing, and the OFF-DIAGONAL wall table.
    """
    from meep_gpu.cuda_kernels import complex_fused_electric_pair as module  # noqa: PLC0415
    from meep_gpu.cuda_kernels import in_seam_coverage  # noqa: PLC0415

    def resolve(fields, grid, pml, arm):
        from meep_gpu.cuda_kernels import complex_pml_kernels  # noqa: PLC0415

        # ``True`` IS THE BACKWARD FLAG, read off complex_emitter.KERNELS['step_D'][1]
        # the same way complex_pml_kernels.step_fused_pml_complex reads it. The
        # forward table is the magnetic twin's and is a wrong phase here.
        flags, values = complex_pml_kernels.bloch_phase_arguments(grid, True)
        return {"tables": module.complex_fused_electric_pair_tables(pml),
                "boundary_codes": complex_pml_kernels.complex_boundary_codes(grid),
                "phase_flags": flags, "phase_values": values,
                "walls": in_seam_coverage.zero_metal_axes(grid),
                "dtdx": float(grid.dt / grid.dx), "arm": arm}

    def launch(fields, arguments, kernel=None):
        return module.launch_complex_fused_electric_pair(
            fields, arguments["tables"], arguments["boundary_codes"],
            arguments["phase_flags"], arguments["phase_values"],
            arguments["walls"], arguments["dtdx"], arguments["arm"], kernel)

    def separate(fields, grid, pml, arm):
        """The certified route: curl, the wall pass where live, constitutive."""
        from meep_gpu.cuda_kernels import complex_pml_kernels  # noqa: PLC0415
        from meep_gpu.cuda_kernels import in_seam_passes  # noqa: PLC0415

        complex_pml_kernels.step_fused_pml_complex(
            "step_D", fields, arm, grid=grid, pml=pml)
        stepping.fill_symmetry_bc_D(fields)
        # The CERTIFIED in-seam kernel refuses complex64 storage by name, so the
        # separate route runs the ARRAY PATH pass here.
        stepping.zero_metal_D(fields)
        stepping.fill_folded_far_ghosts_D(fields)
        complex_pml_kernels.update_fused_pml_complex("E", fields, arm, pml=pml)
        return in_seam_passes  # imported so the reference is visible in a traceback

    return {
        "name": "complex_electric",
        "shape": "cartesian",
        "seam": "D",
        "curl_sub_step": "step_D",
        "constitutive_side": "update_E",
        "seam_constitutive": "E",
        "module": module,
        "specs": COMPLEX_SPECS,
        "build": build_complex_electric,
        "covers": module.covers_complex_fused_electric_pair,
        "resolve": resolve,
        "launch": launch,
        "separate": separate,
        "source": module.complex_fused_electric_pair_source,
        "emitter_module": "meep_gpu.cuda_kernels.complex_fused_electric_pair",
        "emitter_attribute": "complex_fused_electric_pair_source",
        "disjoint": lambda fields, arguments: module.assert_disjoint_bindings(
            fields, arguments["tables"]),
        "mutation_labels": ("wall_xyz", "all_periodic", "bloch_xyz",
                            "bloch_y_wall_xz"),
        "deposit_labels": ("wall_xyz", "all_periodic"),
        "reduced_labels": ("wall_xyz", "bloch_xyz"),
    }


def _cylindrical_family() -> Dict[str, Any]:
    # Same rule as the Cartesian family: nothing that needs a device is imported at
    # family-construction time. ``cylindrical_complex_kernels`` DOES import CuPy at
    # module scope, so it is reached only inside the closures.
    from meep_gpu.cuda_kernels import cylindrical_fused_magnetic_pair as module  # noqa: PLC0415
    from meep_gpu.cuda_kernels import in_seam_coverage  # noqa: PLC0415
    from meep_gpu.cuda_kernels.cylindrical_prefix import cylindrical_prefix  # noqa: PLC0415

    def resolve(fields, grid, pml, arm):
        from meep_gpu.cuda_kernels import cylindrical_complex_kernels as certified  # noqa: PLC0415

        dtdx = float(grid.dt / grid.dx)
        m = int(grid.m)
        return {"tables": module.cylindrical_fused_magnetic_pair_tables(pml),
                "boundary_codes": certified.cylindrical_complex_boundary_codes(grid),
                "imr_rows": certified.imr_rows_for("step_B", grid.xp, m, dtdx,
                                                   int(fields.Bx.shape[0]),
                                                   fields.Bx.dtype),
                "increment_scalars": certified.axis_increment_scalars(m, dtdx),
                "m_class": certified.m_class(m),
                "zero_rows": certified.zero_rows(
                    m, bool(grid.accurate_fields_near_cylorigin)),
                "walls": in_seam_coverage.zero_metal_axes(grid),
                "dtdx": dtdx, "arm": arm, "fields": fields}

    def launch(fields, arguments, kernel=None):
        # THE PREFIX IS RESOLVED PER LAUNCH, never cached beside the tables: it is a
        # cumulative sum over the CURRENT Ep, so a value taken once would be a stale
        # field one step later -- the same class of defect as reading a pre-injection
        # B, arriving through a cache instead of a seam.
        return module.launch_cylindrical_fused_magnetic_pair(
            fields, arguments["tables"], cylindrical_prefix(fields, "step_B"),
            arguments["imr_rows"], arguments["boundary_codes"],
            arguments["increment_scalars"], arguments["m_class"],
            arguments["zero_rows"], arguments["walls"], arguments["dtdx"],
            arguments["arm"], kernel)

    def separate(fields, grid, pml, arm):
        from meep_gpu.cuda_kernels import complex_pml_kernels  # noqa: PLC0415
        from meep_gpu.cuda_kernels import cylindrical_complex_kernels as certified  # noqa: PLC0415

        certified.step_cylindrical_complex("step_B", fields, arm, grid=grid, pml=pml)
        stepping.fill_symmetry_bc_B(fields)
        stepping.zero_metal_B(fields)
        stepping.fill_folded_far_ghosts_B(fields)
        complex_pml_kernels.update_fused_pml_complex("H", fields, arm, pml=pml)
        return certified

    return {
        "name": "cylindrical",
        "shape": "cylindrical",
        "seam": "B",
        "curl_sub_step": "step_B",
        "constitutive_side": "update_H",
        "seam_constitutive": "H",
        "module": module,
        "specs": CYLINDRICAL_SPECS,
        "build": build_cylindrical,
        "covers": module.covers_cylindrical_fused_magnetic_pair,
        "resolve": resolve,
        "launch": launch,
        "separate": separate,
        "source": module.cylindrical_fused_magnetic_pair_source,
        "emitter_module": "meep_gpu.cuda_kernels.cylindrical_fused_magnetic_pair",
        "emitter_attribute": "cylindrical_fused_magnetic_pair_source",
        "disjoint": lambda fields, arguments: module.assert_disjoint_bindings(
            fields, arguments["tables"],
            cylindrical_prefix(fields, "step_B"), arguments["imr_rows"]),
        "mutation_labels": ("m2_z_metallic", "m1_z_metallic", "m3_accurate",
                            "m1_z_periodic", "m0_z_metallic"),
        "deposit_labels": ("m2_z_metallic", "m1_z_periodic", "m0_z_metallic"),
        "reduced_labels": ("m2_z_metallic", "m1_z_periodic", "m0_z_metallic"),
    }


def _cylindrical_electric_family() -> Dict[str, Any]:
    """The Dcyl complex product on the ELECTRIC seam.

    EVERY CYLINDRICAL QUANTITY HERE IS THE ``step_D`` ONE. The radial prefix sums Hp
    rather than Ep, the two i*m/r rows carry the OPPOSITE SIGNS
    (``cylindrical_complex_kernels.IMR_TERMS``), and the axis tail has TWO branches
    rather than one -- Dz alone at |m| = 1 and all six volumes at |m| >= 2. Each of
    those compiles and runs with the B side's value substituted.
    """
    from meep_gpu.cuda_kernels import cylindrical_fused_electric_pair as module  # noqa: PLC0415
    from meep_gpu.cuda_kernels import in_seam_coverage  # noqa: PLC0415
    from meep_gpu.cuda_kernels.cylindrical_prefix import cylindrical_prefix  # noqa: PLC0415

    def resolve(fields, grid, pml, arm):
        from meep_gpu.cuda_kernels import cylindrical_complex_kernels as certified  # noqa: PLC0415

        dtdx = float(grid.dt / grid.dx)
        m = int(grid.m)
        return {"tables": module.cylindrical_fused_electric_pair_tables(pml),
                "boundary_codes": certified.cylindrical_complex_boundary_codes(grid),
                "imr_rows": certified.imr_rows_for("step_D", grid.xp, m, dtdx,
                                                   int(fields.Dx.shape[0]),
                                                   fields.Dx.dtype),
                "increment_scalars": certified.axis_increment_scalars(m, dtdx),
                "m_class": certified.m_class(m),
                "zero_rows": certified.zero_rows(
                    m, bool(grid.accurate_fields_near_cylorigin)),
                "walls": in_seam_coverage.zero_metal_axes(grid),
                "dtdx": dtdx, "arm": arm, "fields": fields}

    def launch(fields, arguments, kernel=None):
        # THE PREFIX IS RESOLVED PER LAUNCH, never cached beside the tables: it is a
        # cumulative sum over the CURRENT Hp, so a value taken once would be a stale
        # field one step later.
        return module.launch_cylindrical_fused_electric_pair(
            fields, arguments["tables"], cylindrical_prefix(fields, "step_D"),
            arguments["imr_rows"], arguments["boundary_codes"],
            arguments["increment_scalars"], arguments["m_class"],
            arguments["zero_rows"], arguments["walls"], arguments["dtdx"],
            arguments["arm"], kernel)

    def separate(fields, grid, pml, arm):
        from meep_gpu.cuda_kernels import complex_pml_kernels  # noqa: PLC0415
        from meep_gpu.cuda_kernels import cylindrical_complex_kernels as certified  # noqa: PLC0415

        certified.step_cylindrical_complex("step_D", fields, arm, grid=grid, pml=pml)
        stepping.fill_symmetry_bc_D(fields)
        stepping.zero_metal_D(fields)
        stepping.fill_folded_far_ghosts_D(fields)
        complex_pml_kernels.update_fused_pml_complex("E", fields, arm, pml=pml)
        return certified

    return {
        "name": "cylindrical_electric",
        "shape": "cylindrical",
        "seam": "D",
        "curl_sub_step": "step_D",
        "constitutive_side": "update_E",
        "seam_constitutive": "E",
        "module": module,
        "specs": CYLINDRICAL_SPECS,
        "build": build_cylindrical_electric,
        "covers": module.covers_cylindrical_fused_electric_pair,
        "resolve": resolve,
        "launch": launch,
        "separate": separate,
        "source": module.cylindrical_fused_electric_pair_source,
        "emitter_module": "meep_gpu.cuda_kernels.cylindrical_fused_electric_pair",
        "emitter_attribute": "cylindrical_fused_electric_pair_source",
        "disjoint": lambda fields, arguments: module.assert_disjoint_bindings(
            fields, arguments["tables"],
            cylindrical_prefix(fields, "step_D"), arguments["imr_rows"]),
        "mutation_labels": ("m2_z_metallic", "m1_z_metallic", "m3_accurate",
                            "m1_z_periodic", "m0_z_metallic"),
        "deposit_labels": ("m2_z_metallic", "m1_z_periodic", "m0_z_metallic"),
        "reduced_labels": ("m2_z_metallic", "m1_z_periodic", "m0_z_metallic"),
    }


def build_complex_electric(spec: Dict[str, Any], value_class: str, rng):
    """``build_complex`` plus the anisotropic material the D seam needs."""
    fields, grid, pml, host = build_complex(spec, value_class, rng)
    anisotropic_epsilon(fields, grid, rng)
    return fields, grid, pml, host


def build_cylindrical_electric(spec: Dict[str, Any], value_class: str, rng):
    """``build_cylindrical`` plus the anisotropic material the D seam needs."""
    fields, grid, pml, host = build_cylindrical(spec, value_class, rng)
    anisotropic_epsilon(fields, grid, rng)
    return fields, grid, pml, host


# ---------------------------------------------------------------------------
# THE FIVE RESIDUAL MAGNETIC FAMILIES, 2026-09-02 -- the board's last buildable cells
# ---------------------------------------------------------------------------
#
# Two are COMPLEX (folded, beta) and ride this harness natively; three are REAL
# (Dcyl m = 0, special_kz, BFAST) and ride it because nothing in the comparison
# machinery is dtype-bound -- the state is compared as raw uint32 words either
# way. The real families' ``covers`` take no licence, so their family dicts wrap
# them to the six-argument shape the legs call with; the arm is resolved and then
# unused on them, which is recorded rather than hidden.
#
# THE TWO FILL-CARRYING FAMILIES DECLARE ALL FIVE PASSES FUSED, so
# ``unfused_passes`` runs NEITHER mirror fill on the subject side -- the launch
# performed them -- while the reference runs the driver's own array passes. That
# is the whole carry claim, measured per complete step. Their per-case CARRY
# FLOOR (leg B of the closure protocol) is :func:`leg_carry_floor`: the same
# shipped kernel re-launched with the fill plan DEGENERATE (near = 0,
# reflect = -1) must DIVERGE from the driver on every folded fixture, or the
# case never exercised the carry and is vacuous.

FOLDED_COMPLEX_SPECS: Tuple[Dict[str, Any], ...] = (
    # The fold axis, both terminations and both full-count parities; walls and
    # an in-plane Bloch phase on the UNFOLDED axes; a two-plane mixed-phase fold
    # for the composed corner. Cells are unequal so an index swap cannot hide.
    {"label": "fold_y_periodic_even", "boundaries": ("metallic", "periodic", "periodic"),
     "symmetry": (("Y", 1),), "k_point": (0.0, 0.0, 0.0), "pml": 2,
     "cell": (9.0, 10.0, 11.0)},
    {"label": "fold_y_periodic_odd", "boundaries": ("metallic", "periodic", "periodic"),
     "symmetry": (("Y", 1),), "k_point": (0.0, 0.0, 0.0), "pml": 2,
     "cell": (9.0, 9.0, 11.0)},
    {"label": "fold_y_odd_phase", "boundaries": ("metallic", "periodic", "periodic"),
     "symmetry": (("Y", -1),), "k_point": (0.0, 0.0, 0.0), "pml": 2,
     "cell": (9.0, 10.0, 11.0)},
    {"label": "fold_y_metallic", "boundaries": ("metallic", "metallic", "periodic"),
     "symmetry": (("Y", 1),), "k_point": (0.0, 0.0, 0.0), "pml": 2,
     "cell": (9.0, 10.0, 11.0)},
    {"label": "fold_x_periodic", "boundaries": ("periodic", "metallic", "periodic"),
     "symmetry": (("X", 1),), "k_point": (0.0, 0.0, 0.0), "pml": 2,
     "cell": (10.0, 9.0, 11.0)},
    # THE COMPOSED CORNER: two folded periodic planes at MIXED phase, so the
    # doubly imaged ghost carries (-1) x (+1) and a dropped factor moves it.
    {"label": "fold_xy_mixed_phase", "boundaries": ("periodic", "periodic", "metallic"),
     "symmetry": (("X", -1), ("Y", 1)), "k_point": (0.0, 0.0, 0.0), "pml": 2,
     "cell": (9.0, 10.0, 11.0)},
    # An in-plane Bloch phase on the PML'd unfolded axis -- the
    # special_kz_2_21_2 / triangular_lattice_oblique shape without the beta.
    {"label": "fold_y_bloch_z", "boundaries": ("metallic", "periodic", "periodic"),
     "symmetry": (("Y", 1),), "k_point": (0.0, 0.0, 0.31), "pml": 2,
     "cell": (9.0, 10.0, 11.0)},
    # Walls beside the fills: zero_metal_B live on x while the y fold fills.
    {"label": "fold_y_wall_x_odd", "boundaries": ("metallic", "periodic", "periodic"),
     "symmetry": (("Y", 1),), "k_point": (0.0, 0.0, 0.0), "pml": 2,
     "cell": (9.0, 9.0, 10.0)},
)

COMPLEX_BETA_SPECS: Tuple[Dict[str, Any], ...] = (
    # The beta corpus splits on the fold and this table serves both sides. Beta
    # runs are 2-D (the analytic z derivative IS the third dimension), so every
    # cell here collapses z.
    {"label": "beta_unfolded_bloch_x", "boundaries": ("periodic",) * 3,
     "symmetry": (), "k_point": (0.9205, 0.0, 0.0), "pml": 2,
     "cell": (12.0, 9.0, 0.0), "beta": -0.3907},
    {"label": "beta_fold_y_periodic_even", "boundaries": ("metallic", "periodic", "periodic"),
     "symmetry": (("Y", 1),), "k_point": (0.0, 0.0, 0.0), "pml": 2,
     "cell": (9.0, 10.0, 0.0), "beta": 0.2},
    {"label": "beta_fold_y_periodic_odd", "boundaries": ("metallic", "periodic", "periodic"),
     "symmetry": (("Y", 1),), "k_point": (0.0, 0.0, 0.0), "pml": 2,
     "cell": (9.0, 9.0, 0.0), "beta": -0.685},
    {"label": "beta_fold_y_odd_phase", "boundaries": ("metallic", "periodic", "periodic"),
     "symmetry": (("Y", -1),), "k_point": (0.0, 0.0, 0.0), "pml": 2,
     "cell": (9.0, 10.0, 0.0), "beta": -0.912},
    # The binary_grating shape: fold Y periodic AND an in-plane kx on the PML'd
    # X axis.
    # The binary_grating shape: fold Y periodic and an in-plane kx, which
    # REQUIRES a periodic x (a k_point and a metallic wall on one axis are
    # mutually exclusive -- grid.py refuses the pair, as MEEP does); the PML
    # rides the periodic x exactly as the corpus row's does.
    {"label": "beta_fold_y_bloch_x", "boundaries": ("periodic", "periodic", "periodic"),
     "symmetry": (("Y", 1),), "k_point": (0.0, 0.0, 0.0), "pml": 2,
     "cell": (10.0, 9.0, 0.0), "beta": -0.685, "in_plane_kx": 0.37},
    {"label": "beta_fold_y_wall_x", "boundaries": ("metallic", "periodic", "periodic"),
     "symmetry": (("Y", 1),), "k_point": (0.0, 0.0, 0.0), "pml": 1,
     "cell": (12.0, 9.0, 0.0), "beta": 0.2},
)

CYLINDRICAL_REAL_SPECS: Tuple[Dict[str, Any], ...] = (
    # m = 0 real Dcyl -- the certified family's whole admission -- at both z
    # terminations and two absorber thicknesses. The r axis is the cylindrical
    # axis on every one, so zero_metal's live wall is z.
    {"label": "m0_z_metallic", "shape": (9, 1, 11), "z_kind": "metallic",
     "absorber": "thick"},
    {"label": "m0_z_periodic", "shape": (9, 1, 11), "z_kind": "periodic",
     "absorber": "thick"},
    {"label": "m0_tall", "shape": (12, 1, 9), "z_kind": "metallic",
     "absorber": "thick"},
    {"label": "m0_thin_absorber", "shape": (11, 1, 10), "z_kind": "metallic",
     "absorber": "thin"},
)

SPECIAL_KZ_SPECS: Tuple[Dict[str, Any], ...] = (
    # Real-storage beta, 2-D. UNFOLDED AND FOLDED SINCE THE 2026-09-02 RESIDUE
    # ROUND: the weld carries the two mirror fills (the real pair's ported
    # ownership inversion), so the table serves both sides the way the
    # complex-beta table does. The corpus's one folded row
    # (``eigsrc_kz_1_real_imag``, fold Y, beta 0.2) is the fold shapes' anchor.
    # NO IN-PLANE k ANYWHERE IN THIS TABLE: the certified real curl refuses a
    # nonzero Bloch k under real storage by name (the wrapped plane carries a
    # phase real storage cannot hold), and the corpus row carries none -- its
    # 0.332 IS the beta.
    {"label": "skz_periodic", "boundaries": ("periodic",) * 3,
     "k_point": (0.0, 0.0, 0.0), "pml": 2,
     "cell": (12.0, 9.0, 0.0), "beta": 0.3321611318837033},
    {"label": "skz_wall_y", "boundaries": ("periodic", "metallic", "periodic"),
     "k_point": (0.0, 0.0, 0.0), "pml": 2, "cell": (10.0, 9.0, 0.0),
     "beta": -0.39},
    {"label": "skz_wall_xy_thin", "boundaries": ("metallic", "metallic", "periodic"),
     "k_point": (0.0, 0.0, 0.0), "pml": 1, "cell": (12.0, 9.0, 0.0),
     "beta": 0.2},
    # THE FOLD SHAPES, mirroring COMPLEX_BETA_SPECS' fold rows under real
    # storage: even and odd stored counts (the far reflect row is n - 3 at an
    # odd full count, which is why the odd fixture exists), an odd mirror
    # phase (+/-1 is exactly representable in real storage), and a wall
    # beside the fills.
    {"label": "skz_fold_y_even", "boundaries": ("metallic", "periodic", "periodic"),
     "symmetry": (("Y", 1),), "k_point": (0.0, 0.0, 0.0), "pml": 2,
     "cell": (9.0, 10.0, 0.0), "beta": 0.2},
    {"label": "skz_fold_y_odd", "boundaries": ("metallic", "periodic", "periodic"),
     "symmetry": (("Y", 1),), "k_point": (0.0, 0.0, 0.0), "pml": 2,
     "cell": (9.0, 9.0, 0.0), "beta": -0.39},
    {"label": "skz_fold_y_odd_phase", "boundaries": ("metallic", "periodic", "periodic"),
     "symmetry": (("Y", -1),), "k_point": (0.0, 0.0, 0.0), "pml": 2,
     "cell": (9.0, 10.0, 0.0), "beta": 0.2},
    {"label": "skz_fold_y_wall_x", "boundaries": ("metallic", "periodic", "periodic"),
     "symmetry": (("Y", 1),), "k_point": (0.0, 0.0, 0.0), "pml": 1,
     "cell": (12.0, 9.0, 0.0), "beta": 0.2},
)

#: THE FOUR IN-PLANE IIR STATES OSCILLATE, AND THE FLOOR SAYS SO BY NAME.
#: Measured on device before this note was written: on a 2-D grid every legal k
#: drives only the z-component pair (``bfast_curl_coefficients`` returned
#: ((0,0),(0,0),(kx,ky)) on the (0.35, 0.21, 0) fixture), so
#: ``f_bfast_{B,D}{x,y}`` advance as ``F -> -F`` (advance = total - 2F with
#: total = 0) -- a PERIOD-2 oscillation that lands back on its seed at any even
#: step count. The final-state movement floor reads that as frozen while the
#: per-step bit-identity watches every intermediate word, so those four are
#: EXCLUDED from the 2-D fixtures' floor with this reason -- never from the
#: comparison -- and the 3-D fixture, where all three k components are live,
#: holds the FULL floor.
_BFAST_2D_FLOOR_EXCLUDES = ("f_bfast_Bx", "f_bfast_By",
                            "f_bfast_Dx", "f_bfast_Dy")
BFAST_SPECS: Tuple[Dict[str, Any], ...] = (
    {"label": "bfast_periodic", "boundaries": ("periodic",) * 3,
     "pml": 2, "cell": (10.0, 9.0, 0.0), "bfast": (0.35, 0.21, 0.0),
     "uniform_floor_excludes": _BFAST_2D_FLOOR_EXCLUDES},
    {"label": "bfast_wall_y", "boundaries": ("periodic", "metallic", "periodic"),
     "pml": 2, "cell": (12.0, 9.0, 0.0), "bfast": (0.35, 0.2, 0.0),
     "uniform_floor_excludes": _BFAST_2D_FLOOR_EXCLUDES},
    {"label": "bfast_wall_xy_thin", "boundaries": ("metallic", "metallic", "periodic"),
     "pml": 1, "cell": (9.0, 12.0, 0.0), "bfast": (0.17, 0.35, 0.0),
     "uniform_floor_excludes": _BFAST_2D_FLOOR_EXCLUDES},
    {"label": "bfast_3d", "boundaries": ("periodic", "metallic", "periodic"),
     "pml": 2, "cell": (9.0, 10.0, 8.0), "bfast": (0.3, 0.2, 0.15)},
)


def _seed_one_real(shape, value_class: str, rng, name: str = "seed") -> np.ndarray:
    """One float32 volume in a value class, as a host array."""
    if value_class == "uniform":
        return np.ascontiguousarray(
            rng.uniform(-1.0, 1.0, size=shape).astype(np.float32))
    if value_class == "subnormal_band":
        return np.ascontiguousarray(np.asarray(
            subnormal_band_hosts((name,), tuple(shape), rng)[name],
            dtype=np.float32))
    raise ValueError(f"value class {value_class!r} is not one of {VALUE_CLASSES}")


def _seed_real_state(fields, grid, value_class: str, rng,
                     names: Sequence[str] = None) -> Dict[str, np.ndarray]:
    """Seed every REAL volume a complete step touches; return the host words.

    The real twin of ``_seed_complex_state``: same movement-floor rationale (the
    auxiliaries start nonzero so a mis-indexed coefficient shows from launch
    one), one float32 volume per name.
    """
    xp = grid.xp
    shape = tuple(int(n) for n in grid.shape)
    words: Dict[str, np.ndarray] = {}
    for name in (STATE_NAMES if names is None else names):
        host = _seed_one_real(shape, value_class, rng, name)
        getattr(fields, name)[...] = xp.asarray(host)
        words[name] = host.view(np.float32).ravel().copy()
    return words


def build_folded_complex_pair(spec: Dict[str, Any], value_class: str, rng):
    """One seeded FOLDED (or folded-beta) Cartesian complex engine."""
    from meep_gpu.fields import Fields  # noqa: PLC0415
    from meep_gpu.grid import Grid, Mirror  # noqa: PLC0415
    from meep_gpu.pml import PML  # noqa: PLC0415

    kwargs: Dict[str, Any] = {}
    if spec.get("beta"):
        kwargs["beta"] = float(spec["beta"])
    if float(spec["cell"][2]) == 0.0:
        kwargs["dimensions"] = 2   # a collapsed z axis is a 2-D declaration
    k_point = list(spec.get("k_point", (0.0, 0.0, 0.0)))
    if spec.get("in_plane_kx"):
        k_point[0] = float(spec["in_plane_kx"])
    grid = Grid(resolution=1.0, cell_size=tuple(spec["cell"]),
                boundaries=tuple(spec["boundaries"]),
                symmetry=tuple(Mirror(axis, phase)
                               for axis, phase in spec.get("symmetry", ())),
                k_point=tuple(k_point), xp=cp, courant=0.5, **kwargs)
    fields = Fields(grid=grid, force_complex_fields=True)
    fields.enable_field_storage()
    fields.enable_pml_storage()
    # THE FOLDED AXIS TAKES NO ABSORBER (its stored window ends at the mirror);
    # the unfolded axes take the spec's thickness where they can hold it.
    thickness = tuple((0, 0) if grid.is_mirrored(axis)
                      else (0, 0) if int(grid.shape[axis]) < 2 * spec["pml"] + 2
                      else (spec["pml"], spec["pml"])
                      for axis in range(3))
    pml = PML(grid=grid, thickness=thickness)
    host = _seed_complex_state(fields, grid, value_class, rng)
    return fields, grid, pml, host


def build_cylindrical_real_pair(spec: Dict[str, Any], value_class: str, rng):
    """One seeded REAL m = 0 Dcyl engine."""
    from meep_gpu.fields import Fields  # noqa: PLC0415
    from meep_gpu.grid import Grid  # noqa: PLC0415
    from meep_gpu.pml import PML  # noqa: PLC0415

    shape = tuple(int(n) for n in spec["shape"])
    grid = Grid(resolution=1.0, cell_size=(float(shape[0]), 0.0, float(shape[2])),
                cylindrical=True, m=0, boundaries={"z": spec["z_kind"]},
                courant=0.5, xp=cp)
    fields = Fields(grid=grid)
    fields.enable_field_storage()
    fields.enable_pml_storage()
    layer = 1 if spec.get("absorber") == "thin" else 2
    pml = PML(grid=grid, thickness={"x": (0, layer), "z": layer})
    host = _seed_real_state(fields, grid, value_class, rng)
    return fields, grid, pml, host


def build_special_kz_pair(spec: Dict[str, Any], value_class: str, rng):
    """One seeded REAL 2-D beta engine, folded where the spec declares a mirror.

    The fold handling is ``build_folded_complex_pair``'s, restated under real
    storage: the folded axis takes NO absorber (its stored window ends at the
    mirror), the unfolded axes take the spec's thickness where they can hold it.
    An unfolded spec builds exactly what this function built before the fold
    rows landed.
    """
    from meep_gpu.fields import Fields  # noqa: PLC0415
    from meep_gpu.grid import Grid, Mirror  # noqa: PLC0415
    from meep_gpu.pml import PML  # noqa: PLC0415

    grid = Grid(resolution=1.0, cell_size=tuple(spec["cell"]),
                boundaries=tuple(spec["boundaries"]),
                symmetry=tuple(Mirror(axis, phase)
                               for axis, phase in spec.get("symmetry", ())),
                k_point=tuple(spec.get("k_point", (0.0, 0.0, 0.0))),
                beta=float(spec["beta"]), xp=cp, courant=0.5, dimensions=2)
    fields = Fields(grid=grid)
    fields.enable_field_storage()
    fields.enable_pml_storage()
    thickness = tuple((0, 0) if grid.is_mirrored(axis)
                      else (0, 0) if int(grid.shape[axis]) < 2 * spec["pml"] + 2
                      else (spec["pml"], spec["pml"]) for axis in range(3))
    pml = PML(grid=grid, thickness=thickness)
    host = _seed_real_state(fields, grid, value_class, rng)
    return fields, grid, pml, host


def build_bfast_pair(spec: Dict[str, Any], value_class: str, rng):
    """One seeded REAL BFAST engine, its six IIR state volumes in the state."""
    from meep_gpu.fields import Fields  # noqa: PLC0415
    from meep_gpu.grid import Grid  # noqa: PLC0415
    from meep_gpu.pml import PML  # noqa: PLC0415

    kwargs = {"dimensions": 2} if float(spec["cell"][2]) == 0.0 else {}
    grid = Grid(resolution=1.0, cell_size=tuple(spec["cell"]),
                boundaries=tuple(spec["boundaries"]),
                bfast_scaled_k=tuple(spec["bfast"]), xp=cp, courant=0.5,
                **kwargs)
    fields = Fields(grid=grid)
    fields.enable_field_storage()
    fields.enable_pml_storage()
    thickness = tuple((0, 0) if int(grid.shape[axis]) < 2 * spec["pml"] + 2
                      else (spec["pml"], spec["pml"]) for axis in range(3))
    pml = PML(grid=grid, thickness=thickness)
    host = _seed_real_state(fields, grid, value_class, rng,
                            names=BFAST_STATE_NAMES)
    return fields, grid, pml, host


#: The BFAST family's comparison set: the 24 stored volumes plus the six IIR
#: state volumes its two curls advance -- a weld that corrupted one would reach
#: B on the next step through a comparison that never looked.
BFAST_STATE_NAMES: Tuple[str, ...] = tuple(STATE_NAMES) + (
    "f_bfast_Bx", "f_bfast_By", "f_bfast_Bz",
    "f_bfast_Dx", "f_bfast_Dy", "f_bfast_Dz")


# THE ELECTRIC-TWIN BUILDERS, 2026-09-02: each is its magnetic sibling's builder
# plus the anisotropic material the D seam needs (see ``anisotropic_epsilon`` --
# on a vacuum fixture every inverse-permittivity mutation is bit-identical, the
# vacuity that function exists to refuse).

def build_folded_complex_electric(spec: Dict[str, Any], value_class: str, rng):
    """``build_folded_complex_pair`` plus the anisotropic material."""
    fields, grid, pml, host = build_folded_complex_pair(spec, value_class, rng)
    anisotropic_epsilon(fields, grid, rng)
    return fields, grid, pml, host


def build_cylindrical_real_electric(spec: Dict[str, Any], value_class: str, rng):
    """``build_cylindrical_real_pair`` plus the anisotropic material."""
    fields, grid, pml, host = build_cylindrical_real_pair(spec, value_class, rng)
    anisotropic_epsilon(fields, grid, rng)
    return fields, grid, pml, host


def build_special_kz_electric(spec: Dict[str, Any], value_class: str, rng):
    """``build_special_kz_pair`` plus the anisotropic material."""
    fields, grid, pml, host = build_special_kz_pair(spec, value_class, rng)
    anisotropic_epsilon(fields, grid, rng)
    return fields, grid, pml, host


def build_bfast_electric(spec: Dict[str, Any], value_class: str, rng):
    """``build_bfast_pair`` plus the anisotropic material."""
    fields, grid, pml, host = build_bfast_pair(spec, value_class, rng)
    anisotropic_epsilon(fields, grid, rng)
    return fields, grid, pml, host


def _folded_refusal_cases(record) -> None:
    """The folded weld's own admission/refusal table, each a REAL configuration."""
    from meep_gpu.fields import Fields  # noqa: PLC0415
    from meep_gpu.grid import Grid, Mirror  # noqa: PLC0415
    from meep_gpu.pml import PML  # noqa: PLC0415

    def build(symmetry=(Mirror("Y", 1),), complex_storage=True, beta=None,
              k_point=(0.0, 0.0, 0.0)):
        kwargs = {"beta": beta, "dimensions": 2} if beta else {}
        grid = Grid(resolution=1.0, cell_size=(9.0, 10.0, 11.0 if not beta else 0.0),
                    boundaries=("metallic", "periodic", "periodic"),
                    symmetry=tuple(symmetry), k_point=k_point, xp=cp, courant=0.5,
                    **kwargs)
        fields = Fields(grid=grid, force_complex_fields=complex_storage)
        fields.enable_field_storage()
        fields.enable_pml_storage()
        thickness = tuple((0, 0) if grid.is_mirrored(axis)
                          else (0, 0) if int(grid.shape[axis]) < 6
                          else (2, 2)
                          for axis in range(3))
        return fields, PML(grid=grid, thickness=thickness), grid

    fields, pml, grid = build()
    record("folded_complex_admitted", "covered", fields, pml, grid, ())
    record("undeclared_source_set", "refused", fields, pml, grid, None)
    fields, pml, grid = build(symmetry=())
    record("unfolded_belongs_to_the_plain_pair", "refused", fields, pml, grid, ())
    fields, pml, grid = build(complex_storage=False)
    record("real_storage", "refused", fields, pml, grid, ())
    fields, pml, grid = build(beta=0.2)
    record("beta_belongs_to_the_beta_weld", "refused", fields, pml, grid, ())
    # A Bloch phase ON the folded axis is unbuildable as a real Grid -- the
    # constructor itself refuses it (grid.py _resolve_bloch) -- so the
    # predicate's own clause for it cannot be recorded through one; it exists
    # for grid-shaped objects that lie and is pinned by the family's laptop
    # tests through proxies. A phase on an UNFOLDED axis is a real
    # configuration and must stay ADMITTED:
    fields, pml, grid = build(k_point=(0.0, 0.0, 0.31))
    record("bloch_phase_on_an_unfolded_axis_admitted", "covered",
           fields, pml, grid, ())


def _beta_refusal_cases(record) -> None:
    """The beta weld's own admission/refusal table."""
    from meep_gpu.fields import Fields  # noqa: PLC0415
    from meep_gpu.grid import Grid, Mirror  # noqa: PLC0415
    from meep_gpu.pml import PML  # noqa: PLC0415

    def build(symmetry=(Mirror("Y", 1),), complex_storage=True, beta=0.2):
        kwargs = {"beta": beta} if beta else {}
        grid = Grid(resolution=1.0, cell_size=(9.0, 10.0, 0.0),
                    boundaries=("metallic", "periodic", "periodic"),
                    symmetry=tuple(symmetry), xp=cp, courant=0.5, dimensions=2,
                    **kwargs)
        fields = Fields(grid=grid, force_complex_fields=complex_storage)
        fields.enable_field_storage()
        fields.enable_pml_storage()
        thickness = tuple((0, 0) if grid.is_mirrored(axis)
                          else (0, 0) if int(grid.shape[axis]) < 6 else (2, 2)
                          for axis in range(3))
        return fields, PML(grid=grid, thickness=thickness), grid

    fields, pml, grid = build()
    record("folded_beta_admitted", "covered", fields, pml, grid, ())
    fields, pml, grid = build(symmetry=())
    record("unfolded_beta_admitted", "covered", fields, pml, grid, ())
    record("undeclared_source_set", "refused", fields, pml, grid, None)
    fields, pml, grid = build(beta=None)
    record("no_beta_belongs_to_the_folded_weld", "refused", fields, pml, grid, ())
    fields, pml, grid = build(symmetry=(), beta=None)
    record("no_beta_unfolded_belongs_to_the_plain_pair", "refused",
           fields, pml, grid, ())


def _cylindrical_real_refusal_cases(record) -> None:
    """The Dcyl real weld's own admission/refusal table."""
    from meep_gpu.fields import Fields  # noqa: PLC0415
    from meep_gpu.grid import Grid  # noqa: PLC0415
    from meep_gpu.pml import PML  # noqa: PLC0415

    def build(complex_storage=False, cylindrical=True):
        if cylindrical:
            grid = Grid(resolution=1.0, cell_size=(9.0, 0.0, 11.0),
                        cylindrical=True, m=0, boundaries={"z": "metallic"},
                        courant=0.5, xp=cp)
            thickness = {"x": (0, 2), "z": 2}
        else:
            grid = Grid(resolution=1.0, cell_size=(9.0, 10.0, 11.0),
                        boundaries=("periodic",) * 3, courant=0.5, xp=cp)
            thickness = 2
        fields = Fields(grid=grid, force_complex_fields=complex_storage)
        fields.enable_field_storage()
        fields.enable_pml_storage()
        return fields, PML(grid=grid, thickness=thickness), grid

    fields, pml, grid = build()
    record("m0_real_dcyl_admitted", "covered", fields, pml, grid, ())
    record("undeclared_source_set", "refused", fields, pml, grid, None)
    fields, pml, grid = build(complex_storage=True)
    record("complex_storage_belongs_to_the_complex_dcyl_weld", "refused",
           fields, pml, grid, ())
    fields, pml, grid = build(cylindrical=False)
    record("cartesian_belongs_to_the_real_pair", "refused", fields, pml, grid, ())


def _special_kz_refusal_cases(record) -> None:
    """The special_kz weld's own admission/refusal table.

    THE FOLD IS ADMITTED SINCE 2026-09-02: the weld carries the two mirror
    fills (the real pair's ported ownership inversion), so the fold case that
    this table used to hold as the weld's own refusal is now its admission --
    recorded through a REAL folded engine so a predicate that silently kept
    the old refusal fails here rather than serving one row fewer."""
    from meep_gpu.fields import Fields  # noqa: PLC0415
    from meep_gpu.grid import Grid, Mirror  # noqa: PLC0415
    from meep_gpu.pml import PML  # noqa: PLC0415

    def build(symmetry=(), complex_storage=False, beta=0.33):
        kwargs = {"beta": beta} if beta else {}
        grid = Grid(resolution=1.0, cell_size=(12.0, 9.0, 0.0),
                    boundaries=("metallic", "periodic", "periodic")
                    if symmetry else ("periodic",) * 3,
                    symmetry=tuple(symmetry), xp=cp, courant=0.5, dimensions=2,
                    **kwargs)
        fields = Fields(grid=grid, force_complex_fields=complex_storage)
        fields.enable_field_storage()
        fields.enable_pml_storage()
        thickness = tuple((0, 0) if grid.is_mirrored(axis)
                          else (0, 0) if int(grid.shape[axis]) < 6 else (2, 2)
                          for axis in range(3))
        return fields, PML(grid=grid, thickness=thickness), grid

    fields, pml, grid = build()
    record("real_beta_admitted", "covered", fields, pml, grid, ())
    record("undeclared_source_set", "refused", fields, pml, grid, None)
    fields, pml, grid = build(symmetry=(Mirror("Y", 1),))
    record("folded_admitted_since_the_fill_carry", "covered", fields, pml, grid, ())
    fields, pml, grid = build(beta=None)
    record("no_beta_belongs_to_the_real_pair", "refused", fields, pml, grid, ())
    fields, pml, grid = build(complex_storage=True)
    record("complex_storage_belongs_to_the_beta_weld", "refused",
           fields, pml, grid, ())


def _bfast_refusal_cases(record) -> None:
    """The BFAST weld's own admission/refusal table."""
    from meep_gpu.fields import Fields  # noqa: PLC0415
    from meep_gpu.grid import Grid, Mirror  # noqa: PLC0415
    from meep_gpu.pml import PML  # noqa: PLC0415

    def build(symmetry=(), bfast=(0.35, 0.0, 0.0)):
        kwargs = {"bfast_scaled_k": bfast} if bfast else {}
        grid = Grid(resolution=1.0, cell_size=(10.0, 9.0, 0.0),
                    boundaries=("metallic", "periodic", "periodic")
                    if symmetry else ("periodic",) * 3,
                    symmetry=tuple(symmetry), xp=cp, courant=0.5, dimensions=2,
                    **kwargs)
        fields = Fields(grid=grid)
        fields.enable_field_storage()
        fields.enable_pml_storage()
        thickness = tuple((0, 0) if grid.is_mirrored(axis)
                          else (0, 0) if int(grid.shape[axis]) < 6 else (2, 2)
                          for axis in range(3))
        return fields, PML(grid=grid, thickness=thickness), grid

    fields, pml, grid = build()
    record("bfast_admitted", "covered", fields, pml, grid, ())
    record("undeclared_source_set", "refused", fields, pml, grid, None)
    fields, pml, grid = build(bfast=None)
    record("no_bfast_belongs_to_the_real_pair", "refused", fields, pml, grid, ())
    fields, pml, grid = build(symmetry=(Mirror("Y", 1),))
    record("folded_refused", "refused", fields, pml, grid, ())


def _real_covers(covers):
    """A real-storage predicate lifted to the six-argument shape the legs call.

    The licence and the policy are ACCEPTED AND UNUSED: these families bind no
    expansion arm (their arithmetic is real float32), and a wrapper that refused
    the extra arguments would fork every leg on the family kind.
    """
    def ask(fields, pml, grid, sources=(), license=None, policy=None):  # noqa: ARG001
        return covers(fields, pml, grid, sources)
    return ask


def _folded_complex_family() -> Dict[str, Any]:
    """The FOLDED-COMPLEX B/H weld -- the ownership-inversion port to cf storage."""
    from meep_gpu.cuda_kernels import complex_fill_carry  # noqa: PLC0415
    from meep_gpu.cuda_kernels import (  # noqa: PLC0415
        complex_folded_fused_magnetic_pair as module,
    )
    from meep_gpu.cuda_kernels import in_seam_coverage  # noqa: PLC0415

    def resolve(fields, grid, pml, arm):
        from meep_gpu.cuda_kernels import complex_folded_kernels  # noqa: PLC0415
        from meep_gpu.cuda_kernels import complex_pml_kernels  # noqa: PLC0415

        codes, refusal = complex_folded_kernels.folded_complex_boundary_codes(grid)
        if refusal is not None:
            raise SystemExit(f"the fixture has no folded boundary codes: {refusal}")
        flags, values = complex_pml_kernels.bloch_phase_arguments(grid, False)
        return {"tables": module.complex_folded_fused_magnetic_pair_tables(pml),
                "boundary_codes": codes,
                "phase_flags": flags, "phase_values": values,
                "walls": in_seam_coverage.zero_metal_axes(grid),
                "fills": complex_fill_carry.fills_plan(grid),
                "dtdx": float(grid.dt / grid.dx), "arm": arm}

    def launch(fields, arguments, kernel=None):
        return module.launch_complex_folded_fused_magnetic_pair(
            fields, arguments["tables"], arguments["boundary_codes"],
            arguments["phase_flags"], arguments["phase_values"],
            arguments["walls"], arguments["fills"], arguments["dtdx"],
            arguments["arm"], kernel)

    def separate(fields, grid, pml, arm):
        """The certified route: the folded curl single, the driver's own array
        fills and wall pass, the certified complex constitutive."""
        from meep_gpu.cuda_kernels import complex_folded_kernels  # noqa: PLC0415
        from meep_gpu.cuda_kernels import complex_pml_kernels  # noqa: PLC0415

        complex_folded_kernels.step_folded_complex(
            "step_B", fields, arm, grid=grid, pml=pml)
        stepping.fill_symmetry_bc_B(fields)
        stepping.zero_metal_B(fields)
        stepping.fill_folded_far_ghosts_B(fields)
        complex_pml_kernels.update_fused_pml_complex("H", fields, arm, pml=pml)

    return {
        "name": "complex_folded",
        "shape": "cartesian",
        "seam": "B",
        "curl_sub_step": "step_B",
        "constitutive_side": "update_H",
        "seam_constitutive": "H",
        "module": module,
        "specs": FOLDED_COMPLEX_SPECS,
        "build": build_folded_complex_pair,
        "covers": module.covers_complex_folded_fused_magnetic_pair,
        "resolve": resolve,
        "launch": launch,
        "separate": separate,
        "source": module.complex_folded_fused_magnetic_pair_source,
        "emitter_module":
            "meep_gpu.cuda_kernels.complex_folded_fused_magnetic_pair",
        "emitter_attribute": "complex_folded_fused_magnetic_pair_source",
        "disjoint": lambda fields, arguments: module.assert_disjoint_bindings(
            fields, arguments["tables"]),
        # ALL FIVE PASSES ARE FUSED: the two mirror fills are carried by the
        # ownership inversion, so the subject runs NO array pass inside the seam.
        "fused_passes": ("step_B", "fill_symmetry_bc_B", "zero_metal_B",
                         "fill_folded_far_ghosts_B", "update_H"),
        "carry_floor": True,
        "refusal_cases": _folded_refusal_cases,
        "certified_curl": (lambda arm, variant=None: __import__(
            "meep_gpu.cuda_kernels.complex_folded_kernels",
            fromlist=["folded_source"]).folded_source("step_B", arm)),
        "mutation_labels": ("fold_y_periodic_even", "fold_y_periodic_odd",
                            "fold_y_odd_phase", "fold_xy_mixed_phase",
                            "fold_y_wall_x_odd", "fold_x_periodic"),
        "deposit_labels": ("fold_y_periodic_even", "fold_y_metallic"),
        "reduced_labels": ("fold_y_periodic_even", "fold_xy_mixed_phase"),
    }


def _complex_beta_family() -> Dict[str, Any]:
    """The COMPLEX-BETA B/H weld -- the folded weld's carry over the beta curl."""
    from meep_gpu.cuda_kernels import complex_fill_carry  # noqa: PLC0415
    from meep_gpu.cuda_kernels import (  # noqa: PLC0415
        complex_beta_fused_magnetic_pair as module,
    )
    from meep_gpu.cuda_kernels import in_seam_coverage  # noqa: PLC0415

    def resolve(fields, grid, pml, arm):
        from meep_gpu.cuda_kernels import complex_folded_kernels  # noqa: PLC0415
        from meep_gpu.cuda_kernels import complex_pml_kernels  # noqa: PLC0415

        codes, refusal = complex_folded_kernels.folded_complex_boundary_codes(grid)
        if refusal is not None:
            raise SystemExit(f"the fixture has no fold-aware boundary codes: "
                             f"{refusal}")
        flags, values = complex_pml_kernels.bloch_phase_arguments(grid, False)
        return {"tables": module.complex_beta_fused_magnetic_pair_tables(pml),
                "boundary_codes": codes,
                "phase_flags": flags, "phase_values": values,
                "beta_words": module.beta_coefficients(grid),
                "walls": in_seam_coverage.zero_metal_axes(grid),
                "fills": complex_fill_carry.fills_plan(grid),
                "dtdx": float(grid.dt / grid.dx), "arm": arm}

    def launch(fields, arguments, kernel=None):
        return module.launch_complex_beta_fused_magnetic_pair(
            fields, arguments["tables"], arguments["boundary_codes"],
            arguments["phase_flags"], arguments["phase_values"],
            arguments["beta_words"], arguments["walls"], arguments["fills"],
            arguments["dtdx"], arguments["arm"], kernel)

    def separate(fields, grid, pml, arm):
        from meep_gpu.cuda_kernels import complex_beta_kernels  # noqa: PLC0415
        from meep_gpu.cuda_kernels import complex_pml_kernels  # noqa: PLC0415

        complex_beta_kernels.step_complex_beta(
            "step_B", fields, arm, grid=grid, pml=pml)
        stepping.fill_symmetry_bc_B(fields)
        stepping.zero_metal_B(fields)
        stepping.fill_folded_far_ghosts_B(fields)
        complex_pml_kernels.update_fused_pml_complex("H", fields, arm, pml=pml)

    return {
        "name": "complex_beta",
        "shape": "cartesian",
        "seam": "B",
        "curl_sub_step": "step_B",
        "constitutive_side": "update_H",
        "seam_constitutive": "H",
        "module": module,
        "specs": COMPLEX_BETA_SPECS,
        "build": build_folded_complex_pair,
        "covers": module.covers_complex_beta_fused_magnetic_pair,
        "resolve": resolve,
        "launch": launch,
        "separate": separate,
        "source": module.complex_beta_fused_magnetic_pair_source,
        "emitter_module": "meep_gpu.cuda_kernels.complex_beta_fused_magnetic_pair",
        "emitter_attribute": "complex_beta_fused_magnetic_pair_source",
        "disjoint": lambda fields, arguments: module.assert_disjoint_bindings(
            fields, arguments["tables"]),
        "fused_passes": ("step_B", "fill_symmetry_bc_B", "zero_metal_B",
                         "fill_folded_far_ghosts_B", "update_H"),
        "carry_floor": True,
        "refusal_cases": _beta_refusal_cases,
        "certified_curl": (lambda arm, variant=None: __import__(
            "meep_gpu.cuda_kernels.complex_beta_kernels",
            fromlist=["beta_source"]).beta_source("step_B", arm)),
        "mutation_labels": ("beta_fold_y_periodic_even",
                            "beta_fold_y_periodic_odd", "beta_fold_y_odd_phase",
                            "beta_unfolded_bloch_x", "beta_fold_y_wall_x"),
        # DEPOSIT LEGS SINCE THE 2026-09-02 FLAG FLIP (fusion-residue audit
        # §1.3): CARRIES_DEPOSIT_REPAIR went True after the "EigenModeSource
        # publishes no point index" premise was re-measured FALSE of the live
        # tree (the lift synthesises VolumeSource equivalent-current sheets and
        # every source class publishes _point_ix/_point_iy/_point_iz at setup).
        # The legs run on a FOLDED fixture (where the repair must image the
        # deposit through the fill closure) and an unfolded one, with the
        # unrepaired null required to DIVERGE on both; the SHEET deposit is the
        # premise itself, carried as index ARRAYS with per-cell amplitudes the
        # way the lifted eigenmode sheets publish theirs.
        "deposit_labels": ("beta_fold_y_periodic_even", "beta_unfolded_bloch_x"),
        "deposit_sheet": True,
        "source_refusal_labels": (),
        "reduced_labels": ("beta_fold_y_periodic_even", "beta_unfolded_bloch_x"),
    }


def _cylindrical_real_family() -> Dict[str, Any]:
    """The REAL m = 0 Dcyl B/H weld -- the certified STORE followed at the axis."""
    from meep_gpu.cuda_kernels import (  # noqa: PLC0415
        cylindrical_real_fused_magnetic_pair as module,
    )
    from meep_gpu.cuda_kernels import in_seam_coverage  # noqa: PLC0415

    def resolve(fields, grid, pml, arm):  # noqa: ARG001 - real family, no arm
        from meep_gpu.cuda_kernels import coverage as cov  # noqa: PLC0415
        from meep_gpu.cuda_kernels import cylindrical_coverage  # noqa: PLC0415

        codes = cylindrical_coverage.cylindrical_boundary_codes(
            tuple(cov.real_pml_boundary_kinds(grid)))
        return {"tables": module.cylindrical_real_fused_magnetic_pair_tables(pml),
                "boundary_codes": tuple(np.int32(code) for code in codes),
                "walls": in_seam_coverage.zero_metal_axes(grid),
                "dtdx": float(grid.dt / grid.dx)}

    def launch(fields, arguments, kernel=None):
        from meep_gpu.cuda_kernels.cylindrical_prefix import (  # noqa: PLC0415
            cylindrical_prefix,
        )

        return module.launch_cylindrical_real_fused_magnetic_pair(
            fields, arguments["tables"],
            cylindrical_prefix(fields, "step_B",
                               scratch=getattr(fields, "scratch", None)),
            arguments["boundary_codes"], arguments["walls"],
            arguments["dtdx"], kernel)

    def separate(fields, grid, pml, arm):  # noqa: ARG001
        from meep_gpu.cuda_kernels import constitutive_kernels  # noqa: PLC0415
        from meep_gpu.cuda_kernels import cylindrical_kernels  # noqa: PLC0415

        stepped = cylindrical_kernels.step_cylindrical(fields, pml, "step_B")
        if not stepped:
            raise SystemExit(
                "the certified Dcyl curl refused the separate route's fixture; "
                "a silent False here would masquerade as a weld divergence")
        stepping.zero_metal_B(fields)
        constitutive_kernels.update_fused_pml_real("H", fields, pml=pml)

    return {
        "name": "cylindrical_real",
        "shape": "cylindrical_real",
        "seam": "B",
        "curl_sub_step": "step_B",
        "constitutive_side": "update_H",
        "seam_constitutive": "H",
        "module": module,
        "specs": CYLINDRICAL_REAL_SPECS,
        "build": build_cylindrical_real_pair,
        "covers": _real_covers(module.covers_cylindrical_real_fused_magnetic_pair),
        "resolve": resolve,
        "launch": launch,
        "separate": separate,
        "source": lambda arm=None: module.cylindrical_real_fused_magnetic_pair_source(),
        "emitter_module":
            "meep_gpu.cuda_kernels.cylindrical_real_fused_magnetic_pair",
        "emitter_attribute": "cylindrical_real_fused_magnetic_pair_source",
        "disjoint": lambda fields, arguments: True,
        "licensed": False,
        "lift_arms": ("REAL",),
        "lift_arithmetic_markers": ("= dtdx * (", "shift_up(", "shift_dn("),
        "memo_key": True,
        "refusal_cases": _cylindrical_real_refusal_cases,
        "certified_curl": (lambda arm, variant=None: __import__(
            "meep_gpu.cuda_kernels.cylindrical_kernels",
            fromlist=["_cyl_step_B_pml_real_kernel_code"])
            ._cyl_step_B_pml_real_kernel_code),
        "certified_constitutive": (lambda arm, variant=None: __import__(
            "meep_gpu.cuda_kernels.constitutive_kernels",
            fromlist=["_update_H_pml_real_kernel_code"])
            ._update_H_pml_real_kernel_code),
        "mutation_labels": ("m0_z_metallic", "m0_z_periodic", "m0_thin_absorber"),
        "deposit_labels": (),
        "source_refusal_labels": ("m0_z_metallic",),
        "reduced_labels": ("m0_z_metallic", "m0_z_periodic"),
    }


def _special_kz_family() -> Dict[str, Any]:
    """The REAL special_kz B/H weld."""
    from meep_gpu.cuda_kernels import (  # noqa: PLC0415
        special_kz_fused_magnetic_pair as module,
    )
    from meep_gpu.cuda_kernels import in_seam_coverage  # noqa: PLC0415

    def resolve(fields, grid, pml, arm):  # noqa: ARG001 - real family, no arm
        from meep_gpu.cuda_kernels import coverage as cov  # noqa: PLC0415

        codes, refusal = cov.real_curl_boundary_codes(grid)
        if refusal is not None:
            raise SystemExit(f"the fixture has no boundary codes: {refusal}")
        return {"tables": module.special_kz_fused_magnetic_pair_tables(pml),
                "boundary_codes": tuple(np.int32(code) for code in codes),
                "walls": in_seam_coverage.zero_metal_axes(grid),
                "beta": module.beta_scalars(grid),
                "fills": module.special_kz_fused_magnetic_pair_fills(grid),
                "dtdx": float(grid.dt / grid.dx)}

    def launch(fields, arguments, kernel=None):
        return module.launch_special_kz_fused_magnetic_pair(
            fields, arguments["tables"], arguments["boundary_codes"],
            arguments["walls"], arguments["beta"], arguments["fills"],
            arguments["dtdx"], kernel)

    def separate(fields, grid, pml, arm):  # noqa: ARG001
        from meep_gpu.cuda_kernels import constitutive_kernels  # noqa: PLC0415
        from meep_gpu.cuda_kernels import special_kz_curl  # noqa: PLC0415
        from meep_gpu.cuda_kernels import step_curl_kernels  # noqa: PLC0415
        from meep_gpu.cuda_kernels import coverage as cov  # noqa: PLC0415

        codes, refusal = cov.real_curl_boundary_codes(grid)
        assert refusal is None, refusal
        special_kz_curl.step_special_kz(
            "step_B", fields, step_curl_kernels.real_pml_curl_tables(pml, True),
            tuple(np.int32(code) for code in codes), float(grid.dt / grid.dx),
            module.beta_scalars(grid))
        stepping.fill_symmetry_bc_B(fields)
        stepping.zero_metal_B(fields)
        stepping.fill_folded_far_ghosts_B(fields)
        constitutive_kernels.update_fused_pml_real("H", fields, pml=pml)

    def separate_without_fills(fields, grid, pml, arm):  # noqa: ARG001
        from meep_gpu.cuda_kernels import constitutive_kernels  # noqa: PLC0415
        from meep_gpu.cuda_kernels import special_kz_curl  # noqa: PLC0415
        from meep_gpu.cuda_kernels import step_curl_kernels  # noqa: PLC0415
        from meep_gpu.cuda_kernels import coverage as cov  # noqa: PLC0415

        codes, refusal = cov.real_curl_boundary_codes(grid)
        assert refusal is None, refusal
        special_kz_curl.step_special_kz(
            "step_B", fields, step_curl_kernels.real_pml_curl_tables(pml, True),
            tuple(np.int32(code) for code in codes), float(grid.dt / grid.dx),
            module.beta_scalars(grid))
        stepping.zero_metal_B(fields)
        constitutive_kernels.update_fused_pml_real("H", fields, pml=pml)

    return {
        "name": "special_kz",
        "shape": "real_beta",
        "seam": "B",
        "curl_sub_step": "step_B",
        "constitutive_side": "update_H",
        "seam_constitutive": "H",
        "module": module,
        "specs": SPECIAL_KZ_SPECS,
        "build": build_special_kz_pair,
        "covers": _real_covers(module.covers_special_kz_fused_magnetic_pair),
        "resolve": resolve,
        "launch": launch,
        "separate": separate,
        "separate_without_fills": separate_without_fills,
        "source": lambda arm=None: module.special_kz_fused_magnetic_pair_source(),
        "emitter_module": "meep_gpu.cuda_kernels.special_kz_fused_magnetic_pair",
        "emitter_attribute": "special_kz_fused_magnetic_pair_source",
        "disjoint": lambda fields, arguments: module.assert_disjoint_bindings(
            fields, arguments["tables"]),
        "licensed": False,
        "lift_arms": ("REAL",),
        "lift_arithmetic_markers": ("= dtdx * (", "shift_up(", "shift_dn("),
        # ALL FIVE PASSES SINCE 2026-09-02: the weld ported the real pair's fill
        # carry, so the two mirror fills are fused rather than refused, and the
        # carry floor is the armed closure on every folded fixture.
        "fused_passes": ("step_B", "fill_symmetry_bc_B", "zero_metal_B",
                         "fill_folded_far_ghosts_B", "update_H"),
        "carry_floor": True,
        "refusal_cases": _special_kz_refusal_cases,
        "certified_curl": (lambda arm, variant=None: __import__(
            "meep_gpu.cuda_kernels.special_kz_curl",
            fromlist=["kernel_source"]).kernel_source("step_B_special_kz_real")),
        "certified_constitutive": (lambda arm, variant=None: __import__(
            "meep_gpu.cuda_kernels.constitutive_kernels",
            fromlist=["_update_H_pml_real_kernel_code"])
            ._update_H_pml_real_kernel_code),
        "mutation_labels": ("skz_periodic", "skz_wall_y", "skz_wall_xy_thin",
                            "skz_fold_y_even", "skz_fold_y_odd"),
        # DEPOSIT LEGS SINCE THE 2026-09-02 FLAG FLIP (fusion-residue audit
        # §1.3): the "publishes no point index" premise was re-measured FALSE
        # of the live tree, so the bracket carries the seam and the legs prove
        # it -- one unfolded fixture, one FOLDED (the repair must image the
        # deposit through the fill closure), the sheet variant on both.
        "deposit_labels": ("skz_wall_y", "skz_fold_y_even"),
        "deposit_sheet": True,
        "source_refusal_labels": (),
        "reduced_labels": ("skz_periodic", "skz_wall_y", "skz_fold_y_even"),
    }


def _bfast_family() -> Dict[str, Any]:
    """The REAL BFAST B/H weld."""
    from meep_gpu.cuda_kernels import (  # noqa: PLC0415
        bfast_fused_magnetic_pair as module,
    )
    from meep_gpu.cuda_kernels import in_seam_coverage  # noqa: PLC0415

    def resolve(fields, grid, pml, arm):  # noqa: ARG001 - real family, no arm
        from meep_gpu.cuda_kernels import coverage as cov  # noqa: PLC0415

        codes, refusal = cov.real_curl_boundary_codes(grid)
        if refusal is not None:
            raise SystemExit(f"the fixture has no boundary codes: {refusal}")
        return {"tables": module.bfast_fused_magnetic_pair_tables(pml),
                "boundary_codes": tuple(np.int32(code) for code in codes),
                "walls": in_seam_coverage.zero_metal_axes(grid),
                "coefficients": module.bfast_scalars(grid),
                "dtdx": float(grid.dt / grid.dx)}

    def launch(fields, arguments, kernel=None):
        return module.launch_bfast_fused_magnetic_pair(
            fields, arguments["tables"], arguments["boundary_codes"],
            arguments["walls"], arguments["coefficients"], arguments["dtdx"],
            kernel)

    def separate(fields, grid, pml, arm):  # noqa: ARG001
        from meep_gpu.cuda_kernels import bfast_curl  # noqa: PLC0415
        from meep_gpu.cuda_kernels import constitutive_kernels  # noqa: PLC0415
        from meep_gpu.cuda_kernels import step_curl_kernels  # noqa: PLC0415
        from meep_gpu.cuda_kernels import coverage as cov  # noqa: PLC0415

        codes, refusal = cov.real_curl_boundary_codes(grid)
        assert refusal is None, refusal
        bfast_curl.step_bfast(
            "step_B", fields, step_curl_kernels.real_pml_curl_tables(pml, True),
            tuple(np.int32(code) for code in codes), float(grid.dt / grid.dx),
            module.bfast_scalars(grid))
        stepping.zero_metal_B(fields)
        constitutive_kernels.update_fused_pml_real("H", fields, pml=pml)

    return {
        "name": "bfast",
        "shape": "bfast",
        "seam": "B",
        "curl_sub_step": "step_B",
        "constitutive_side": "update_H",
        "seam_constitutive": "H",
        "module": module,
        "specs": BFAST_SPECS,
        "build": build_bfast_pair,
        "covers": _real_covers(module.covers_bfast_fused_magnetic_pair),
        "resolve": resolve,
        "launch": launch,
        "separate": separate,
        "source": lambda arm=None: module.bfast_fused_magnetic_pair_source(),
        "emitter_module": "meep_gpu.cuda_kernels.bfast_fused_magnetic_pair",
        "emitter_attribute": "bfast_fused_magnetic_pair_source",
        "disjoint": lambda fields, arguments: module.assert_disjoint_bindings(
            fields, arguments["tables"]),
        "licensed": False,
        "state_names": BFAST_STATE_NAMES,
        "lift_arms": ("REAL",),
        "lift_arithmetic_markers": ("= dtdx * (", "shift_up(", "shift_dn("),
        "refusal_cases": _bfast_refusal_cases,
        "certified_curl": (lambda arm, variant=None: __import__(
            "meep_gpu.cuda_kernels.bfast_curl",
            fromlist=["kernel_source"]).kernel_source("step_B_bfast_real")),
        "certified_constitutive": (lambda arm, variant=None: __import__(
            "meep_gpu.cuda_kernels.constitutive_kernels",
            fromlist=["_update_H_pml_real_kernel_code"])
            ._update_H_pml_real_kernel_code),
        "mutation_labels": ("bfast_periodic", "bfast_wall_y", "bfast_wall_xy_thin"),
        "deposit_labels": (),
        "source_refusal_labels": ("bfast_wall_y",),
        "reduced_labels": ("bfast_periodic", "bfast_wall_y"),
    }


# ---------------------------------------------------------------------------
# THE FIVE RESIDUAL ELECTRIC FAMILIES, 2026-09-02 -- the D-side twins
# ---------------------------------------------------------------------------
#
# Each is its magnetic sibling three seam arguments apart -- the BACKWARD Bloch
# table where one exists, the mirror-image sub-lattice pairing (``kms_half_*``),
# and the OFF-DIAGONAL wall table -- plus the three inverse-permittivity
# pointers only the electric seam binds, which is why every builder installs the
# anisotropic material. EVERY ROW OF EVERY TWIN'S BOARD CELL DECLARES AN
# ELECTRIC DEPOSIT INSIDE THE SEAM (measured off the residualwelds census), so
# each family runs deposit legs rather than source-refusal legs: the flag IS the
# product on these cells.

def _folded_electric_refusal_cases(record) -> None:
    """The folded ELECTRIC weld's own admission/refusal table."""
    from meep_gpu.fields import Fields  # noqa: PLC0415
    from meep_gpu.grid import Grid, Mirror  # noqa: PLC0415
    from meep_gpu.pml import PML  # noqa: PLC0415

    def build(symmetry=(Mirror("Y", 1),), complex_storage=True, beta=None,
              k_point=(0.0, 0.0, 0.0)):
        kwargs = {"beta": beta, "dimensions": 2} if beta else {}
        grid = Grid(resolution=1.0, cell_size=(9.0, 10.0, 11.0 if not beta else 0.0),
                    boundaries=("metallic", "periodic", "periodic"),
                    symmetry=tuple(symmetry), k_point=k_point, xp=cp, courant=0.5,
                    **kwargs)
        fields = Fields(grid=grid, force_complex_fields=complex_storage)
        fields.enable_field_storage()
        fields.enable_pml_storage()
        thickness = tuple((0, 0) if grid.is_mirrored(axis)
                          else (0, 0) if int(grid.shape[axis]) < 6
                          else (2, 2)
                          for axis in range(3))
        return fields, PML(grid=grid, thickness=thickness), grid

    fields, pml, grid = build()
    record("folded_complex_admitted", "covered", fields, pml, grid, ())
    record("undeclared_source_set", "refused", fields, pml, grid, None)
    fields, pml, grid = build(symmetry=())
    record("unfolded_belongs_to_the_plain_pair", "refused", fields, pml, grid, ())
    fields, pml, grid = build(complex_storage=False)
    record("real_storage", "refused", fields, pml, grid, ())
    fields, pml, grid = build(beta=0.2)
    record("beta_belongs_to_the_beta_weld", "refused", fields, pml, grid, ())

    # THE ELECTRIC SEAM'S OWN TWO: a conductivity re-routes the deposit through
    # _inject_electric_through_conductivity, on which the repair has no verdict;
    # an in-seam D source with no index has nothing the repair can save.
    live, live_pml, live_grid = build()

    class _Conductive:
        def __getattr__(self, item):
            return getattr(live, item)

        has_conductivity = True

    record("engine_conductivity", "refused", _Conductive(), live_pml, live_grid, ())

    class _Opaque:
        field_type = "D"

    record("in_seam_D_source_without_an_index", "refused", live, live_pml,
           live_grid, (_Opaque(),))


def _beta_electric_refusal_cases(record) -> None:
    """The beta ELECTRIC weld's own admission/refusal table."""
    from meep_gpu.fields import Fields  # noqa: PLC0415
    from meep_gpu.grid import Grid, Mirror  # noqa: PLC0415
    from meep_gpu.pml import PML  # noqa: PLC0415

    def build(symmetry=(Mirror("Y", 1),), complex_storage=True, beta=0.2):
        kwargs = {"beta": beta} if beta else {}
        grid = Grid(resolution=1.0, cell_size=(9.0, 10.0, 0.0),
                    boundaries=("metallic", "periodic", "periodic"),
                    symmetry=tuple(symmetry), xp=cp, courant=0.5, dimensions=2,
                    **kwargs)
        fields = Fields(grid=grid, force_complex_fields=complex_storage)
        fields.enable_field_storage()
        fields.enable_pml_storage()
        thickness = tuple((0, 0) if grid.is_mirrored(axis)
                          else (0, 0) if int(grid.shape[axis]) < 6 else (2, 2)
                          for axis in range(3))
        return fields, PML(grid=grid, thickness=thickness), grid

    fields, pml, grid = build()
    record("folded_beta_admitted", "covered", fields, pml, grid, ())
    fields, pml, grid = build(symmetry=())
    record("unfolded_beta_admitted", "covered", fields, pml, grid, ())
    record("undeclared_source_set", "refused", fields, pml, grid, None)
    fields, pml, grid = build(beta=None)
    record("no_beta_belongs_to_the_folded_weld", "refused", fields, pml, grid, ())
    fields, pml, grid = build(symmetry=(), beta=None)
    record("no_beta_unfolded_belongs_to_the_plain_pair", "refused",
           fields, pml, grid, ())
    live, live_pml, live_grid = build()

    class _Conductive:
        def __getattr__(self, item):
            return getattr(live, item)

        has_conductivity = True

    record("engine_conductivity", "refused", _Conductive(), live_pml, live_grid, ())

    class _Opaque:
        field_type = "D"

    record("in_seam_D_source_without_an_index", "refused", live, live_pml,
           live_grid, (_Opaque(),))


def _cylindrical_real_electric_refusal_cases(record) -> None:
    """The Dcyl real ELECTRIC weld's own admission/refusal table."""
    from meep_gpu.fields import Fields  # noqa: PLC0415
    from meep_gpu.grid import Grid  # noqa: PLC0415
    from meep_gpu.pml import PML  # noqa: PLC0415

    def build(complex_storage=False, cylindrical=True):
        if cylindrical:
            grid = Grid(resolution=1.0, cell_size=(9.0, 0.0, 11.0),
                        cylindrical=True, m=0, boundaries={"z": "metallic"},
                        courant=0.5, xp=cp)
            thickness = {"x": (0, 2), "z": 2}
        else:
            grid = Grid(resolution=1.0, cell_size=(9.0, 10.0, 11.0),
                        boundaries=("periodic",) * 3, courant=0.5, xp=cp)
            thickness = 2
        fields = Fields(grid=grid, force_complex_fields=complex_storage)
        fields.enable_field_storage()
        fields.enable_pml_storage()
        return fields, PML(grid=grid, thickness=thickness), grid

    fields, pml, grid = build()
    record("m0_real_dcyl_admitted", "covered", fields, pml, grid, ())
    record("undeclared_source_set", "refused", fields, pml, grid, None)
    fields, pml, grid = build(complex_storage=True)
    record("complex_storage_belongs_to_the_complex_dcyl_weld", "refused",
           fields, pml, grid, ())
    fields, pml, grid = build(cylindrical=False)
    record("cartesian_belongs_to_the_real_pair", "refused", fields, pml, grid, ())
    live, live_pml, live_grid = build()

    class _Conductive:
        def __getattr__(self, item):
            return getattr(live, item)

        has_conductivity = True

    record("engine_conductivity", "refused", _Conductive(), live_pml, live_grid, ())

    class _Opaque:
        field_type = "D"

    record("in_seam_D_source_without_an_index", "refused", live, live_pml,
           live_grid, (_Opaque(),))


def _special_kz_electric_refusal_cases(record) -> None:
    """The special_kz ELECTRIC weld's own admission/refusal table -- the fold
    admitted, exactly as the magnetic twin's is since the fill carry."""
    from meep_gpu.fields import Fields  # noqa: PLC0415
    from meep_gpu.grid import Grid, Mirror  # noqa: PLC0415
    from meep_gpu.pml import PML  # noqa: PLC0415

    def build(symmetry=(), complex_storage=False, beta=0.33):
        kwargs = {"beta": beta} if beta else {}
        grid = Grid(resolution=1.0, cell_size=(12.0, 9.0, 0.0),
                    boundaries=("metallic", "periodic", "periodic")
                    if symmetry else ("periodic",) * 3,
                    symmetry=tuple(symmetry), xp=cp, courant=0.5, dimensions=2,
                    **kwargs)
        fields = Fields(grid=grid, force_complex_fields=complex_storage)
        fields.enable_field_storage()
        fields.enable_pml_storage()
        thickness = tuple((0, 0) if grid.is_mirrored(axis)
                          else (0, 0) if int(grid.shape[axis]) < 6 else (2, 2)
                          for axis in range(3))
        return fields, PML(grid=grid, thickness=thickness), grid

    fields, pml, grid = build()
    record("real_beta_admitted", "covered", fields, pml, grid, ())
    record("undeclared_source_set", "refused", fields, pml, grid, None)
    fields, pml, grid = build(symmetry=(Mirror("Y", 1),))
    record("folded_admitted_since_the_fill_carry", "covered", fields, pml, grid, ())
    fields, pml, grid = build(beta=None)
    record("no_beta_belongs_to_the_real_pair", "refused", fields, pml, grid, ())
    fields, pml, grid = build(complex_storage=True)
    record("complex_storage_belongs_to_the_beta_weld", "refused",
           fields, pml, grid, ())
    live, live_pml, live_grid = build()

    class _Conductive:
        def __getattr__(self, item):
            return getattr(live, item)

        has_conductivity = True

    record("engine_conductivity", "refused", _Conductive(), live_pml, live_grid, ())

    class _Opaque:
        field_type = "D"

    record("in_seam_D_source_without_an_index", "refused", live, live_pml,
           live_grid, (_Opaque(),))


def _bfast_electric_refusal_cases(record) -> None:
    """The BFAST ELECTRIC weld's own admission/refusal table."""
    from meep_gpu.fields import Fields  # noqa: PLC0415
    from meep_gpu.grid import Grid, Mirror  # noqa: PLC0415
    from meep_gpu.pml import PML  # noqa: PLC0415

    def build(symmetry=(), bfast=(0.35, 0.0, 0.0)):
        kwargs = {"bfast_scaled_k": bfast} if bfast else {}
        grid = Grid(resolution=1.0, cell_size=(10.0, 9.0, 0.0),
                    boundaries=("metallic", "periodic", "periodic")
                    if symmetry else ("periodic",) * 3,
                    symmetry=tuple(symmetry), xp=cp, courant=0.5, dimensions=2,
                    **kwargs)
        fields = Fields(grid=grid)
        fields.enable_field_storage()
        fields.enable_pml_storage()
        thickness = tuple((0, 0) if grid.is_mirrored(axis)
                          else (0, 0) if int(grid.shape[axis]) < 6 else (2, 2)
                          for axis in range(3))
        return fields, PML(grid=grid, thickness=thickness), grid

    fields, pml, grid = build()
    record("bfast_admitted", "covered", fields, pml, grid, ())
    record("undeclared_source_set", "refused", fields, pml, grid, None)
    fields, pml, grid = build(bfast=None)
    record("no_bfast_belongs_to_the_real_pair", "refused", fields, pml, grid, ())
    fields, pml, grid = build(symmetry=(Mirror("Y", 1),))
    record("folded_refused", "refused", fields, pml, grid, ())
    live, live_pml, live_grid = build()

    class _Conductive:
        def __getattr__(self, item):
            return getattr(live, item)

        has_conductivity = True

    record("engine_conductivity", "refused", _Conductive(), live_pml, live_grid, ())

    class _Opaque:
        field_type = "D"

    record("in_seam_D_source_without_an_index", "refused", live, live_pml,
           live_grid, (_Opaque(),))


def _folded_complex_electric_family() -> Dict[str, Any]:
    """The FOLDED-COMPLEX D/E weld -- the D-side mirror of the folded magnetic
    weld, through the shared ``complex_electric_fill_carry``."""
    from meep_gpu.cuda_kernels import (  # noqa: PLC0415
        complex_folded_fused_electric_pair as module,
    )
    from meep_gpu.cuda_kernels import in_seam_coverage  # noqa: PLC0415

    def resolve(fields, grid, pml, arm):
        from meep_gpu.cuda_kernels import complex_folded_kernels  # noqa: PLC0415
        from meep_gpu.cuda_kernels import complex_pml_kernels  # noqa: PLC0415

        codes, refusal = complex_folded_kernels.folded_complex_boundary_codes(grid)
        if refusal is not None:
            raise SystemExit(f"the fixture has no folded boundary codes: {refusal}")
        # ``True`` IS THE BACKWARD FLAG -- the step_D sub-step's own conjugation.
        flags, values = complex_pml_kernels.bloch_phase_arguments(grid, True)
        return {"tables": module.complex_folded_fused_electric_pair_tables(pml),
                "boundary_codes": codes,
                "phase_flags": flags, "phase_values": values,
                "walls": in_seam_coverage.zero_metal_axes(grid),
                "fills": module.complex_folded_fused_electric_pair_fills(grid),
                "dtdx": float(grid.dt / grid.dx), "arm": arm}

    def launch(fields, arguments, kernel=None):
        return module.launch_complex_folded_fused_electric_pair(
            fields, arguments["tables"], arguments["boundary_codes"],
            arguments["phase_flags"], arguments["phase_values"],
            arguments["walls"], arguments["fills"], arguments["dtdx"],
            arguments["arm"], kernel)

    def separate(fields, grid, pml, arm):
        from meep_gpu.cuda_kernels import complex_folded_kernels  # noqa: PLC0415
        from meep_gpu.cuda_kernels import complex_pml_kernels  # noqa: PLC0415

        complex_folded_kernels.step_folded_complex(
            "step_D", fields, arm, grid=grid, pml=pml)
        stepping.fill_symmetry_bc_D(fields)
        stepping.zero_metal_D(fields)
        stepping.fill_folded_far_ghosts_D(fields)
        complex_pml_kernels.update_fused_pml_complex("E", fields, arm, pml=pml)

    def separate_without_fills(fields, grid, pml, arm):
        from meep_gpu.cuda_kernels import complex_folded_kernels  # noqa: PLC0415
        from meep_gpu.cuda_kernels import complex_pml_kernels  # noqa: PLC0415

        complex_folded_kernels.step_folded_complex(
            "step_D", fields, arm, grid=grid, pml=pml)
        stepping.zero_metal_D(fields)
        complex_pml_kernels.update_fused_pml_complex("E", fields, arm, pml=pml)

    return {
        "name": "folded_complex_electric",
        "shape": "cartesian",
        "seam": "D",
        "curl_sub_step": "step_D",
        "constitutive_side": "update_E",
        "seam_constitutive": "E",
        "module": module,
        "specs": FOLDED_COMPLEX_SPECS,
        "build": build_folded_complex_electric,
        "covers": module.covers_complex_folded_fused_electric_pair,
        "resolve": resolve,
        "launch": launch,
        "separate": separate,
        "separate_without_fills": separate_without_fills,
        "source": module.complex_folded_fused_electric_pair_source,
        "emitter_module":
            "meep_gpu.cuda_kernels.complex_folded_fused_electric_pair",
        "emitter_attribute": "complex_folded_fused_electric_pair_source",
        "disjoint": lambda fields, arguments: module.assert_disjoint_bindings(
            fields, arguments["tables"]),
        "fused_passes": ("step_D", "fill_symmetry_bc_D", "zero_metal_D",
                         "fill_folded_far_ghosts_D", "update_E"),
        "carry_floor": True,
        "refusal_cases": _folded_electric_refusal_cases,
        "certified_curl": (lambda arm, variant=None: __import__(
            "meep_gpu.cuda_kernels.complex_folded_kernels",
            fromlist=["folded_source"]).folded_source("step_D", arm)),
        "mutation_labels": ("fold_y_periodic_even", "fold_y_periodic_odd",
                            "fold_y_odd_phase", "fold_xy_mixed_phase",
                            "fold_y_wall_x_odd", "fold_x_periodic"),
        "deposit_labels": ("fold_y_periodic_even", "fold_y_metallic"),
        "deposit_sheet": True,
        "reduced_labels": ("fold_y_periodic_even", "fold_xy_mixed_phase"),
    }


def _complex_beta_electric_family() -> Dict[str, Any]:
    """The COMPLEX-BETA D/E weld -- the folded electric weld plus the beta insert."""
    from meep_gpu.cuda_kernels import (  # noqa: PLC0415
        complex_beta_fused_electric_pair as module,
    )
    from meep_gpu.cuda_kernels import in_seam_coverage  # noqa: PLC0415

    def resolve(fields, grid, pml, arm):
        from meep_gpu.cuda_kernels import complex_folded_kernels  # noqa: PLC0415
        from meep_gpu.cuda_kernels import complex_pml_kernels  # noqa: PLC0415

        codes, refusal = complex_folded_kernels.folded_complex_boundary_codes(grid)
        if refusal is not None:
            raise SystemExit(f"the fixture has no fold-aware boundary codes: "
                             f"{refusal}")
        flags, values = complex_pml_kernels.bloch_phase_arguments(grid, True)
        return {"tables": module.complex_beta_fused_electric_pair_tables(pml),
                "boundary_codes": codes,
                "phase_flags": flags, "phase_values": values,
                "beta_words": module.beta_coefficients(grid),
                "walls": in_seam_coverage.zero_metal_axes(grid),
                "fills": module.complex_beta_fused_electric_pair_fills(grid),
                "dtdx": float(grid.dt / grid.dx), "arm": arm}

    def launch(fields, arguments, kernel=None):
        return module.launch_complex_beta_fused_electric_pair(
            fields, arguments["tables"], arguments["boundary_codes"],
            arguments["phase_flags"], arguments["phase_values"],
            arguments["beta_words"], arguments["walls"], arguments["fills"],
            arguments["dtdx"], arguments["arm"], kernel)

    def separate(fields, grid, pml, arm):
        from meep_gpu.cuda_kernels import complex_beta_kernels  # noqa: PLC0415
        from meep_gpu.cuda_kernels import complex_pml_kernels  # noqa: PLC0415

        complex_beta_kernels.step_complex_beta(
            "step_D", fields, arm, grid=grid, pml=pml)
        stepping.fill_symmetry_bc_D(fields)
        stepping.zero_metal_D(fields)
        stepping.fill_folded_far_ghosts_D(fields)
        complex_pml_kernels.update_fused_pml_complex("E", fields, arm, pml=pml)

    def separate_without_fills(fields, grid, pml, arm):
        from meep_gpu.cuda_kernels import complex_beta_kernels  # noqa: PLC0415
        from meep_gpu.cuda_kernels import complex_pml_kernels  # noqa: PLC0415

        complex_beta_kernels.step_complex_beta(
            "step_D", fields, arm, grid=grid, pml=pml)
        stepping.zero_metal_D(fields)
        complex_pml_kernels.update_fused_pml_complex("E", fields, arm, pml=pml)

    return {
        "name": "complex_beta_electric",
        "shape": "cartesian",
        "seam": "D",
        "curl_sub_step": "step_D",
        "constitutive_side": "update_E",
        "seam_constitutive": "E",
        "module": module,
        "specs": COMPLEX_BETA_SPECS,
        "build": build_folded_complex_electric,
        "covers": module.covers_complex_beta_fused_electric_pair,
        "resolve": resolve,
        "launch": launch,
        "separate": separate,
        "separate_without_fills": separate_without_fills,
        "source": module.complex_beta_fused_electric_pair_source,
        "emitter_module": "meep_gpu.cuda_kernels.complex_beta_fused_electric_pair",
        "emitter_attribute": "complex_beta_fused_electric_pair_source",
        "disjoint": lambda fields, arguments: module.assert_disjoint_bindings(
            fields, arguments["tables"]),
        "fused_passes": ("step_D", "fill_symmetry_bc_D", "zero_metal_D",
                         "fill_folded_far_ghosts_D", "update_E"),
        "carry_floor": True,
        "refusal_cases": _beta_electric_refusal_cases,
        "certified_curl": (lambda arm, variant=None: __import__(
            "meep_gpu.cuda_kernels.complex_beta_kernels",
            fromlist=["beta_source"]).beta_source("step_D", arm)),
        "mutation_labels": ("beta_fold_y_periodic_even",
                            "beta_fold_y_periodic_odd", "beta_fold_y_odd_phase",
                            "beta_unfolded_bloch_x", "beta_fold_y_wall_x"),
        "deposit_labels": ("beta_fold_y_periodic_even", "beta_unfolded_bloch_x"),
        "deposit_sheet": True,
        "reduced_labels": ("beta_fold_y_periodic_even", "beta_unfolded_bloch_x"),
    }


def _cylindrical_real_electric_family() -> Dict[str, Any]:
    """The REAL m = 0 Dcyl D/E weld."""
    from meep_gpu.cuda_kernels import (  # noqa: PLC0415
        cylindrical_real_fused_electric_pair as module,
    )
    from meep_gpu.cuda_kernels import in_seam_coverage  # noqa: PLC0415

    def resolve(fields, grid, pml, arm):  # noqa: ARG001 - real family, no arm
        from meep_gpu.cuda_kernels import coverage as cov  # noqa: PLC0415
        from meep_gpu.cuda_kernels import cylindrical_coverage  # noqa: PLC0415

        codes = cylindrical_coverage.cylindrical_boundary_codes(
            tuple(cov.real_pml_boundary_kinds(grid)))
        return {"tables": module.cylindrical_real_fused_electric_pair_tables(pml),
                "boundary_codes": tuple(np.int32(code) for code in codes),
                "walls": in_seam_coverage.zero_metal_axes(grid),
                "dtdx": float(grid.dt / grid.dx)}

    def launch(fields, arguments, kernel=None):
        from meep_gpu.cuda_kernels.cylindrical_prefix import (  # noqa: PLC0415
            cylindrical_prefix,
        )

        return module.launch_cylindrical_real_fused_electric_pair(
            fields, arguments["tables"],
            cylindrical_prefix(fields, "step_D",
                               scratch=getattr(fields, "scratch", None)),
            arguments["boundary_codes"], arguments["walls"],
            arguments["dtdx"], kernel)

    def separate(fields, grid, pml, arm):  # noqa: ARG001
        from meep_gpu.cuda_kernels import constitutive_kernels  # noqa: PLC0415
        from meep_gpu.cuda_kernels import cylindrical_kernels  # noqa: PLC0415

        stepped = cylindrical_kernels.step_cylindrical(fields, pml, "step_D")
        if not stepped:
            raise SystemExit(
                "the certified Dcyl curl refused the separate route's fixture; "
                "a silent False here would masquerade as a weld divergence")
        stepping.zero_metal_D(fields)
        constitutive_kernels.update_fused_pml_real("E", fields, pml=pml)

    return {
        "name": "cylindrical_real_electric",
        "shape": "cylindrical_real",
        "seam": "D",
        "curl_sub_step": "step_D",
        "constitutive_side": "update_E",
        "seam_constitutive": "E",
        "module": module,
        "specs": CYLINDRICAL_REAL_SPECS,
        "build": build_cylindrical_real_electric,
        "covers": _real_covers(module.covers_cylindrical_real_fused_electric_pair),
        "resolve": resolve,
        "launch": launch,
        "separate": separate,
        "source": lambda arm=None: module.cylindrical_real_fused_electric_pair_source(),
        "emitter_module":
            "meep_gpu.cuda_kernels.cylindrical_real_fused_electric_pair",
        "emitter_attribute": "cylindrical_real_fused_electric_pair_source",
        "disjoint": lambda fields, arguments: True,
        "licensed": False,
        "lift_arms": ("REAL",),
        "lift_arithmetic_markers": ("= dtdx * (", "shift_up(", "shift_dn("),
        "refusal_cases": _cylindrical_real_electric_refusal_cases,
        "certified_curl": (lambda arm, variant=None: __import__(
            "meep_gpu.cuda_kernels.cylindrical_kernels",
            fromlist=["_cyl_step_D_pml_real_kernel_code"])
            ._cyl_step_D_pml_real_kernel_code),
        "certified_constitutive": (lambda arm, variant=None: __import__(
            "meep_gpu.cuda_kernels.constitutive_kernels",
            fromlist=["_update_E_pml_real_kernel_code"])
            ._update_E_pml_real_kernel_code),
        "mutation_labels": ("m0_z_metallic", "m0_z_periodic", "m0_thin_absorber"),
        "deposit_labels": ("m0_z_metallic", "m0_z_periodic"),
        "reduced_labels": ("m0_z_metallic", "m0_z_periodic"),
    }


def _special_kz_electric_family() -> Dict[str, Any]:
    """The REAL special_kz D/E weld -- the real electric pair's fill carry over
    the beta curl, all five passes fused."""
    from meep_gpu.cuda_kernels import (  # noqa: PLC0415
        special_kz_fused_electric_pair as module,
    )
    from meep_gpu.cuda_kernels import in_seam_coverage  # noqa: PLC0415

    def resolve(fields, grid, pml, arm):  # noqa: ARG001 - real family, no arm
        from meep_gpu.cuda_kernels import coverage as cov  # noqa: PLC0415

        codes, refusal = cov.real_curl_boundary_codes(grid)
        if refusal is not None:
            raise SystemExit(f"the fixture has no boundary codes: {refusal}")
        return {"tables": module.special_kz_fused_electric_pair_tables(pml),
                "boundary_codes": tuple(np.int32(code) for code in codes),
                "walls": in_seam_coverage.zero_metal_axes(grid),
                "beta": module.beta_scalars(grid),
                "fills": module.special_kz_fused_electric_pair_fills(grid),
                "dtdx": float(grid.dt / grid.dx)}

    def launch(fields, arguments, kernel=None):
        return module.launch_special_kz_fused_electric_pair(
            fields, arguments["tables"], arguments["boundary_codes"],
            arguments["walls"], arguments["beta"], arguments["fills"],
            arguments["dtdx"], kernel)

    def separate(fields, grid, pml, arm):  # noqa: ARG001
        from meep_gpu.cuda_kernels import constitutive_kernels  # noqa: PLC0415
        from meep_gpu.cuda_kernels import special_kz_curl  # noqa: PLC0415
        from meep_gpu.cuda_kernels import step_curl_kernels  # noqa: PLC0415
        from meep_gpu.cuda_kernels import coverage as cov  # noqa: PLC0415

        codes, refusal = cov.real_curl_boundary_codes(grid)
        assert refusal is None, refusal
        special_kz_curl.step_special_kz(
            "step_D", fields, step_curl_kernels.real_pml_curl_tables(pml, False),
            tuple(np.int32(code) for code in codes), float(grid.dt / grid.dx),
            module.beta_scalars(grid))
        stepping.fill_symmetry_bc_D(fields)
        stepping.zero_metal_D(fields)
        stepping.fill_folded_far_ghosts_D(fields)
        constitutive_kernels.update_fused_pml_real("E", fields, pml=pml)

    def separate_without_fills(fields, grid, pml, arm):  # noqa: ARG001
        from meep_gpu.cuda_kernels import constitutive_kernels  # noqa: PLC0415
        from meep_gpu.cuda_kernels import special_kz_curl  # noqa: PLC0415
        from meep_gpu.cuda_kernels import step_curl_kernels  # noqa: PLC0415
        from meep_gpu.cuda_kernels import coverage as cov  # noqa: PLC0415

        codes, refusal = cov.real_curl_boundary_codes(grid)
        assert refusal is None, refusal
        special_kz_curl.step_special_kz(
            "step_D", fields, step_curl_kernels.real_pml_curl_tables(pml, False),
            tuple(np.int32(code) for code in codes), float(grid.dt / grid.dx),
            module.beta_scalars(grid))
        stepping.zero_metal_D(fields)
        constitutive_kernels.update_fused_pml_real("E", fields, pml=pml)

    return {
        "name": "special_kz_electric",
        "shape": "real_beta",
        "seam": "D",
        "curl_sub_step": "step_D",
        "constitutive_side": "update_E",
        "seam_constitutive": "E",
        "module": module,
        "specs": SPECIAL_KZ_SPECS,
        "build": build_special_kz_electric,
        "covers": _real_covers(module.covers_special_kz_fused_electric_pair),
        "resolve": resolve,
        "launch": launch,
        "separate": separate,
        "separate_without_fills": separate_without_fills,
        "source": lambda arm=None: module.special_kz_fused_electric_pair_source(),
        "emitter_module": "meep_gpu.cuda_kernels.special_kz_fused_electric_pair",
        "emitter_attribute": "special_kz_fused_electric_pair_source",
        "disjoint": lambda fields, arguments: module.assert_disjoint_bindings(
            fields, arguments["tables"]),
        "licensed": False,
        "lift_arms": ("REAL",),
        "lift_arithmetic_markers": ("= dtdx * (", "shift_up(", "shift_dn("),
        "fused_passes": ("step_D", "fill_symmetry_bc_D", "zero_metal_D",
                         "fill_folded_far_ghosts_D", "update_E"),
        "carry_floor": True,
        "refusal_cases": _special_kz_electric_refusal_cases,
        "certified_curl": (lambda arm, variant=None: __import__(
            "meep_gpu.cuda_kernels.special_kz_curl",
            fromlist=["kernel_source"]).kernel_source("step_D_special_kz_real")),
        "certified_constitutive": (lambda arm, variant=None: __import__(
            "meep_gpu.cuda_kernels.constitutive_kernels",
            fromlist=["_update_E_pml_real_kernel_code"])
            ._update_E_pml_real_kernel_code),
        "mutation_labels": ("skz_periodic", "skz_wall_y", "skz_wall_xy_thin",
                            "skz_fold_y_even", "skz_fold_y_odd"),
        "deposit_labels": ("skz_wall_y", "skz_fold_y_even"),
        "deposit_sheet": True,
        "reduced_labels": ("skz_periodic", "skz_wall_y", "skz_fold_y_even"),
    }


def _bfast_electric_family() -> Dict[str, Any]:
    """The REAL BFAST D/E weld."""
    from meep_gpu.cuda_kernels import (  # noqa: PLC0415
        bfast_fused_electric_pair as module,
    )
    from meep_gpu.cuda_kernels import in_seam_coverage  # noqa: PLC0415

    def resolve(fields, grid, pml, arm):  # noqa: ARG001 - real family, no arm
        from meep_gpu.cuda_kernels import bfast_curl  # noqa: PLC0415
        from meep_gpu.cuda_kernels import coverage as cov  # noqa: PLC0415

        codes, refusal = cov.real_curl_boundary_codes(grid)
        if refusal is not None:
            raise SystemExit(f"the fixture has no boundary codes: {refusal}")
        return {"tables": module.bfast_fused_electric_pair_tables(pml),
                "boundary_codes": tuple(np.int32(code) for code in codes),
                "walls": in_seam_coverage.zero_metal_axes(grid),
                "coefficients": bfast_curl.bfast_curl_coefficients(grid, "step_D"),
                "dtdx": float(grid.dt / grid.dx)}

    def launch(fields, arguments, kernel=None):
        return module.launch_bfast_fused_electric_pair(
            fields, arguments["tables"], arguments["boundary_codes"],
            arguments["walls"], arguments["coefficients"], arguments["dtdx"],
            kernel)

    def separate(fields, grid, pml, arm):  # noqa: ARG001
        from meep_gpu.cuda_kernels import bfast_curl  # noqa: PLC0415
        from meep_gpu.cuda_kernels import constitutive_kernels  # noqa: PLC0415
        from meep_gpu.cuda_kernels import step_curl_kernels  # noqa: PLC0415
        from meep_gpu.cuda_kernels import coverage as cov  # noqa: PLC0415

        codes, refusal = cov.real_curl_boundary_codes(grid)
        assert refusal is None, refusal
        bfast_curl.step_bfast(
            "step_D", fields, step_curl_kernels.real_pml_curl_tables(pml, False),
            tuple(np.int32(code) for code in codes), float(grid.dt / grid.dx),
            bfast_curl.bfast_curl_coefficients(grid, "step_D"))
        stepping.zero_metal_D(fields)
        constitutive_kernels.update_fused_pml_real("E", fields, pml=pml)

    return {
        "name": "bfast_electric",
        "shape": "bfast",
        "seam": "D",
        "curl_sub_step": "step_D",
        "constitutive_side": "update_E",
        "seam_constitutive": "E",
        "module": module,
        "specs": BFAST_SPECS,
        "build": build_bfast_electric,
        "covers": _real_covers(module.covers_bfast_fused_electric_pair),
        "resolve": resolve,
        "launch": launch,
        "separate": separate,
        "source": lambda arm=None: module.bfast_fused_electric_pair_source(),
        "emitter_module": "meep_gpu.cuda_kernels.bfast_fused_electric_pair",
        "emitter_attribute": "bfast_fused_electric_pair_source",
        "disjoint": lambda fields, arguments: module.assert_disjoint_bindings(
            fields, arguments["tables"]),
        "licensed": False,
        "state_names": BFAST_STATE_NAMES,
        "lift_arms": ("REAL",),
        "lift_arithmetic_markers": ("= dtdx * (", "shift_up(", "shift_dn("),
        "refusal_cases": _bfast_electric_refusal_cases,
        "certified_curl": (lambda arm, variant=None: __import__(
            "meep_gpu.cuda_kernels.bfast_curl",
            fromlist=["kernel_source"]).kernel_source("step_D_bfast_real")),
        "certified_constitutive": (lambda arm, variant=None: __import__(
            "meep_gpu.cuda_kernels.constitutive_kernels",
            fromlist=["_update_E_pml_real_kernel_code"])
            ._update_E_pml_real_kernel_code),
        "mutation_labels": ("bfast_periodic", "bfast_wall_y", "bfast_wall_xy_thin"),
        "deposit_labels": ("bfast_periodic", "bfast_wall_y"),
        "reduced_labels": ("bfast_periodic", "bfast_wall_y"),
    }


FAMILIES: Dict[str, Callable[[], Dict[str, Any]]] = {
    "complex": _complex_family,
    "cylindrical": _cylindrical_family,
    "complex_electric": _complex_electric_family,
    "cylindrical_electric": _cylindrical_electric_family,
    "no_pml_complex_electric": _no_pml_complex_electric_family,
    # THE FIVE RESIDUAL MAGNETIC FAMILIES, 2026-09-02.
    "complex_folded": _folded_complex_family,
    "complex_beta": _complex_beta_family,
    "cylindrical_real": _cylindrical_real_family,
    "special_kz": _special_kz_family,
    "bfast": _bfast_family,
    # THE FIVE RESIDUAL ELECTRIC FAMILIES, 2026-09-02 (the residue round).
    "folded_complex_electric": _folded_complex_electric_family,
    "complex_beta_electric": _complex_beta_electric_family,
    "cylindrical_real_electric": _cylindrical_real_electric_family,
    "special_kz_electric": _special_kz_electric_family,
    "bfast_electric": _bfast_electric_family,
}


# ---------------------------------------------------------------------------
# VARIANTS: one product, more than one emitted kernel
# ---------------------------------------------------------------------------
#
# THE FOUR PML PRODUCTS EMIT ONE KERNEL EACH and the no-absorber one emits TWO -- a
# plain-tail weld and a conductive-tail weld, because its certified curl half ships
# both and every corpus row on its cell takes the second. So the source legs and the
# device mutations are parametrised over VARIANTS, and a family that declares none has
# exactly one whose label is ``None``. The four defaults below are what makes that
# reduction exact rather than approximate: for a single-variant family every accessor
# returns what it returned before this parametrisation landed.

def variants_of(family: Dict[str, Any]) -> Tuple[Any, ...]:
    return tuple(family.get("variants", (None,)))


def variant_for_spec(family: Dict[str, Any], spec: Dict[str, Any]) -> Any:
    """Which emitted kernel this fixture drives. One product may emit more than one."""
    chooser = family.get("variant_of")
    return chooser(spec) if chooser is not None else None


def fused_source(family: Dict[str, Any], arm: str, variant: Any = None) -> str:
    source = family["source"]
    try:
        return source(arm, variant)
    except TypeError:
        return source(arm)


def kernel_symbol(family: Dict[str, Any], variant: Any = None) -> str:
    """The ``extern "C"`` symbol NVRTC is asked for, per variant.

    A MULTI-VARIANT FAMILY'S ``KERNEL_NAME`` IS A LABEL AND NOT A SYMBOL -- the
    no-absorber module spells it in the bracket form its own arm table uses -- so a
    gate that compiled against ``module.KERNEL_NAME`` would ask NVRTC for a name no
    source declares and fail three frames from the cause.
    """
    resolver = family.get("kernel_symbol")
    return resolver(variant) if resolver is not None else family["module"].KERNEL_NAME


def certified_halves_source(family: Dict[str, Any], arm: str,
                            variant: Any = None) -> Tuple[str, str]:
    """The two certified sources this family splices, for one variant."""
    from meep_gpu.cuda_kernels import complex_emitter  # noqa: PLC0415

    curl = family.get("certified_curl")
    constitutive = family.get("certified_constitutive")
    return (curl(arm, variant) if curl is not None
            else _certified_curl_source(family, arm),
            constitutive(arm, variant) if constitutive is not None
            else complex_emitter.complex_source(family["constitutive_side"], arm))


# ---------------------------------------------------------------------------
# Word-level comparison
# ---------------------------------------------------------------------------

def words(array: Any) -> np.ndarray:
    """One array as raw uint32 WORDS. Byte compares, never ``allclose``: ``-0.0 ==
    0.0`` and ``NaN != NaN`` both lie, and the band class puts signed zeros in the
    operands deliberately. A complex64 volume is viewed as its float32 word pairs."""
    host = to_host(array)
    host = np.ascontiguousarray(host)
    if host.dtype == np.complex64:
        host = host.view(np.float32)
    return np.ascontiguousarray(host, dtype=np.float32).ravel().view(np.uint32)


def differing(left: Any, right: Any) -> int:
    a, b = words(left), words(right)
    if a.shape != b.shape:
        return max(a.size, b.size)
    return int(np.count_nonzero(a != b))


def state_of(fields, names: Sequence[str] = STATE_NAMES) -> Dict[str, Any]:
    return {name: _slot_array(fields, name) for name in names}


def frozen(fields, names: Sequence[str] = STATE_NAMES) -> Dict[str, np.ndarray]:
    return {name: words(value).copy()
            for name, value in state_of(fields, names).items()}


def compare(left, right, names: Sequence[str] = STATE_NAMES) -> Dict[str, int]:
    a, b = state_of(left, names), state_of(right, names)
    return {name: n for name in names if (n := differing(a[name], b[name]))}


def read_only_snapshot(family: Dict[str, Any], fields, pml) -> Dict[str, np.ndarray]:
    """Every volume a run must leave BIT-UNCHANGED, per family.

    THE PML COEFFICIENT VECTORS ARE ONE ANSWER AND NOT THE ONLY ONE. A no-absorber
    family has none, and a snapshot that asked for them would come back EMPTY -- a
    check that inspected nothing and reported no drift, which is exactly the vacuity
    this gate refuses everywhere else. That family names its MATERIAL volumes instead:
    the three inverse permittivities the constitutive half multiplies by, the six
    conductivity coefficients the conductive tail reads, and every susceptibility
    sigma. All are read-only to a launch, and all are bound as pointers to it.
    """
    out: Dict[str, np.ndarray] = {}
    for name in IMMUTABLE_PML:
        value = getattr(pml, name, None)
        if value is not None:
            out[f"pml.{name}"] = words(value).copy()
    reader = family.get("read_only_material")
    if reader is not None:
        for name, value in reader(fields).items():
            out[name] = words(value).copy()
    return out


def read_only_drift(before: Dict[str, np.ndarray], family: Dict[str, Any],
                    fields, pml) -> List[str]:
    after = read_only_snapshot(family, fields, pml)
    return sorted(name for name in before
                  if name not in after
                  or before[name].shape != after[name].shape
                  or not np.array_equal(before[name], after[name]))


# ---------------------------------------------------------------------------
# Launch counting: two independent counters
# ---------------------------------------------------------------------------

class _CountingKernel:
    """A ``cp.RawKernel`` that counts its launches and delegates everything else."""

    __slots__ = ("_kernel", "_counter", "_name")

    def __init__(self, kernel: Any, counter: Dict[str, int], name: str) -> None:
        self._kernel, self._counter, self._name = kernel, counter, name

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self._counter[self._name] = self._counter.get(self._name, 0) + 1
        self._counter["_total"] = self._counter.get("_total", 0) + 1
        return self._kernel(*args, **kwargs)

    def __getattr__(self, item: str) -> Any:
        return getattr(self._kernel, item)


class MemoLaunchCounter:
    """Wrap every memoized device kernel; restore on exit.

    THE MEMO IS THE RIGHT SEAM and it is the shipped one. Every launcher in this
    package reaches its kernel through ``compile_cache.get_or_compile``, which returns
    ``_compiled_kernels[key]`` on a hit, so replacing the stored object counts every
    launch through every shipped route without editing one byte under ``meep_gpu/``.
    NOTHING IS INSTALLED UNLESS THE MEMO IS ALREADY WARM, which is why every case
    warms it first: a cold memo would have the factory store a raw kernel and the
    count would read zero for a kernel that ran.
    """

    def __init__(self) -> None:
        self.counts: Dict[str, int] = {}
        self._saved: Dict[Any, Any] = {}

    def __enter__(self) -> "MemoLaunchCounter":
        from meep_gpu.cuda_kernels import compile_cache  # noqa: PLC0415
        store = compile_cache._compiled_kernels  # noqa: SLF001 - the shipped memo
        self._saved = dict(store)
        for key, kernel in list(store.items()):
            store[key] = _CountingKernel(kernel, self.counts, str(key[0]))
        return self

    def __exit__(self, *exc: Any) -> None:
        from meep_gpu.cuda_kernels import compile_cache  # noqa: PLC0415
        store = compile_cache._compiled_kernels  # noqa: SLF001
        for key, kernel in list(store.items()):
            if isinstance(kernel, _CountingKernel) and key in self._saved:
                store[key] = self._saved[key]

    @property
    def total(self) -> int:
        return int(self.counts.get("_total", 0))

    def named(self) -> Dict[str, int]:
        return {k: v for k, v in sorted(self.counts.items()) if k != "_total"}


def _memo_key_for_fused(family: Dict[str, Any], arm: str,
                        variant: Any = None) -> str:
    """The name the memo proxy reports for the fused kernel.

    ``compile_cache.kernel_cache_key`` puts the launcher's own label in slot 0 and
    every module spells it ``<symbol>_arm<code>``, so the proxy's key is that string
    rather than the device entry point. Derived here from the shipped module's own
    constants rather than typed, so a renamed kernel is a mismatch the counter reports
    instead of a name this file quietly stops finding.

    THE SYMBOL IS PER VARIANT, and on the no-absorber family that is the difference
    between watching the kernel that ran and watching a name nothing compiles: its
    ``KERNEL_NAME`` is the bracket LABEL of two symbols, and its launcher keys the memo
    on whichever one the run's curl arm selected.
    """
    from meep_gpu.cuda_kernels import complex_emitter  # noqa: PLC0415

    if family.get("licensed", True) is False:
        # A REAL family's launcher keys the memo on the bare symbol -- its
        # arithmetic binds no expansion arm, so there is no arm code in the key.
        return kernel_symbol(family, variant)
    return (f"{kernel_symbol(family, variant)}_"
            f"arm{complex_emitter.normalized_expansion(arm)}")


def warm_memo(family: Dict[str, Any], specs: Sequence[Dict[str, Any]],
              arm: str) -> None:
    """Compile and launch every kernel this file drives, once, off the counted path.

    A REFUSED WARM-UP IS FATAL RATHER THAN SILENT: a spec the predicate does not
    admit would otherwise reach the product sweep and fail there with a reason about
    coverage, one leg later than the table that was wrong.

    EVERY VARIANT MUST BE REACHED. A multi-variant family whose fixture table drove
    only one of its emitted kernels would leave the other with a cold memo -- so the
    counter would read zero for a kernel that never ran, and the sweep would be gating
    one kernel while shipping two. The check is on the SPECS rather than on the launch,
    so the failure names the table that is short.
    """
    reached = {variant_for_spec(family, warm) for warm in specs}
    missing = [variant for variant in variants_of(family) if variant not in reached]
    if missing:
        raise SystemExit(
            f"the fixture table drives {sorted(map(str, reached))} but this family "
            f"emits {sorted(map(str, variants_of(family)))}; {sorted(map(str, missing))} "
            f"would ship ungated. Add a spec that drives it.")
    for warm in specs:
        fields, grid, pml, _ = family["build"](warm, "uniform",
                                               np.random.default_rng(1))
        covered, reason = family["covers"](fields, pml, grid, (), LICENCE["verdict"],
                                           POLICY["name"])
        if not covered:
            raise SystemExit(f"the memo warm-up is refused on {warm['label']}: "
                             f"{reason}")
        arguments = family["resolve"](fields, grid, pml, arm)
        family["launch"](fields, arguments)
        family["separate"](fields, grid, pml, arm)
    cp.cuda.runtime.deviceSynchronize()


# ---------------------------------------------------------------------------
# One product case
# ---------------------------------------------------------------------------

def run_case(family: Dict[str, Any], spec: Dict[str, Any], value_class: str,
             steps: int, arm: str, kernel: Optional[Any] = None,
             patch: Optional[Callable[[Dict[str, Any], Any], Dict[str, Any]]] = None,
             label_suffix: str = "") -> Dict[str, Any]:
    """Step THREE engines side by side and compare per COMPLETE DRIVER STEP.

    THE THIRD ENGINE IS THE SEPARATE CERTIFIED ROUTE, and it earned its place on
    2026-08-30 by ATTRIBUTING A DIVERGENCE THAT WAS NOT THIS WELD'S. On the first
    device run the fused route diverged from the array path on every phased
    subnormal-band fixture; the separate certified route diverged in the SAME words,
    on the same cells, with no fused kernel in the picture, and a re-run of
    ``gate_cuda_complex.py`` reproduced it (``single=29/36 passed=False``). The cause
    was neither kernel: that run had not installed the float32 subnormal policy, so
    ``-ftz=true`` binaries were compiled into a cache directory NAMED for the stripped
    policy and then served to every later run against it -- exactly the hazard
    ``subnormal_policy`` documents, since the CuPy cache key is computed ABOVE the
    strip seam. With ``install_subnormal_policy_for_run`` driven at the top of
    :func:`main`, both routes are bit-identical to the array path on every fixture and
    the certified family's own gate returns 36/36.

    THE THIRD ENGINE STAYS, because that episode is the argument for it: without it
    the run would have reported a fused-product defect, and the defect was in neither
    product. It also makes every passing case carry a strictly stronger statement.

    SO THE CASE MEASURES THREE THINGS AND CONFLATES NONE:

    * ``weld_identity`` -- the fused launch against the SEPARATE CERTIFIED SEQUENCE
      it replaces, word for word. THIS IS THE PRODUCT'S OWN CLAIM and it is required
      unconditionally, on every case and every class. It is a strictly stronger
      comparison than this gate would otherwise make, not a weaker one.
    * ``array_identity`` -- the fused launch against ``stepping``'s eleven passes.
      The full claim, required wherever the halves themselves achieve it.
    * ``half_identity`` -- the separate certified route against the array path. The
      ATTRIBUTION: where this is false the divergence belongs to a half, and the case
      then requires the fused route's divergence set to be IDENTICAL to it, cell for
      cell, rather than merely also nonzero.

    A case passes on ``weld_identity`` AND (``array_identity`` OR a divergence proved
    inherited). Nothing is skipped, nothing is tolerated on assertion: an inherited
    divergence is recorded with both word maps so a reader can check the claim.

    ``kernel`` and ``patch`` are THE GATE'S DOORS -- the shipped launcher takes a
    kernel and every resolved argument precisely so a gate can arm a defect. A harness
    that could not hand in its own kernel could not arm a single mutation, and every
    mutation leg would launch the shipped kernel and report the defect as uncaught.
    THEY REACH THE FUSED ENGINE ONLY: the separate route is never mutated, which is
    what makes ``weld_identity`` the thing a mutation is scored on.
    """
    started = time.time()
    label = f"{family['name']}|{spec['label']}|{value_class}{label_suffix}"
    case: Dict[str, Any] = {"family": family["name"], "label": spec["label"],
                            "value_class": value_class, "case_seed_label": label,
                            "steps_requested": steps, "spec": {
                                k: (list(v) if isinstance(v, tuple) else v)
                                for k, v in spec.items()}}

    reference, ref_grid, ref_pml, host = family["build"](spec, value_class,
                                                         case_rng(label))
    actual, grid, pml, _ = family["build"](spec, value_class, case_rng(label))
    apart, apart_grid, apart_pml, _ = family["build"](spec, value_class,
                                                      case_rng(label))
    case["shape"] = [int(n) for n in grid.shape]
    case["dtdx"] = float(grid.dt / grid.dx)
    case["operand_census"] = operand_census(host)
    # THE COMPARED SET IS THE FAMILY'S, and on a dispersive family it is read off the
    # ENGINE: the pole buffers are part of the state a complete driver step touches.
    names = state_names_for(family, reference)
    case["arrays_compared_by_name"] = list(names)

    drift = compare(reference, actual, names)
    drift_apart = compare(reference, apart, names)
    if drift or drift_apart:
        case["passed"] = False
        case["why"] = (f"the three builds are not identical: "
                       f"{drift or drift_apart}")
        return case

    covered, reason = family["covers"](actual, pml, grid, (), LICENCE["verdict"],
                                       POLICY["name"])
    case["predicate"] = {"covered": bool(covered), "reason": reason}
    if not covered:
        case["passed"] = False
        case["why"] = f"the predicate refused this fixture: {reason}"
        return case

    arguments = family["resolve"](actual, grid, pml, arm)
    case["distinct_allocations_checked"] = family["disjoint"](actual, arguments)
    # ``walls`` IS ABSENT ON A PRODUCT THAT CARRIES NO WALL PASS, and recorded as the
    # empty tuple rather than as three Falses: three Falses would read as "this launch
    # was told no axis is walled" when in fact it is told nothing, because it refuses
    # every walled run and has no such parameter.
    case["walls"] = [bool(w) for w in arguments.get("walls", ())]
    case["boundary_codes"] = [int(c) for c in arguments["boundary_codes"]]
    case["curl_arm"] = arguments.get("curl_arm")
    if patch is not None:
        arguments = patch(arguments, grid)
    case["walls_used"] = [bool(w) for w in arguments.get("walls", ())]
    case["boundary_codes_used"] = [int(c) for c in arguments["boundary_codes"]]
    case["arm_used"] = arguments.get("arm")   # absent on the unlicensed families
    case["curl_arm_used"] = arguments.get("curl_arm")

    before = frozen(actual, names)
    read_only = read_only_snapshot(family, actual, pml)
    launched = 0
    counter = MemoLaunchCounter()
    #: A SECOND counter, entered only around the fused launch. The outer one sees the
    #: separate certified route's launches too (they share the shipped memo), so the
    #: "only this kernel ran" question has to be asked of a window that contains
    #: nothing but the fused launch.
    fused_counter = MemoLaunchCounter()
    per_step: List[Dict[str, Any]] = []
    with counter:
        for step in range(1, steps + 1):
            array_step(reference, ref_pml)
            # THE SEPARATE CERTIFIED ROUTE runs OUTSIDE the launch counter's
            # bookkeeping question but inside the same loop, so both subjects see the
            # same step number and the same seeded history. Its launches land in the
            # counter too, which is why `other_kernels` below is read from a SECOND
            # counter rather than from this one.
            #
            # THE PASSES BEFORE THE SEAM RUN FIRST ON BOTH SUBJECTS, and on the
            # ELECTRIC seam that is the whole B/H half rather than nothing: a D-side
            # subject that launched before running it would be compared against an
            # oracle half a step ahead of it. Empty on the magnetic seam, where this
            # is bit-for-bit the sequence this gate ran before it was parametrised.
            before_passes(family, apart, apart_pml)
            family["separate"](apart, apart_grid, apart_pml, arm)
            unfused_passes_after_seam(family, apart, apart_pml)
            before_passes(family, actual, pml)
            with fused_counter:
                report = family["launch"](actual, arguments, kernel)
            launched += int(bool(report.get("launched")))
            unfused_passes(family, actual, pml)
            cp.cuda.runtime.deviceSynchronize()
            difference = compare(reference, actual, names)
            weld = compare(apart, actual, names)
            half = compare(reference, apart, names)
            census = operand_census(
                {n: words(v) for n, v in state_of(reference, names).items()})
            per_step.append({"step": step,
                             "differing_words": sum(difference.values()),
                             "differing_arrays": dict(sorted(difference.items())),
                             "weld_differing_words": sum(weld.values()),
                             "weld_differing_arrays": dict(sorted(weld.items())),
                             "half_differing_words": sum(half.values()),
                             "half_differing_arrays": dict(sorted(half.items())),
                             "reference_subnormals": census["subnormals"],
                             "reference_negative_zeros": census["negative_zeros"]})
            if weld or (difference and difference != half):
                break

    after = state_of(actual, names)
    moved = {name: int(np.count_nonzero(before[name] != words(after[name])))
             for name in names}
    still = sorted(name for name, count in moved.items() if count == 0)
    excluded = tuple(spec.get("uniform_floor_excludes", ()))
    required = (tuple(n for n in names if n not in excluded)
                if value_class == "uniform" else band_must_move(family))
    unmoved_required = sorted(name for name in required if moved[name] == 0)
    complete = len(per_step) == steps
    identical = complete and all(row["differing_words"] == 0 for row in per_step)
    weld_identical = complete and all(row["weld_differing_words"] == 0
                                      for row in per_step)
    half_identical = complete and all(row["half_differing_words"] == 0
                                      for row in per_step)
    # AN INHERITED DIVERGENCE IS PROVED, NEVER ASSERTED: the fused route's word map
    # must equal the separate certified route's, array for array and count for count,
    # on every step. "Both are nonzero" would not distinguish a shared inheritance
    # from two different defects that happen to coincide in size.
    inherited = complete and not identical and all(
        row["differing_arrays"] == row["half_differing_arrays"] for row in per_step)
    drifted = read_only_drift(read_only, family, actual, pml)
    memo_name = _memo_key_for_fused(family, arm, variant_for_spec(family, spec))
    fused_launches = fused_counter.named().get(memo_name, 0)
    # THE TWO COUNTERS MUST AGREE, AND THE PROXY MUST NAME ONLY THIS KERNEL inside the
    # window that holds only the fused launch. A run that launched the certified curl
    # beside the fused kernel added a launch instead of removing two, and
    # `other_kernels` is what says so.
    other_kernels = {k: v for k, v in fused_counter.named().items()
                     if k != memo_name}
    launch_ok = (launched == len(per_step)
                 and fused_launches == (len(per_step) if kernel is None else 0)
                 and not other_kernels)
    case.update({
        "passed": bool(weld_identical and (identical or inherited)
                       and not unmoved_required and not drifted and launch_ok),
        "bit_identical": identical,
        "weld_bit_identical": weld_identical,
        "certified_halves_bit_identical_to_the_array_path": half_identical,
        "divergence_is_inherited_from_a_half": inherited,
        "steps_run": len(per_step),
        "first_divergence": next((row["step"] for row in per_step
                                  if row["differing_words"]), None),
        "first_weld_divergence": next((row["step"] for row in per_step
                                       if row["weld_differing_words"]), None),
        "per_step": per_step,
        "differing_words": per_step[-1]["differing_words"] if per_step else -1,
        "differing_arrays": per_step[-1]["differing_arrays"] if per_step else {},
        "weld_differing_words": (per_step[-1]["weld_differing_words"]
                                 if per_step else -1),
        "half_differing_words": (per_step[-1]["half_differing_words"]
                                 if per_step else -1),
        "arrays_compared": len(names),
        "arrays_that_never_moved": still,
        "movement_floor": {"class": value_class, "required": list(required),
                           "unmet": unmoved_required, "note": MOVEMENT_FLOOR_NOTE},
        "read_only_volumes_checked": sorted(read_only),
        "read_only_volumes_that_moved": drifted,
        "launch_counts": {
            "launcher_reports": launched,
            "memo_proxy_around_the_fused_launch": fused_counter.named(),
            "memo_proxy_around_the_fused_launch_total": fused_counter.total,
            "memo_proxy_whole_step_both_routes": counter.named(),
            "memo_proxy_whole_step_total": counter.total,
            "memo_key_watched": memo_name,
            "other_kernels_launched": other_kernels,
            "kernel_supplied_by_gate": kernel is not None,
            "agree": bool(launch_ok),
            "expected_per_step": 1,
        },
        "seconds": time.time() - started,
    })
    return case


# ---------------------------------------------------------------------------
# The separate composition: what the fusion actually removes, measured
# ---------------------------------------------------------------------------

def leg_separate_control(family: Dict[str, Any], spec: Dict[str, Any], steps: int,
                         arm: str) -> Dict[str, Any]:
    """THREE engines from one seed: array path, separate certified products, fused.

    THE POINT IS NOT THAT THE FUSED ROUTE PASSES -- the product sweep says that. It is
    that the SEPARATE certified route also reproduces the array path on this fixture,
    so a divergence in the fused one is attributable to the WELD rather than to either
    half; and that the fused route makes strictly FEWER device launches, counted
    rather than asserted.
    """
    started = time.time()
    label = f"{family['name']}|{spec['label']}|separate"
    out: Dict[str, Any] = {"family": family["name"], "label": spec["label"],
                           "steps_requested": steps}
    reference, _, ref_pml, _ = family["build"](spec, "uniform", case_rng(label))
    apart, apart_grid, apart_pml, _ = family["build"](spec, "uniform", case_rng(label))
    fused, grid, pml, _ = family["build"](spec, "uniform", case_rng(label))
    names = state_names_for(family, reference)

    arguments = family["resolve"](fused, grid, pml, arm)
    apart_counter, fused_counter = MemoLaunchCounter(), MemoLaunchCounter()
    apart_steps: List[int] = []
    fused_steps: List[int] = []
    for step in range(1, steps + 1):
        array_step(reference, ref_pml)
        before_passes(family, apart, apart_pml)
        with apart_counter:
            family["separate"](apart, apart_grid, apart_pml, arm)
            unfused_passes_after_seam(family, apart, apart_pml)
        before_passes(family, fused, pml)
        with fused_counter:
            family["launch"](fused, arguments)
            unfused_passes(family, fused, pml)
        cp.cuda.runtime.deviceSynchronize()
        apart_steps.append(sum(compare(reference, apart, names).values()))
        fused_steps.append(sum(compare(reference, fused, names).values()))
        if apart_steps[-1] or fused_steps[-1]:
            break
    out.update({
        "passed": bool(len(apart_steps) == steps
                       and not any(apart_steps) and not any(fused_steps)
                       and fused_counter.total < apart_counter.total),
        "steps_run": len(apart_steps),
        "separate_differing_words": apart_steps,
        "fused_differing_words": fused_steps,
        "separate_launches": apart_counter.named(),
        "separate_launch_total": apart_counter.total,
        "fused_launches": fused_counter.named(),
        "fused_launch_total": fused_counter.total,
        "launches_removed_per_step": (
            (apart_counter.total - fused_counter.total) / max(1, len(apart_steps))),
        "seconds": time.time() - started,
    })
    return out


# ---------------------------------------------------------------------------
# The lift: the emitted source IS the certified text
# ---------------------------------------------------------------------------

def leg_lift(family: Dict[str, Any]) -> Dict[str, Any]:
    """Every line of the emitted kernel is certified text or a declared edit.

    THE CLAIM THE MODULES MAKE is that they SPLICE rather than transcribe. This checks
    it from the other side: the fused source is diffed against the two certified
    sources line by line, and every line of the fused body that is not in either
    certified body must be accounted for by :data:`LIFT_EDITS`, the signature, or the
    carried in-seam pass.

    A LINE COUNT ALONE WOULD BE VACUOUS, so the leg also asserts that the certified
    ARITHMETIC lines survive: every ``constitutive_apply`` and ``pml_apply`` call, the
    curl groupings, and the masks.
    """
    from meep_gpu.cuda_kernels import complex_emitter  # noqa: PLC0415

    module = family["module"]
    out: Dict[str, Any] = {"family": family["name"], "arms": {},
                           "variants": [str(v) for v in variants_of(family)]}
    ok = True
    # EVERY (EXPANSION ARM, VARIANT) PAIR, because a product that emits two kernels has
    # two splices to be right about. On a single-variant family this is the loop it
    # always was, with the key spelled ``<arm>`` rather than ``<arm>|None``.
    lift_arms = tuple(family.get("lift_arms")
                      or sorted(complex_emitter.EXPANSIONS))
    for arm in lift_arms:
      for variant in variants_of(family):
        key = arm if variant is None else f"{arm}|{variant}"
        fused = fused_source(family, arm, variant)
        certified_curl, certified_const = certified_halves_source(
            family, arm, variant)
        certified_lines = set(certified_curl.splitlines()) | set(
            certified_const.splitlines())
        novel = [line for line in fused.splitlines()
                 if line not in certified_lines and line.strip()]
        # EVERY CERTIFIED ARITHMETIC STATEMENT MUST SURVIVE THE SPLICE, and a
        # statement that did not must be one a declared LIFT_EDIT names -- with the
        # EXPRESSION it computed still present verbatim in the fused source. That
        # second half is what keeps the edit list from being an excuse: an edit may
        # rename a value or add a store, and may not move a parenthesis.
        markers = tuple(family.get("lift_arithmetic_markers")
                        or ("mul_coefficient_left(dtdx", "cf_sub(", "cshift_"))
        arithmetic = [line.strip() for line in certified_curl.splitlines()
                      if any(marker in line for marker in markers)]
        absent = [line for line in arithmetic if line not in fused]
        declared = {edit["line"].strip(): edit for edit in module.LIFT_EDITS}
        missing = []
        accounted: List[Dict[str, str]] = []
        for line in absent:
            edit = declared.get(line)
            expression = _expression_of(line)
            if edit is None or expression not in fused:
                missing.append(line)
                continue
            accounted.append({"certified_line": line,
                              "declared_became": edit["became"],
                              "expression_still_verbatim": expression})
        block = {
            "variant": variant,
            "fused_sha256": hashlib.sha256(fused.encode("utf-8")).hexdigest(),
            "certified_curl_sha256": hashlib.sha256(
                certified_curl.encode("utf-8")).hexdigest(),
            "certified_constitutive_sha256": hashlib.sha256(
                certified_const.encode("utf-8")).hexdigest(),
            "fused_lines": len(fused.splitlines()),
            "lines_not_in_either_certified_source": len(novel),
            "certified_arithmetic_lines_checked": len(arithmetic),
            "certified_arithmetic_lines_missing": missing,
            "certified_arithmetic_lines_changed_by_a_declared_edit": accounted,
            "declared_lift_edits": len(module.LIFT_EDITS),
            "kernel_names_emitted": sorted(
                complex_emitter.shipped_kernel_names(fused)),
            "kernel_name_expected": kernel_symbol(family, variant),
        }
        block["passed"] = bool(not missing
                               and block["kernel_names_emitted"]
                               == [kernel_symbol(family, variant)]
                               and len(arithmetic) >= 9)
        ok = ok and block["passed"]
        out["arms"][key] = block
    # EVERY EMITTED KERNEL IS DISTINCT. A multi-variant family whose two sources
    # collapsed to one string would pass every block above while shipping one kernel
    # under two names -- and on this product that is the plain tail serving a
    # conductive run, which is a wrong answer on every word rather than a crash.
    digests = {block["fused_sha256"] for block in out["arms"].values()}
    out["distinct_emitted_sources"] = len(digests)
    out["distinct_emitted_sources_expected"] = (
        len(lift_arms) * len(variants_of(family)))
    out["passed"] = bool(
        ok and out["distinct_emitted_sources"]
        == out["distinct_emitted_sources_expected"])
    return out


def _expression_of(statement: str) -> str:
    """The right-hand side one certified statement computes, as text.

    ``cf_store(f, idx, EXPR);`` -> ``EXPR``; ``lhs = EXPR;`` -> ``EXPR``; anything
    else is returned whole. Used to check that an edit which RENAMED a value left the
    expression, its parenthesisation and its operand order untouched -- which is the
    only kind of edit the lift declares.
    """
    text = statement.strip().rstrip(";")
    if text.startswith("cf_store(") and text.endswith(")"):
        inner = text[len("cf_store("):-1]
        parts = inner.split(", ", 2)
        if len(parts) == 3:
            return parts[2]
    if " = " in text:
        return text.split(" = ", 1)[1]
    return text


def _certified_curl_source(family: Dict[str, Any], arm: str) -> str:
    """The certified curl text this family SPLICES -- its own coordinate system AND
    its own sub-step, both read off the family rather than assumed. A leg that lifted
    ``step_B``'s text for a ``step_D`` product would diff the fused source against a
    kernel it does not contain and report every line as novel."""
    from meep_gpu.cuda_kernels import complex_emitter  # noqa: PLC0415

    if family["shape"] == "cylindrical":
        from meep_gpu.cuda_kernels import cylindrical_complex_kernels  # noqa: PLC0415
        return cylindrical_complex_kernels.cylindrical_complex_source(
            family["curl_sub_step"], arm)
    return complex_emitter.complex_source(family["curl_sub_step"], arm)


# ---------------------------------------------------------------------------
# The refusals: rule 3 of this round, measured
# ---------------------------------------------------------------------------

def leg_refusal(family: Dict[str, Any], arm: str) -> Dict[str, Any]:
    """Every configuration each module refuses BY NAME, built and refused.

    A PREDICATE ADMITTING A ROW THE LAUNCH CANNOT SERVE IS THE ONE FAILURE MODE NO
    BIT-COMPARISON CATCHES: the launch would run, produce a number, and be compared
    against an oracle for a configuration nobody welded. So each refusal is a REAL
    configuration here rather than a stub, and the reason is recorded so a reader can
    check that the refusal names what it refuses.
    """
    from meep_gpu.fields import Fields  # noqa: PLC0415
    from meep_gpu.grid import Grid, Mirror  # noqa: PLC0415
    from meep_gpu.pml import PML  # noqa: PLC0415

    covers = family["covers"]
    checks: List[Dict[str, Any]] = []

    def record(name: str, expectation: str, fields, pml, grid, sources=(),
               license=None, policy=None) -> None:
        covered, reason = covers(fields, pml, grid, sources,
                                 LICENCE["verdict"] if license is None else license,
                                 POLICY["name"] if policy is None else policy)
        checks.append({"case": name, "expected": expectation,
                       "covered": bool(covered), "reason": str(reason),
                       "passed": (not covered) if expectation == "refused"
                                 else bool(covered)})

    # A FAMILY MAY SUPPLY ITS OWN CASE TABLE. The residual magnetic families'
    # refusal geometry inverts several of the built-in cases (a REAL-storage run
    # is three of their cells rather than a refusal), so each carries a builder
    # that records its own admissions and refusals through the same ``record``.
    hook = family.get("refusal_cases")
    if hook is not None:
        hook(record)
        passed = all(check["passed"] for check in checks)
        return {"family": family["name"], "passed": passed, "checks": checks,
                "admitted_control_present": any(
                    c["expected"] == "covered" and c["passed"] for c in checks)}

    #: Does this family's product REQUIRE an inert layer rather than an active one?
    #: Every clause below that mentions the absorber has two answers, and they are
    #: mirror images rather than variations: the four PML products refuse a run with no
    #: active layer and this one refuses a run that HAS one, which is exactly what
    #: keeps four candidates on one seam disjoint.
    no_pml = bool(family.get("requires_inert_layer"))

    def cartesian(**kwargs):
        grid = Grid(resolution=1.0, cell_size=(9.0, 10.0, 11.0),
                    boundaries=kwargs.pop("boundaries",
                                          ("periodic", "periodic", "periodic")),
                    xp=cp, courant=0.5, **kwargs)
        fields = Fields(grid=grid,
                        force_complex_fields=kwargs.get("_complex", True))
        fields.enable_field_storage()
        if not no_pml:
            fields.enable_pml_storage()
        return fields, (None if no_pml else PML(grid=grid, thickness=2)), grid

    # 1. REAL FLOAT32 STORAGE. Both products are complex-only; a real run belongs to
    #    the shipped real pair, and admitting one here would be two products claiming
    #    one seam -- which install_fused_pairs answers by leaving it UNFUSED.
    grid = Grid(resolution=1.0, cell_size=(9.0, 10.0, 11.0),
                boundaries=("periodic",) * 3, xp=cp, courant=0.5)
    real_fields = Fields(grid=grid)
    real_fields.enable_field_storage()
    if not no_pml:
        real_fields.enable_pml_storage()
    record("real_storage", "refused", real_fields,
           None if no_pml else PML(grid=grid, thickness=2), grid)

    # 2. THE ABSORBER CLAUSE, IN WHICHEVER DIRECTION THIS PRODUCT MAKES IT. The four
    #    split-field products refuse a run with no active layer; the no-absorber one
    #    ADMITS exactly that and refuses a run that has one. Both directions are
    #    recorded here rather than one, because a predicate that refused BOTH would
    #    pass a one-sided check while serving nothing.
    inert_fields, _, inert_grid = cartesian()
    record("no_active_absorber", "covered" if no_pml else "refused",
           inert_fields, None, inert_grid)
    if no_pml:
        #    THE MIRROR, and it is the whole disjointness proof: an ACTIVE layer must
        #    be refused BY NAME here, or this product and the three PML ones would
        #    co-admit and install_fused_pairs would leave the seam UNFUSED naming all
        #    four.
        active_fields = Fields(grid=inert_grid, force_complex_fields=True)
        active_fields.enable_field_storage()
        active_fields.enable_pml_storage()
        record("an_active_absorber", "refused", active_fields,
               PML(grid=inert_grid, thickness=2), inert_grid)
        #    AND PML STORAGE MODE WITH AN INERT LAYER. `enable_pml_storage` is a
        #    one-way switch that also changes what `get_H` returns, so a curl would
        #    difference a frozen H. The array path has the same hazard; this refuses
        #    the configuration rather than reproducing it.
        stored = Fields(grid=inert_grid, force_complex_fields=True)
        stored.enable_field_storage()
        stored.enable_pml_storage()
        record("pml_storage_mode_with_an_inert_layer", "refused", stored, None,
               inert_grid)
        #    AND A METALLIC WALL. `zero_metal_D` (driver.py:3310) runs inside this
        #    seam on a walled run and this pair does not carry it -- which is the
        #    clause that makes its REPLACES a contiguous run of two. There is no
        #    walled fixture in its sweep, by design; this is where a walled grid is
        #    exercised, as the refusal it is.
        for axis, kinds in (("x", ("metallic", "periodic", "periodic")),
                            ("y", ("periodic", "metallic", "periodic")),
                            ("z", ("periodic", "periodic", "metallic")),
                            ("xyz", ("metallic", "metallic", "metallic"))):
            walled, walled_pml, walled_grid = cartesian(boundaries=kinds)
            record(f"metallic_wall_{axis}", "refused", walled, walled_pml,
                   walled_grid)

    # 3. A FOLDED GRID. The two mirror fills run inside this seam and no product here
    #    carries them; the refusal is what makes REPLACES honest.
    folded_grid = Grid(resolution=1.0, cell_size=(9.0, 10.0, 11.0),
                       boundaries=("periodic",) * 3, symmetry=(Mirror("Y", 1),),
                       xp=cp, courant=0.5)
    folded = Fields(grid=folded_grid, force_complex_fields=True)
    folded.enable_field_storage()
    if not no_pml:
        folded.enable_pml_storage()
    record("folded", "refused", folded,
           None if no_pml
           else PML(grid=folded_grid, thickness=((2, 2), (0, 2), (2, 2))),
           folded_grid)

    # 4. AN UNDECLARED SOURCE SET. Ignorance is never an empty set: Fields does not
    #    hold the source list, so a predicate that inferred "no sources" from not
    #    being told would be over-covering.
    live, live_pml, live_grid = cartesian()
    record("sources_undeclared", "refused", live, live_pml, live_grid, None)

    # 5. AN IN-SEAM SOURCE THAT PUBLISHES NO INDEX. The repair has nothing to save
    #    and restore, so the deposit cannot be carried. THE FIELD TYPE IS THIS SEAM'S:
    #    a magnetic source is not in the electric seam at all and an electric one is
    #    not in the magnetic seam, so a leg that used the wrong letter would record a
    #    predicate ADMITTING a harmless source as if it had refused a fatal one.
    class _Opaque:
        field_type = family["seam"]

    record(f"in_seam_{family['seam']}_source_without_an_index", "refused", live,
           live_pml, live_grid, (_Opaque(),))

    # 5b. A CONDUCTIVITY, ON THE ELECTRIC SEAM ONLY. The driver deposits an electric
    #     source through `_inject_electric_through_conductivity` (driver.py:3305) when
    #     one is present, which rescales the increment by `condinv`; the repair has no
    #     verdict on that route. BOTH COMPLEX HALVES ADMIT A CONDUCTIVITY BY NAME, so
    #     unlike the real electric pair this refusal is the PRODUCT'S OWN and is the
    #     only thing standing between it and that route.
    if family["seam"] == "D" and not no_pml:
        class _Conductive:
            def __getattr__(self, item):
                return getattr(live, item)

            has_conductivity = True

        record("engine_conductivity", "refused", _Conductive(), live_pml, live_grid,
               ())
    elif no_pml:
        # 5c. THE MIRROR CLAUSE, AND IT IS AN ADMISSION RATHER THAN A REFUSAL. That
        #     driver route runs only `if electric and self.fields.has_conductivity`
        #     (driver.py:3304), and this predicate refuses every electric source
        #     outright, so the route is unreachable and refusing a conductivity would
        #     be a refusal with no line behind it -- one that would cost the WHOLE
        #     board cell, since every corpus row there records has_conductivity True.
        #     Built as a real lossy engine rather than a flag proxy: the arm this
        #     admission selects is read from `condfac_for`, which a flag does not set.
        conductive_spec = next(s for s in family["specs"] if s["conductive"])
        lossy, lossy_grid, lossy_pml, _ = family["build"](
            conductive_spec, "uniform", np.random.default_rng(13))
        record("engine_conductivity_admitted", "covered", lossy, lossy_pml,
               lossy_grid, ())
        checks.append({
            "case": "engine_conductivity_selects_the_conductive_arm",
            "expected": "conductive",
            "covered": True,
            "reason": family["module"].no_pml_complex_curl_arm(lossy, "step_D"),
            "passed": family["module"].no_pml_complex_curl_arm(
                lossy, "step_D") == "conductive"})

    # 6. NO EXPANSION LICENCE. The arm is a measured platform fact and both arms
    #    compile; a guessed one is a wrong ANSWER rather than a crash.
    record("no_licence", "refused", live, live_pml, live_grid, (), license=False)

    # 7. THE POSITIVE CONTROL, AND IT IS THIS FAMILY'S OWN FIXTURE. A refusal leg
    #    with no admitted case cannot tell a predicate that refuses correctly from
    #    one that refuses everything -- and the control has to be built by the same
    #    builder the product sweep uses, or it tests the other family's predicate.
    #    Measured 2026-08-30: a Cartesian control handed to the Dcyl predicate is
    #    refused for "not a cylindrical (Dcyl) grid", which is the RIGHT answer to
    #    the WRONG question and failed the leg.
    admitted, admitted_grid, admitted_pml, _ = family["build"](
        family["specs"][0], "uniform", np.random.default_rng(11))
    record("the_fixture_this_gate_sweeps", "covered", admitted, admitted_pml,
           admitted_grid, ())

    # 8. THE COORDINATE SYSTEM EACH PRODUCT IS NOT FOR -- the clause that keeps the
    #    two complex products from both claiming one seam.
    cyl_grid = Grid(resolution=1.0, cell_size=(9.0, 0.0, 11.0), cylindrical=True,
                    m=2, boundaries={"z": "metallic"}, courant=0.5, xp=cp)
    cyl = Fields(grid=cyl_grid, force_complex_fields=True)
    cyl.enable_field_storage()
    if not no_pml:
        cyl.enable_pml_storage()
    cyl_pml = None if no_pml else PML(grid=cyl_grid, thickness={"x": (0, 2), "z": 2})
    if family["shape"] == "cartesian":
        record("cylindrical_grid", "refused", cyl, cyl_pml, cyl_grid, ())
    else:
        record("cartesian_grid", "refused", live, live_pml, live_grid, ())
        # m = 0 SPLITS ON STORAGE ALONE since 2026-09-04. The complex-storage m = 0
        # triple is this family's third m_class arm and is ADMITTED -- built by this
        # family's own builder, so the electric product's material and the seam's
        # own fixture shape are the ones the sweep uses; the REAL-storage m = 0
        # triple is the real pair's and is refused by name, so the two cylindrical
        # products can never both admit a triple.
        m0_spec = next(spec for spec in family["specs"] if int(spec["m"]) == 0)
        m0, m0_grid, m0_pml, _ = family["build"](m0_spec, "uniform",
                                                 np.random.default_rng(13))
        record("cylindrical_m_zero_complex_storage", "covered", m0, m0_pml, m0_grid,
               ())
        real_m0_grid = Grid(resolution=1.0, cell_size=(9.0, 0.0, 11.0),
                            cylindrical=True, m=0, boundaries={"z": "metallic"},
                            courant=0.5, xp=cp)
        real_m0 = Fields(grid=real_m0_grid)
        real_m0.enable_field_storage()
        real_m0.enable_pml_storage()
        record("cylindrical_m_zero_real_storage", "refused", real_m0,
               PML(grid=real_m0_grid, thickness={"x": (0, 2), "z": 2}),
               real_m0_grid, ())

    passed = all(check["passed"] for check in checks)
    return {"family": family["name"], "passed": passed, "checks": checks,
            "admitted_control_present": any(
                c["expected"] == "covered" and c["passed"] for c in checks)}


# ---------------------------------------------------------------------------
# The deposit: an in-seam source, carried
# ---------------------------------------------------------------------------

class _PointSource:
    """A point source at a declared index -- the shape the repair can invert.

    NOT A MOCK OF THE ENGINE'S SOURCE CLASS: it publishes exactly the three attributes
    ``deposit_repair._deposit_index`` reads and injects exactly the way the driver
    does (``source.inject(fields, time)``), so the repair sees the same thing it sees
    in a real run.

    ``field_type`` IS AN ARGUMENT AND NOT A CLASS CONSTANT, because it is what decides
    which seam the source is in: ``deposit_repair._in_seam_indexed`` reads it, and a
    magnetic source handed to the electric seam is simply not in it -- so a leg built
    on the wrong letter would run a clean seam and report the null control as failing
    to diverge, which reads as a defect in the repair.
    """

    def __init__(self, field_type: str, component: str,
                 index: Tuple[int, int, int], amplitude: complex) -> None:
        self.field_type = field_type
        self.component = component
        self._point_ix, self._point_iy, self._point_iz = index
        self.amplitude = amplitude

    def inject(self, fields, when: float) -> None:
        value = self.amplitude * np.float32(np.cos(3.1 * float(when)))
        volume = getattr(fields, self.component)
        volume[self._point_ix, self._point_iy, self._point_iz] += (
            volume.dtype.type(value))


class _SheetSource:
    """A MULTI-CELL deposit publishing index ARRAYS with per-cell amplitudes.

    THE FLAG-FLIP PREMISE, CARRIED ONTO THE DEVICE LEG: the 2026-09-02 residue
    round flipped two CARRIES_DEPOSIT_REPAIR flags after re-measuring that the
    lifted eigenmode sources deposit through equivalent-current SHEETS whose
    source objects publish ``_point_ix/_point_iy/_point_iz`` as flat index
    columns with per-cell amps (``sources._setup_source_points``; the
    ``amp_func`` shape). This class is that shape reduced to what the repair
    reads: three parallel index arrays of arbitrary length and an amplitude PER
    CELL, injected as ``host scalar x setup-time per-cell amps`` exactly as the
    engine's own inject paths do. ``deposit_repair.save/apply`` make no
    point-cardinality assumption -- which is the claim, and what the unrepaired
    null control diverging proves was exercised.
    """

    def __init__(self, field_type: str, component: str,
                 indices: Tuple[Any, Any, Any], amplitudes: Any) -> None:
        self.field_type = field_type
        self.component = component
        self._point_ix, self._point_iy, self._point_iz = (
            cp.asarray(np.asarray(column, dtype=np.int64))
            for column in indices)
        self.amplitudes = amplitudes  # host array, one value per cell

    def inject(self, fields, when: float) -> None:
        volume = getattr(fields, self.component)
        values = np.asarray(self.amplitudes) * np.float32(
            np.cos(3.1 * float(when)))
        volume[self._point_ix, self._point_iy, self._point_iz] += cp.asarray(
            values.astype(volume.dtype))


def _index_as_list(column: Any) -> Any:
    """A published index column as JSON-ready data: an int stays an int, an
    array (host or device) becomes a list."""
    if isinstance(column, (int, np.integer)):
        return int(column)
    return np.asarray(to_host(column)).tolist()


def _sheet_deposit_for(family: Dict[str, Any], shape: Tuple[int, ...],
                       is_complex: bool) -> _SheetSource:
    """One sheet deposit for this seam, spanning the longest axis of the grid.

    The cells run 0..n-1 along the longest axis at fixed transverse index 1
    where the grid can hold it -- so on a folded fixture the sheet CROSSES the
    rows the two fills read from and the repair must image it, and on a walled
    one it touches the cleared plane. Amplitudes are distinct per cell (a
    constant sheet would leave an index permutation invisible).
    """
    axis = int(np.argmax(shape))
    count = max(2, min(int(shape[axis]), 8))
    columns = [np.full(count, min(1, extent - 1), dtype=np.int64)
               for extent in shape]
    columns[axis] = np.arange(count, dtype=np.int64)
    ramp = 1.0 + 0.1 * np.arange(count)
    if is_complex:
        amplitudes = (0.29 + 0.13j) * ramp * np.exp(0.21j * np.arange(count))
        amplitudes = amplitudes.astype(np.complex64)
    else:
        amplitudes = (np.float32(0.29) * ramp).astype(np.float32)
    component = "By" if family["seam"] == "B" else "Dy"
    return _SheetSource(family["seam"], component, tuple(columns), amplitudes)


def leg_deposit(family: Dict[str, Any], spec: Dict[str, Any], steps: int, arm: str,
                repaired: bool = True) -> Dict[str, Any]:
    """A real in-seam source in THIS family's seam, carried by the two repair plans.

    THE DRIVER INJECTS BETWEEN THE TWO HALVES -- a magnetic source at driver.py:3293
    and an electric one at :3305/:3308 -- so the fused launch computes the constitutive
    half against a pre-injection flux density. The repair saves the deposit's closure
    immediately before the launch and recomputes the constitutive result after the
    driver has injected and cleared walls, which is the only point at which the
    injected field is final and the launch's pre-injection accumulation is still known.

    THE INJECTION INSTANT IS PART OF THE SEAM AND NOT A DETAIL. The electric seam
    injects at ``when + 0.5*dt`` (driver.py:3303) and the magnetic one at ``when``; a
    leg that used one instant for both would compare two different runs and could pass
    only by cancelling its own error.

    ``repaired=False`` IS THE NULL CONTROL AND MUST DIVERGE. A leg whose control also
    passed would mean the repair was doing nothing and the seam had no deposit in it.
    """
    started = time.time()
    seam = family["seam"]
    span = seam_span(family)
    label = f"{family['name']}|{spec['label']}|deposit"
    out: Dict[str, Any] = {"family": family["name"], "label": spec["label"],
                           "seam": seam, "repaired": repaired,
                           "steps_requested": steps}
    reference, ref_grid, ref_pml, _ = family["build"](spec, "uniform",
                                                      case_rng(label))
    actual, grid, pml, _ = family["build"](spec, "uniform", case_rng(label))
    names = state_names_for(family, reference)
    shape = tuple(int(n) for n in grid.shape)
    # THREE SOURCES, AND THE SPLIT IS THE POINT.
    #
    # ONE LANDS WHERE THE WALL CLEAR WRITES. That is the one asymmetry between the
    # fused route and the driver's -- the kernel clears BEFORE the injection and the
    # driver AFTER it -- and a deposit sweep that avoided the wall could not see it.
    # The component is THIS SEAM'S: on the B side the table is the DIAGONAL, so Bx is
    # wiped at stored cell 0 of a walled x axis; on the D side it is the OFF-DIAGONAL
    # complement, so the component the x wall clears is Dy, never Dx.
    #
    # TWO LAND WHERE NO WALL CLEARS, and they are what makes the NULL CONTROL able to
    # diverge at all. Measured 2026-08-30 before this note was written: with both
    # deposits on cleared cells the un-repaired control was BIT-IDENTICAL, because the
    # driver's own wall clear destroys the deposit before the constitutive half ever
    # reads it -- a vacuous control reporting a pass. A leg whose control cannot
    # diverge measures nothing, so the fixture carries deposits the clear does not
    # reach.
    off_x = min(1, shape[0] - 1)
    off_y = min(1, shape[1] - 1)
    off_z = min(1, shape[2] - 1)
    on_wall, off_wall_a, off_wall_b = (
        ("Bx", "Bx", "Bz") if seam == "B" else ("Dy", "Dy", "Dz"))
    # AMPLITUDES FOLLOW THE STORAGE, exactly as leg_source_refusal's do: a
    # complex amplitude into a float32 volume is a cast error, not a deposit --
    # and the residue round put REAL families on this leg for the first time.
    is_complex = bool(np.iscomplexobj(to_host(getattr(actual, "Bx"))))

    def _amp(re, im):
        return complex(re, im) if is_complex else float(re)

    sources = [_PointSource(seam, on_wall, (0, 0, min(2, shape[2] - 1)),
                            _amp(0.37, 0.11)),
               _PointSource(seam, off_wall_a, (off_x, off_y, off_z),
                            _amp(-0.19, 0.27)),
               _PointSource(seam, off_wall_b, (off_x, off_y, off_z),
                            _amp(-0.23, 0.4))]
    # THE SHEET DEPOSIT, where the family declares one: the flag-flip premise
    # (index ARRAYS, per-cell amps) carried as a device measurement rather
    # than a host argument. Recorded per leg so the artifact shows the
    # multi-cell footprint that ran.
    if family.get("deposit_sheet"):
        sources.append(_sheet_deposit_for(family, shape, is_complex))
        out["sheet_deposit_cells"] = int(
            np.asarray(sources[-1].amplitudes).size)
    covered, reason = family["covers"](actual, pml, grid, sources, LICENCE["verdict"],
                                       POLICY["name"])
    out["predicate"] = {"covered": bool(covered), "reason": reason}
    if not covered:
        out["passed"] = False
        out["why"] = f"the predicate refused the deposit fixture: {reason}"
        return out

    arguments = family["resolve"](actual, grid, pml, arm)
    ok, why = deposit_repair.repairable(actual, seam, pml)
    out["repairable"] = {"ok": bool(ok), "reasons": list(why)}
    if not ok:
        out["passed"] = False
        out["why"] = f"deposit_repair refused this seam: {why}"
        return out

    # The three in-seam passes, and the driver's tail after the constitutive half.
    in_seam = tuple(name for name in span if name != span[0])
    tail = DRIVER_ORDER[DRIVER_ORDER.index(span[-1]) + 1:]
    out["injection_offset_in_dt"] = 0.0 if seam == "B" else 0.5
    per_step: List[int] = []
    dt = float(grid.dt)
    for step in range(1, steps + 1):
        when = (step - 1) * dt + (0.0 if seam == "B" else 0.5 * dt)
        # The oracle: the driver's own order, injection included.
        before_passes(family, reference, ref_pml)
        run_pass(span[0], reference, ref_pml)
        for source in sources:
            source.inject(reference, when)
        for name in in_seam:
            run_pass(name, reference, ref_pml)
        for name in tail:
            run_pass(name, reference, ref_pml)

        # The subject: save, fused launch, inject, the in-seam passes, repair.
        before_passes(family, actual, pml)
        saved = deposit_repair.save(actual, sources, seam, pml) if repaired else {}
        family["launch"](actual, arguments)
        for source in sources:
            source.inject(actual, when)
        for name in in_seam[:-1]:
            run_pass(name, actual, pml)
        if repaired:
            deposit_repair.apply(actual, pml, sources, seam, saved)
        for name in tail:
            run_pass(name, actual, pml)
        cp.cuda.runtime.deviceSynchronize()
        per_step.append(sum(compare(reference, actual, names).values()))
        if per_step[-1]:
            break

    identical = len(per_step) == steps and not any(per_step)
    out.update({
        "passed": bool(identical if repaired else not identical),
        "bit_identical": identical,
        "steps_run": len(per_step),
        "first_divergence": next((i + 1 for i, n in enumerate(per_step) if n), None),
        "differing_words_per_step": per_step,
        "deposit_indices": [
            [_index_as_list(s._point_ix), _index_as_list(s._point_iy),  # noqa: SLF001
             _index_as_list(s._point_iz)]  # noqa: SLF001
            for s in sources],
        "seconds": time.time() - started,
    })
    return out


# ---------------------------------------------------------------------------
# The deposit NOT carried: the refusal, and the divergence behind it
# ---------------------------------------------------------------------------

def leg_source_refusal(family: Dict[str, Any], spec: Dict[str, Any], steps: int,
                       arm: str) -> Dict[str, Any]:
    """What :func:`leg_deposit` proves for a product that DOES carry the repair,
    proved for one that deliberately does not.

    ``CARRIES_DEPOSIT_REPAIR = False`` IS A MEASUREMENT ON THIS PRODUCT'S BOARD CELL --
    all four of its corpus rows declare a MAGNETIC source and nothing in the D seam --
    and the flag is only SOUND if two things hold, both of which are measured here
    rather than argued:

    1. **The predicate refuses every in-seam electric source**, so no configuration
       with a deposit can reach ``_install_fused_pair``, where the ``NoopPlan`` branch
       would put a cheap sentinel over a launch that consumed a pre-injection D.
    2. **That refusal is not decorative.** A fused launch spanning the deposit really
       does diverge from the driver's own order -- measured by running exactly that,
       with the same sources, and requiring the words to differ. Without this half,
       clause 1 would be a clause with no consequence behind it, which is the vacuity
       a deposit leg's ``repaired=False`` control exists to prevent on the products
       that carry one.

    AND A MAGNETIC SOURCE IS ADMITTED, which is what makes the product serve its cell
    at all: the driver injects it one seam earlier (driver.py:3293) and it is not in
    this seam. Recorded beside the refusal so a reader can see the predicate
    distinguishing rather than refusing everything.
    """
    started = time.time()
    seam = family["seam"]
    span = seam_span(family)
    label = f"{family['name']}|{spec['label']}|source_refusal"
    out: Dict[str, Any] = {"family": family["name"], "label": spec["label"],
                           "seam": seam,
                           "carries_deposit_repair": bool(
                               family["module"].CARRIES_DEPOSIT_REPAIR),
                           "steps_requested": steps}
    reference, ref_grid, ref_pml, _ = family["build"](spec, "uniform",
                                                      case_rng(label))
    actual, grid, pml, _ = family["build"](spec, "uniform", case_rng(label))
    names = state_names_for(family, reference)
    shape = tuple(int(n) for n in grid.shape)
    off = tuple(min(1, n - 1) for n in shape)
    # THE SEAM'S OWN LETTER DECIDES WHICH SOURCES ARE IN IT -- this leg served
    # the D seam alone until the residual B-side families landed, and a B-side
    # run of the electric spelling would have run a CLEAN seam and reported the
    # divergence floor as a defect. Amplitudes follow the storage: a complex
    # amplitude into a float32 volume is a cast error, not a deposit.
    is_complex = bool(np.iscomplexobj(to_host(getattr(actual, "Bx"))))
    def _amp(re, im):
        return complex(re, im) if is_complex else float(re)
    if family["seam"] == "B":
        in_seam = [_PointSource("B", "Bx", (0, 0, min(2, shape[2] - 1)),
                                _amp(0.37, 0.11)),
                   _PointSource("B", "By", off, _amp(-0.23, 0.4))]
        other = [_PointSource("D", "Dy", off, _amp(0.31, -0.17))]
        in_seam_needle, in_seam_key, other_key = (
            "is magnetic", "magnetic_source", "electric_source")
    else:
        in_seam = [_PointSource("D", "Dy", (0, 0, min(2, shape[2] - 1)),
                                _amp(0.37, 0.11)),
                   _PointSource("D", "Dz", off, _amp(-0.23, 0.4))]
        other = [_PointSource("B", "Bx", off, _amp(0.31, -0.17))]
        in_seam_needle, in_seam_key, other_key = (
            "is electric", "electric_source", "magnetic_source")
    electric = in_seam  # the injected set below; the name is historical

    covered, reason = family["covers"](actual, pml, grid, in_seam,
                                       LICENCE["verdict"], POLICY["name"])
    out[in_seam_key] = {"covered": bool(covered), "reason": str(reason),
                        "passed": not covered and in_seam_needle in str(reason)}
    covered, reason = family["covers"](actual, pml, grid, other,
                                       LICENCE["verdict"], POLICY["name"])
    out[other_key] = {"covered": bool(covered), "reason": str(reason),
                      "passed": bool(covered)}
    covered, reason = family["covers"](actual, pml, grid, None,
                                       LICENCE["verdict"], POLICY["name"])
    out["undeclared_source_set"] = {"covered": bool(covered), "reason": str(reason),
                                    "passed": not covered}

    # THE CONSEQUENCE. The fused launch is run ANYWAY, against the same sources the
    # predicate just refused, and the words are required to differ from the driver's
    # own order. `family["launch"]` is the shipped launcher, so this is the real
    # kernel spanning the real deposit -- not a stand-in.
    arguments = family["resolve"](actual, grid, pml, arm)
    in_seam = tuple(name for name in span if name != span[0] and name != span[-1])
    tail = DRIVER_ORDER[DRIVER_ORDER.index(span[-1]) + 1:]
    dt = float(grid.dt)
    per_step: List[int] = []
    for step in range(1, steps + 1):
        when = (step - 1) * dt + 0.5 * dt
        before_passes(family, reference, ref_pml)
        run_pass(span[0], reference, ref_pml)
        for source in electric:
            source.inject(reference, when)
        for name in in_seam:
            run_pass(name, reference, ref_pml)
        run_pass(span[-1], reference, ref_pml)
        for name in tail:
            run_pass(name, reference, ref_pml)

        before_passes(family, actual, pml)
        family["launch"](actual, arguments)
        for source in electric:
            source.inject(actual, when)
        for name in in_seam:
            run_pass(name, actual, pml)
        for name in tail:
            run_pass(name, actual, pml)
        cp.cuda.runtime.deviceSynchronize()
        per_step.append(sum(compare(reference, actual, names).values()))
        if per_step[-1]:
            break

    diverged = any(per_step)
    out.update({
        "unrepaired_span_diverges": diverged,
        "unrepaired_first_divergence": next(
            (i + 1 for i, n in enumerate(per_step) if n), None),
        "unrepaired_differing_words_per_step": per_step,
        "deposit_indices": [[s._point_ix, s._point_iy, s._point_iz]  # noqa: SLF001
                            for s in electric],
        "passed": bool(out[in_seam_key]["passed"]
                       and out[other_key]["passed"]
                       and out["undeclared_source_set"]["passed"]
                       and diverged
                       and not family["module"].CARRIES_DEPOSIT_REPAIR),
        "seconds": time.time() - started,
    })
    return out


# ---------------------------------------------------------------------------
# The mutations
# ---------------------------------------------------------------------------

def needle(source: str, old: str, new: str, count: int = 1) -> Tuple[str, int]:
    """Replace ``old`` with ``new``, reporting how many sites were hit.

    A mutation that matched NOTHING is not a mutation, and a leg that reported
    "uncaught" for one would be reporting on its own needle rather than on the gate.
    """
    hits = source.count(old)
    return (source.replace(old, new, count) if hits else source), hits


def compile_source(source: str, name: str, options: Sequence[str]):
    return cp.RawKernel(source, name, options=tuple(options))


### THE MUTATIONS, EACH WITH AN EXPECTATION AND A LIVENESS
#
# "EVERY MUTATION MUST BE CAUGHT EVERYWHERE" IS FALSE BY CONSTRUCTION HERE, and the
# first full run measured exactly which legs it is false for: a wall mutation on an
# unwalled grid changes no byte, an arm mutation at k = 0 reaches no complex product,
# and the seam's own register hand-off is a PURE OPTIMISATION whose whole claim is
# that reading the word back from global memory instead is the identity. So each leg
# declares one of three expectations, and the leg's verdict is the match:
#
#   ``caught``      -- must diverge on every fixture it is scored on;
#   ``caught_where_live`` -- must diverge on every fixture ``live`` admits, must be
#                     live on at least one, and must be INERT where it is not live
#                     (a wall mutation that moved a word on an unwalled grid would be
#                     writing a plane nobody asked for);
#   ``inert``       -- must diverge NOWHERE. These are the null controls, and they
#                     are the strongest statements in the table rather than the
#                     weakest: they say the edit this product makes to the certified
#                     text changes no arithmetic at all.
#
# ``live`` takes (family, spec) and is spelled per leg. A leg with no live fixture in
# the scored set is a FAILURE, not a pass: it would be a defect nothing could catch.


def _spec_has_a_wall(family: Dict[str, Any], spec: Dict[str, Any]) -> bool:
    if "boundaries" in spec:
        return any(b == "metallic" for b in spec["boundaries"])
    return spec["z_kind"] == "metallic"


def _spec_walls(family: Dict[str, Any], spec: Dict[str, Any]) -> Tuple[bool, ...]:
    """The three walled-axis flags this fixture drives, from the spec alone.

    On a Dcyl grid axis 0 is the r axis: ``zero_metal_axes`` reads
    ``grid.is_metallic`` there, and the cylindrical axis is not a metallic wall, so
    only z can carry one.
    """
    if "boundaries" in spec:
        # Every Cartesian-shaped family, the residual real ones included: the
        # walls are the metallic boundaries the spec declares.
        return tuple(b == "metallic" for b in spec["boundaries"])
    return (False, False, spec["z_kind"] == "metallic")


def _arm_is_live(family: Dict[str, Any], spec: Dict[str, Any]) -> bool:
    """Is there a COMPLEX-BY-COMPLEX product for the expansion arm to change?

    THE TWO FAMILIES ANSWER DIFFERENTLY AND THE FIRST RUN MEASURED IT. The arms
    differ only on the full complex product; a real coefficient times a complex
    field (``mul_coefficient_left``, which is all ``constitutive_apply`` does) is
    the same bits under either.

    * Cartesian complex: the only full product in this seam is the BLOCH ROTATION on
      the wrapped lane, so the leg is live exactly where a phase exists. Measured:
      UNCAUGHT on every k = 0 fixture and caught on every phased one.
    * Dcyl: NEVER, and this took two runs to establish rather than one reading. The
      family carries no phase -- the predicate refuses one -- but the curl does carry
      full complex products: ``mul_complex_left(cf_load(imr0, i), f_1)``, the i*m/r
      coupling, and the |m| = 1 axis increment's ``mul_complex_left(q, e1)``. The
      first run declared the leg NEVER LIVE, the second declared it live everywhere,
      and the device answered UNCAUGHT on all four fixtures both times. The reason is
      in the OPERANDS, measured off the shipped builders on 2026-08-30: every one of
      those left operands is PURE IMAGINARY -- ``imr_coefficient_row`` returns
      ``0 - i*m/r`` (m = 1 gives ``[0-1j, 0-0.5j, 0-0.25j]``) and
      ``axis_increment_scalars`` returns ``(0.0, m/2)``. With ``c.re`` an exact zero
      each output word is a SINGLE product rather than a sum of two, so there is no
      grouping for the arm to change and ``fma(0, x, -(c.im*z.im))`` is the naive
      expression to the bit. The leg stays, as a null control on the claim that this
      family's arm argument is inert.
    """
    if family["shape"] == "cylindrical":
        return False
    if family.get("licensed", True) is False:
        # A REAL family binds no arm at all; its launcher ignores the argument
        # entirely, so the leg is a null control there, never live.
        return False
    return any(float(k) != 0.0 for k in spec.get("k_point", ()))


def _walls_and_codes_can_differ(family: Dict[str, Any],
                                spec: Dict[str, Any]) -> bool:
    """Can ``zero_metal_axes`` and the curl's boundary codes disagree here?

    THE ANSWER IS DIFFERENT ON THE TWO FAMILIES, and the first run measured it after
    this function had declared one answer for both.

    * Cartesian complex: NO. ``zero_metal_axes`` is ``is_metallic and not
      is_mirrored`` (stepping.py:2284-2286) and the product refuses a folded grid, so
      on every admitted row the two readings coincide and the leg is inert.
    * Dcyl at |m| = 1: YES, and on the r axis specifically.
      ``cylindrical_boundary_codes`` maps the cylindrical axis onto ``BC_METALLIC``
      -- that is the ghost rule the curl's stencil needs there -- while
      ``zero_metal_axes`` reads ``grid.is_metallic``, which the r axis is not. So
      reading the wall plan off the boundary codes wipes a plane of Br the array path
      keeps.
    * Dcyl at |m| >= 2: NO, and the reason is the certified axis tail rather than the
      wall plan. ``zero_rows`` is 1 to 3 there (measured off the shipped builder:
      m = 2 gives 2, m = 3 gives 3, and ``accurate_fields_near_cylorigin`` gives 1),
      so ``stepping._cylindrical_axis_zero_B`` has ALREADY written exact zero into
      row 0 of all three components before the wall clear runs. Wiping a row that is
      already zero changes no word. At |m| = 1 the tail does not run at all
      (``m_class`` is 1 and ``zero_rows`` is 0), which is what leaves the leg live.
    * Dcyl at m = 0 (2026-09-04): DIFFERENT PER SEAM, and the first device run of the
      m = 0 arm measured it. On the B seam the m = 0 tail zeroes Br on the axis row
      and a spurious r wall wipes exactly Br there (the DIAGONAL table) -- a row
      already zero, so the leg is inert. On the D seam the tail zeroes Dp and
      POST-ADDS into Dz on the axis row, and the spurious r wall wipes Dp AND Dz
      (the OFF-DIAGONAL table), so the post-added Dz is lost and the leg is live.
      Measured: NOT-LIVE-BUT-DIVERGED on ``m0_z_metallic`` for the electric family
      when this function said "not live" for both seams.
    """
    if family["shape"] != "cylindrical":
        return False
    m = int(spec["m"])
    if abs(m) == 1:
        return True
    if m == 0:
        return family["seam"] == "D"
    return False


def _codes_all_periodic_is_live(family: Dict[str, Any],
                                spec: Dict[str, Any]) -> bool:
    """Where does dropping the metallic ghost rule from the curl's stencil MOVE a word?

    THREE ANSWERS, AND THE THIRD WAS MEASURED BY THE FIRST DEVICE RUN OF THE ELECTRIC
    FAMILIES rather than reasoned to. This leg used to read "live on every Dcyl fixture,
    wall or no wall", which is true of the B seam and false of the D seam, and it scored
    UNCAUGHT on ``m3_accurate`` -- correctly, because the mutation there really does
    change no word.

    * **Cartesian, either seam** -- live exactly where an axis is walled. With no
      metallic code in the table there is nothing for the mutation to drop.
    * **Dcyl, MAGNETIC seam** -- live everywhere. ``cylindrical_complex_boundary_codes``
      maps the r axis onto ``BC_METALLIC`` on every Dcyl grid, and ``step_B`` reads it
      through ``cshift_up``, whose bc branch is taken at the TOP row (``ia + 1 == na``,
      complex_emitter's ``_TAIL``). No axis tail ever writes that row, so the change
      survives at every |m|.
    * **Dcyl, ELECTRIC seam** -- live only where z is walled or |m| = 1. ``step_D``
      reads bc_x through ``cshift_dn``, whose bc branch is taken at ROW 0
      (``ia > 0`` fails), and both of its own ``bc_x == BC_METALLIC`` masks are
      ``i == 0`` -- so the whole r-axis change is confined to row 0. At |m| >= 2
      ``stepping._cylindrical_axis_zero_D`` then zeroes all three components and their
      three ``fu`` volumes on rows ``[0:zero_rows]`` AFTER the recurrence, and
      ``zero_rows`` is 1 or 2 on every |m| >= 2 fixture here (measured off the shipped
      builder: m = 2 gives 2, m = 2 accurate 1, m = 3 accurate 1, m = -2 gives 2), so
      the mutated row is overwritten with exact zero before anything reads it. At
      |m| = 1 the tail zeroes Dz at row 0 ALONE (:588), leaving Dx and Dy to carry it.

    The leg still RUNS on every fixture: where it is declared not live, ``_score``
    requires INERTNESS, so a divergence there fails rather than passing quietly.
    """
    if family["shape"] not in ("cylindrical", "cylindrical_real"):
        return _spec_has_a_wall(family, spec)
    if _spec_walls(family, spec)[2]:
        return True
    if family["seam"] == "B":
        return True
    # THE m = 0 D SEAM (the 2026-09-02 real electric twin): its certified axis
    # tail zeroes Dy alone at row 0 and ADDS to Dz there, so the r-axis
    # metallic code's change at row 0 survives in Dx and inside the Dz add --
    # unlike |m| >= 2, where the tail overwrites the whole row. The m0 spec
    # table carries no "m" key; 0 is its one value.
    return abs(int(spec.get("m", 0))) == 1 or int(spec.get("m", 0)) == 0


def _wall_x_is_live(family: Dict[str, Any], spec: Dict[str, Any]) -> bool:
    """Is the X-AXIS wall row live on this fixture?

    The ``zero_metal_clears_no_register`` needle rewrites the Bx/wall_x line
    alone, so a fixture whose only wall is y or z arms nothing there --
    measured on the folded sweep (UNCAUGHT on fold_x_periodic, whose wall is
    y), where ``_spec_has_a_wall`` over-declared it. A folded X carries no wall
    whatever the boundary says.
    """
    if "boundaries" not in spec:
        return False
    if tuple(spec["boundaries"])[0] != "metallic":
        return False
    return not any(axis == "X" for axis, _p in spec.get("symmetry", ()))


HOST_MUTATIONS: Tuple[Dict[str, Any], ...] = (
    {"name": "swap_constitutive_sublattice", "expect": "caught",
     "family": ("complex", "cylindrical", "complex_electric", "cylindrical_electric",
                "complex_folded", "complex_beta",
                "folded_complex_electric", "complex_beta_electric"),
     "why": "hand update_H the HALF-INTEGER coefficient vectors the B curl reads. "
            "Both compile and both run; the difference is half a cell in the "
            "absorber profile -- converged, smooth and wrong."},
    {"name": "swap_curl_sublattice", "expect": "caught",
     "family": ("complex", "cylindrical", "complex_electric", "cylindrical_electric",
                "complex_folded", "complex_beta",
                "folded_complex_electric", "complex_beta_electric"),
     "why": "the same swap from the other side: the B curl reading update_H's "
            "INTEGER positions."},
    {"name": "drop_walls", "expect": "caught_where_live", "live": _spec_has_a_wall,
     "family": ("complex", "cylindrical", "complex_electric", "cylindrical_electric",
                "complex_folded", "complex_beta",
                "folded_complex_electric", "complex_beta_electric"),
     "why": "tell the kernel no axis is walled. zero_metal_B is CARRIED, so a "
            "dropped wall is a plane of un-wiped B feeding update_H. Inert on a "
            "grid with no wall, where the flags are already all zero."},
    {"name": "walls_rotated", "expect": "caught_where_live",
     "family": ("complex", "cylindrical", "complex_electric", "cylindrical_electric"),
     "live": lambda family, spec: (
         len(set(_spec_walls(family, spec))) > 1),
     "why": "the wall table, rotated by one axis. zero_metal_B's B-side table is "
            "the DIAGONAL -- Bx on x, By on y, Bz on z -- and the D side's is the "
            "off-diagonal complement; a pair that reused the wrong one clears the "
            "wrong plane. Live only where the three flags are not all equal, since "
            "a rotation of (1,1,1) or (0,0,0) is the identity."},
    {"name": "walls_from_boundary_codes", "expect": "caught_where_live",
     "family": ("complex", "cylindrical", "complex_electric", "cylindrical_electric"),
     "live": _walls_and_codes_can_differ,
     # Live on the Dcyl family at |m| = 1 and NOWHERE on the Cartesian complex one,
     # where the two grid readings coincide by construction. So the "must be live
     # somewhere" demand is narrowed to the family that can satisfy it, and the leg
     # still RUNS on every Cartesian fixture as a null control whose divergence would
     # mean that coincidence is false.
     "live_somewhere_on": ("cylindrical", "cylindrical_electric"),
     "why": "READ THE WALL PLAN OFF THE CURL'S GHOST-RULE RESOLUTION. It is the "
            "confusion this family's two grid readings exist to prevent, and the "
            "two families answer differently: on a Cartesian complex grid the "
            "readings coincide (zero_metal_axes is `is_metallic and not "
            "is_mirrored` and folds are refused), so the leg is inert; on a Dcyl "
            "grid the r axis maps to BC_METALLIC in the codes and is NOT a wall in "
            "zero_metal_axes, so the leg wipes a plane of Br on every row."},
    {"name": "codes_all_periodic", "expect": "caught_where_live",
     "live": _codes_all_periodic_is_live,
     # NOT LIVE ANYWHERE ON THE NO-ABSORBER FAMILY, and that is a REFUSAL rather
     # than a gap in its fixture table: that product declines every metallic run
     # by name, so no admitted grid carries a metallic code for this leg to drop
     # and no fixture could be built that does. The leg still RUNS on every one of
     # its fixtures, where `_score` requires INERTNESS -- a divergence there would
     # mean the emitted curl reads a boundary code the predicate says cannot be
     # metallic.
     "live_somewhere_on": ("complex", "cylindrical", "complex_electric",
                           "cylindrical_electric"),
     "why": "drop the metallic ghost rule from the curl's neighbour stencil. On a "
            "Dcyl grid the r axis is always BC_METALLIC (the codes map it onto that), "
            "but WHERE the code is read differs by seam and that decides liveness -- "
            "see _codes_all_periodic_is_live, which was narrowed by a measurement "
            "rather than a reading."},
    # ------------------------------------ the residual REAL families' own legs
    {"name": "swap_constitutive_sublattice_real", "expect": "caught",
     "family": ("cylindrical_real", "special_kz", "bfast",
                "cylindrical_real_electric", "special_kz_electric",
                "bfast_electric"),
     "why": "hand the real update_H the HALF-INTEGER coefficient vectors the B "
            "curl reads -- the same half-cell error as the complex leg, in the "
            "real families' own table resolvers."},
    {"name": "swap_curl_sublattice_real", "expect": "caught",
     "family": ("cylindrical_real", "special_kz", "bfast",
                "cylindrical_real_electric", "special_kz_electric",
                "bfast_electric"),
     "why": "the same swap from the other side: the real B curl reading "
            "update_H's INTEGER positions."},
    {"name": "drop_walls_real", "expect": "caught_where_live",
     "family": ("cylindrical_real", "special_kz", "bfast",
                "cylindrical_real_electric", "special_kz_electric",
                "bfast_electric"),
     "live": _spec_has_a_wall,
     "why": "tell the kernel no axis is walled. zero_metal_B is CARRIED by all "
            "three residual real welds, so a dropped wall is a plane of un-wiped "
            "B feeding update_H. Inert on a grid with no wall."},
    {"name": "wrong_arm", "expect": "caught_where_live", "live": _arm_is_live,
     # WHERE THIS LEG MUST BE LIVE, and it is a per-family declaration because the
     # answer is: on the Cartesian complex family, whose Bloch rotation is a full
     # complex product. On the Dcyl family it is a NULL CONTROL -- ``_arm_is_live``
     # returns False there, so ``_score`` requires INERTNESS rather than a catch, and
     # the leg still runs on every Dcyl fixture where a divergence would fail it.
     "live_somewhere_on": ("complex", "complex_electric",
                           "no_pml_complex_electric"),
     "why": "compile the arm this platform was NOT licensed for. Both compile, "
            "both run, and they differ in the last bits of about a quarter of the "
            "words. Live wherever a FULL complex product exists: the Bloch "
            "rotation on a phased Cartesian grid, and the i*m/r coupling on every "
            "Dcyl row. update_H's constitutive_apply is a real coefficient times a "
            "complex field, which both arms compute identically, so an unphased "
            "Cartesian fixture has nothing for the arm to change."},

    # ------------------------------------------- the NO-ABSORBER family's own legs
    #
    # THREE LEGS THAT EXIST NOWHERE ELSE IN THIS FILE, because they arm the two things
    # this product has that no sibling does: a curl half that ships TWO kernels, and a
    # POLE BANK. Neither is reachable through the four legs above -- that product has
    # no coefficient table to swap and no wall flags to rotate -- so a sweep that
    # reused the shared table would have run four INERT legs and armed nothing.
    {"name": "wrong_curl_arm", "expect": "caught_where_live",
     "family": ("no_pml_complex_electric",),
     "live": lambda family, spec: bool(spec.get("conductive")),
     "why": "LAUNCH THE PLAIN-TAIL WELD ON A CONDUCTIVE RUN. Both kernels compile, "
            "both run, and the difference is `field *= condfac; field -= curl; field "
            "*= condinv` against `field -= curl` -- every word of the flux density, on "
            "every cell, from the first step. THIS IS THE DEFECT THE WHOLE TWO-ARM "
            "SHAPE EXISTS TO PREVENT: every corpus row on this product's cell records "
            "arm=conductive, so a product built on the plain arm alone would have "
            "served the entire cell with this defect. Live on a conductive fixture "
            "only -- on a lossless one the two arms ARE the same launch, since the "
            "classifier returns 'plain' and the mutation asks for 'plain'."},
    {"name": "pole_banks_rotated", "expect": "caught_where_live",
     "family": ("no_pml_complex_electric",),
     "live": lambda family, spec: int(spec.get("poles", 0)) > 0,
     "why": "give component 0 component 1's pole bank. `update_E`'s source is D minus "
            "every driving P FOR THAT COMPONENT (fields.py:1095-1105); a rotated bank "
            "subtracts the wrong polarization and is a smooth, converged, wrong "
            "dispersion. Inert on the pole-free fixture, where every bank is the "
            "unused-slot filler and the runtime np guard is false for all of it."},
    {"name": "pole_counts_zeroed", "expect": "caught_where_live",
     "family": ("no_pml_complex_electric",),
     "live": lambda family, spec: int(spec.get("poles", 0)) > 0,
     "why": "tell the kernel every component has ZERO poles. `minus_poles_reg`'s "
            "eight guards are all runtime `np` tests, so this is the whole "
            "polarization dropped from update_E's source while update_P keeps "
            "advancing it -- the ordinary constitutive update wearing a dispersive "
            "label. Inert where there is no pole to drop, which is the fixture that "
            "proves the pole-free store is a real write rather than a no-op."},
)


def _host_patch(family: Dict[str, Any], name: str, pml: Any,
                arm: str) -> Callable[[Dict[str, Any], Any], Dict[str, Any]]:
    """One host-side defect, armed through the shipped launcher's own doors.

    THE TWO SUB-LATTICE SWAPS ARE DERIVED FROM THE FAMILY, never typed, and the first
    device run of the electric families is why. Until 2026-08-31 this function handed
    the constitutive group ``half_integer=True`` and the curl group ``False`` -- which
    IS the swap on the B/H seam, where the curl reads HALF-INTEGER and ``update_H``
    INTEGER. THE D/E PAIRING IS THE MIRROR IMAGE, so on the two electric families those
    same two constants are the SHIPPED tables: the mutation changed no byte, both legs
    scored UNCAUGHT, and the run reported a defect this gate could not see when what it
    had actually done was arm nothing. The shipped sub-lattice is now read from
    ``complex_emitter.HALF_INTEGER`` -- the same table both launchers derive theirs
    from -- and the mutation supplies its negation.
    """
    from meep_gpu.cuda_kernels import complex_emitter  # noqa: PLC0415
    from meep_gpu.cuda_kernels import complex_pml_kernels  # noqa: PLC0415

    curl_shipped = complex_emitter.HALF_INTEGER[family["curl_sub_step"]]
    constitutive_shipped = complex_emitter.HALF_INTEGER[family["seam_constitutive"]]

    def patch(arguments: Dict[str, Any], grid: Any) -> Dict[str, Any]:
        out = dict(arguments)
        # THE TABLE COPY IS MADE ONLY WHERE A LEG NEEDS ONE. A product with no
        # absorber binds no coefficient vector and its resolver returns no ``tables``
        # key at all, so copying unconditionally would raise before the branch that
        # decides whether this leg is even about a table.
        if name in ("swap_constitutive_sublattice", "swap_curl_sublattice"):
            out["tables"] = dict(arguments["tables"])
        if name == "swap_constitutive_sublattice":
            out["tables"]["constitutive"] = complex_pml_kernels._tables(  # noqa: SLF001
                pml, not constitutive_shipped, ("kps", "kms"))
        elif name == "swap_curl_sublattice":
            out["tables"]["curl"] = complex_pml_kernels._tables(  # noqa: SLF001
                pml, not curl_shipped, ("kms", "sinv"))
        elif name == "drop_walls":
            out["walls"] = (0, 0, 0)
        elif name == "walls_rotated":
            walls = tuple(arguments["walls"])
            out["walls"] = (walls[1], walls[2], walls[0])
        elif name == "walls_from_boundary_codes":
            out["walls"] = tuple(int(int(c) == 1)
                                 for c in arguments["boundary_codes"])
        elif name == "codes_all_periodic":
            out["boundary_codes"] = tuple(type(c)(0)
                                          for c in arguments["boundary_codes"])
        elif name == "wrong_arm":
            out["arm"] = "NAIVE" if arm == "FMA_V1" else "FMA_V1"
        elif name == "wrong_curl_arm":
            # THE OTHER ARM, and on this family it is a different KERNEL rather than a
            # different expansion of one. ONE DIRECTION ONLY, deliberately: a lossless
            # fixture has no conductivity volumes to bind, so asking for the conductive
            # weld there would be a crash rather than a defect. Leaving it alone arms
            # the IDENTITY, which is what the leg's liveness declares -- it is not live
            # on a lossless fixture and `_score` requires it to move no word there.
            if arguments["curl_arm"] == "conductive":
                out["curl_arm"] = "plain"
        elif name == "pole_banks_rotated":
            out["poles_mutation"] = "rotate"
        elif name == "pole_counts_zeroed":
            out["poles_mutation"] = "zero"
        elif name == "swap_constitutive_sublattice_real":
            # DERIVED, NOT TYPED -- the same lesson the complex branch's
            # docstring records: on the B seam the shipped constitutive is
            # INTEGER so the swap hands half-integer, and on the D seam
            # (the 2026-09-02 electric twins) it is the mirror image.
            from meep_gpu.cuda_kernels import constitutive_kernels  # noqa: PLC0415
            out["tables"] = dict(arguments["tables"])
            out["tables"]["constitutive"] = \
                constitutive_kernels.real_constitutive_tables(
                    pml, not constitutive_shipped)
        elif name == "swap_curl_sublattice_real":
            out["tables"] = dict(arguments["tables"])
            if family["shape"] == "cylindrical_real":
                from meep_gpu.cuda_kernels import cylindrical_kernels  # noqa: PLC0415
                out["tables"]["curl"] = \
                    cylindrical_kernels.cylindrical_curl_tables(
                        pml, not curl_shipped)
            else:
                from meep_gpu.cuda_kernels import step_curl_kernels  # noqa: PLC0415
                out["tables"]["curl"] = \
                    step_curl_kernels.real_pml_curl_tables(
                        pml, not curl_shipped)
        elif name == "drop_walls_real":
            out["walls"] = (0, 0, 0)
        else:
            raise ValueError(f"unknown host mutation {name!r}")
        return out

    return patch


DEVICE_MUTATIONS: Tuple[Dict[str, Any], ...] = (
    {"name": "seam_reloads_from_memory",
     "old": "cf s0 = b0;", "new": "cf s0 = cf_load(f0, idx);",
     "family": ("complex", "complex_folded", "complex_beta"), "expect": "inert",
     "why": "THE SEAM ITSELF, REMOVED -- and it is a NULL CONTROL rather than a "
            "defect, which is the whole claim this product makes about it. Reading "
            "B back from global instead of from the register must be the identity "
            "on the bits: the thread that wrote the word is the thread that reads "
            "it, at its own index, with no fill and no neighbour in between. A "
            "DIVERGENCE HERE WOULD MEAN THE WELD CHANGED THE ARITHMETIC, which is "
            "exactly what LIFT_EDITS says it does not."},
    {"name": "seam_reloads_from_memory_cyl",
     "old": "cf s0 = b0;", "new": "cf s0 = cf_load(Bx, idx);",
     "family": ("cylindrical",), "expect": "inert",
     "why": "the same null control on the Dcyl kernel."},
    {"name": "zero_metal_clears_no_register",
     "old": "{ b0 = cf_zero(); cf_store(f0, idx, b0); }",
     "new": "{ cf_store(f0, idx, cf_zero()); }",
     "family": ("complex", "complex_folded", "complex_beta"),
     "expect": "caught_where_live", "live": _wall_x_is_live,
     "why": "carry the wall clear to memory but NOT to the register. update_H then "
            "reads the un-wiped displacement -- exactly the defect the register "
            "hand-off makes possible and the reason the clear is beside the store. "
            "Inert where no axis is walled, because the guarded block never runs."},
    {"name": "zero_metal_clears_no_register_cyl",
     # THE Z COMPONENT, NOT X, and the first run is why. On a Dcyl grid
     # ``zero_metal_axes`` never sets ``wall_x`` -- axis 0 is the cylindrical axis,
     # which ``grid.is_metallic`` does not report as a wall -- so the Bx-guarded
     # block this leg used to needle never runs and the leg scored UNCAUGHT on both
     # walled fixtures. The wall a Dcyl run can carry is on z, and the register it
     # feeds to update_H is b2.
     "old": "{ b2 = cf_zero(); cf_store(Bz, idx, b2); }",
     "new": "{ cf_store(Bz, idx, cf_zero()); }",
     "family": ("cylindrical",), "expect": "caught_where_live",
     "live": lambda family, spec: bool(_spec_walls(family, spec)[2]),
     "why": "carry the wall clear to memory but NOT to the register, on the one "
            "axis a Dcyl grid can wall. update_H then reads the un-wiped "
            "displacement -- the defect the register hand-off makes possible and "
            "the reason the clear sits beside the store."},
    {"name": "axis_tail_m0_not_carried",
     "old": ("if (m_class == 0 && i == 0) {\n"
             "        b0 = cf_zero(); cf_store(Bx, idx, b0);\n    }"),
     "new": ("if (m_class == 0 && i == 0) {\n"
             "        cf_store(Bx, idx, cf_zero());\n    }"),
     "family": ("cylindrical",), "expect": "caught_where_live",
     "live": lambda family, spec: int(spec["m"]) == 0,
     "why": "THE m = 0 BRANCH OF THE AXIS TAIL (2026-09-04), UNCARRIED. It runs "
            "AFTER the recurrence and zeroes Br on the axis row; a weld that "
            "captured at pml_apply and stopped hands update_H the pre-zeroing "
            "displacement there. Live at m = 0 only: the guard never runs at "
            "|m| >= 1."},
    {"name": "axis_tail_not_carried",
     "old": "b0 = cf_zero(); cf_store(Bx, idx, b0);   // THE AXIS TAIL, CARRIED",
     "new": "cf_store(Bx, idx, cf_zero());",
     "family": ("cylindrical",), "expect": "caught_where_live",
     "live": lambda family, spec: abs(int(spec["m"])) >= 2,
     "why": "THE ONE THING THIS FAMILY ADDS. The |m| >= 2 axis rule runs AFTER the "
            "recurrence; a weld that captured at pml_apply and stopped hands "
            "update_H the pre-zeroing displacement on rows [0:zero_rows]. Live at "
            "|m| >= 2 only -- at |m| = 1 the array path runs no axis zeroing at "
            "all, which is a fact READ OFF stepping._cylindrical_axis_zero_B rather "
            "than inferred from the D side being nearby."},
    {"name": "constitutive_reads_curl_sublattice",
     "old": "kms_int_x[i]", "new": "kms_x[i]",
     "family": ("complex", "cylindrical"), "expect": "caught",
     "why": "the half-cell error the kms_int rename exists to prevent, planted in "
            "the DEVICE text rather than in the binding."},
    {"name": "constitutive_reads_curl_sublattice_electric",
     # THE SAME COLLISION, SPELLED THE OTHER WAY. On the D seam the CURL reads the
     # INTEGER vector and update_E the HALF-INTEGER one, so the rename that resolves
     # it is `kms_half_*` rather than `kms_int_*` -- the mirror image of the B/H
     # pairing, and a leg that reused the B spelling would needle nothing here.
     "old": "kms_half_x[i]", "new": "kms_x[i]",
     "family": ("complex_electric", "cylindrical_electric"), "expect": "caught",
     "why": "the half-cell error the kms_half rename exists to prevent, planted in "
            "the DEVICE text rather than in the binding."},
    {"name": "wall_on_the_wrong_axis",
     "old": "if (wall_x && i == 0)", "new": "if (wall_x && j == 0)",
     "family": ("complex",), "expect": "caught_where_live",
     "live": lambda family, spec: bool(_spec_walls(family, spec)[0]),
     "why": "zero_metal_B's diagonal, rotated. Bx is wiped on the x wall and only "
            "there; a kernel that used the D-side off-diagonal complement would "
            "clear this plane. Live only where the X axis is walled -- the guard "
            "is `wall_x && ...`, so with wall_x zero neither spelling runs."},
    {"name": "wall_on_the_wrong_axis_cyl",
     "old": "if (wall_z && k == 0)", "new": "if (wall_z && i == 0)",
     "family": ("cylindrical",), "expect": "caught_where_live",
     "live": lambda family, spec: bool(_spec_walls(family, spec)[2]),
     "why": "THE SAME ROTATION ON THE ONE AXIS A Dcyl RUN CAN WALL. Bz is wiped at "
            "stored cell 0 of z; pointing the guard at the r index instead clears "
            "the axis row, which is exactly the plane the |m| >= 2 tail and the "
            "i*m/r coupling make most visible. Written as its own leg rather than "
            "shared with the Cartesian one because wall_x is never set on Dcyl and "
            "the shared spelling was NEVER LIVE there -- measured, not assumed."},

    # ---------------------------------------------------------------- the D seam
    #
    # EVERY ONE OF THESE IS ITS OWN LEG RATHER THAN A SHARED SPELLING, and that is
    # forced rather than tidy: the two electric kernels name their carried registers
    # `d*` where the magnetic ones name them `b*`, the wall table they carry is the
    # OFF-DIAGONAL complement so the block a needle has to find holds TWO stores, and
    # the sub-lattice rename is `kms_half_*` rather than `kms_int_*`. A leg that
    # reused a B-side needle would match nothing, which `run_mutations` reports as
    # NEEDLE MISSED and scores as a failure -- but a leg that matched the WRONG block
    # would score UNCAUGHT while looking armed, which is why each needle below was
    # counted against the emitted source before it was written down.
    {"name": "seam_reloads_from_memory_electric",
     "old": "cf s0 = d0;", "new": "cf s0 = cf_load(f0, idx);",
     "family": ("complex_electric",), "expect": "inert",
     "why": "THE SEAM ITSELF, REMOVED -- a NULL CONTROL rather than a defect, and "
            "the whole claim this product makes about it. Reading D back from global "
            "instead of from the register must be the identity on the bits: the "
            "thread that wrote the word is the thread that reads it, at its own "
            "index, with no fill and no neighbour between. A DIVERGENCE HERE WOULD "
            "MEAN THE WELD CHANGED THE ARITHMETIC."},
    {"name": "seam_reloads_from_memory_electric_cyl",
     "old": "cf s0 = d0;", "new": "cf s0 = cf_load(Dx, idx);",
     "family": ("cylindrical_electric",), "expect": "inert",
     "why": "the same null control on the Dcyl electric kernel."},
    {"name": "zero_metal_clears_no_register_electric",
     # THE WHOLE wall_x BLOCK, because the D table wipes TWO components there and
     # the inner statement `d1 = cf_zero(); cf_store(f1, idx, d1);` also appears in
     # the wall_z line -- a needle on the statement alone would rewrite whichever
     # came first and the leg would be reporting on the wrong wall.
     "old": ("{ d1 = cf_zero(); cf_store(f1, idx, d1); "
             "d2 = cf_zero(); cf_store(f2, idx, d2); }"),
     "new": "{ cf_store(f1, idx, cf_zero()); cf_store(f2, idx, cf_zero()); }",
     "family": ("complex_electric",), "expect": "caught_where_live",
     "live": lambda family, spec: bool(_spec_walls(family, spec)[0]),
     "why": "carry the wall clear to memory but NOT to the registers. update_E then "
            "reads the un-wiped displacement on BOTH components the x wall clears -- "
            "exactly the defect the register hand-off makes possible and the reason "
            "the clears sit beside the stores."},
    {"name": "zero_metal_clears_no_register_electric_cyl",
     # THE Z WALL, for the reason the magnetic Dcyl leg gives: `zero_metal_axes`
     # never sets wall_x on a Dcyl grid, so an r-guarded block never runs there.
     "old": ("{ d0 = cf_zero(); cf_store(Dx, idx, d0); "
             "d1 = cf_zero(); cf_store(Dy, idx, d1); }"),
     "new": "{ cf_store(Dx, idx, cf_zero()); cf_store(Dy, idx, cf_zero()); }",
     "family": ("cylindrical_electric",), "expect": "caught_where_live",
     "live": lambda family, spec: bool(_spec_walls(family, spec)[2]),
     "why": "the same, on the one axis a Dcyl run can wall: Dr and Dp are the two "
            "components with Yee shift 0 on z."},
    {"name": "wall_on_the_wrong_axis_electric",
     "old": "if (wall_x && i == 0)", "new": "if (wall_x && j == 0)",
     "family": ("complex_electric",), "expect": "caught_where_live",
     "live": lambda family, spec: bool(_spec_walls(family, spec)[0]),
     "why": "zero_metal_D's OFF-DIAGONAL table, pointed at the wrong coordinate. Dy "
            "and Dz are wiped at stored cell 0 of x; testing j instead clears a "
            "plane the array path keeps. Live only where the X axis is walled."},
    {"name": "wall_on_the_wrong_axis_electric_cyl",
     "old": "if (wall_z && k == 0)", "new": "if (wall_z && i == 0)",
     "family": ("cylindrical_electric",), "expect": "caught_where_live",
     "live": lambda family, spec: bool(_spec_walls(family, spec)[2]),
     "why": "the same rotation on the one axis a Dcyl run can wall."},
    {"name": "axis_tail_m2_not_carried_electric",
     # ANCHORED AS A WHOLE LINE WITH ITS INDENTATION: the same statement appears
     # inline inside two zero_metal blocks, and only the tail spells it on its own
     # eight-space line.
     "old": "\n        d0 = cf_zero(); cf_store(Dx, idx, d0);\n",
     "new": "\n        cf_store(Dx, idx, cf_zero());\n",
     "family": ("cylindrical_electric",), "expect": "caught_where_live",
     "live": lambda family, spec: abs(int(spec["m"])) >= 2,
     "why": "THE |m| >= 2 BRANCH OF THE AXIS TAIL, UNCARRIED. It runs AFTER the "
            "recurrence; a weld that captured at pml_apply and stopped hands "
            "update_E the pre-zeroing displacement on rows [0:zero_rows]."},
    {"name": "axis_tail_m0_not_carried_electric",
     "old": ("if (m_class == 0 && i == 0) {\n"
             "        d2 = cf_add(cf_load(Dz, idx),\n"
             "                    mul_coefficient_left(axis_coef, cf_load(Hy, idx)));\n"
             "        cf_store(Dz, idx, d2);\n"
             "        d1 = cf_zero(); cf_store(Dy, idx, d1);\n    }"),
     "new": ("if (m_class == 0 && i == 0) {\n"
             "        cf_store(Dz, idx, cf_add(cf_load(Dz, idx),\n"
             "                    mul_coefficient_left(axis_coef, cf_load(Hy, idx))));\n"
             "        cf_store(Dy, idx, cf_zero());\n    }"),
     "family": ("cylindrical_electric",), "expect": "caught_where_live",
     "live": lambda family, spec: int(spec["m"]) == 0,
     "why": "THE m = 0 BRANCH OF THE AXIS TAIL (2026-09-04), UNCARRIED on the D "
            "side: the on-axis Dz post-add and Dp = 0 reach memory but not the "
            "registers, so update_E reads the pre-add Dz and the pre-zeroing Dy on "
            "the axis row. Live at m = 0 only."},
    {"name": "axis_tail_m0_post_add_dropped_electric",
     "old": ("        d2 = cf_add(cf_load(Dz, idx),\n"
             "                    mul_coefficient_left(axis_coef, cf_load(Hy, idx)));\n"
             "        cf_store(Dz, idx, d2);\n"),
     "new": "",
     "family": ("cylindrical_electric",), "expect": "caught_where_live",
     "live": lambda family, spec: int(spec["m"]) == 0,
     "why": "the m = 0 on-axis Dz post-add (stepping.py:585) removed outright -- the "
            "|m| >= 1 kernel's tail wearing the m = 0 label. Live at m = 0 only."},
    {"name": "axis_tail_m1_not_carried_electric",
     # THE BRANCH THE B SIDE DOES NOT HAVE. stepping._cylindrical_axis_zero_D zeroes
     # Dz on the axis row at |m| = 1 -- the FIELD ONLY, never fu_Dz (:588) -- where
     # _cylindrical_axis_zero_B does nothing at all. A port of the magnetic family
     # would carry one branch and silently drop this one.
     "old": ("if (m_class == 1 && i == 0) {\n"
             "        d2 = cf_zero(); cf_store(Dz, idx, d2);\n    }"),
     "new": ("if (m_class == 1 && i == 0) {\n"
             "        cf_store(Dz, idx, cf_zero());\n    }"),
     "family": ("cylindrical_electric",), "expect": "caught_where_live",
     "live": lambda family, spec: abs(int(spec["m"])) == 1,
     "why": "THE |m| = 1 BRANCH, UNCARRIED, and the one this family adds over its "
            "magnetic twin. Live at |m| = 1 only: at |m| >= 2 the guard never runs."},
    {"name": "inverse_permittivity_binding_rotated",
     # THE THREE POINTERS THIS SIGNATURE BINDS THAT THE MAGNETIC ONE DOES NOT, and
     # the reason the electric fixtures carry an anisotropic material at all: on a
     # vacuum fixture all three are one allocation of ones and this leg is inert.
     "old": "s0 = mul_field_left(s0, inv_eps_0[idx]);",
     "new": "s0 = mul_field_left(s0, inv_eps_1[idx]);",
     "family": ("complex_electric", "cylindrical_electric"), "expect": "caught",
     "why": "give component 0 component 1's inverse permittivity. The three volumes "
            "are the one group in this signature that is NOT __restrict__ -- an "
            "isotropic run hands one pointer three times -- so nothing about the "
            "binding is checked by the aliasing guard, and only a fixture with three "
            "DISTINCT materials can tell a rotated binding from the right one."},

    # -------------------------------------------------------- the NO-ABSORBER weld
    #
    # EVERY NEEDLE HERE IS ITS OWN, and none is shared with the four PML legs above,
    # because none of that text exists in this kernel: it carries no ``kms``, no
    # ``sinv``, no ``fu``, no wall block and no ``s0`` temporary. What it carries
    # instead -- the register hand-off through ``minus_poles_reg``, the eight guarded
    # pole subtractions and the conductive tail's three-operation order -- is what
    # these needles find. Each was counted against the emitted source before it was
    # written down; ``run_mutations`` reports NEEDLE MISSED and FAILS the leg for one
    # that matches nothing, which is what keeps that true.
    {"name": "seam_reloads_from_memory_no_pml",
     "old": "minus_poles_reg(d0, a0,",
     "new": "minus_poles_reg(cf_load(f0, idx), a0,",
     "family": ("no_pml_complex_electric",), "expect": "inert",
     "why": "THE SEAM ITSELF, REMOVED -- and it is a NULL CONTROL rather than a "
            "defect, which is the whole claim this product makes about it. Reading D "
            "back from global instead of from the register must be the identity on "
            "the bits: the thread that wrote the word is the thread that reads it, at "
            "its own index, with no fill, no wall clear and no neighbour between. A "
            "DIVERGENCE HERE WOULD MEAN THE WELD CHANGED THE ARITHMETIC, which is "
            "exactly what LIFT_EDITS says it does not."},
    {"name": "poles_pre_summed_no_pml",
     # THE FIRST TWO GUARDED SUBTRACTIONS, TURNED INTO A PRE-SUM. Anchored on both
     # lines together so the needle cannot attach to one guard and leave the other.
     "old": ("    if (np > 0) s = cf_sub(s, cf_load(p0, idx));\n"
             "    if (np > 1) s = cf_sub(s, cf_load(p1, idx));\n"),
     "new": ("    if (np > 1) s = cf_sub(s, cf_add(cf_load(p0, idx), "
             "cf_load(p1, idx)));\n"
             "    if (np == 1) s = cf_sub(s, cf_load(p0, idx));\n"),
     "family": ("no_pml_complex_electric",), "expect": "caught_where_live",
     "live": lambda family, spec: int(spec.get("poles", 0)) >= 2,
     "why": "((D - P0) - P1) against D - (P0 + P1). MEASURED at 50279 of 400000 "
            "float32 words on a random two-pole fixture (test_complex_no_pml.py), and "
            "the reason minus_poles subtracts one contributor at a time in "
            "REGISTRATION ORDER. Live at two poles only: with one pole the mutated "
            "branch computes the same single subtraction, and with none neither runs."},
    {"name": "conductive_tail_reassociated_no_pml",
     "old": ("    cf t = mul_field_left(cf_load(f, idx), condfac);\n"
             "    t = cf_sub(t, curl);\n"),
     "new": ("    cf t = mul_field_left(cf_sub(cf_load(f, idx), curl), condfac);\n"),
     "family": ("no_pml_complex_electric",), "expect": "caught",
     "variants": ("conductive",),
     "why": "THE ORDER IS THE ARITHMETIC. `field *= condfac; field -= curl; field *= "
            "condinv` reassociated to `((field - curl) * condfac) * condinv` differs "
            "on 400000 of 400000 float32 words (measured, test_complex_no_pml.py) -- "
            "the same three operations, three roundings in a different order. Scored "
            "on the CONDUCTIVE variant only, because the plain weld has no such tail "
            "and the needle would match nothing there."},
    {"name": "plain_tail_operands_swapped_no_pml",
     "old": "    cf value = cf_sub(cf_load(f, idx), curl);\n",
     "new": "    cf value = cf_sub(curl, cf_load(f, idx));\n",
     "family": ("no_pml_complex_electric",), "expect": "caught",
     "variants": ("plain",),
     "why": "`curl - target` instead of `target -= curl` (stepping._apply_curl's "
            "FOURTH tail, stepping.py:539). The ONE hand-touched statement in the "
            "plain weld is this helper's, and the lift's whole claim about it is that "
            "the expression, its parenthesisation and its OPERAND ORDER are the "
            "certified ones -- so the operand order is what this leg arms. Scored on "
            "the PLAIN variant only: the conductive weld reaches its subtraction "
            "through conductive_apply_reg and the needle has no site there."},
    {"name": "inverse_permittivity_binding_rotated_no_pml",
     # THE SAME DEFECT AS THE PML SIBLINGS' AND A DIFFERENT NEEDLE, because this
     # kernel stores through `h*` and multiplies inside the `cf_store` rather than
     # through an `s0` temporary.
     "old": "        inv_eps_0[idx]));", "new": "        inv_eps_1[idx]));",
     "family": ("no_pml_complex_electric",), "expect": "caught",
     "why": "give component 0 component 1's inverse permittivity. The three volumes "
            "are NOT __restrict__ -- an isotropic run hands one pointer three times -- "
            "so nothing about the binding is checked by the aliasing guard, and only "
            "a fixture with three DISTINCT materials can tell a rotated binding from "
            "the right one. THIS LAUNCH BINDS THEM TWICE, in the unused pole slots as "
            "well, so a rotation here is also the only leg that would see the second "
            "binding drift from the first."},
    {"name": "constitutive_stores_to_the_flux_density_no_pml",
     "old": "    cf_store(h0, idx, mul_field_left(",
     "new": "    cf_store(f0, idx, mul_field_left(",
     "family": ("no_pml_complex_electric",), "expect": "caught",
     "why": "THE TARGET RENAME, UNDONE. `f0` is D in the curl body and E in the "
            "constitutive one; letting the second write the first stores the "
            "constitutive product back into the flux density and leaves Ex untouched. "
            "It COMPILES, which is what makes it worth arming: the collision this "
            "rename resolves is the harmless half only because the rename is there."},
    # --------------------------- the cf fill carry's own needles, 2026-09-02
    #
    # EVERY ONE IS SCORED ONLY WHERE THE PASS IT BREAKS IS LIVE, exactly as the
    # real pair's catalog scores its fills; the liveness reads the SPEC's own
    # fold declaration. The needles anchor component-specific emitted lines, so
    # a moved emitter fails as NEEDLE MISSED rather than arming nothing.
    {"name": "cf_ownership_guard_inverted",
     "old": "    if (!owned) return cf_zero();",
     "new": "    if (owned) return cf_zero();",
     "family": ("complex_folded", "complex_beta",
                "folded_complex_electric", "complex_beta_electric"),
     "expect": "caught",
     "why": "the OWNER returns before storing B and the destination stores "
            "instead, so every owned cell keeps its previous step's flux "
            "density. Deterministic -- the owner simply stops writing -- and "
            "caught on unfolded rows too, where owned is identically 1. The "
            "electric twins share the guard's spelling through the shared "
            "carry, so the same needle arms their D."},
    {"name": "cf_constitutive_ownership_inverted",
     "old": "    if (own_0) {",
     "new": "    if (!own_0) {",
     "family": ("complex_folded", "complex_beta",
                "folded_complex_electric", "complex_beta_electric"),
     "expect": "caught",
     "why": "the constitutive half runs at the ghosts and not at the owned "
            "cells: H and f_w_H (E and f_w_E on the electric twins) keep the "
            "previous step's value everywhere the thread owns. Deterministic, "
            "no contended word."},
    {"name": "cf_near_fill_images_the_wrong_row",
     "old": "int g1_n_i = idx - 2 * sy;",
     "new": "int g1_n_i = idx - 1 * sy;",
     "family": ("complex_folded", "complex_beta"), "expect": "caught_where_live",
     "live": lambda family, spec: any(
         axis == "Y" for axis, _phase in spec.get("symmetry", ())),
     "why": "the near fill's source row is stepping.MIRROR_SOURCE_INDEX = 2, "
            "MEEP's halved-grid io = -2; imaging row 1 is a whole cell wrong. "
            "Component 1's needle, so the y-fold fixtures drive it."},
    {"name": "cf_near_fill_parity_negated",
     "old": "cf g1_n_v = mul_coefficient_left(phase_y, b1);",
     "new": "cf g1_n_v = mul_coefficient_left((phase_y * -1.0f), b1);",
     "family": ("complex_folded", "complex_beta"), "expect": "caught_where_live",
     "live": lambda family, spec: any(
         axis == "Y" for axis, _phase in spec.get("symmetry", ())),
     "why": "the near parity is +phase (the imaged component's Yee shift is 0 "
            "there); negating it is the doubly mirrored value with the wrong "
            "sign -- smooth, converged and wrong."},
    {"name": "cf_far_fill_parity_unnegated",
     "old": "cf g0_y_v = mul_coefficient_left((phase_y * -1.0f), b0);",
     "new": "cf g0_y_v = mul_coefficient_left(phase_y, b0);",
     "family": ("complex_folded", "complex_beta"), "expect": "caught_where_live",
     "live": lambda family, spec: (
         any(axis == "Y" for axis, _phase in spec.get("symmetry", ()))
         and tuple(spec["boundaries"])[1] == "periodic"),
     "why": "the far parity is -phase (the imaged component's Yee shift is 1 "
            "there); dropping the negation flips every far ghost of Bx along "
            "y. Live only on a folded PERIODIC y, where the far pass runs."},
    {"name": "cf_far_fill_bakes_n_minus_two",
     "old": "int g0_y_i = idx + (ny - 1 - reflect_y) * sy;",
     "new": "int g0_y_i = idx + (ny - 1 - (ny - 2)) * sy;",
     "family": ("complex_folded", "complex_beta"), "expect": "caught_where_live",
     "live": lambda family, spec: (
         any(axis == "Y" for axis, _phase in spec.get("symmetry", ()))
         and tuple(spec["boundaries"])[1] == "periodic"
         and int(round(spec["cell"][1])) % 2 == 1),
     "why": "the reflect row is n_full - stored + 2 -- stored - 2 at an even "
            "full count and stored - 3 at an odd one. Baking n - 2 reflects "
            "about the window top instead of about the mirror: exact on even "
            "counts, a whole cell wrong on the odd-count fixtures, which is "
            "why they exist."},
    {"name": "cf_composed_parity_drops_the_near_factor",
     "old": "cf g0_yn_v = mul_coefficient_left(phase_x, b0);",
     "new": "cf g0_yn_v = b0;",
     "family": ("complex_folded", "complex_beta"), "expect": "caught_where_live",
     # LIVE ONLY ON A TWO-PLANE FOLD, which only the folded family's sweep can
     # build: a 2-D beta cell folded on BOTH in-plane axes has no axis left to
     # carry an active absorber, so on the beta family this leg is a permanent
     # null control -- the demand is narrowed to the family that can satisfy
     # it, exactly as wrong_arm's is.
     "live_somewhere_on": ("complex_folded",),
     "live": lambda family, spec: (
         any(axis == "X" and int(phase) == -1
             for axis, phase in spec.get("symmetry", ()))
         and any(axis == "Y" for axis, _phase in spec.get("symmetry", ()))),
     "why": "the corner ghost carries the near application INSIDE the far one; "
            "dropping the near factor leaves the far parity alone. Live on the "
            "mixed-phase two-plane fixture, where the dropped factor is -1 and "
            "the corner value flips sign; with the factor +1 the drop is "
            "byte-exact except at signed zeros, which is why the liveness "
            "demands the odd phase."},
    {"name": "cf_ghost_reads_the_wrong_lane",
     "old": "cf g0_y_v = mul_coefficient_left((phase_y * -1.0f), b0);",
     "new": "cf g0_y_v = mul_coefficient_left((phase_y * -1.0f), b1);",
     "family": ("complex_folded", "complex_beta"), "expect": "caught_where_live",
     "live": lambda family, spec: (
         any(axis == "Y" for axis, _phase in spec.get("symmetry", ()))
         and tuple(spec["boundaries"])[1] == "periodic"),
     "why": "the far ghost of Bx built from BY'S register: a plausible complex "
            "number in the right cell of the wrong field. The certified fill "
            "images each component from ITSELF."},
    # ----------------------------- the residual REAL families' device needles
    {"name": "real_seam_reads_the_wrong_register",
     "old": "constitutive_apply(Hx, f_w_Hx, idx, b_x,",
     "new": "constitutive_apply(Hx, f_w_Hx, idx, b_y,",
     "family": ("cylindrical_real", "special_kz", "bfast"), "expect": "caught",
     "why": "Hx built from By's displacement: the seam handing the right value "
            "to the wrong component."},
    {"name": "real_seam_reloads_from_memory",
     "old": "constitutive_apply(Hx, f_w_Hx, idx, b_x,",
     "new": "constitutive_apply(Hx, f_w_Hx, idx, Bx[idx],",
     "family": ("cylindrical_real", "special_kz", "bfast"), "expect": "inert",
     "why": "THE SEAM NULL: reading B back from global instead of from the "
            "register must be the identity on the bits -- these welds carry no "
            "fill, so the thread that wrote the word is the thread reading it. "
            "A divergence here would mean the weld changed the arithmetic."},
    {"name": "real_sublattice_shadow",
     "old": "kms_int_x[i]);",
     "new": "kms_x[i]);",
     "family": ("cylindrical_real", "special_kz", "bfast"), "expect": "caught",
     "why": "the fused scope's one name collision, resolved the wrong way: "
            "update_H reading the curl's HALF-INTEGER absorber profile -- "
            "converged, smooth and half a cell wrong."},
    {"name": "cyl_axis_tail_skips_the_register",
     "old": "if (i == 0) { b_x = 0.0f; Bx[idx] = 0.0f; }",
     "new": "if (i == 0) { Bx[idx] = 0.0f; }",
     "family": ("cylindrical_real",), "expect": "caught",
     "why": "THE CYLINDRICAL SUBTLETY, ARMED: the certified curl does not end "
            "at pml_apply -- the m = 0 axis tail stores zero into Br AFTER the "
            "recurrence -- and a weld that cleared memory but not the register "
            "hands update_H the pre-tail displacement on the axis row."},
    {"name": "real_zero_metal_clears_no_register",
     "old": "if (wall_y && j == 0) { b_y = 0.0f; By[idx] = 0.0f; }",
     "new": "if (wall_y && j == 0) { By[idx] = 0.0f; }",
     "family": ("bfast",), "expect": "caught_where_live",
     "live": lambda family, spec: tuple(spec["boundaries"])[1] == "metallic",
     "why": "carry the wall clear to memory but not to the register: update_H "
            "reads the un-wiped displacement on the y wall plane. NARROWED to "
            "the BFAST weld on 2026-09-02: the special_kz weld's clear gained "
            "the ownership guard with the fill carry, so its spelling moved to "
            "skz_zero_metal_clears_no_register."},
    {"name": "skz_zero_metal_clears_no_register",
     "old": "if (own_y && wall_y && j == 0) { b_y = 0.0f; By[idx] = 0.0f; }",
     "new": "if (own_y && wall_y && j == 0) { By[idx] = 0.0f; }",
     "family": ("special_kz",), "expect": "caught_where_live",
     "live": lambda family, spec: tuple(spec["boundaries"])[1] == "metallic",
     "why": "the same defect on the five-pass special_kz weld, whose clear is "
            "GUARDED on own_y since the fill carry (a cell a fill images "
            "belongs to its source thread): memory wiped, register kept, "
            "update_H reads the un-wiped displacement on the y wall plane."},
    {"name": "cyl_zero_metal_clears_no_register",
     "old": "if (wall_z && k == 0) { b_z = 0.0f; Bz[idx] = 0.0f; }",
     "new": "if (wall_z && k == 0) { Bz[idx] = 0.0f; }",
     "family": ("cylindrical_real",), "expect": "caught_where_live",
     "live": lambda family, spec: spec.get("z_kind") == "metallic",
     "why": "the same defect on the one axis a Dcyl grid can wall: z."},
    {"name": "bfast_substitution_dropped",
     "old": "curl = curl - advance;",
     "new": "curl = curl;",
     "family": ("bfast",), "expect": "caught",
     "why": "the BFAST state substitution removed from Bx's curl while the IIR "
            "state keeps advancing: the plain real pair wearing the BFAST "
            "label. One site (component x), so the state divergence reaches "
            "every component within a step through the array-path D half."},
    # ------------------------- the residual ELECTRIC families' device needles
    # Each is its magnetic sibling's defect restated in the D seam's own
    # spellings: the register hand-off is ``src_<a>`` (post inverse-epsilon),
    # the sub-lattice rename is ``kms_half_*``, the wall table is the
    # OFF-DIAGONAL complement, and three inverse-permittivity pointers exist
    # that the B seam has no counterpart for.
    {"name": "real_electric_seam_reads_the_wrong_register",
     "old": "constitutive_apply(Ex, f_w_Ex, idx, src_x,",
     "new": "constitutive_apply(Ex, f_w_Ex, idx, src_y,",
     "family": ("cylindrical_real_electric", "special_kz_electric",
                "bfast_electric"),
     "expect": "caught",
     "why": "Ex built from Dy's scaled displacement: the seam handing the "
            "right value to the wrong component."},
    {"name": "real_electric_seam_reloads_from_memory",
     "old": "float src_x = d_x * inv_eps_Ex[idx];",
     "new": "float src_x = Dx[idx] * inv_eps_Ex[idx];",
     "family": ("cylindrical_real_electric", "special_kz_electric",
                "bfast_electric"),
     "expect": "inert",
     "why": "THE SEAM NULL: reading D back from global instead of from the "
            "register must be the identity on the bits. At every cell whose "
            "own constitutive RUNS (own_* holds), the thread wrote the word "
            "itself; at a ghost destination the mutated load is concurrent "
            "with the source thread's write but its value is DEAD -- the "
            "own_* guard skips the apply -- so no stored word can move. A "
            "divergence would mean the weld changed the arithmetic."},
    {"name": "real_electric_sublattice_shadow",
     "old": "kms_half_x[i]);",
     "new": "kms_x[i]);",
     "family": ("cylindrical_real_electric", "special_kz_electric",
                "bfast_electric"),
     "expect": "caught",
     "why": "the fused scope's one name collision, resolved the wrong way: "
            "update_E reading the curl's INTEGER absorber profile -- the "
            "mirror image of the B seam's shadow, half a cell wrong."},
    {"name": "real_electric_inverse_permittivity_rotated",
     "old": "float src_x = d_x * inv_eps_Ex[idx];",
     "new": "float src_x = d_x * inv_eps_Ey[idx];",
     "family": ("cylindrical_real_electric", "special_kz_electric",
                "bfast_electric"),
     "expect": "caught",
     "why": "Ex scaled by Ey's permittivity: the binding rotation the "
            "anisotropic fixture exists to make visible (on a vacuum fixture "
            "all three pointers are one allocation and this is inert)."},
    {"name": "real_electric_zero_metal_clears_no_register",
     "old": "if (wall_y && j == 0) { d_x = 0.0f; Dx[idx] = d_x; "
            "d_z = 0.0f; Dz[idx] = d_z; }",
     "new": "if (wall_y && j == 0) { Dx[idx] = 0.0f; Dz[idx] = 0.0f; }",
     "family": ("bfast_electric",),
     "expect": "caught_where_live",
     "live": lambda family, spec: tuple(spec["boundaries"])[1] == "metallic",
     "why": "carry the OFF-DIAGONAL wall clear to memory but not to the "
            "registers: update_E reads the un-wiped displacements on the y "
            "wall plane."},
    {"name": "cyl_electric_zero_metal_clears_no_register",
     "old": "if (wall_z && k == 0) { d_x = 0.0f; Dx[idx] = d_x; "
            "d_y = 0.0f; Dy[idx] = d_y; }",
     "new": "if (wall_z && k == 0) { Dx[idx] = 0.0f; Dy[idx] = 0.0f; }",
     "family": ("cylindrical_real_electric",),
     "expect": "caught_where_live",
     "live": lambda family, spec: spec.get("z_kind") == "metallic",
     "why": "the same defect on the one axis a Dcyl grid can wall: z, whose "
            "OFF-DIAGONAL clear wipes Dx and Dy at the k = 0 plane."},
    {"name": "skz_electric_zero_metal_clears_no_register",
     "old": "if (own_y && clr_y) { d_y = 0.0f; Dy[idx] = d_y; }",
     "new": "if (own_y && clr_y) { Dy[idx] = 0.0f; }",
     "family": ("special_kz_electric",), "expect": "caught_where_live",
     "live": lambda family, spec: (
         tuple(spec["boundaries"])[0] == "metallic"
         or tuple(spec["boundaries"])[2] == "metallic"),
     "why": "the same defect on the five-pass electric weld, whose clear is "
            "flag-driven and GUARDED on own_y since the fill carry: memory "
            "wiped, register kept. clr_y fires from a wall on x or z (the "
            "off-diagonal complement), so the x-walled fixtures drive it."},
    {"name": "cyl_electric_axis_tail_skips_the_register",
     "old": "d_y = 0.0f;\n        Dy[idx] = 0.0f;",
     "new": "Dy[idx] = 0.0f;",
     "family": ("cylindrical_real_electric",), "expect": "caught",
     "why": "THE CYLINDRICAL SUBTLETY on the D side: the certified curl's "
            "m = 0 axis tail stores zero into Dp (= Dy) after the recurrence, "
            "and a weld that cleared memory but not the register hands "
            "update_E the pre-tail displacement on the axis row."},
    {"name": "bfast_electric_substitution_dropped",
     "old": "curl = curl - advance;",
     "new": "curl = curl;",
     "family": ("bfast_electric",), "expect": "caught",
     "why": "the BFAST state substitution removed from the D curl while the "
            "IIR state keeps advancing: the plain real pair wearing the BFAST "
            "label, on the electric seam (three sites, one per component)."},
    {"name": "cf_electric_seam_reloads_from_memory",
     "old": "cf s0 = d0;", "new": "cf s0 = cf_load(f0, idx);",
     "family": ("folded_complex_electric", "complex_beta_electric"),
     "expect": "inert",
     "why": "THE SEAM NULL on the complex twins: reading D back from global "
            "instead of from the register must be the identity on the bits -- "
            "the own-cell constitutive runs only where own_0 holds, and there "
            "the thread wrote the word itself; at a ghost destination the "
            "mutated load's value is dead behind the guard."},
    {"name": "cf_electric_zero_metal_clears_no_register",
     "old": "if (own_1 && clr_1) { d1 = cf_zero(); cf_store(f1, idx, d1); }",
     "new": "if (own_1 && clr_1) { cf_store(f1, idx, cf_zero()); }",
     "family": ("folded_complex_electric", "complex_beta_electric"),
     "expect": "caught_where_live",
     "live": lambda family, spec: (
         tuple(spec["boundaries"])[0] == "metallic"
         or tuple(spec["boundaries"])[2] == "metallic"),
     "why": "carry component 1's wall clear to memory but not to the "
            "register: clr_1 fires from a wall on x or z (the off-diagonal "
            "complement), and every folded fixture walls x, so the leg is "
            "live across both twins' sweeps."},
    {"name": "cf_electric_sublattice_shadow",
     "old": "kms_half_x[i]", "new": "kms_x[i]",
     "family": ("folded_complex_electric", "complex_beta_electric"),
     "expect": "caught_where_live",
     # NARROWED 2026-09-02 FROM AN UNCONDITIONAL "caught", ON A MEASURED
     # IDENTITY AND NOT ON A PREFERENCE. The first run of this leg reported it
     # UNCAUGHT on `fold_x_periodic` and `fold_xy_mixed_phase` -- and on those
     # two fixtures, and only those, X IS THE FOLDED AXIS, which
     # `build_folded_complex_pair` gives NO ABSORBER by construction
     # ("THE FOLDED AXIS TAKES NO ABSORBER (its stored window ends at the
     # mirror)", :1481-1486). With no absorber on x the half-integer and
     # integer profiles ARE the same vector, so the shadow is the identity on
     # the bits -- there is no half cell to be wrong by. Declaring it live
     # there asked the leg to catch a change that changes nothing.
     "live": lambda family, spec: (
         not any(axis == "X" for axis, _phase in spec.get("symmetry", ()))
         and int(round(spec["cell"][0])) >= 2 * int(spec["pml"]) + 2),
     "why": "update_E reading the curl's INTEGER absorber profile through the "
            "fused scope's name collision -- all four sites (the own cell and "
            "the carried ghosts) shadowed together, half a cell wrong. Live "
            "only where X carries an absorber at all: the builder gives a "
            "FOLDED axis none, and an axis too short to hold the spec's "
            "thickness none either, and with none the two sub-lattice "
            "profiles are the same vector."},
    {"name": "cf_electric_inverse_permittivity_rotated",
     "old": "s0 = mul_field_left(s0, inv_eps_0[idx]);",
     "new": "s0 = mul_field_left(s0, inv_eps_1[idx]);",
     "family": ("folded_complex_electric", "complex_beta_electric"),
     "expect": "caught",
     "why": "component 0 scaled by component 1's permittivity: the rotation "
            "the anisotropic fixture makes visible."},
    {"name": "cf_electric_near_fill_images_the_wrong_row",
     "old": "int g0_ny_i = idx - 2 * sy;",
     "new": "int g0_ny_i = idx - 1 * sy;",
     "family": ("folded_complex_electric", "complex_beta_electric"),
     "expect": "caught_where_live",
     "live": lambda family, spec: any(
         axis == "Y" for axis, _phase in spec.get("symmetry", ())),
     "why": "the near fill's source row is stepping.MIRROR_SOURCE_INDEX = 2; "
            "imaging row 1 is a whole cell wrong. Component 0's y needle, so "
            "the y-fold fixtures drive it."},
    {"name": "cf_electric_near_fill_parity_negated",
     "old": "cf g0_ny_v = mul_coefficient_left(phase_y, pre0);",
     "new": "cf g0_ny_v = mul_coefficient_left((phase_y * -1.0f), pre0);",
     "family": ("folded_complex_electric", "complex_beta_electric"),
     "expect": "caught_where_live",
     "live": lambda family, spec: any(
         axis == "Y" for axis, _phase in spec.get("symmetry", ())),
     "why": "the D family's near parity is +phase (the imaged component's Yee "
            "shift is 0 on the two NEAR axes); negating it is the doubly "
            "mirrored value with the wrong sign."},
    {"name": "cf_electric_far_fill_parity_applied",
     "old": "cf g1_y_v = d1;",
     "new": "cf g1_y_v = mul_coefficient_left(phase_y, d1);",
     "family": ("folded_complex_electric", "complex_beta_electric"),
     "expect": "caught_where_live",
     # NARROWED 2026-09-02 ON A MEASURED IDENTITY. The clause that stood here
     # said this leg was "inert at phase +1 only at exact zeros", and that is
     # FALSE: `phase_y` is the plane's MIRROR PARITY, +1 or -1 and nothing else
     # (`complex_electric_fill_carry.fills_plan` packs
     # `float(mirror_fill_phases(grid)[axis])` and RAISES on any other value),
     # and `mul_coefficient_left(1.0f, z)` is `1*re - 0*im, 1*im + 0*re` --
     # the identity on every finite word, not merely on zeros. The first run
     # measured exactly that: UNCAUGHT on every +1 fixture of both twins
     # (fold_y_periodic_even/odd, fold_xy_mixed_phase, fold_y_wall_x_odd,
     # beta_fold_y_periodic_even/odd, beta_fold_y_wall_x) and CAUGHT on the
     # -1 ones (fold_y_odd_phase, beta_fold_y_odd_phase). So the parity is
     # part of the liveness, and the leg stays REQUIRED-CAUGHT where the
     # mutation can move a word -- which both sweeps still reach.
     "live": lambda family, spec: (
         any(axis == "Y" and int(phase) == -1
             for axis, phase in spec.get("symmetry", ()))
         and tuple(spec["boundaries"])[1] == "periodic"),
     "why": "the D family's far ghost on the component's OWN axis carries the "
            "IDENTITY (the inversion of the B family's -phase); applying a "
            "phase there is the B family's rule worn by the wrong seam. Live "
            "on a folded PERIODIC y whose plane parity is -1: the far pass "
            "runs there, and the applied factor is -1 rather than the exact "
            "identity multiplication by +1 would be."},
    {"name": "cf_electric_far_fill_bakes_n_minus_two",
     "old": "int g1_y_i = idx + (ny - 1 - reflect_y) * sy;",
     "new": "int g1_y_i = idx + (ny - 1 - (ny - 2)) * sy;",
     "family": ("folded_complex_electric", "complex_beta_electric"),
     "expect": "caught_where_live",
     "live": lambda family, spec: (
         any(axis == "Y" for axis, _phase in spec.get("symmetry", ()))
         and tuple(spec["boundaries"])[1] == "periodic"
         and int(round(spec["cell"][1])) % 2 == 1),
     "why": "the reflect row is n_full - stored + 2 -- stored - 2 at an even "
            "full count and stored - 3 at an odd one. Baking n - 2 is exact "
            "on even counts and a whole cell wrong on the odd-count fixtures, "
            "which is why they exist."},
    {"name": "cf_electric_composed_parity_drops_the_near_factor",
     "old": "cf g1_ynx_v = mul_coefficient_left(phase_x, pre1);",
     "new": "cf g1_ynx_v = pre1;",
     "family": ("folded_complex_electric", "complex_beta_electric"),
     "expect": "caught_where_live",
     # LIVE ONLY ON A TWO-PLANE FOLD with the near-x phase odd, which only the
     # folded family's sweep can build (a 2-D beta cell folded on both
     # in-plane axes has no axis left for an active absorber) -- the same
     # narrowing as the B family's composed leg.
     "live_somewhere_on": ("folded_complex_electric",),
     "live": lambda family, spec: (
         any(axis == "X" and int(phase) == -1
             for axis, phase in spec.get("symmetry", ()))
         and any(axis == "Y" for axis, _phase in spec.get("symmetry", ()))
         and tuple(spec["boundaries"])[1] == "periodic"),
     "why": "the corner ghost (far y composed with near x) carries the near "
            "application inside the far one; dropping the near factor leaves "
            "the far identity alone. Live on the mixed-phase two-plane "
            "fixture, where the dropped factor is -1 and the corner value "
            "flips sign."},
    {"name": "cf_electric_ghost_reads_the_wrong_lane",
     "old": "cf g1_y_v = d1;",
     "new": "cf g1_y_v = d0;",
     "family": ("folded_complex_electric", "complex_beta_electric"),
     "expect": "caught_where_live",
     "live": lambda family, spec: (
         any(axis == "Y" for axis, _phase in spec.get("symmetry", ()))
         and tuple(spec["boundaries"])[1] == "periodic"),
     "why": "the far ghost of Dy built from DX'S register: a plausible "
            "complex number in the right cell of the wrong field. The "
            "certified fill images each component from ITSELF."},
    {"name": "skz_electric_near_fill_images_the_wrong_row",
     "old": "int gx_ny_i = idx - 2 * sy;",
     "new": "int gx_ny_i = idx - 1 * sy;",
     "family": ("special_kz_electric",), "expect": "caught_where_live",
     "live": lambda family, spec: any(
         axis == "Y" for axis, _phase in spec.get("symmetry", ())),
     "why": "the real electric carry's near needle: the source row is "
            "stepping.MIRROR_SOURCE_INDEX = 2, and imaging row 1 is a whole "
            "cell wrong."},
    {"name": "skz_electric_far_fill_parity_dropped",
     "old": "gy_y_v = (-phase_y) * gy_y_v;",
     "new": "gy_y_v = phase_y * gy_y_v;",
     "family": ("special_kz_electric",), "expect": "caught_where_live",
     "live": lambda family, spec: (
         any(axis == "Y" for axis, _phase in spec.get("symmetry", ()))
         and tuple(spec["boundaries"])[1] == "periodic"),
     "why": "the real D far ghost on the component's own axis carries -phase "
            "(the shift-1 parity); dropping the negation flips every far "
            "ghost of Dy along y. Live only on a folded PERIODIC y."},
    {"name": "skz_electric_far_fill_bakes_n_minus_two",
     "old": "int gy_y_i = idx + (ny - 1 - reflect_y) * sy;",
     "new": "int gy_y_i = idx + (ny - 1 - (ny - 2)) * sy;",
     "family": ("special_kz_electric",), "expect": "caught_where_live",
     "live": lambda family, spec: (
         any(axis == "Y" for axis, _phase in spec.get("symmetry", ()))
         and tuple(spec["boundaries"])[1] == "periodic"
         and int(round(spec["cell"][1])) % 2 == 1),
     "why": "the reflect row at an odd full count is stored - 3; baking "
            "n - 2 reflects about the window top instead of the mirror, a "
            "whole cell wrong on the odd-count fixture."},
    {"name": "skz_electric_ghost_reads_the_wrong_lane",
     "old": "float gy_y_v = d_y;",
     "new": "float gy_y_v = d_x;",
     "family": ("special_kz_electric",), "expect": "caught_where_live",
     "live": lambda family, spec: (
         any(axis == "Y" for axis, _phase in spec.get("symmetry", ()))
         and tuple(spec["boundaries"])[1] == "periodic"),
     "why": "the far ghost of Dy built from Dx's register: right cell, wrong "
            "field. The certified fill images each component from itself."},
    {"name": "skz_magnetic_near_fill_images_the_wrong_row",
     "old": "int gy_n_i = idx - 2 * sy;",
     "new": "int gy_n_i = idx - 1 * sy;",
     "family": ("special_kz",), "expect": "caught_where_live",
     "live": lambda family, spec: any(
         axis == "Y" for axis, _phase in spec.get("symmetry", ())),
     "why": "the five-pass magnetic weld's ported near fill, armed the way "
            "the complex families' is: the source row is 2 and imaging row 1 "
            "is a whole cell wrong."},
    {"name": "skz_magnetic_far_fill_bakes_n_minus_two",
     "old": "int gx_y_i = idx + (ny - 1 - reflect_y) * sy;",
     "new": "int gx_y_i = idx + (ny - 1 - (ny - 2)) * sy;",
     "family": ("special_kz",), "expect": "caught_where_live",
     "live": lambda family, spec: (
         any(axis == "Y" for axis, _phase in spec.get("symmetry", ()))
         and tuple(spec["boundaries"])[1] == "periodic"
         and int(round(spec["cell"][1])) % 2 == 1),
     "why": "the ported far fill's reflect row, wrong by a cell at an odd "
            "full count -- the defect the odd-count fold fixture exists for."},
    {"name": "skz_magnetic_ghost_reads_the_wrong_lane",
     "old": "float gx_y_v = gx_y_p * b_x;",
     "new": "float gx_y_v = gx_y_p * b_y;",
     "family": ("special_kz",), "expect": "caught_where_live",
     "live": lambda family, spec: (
         any(axis == "Y" for axis, _phase in spec.get("symmetry", ()))
         and tuple(spec["boundaries"])[1] == "periodic"),
     "why": "the far ghost of Bx built from By's register: right cell, wrong "
            "field, through the ported real carry."},
    {"name": "real_ownership_guard_inverted",
     "old": "    if (!owned) return 0.0f;",
     "new": "    if (owned) return 0.0f;",
     "family": ("special_kz", "special_kz_electric"), "expect": "caught",
     "why": "the real-storage spelling of cf_ownership_guard_inverted: the "
            "owner returns before storing and every owned cell keeps its "
            "previous step's flux density. Caught on unfolded fixtures too, "
            "where owned is identically 1."},
)


def _live(mutation: Dict[str, Any], family: Dict[str, Any],
          spec: Dict[str, Any]) -> bool:
    """Is this leg observable on this fixture? Declared per leg, never guessed."""
    if mutation["expect"] == "caught":
        return True
    predicate = mutation.get("live")
    return bool(predicate(family, spec)) if predicate is not None else False


def _score(mutation: Dict[str, Any], live: bool, diverged: bool) -> Tuple[bool, str]:
    """Did this leg do what it declared? ``(passed, verdict)``."""
    expect = mutation["expect"]
    if expect == "inert":
        return (not diverged), ("INERT-AS-DECLARED" if not diverged
                                else "DIVERGED-BUT-DECLARED-INERT")
    if not live:
        return (not diverged), ("NOT-LIVE-AND-INERT" if not diverged
                                else "NOT-LIVE-BUT-DIVERGED")
    return diverged, ("CAUGHT" if diverged else "UNCAUGHT")


def leg_carry_floor(family: Dict[str, Any], spec: Dict[str, Any], steps: int,
                    arm: str) -> Dict[str, Any]:
    """THE ARMED THREE-LEG CLOSURE PROTOCOL for a fill-carrying weld, per fixture.

    Leg A is the main sweep (full carry, 0 differing words, measured by
    :func:`run_case` on every case). This leg is B and C, and each MUST DIVERGE
    on a folded fixture -- a fixture where leg B does not diverge never
    exercised the carry at all and the case is VACUOUS, which is scored as a
    FAILURE here rather than as a pass:

    * **B, the degenerate carry**: the SAME shipped kernel launched with the
      fill plan zeroed (near = 0, reflect = -1) while ``REPLACES`` still claims
      the fills -- the weld's own composition with the carry cut out.
    * **C, no carry at all**: the separate certified route with the two array
      fills SKIPPED -- what the seam would compute if nothing performed them.

    Both are compared against the driver's own order over ``steps`` complete
    steps; the first divergent step is recorded. ON AN UNFOLDED FIXTURE both
    are required INERT instead (the fills are no-ops there), which is what
    stops a degenerate plan from being scored where it means nothing.
    """
    from meep_gpu.cuda_kernels import complex_fill_carry  # noqa: PLC0415

    started = time.time()
    label = f"{family['name']}|{spec['label']}|carry_floor"
    out: Dict[str, Any] = {"family": family["name"], "label": spec["label"]}
    folded = bool(spec.get("symmetry"))

    def sweep(subject_step) -> Tuple[bool, Optional[int], List[int]]:
        reference, _rg, ref_pml, _ = family["build"](spec, "uniform",
                                                     case_rng(label))
        actual, grid, pml, _ = family["build"](spec, "uniform", case_rng(label))
        names = state_names_for(family, reference)
        per_step: List[int] = []
        arguments = family["resolve"](actual, grid, pml, arm)
        for _step in range(1, steps + 1):
            array_step(reference, ref_pml)
            subject_step(actual, grid, pml, arguments)
            for name in DRIVER_ORDER[DRIVER_ORDER.index(
                    seam_span(family)[-1]) + 1:]:
                run_pass(name, actual, pml)
            cp.cuda.runtime.deviceSynchronize()
            per_step.append(sum(compare(reference, actual, names).values()))
            if per_step[-1]:
                break
        diverged = any(per_step)
        first = next((i + 1 for i, n in enumerate(per_step) if n), None)
        return diverged, first, per_step

    def degenerate(actual, grid, pml, arguments):
        before_passes(family, actual, pml)
        cut = dict(arguments)
        cut["fills"] = {"near": (0, 0, 0), "reflect": (-1, -1, -1),
                        "phase": (0.0, 0.0, 0.0)}
        family["launch"](actual, cut)

    def no_carry(actual, grid, pml, arguments):  # noqa: ARG001
        before_passes(family, actual, pml)
        _separate_without_fills(family, actual, grid, pml, arm)

    def _separate_without_fills(family, fields, grid, pml, arm):
        """The certified singles and the wall pass; the two fills SKIPPED."""
        module_separate = family["separate"]
        # A FAMILY MAY CARRY ITS OWN NO-FILLS ROUTE (the residue-round families
        # do -- their certified pieces differ per curl), consulted before the
        # original two-family chain so those two stay byte-for-byte the code
        # the 2026-09-02 records ran.
        hook = family.get("separate_without_fills")
        if hook is not None:
            hook(fields, grid, pml, arm)
            return
        # The separate closure runs the fills through ``stepping``; rather than
        # fork it per family, the two fills are made no-ops for this call by
        # running the closure on a grid the fills skip -- impossible without a
        # proxy -- so the route is spelled here from the same certified pieces.
        if family["name"] == "complex_folded":
            from meep_gpu.cuda_kernels import complex_folded_kernels  # noqa: PLC0415
            from meep_gpu.cuda_kernels import complex_pml_kernels  # noqa: PLC0415
            complex_folded_kernels.step_folded_complex(
                "step_B", fields, arm, grid=grid, pml=pml)
            stepping.zero_metal_B(fields)
            complex_pml_kernels.update_fused_pml_complex("H", fields, arm, pml=pml)
        elif family["name"] == "complex_beta":
            from meep_gpu.cuda_kernels import complex_beta_kernels  # noqa: PLC0415
            from meep_gpu.cuda_kernels import complex_pml_kernels  # noqa: PLC0415
            complex_beta_kernels.step_complex_beta(
                "step_B", fields, arm, grid=grid, pml=pml)
            stepping.zero_metal_B(fields)
            complex_pml_kernels.update_fused_pml_complex("H", fields, arm, pml=pml)
        else:
            raise SystemExit(f"{family['name']} declares no carry floor")

    b_diverged, b_first, b_steps = sweep(degenerate)
    c_diverged, c_first, c_steps = sweep(no_carry)
    out.update({
        "folded_fixture": folded,
        "degenerate_carry_diverges": b_diverged,
        "degenerate_first_divergence": b_first,
        "degenerate_differing_words_per_step": b_steps,
        "no_carry_diverges": c_diverged,
        "no_carry_first_divergence": c_first,
        "no_carry_differing_words_per_step": c_steps,
        # A FOLDED fixture where leg B does not diverge is VACUOUS -- it never
        # exercised the carry -- and vacuous is a FAILURE, never a pass. An
        # unfolded fixture requires the exact opposite: both legs inert.
        "passed": bool((b_diverged and c_diverged) if folded
                       else (not b_diverged and not c_diverged)),
        "seconds": time.time() - started,
    })
    return out


def run_mutations(family: Dict[str, Any], specs: Sequence[Dict[str, Any]],
                  steps: int, arm: str) -> List[Dict[str, Any]]:
    """Every mutation, armed on every fixture, and scored against its DECLARATION.

    EVERY LEG IS RUN EVERYWHERE, including where it is declared not live: a leg that
    is inert where it should be inert is a measurement, and a leg that MOVED A WORD
    where nothing should have moved is a defect this table would otherwise miss. What
    the liveness decides is the EXPECTATION, never whether the run happens.

    A ``caught_where_live`` leg with no live fixture in the scored set FAILS. It would
    be a defect nothing in this sweep could catch, and reporting it as a pass is the
    vacuity the whole track guards against.
    """
    legs: List[Dict[str, Any]] = []

    def arm_leg(mutation: Dict[str, Any], spec: Dict[str, Any],
                case: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
        live = _live(mutation, family, spec)
        diverged = not case.get("weld_bit_identical", False)
        passed, verdict = _score(mutation, live, diverged)
        leg = {"leg": mutation["name"], "label": spec["label"],
               "family": family["name"], "scored": True, "live": live,
               "expect": mutation["expect"], "diverged": diverged,
               "caught": diverged and live, "verdict": verdict, "passed": passed,
               "why": mutation["why"], "scored_on": "weld_bit_identical",
               "first_weld_divergence": case.get("first_weld_divergence"),
               "weld_differing_words": case.get("weld_differing_words"),
               "array_differing_words": case.get("differing_words")}
        leg.update(extra)
        log(f"    mutation {mutation['name']} on {spec['label']}: {verdict}"
            f"{'' if passed else '  <-- LEG FAILED'}")
        return leg

    # HOST LEGS ARE FAMILY-GATED, exactly as the device legs already were. Four of the
    # seven arm a coefficient table or a wall flag that the no-absorber product does
    # not have, and three arm a curl arm and a pole bank that only it has -- so a
    # shared loop would have run four legs that patch a key nothing reads (an INERT
    # leg reporting a pass) and left that product's own two seams unarmed.
    for mutation in HOST_MUTATIONS:
        if family["name"] not in mutation.get("family", tuple(FAMILIES)):
            continue
        for spec in specs:
            _f, _g, pml, _h = family["build"](spec, "uniform",
                                              np.random.default_rng(7))
            case = run_case(family, spec, "uniform", steps, arm,
                            patch=_host_patch(family, mutation["name"], pml, arm),
                            label_suffix=f"|{mutation['name']}")
            legs.append(arm_leg(mutation, spec, case, {"kind": "host"}))
    # DEVICE LEGS ARE COMPILED PER VARIANT, because a product may emit more than one
    # kernel and a fixture drives exactly one of them. Compiling once against the
    # first variant and launching it on every fixture would hand a conductive run the
    # plain weld -- a wrong answer scored as if it were the mutation's.
    for mutation in DEVICE_MUTATIONS:
        if family["name"] not in mutation["family"]:
            continue
        scored_variants = mutation.get("variants")
        compiled: Dict[Any, Any] = {}
        missed: List[Any] = []
        for spec in specs:
            variant = variant_for_spec(family, spec)
            # A LEG MAY DECLARE WHICH VARIANTS IT SCORES ON. The conductive tail's
            # reassociation has no site in the plain weld, so a needle scored there
            # would report NEEDLE MISSED for a leg that is simply not about that
            # kernel -- which is a different statement from a needle that has gone
            # stale, and the two may not be conflated.
            if scored_variants is not None and variant not in scored_variants:
                continue
            if variant not in compiled:
                source, hits = needle(fused_source(family, arm, variant),
                                      mutation["old"], mutation["new"])
                if not hits:
                    missed.append(variant)
                    compiled[variant] = None
                else:
                    compiled[variant] = (
                        compile_source(source, kernel_symbol(family, variant),
                                       GUARD_OPTIONS), hits)
            if compiled[variant] is None:
                continue
            kernel, hits = compiled[variant]
            case = run_case(family, spec, "uniform", steps, arm, kernel=kernel,
                            label_suffix=f"|{mutation['name']}")
            legs.append(arm_leg(mutation, spec, case,
                                {"kind": "device", "needle_sites": hits,
                                 "variant": variant}))
        for variant in missed:
            legs.append({"leg": mutation["name"], "family": family["name"],
                         "variant": variant,
                         "scored": False, "passed": False, "kind": "device",
                         "why_not_scored": (
                             f"the needle {mutation['old']!r} matched nothing in the "
                             f"emitted source for variant {variant!r}; a mutation "
                             f"that changed no byte scores nothing")})
            log(f"    mutation {mutation['name']} [{variant}]: NEEDLE MISSED")

    # A ``caught_where_live`` LEG MUST HAVE BEEN LIVE SOMEWHERE -- checked here, over
    # the scored set, rather than left to a reader of the table. ``live_somewhere_on``
    # narrows that demand to the families where the leg CAN be live; a leg that is a
    # measured null control on the other family would otherwise have to be either
    # dropped there (losing the control) or declared live (a false statement).
    for mutation in tuple(
            m for m in HOST_MUTATIONS
            if family["name"] in m.get("family", tuple(FAMILIES))) + tuple(
            m for m in DEVICE_MUTATIONS
            if family["name"] in m["family"]):
        if mutation["expect"] != "caught_where_live":
            continue
        if family["name"] not in mutation.get("live_somewhere_on",
                                              tuple(FAMILIES)):
            continue
        scored = [leg for leg in legs if leg["leg"] == mutation["name"]
                  and leg.get("scored")]
        if scored and not any(leg["live"] for leg in scored):
            for leg in scored:
                leg["passed"] = False
                leg["verdict"] = "NEVER-LIVE-IN-THIS-SWEEP"
            log(f"    mutation {mutation['name']}: NEVER LIVE in this sweep "
                f"<-- LEG FAILED")
    return legs


# ---------------------------------------------------------------------------
# The licence and the policy
# ---------------------------------------------------------------------------

LICENCE: Dict[str, Any] = {"verdict": None}
POLICY: Dict[str, Any] = {"name": None}


def load_licence(path: Optional[str], policy: Optional[str]) -> Dict[str, Any]:
    """Read the probe artifact, run the licence, and report the whole verdict.

    THE ARBITER IS ``complex_fields.expansion_license`` AND IS NOT REIMPLEMENTED.
    What this adds is the two things a verdict cannot know: which artifact was read,
    and which policy the consuming run installs.
    """
    out: Dict[str, Any] = {"path": path, "policy_required": policy,
                           "record_sha256": None, "verdict": None,
                           "policy_reasons": [], "certification_reasons": [],
                           "arm": None, "usable": False}
    if not path:
        path = os.environ.get(complex_fields.PROBE_PATH_ENVIRONMENT)
        out["path"] = path
    if not path:
        out["policy_reasons"] = [
            "no expansion probe artifact was given and "
            f"{complex_fields.PROBE_PATH_ENVIRONMENT} is unset; the arm is a "
            "measured platform fact and may not be guessed"]
        return out
    try:
        with open(path, "rb") as handle:
            raw = handle.read()
    except OSError as exc:
        out["policy_reasons"] = [f"the probe artifact could not be read: {exc}"]
        return out
    out["record_sha256"] = hashlib.sha256(raw).hexdigest()
    try:
        record = json.loads(raw.decode("utf-8"))
    except Exception as exc:  # noqa: BLE001 - unreadable is a refusal
        out["policy_reasons"] = [f"the probe artifact is not readable JSON: {exc}"]
        return out
    verdict = complex_fields.expansion_license(record)
    out["verdict"] = verdict
    out["policy_reasons"] = list(
        complex_fields.expansion_policy_reasons(record, policy))
    out["certification_reasons"] = list(
        complex_fields.expansion_certification_reasons(policy))
    out["arm"] = verdict.get("arm")
    out["basis"] = verdict.get("basis")
    out["usable"] = bool(verdict.get("arm") and not verdict.get("refusals")
                         and not out["policy_reasons"])
    return out


# ---------------------------------------------------------------------------
# Driving
# ---------------------------------------------------------------------------

def save(results: Dict[str, Any], out_path: str) -> None:
    """Re-stamp and rewrite the artifact. Called after EVERY case, not at the end.

    THE STAMP IS TAKEN HERE, at the write, rather than once at startup: every write
    site must record what the tree looked like when it was written, and an artifact
    rewritten twenty times from a stamp taken once would carry provenance for a tree
    the later writes did not run against. The temp-and-replace is what keeps an
    interrupted run's artifact readable rather than truncated.
    """
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    gate_provenance.stamp(results)
    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, sort_keys=True, default=str)
    os.replace(tmp, out_path)


def summarize(results: Dict[str, Any], steps: int) -> Dict[str, Any]:
    cases = results.get("cases", [])
    mutations = results.get("mutations", [])
    scored = [m for m in mutations if m.get("scored")]
    caught = [m for m in scored if m.get("caught")]
    live = [m for m in scored if m.get("live")]
    failed_legs = [f"{m['leg']}|{m.get('label')}:{m.get('verdict', 'UNSCORED')}"
                   for m in mutations if not m.get("passed")]
    census_ok = {
        "band_contains_subnormals": all(
            c["operand_census"]["subnormals"] > 0 for c in cases
            if c.get("value_class") == "subnormal_band" and "operand_census" in c),
        "uniform_contains_none": all(
            c["operand_census"]["subnormals"] == 0 for c in cases
            if c.get("value_class") == "uniform" and "operand_census" in c),
    }
    inherited = [c["case_seed_label"] for c in cases
                 if c.get("divergence_is_inherited_from_a_half")]
    verdict = {
        "steps_per_case": steps,
        "cases": len(cases),
        "cases_passed": sum(1 for c in cases if c.get("passed")),
        "cases_bit_identical_to_the_array_path": sum(
            1 for c in cases if c.get("bit_identical")),
        "cases_weld_bit_identical": sum(
            1 for c in cases if c.get("weld_bit_identical")),
        # THE TWO CLAIMS, KEPT APART. `weld_released` is this product's own: one
        # launch reproduces, word for word, the certified sequence it replaces, on
        # every case and every operand class. `array_identity_released` is the full
        # claim, and it can only hold where the CERTIFIED HALVES themselves reproduce
        # the array path -- so the two are reported separately and a case whose
        # divergence is proved inherited names the half rather than this weld. Under a
        # correctly installed policy both hold on every fixture here; the split is
        # what made that provable rather than assumed.
        "cases_with_a_divergence_inherited_from_a_certified_half": inherited,
        "mutations_scored": len(scored),
        "mutations_live": len(live),
        "mutations_caught": len(caught),
        "mutations_caught_of_live": f"{len(caught)}/{len(live)}",
        "mutation_legs_that_failed": failed_legs,
        "mutations_unscored": [m["leg"] for m in mutations if not m.get("scored")
                               and not m.get("passed")],
        "operand_census_preconditions": census_ok,
        "refusal_leg_passed": bool(results.get("refusal", {}).get("passed")),
        "lift_leg_passed": bool(results.get("lift", {}).get("passed")),
        "driver_order_passed": bool(results.get("driver_order", {}).get("passed")),
        "separate_controls_passed": all(
            c.get("passed") for c in results.get("separate_controls", [])),
        "deposit_legs_passed": all(d.get("passed")
                                   for d in results.get("deposit", [])),
        # THE COMPLEMENT LEG, and it is required to have RUN on a family that declares
        # no repair. A product with CARRIES_DEPOSIT_REPAIR False and no source-refusal
        # leg would have neither half of the deposit question measured: not the carry
        # (it carries none) and not the refusal (nothing asked). ``None`` means the
        # family does not need it; an empty list on a family that does is a FAILURE.
        "source_refusal_legs_passed": (
            None if not results.get("source_refusal_expected")
            else bool(results.get("source_refusal")
                      and all(d.get("passed")
                              for d in results.get("source_refusal", [])))),
        # THE CARRY FLOOR (fill-carrying families only): every folded fixture's
        # degenerate-carry and no-carry legs DIVERGED (a case where they do not
        # is vacuous and fails), every unfolded fixture's were inert, and at
        # least one folded fixture was scored.
        "carry_floor_legs_passed": (
            None if not results.get("carry_floor_expected")
            else bool(results.get("carry_floor")
                      and all(d.get("passed")
                              for d in results.get("carry_floor", []))
                      and any(d.get("folded_fixture")
                              for d in results.get("carry_floor", [])))),
    }
    #: THE WELD'S OWN CLAIM: one launch reproduces, word for word, the certified
    #: sequence it replaces, on every case and every operand class -- with every
    #: mutation caught on that same comparison, every refusal standing, and the launch
    #: counters agreeing that the fused kernel and nothing else ran.
    verdict["weld_released"] = bool(
        cases and verdict["cases_passed"] == len(cases)
        and verdict["cases_weld_bit_identical"] == len(cases)
        # EVERY LEG MUST MATCH ITS DECLARATION -- caught where live, inert where
        # declared inert, and live somewhere for every caught_where_live leg. Not
        # "every leg diverged": a wall mutation on an unwalled grid changes no byte
        # by construction, and requiring a divergence there would be requiring the
        # kernel to write a plane nobody asked for.
        and scored and live and not failed_legs
        and not verdict["mutations_unscored"]
        and all(census_ok.values())
        and verdict["refusal_leg_passed"] and verdict["lift_leg_passed"]
        and verdict["driver_order_passed"] and verdict["separate_controls_passed"]
        and verdict["deposit_legs_passed"]
        and verdict["source_refusal_legs_passed"] is not False
        and verdict["carry_floor_legs_passed"] is not False)
    #: THE FULL CLAIM: the fused launch also reproduces ``stepping``'s own eleven
    #: passes. It can only hold where the CERTIFIED HALVES do, so it is reported
    #: separately rather than folded into the weld's verdict -- and where it is False
    #: with ``weld_released`` True, ``cases_with_a_divergence_inherited_from_a_certified_half``
    #: names every case and the record carries both word maps.
    verdict["array_identity_released"] = bool(
        verdict["weld_released"] and not inherited
        and verdict["cases_bit_identical_to_the_array_path"] == len(cases))
    verdict["released"] = verdict["array_identity_released"]
    return verdict


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=sorted(FAMILIES), required=True)
    parser.add_argument("--product", choices=("full", "reduced"), default="full")
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--subnormal-policy", default=None)
    parser.add_argument("--import-meep-for-host-policy", action="store_true")
    parser.add_argument("--expansion-probe", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--skip-mutations", action="store_true")
    args = parser.parse_args(argv)

    started = time.time()
    results: Dict[str, Any] = {
        "gate": "cuda_fused_complex_pairs",
        "family": args.family,
        # MACHINE STAMPS, not prose: a certification block reads the start time
        # and the host off this payload, and test_certification_metadata is what
        # requires both to have come from the artifact rather than from memory.
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "host": __import__("socket").gethostname(),
        "argv": list(argv or sys.argv[1:]),
        "steps": args.steps,
        "subnormal_policy": args.subnormal_policy,
    }
    gate_provenance.stamp(results)
    POLICY["name"] = args.subnormal_policy

    # THE POLICY IS INSTALLED HERE, BEFORE ANY COMPILE, and passing it as a NAME
    # would not have been enough. CuPy appends ``-ftz=true`` to every NVRTC compile
    # (compiler.py:552) and ``subnormal_policy`` strips it at that seam; a run that
    # only carried the name would measure FLUSHED bytes and stamp them ``keep``.
    # ``flush`` additionally needs the HOST half, which only a MEEP import can attain
    # -- the package may not import MEEP, a parity gate may, and it does so only on
    # request.
    if args.import_meep_for_host_policy:
        results["meep_import_for_host_policy"] = probe.import_meep_for_host_policy()
    if args.subnormal_policy and cp is not None:
        results["subnormal_policy_install"] = probe.install_subnormal_policy_for_run(
            args.subnormal_policy, _REPO_API)
    elif args.subnormal_policy:
        results["subnormal_policy_install"] = {
            "installed": False, "policy": args.subnormal_policy,
            "why": "no CuPy on this host: the policy is installed at CuPy's NVRTC "
                   "seam and there is nothing here to install it at. The NAME is "
                   "still carried because the licence clause is a policy comparison"}
    results["subnormal_policy_stamp"] = probe.subnormal_policy_stamp(_REPO_API)

    licence = load_licence(args.expansion_probe, args.subnormal_policy)
    results["licence"] = licence
    LICENCE["verdict"] = licence["verdict"]

    family = FAMILIES[args.family]()
    results["module"] = {
        "family": family["module"].FAMILY,
        "kernel": family["module"].KERNEL_NAME,
        "kernel_symbols": [kernel_symbol(family, variant)
                           for variant in variants_of(family)],
        "variants": [str(variant) for variant in variants_of(family)],
        "replaces": list(family["module"].REPLACES),
        "carries_deposit_repair": bool(family["module"].CARRIES_DEPOSIT_REPAIR),
        "certified_kernels": list(family["module"].CERTIFIED_KERNELS),
        "uncertified_kernels": dict(family["module"].UNCERTIFIED_KERNELS),
        "lift_edits": len(family["module"].LIFT_EDITS),
    }
    results["driver_order"] = leg_driver_order(family)
    log(f"driver order: {'PASS' if results['driver_order']['passed'] else 'FAIL'}")
    # THE LIFT LEG RUNS BEFORE THE DEVICE CHECK, deliberately: it is a claim about
    # SOURCE TEXT, it needs no CuPy, and a splice that stopped being the certified
    # arithmetic should be caught at the merge bar rather than only on the one host
    # with a GPU.
    results["lift"] = leg_lift(family)
    log(f"lift: {'PASS' if results['lift']['passed'] else 'FAIL'}")
    save(results, args.out)

    if cp is None:
        results["verdict"] = {"released": False,
                              "why": "CuPy is absent; only the source legs ran"}
        save(results, args.out)
        return 1
    if not licence["usable"]:
        results["verdict"] = {"released": False,
                              "why": f"the expansion licence is unusable: "
                                     f"{licence['policy_reasons'] or licence['verdict']}"}
        save(results, args.out)
        return 1
    arm = licence["arm"]
    results["arm"] = arm
    log(f"arm {arm} under policy {args.subnormal_policy}")

    specs = family["specs"]
    if args.product == "reduced":
        specs = tuple(s for s in specs if s["label"] in family["reduced_labels"])
    warm_memo(family, specs, arm)
    log(f"memo warm on {len(specs)} specs")

    results["refusal"] = leg_refusal(family, arm)
    log(f"refusal: {'PASS' if results['refusal']['passed'] else 'FAIL'}")
    save(results, args.out)

    cases: List[Dict[str, Any]] = []
    results["cases"] = cases
    for spec in specs:
        for value_class in VALUE_CLASSES:
            case = run_case(family, spec, value_class, args.steps, arm)
            cases.append(case)
            log(f"  case {spec['label']}/{value_class}: "
                f"{'PASS' if case.get('passed') else 'FAIL'} "
                f"steps={case.get('steps_run')} "
                f"differing={case.get('differing_words')} "
                f"launches={case.get('launch_counts', {}).get('launcher_reports')} "
                f"({case.get('seconds', 0.0):.1f} s)")
            save(results, args.out)

    controls: List[Dict[str, Any]] = []
    results["separate_controls"] = controls
    for spec in [s for s in specs if s["label"] in family["reduced_labels"]]:
        control = leg_separate_control(family, spec, min(12, args.steps), arm)
        controls.append(control)
        log(f"  separate control {spec['label']}: "
            f"{'PASS' if control['passed'] else 'FAIL'} "
            f"separate={control['separate_launch_total']} "
            f"fused={control['fused_launch_total']}")
        save(results, args.out)

    floors: List[Dict[str, Any]] = []
    results["carry_floor"] = floors
    results["carry_floor_expected"] = bool(family.get("carry_floor"))
    if family.get("carry_floor"):
        for spec in specs:
            leg = leg_carry_floor(family, spec, min(12, args.steps), arm)
            floors.append(leg)
            log(f"  carry floor {spec['label']}: "
                f"{'PASS' if leg['passed'] else 'FAIL'} "
                f"folded={leg['folded_fixture']} "
                f"degenerate_diverges={leg['degenerate_carry_diverges']} "
                f"no_carry_diverges={leg['no_carry_diverges']}")
            save(results, args.out)

    deposits: List[Dict[str, Any]] = []
    results["deposit"] = deposits
    for spec in [s for s in specs if s["label"] in family["deposit_labels"]]:
        for repaired in (True, False):
            leg = leg_deposit(family, spec, min(12, args.steps), arm, repaired)
            deposits.append(leg)
            log(f"  deposit {spec['label']} repaired={repaired}: "
                f"{'PASS' if leg['passed'] else 'FAIL'} "
                f"identical={leg.get('bit_identical')}")
            save(results, args.out)

    # THE OTHER HALF OF THE DEPOSIT QUESTION, for a product that declares no repair.
    # A family with `deposit_labels` carries the deposit and proves the carry; a family
    # with `source_refusal_labels` refuses it and proves BOTH the refusal and that the
    # refusal has a consequence. Every family must do one or the other, which
    # `source_refusal_expected` is what makes checkable.
    refusals: List[Dict[str, Any]] = []
    results["source_refusal"] = refusals
    results["source_refusal_expected"] = bool(family.get("source_refusal_labels"))
    for spec in [s for s in specs
                 if s["label"] in family.get("source_refusal_labels", ())]:
        leg = leg_source_refusal(family, spec, min(12, args.steps), arm)
        refusals.append(leg)
        log(f"  source refusal {spec['label']}: "
            f"{'PASS' if leg['passed'] else 'FAIL'} "
            f"electric_refused={leg['electric_source']['passed']} "
            f"magnetic_admitted={leg['magnetic_source']['passed']} "
            f"unrepaired_diverges={leg.get('unrepaired_span_diverges')}")
        save(results, args.out)

    if not args.skip_mutations:
        mutation_specs = [s for s in specs
                          if s["label"] in family["mutation_labels"]]
        results["mutations"] = run_mutations(family, mutation_specs,
                                             min(12, args.steps), arm)
        save(results, args.out)
    else:
        results["mutations"] = []

    results["verdict"] = summarize(results, args.steps)
    results["seconds"] = time.time() - started
    save(results, args.out)
    v = results["verdict"]
    log(f"VERDICT weld_released={v['weld_released']} "
        f"array_identity_released={v['array_identity_released']} "
        f"cases={v['cases_passed']}/{v['cases']} "
        f"weld_identical={v['cases_weld_bit_identical']}/{v['cases']} "
        f"array_identical={v['cases_bit_identical_to_the_array_path']}/{v['cases']} "
        f"mutations caught={v['mutations_caught_of_live']} live "
        f"scored={v['mutations_scored']} failed={len(v['mutation_legs_that_failed'])}")
    if v["mutation_legs_that_failed"]:
        log("FAILED LEGS: " + ", ".join(v["mutation_legs_that_failed"]))
    if v["cases_with_a_divergence_inherited_from_a_certified_half"]:
        log("INHERITED (a certified half, not this weld): "
            + ", ".join(v["cases_with_a_divergence_inherited_from_a_certified_half"]))
    # EXIT 0 ON THE WELD'S OWN CLAIM. The full array identity is reported and is what
    # `released` records; it depends on a half this product does not own, and an exit
    # code that conflated the two would report a defect in complex_pml_kernels as a
    # failure of this weld.
    return 0 if v["weld_released"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
