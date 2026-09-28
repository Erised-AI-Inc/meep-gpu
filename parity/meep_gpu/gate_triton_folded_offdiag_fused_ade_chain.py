#!/usr/bin/env python3
"""CUDA byte gate for the FOLDED OFF-DIAGONAL DISPERSIVE E->P chain.

WHAT IS BEING CERTIFIED, and it is not a sub-step. This product spans TWO driver
passes — ``update_E`` at driver.py:3304 and ``update_P`` at :3306, with nothing
between them — over the folded, tensor-epsilon, dispersive constitutive body. A
per-sub-step comparison could not see it at all: the whole claim is about the
SEAM. The comparison is therefore per COMPLETE SEAM STEP over a stated budget,
against

  * the CuPy array path (``stepping.update_E`` then ``stepping.update_P``), and
  * the SEPARATELY CERTIFIED Triton products this one replaces — the E half
    (``folded_offdiag_dispersive_update_e.plan_folded_offdiag_dispersive_
    constitutive``) and the ADE half (``launch.plan_ade_update_p``, one plan per
    susceptibility) dispatched as separate launches,

over every allocated volume, compared as uint32 WORDS. ``allclose`` appears
nowhere, and the FIRST divergent step is reported rather than a final pass/fail.

THE EIGHT THINGS THIS GATE REFUSES TO LET PASS SILENTLY:

1. **A silent fallback.** Bytes alone cannot prove the fused path ran: a plan that
   was never launched is byte-identical to the oracle BY CONSTRUCTION. Every case
   asserts the exact LAUNCH COUNT through a proxy that owns the kernel object, a
   nonzero ``runs``, one alias check per launch, and a NON-VACUITY FLOOR SPLIT BY
   ROLE — the arrays this seam WRITES (E, f_w_E, every polarization volume) must
   all have moved, and the arrays it only READS (B, D, H, fu_*, the material, the
   six off-diagonal row volumes) must NOT have.
2. **A hollow pass.** Source mutations and HOST mutations are armed, and each is
   scored CAUGHT or NULL CONFIRMED with its evidence; a mutant that compiled to
   PTX already present for the shipped kernel is reported PTX-IDENTICAL rather
   than as a passing null. Leg ``disarm`` reruns the identical case with the
   shipped bytes and requires zero.
3. **A mutant that could not compile.** This kernel calls THREE JIT helpers and
   compares against THREE constexprs that live in the two certified folded
   off-diagonal modules, and a mutant module compiled under a bare
   ``import triton`` header would ``NameError`` on every one of them — scoring
   every kernel mutation UNCOMPILABLE while reporting nothing. :func:`compile_
   mutant` writes those six names into the mutant's header, by import, from the
   modules the shipped kernel resolves them through.
4. **A dead mutation.** The E chain and the ADE arms are unrolled by ``NP``
   constexprs, so a rewrite of a slot the scored case does not carry is emitted
   nowhere and measures nothing. Every armed mutation declares the SLOT it edits
   and :func:`leg_no_dead_mutations` refuses any whose slot its own case does not
   reach.
5. **A mirrored evaluator.** :func:`leg_transcription` PARSES the certified body
   out of ``folded_offdiag_dispersive_update_e.py`` and the recurrence out of
   ``kernels.py``, and requires the first to be an AST-IDENTICAL PREFIX of this
   kernel's body and the second to be present per arm modulo the six declared
   renames. It re-implements nothing.
6. **A vacuous value class.** ``uniform`` is the physical band; ``pm_zero_lattice``
   is exempt from the movement floor (a zero state is a fixed point of this seam)
   and must instead carry BOTH signs of zero in the reference output;
   ``subnormal_band`` is COMPARED rather than refused — Triton has both float32
   policies and CuPy's flush is strippable — and its census must FIRE or the row
   is VACUOUS, not passed.
7. **An unmeasured shape.** The whole licence for this product is that one scratch
   per driven component makes the launch's write set disjoint from its read set at
   every position of the rotation's orbit — the six row volumes the E half gathers
   at neighbour offsets INCLUDED in the read inventory. :func:`leg_shape` walks the
   orbit to closure THROUGH THE SHIPPED PLAN with the alias check armed, and
   :func:`leg_plan_refusals` breaks the chain on purpose and requires a refusal.
8. **A verdict that cannot fail.** ``--plant`` rewrites the SHIPPED kernel source
   for the whole process and the release verdict must come back FAIL.

BOTH FLOAT32 SUBNORMAL POLICIES, ONE PER PROCESS. ``--subnormal-policy`` drives
host, CuPy and Triton before the first device compile, and the runner
(``run_triton_folded_offdiag_fused_ade_chain_direct.sh``) executes this gate twice
through the fleet driver with distinct CuPy and Triton cache directories. One
process per policy is not fastidiousness: CuPy computes its kernel cache key ABOVE
the seam where the ``-ftz=true`` strip installs, so a cache shared between the two
policies serves flushed binaries under the keeping record's name.

THE ARTIFACT IS WRITTEN IN BOTH SHAPES. ``verdict``/``rows``/``counts`` is what a
reader and ``rebind_triton_welds`` read; ``device_status``/``release``/``passed``
is the shape ``build_triton_fusion_matrix``'s GATE_BOUND branch reads, and the
``environment`` block carries hostname, device, device_name and
``compute_capability`` because ``seed_triton_welds`` refuses to mint a ledger
entry from an artifact that names neither the device nor the architecture its PTX
was generated for.

Rule 7: one flushed line per case, every row appended and fsynced as it lands, and
the artifact re-written (provenance-stamped) after every leg.

Usage::

    # laptop, no CUDA and no Triton — the legs that need no device
    KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. python -u \\
        parity/meep_gpu/gate_triton_folded_offdiag_fused_ade_chain.py --no-device \\
        --out <scratch>/no_device.json

    # CUDA host, verified-empty device
    CUDA_VISIBLE_DEVICES=<verified-empty device> python -u \\
        parity/meep_gpu/gate_triton_folded_offdiag_fused_ade_chain.py \\
        --subnormal-policy keep --out <fresh-dir>/gate.json
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import argparse
import ast
import dataclasses
import hashlib
import importlib.util
import inspect
import json
import os
import re
import socket
import sys
import tempfile
import textwrap
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
API_ROOT = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
for _path in (HERE, API_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

#: Complete seam steps every product case walks. SIXTY, not one: ``f_w_E`` and the
#: whole ADE history are state carried between steps and the rotation's own period
#: is three, so the defects this gate exists for COMPOUND. The comparison is per
#: COMPLETE STEP, so a green row means identical at ALL sixty — and step 1 is
#: reported separately, because "identical at one launch" and "identical at sixty"
#: are different claims and both are made here.
STEPS = 60

#: Steps per armed mutation. A mutation needing more than this to become
#: byte-visible is reported as a null WITH its launch and PTX evidence.
MUTATION_STEPS = 3

#: Base for every per-case seed. The seed itself is ``SEED + sha256(label|leg)``,
#: never ``hash()``: Python salts ``hash()`` of a string with ``PYTHONHASHSEED``,
#: so a hash-seeded gate draws a different fixture every process and a failing case
#: cannot be replayed.
SEED = 91_000

#: The S2 sweep's block sizes. BLOCK is a constexpr, so each size is a DIFFERENT
#: compiled program over the same arithmetic; a body whose answer moved with the
#: schedule would be a reordering this weld introduced, and the certified E half
#: is only certified at its own default.
BLOCK_SIZES: Tuple[int, ...] = (128, 256, 512, 1024)

STATE_NAMES: Tuple[str, ...] = tuple(
    [f"{stem}{axis}" for stem in ("B", "D", "E", "H") for axis in "xyz"]
    + [f"fu_{stem}{axis}" for stem in ("B", "D") for axis in "xyz"]
    + [f"f_w_{stem}{axis}" for stem in ("E", "H") for axis in "xyz"])

E_NAMES: Tuple[str, str, str] = ("Ex", "Ey", "Ez")

#: The arrays this seam WRITES and the ones it only READS. The movement floor is
#: split on exactly this line: demanding movement from all of them fails a correct
#: run, and demanding it from none passes a run that launched nothing. ``row`` is
#: on the READ side and that is the addition over the plain chain — the six
#: off-diagonal coefficient volumes are gathered at NEIGHBOUR offsets while the
#: ADE outputs are being written, so a launch that disturbed one would make the
#: answer depend on the block schedule.
WRITTEN_PREFIXES = ("E", "f_w_E", "P[", "P_prev[", "scratch[")
READ_ONLY_PREFIXES = ("B", "D", "H", "fu_", "f_w_H", "eps", "inv_eps", "row")

_TEMPORARY: List[str] = []

#: The float32 subnormal policy this process INSTALLED, set once in :func:`main`
#: before the first device compile. Two legs read it, and both would otherwise
#: assert a policy-dependent fact as if it were policy-free.
_POLICY: Optional[str] = None

#: Legs whose failure flips the RELEASE verdict. The corpus-lift leg is carried
#: for a later round and is deliberately not among them: neither Triton chain gate
#: has one, the row's ADMISSION is measured by the census on the lifted objects,
#: and its bytes by the scaled shape case below. A non-gating leg that fails is
#: still reported, under its own key, rather than silently dropped.
RELEASE_GATING_LEGS = ("transcription", "shape", "refusals", "mutation",
                       "product", "separate_control", "block_sweep",
                       "value_classes", "disarm")


@dataclasses.dataclass(frozen=True)
class Case:
    """One configuration of the cell: a fold, a tensor row set and poles."""

    label: str
    fold_axis: str = "Y"
    fold_phase: int = +1
    poles: int = 1
    #: "scalar" | "volume" | "mixed" (scalar state 0, volume state 1) |
    #: "one_component" (Ey/Ez un-driven by a 0.0 sigma)
    sigma: str = "scalar"
    kind: str = "lorentzian"            # lorentzian | drude | mixed
    #: "all" (six slots) | "one" (the Ex/Ey slot alone) | "none" (the diagonal
    #: family's product, which this one must refuse)
    rows: str = "all"
    cell: Tuple[float, float, float] = (1.3, 1.6, 0.9)
    dimensions: int = 3
    boundaries: Any = None              # None is periodic on every axis
    complex_storage: bool = False
    storage: str = "pml"                # "pml" (f_w allocated) | "field"
    value_class: str = "uniform"
    scale: float = 1.0


#: THE CASE MATRIX. Eight configurations of ONE cell, chosen so that every
#: constexpr this launch specialises on takes both of its values somewhere: the
#: fold axis (X and Y), its parity (+1 and -1), the row mask (all six slots and
#: one), the driven-component count (1 and 3), the pole count (1 and 2 =
#: CHAIN_MAX_POLES), the sigma kind (scalar, volume, and both in one launch), the
#: boundary codes (periodic, mirror and metallic) and the wall mask.
#:
#: THE LAST ROW IS THE CORPUS ROW'S OWN SHAPE. ``examples:absorbed_power_density
#: .py`` is the single seam-instance this product exists for, and the census
#: records its boundary kinds as ``[periodic, mirror, periodic]`` — so that is what
#: is carried here, at the row's 2-D cell, rather than the E-half gate's metallic
#: variant of the same script.
CASES: Tuple[Case, ...] = (
    Case("one_state_all_components"),
    Case("one_state_one_component", sigma="one_component"),
    Case("one_state_all_rows_odd_fold", fold_phase=-1),
    Case("one_state_volume_sigma", sigma="volume"),
    Case("two_states_mixed_sigma", poles=2, sigma="mixed", kind="mixed"),
    Case("three_d_fold_x", fold_axis="X", cell=(1.6, 1.3, 0.9)),
    Case("metallic_terminated_fold", poles=2, sigma="mixed", kind="mixed",
         boundaries={"x": "metallic", "y": "periodic", "z": "metallic"}),
    Case("absorbed_power_density_shape", cell=(2.4, 2.4, 0.0), dimensions=2,
         boundaries={"x": "periodic", "y": "periodic", "z": "periodic"}),
)

#: The case every mutation is scored on unless it names another. TWO poles with a
#: MIXED sigma, so both ADE slots are live on every component, slot 0 binds a
#: scalar and slot 1 a volume, and the two states carry DIFFERENT recurrence
#: triples — which is what keeps the coefficient and sigma host mutations from
#: being nulls under another name.
MUTATION_CASE = "two_states_mixed_sigma"

#: The case the wall-mask mutation is scored on: the row sum is masked at a
#: METALLIC wall and nowhere else, so on a periodic-and-mirrored grid the edit
#: would rewrite a real line that computes the same number.
MUTATION_CASE_METALLIC = "metallic_terminated_fold"


def log(message: str) -> None:
    print(message, flush=True)


def case_seed(label: str, leg: str) -> int:
    """A replayable per-case seed: ``SEED + sha256(label|leg)``, never ``hash()``."""
    digest = hashlib.sha256(f"{label}|{leg}".encode("utf-8")).digest()[:4]
    return SEED + int.from_bytes(digest, "big")


def sha256(path: str) -> str:
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def source_hashes() -> Dict[str, str]:
    """The bytes under certification, curated.

    EIGHTEEN PATHS, AND THE TWO E-HALF MODULES ARE WHY THIS PRODUCT IS BOARD-BOUND
    BY ITS GATE RATHER THAN BY A FINGERPRINT. ``seed_triton_welds.curated_for``
    pins the family module, the SHARED set and the parity scripts naming the stem;
    it does not know that this kernel EXECUTES ``_load_d_minus_p`` and
    ``_folded_dispersive_term`` out of ``folded_offdiag_dispersive_update_e.py``
    and ``_masked_row_sum`` out of ``folded_offdiag_update_e.py``. The board
    re-hashes every path recorded here on every cut, so naming them here is what
    makes an edit to either one withdraw this product's credit.

    See ``gate_provenance``'s docstring for why this is kept apart from
    ``imported_source_sha256``: this is the curated claim, that is the measurement.
    """
    names = (
        "meep_gpu/driver.py", "meep_gpu/stepping.py", "meep_gpu/dispersion.py",
        "meep_gpu/fields.py", "meep_gpu/subnormal_policy.py",
        "meep_gpu/triton_kernels/coverage.py",
        "meep_gpu/triton_kernels/kernels.py",
        "meep_gpu/triton_kernels/launch.py",
        "meep_gpu/triton_kernels/symmetry.py",
        "meep_gpu/triton_kernels/offdiag_update_e.py",
        "meep_gpu/triton_kernels/folded_offdiag_update_e.py",
        "meep_gpu/triton_kernels/dispersive_update_e.py",
        "meep_gpu/triton_kernels/folded_offdiag_dispersive_update_e.py",
        "meep_gpu/triton_kernels/folded_offdiag_fused_ade_chain.py",
        "meep_gpu/test_triton_folded_offdiag_fused_ade_chain.py",
        "parity/meep_gpu/gate_provenance.py",
        # The module that decides the SPELLING of the identity block the minting
        # tool reads. It is pinned for the same reason the two E-half modules are:
        # an edit there changes what this artifact says about the device, and a
        # credit that survives such an edit rests on a record nothing re-checked.
        "parity/meep_gpu/triton_device_identity.py",
        "parity/meep_gpu/gate_triton_folded_offdiag_fused_ade_chain.py",
    )
    return {name: sha256(os.path.join(API_ROOT, name)) for name in names}


def save(payload: Dict[str, Any], path: str) -> None:
    """Serialise the payload, provenance-stamped, atomically.

    THE STAMP IS HERE AND NOT AT THE CALL SITES. ``main`` calls this after every
    leg, and a stamp bolted onto the last call would leave every partial artifact —
    including the one an aborted run leaves behind, which is the artifact a failure
    is read from — unattributable. The POLICY stamp is RE-READ rather than carried:
    taken once at install time it records every executor counter at zero, because
    nothing had compiled yet.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    temporary = path + ".tmp"
    if "subnormal_policy" in payload:
        try:
            from meep_gpu import subnormal_policy as _policy  # noqa: PLC0415

            payload["subnormal_policy"] = _policy.policy_stamp()
        except Exception as exc:  # noqa: BLE001
            payload["subnormal_policy_reread_error"] = repr(exc)
    from gate_provenance import stamp as _stamp_provenance  # noqa: PLC0415

    _stamp_provenance(payload)  # bytes THIS process imported; see gate_provenance
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


# ---------------------------------------------------------------------------
# The configuration
# ---------------------------------------------------------------------------

def _sigma_for(case: Case, order: int, shape, rng, xp) -> Any:
    """One susceptibility's sigma, in the form the case declares.

    THE PER-COMPONENT VALUES ARE DISTINCT ON PURPOSE. Two host mutations — the
    sigma bound for the wrong component, and the inverse epsilon bound for the
    wrong one — are NULLS wherever the three components share an object, and a
    null scored as an uncaught defect is the worst reading available to either.
    """
    base = {"Ex": 0.31 + 0.05 * order, "Ey": 0.24 + 0.04 * order,
            "Ez": 0.19 + 0.06 * order}
    if case.sigma == "one_component":
        return {"Ex": base["Ex"], "Ey": 0.0, "Ez": 0.0}
    volume = case.sigma == "volume" or (case.sigma == "mixed" and order == 1)
    if not volume:
        return base
    return {name: xp.asarray(np.ascontiguousarray(
        (value * (0.7 + 0.6 * rng.random(shape))).astype(np.float32)))
        for name, value in base.items()}


def build(xp, case: Case, seed: int, scale: Optional[float] = None):
    """One seeded engine. Called once per route, identically, from one seed.

    EVERY VOLUME IS FILLED WITH PHYSICAL-BAND VALUES unless a value class says
    otherwise, and that is the first thing this gate establishes rather than an
    afterthought: a zero-initialised state is a FIXED POINT of this seam — with
    D = P = P_prev = 0 the constitutive gives E = 0 and the recurrence gives P = 0
    forever — so a deliberately wrong kernel compared against a zero oracle reports
    IDENTICAL.
    """
    from meep_gpu.dispersion import (  # noqa: PLC0415
        DRUDE, LORENTZIAN, PolarizationState, Susceptibility,
    )
    from meep_gpu.fields import Fields  # noqa: PLC0415
    from meep_gpu.grid import Grid, Mirror  # noqa: PLC0415
    from meep_gpu.pml import PML  # noqa: PLC0415

    amplitude = case.scale if scale is None else scale
    grid = Grid(resolution=10.0, cell_size=tuple(case.cell),
                boundaries=case.boundaries, dimensions=case.dimensions,
                courant=0.35, k_point=(0.0, 0.0, 0.0),
                symmetry=((Mirror(case.fold_axis, case.fold_phase),)
                          if case.fold_axis else ()), xp=xp)
    fields = Fields(grid=grid, force_complex_fields=case.complex_storage)
    dtype = np.complex64 if case.complex_storage else np.float32
    shape = tuple(grid.shape)
    rng = np.random.default_rng(seed)

    # A FULL inverse-permittivity ROW per component, the way subpixel smoothing of
    # a curved interface produces one. The off-diagonal entries are raw tensor
    # entries and are free to be negative — a uniform positive block would be a
    # weaker needle — and the three DIAGONAL volumes are distinct, which is what
    # makes the inverse-epsilon host mutation live rather than an aliased no-op.
    diagonal: Dict[str, Any] = {}
    inverse: Dict[str, Any] = {}
    for index, name in enumerate(E_NAMES):
        epsilon = ((1.45 + 0.40 * index) + 0.30 * rng.random(shape)).astype(np.float32)
        diagonal[name] = xp.asarray(np.ascontiguousarray(epsilon))
        inverse[name] = xp.asarray(np.ascontiguousarray(
            (np.float32(1.0) / epsilon).astype(np.float32)))
    rows: Optional[Dict[str, Dict[str, Any]]]
    if case.rows == "all":
        rows = {
            row: {partner: xp.asarray(np.ascontiguousarray(
                rng.uniform(-0.08, 0.08, shape).astype(np.float32)))
                for partner in E_NAMES if partner != row}
            for row in E_NAMES}
    elif case.rows == "one":
        rows = {"Ex": {"Ey": xp.asarray(np.ascontiguousarray(
            rng.uniform(-0.08, 0.08, shape).astype(np.float32)))}}
    else:
        rows = None
    fields.set_epsilon_volumes(diagonal, inverse, rows)

    kinds = ((DRUDE, LORENTZIAN) if case.kind == "mixed"
             else ((DRUDE,) if case.kind == "drude" else (LORENTZIAN,)))
    for order in range(case.poles):
        fields.polarizations.append(PolarizationState(
            Susceptibility(0.62 + 0.19 * order, 0.04 + 0.012 * order,
                           kinds[order % len(kinds)]),
            _sigma_for(case, order, shape, rng, xp), grid, dtype))

    # THE E HALF REQUIRES AN ACTIVE ABSORBER and PML storage: the register this
    # kernel hands the recurrence is what it stored to f_w, and without the layer
    # ``drive_field`` returns the stored E instead (fields.py:1160-1162).
    if case.storage == "pml":
        fields.enable_pml_storage()
    else:
        fields.enable_field_storage()
    # A MIRRORED axis takes the HIGH FACE ONLY: cell 0 lies on the mirror plane,
    # which is a boundary condition and not an absorber (pml.py:408 refuses the
    # low face by name). That is the engine's own rule, not a harness convenience.
    request: Dict[str, Any] = {}
    for axis, letter in enumerate("xyz"):
        if shape[axis] < 6:
            continue
        request[letter] = {"high": 2} if grid.is_mirrored(axis) else 2
    pml = PML(grid=grid, thickness=request or 0)

    if case.value_class == "pm_zero_lattice":
        # A LATTICE of +0.0 and -0.0 over every array the seam consumes, with the
        # inverse epsilon left NORMAL so the sign propagates through a multiply.
        # Exempt from the movement floor by construction; the floor it answers to
        # instead is "the reference output carries BOTH signs".
        flat = np.arange(int(np.prod(shape))).reshape(shape)
        lattice = np.ascontiguousarray(
            np.where((flat % 2) == 0, np.float32(0.0), np.float32(-0.0))
            .astype(np.float32))
        alternate = np.ascontiguousarray(
            np.where(((flat // 2) % 2) == 0, np.float32(-0.0), np.float32(0.0))
            .astype(np.float32))
        for name in STATE_NAMES:
            if getattr(fields, name, None) is not None:
                getattr(fields, name)[...] = xp.asarray(lattice)
        for state in fields.polarizations:
            for component in state.driven():
                state.P[component][...] = xp.asarray(lattice)
                state.P_prev[component][...] = xp.asarray(alternate)
        return grid, fields, pml

    for name in STATE_NAMES:
        if getattr(fields, name, None) is not None:
            getattr(fields, name)[...] = xp.asarray(np.ascontiguousarray(
                (rng.standard_normal(shape) * 0.37 * amplitude).astype(np.float32)))
    for state in fields.polarizations:
        for component in state.driven():
            state.P[component][...] = xp.asarray(np.ascontiguousarray(
                (rng.standard_normal(shape) * 0.21 * amplitude).astype(np.float32)))
            state.P_prev[component][...] = xp.asarray(np.ascontiguousarray(
                (rng.standard_normal(shape) * 0.19 * amplitude).astype(np.float32)))
    return grid, fields, pml


def state_of(fields: Any, plan: Any = None) -> Dict[str, Any]:
    """Every semantically live array, with the ADE roles NAMED rather than aliased.

    ``update_P`` ROTATES which physical allocation holds ``P``, so a static string
    table could not name them without losing the semantic role — which is the one
    thing the comparison is about.

    THE SCRATCH IS THE ONE ASYMMETRY AND IT IS NAMED, NOT DROPPED. The reference
    keeps ONE retired buffer per susceptibility and this product keeps one per
    driven component (see the module's shape note), so only the LAST driven
    component's is comparable: the reference overwrote the others with the next
    component's result. That one IS compared; the rest are dead state the next
    launch overwrites before reading, and pretending otherwise would be comparing
    the harness.

    THE MATERIAL IS IN HERE TOO, per component and per off-diagonal slot. This
    seam reads eleven material volumes and writes none of them, and the movement
    floor can only say so if they are in the inventory.
    """
    state = {name: getattr(fields, name) for name in STATE_NAMES
             if getattr(fields, name, None) is not None}
    for index, polarization in enumerate(tuple(fields.polarizations)):
        driven = tuple(polarization.driven())
        for component in driven:
            state[f"P[{index}].{component}"] = polarization.P[component]
            state[f"P_prev[{index}].{component}"] = polarization.P_prev[component]
        if driven:
            state[f"scratch[{index}]"] = (
                plan._scratch[(index, driven[-1])] if plan is not None
                else polarization._scratch)
    for name, attribute in (("eps", "_eps_components"),
                            ("inv_eps", "_inv_eps_components")):
        volumes = getattr(fields, attribute, None)
        if isinstance(volumes, dict):
            for component in sorted(volumes):
                state[f"{name}[{component}]"] = volumes[component]
    seen: List[int] = []
    for row in E_NAMES:
        entries = fields.chi1inv_offdiagonal_for(row) or {}
        for partner in sorted(entries):
            volume = entries[partner]
            if any(volume is other for other in seen):
                continue
            seen.append(volume)
            state[f"row[{row}.{partner}]"] = volume
    return state


def words(array: Any) -> np.ndarray:
    """One array as uint32 WORDS. Byte compares, never allclose — and the only
    comparison that can tell ``-0.0`` from ``+0.0``, which the lattice class needs."""
    host = array if isinstance(array, np.ndarray) else array.get()
    return np.ascontiguousarray(host).view(np.uint32).ravel()


def snapshot(fields: Any, plan: Any = None) -> Dict[str, np.ndarray]:
    return {name: words(value) for name, value in state_of(fields, plan).items()}


def differing(left: np.ndarray, right: np.ndarray) -> int:
    if left.shape != right.shape:
        return max(left.size, right.size)
    return int(np.count_nonzero(left != right))


def compare(left: Dict[str, np.ndarray], right: Dict[str, np.ndarray]
            ) -> Dict[str, int]:
    assert set(left) == set(right), sorted(set(left) ^ set(right))
    return {name: count for name in sorted(left)
            if (count := differing(left[name], right[name]))}


def seam_step(fields: Any, pml: Any) -> None:
    """The two array-path passes this product replaces, in driver order."""
    from meep_gpu import stepping  # noqa: PLC0415

    stepping.update_E(fields, pml)          # driver.py:3304
    stepping.update_P(fields, pml)          # driver.py:3306


def subnormal_words(state: Dict[str, np.ndarray]) -> int:
    """float32 words in the subnormal band: exponent zero, mantissa nonzero."""
    total = 0
    for value in state.values():
        exponent = (value >> np.uint32(23)) & np.uint32(0xFF)
        mantissa = value & np.uint32(0x7FFFFF)
        total += int(np.count_nonzero((exponent == 0) & (mantissa != 0)))
    return total


def zero_words(state: Dict[str, np.ndarray], negative: bool) -> int:
    needle = np.uint32(0x80000000) if negative else np.uint32(0)
    return sum(int(np.count_nonzero(value == needle)) for value in state.values())


def _split_movement(before: Dict[str, np.ndarray], after: Dict[str, np.ndarray]
                    ) -> Tuple[List[str], List[str], int]:
    moved = {name: differing(before[name], after[name]) for name in before}
    written = [name for name in moved if name.startswith(WRITTEN_PREFIXES)]
    read_only = [name for name in moved if name.startswith(READ_ONLY_PREFIXES)]
    assert set(written) | set(read_only) == set(moved), sorted(
        set(moved) - set(written) - set(read_only))
    still = sorted(name for name in written if moved[name] == 0)
    disturbed = sorted(name for name in read_only if moved[name])
    return still, disturbed, int(sum(moved.values()))


# ---------------------------------------------------------------------------
# Launch counting
# ---------------------------------------------------------------------------

class CountingKernel:
    """Owns the JIT kernel and counts every launch the plan makes through it.

    This is the proof-of-execution instrument. It is NOT a wrapper around the plan:
    the plan calls ``kernel[grid](...)`` and nothing else, so a plan that silently
    did not launch shows up as a count of zero rather than as a passing byte
    comparison.
    """

    __slots__ = ("jit", "calls", "grids")

    def __init__(self, jit: Any) -> None:
        self.jit = jit
        self.calls = 0
        self.grids: List[Any] = []

    def __getitem__(self, grid):
        launcher = self.jit[grid]

        def run(*arguments, **keywords):
            self.calls += 1
            self.grids.append(tuple(int(value) for value in grid))
            return launcher(*arguments, **keywords)

        return run


def kernel_ptx(jit: Any) -> List[str]:
    """Every compiled specialization's PTX, for the stale-binary tripwire."""
    out: List[str] = []
    for per_device in (getattr(jit, "cache", None) or {}).values():
        for compiled in per_device.values():
            assembly = getattr(compiled, "asm", None)
            if assembly and "ptx" in assembly:
                out.append(assembly["ptx"])
    return out


# ---------------------------------------------------------------------------
# One case
# ---------------------------------------------------------------------------

def _case(label: str) -> Case:
    return next(case for case in CASES if case.label == label)


def run_case(xp, case: Case, leg: str, steps: int, kernel: Any = None,
             patch: Optional[Callable[[Any], Dict[str, Any]]] = None,
             block: Optional[int] = None) -> Dict[str, Any]:
    """Step the two engines side by side and compare per COMPLETE SEAM STEP."""
    from meep_gpu.triton_kernels import (  # noqa: PLC0415
        folded_offdiag_fused_ade_chain as family,
    )

    # A CLASS PATCH FROM AN EARLIER ROW WOULD MAKE THIS ONE A MEASUREMENT OF THE
    # WRONG PROGRAM. The two ordering host mutations have to land on the class
    # (the plan declares __slots__), so every entry drains first.
    _drain_undo()
    seed = case_seed(case.label, leg)
    _, reference, reference_pml = build(xp, case, seed)
    _, actual, actual_pml = build(xp, case, seed)

    counter = CountingKernel(
        kernel if kernel is not None
        else family.folded_offdiag_fused_ade_chain_kernel())
    plan = family.plan_folded_offdiag_fused_ade_chain(
        actual, actual_pml, block=block, num_warps=1, kernel=counter)
    if plan is None:
        return {"passed": False, "reason": "the fused chain was refused",
                "refusals": list(family.folded_offdiag_fused_ade_chain_coverage(
                    actual, actual_pml).reasons)}
    note: Dict[str, Any] = {}
    if patch is not None:
        note = patch(plan) or {}

    drift = compare(snapshot(reference), snapshot(actual, plan))
    assert not drift, f"{case.label}: the two builds are not identical: {drift}"

    before = snapshot(actual, plan)
    per_step: List[Dict[str, Any]] = []
    census_trail: List[int] = []
    error = None
    for step in range(1, steps + 1):
        seam_step(reference, reference_pml)
        try:
            plan.run()
        except Exception as exc:  # noqa: BLE001 - a refusal IS a measurement here
            error = f"{type(exc).__name__}: {exc}"[:400]
            break
        difference = compare(snapshot(reference), snapshot(actual, plan))
        census = subnormal_words(snapshot(reference))
        census_trail.append(census)
        per_step.append({"step": step, "differing_words": sum(difference.values()),
                         "differing_arrays": dict(sorted(difference.items())),
                         "reference_subnormals": census})
        if difference:
            break

    after = snapshot(actual, plan)
    still, disturbed, moved_words = _split_movement(before, after)
    identical = (error is None and len(per_step) == steps
                 and all(row["differing_words"] == 0 for row in per_step))
    launches_ok = (plan.runs == len(per_step)
                   and plan.launches == plan.launches_per_run * len(per_step)
                   and plan.alias_checks == plan.launches
                   and counter.calls == plan.launches)
    banded = case.value_class == "subnormal_band"
    vacuity_exempt = case.value_class == "pm_zero_lattice"
    floor_ok = (not disturbed) and (vacuity_exempt or not still)
    lattice_live = census_live = None
    if vacuity_exempt:
        reference_state = snapshot(reference)
        lattice_live = bool(zero_words(reference_state, True)
                            and zero_words(reference_state, False))
        floor_ok = floor_ok and lattice_live
    band_reach = None
    if banded:
        # THE BAND IS A CLASS ONLY WHERE THE POLICY LETS ONE EXIST, and this is
        # the three-way rule rather than a pass/fail: a row that COULD NOT have
        # reached the class must not be failed for a property of the
        # configuration, and a row that could have and did not must not be
        # passed.
        #
        # * under ``keep`` the census MUST fire, or the row measured the physical
        #   band under another name — that is VACUOUS;
        # * under ``flush`` it must NOT, because flushing the band to zero IS the
        #   policy. The row then measures something the keeping run cannot: that
        #   the fused launch and the array path agree word for word about a band
        #   the executor has removed from under both of them.
        peak = max(census_trail or [0])
        band_reach = ("live" if peak else
                      "flushed_by_policy" if _POLICY == "flush" else "VACUOUS")
        census_live = band_reach != "VACUOUS"
        floor_ok = floor_ok and census_live
    # A UNIFORM ROW WHOSE CENSUS FIRED is a banded row under another name: the
    # value classes are only classes while each one stays in its own band.
    clean = (case.value_class != "uniform") or (max(census_trail or [0]) == 0)
    return {
        "passed": bool(identical and launches_ok and floor_ok and clean),
        "bit_identical": identical,
        "error": error,
        "subnormal_free": clean,
        "identical_at_step_1": bool(per_step and per_step[0]["differing_words"] == 0),
        "identical_at_last_step": bool(
            per_step and per_step[-1]["differing_words"] == 0
            and len(per_step) == steps),
        "first_divergence": next((row["step"] for row in per_step
                                  if row["differing_words"]), None),
        "steps_compared": len(per_step),
        "differing_words": per_step[-1]["differing_words"] if per_step else None,
        "differing_arrays": per_step[-1]["differing_arrays"] if per_step else None,
        "reference_subnormal_peak": max(census_trail or [0]),
        "value_class": case.value_class,
        "value_class_live": (lattice_live if vacuity_exempt else census_live),
        "subnormal_band_reach": band_reach,
        "subnormal_policy": _POLICY,
        "arrays_compared": len(before),
        "written_arrays_that_never_moved": still,
        "read_only_arrays_the_seam_disturbed": disturbed,
        "moved_words": moved_words,
        "runs": plan.runs, "launches": plan.launches,
        "kernel_calls": counter.calls, "alias_checks": plan.alias_checks,
        "launches_per_run": plan.launches_per_run,
        "launches_expected": plan.launches_per_run * len(per_step),
        "replaces": list(plan.replaces_sub_steps),
        "poles": list(plan.counts), "block": plan.block,
        "row_mask": list(plan._e.row_mask),
        "boundary_codes": list(plan._e._base.boundary_codes),
        "wall_axes": list(plan._e._base.wall_axes),
        "extra_scratch_volumes": plan.extra_scratch_volumes,
        "grid": [list(g) for g in counter.grids[:1]],
        "shape": list(plan.shape), "seed": seed, "patch": note,
    }


def run_block_sweep(xp, case: Case, sizes: Sequence[int], steps: int
                    ) -> Dict[str, Any]:
    """S2: the same arithmetic, four compiled programs.

    ``BLOCK`` is a constexpr, so each size is a different program over the same
    body, and the certified E half is certified at ONE of them. A weld that
    reordered anything the schedule can see would show up here and nowhere else in
    this gate — every other leg compiles a single block size.
    """
    rows: List[Dict[str, Any]] = []
    for size in sizes:
        result = run_case(xp, case, f"block:{size}", steps, block=int(size))
        rows.append({"block": int(size), "passed": bool(result.get("passed")),
                     "differing_words": result.get("differing_words"),
                     "first_divergence": result.get("first_divergence"),
                     "launches": result.get("launches"),
                     "grid": result.get("grid"), "error": result.get("error")})
        log(f"    block {size}: passed={rows[-1]['passed']} "
            f"words={rows[-1]['differing_words']}")
    return {"passed": all(row["passed"] for row in rows), "case": case.label,
            "steps": steps, "rows": rows}


def run_separate_control(xp, case: Case, steps: int) -> Dict[str, Any]:
    """The SEPARATE certified products beside the fused one, on identical state.

    THIS IS THE LEG THAT MAKES "FUSION" A MEASUREMENT RATHER THAN A NAME. Three
    engines from one seed: the array path; the two ALREADY CERTIFIED Triton
    products (``folded_offdiag_dispersive_update_e`` on ``update_E`` and
    ``launch.plan_ade_update_p`` on ``update_P``) dispatching the seam as separate
    launches; and the fused product. All three must agree word for word at every
    complete seam step, and the LAUNCH COUNTS are recorded on both sides — the only
    place the difference between the compositions shows up at all, since a correct
    fusion is byte-neutral by construction.

    The separate side is not a strawman: those two products are what the corpus row
    would be stepped by today, so disagreement here would be a defect in the fused
    product rather than in an invented comparator.
    """
    from meep_gpu.triton_kernels import (  # noqa: PLC0415
        folded_offdiag_dispersive_update_e as electric_module,
        folded_offdiag_fused_ade_chain as family,
        launch as launch_module,
    )

    seed = case_seed(case.label, "separate_control")
    _, reference, reference_pml = build(xp, case, seed)
    _, separate, separate_pml = build(xp, case, seed)
    _, fused, fused_pml = build(xp, case, seed)

    electric = electric_module.plan_folded_offdiag_dispersive_constitutive(
        separate, separate_pml)
    polarizations = [launch_module.plan_ade_update_p(separate, state)
                     for state in separate.polarizations]
    plan = family.plan_folded_offdiag_fused_ade_chain(fused, fused_pml,
                                                      num_warps=1)
    if electric is None or plan is None or any(p is None for p in polarizations):
        return {"passed": False, "reason": "a declared product was refused",
                "electric": electric is not None,
                "polarizations": [p is not None for p in polarizations],
                "fused": plan is not None}

    before = snapshot(fused, plan)
    per_step: List[Dict[str, Any]] = []
    for step in range(1, steps + 1):
        seam_step(reference, reference_pml)
        electric.run()
        for entry in polarizations:
            entry.run(separate.drive_field)
        plan.run()
        row = {
            "step": step,
            "separate_vs_array": sum(compare(snapshot(reference),
                                             snapshot(separate)).values()),
            "fused_vs_array": sum(compare(snapshot(reference),
                                          snapshot(fused, plan)).values()),
            "fused_vs_separate": sum(compare(snapshot(separate),
                                             snapshot(fused, plan)).values()),
        }
        per_step.append(row)
        if any(value for key, value in row.items() if key != "step"):
            break

    still, disturbed, _ = _split_movement(before, snapshot(fused, plan))
    identical = (len(per_step) == steps
                 and all(row["separate_vs_array"] == row["fused_vs_array"]
                         == row["fused_vs_separate"] == 0 for row in per_step))
    separate_dispatches = 1 + sum(len(tuple(state.driven()))
                                  for state in separate.polarizations)
    return {
        "passed": bool(identical and plan.launches and not still and not disturbed),
        "per_step_tail": per_step[-3:],
        "steps_compared": len(per_step),
        "separate_dispatches_per_step": separate_dispatches,
        "fused_dispatches_per_step": plan.launches_per_run,
        "dispatches_removed_per_step": separate_dispatches - plan.launches_per_run,
        "fused_launches": plan.launches,
        "written_arrays_that_never_moved": still,
        "read_only_arrays_the_seam_disturbed": disturbed,
        "extra_scratch_volumes": plan.extra_scratch_volumes,
        "seed": seed,
    }


# ---------------------------------------------------------------------------
# Static legs — no device, no Triton
# ---------------------------------------------------------------------------

KERNELS_DIR = os.path.join(API_ROOT, "meep_gpu", "triton_kernels")
FAMILY_PATH = os.path.join(KERNELS_DIR, "folded_offdiag_fused_ade_chain.py")
CERTIFIED_PATH = os.path.join(KERNELS_DIR, "folded_offdiag_dispersive_update_e.py")
CERTIFIED_STEP = "folded_offdiag_dispersive_constitutive_step"


def _function_node(path: str, name: str) -> ast.FunctionDef:
    tree = ast.parse(open(path, encoding="utf-8").read())
    return next(child for child in ast.walk(tree)
                if isinstance(child, ast.FunctionDef) and child.name == name)


def _function_text(path: str, name: str) -> str:
    source = open(path, encoding="utf-8").read()
    segment = ast.get_source_segment(source, _function_node(path, name))
    assert segment, f"{name} has no source segment in {path}"
    return segment


def _normalised(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def _statements(node: ast.FunctionDef) -> List[Any]:
    return node.body[1:] if ast.get_docstring(node) else list(node.body)


def _runtime_names(node: ast.FunctionDef) -> List[str]:
    return [a.arg for a in node.args.args if a.annotation is None]


def _constexpr_names(node: ast.FunctionDef) -> List[str]:
    return [a.arg for a in node.args.args if a.annotation is not None]


def kernel_text() -> str:
    """The shipped kernel's EXECUTABLE source, read from the FILE.

    The DOCSTRING IS DROPPED. It names the two loads this kernel eliminates, so a
    needle test that read it would be satisfied by prose rather than by code — the
    same reason the laptop test file strips it.
    """
    from meep_gpu.triton_kernels import (  # noqa: PLC0415
        folded_offdiag_fused_ade_chain as family,
    )

    source = open(FAMILY_PATH, encoding="utf-8").read()
    node = _function_node(FAMILY_PATH, family.KERNEL_NAME)
    lines = source.splitlines(keepends=True)
    head = "".join(lines[node.lineno - 1:node.body[0].lineno - 1])
    body = "".join(lines[node.body[1].lineno - 1:node.end_lineno])
    assert ast.get_docstring(node), "the kernel carries a docstring"
    return head + body


def leg_transcription() -> Dict[str, Any]:
    """The E half is the certified body; the ADE half is ``ade_update_p``. PARSED.

    THE MIRRORED EVALUATOR is a defect class this project has paid for: a check
    that re-implements a kernel's arithmetic mirrors a defect instead of executing
    it. So nothing here re-derives a line. The certified kernel's own file is
    parsed and its statement list is required to be an AST-IDENTICAL PREFIX of this
    one's; ``kernels.ade_update_p``'s own file is parsed and each of its six lines
    is required present per arm under the DECLARED renames; and the two eliminated
    loads are required ABSENT.
    """
    from meep_gpu.triton_kernels import (  # noqa: PLC0415
        folded_offdiag_fused_ade_chain as family,
    )

    fused_node = _function_node(FAMILY_PATH, family.KERNEL_NAME)
    certified_node = _function_node(CERTIFIED_PATH, CERTIFIED_STEP)
    fused = kernel_text()
    normalised = _normalised(fused)
    ade = _normalised(_function_text(os.path.join(KERNELS_DIR, "kernels.py"),
                                     "ade_update_p"))
    expression = "((p * c_now) + (c_prev * q)) + (c_drive * (s * w))"
    rows: List[Dict[str, Any]] = []

    def row(name: str, agrees: bool, **detail: Any) -> None:
        rows.append({"line": name, "agrees": bool(agrees), **detail})

    certified_body = [ast.dump(node) for node in _statements(certified_node)]
    fused_body = [ast.dump(node) for node in _statements(fused_node)]
    row("the_certified_body_is_an_ast_identical_prefix",
        len(certified_body) >= 40 and fused_body[:len(certified_body)] == certified_body,
        certified_statements=len(certified_body), fused_statements=len(fused_body))
    trailing = _statements(fused_node)[len(certified_body):]
    row("all_three_components_in_one_launch",
        len(trailing) == 3 * family.CHAIN_MAX_POLES
        and all(isinstance(node, ast.If)
                and re.fullmatch(r"NP[012] > \d", ast.unparse(node.test))
                for node in trailing),
        trailing_statements=len(trailing),
        guards=[ast.unparse(node.test) for node in trailing
                if isinstance(node, ast.If)])

    certified_runtime = _runtime_names(certified_node)
    certified_constexpr = _constexpr_names(certified_node)
    fused_runtime = _runtime_names(fused_node)
    fused_constexpr = _constexpr_names(fused_node)
    everything = [a.arg for a in fused_node.args.args]
    expected_ade = [f"{stem}_{letter}{slot}" for letter in family.LETTERS
                    for stem in ("p_out", "p_prev", "sigma", "cnow", "cprev",
                                 "cdrive")
                    for slot in range(family.CHAIN_MAX_POLES)]
    row("both_signature_halves_are_certified_prefixes",
        (fused_runtime[:len(certified_runtime)] == certified_runtime
         and fused_constexpr[:len(certified_constexpr)] == certified_constexpr
         and len(certified_runtime) == 55 and len(certified_constexpr) == 19),
        certified_runtime=len(certified_runtime),
        certified_constexpr=len(certified_constexpr),
        fused_runtime=len(fused_runtime), fused_constexpr=len(fused_constexpr))
    row("the_ade_block_sits_between_n_elem_and_NP0",
        (everything[everything.index("n_elem") + 1] == "p_out_a0"
         and everything[everything.index("NP0") - 1]
         == f"cdrive_c{family.CHAIN_MAX_POLES - 1}"
         and fused_runtime[len(certified_runtime):] == expected_ade),
        after_n_elem=everything[everything.index("n_elem") + 1],
        before_NP0=everything[everything.index("NP0") - 1])
    row("the_SV_flags_follow_BLOCK",
        (certified_constexpr[-1] == "BLOCK"
         and fused_constexpr[len(certified_constexpr):]
         == [f"SV_{letter}{slot}" for letter in family.LETTERS
             for slot in range(family.CHAIN_MAX_POLES)]),
        tail=fused_constexpr[len(certified_constexpr):])
    row("the_launcher_and_the_signature_agree_on_91",
        len(fused_runtime) == family.RUNTIME_ARGUMENTS,
        declared=family.RUNTIME_ARGUMENTS, counted=len(fused_runtime))

    row("pole_bound_exactly_once",
        all(everything.count(f"{letter}{slot}") == 1
            for letter in family.LETTERS for slot in range(family.MAX_POLES)),
        pole_parameters=3 * family.MAX_POLES)
    row("p_now_is_not_a_parameter",
        "p_now" not in everything and "drive" not in everything)

    for index, letter in enumerate(family.LETTERS):
        for slot in range(family.CHAIN_MAX_POLES):
            checks = {
                "store": f"tl.store(p_out_{letter}{slot} + idx, {expression}, "
                         f"mask=live)",
                "history": f"q = tl.load(p_prev_{letter}{slot} + idx, mask=live, "
                           f"other=0.0)",
                "sigma_volume": f"s = tl.load(sigma_{letter}{slot} + idx, "
                                f"mask=live, other=0.0)",
                "sigma_scalar": f"s = sigma_{letter}{slot}",
                "pole_reread": f"p = tl.load({letter}{slot} + idx, mask=live, "
                               f"other=0.0)",
                "drive_register": f"w = src{index}",
                "c_now": f"c_now = cnow_{letter}{slot}",
                "c_prev": f"c_prev = cprev_{letter}{slot}",
                "c_drive": f"c_drive = cdrive_{letter}{slot}",
            }
            missing = sorted(key for key, text in checks.items()
                             if text not in normalised)
            row(f"the_recurrence_is_the_certified_line[{letter}{slot}]",
                not missing, missing=missing,
                certified_source="kernels.ade_update_p")
    row("the_certified_recurrence_is_what_was_renamed",
        all(text in ade for text in (
            f"tl.store(p_out + idx, {expression}, mask=live)",
            "q = tl.load(p_prev + idx, mask=live, other=0.0)",
            "s = tl.load(sigma + idx, mask=live, other=0.0)",
            "p = tl.load(p_now + idx, mask=live, other=0.0)",
            "w = tl.load(drive + idx, mask=live, other=0.0)")),
        certified_source="kernels.ade_update_p")
    row("the_drive_is_the_register",
        "tl.load(drive" not in normalised and "w = tl.load(" not in normalised)
    row("the_pole_is_read_through_the_bound_pointer",
        "p_now" not in normalised)
    row("the_entry_point_is_its_own",
        family.KERNEL_NAME in fused and CERTIFIED_STEP not in fused,
        kernel_name=family.KERNEL_NAME)

    guards: Dict[str, set] = {}
    for node in ast.walk(fused_node):
        if not isinstance(node, ast.If):
            continue
        test = ast.unparse(node.test)
        for statement in node.body:
            for child in ast.walk(statement):
                if isinstance(child, ast.Call) and getattr(
                        child.func, "attr", None) == "store":
                    guards.setdefault(ast.unparse(child.args[0]), set()).add(test)
    for index, letter in enumerate(family.LETTERS):
        for slot in range(family.CHAIN_MAX_POLES):
            target = f"p_out_{letter}{slot} + idx"
            row(f"every_arm_is_guarded_by_its_own_slot[{letter}{slot}]",
                guards.get(target) == {f"NP{index} > {slot}"},
                guard=sorted(guards.get(target) or ()))
    row("the_callers_NP_guards_are_the_ade_arms_alone",
        all(len(re.findall(rf"if NP{index} > \d+:", fused))
            == family.CHAIN_MAX_POLES for index in range(3))
        and not re.search(r"if NP[012] > \d+:",
                          _function_text(CERTIFIED_PATH, CERTIFIED_STEP)))

    module_source = open(FAMILY_PATH, encoding="utf-8").read()
    bindings = ("_masked_row_sum = _folded._masked_row_sum",
                "_load_d_minus_p = _fod._load_d_minus_p",
                "_folded_dispersive_term = _fod._folded_dispersive_term",
                "PERIODIC = tl.constexpr(_fod.CODE_PERIODIC)",
                "METALLIC = tl.constexpr(_fod.CODE_METALLIC)",
                "MIRROR_ROW = tl.constexpr(_fod.MIRROR_SOURCE_INDEX)")
    row("the_helpers_are_the_certified_modules_objects",
        all(line in module_source for line in bindings)
        and all(name in fused for name in (
            "_load_d_minus_p", "_folded_dispersive_term", "_masked_row_sum",
            "PERIODIC", "METALLIC", "MIRROR_ROW")),
        bindings=[line for line in bindings if line not in module_source])

    edits = family.LIFT_EDITS
    row("lift_edits", len(edits) == 6
        and all({"line", "became", "why"} <= set(edit) for edit in edits),
        declared=len(edits), lines=[edit.get("line", "")[:60] for edit in edits])
    return {"passed": all(entry["agrees"] for entry in rows),
            "rows_checked": len(rows),
            "failed_rows": [entry for entry in rows if not entry["agrees"]]}


def _expected_extra(fields: Any) -> int:
    """``K * (d - 1)`` summed over the susceptibilities this engine registers."""
    return sum(max(len(tuple(state.driven())) - 1, 0)
               for state in fields.polarizations)


def leg_shape(xp) -> Dict[str, Any]:
    """The rotation orbit, walked to closure THROUGH THE SHIPPED PLAN.

    A product whose whole licence is a buffer-disjointness argument has to have
    that argument checked against the object that allocates the buffers, not
    against a model of it. Nine positions is three full orbits, and the ROW VOLUMES
    are among the plan's reads here — the addition this body makes over the plain
    chain, since the E half gathers them at neighbour offsets while the ADE outputs
    are written.
    """
    from meep_gpu.triton_kernels import (  # noqa: PLC0415
        folded_offdiag_fused_ade_chain as family,
    )

    rows: List[Dict[str, Any]] = []
    for case in CASES:
        _, fields, pml = build(xp, case, case_seed(case.label, "shape"))
        plan = family.FoldedOffdiagFusedAdeChainPlan(fields, pml, 256, num_warps=1)
        positions: List[Tuple[int, ...]] = []
        conflicts = 0
        for _turn in range(9):
            groups = plan._e._poles.arrays()
            chain = plan._resolve(groups)
            try:
                plan._check_aliasing(chain)
            except RuntimeError:
                conflicts += 1
            outputs = {id(entry[2]) for entries in chain for entry in entries}
            inputs = {id(entry[3]) for entries in chain for entry in entries}
            inputs |= {id(entry[4]) for entries in chain for entry in entries}
            inputs |= set(plan._read_ids)
            conflicts += len(outputs & inputs)
            positions.append(tuple(id(state.P[name])
                                   for state in fields.polarizations
                                   for name in state.driven()))
            plan._rotate(chain)
        expected = _expected_extra(fields)
        driven = [len(tuple(state.driven())) for state in fields.polarizations]
        rows.append({
            "case": case.label, "poles": case.poles, "driven_per_state": driven,
            "positions_walked": len(positions), "conflicts": conflicts,
            "orbit_is_three": positions[0] == positions[3] == positions[6],
            "orbit_is_not_one": positions[0] != positions[1],
            "extra_scratch_volumes": plan.extra_scratch_volumes,
            "extra_expected_K_times_d_minus_1": expected,
            "row_volumes_in_the_read_set": len(plan._read_ids),
            "alias_checks": plan.alias_checks,
            "agrees": (conflicts == 0
                       and plan.extra_scratch_volumes == expected
                       and plan.alias_checks == 9
                       and positions[0] == positions[3] == positions[6]
                       and positions[0] != positions[1])})
        log(f"    shape {case.label}: conflicts={conflicts} "
            f"extra={plan.extra_scratch_volumes}/{expected} orbit=3")
    return {"passed": all(entry["agrees"] for entry in rows), "rows": rows}


def leg_plan_refusals(xp) -> Dict[str, Any]:
    """Break the chain on purpose; the plan must refuse rather than launch.

    A guard never demonstrated to fire is decoration, and these are the guards the
    shape rests on: no two arms may share an output, no output may be an array the
    launch reads — INCLUDING one of the six row volumes — and the pole ORDER the E
    chain resolves must still be the one the ADE arms were paired against. Driven
    through the plan's own pre-launch resolution, so the leg runs on a host with no
    Triton too, which is the merge bar.
    """
    from meep_gpu.triton_kernels import (  # noqa: PLC0415
        folded_offdiag_fused_ade_chain as family,
        offdiag_update_e as offdiag,
    )

    case = _case(MUTATION_CASE)
    rows: List[Dict[str, Any]] = []

    def fresh():
        _, fields, pml = build(xp, case, case_seed(case.label, "plan_refusals"))
        return fields, family.FoldedOffdiagFusedAdeChainPlan(
            fields, pml, 256, num_warps=1)

    def refusal(defect: str, needle: str, action: Callable[..., Any],
                error=RuntimeError) -> None:
        fields, plan = fresh()
        caught = ""
        try:
            action(plan, fields)
        except error as exc:
            caught = str(exc)
        rows.append({"defect": defect, "refused": bool(caught),
                     "reason": caught[:220], "expected_reason": needle,
                     "agrees": needle in caught})
        log(f"    plan refusal: {defect} -> refused={bool(caught)}")

    def shared_scratch(plan: Any, _fields: Any) -> None:
        plan._scratch[(1, "Ex")] = plan._scratch[(0, "Ex")]
        plan._check_aliasing(plan._resolve(plan._e._poles.arrays()))

    def output_is_an_input(plan: Any, _fields: Any) -> None:
        chain = plan._resolve(plan._e._poles.arrays())
        plan._check_aliasing(chain)                     # clean before the break
        broken = [list(entries) for entries in chain]
        entry = list(chain[0][0])
        entry[2] = entry[3]
        broken[0][0] = tuple(entry)
        plan._check_aliasing(broken)

    def scratch_is_a_row_volume(plan: Any, fields: Any) -> None:
        volumes = [value for value in offdiag.row_volumes_for(fields)
                   if value is not None]
        assert volumes, "the mutation case installs no off-diagonal row"
        plan._scratch[(0, "Ex")] = volumes[0]
        plan._check_aliasing(plan._resolve(plan._e._poles.arrays()))

    def order_reversed_under_the_arms(plan: Any, _fields: Any) -> None:
        plan._order["Ex"] = tuple(reversed(plan._order["Ex"]))
        plan._resolve(plan._e._poles.arrays())

    def order_changed_under_the_live_binding(plan: Any, _fields: Any) -> None:
        binding = plan._e._poles
        binding.order["Ex"] = tuple(reversed(binding.order["Ex"]))
        plan._resolve(binding.arrays())

    refusal("two susceptibilities share one Ex scratch",
            "SAME output buffer", shared_scratch)
    refusal("an ADE output is also a P this launch reads",
            "also read by this launch", output_is_an_input)
    refusal("an ADE output IS one of the six off-diagonal row volumes",
            "also read by this launch", scratch_is_a_row_volume)
    refusal("the Ex ADE arms are paired against a reversed pole order",
            "is not the array the subtraction slot resolved",
            order_reversed_under_the_arms)
    refusal("the live pole binding sees a different order than the plan was built for",
            "no longer valid", order_changed_under_the_live_binding)

    # THE CAP, AS THE PLAN ENFORCES IT. The predicate refuses a third pole by
    # name; this is the second door, on the object that would bind the arms.
    third = dataclasses.replace(_case(MUTATION_CASE), label="three_poles", poles=3)
    _, fields, pml = build(xp, third, case_seed(third.label, "plan_refusals"))
    caught = ""
    try:
        family.FoldedOffdiagFusedAdeChainPlan(fields, pml, 256, num_warps=1)
    except ValueError as exc:
        caught = str(exc)
    rows.append({"defect": "counts past CHAIN_MAX_POLES", "refused": bool(caught),
                 "reason": caught[:220],
                 "expected_reason": "ADE arms per component",
                 "agrees": "ADE arms per component" in caught})
    log(f"    plan refusal: counts past the cap -> refused={bool(caught)}")
    return {"passed": all(entry["agrees"] for entry in rows), "rows": rows}


def leg_refusals(xp) -> Dict[str, Any]:
    """Configurations the product must refuse, on the host's own arrays.

    AND THE TWO DISJOINTNESS DIRECTIONS. ``fused_ade_chain`` must refuse this cell
    on ALL THREE of its arms and ``complex_fused_ade_chain`` on storage, or the
    board's at-most-one-E->P-admitter assertion would find a clash on the one row
    this product exists for.
    """
    from meep_gpu.triton_kernels import (  # noqa: PLC0415
        complex_fused_ade_chain as complex_chain,
        folded_offdiag_dispersive_update_e as electric_module,
        folded_offdiag_fused_ade_chain as family,
        fused_ade_chain as chain,
    )

    rows: List[Dict[str, Any]] = []
    device = xp is not np
    target = _case("one_state_all_components")

    def engine(label: str, case: Case, **changes: Any):
        adjusted = dataclasses.replace(case, label=label, **changes)
        return build(xp, adjusted, case_seed(label, "refusals"))

    def record(name: str, fields: Any, pml: Any, expected: bool,
               needles: Sequence[str] = ()) -> None:
        verdict = family.folded_offdiag_fused_ade_chain_coverage(fields, pml)
        # ``covered_modulo_backend`` is the convention every predicate record on
        # this board is read through: on a host with no CuPy every verdict is False
        # for that reason ALONE, which is a fact about the host and not about the
        # configuration. On the device host the two columns are the same number and
        # ``planned`` is what proves it.
        residual = [reason for reason in verdict.reasons if "not cupy" not in reason]
        planned = family.plan_folded_offdiag_fused_ade_chain(fields, pml) is not None
        joined = " | ".join(residual)
        missing = [needle for needle in needles if needle not in joined]
        agrees = (not residual) == expected and not missing
        if device:
            agrees = agrees and verdict.covered == expected == planned
        rows.append({"case": name, "covered": verdict.covered,
                     "covered_modulo_backend": not residual, "planned": planned,
                     "expected": expected, "device": device, "agrees": agrees,
                     "needles_missing": missing,
                     "reasons": residual[:5]})
        log(f"    refusal {name}: modulo_backend={not residual} "
            f"expected={expected} missing={missing}")

    # --- the target, and the E half alone on it ------------------------------
    _, fields, pml = engine("target", target)
    record("target", fields, pml, True)
    electric = electric_module.folded_offdiag_dispersive_constitutive_coverage(
        fields, pml)
    electric_residual = [r for r in electric.reasons if "not cupy" not in r]
    rows.append({"case": "the_E_half_alone_admits_the_target",
                 "covered_modulo_backend": not electric_residual,
                 "expected": True, "agrees": not electric_residual,
                 "reasons": electric_residual[:5]})
    # --- disjointness, both directions ---------------------------------------
    arms = {arm: chain.fused_ade_chain_coverage(fields, pml, arm)
            for arm in chain.ARMS}
    arm_rows = {arm: [r for r in verdict.reasons if "not cupy" not in r]
                for arm, verdict in arms.items()}
    rows.append({"case": "fused_ade_chain_refuses_this_cell_on_every_arm",
                 "expected": False, "arms": sorted(arm_rows),
                 "agrees": all(bool(reasons) for reasons in arm_rows.values()),
                 "reasons": {arm: reasons[:2] for arm, reasons in arm_rows.items()}})
    complex_reasons = [r for r in complex_chain.complex_fused_ade_chain_coverage(
        fields, pml, None).reasons if "not cupy" not in r]
    rows.append({"case": "complex_fused_ade_chain_refuses_real_storage",
                 "expected": False, "agrees": bool(complex_reasons),
                 "reasons": complex_reasons[:3]})

    # --- the refusal table ----------------------------------------------------
    _, fields, pml = engine("no_fold", target, fold_axis="")
    record("no_fold", fields, pml, False, ("E half:",))
    _, fields, pml = engine("no_rows", target, rows="none")
    record("no_rows", fields, pml, False, ("E half:",))
    _, fields, pml = engine("no_poles", target, poles=0)
    record("no_poles", fields, pml, False,
           ("no polarization drives any electric component",))
    _, fields, pml = engine("three_poles", target, poles=3)
    record("three_poles", fields, pml, False, ("frozen P",))
    _, fields, pml = engine("inactive_pml", target)
    inert = _inert_pml(fields)
    record("inactive_pml", fields, inert, False, ("E half:",))
    _, fields, pml = engine("no_pml_storage", target, storage="field")
    record("no_pml_storage", fields, pml, False,
           ("Fields is not in PML storage mode", "f_w_Ex is not allocated"))
    _, fields, pml = engine("complex_storage", target, complex_storage=True)
    record("complex_storage", fields, pml, False, ("E half:",))
    _, fields, pml = engine("nonlinear", target)
    # chi3 is mirror-compatible on every component; chi2 on the mirror-odd one is
    # refused by Fields before this predicate is reached, which would measure the
    # installer rather than the clause.
    fields.set_nonlinear_volumes({}, {name: 0.05 for name in E_NAMES})
    record("nonlinear", fields, pml, False, ("E half:",))

    # --- the three SEAM breaks ------------------------------------------------
    _, fields, pml = engine("drive_rebound", target)
    stored = {name: getattr(fields, name) for name in E_NAMES}
    fields.drive_field = lambda component: stored[component]  # type: ignore[method-assign]
    record("drive_field_returns_the_stored_E", fields, pml, False,
           ("this kernel's register is the value it stores THERE",))

    _, fields, pml = engine("driven_disagrees", target)
    # ``driven()`` ALONE, not ``_driven``: the E chain's pole partition is derived
    # from ``drives()`` (dispersive_update_e.poles_per_component) and the ADE half
    # from ``driven()``, so rewriting the underlying tuple moves both together and
    # would measure nothing. The clause exists for the case where the two READERS
    # disagree, which is what this patches.
    state = fields.polarizations[0]
    state.driven = lambda: ("Ex",)  # type: ignore[method-assign]
    record("driven_omits_a_subtracted_component", fields, pml, False,
           ("disagrees with the components whose D - sum P chain subtracts it",))

    _, fields, pml = engine("resized_P", target)
    state = fields.polarizations[0]
    component = tuple(state.driven())[0]
    state.P[component] = state.P[component].ravel()[:-1]
    record("a_P_volume_is_shorter_than_the_launch_walks", fields, pml, False,
           ("this weld drops the certified update_P guard",))
    return {"passed": all(entry["agrees"] for entry in rows), "rows": rows}


def _inert_pml(fields: Any) -> Any:
    """A zero-thickness layer on the same grid: an absorber that absorbs nothing."""
    from meep_gpu.pml import PML  # noqa: PLC0415

    return PML(grid=fields.grid, thickness=0)


#: Scale, and what the per-step census MUST do at it. The clean rows are what make
#: the firing rows mean something — a detector that fired on everything would refuse
#: the physical band too and certify nothing.
PRECONDITION_SCALES: Tuple[Tuple[str, float, Optional[bool]], ...] = (
    ("physical", 1.0, True),
    ("small_normal", 1e-25, True),
    ("band_edge", 1e-30, None),
    ("subnormal_band", 1e-34, False),
    ("deep_subnormal", 1e-41, False),
)


def leg_precondition(xp, case: Case, budget: int = 8) -> Dict[str, Any]:
    """A precondition never demonstrated to FIRE is decoration.

    WHAT THIS LEG MEANS DEPENDS ON THE INSTALLED POLICY, and both meanings are
    asserted rather than one being quietly assumed:

    * under ``keep`` it is the detector's own calibration — the census stays
      silent through the physical band and FIRES in the subnormal one, so a row
      elsewhere that reports "the band fired" is reporting something real;
    * under ``flush`` every row must come back CLEAN, at every scale, including
      the two that fire under ``keep``. That is not the detector failing: it is
      the policy working, measured end to end through the same array path the
      byte comparison runs against.
    """
    flushing = _POLICY == "flush"
    rows: List[Dict[str, Any]] = []
    for label, scale, expect_clean in PRECONDITION_SCALES:
        if flushing:
            expect_clean = True
        _, reference, reference_pml = build(
            xp, case, case_seed(case.label, f"precondition:{label}"), scale=scale)
        per_step: List[int] = []
        for _ in range(budget):
            seam_step(reference, reference_pml)
            per_step.append(subnormal_words(snapshot(reference)))
        fired = [index for index, count in enumerate(per_step) if count]
        agrees = (expect_clean is None
                  or (expect_clean and not fired)
                  or (not expect_clean and bool(fired)))
        rows.append({"scale": label, "factor": scale,
                     "subnormal_words_per_step": per_step,
                     "census_fired": bool(fired),
                     "first_step": fired[0] if fired else None,
                     "expected_clean": expect_clean, "agrees": agrees,
                     "subnormal_policy": _POLICY})
        log(f"    precondition {label} factor={scale:g} fired={bool(fired)} "
            f"expected_clean={expect_clean}")
    return {"passed": all(entry["agrees"] for entry in rows), "rows": rows,
            "subnormal_policy": _POLICY,
            "reading": ("under 'flush' EVERY scale must come back clean — the "
                        "policy removed the band; under 'keep' the two banded "
                        "scales must FIRE and the physical ones must not")}


# ---------------------------------------------------------------------------
# Armed kernel mutations
# ---------------------------------------------------------------------------

def shipped_source(family) -> str:
    """The shipped kernel's own text, dedented, decorator included."""
    return textwrap.dedent(
        inspect.getsource(family.folded_offdiag_fused_ade_chain_step.fn))


#: THE MUTANT'S HEADER, AND WHY IT IS NOT ``import triton`` ALONE. This kernel
#: CALLS three JIT helpers and COMPARES against three constexprs that are module
#: globals of the two certified folded off-diagonal modules — ``@triton.jit``
#: resolves both kinds through the defining module's globals — so a mutant module
#: compiled under the plain chain gate's header would raise ``NameError`` on the
#: first one and every kernel mutation would score UNCOMPILABLE while measuring
#: nothing. The names are IMPORTED from the modules the shipped kernel resolves
#: them through, so a mutant executes the same helper bytes the product does.
MUTANT_HEADER = (
    "import triton\n"
    "import triton.language as tl\n"
    "from meep_gpu.triton_kernels.folded_offdiag_dispersive_update_e import (\n"
    "    _load_d_minus_p, _folded_dispersive_term, PERIODIC, METALLIC,\n"
    "    MIRROR_ROW)\n"
    "from meep_gpu.triton_kernels.folded_offdiag_update_e import _masked_row_sum\n"
    "\n\n"
)


def compile_mutant(source: str, kernel_name: str) -> Any:
    """Compile a renamed mutant. The rename is what keeps the JIT cache honest."""
    handle = tempfile.NamedTemporaryFile(
        "w", suffix="_mutated_fod_ade_chain.py", delete=False, encoding="utf-8")
    handle.write(MUTANT_HEADER + source)
    handle.close()
    _TEMPORARY.append(handle.name)
    spec = importlib.util.spec_from_file_location(
        "triton_mutated_fod_ade_chain_" + str(len(_TEMPORARY)), handle.name)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[spec.name] = module  # type: ignore[union-attr]
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return getattr(module, kernel_name)


def _lines_at(source: str, needle: str) -> Tuple[int, str, List[str]]:
    lines = source.split("\n")
    hits = [index for index, line in enumerate(lines) if needle in line]
    if len(hits) != 1:
        return -1, "", lines
    index = hits[0]
    indent = lines[index][:len(lines[index]) - len(lines[index].lstrip())]
    return index, indent, lines


def _replace_span(source: str, needle: str, span: int,
                  replacement: Sequence[str]) -> Tuple[str, int]:
    """Replace the ``span`` lines starting at the UNIQUE line holding ``needle``."""
    index, indent, lines = _lines_at(source, needle)
    if index < 0:
        return source, 0
    lines[index:index + span] = [indent + line for line in replacement]
    return "\n".join(lines), 1


def _swap_two(source: str, first: str, second: str,
              replacement: Sequence[str]) -> Tuple[str, int]:
    index, indent, lines = _lines_at(source, first)
    if index < 0 or second not in lines[index + 1]:
        return source, 0
    lines[index:index + 2] = [indent + line for line in replacement]
    return "\n".join(lines), 1


def _replace_block(source: str, needles: Sequence[str],
                   replacement: Sequence[str]) -> Tuple[str, int]:
    """Replace EVERY contiguous run of lines whose STRIPPED text is ``needles``.

    A LINE-WISE NEEDLE CANNOT REACH A WRAPPED CALL. The certified body's partner
    terms and masked row sums are multi-line calls whose OPENING line is not
    unique (the ``R01`` branch and the ``R02``-only else branch spell the same
    call), so the anchor is the whole stripped run and the hit count is reported
    rather than required to be one.
    """
    lines = source.split("\n")
    stripped = [line.strip() for line in lines]
    span = len(needles)
    wanted = list(needles)
    hits = [index for index in range(len(lines) - span + 1)
            if stripped[index:index + span] == wanted]
    if not hits:
        return source, 0
    for index in reversed(hits):
        indent = lines[index][:len(lines[index]) - len(lines[index].lstrip())]
        lines[index:index + span] = [indent + line for line in replacement]
    return "\n".join(lines), len(hits)


#: The five stripped lines of Ex's FIRST partner term, under ``if R01:``. Anchored
#: as a block because the opening line alone appears twice in the certified body.
_EX_PARTNER_TERM = (
    "total0 = _folded_dispersive_term(",
    "g1, b0, b1, b2, b3, b4, b5, b6, b7, u01,",
    "idx, i * nyz + dj * nz + k, ui * nyz + j * nz + k,",
    "ui * nyz + dj * nz + k, live, dvy, uvx, uvx & dvy,",
    "wy, NP1, MG_Y)",
)

#: Ex's masked row sum, in BOTH branches that spell it.
_EX_ROW_SUM = (
    "src0 = _masked_row_sum(",
    "gs0 * us0, total0, at_y, at_z, WM_Y, WM_Z)",
)


#: ``(id, case, slot, why, expectation, rewrite)``. ``slot`` is the pole slot the
#: edit lands in and :func:`leg_no_dead_mutations` refuses any the scored case does
#: not reach — the constexpr chains mean an edit to an unlived slot is emitted
#: NOWHERE and measures nothing. ``expectation`` is what the design predicts; the
#: artifact records what was MEASURED, and a predicted null that turns out visible
#: (or the reverse) is reported either way.
def mutation_table() -> Tuple[Tuple[str, str, int, str, str,
                                    Callable[[str], Tuple[str, int]]], ...]:

    def m1_drive_reloaded_from_f_w(source: str) -> Tuple[str, int]:
        needle = "w = src0"
        return (source.replace(
            needle, "w = tl.load(w0 + idx, mask=live, other=0.0)"),
            source.count(needle))

    def m1s_drive_from_the_stored_E(source: str) -> Tuple[str, int]:
        needle = "w = src0"
        return (source.replace(
            needle, "w = tl.load(f0 + idx, mask=live, other=0.0)"),
            source.count(needle))

    def m2_seam_takes_the_pre_constitutive_source(source: str) -> Tuple[str, int]:
        needle = "w = src0"
        return source.replace(needle, "w = gs0"), source.count(needle)

    def m3_pole_read_through_the_wrong_component(source: str) -> Tuple[str, int]:
        needle = "p = tl.load(a0 + idx"
        return source.replace(needle, "p = tl.load(b0 + idx"), source.count(needle)

    def m4_ade_association_flattened(source: str) -> Tuple[str, int]:
        needle = "((p * c_now) + (c_prev * q)) + (c_drive * (s * w))"
        return (source.replace(needle, "p * c_now + c_prev * q + c_drive * s * w"),
                source.count(needle))

    def m5_ade_p_and_q_swapped(source: str) -> Tuple[str, int]:
        return _swap_two(source, "p = tl.load(a0 + idx",
                         "q = tl.load(p_prev_a0 + idx", (
                             "p = tl.load(p_prev_a0 + idx, mask=live, other=0.0)",
                             "q = tl.load(a0 + idx, mask=live, other=0.0)"))

    def m6_c_now_and_c_prev_swapped(source: str) -> Tuple[str, int]:
        return _swap_two(source, "c_now = cnow_a0", "c_prev = cprev_a0",
                         ("c_now = cprev_a0", "c_prev = cnow_a0"))

    def m7_second_arm_never_runs(source: str) -> Tuple[str, int]:
        # THE E-CHAIN DROP IS NOT REACHABLE IN THE CALLER, so this is its ADE
        # twin: the guard is raised past every live count and slot 1's recurrence
        # never advances while the E chain keeps subtracting it — a frozen P. The
        # certified helper's own `if NP > k` links are covered by the E half's
        # released weld and by the hand-CUDA sibling's pole_chain_dropped.
        return _replace_span(source, "if NP0 > 1:", 1, ("if NP0 > 7:",))

    def m8_ade_arm_one_never_stores(source: str) -> Tuple[str, int]:
        return _replace_span(source, "tl.store(p_out_a1 + idx,", 3, ("pass",))

    def m9_sigma_index_off_by_one(source: str) -> Tuple[str, int]:
        needle = "s = tl.load(sigma_a1 + idx, mask=live, other=0.0)"
        return (source.replace(
            needle, "s = tl.load(sigma_a0 + idx, mask=live, other=0.0)"),
            source.count(needle))

    def m10_commuted_multiply(source: str) -> Tuple[str, int]:
        needle = "(c_drive * (s * w))"
        return source.replace(needle, "(c_drive * (w * s))"), source.count(needle)

    def m11_pml_accumulations_reversed(source: str) -> Tuple[str, int]:
        return _swap_two(source, "value0 = value0 + kp_0 * src0",
                         "value0 = value0 - km_0 * prev0",
                         ("value0 = value0 - km_0 * src0",
                          "value0 = value0 + kp_0 * prev0"))

    def m12_fw_store_dropped(source: str) -> Tuple[str, int]:
        return _replace_span(source, "tl.store(w0 + idx, src0, mask=live)", 1,
                             ("pass",))

    def m14_arm_reads_another_components_drive(source: str) -> Tuple[str, int]:
        # EZ'S ARMS, READING EX'S DRIVE. The direction matters: an EARLIER
        # component reading a LATER one's register is a NameError the compiler
        # refuses, which measures the emission order rather than the seam. This
        # direction compiles and is a wrong answer.
        needle = "w = src2"
        return source.replace(needle, "w = src0"), source.count(needle)

    def m15_partner_term_dropped(source: str) -> Tuple[str, int]:
        # THE ROW-PRODUCT NEEDLE. One of Ex's two off-diagonal partner terms
        # zeroed: the tensor coupling this whole cell exists for, gone, with the
        # diagonal answer left standing.
        return _replace_block(source, _EX_PARTNER_TERM, ("total0 = gs0 * 0.0",))

    def m16_mirror_ghost_without_parity(source: str) -> Tuple[str, int]:
        needle = "wy = tl.where(at_y, gwy, 1.0)"
        return source.replace(needle, "wy = 1.0"), source.count(needle)

    def m17_row_sum_before_the_wall_mask(source: str) -> Tuple[str, int]:
        return _replace_block(source, _EX_ROW_SUM, ("src0 = gs0 * us0 + total0",))

    target = MUTATION_CASE
    return (
        ("m1_drive_reloaded_from_f_w", target, 0,
         "the fusion itself, undone: read f_w back instead of keeping the register",
         "null", m1_drive_reloaded_from_f_w),
        ("m1s_drive_from_the_stored_E", target, 0,
         "the stored E and f_w agree everywhere EXCEPT inside the layer",
         "caught", m1s_drive_from_the_stored_E),
        ("m2_seam_takes_the_pre_constitutive_source", target, 0,
         "feed update_P the D-minus-P before the inverse epsilon and the row sum",
         "caught", m2_seam_takes_the_pre_constitutive_source),
        ("m3_pole_read_through_the_wrong_component", target, 0,
         "Ex's arm advances from Ey's P: the re-read pointer, mis-bound",
         "caught", m3_pole_read_through_the_wrong_component),
        ("m4_ade_association_flattened", target, 0,
         "ade_update_p's parentheses ARE the float32 answer",
         "caught", m4_ade_association_flattened),
        ("m5_ade_p_and_q_swapped", target, 0,
         "advance the recurrence from the history and the history from P",
         "caught", m5_ade_p_and_q_swapped),
        ("m6_c_now_and_c_prev_swapped", target, 0,
         "the recurrence constants, crossed",
         "caught", m6_c_now_and_c_prev_swapped),
        ("m7_second_arm_never_runs", target, 1,
         "a frozen P: the E chain keeps subtracting a pole the ADE half stopped "
         "advancing",
         "caught", m7_second_arm_never_runs),
        ("m8_ade_arm_one_never_stores", target, 1,
         "the same frozen P, from the store side",
         "caught", m8_ade_arm_one_never_stores),
        ("m9_sigma_index_off_by_one", target, 1,
         "pole 1's recurrence driven through pole 0's sigma volume",
         "caught", m9_sigma_index_off_by_one),
        ("m10_commuted_multiply", target, 0,
         "float32 multiplication is bitwise commutative: a DECLARED null control",
         "null", m10_commuted_multiply),
        ("m11_pml_accumulations_reversed", target, 0,
         "the split-field accumulation with its two coefficients crossed",
         "caught", m11_pml_accumulations_reversed),
        ("m12_fw_store_dropped", target, 0,
         "a stale f_w: next step's PML history term is one step old, and the "
         "drive this launch hands the recurrence never lands",
         "caught", m12_fw_store_dropped),
        ("m14_arm_reads_another_components_drive", target, 0,
         "Ez's poles advanced from Ex's drive — the cross-component seam error",
         "caught", m14_arm_reads_another_components_drive),
        ("m15_partner_term_dropped", target, 0,
         "Ex's first off-diagonal partner term zeroed: the tensor coupling gone",
         "caught", m15_partner_term_dropped),
        ("m16_mirror_ghost_without_parity", target, 0,
         "the mirror image taken without its parity weight",
         "caught", m16_mirror_ghost_without_parity),
        ("m17_row_sum_before_the_wall_mask", MUTATION_CASE_METALLIC, 0,
         "the row sum applied THROUGH a metallic wall the mask exists to cut",
         "caught", m17_row_sum_before_the_wall_mask),
    )


def leg_no_dead_mutations(xp) -> Dict[str, Any]:
    """Every armed edit must land in a slot the scored case actually carries.

    THE DEAD MUTATION is a defect class this project has paid for: an edit rewrites
    real lines the scored configuration never emits — here because the ``NP``
    constexpr chain unrolls only the live slots — and then reports UNCAUGHT while
    measuring nothing.
    """
    from meep_gpu.triton_kernels import (  # noqa: PLC0415
        folded_offdiag_fused_ade_chain as family,
    )

    counts: Dict[str, Tuple[int, ...]] = {}
    walls: Dict[str, Tuple[int, ...]] = {}
    for label in {MUTATION_CASE, MUTATION_CASE_METALLIC}:
        case = _case(label)
        _, fields, pml = build(xp, case, case_seed(case.label, "dead"))
        plan = family.FoldedOffdiagFusedAdeChainPlan(fields, pml, 256, num_warps=1)
        counts[label] = tuple(plan.counts)
        walls[label] = tuple(plan._e._base.wall_axes)
    rows = []
    for identifier, label, slot, _why, _expectation, _rewrite in mutation_table():
        live = slot < min(counts[label])
        # THE WALL-MASK EDIT NEEDS A WALL. Its case must actually terminate an
        # axis metallically, or the rewritten line computes the same number and
        # the row would report UNCAUGHT about a mask it never crossed.
        needs_wall = identifier == "m17_row_sum_before_the_wall_mask"
        armed = live and (not needs_wall or any(walls[label]))
        rows.append({"mutation": identifier, "case": label, "slot": slot,
                     "case_pole_counts": list(counts[label]),
                     "wall_axes": list(walls[label]),
                     "slot_is_live": live, "agrees": armed})
    return {"passed": all(entry["agrees"] for entry in rows),
            "pole_counts": {label: list(value) for label, value in counts.items()},
            "wall_axes": {label: list(value) for label, value in walls.items()},
            "rows": rows}


def run_mutations(xp, family, pristine_ptx: Sequence[str],
                  steps: int) -> List[Dict[str, Any]]:
    source = shipped_source(family)
    rows: List[Dict[str, Any]] = []
    for identifier, label, slot, why, expectation, rewrite in mutation_table():
        case = _case(label)
        mutated, hits = rewrite(source)
        entry: Dict[str, Any] = {
            "mutation": identifier, "case": case.label, "slot": slot, "why": why,
            "expectation": expectation, "needle_hits": hits,
            "steps_budget": steps}
        if hits == 0 or mutated == source:
            entry["status"] = "NEEDLE-MISSED"
            entry["passed"] = False
            entry["failure"] = ("the rewrite matched nothing; this mutation was "
                                "never armed")
            rows.append(entry)
            log(f"  {identifier}: NEEDLE-MISSED")
            continue
        kernel_name = f"mutant_{identifier}"
        body = mutated.replace("def folded_offdiag_fused_ade_chain_step(",
                               f"def {kernel_name}(")
        try:
            mutant = compile_mutant(body, kernel_name)
        except Exception as exc:  # noqa: BLE001
            entry["status"] = "UNCOMPILABLE"
            entry["passed"] = False
            entry["failure"] = f"{type(exc).__name__}: {exc}"[:400]
            rows.append(entry)
            log(f"  {identifier}: UNCOMPILABLE {exc}")
            continue
        result = run_case(xp, case, f"mutation:{identifier}", steps, kernel=mutant)
        normalise = (lambda text: text.replace(
            kernel_name, "folded_offdiag_fused_ade_chain_step"))
        mutant_ptx = [normalise(text) for text in kernel_ptx(mutant)]
        entry["launches"] = result.get("launches")
        entry["error"] = result.get("error")
        entry["ptx_specializations"] = len(mutant_ptx)
        entry["ptx_differs_from_shipped"] = bool(
            mutant_ptx and pristine_ptx
            and all(text not in set(pristine_ptx) for text in mutant_ptx))
        entry["first_divergence"] = result.get("first_divergence")
        entry["differing_words"] = result.get("differing_words")
        caught = result.get("first_divergence") is not None or bool(
            result.get("error"))
        if not entry["launches"] and not result.get("error"):
            entry["status"] = "DISARMED"
            entry["note"] = "the mutant kernel was never launched"
        elif caught:
            entry["status"] = "CAUGHT"
        elif not entry["ptx_differs_from_shipped"]:
            entry["status"] = "PTX-IDENTICAL"
            entry["note"] = (
                "the mutant compiled to PTX already present for the shipped "
                "kernel; the source change was canonicalized away and nothing "
                "was tested by it")
        else:
            entry["status"] = "NULL CONFIRMED"
            entry["note"] = (
                "launched, PTX differs, and the gate saw no byte difference: "
                "recorded as a measured null with its evidence")
        entry["passed"] = bool(
            (expectation == "caught" and entry["status"] == "CAUGHT")
            or (expectation == "null"
                and entry["status"] in ("NULL CONFIRMED", "PTX-IDENTICAL")))
        rows.append(entry)
        log(f"  {identifier}: {entry['status']} launches={entry['launches']} "
            f"ptx_differs={entry['ptx_differs_from_shipped']} "
            f"expected={expectation}")
    return rows


# ---------------------------------------------------------------------------
# Armed HOST mutations — defects no kernel edit can reach
# ---------------------------------------------------------------------------

def swap_sigma_between_components(plan: Any) -> Dict[str, Any]:
    """Bind Ey's sigma to Ex's arms.

    The kernel takes a pointer (or a scalar) per arm and never asks which
    component's sigma it is. A per-component sigma is exactly what an anisotropic
    material grid gives, so this is a silent wrong coupling rather than a crash,
    and no kernel edit can reach it.
    """
    changed = 0
    for slot in range(len(plan._order["Ex"])):
        left, right = plan._sigma[("Ex", slot)], plan._sigma[("Ey", slot)]
        if hasattr(left, "array") or hasattr(right, "array"):
            assert getattr(left, "array", left) is not getattr(right, "array", left), (
                "Ex and Ey hold the SAME sigma array, so this mutation is a null "
                "and must not be scored caught")
        else:
            assert left != right, (
                "Ex and Ey hold the SAME scalar sigma, so this mutation is a null")
        plan._sigma[("Ex", slot)], plan._sigma[("Ey", slot)] = right, left
        changed += 1
    return {"slots_swapped": changed}


def reverse_ade_coefficients_on_Ex(plan: Any) -> Dict[str, Any]:
    """Pair pole k's displacement subtraction with pole j's recurrence.

    THE MISPAIRED ARM. The E chain still subtracts the poles in order; only the
    coefficients the ADE arms use are reversed, so every arm advances the right
    volume with the wrong susceptibility's constants — smooth, plausible and wrong.
    """
    slots = len(plan._order["Ex"])
    triples = [plan._coeff[("Ex", slot)] for slot in range(slots)]
    assert len(set(triples)) > 1, "the poles share one coefficient triple"
    for slot, triple in enumerate(reversed(triples)):
        plan._coeff[("Ex", slot)] = triple
    return {"slots_reversed": slots}


def swap_c_prev_and_c_drive(plan: Any) -> Dict[str, Any]:
    """The recurrence constants come from the susceptibility and dt
    (dispersion.py:644). Swapping two of them is a plausible packing slip that no
    kernel edit can reach — the body reads ``cprev_a0`` either way."""
    changed = 0
    for key, (c_now, c_prev, c_drive) in list(plan._coeff.items()):
        plan._coeff[key] = (c_now, c_drive, c_prev)
        changed += 1
    return {"triples_swapped": changed}


def swap_inverse_epsilon(plan: Any) -> Dict[str, Any]:
    """LIVE on this configuration, and the identity that makes it live is asserted.

    The fixture installs DISTINCT per-component diagonal volumes through
    ``set_epsilon_volumes``, which is what an anisotropic grid gives and what the
    plain chain's isotropic fixture could not carry — so binding Ey's inverse
    epsilon to Ex's slot is a wrong answer here rather than a rebinding of the
    identical object.
    """
    base = plan._e._base
    first, second = base._inv_eps[0], base._inv_eps[1]
    assert first.array is not second.array, (
        "the two components hold the SAME inverse-epsilon array, so this "
        "mutation is a null and must not be scored caught")
    inverse = list(base._inv_eps)
    inverse[0], inverse[1] = second, first
    base._inv_eps = tuple(inverse)
    return {"components_swapped": ["Ex", "Ey"]}


def swap_row_volume_slots(plan: Any) -> Dict[str, Any]:
    """Bind Ex's z-partner coefficient where its y-partner belongs.

    ``ROW_SLOTS`` pairs a coefficient volume with a partner term, and the pairing
    is the one thing a launcher can get wrong without any array being invalid. No
    kernel edit reaches it: the body reads ``u01`` and ``u02`` either way.
    """
    base = plan._e._base
    rows = list(base._rows)
    assert rows[0] is not rows[1], "the two Ex row slots are the same pointer"
    assert getattr(rows[0], "array", None) is not getattr(rows[1], "array", 1), (
        "the two Ex row slots hold one array, so this mutation is a null")
    rows[0], rows[1] = rows[1], rows[0]
    base._rows = tuple(rows)
    return {"slots_swapped": ["u01", "u02"]}


def lie_about_the_sigma_kind(plan: Any) -> Dict[str, Any]:
    """Tell the constexpr the volume sigma is a scalar.

    ``coverage.sigma_is_volume`` is the ONE place that decides ``SV_*`` precisely
    so the clause and the constexpr cannot disagree; this is what disagreeing looks
    like from the launcher's side — a pointer read as a float.
    """
    assert plan._sv["SV_a1"] == 1, "the scored case has no volume sigma on Ex slot 1"
    plan._sv["SV_a1"] = 0
    return {"flag": "SV_a1 forced to 0 while a pointer is bound"}


#: Undo callbacks for the two ordering mutations. THE PLAN DECLARES ``__slots__``,
#: so neither ``run`` nor ``_rotate`` can be rebound on an instance and the patch
#: has to land on the CLASS — which would then outlive its case. Every patch
#: registers its restore here, :func:`_drain_undo` runs it, and ``run_case`` drains
#: on entry as well as ``main`` in a ``finally``: a leaked class patch would make
#: every LATER row a measurement of the wrong program.
_UNDO: List[Callable[[], None]] = []


def _patch_class(owner: Any, name: str, replacement: Any) -> None:
    original = getattr(owner, name)
    setattr(owner, name, replacement)
    _UNDO.append(lambda: setattr(owner, name, original))


def _drain_undo() -> int:
    count = len(_UNDO)
    while _UNDO:
        _UNDO.pop()()
    return count


def rotate_before_the_launch(plan: Any) -> Dict[str, Any]:
    """Rotate first, launch second — the ordering the plan deliberately does not use.

    A plan that rotated first would hand each arm the buffer the NEXT step's roles
    name, so P and P_prev are one cycle out for the whole run.
    """
    original = type(plan).run

    def run(self, guard=None):
        self._rotate(self._resolve(self._e._poles.arrays()))
        original(self, guard)

    _patch_class(type(plan), "run", run)
    return {"ordering": "rotate then launch"}


def skip_the_rotation(plan: Any) -> Dict[str, Any]:
    """Never rotate: every step writes its result into the same scratch and the
    history the next step reads is the one before last."""
    _patch_class(type(plan), "_rotate", lambda self, chain: None)
    return {"rotation": "disabled"}


#: name -> (patch, expected outcome). "caught" must diverge (or raise); "null" must
#: NOT, and its patch asserts the identity that makes it a null.
HOST_MUTATIONS: Dict[str, Tuple[Callable[[Any], Dict[str, Any]], str]] = {
    "sigma_bound_for_the_wrong_component": (swap_sigma_between_components, "caught"),
    "ade_coefficients_reversed_on_Ex": (reverse_ade_coefficients_on_Ex, "caught"),
    "c_prev_and_c_drive_swapped": (swap_c_prev_and_c_drive, "caught"),
    "sigma_constexpr_disagrees_with_the_binding": (lie_about_the_sigma_kind, "caught"),
    "inverse_epsilon_bound_for_the_wrong_component": (swap_inverse_epsilon, "caught"),
    "row_volume_bound_to_the_wrong_slot": (swap_row_volume_slots, "caught"),
    "rotate_before_launch": (rotate_before_the_launch, "caught"),
    "rotation_skipped": (skip_the_rotation, "caught"),
}


# ---------------------------------------------------------------------------
# The planted defect — the verdict itself, shown to fail
# ---------------------------------------------------------------------------

#: A mutation leg proves that ONE case diverges when ONE kernel is wrong; it does
#: not prove the gate's RELEASE VERDICT can fail. These edits replace the family's
#: SHIPPED kernel for the whole process, so the complete gate — product, separate
#: control, block sweep, value classes, disarm — runs against defective bytes and
#: the top-level verdict must flip to FAIL. Run with ``--plant <name>`` and require
#: a nonzero exit.
PLANTED: Dict[str, Tuple[Tuple[str, str], ...]] = {
    "ade_association_flattened_everywhere": (
        ("((p * c_now) + (c_prev * q)) + (c_drive * (s * w))",
         "p * c_now + c_prev * q + c_drive * s * w"),),
    "seam_reads_the_stale_pole_instead_of_the_drive": (
        ("w = src0", "w = p"), ("w = src1", "w = p"), ("w = src2", "w = p")),
}


def plant(family, name: str) -> Dict[str, Any]:
    """Replace the family's shipped kernel with a defective one, process-wide."""
    source = shipped_source(family)
    mutated = source
    hits = {}
    for needle, replacement in PLANTED[name]:
        hits[needle] = mutated.count(needle)
        if not hits[needle]:
            raise SystemExit(
                f"planted defect {name!r} matches nothing at {needle!r}; a marker "
                f"that matches nothing disables the check it feeds SILENTLY")
        mutated = mutated.replace(needle, replacement)
    kernel_name = "planted_" + name
    body = mutated.replace("def folded_offdiag_fused_ade_chain_step(",
                           f"def {kernel_name}(")
    family.folded_offdiag_fused_ade_chain_step = compile_mutant(body, kernel_name)
    return {"planted": name, "needle_hits": hits}


# ---------------------------------------------------------------------------
# The optional corpus-lift leg — NOT release-gating
# ---------------------------------------------------------------------------

def leg_corpus_lift(xp, corpus_root: Optional[str], steps: int = 20
                    ) -> Dict[str, Any]:
    """The real ``absorbed_power_density.py`` objects, stepped through this weld.

    NOT RELEASE-GATING, and the reason is measured rather than preferred: neither
    Triton chain gate has such a leg, the row's ADMISSION is what the census
    measures on the lifted objects, and its BYTES are measured by the scaled
    ``absorbed_power_density_shape`` case above. This leg is carried so that a
    later round can promote it without inventing a harness, and it records SKIPPED
    by name when no corpus is staged.
    """
    if not corpus_root:
        return {"passed": True, "measured": False, "status": "SKIPPED",
                "reason": "no --corpus-root; the row's admission is the census's "
                          "and its bytes are the scaled shape case's",
                "release_gating": False}
    script = os.path.join(corpus_root, "examples", "absorbed_power_density.py")
    record: Dict[str, Any] = {"script": script, "release_gating": False,
                              "steps": steps}
    if not os.path.exists(script):
        record.update(passed=False, measured=False, status="MISSING",
                      reason=f"{script} does not exist")
        return record
    try:
        from sweep_corpus_lift_parity import capture_simulation  # noqa: PLC0415
        import meep_gpu  # noqa: PLC0415
        from meep_gpu.triton_kernels import (  # noqa: PLC0415
            folded_offdiag_fused_ade_chain as family,
        )

        drivers = []
        for _route in range(2):
            _, simulation, restore = capture_simulation(script)
            if simulation is None:
                record.update(passed=False, measured=False, status="NO_SIMULATION")
                return record
            restore()
            drivers.append(meep_gpu.lift_simulation(simulation, prefer_gpu=True))
        reference, actual = drivers
        plan = family.plan_folded_offdiag_fused_ade_chain(
            actual.fields, actual.pml, num_warps=1)
        if plan is None:
            record.update(
                passed=False, measured=True, status="REFUSED",
                reasons=list(family.folded_offdiag_fused_ade_chain_coverage(
                    actual.fields, actual.pml).reasons)[:6])
            return record
        per_step = []
        for step in range(1, steps + 1):
            seam_step(reference.fields, reference.pml)
            plan.run()
            difference = compare(snapshot(reference.fields),
                                 snapshot(actual.fields, plan))
            per_step.append({"step": step,
                             "differing_words": sum(difference.values())})
            if difference:
                break
        record.update(
            measured=True, status="MEASURED",
            steps_compared=len(per_step),
            differing_words=per_step[-1]["differing_words"] if per_step else None,
            launches=plan.launches,
            passed=bool(len(per_step) == steps
                        and all(row["differing_words"] == 0 for row in per_step)))
        for driver in drivers:
            try:
                driver.close()
            except Exception:  # noqa: BLE001
                pass
    except Exception as exc:  # noqa: BLE001
        record.update(passed=False, measured=False, status="UNAVAILABLE",
                      error=f"{type(exc).__name__}: {exc}"[:400])
    return record


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def emit(handle: Any, row: Dict[str, Any]) -> None:
    handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
    handle.flush()
    os.fsync(handle.fileno())
    log(f"{row['index']}/{row['total']} [{row['leg']}] {row['label']}: "
        f"passed={row.get('passed')} diff={row.get('differing_words', '-')} "
        f"launches={row.get('launches', '-')} "
        f"elapsed={row.get('elapsed_seconds', '-')}")


def environment(xp) -> Dict[str, Any]:
    """What ran this, in the shape the WELD PIPELINE reads.

    ``hostname``, ``device``, ``compute_capability``, ``triton``, ``cupy`` and
    ``cuda_visible_devices`` are the six ``rebind_triton_welds._host_line`` builds
    a weld's ``host`` string from, and it returns None — SKIPPING the entry rather
    than binding it half-known — if any is missing; ``seed_triton_welds`` refuses
    an artifact lacking the device and the compute capability outright. The
    capability is not decoration: ``test_triton_weld_contract`` binds it and the
    Triton version to the record's ``validated_compute_capabilities`` /
    ``validated_triton_versions``, because Triton generates PTX for an
    ARCHITECTURE and a weld cut on an undeclared one must fail until the
    declaration is widened deliberately.

    ``device`` and ``device_name`` carry the SAME string under both spellings: the
    minting tool reads one and the fleet driver's campaign row reads the other.

    THE BLOCK IS THEN HANDED TO ``triton_device_identity.record``, which is the one
    place on this backend that decides the SPELLING. It ``setdefault``s — so every
    value measured below is kept, since this gate measures them from the device it
    launched through — and it NORMALISES ``compute_capability`` to ``"8.6"``,
    because CuPy hands the number out as ``"86"`` too and the weld contract tests a
    SUBSTRING of the composed host line against ``validated_compute_capabilities``.
    Two spellings of one architecture is what makes that test decide a gate by
    which reader answered first.
    """
    record: Dict[str, Any] = {
        "hostname": socket.gethostname(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "argv": list(sys.argv),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cupy_cache_dir": os.environ.get("CUPY_CACHE_DIR"),
        "triton_cache_dir": os.environ.get("TRITON_CACHE_DIR"),
        # The chain gate's own spellings, kept so a reader of either artifact
        # finds the same fact under the name it knows.
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "CUPY_CACHE_DIR": os.environ.get("CUPY_CACHE_DIR"),
        "TRITON_CACHE_DIR": os.environ.get("TRITON_CACHE_DIR"),
    }
    try:
        import triton  # noqa: PLC0415

        record["triton"] = getattr(triton, "__version__", "?")
    except Exception as exc:  # noqa: BLE001
        record["triton"] = None
        record["triton_error"] = repr(exc)
    if xp is not np:
        record["cupy"] = getattr(xp, "__version__", "?")
        try:
            device = xp.cuda.Device()
            properties = xp.cuda.runtime.getDeviceProperties(device.id)
            name = properties["name"].decode()
            record["device_id"] = int(device.id)
            record["device"] = name
            record["device_name"] = name
            record["compute_capability"] = (
                f"{properties['major']}.{properties['minor']}")
            free, total = xp.cuda.runtime.memGetInfo()
            record["device_free_bytes"] = int(free)
            record["device_total_bytes"] = int(total)
        except Exception as exc:  # noqa: BLE001
            record["device_error"] = repr(exc)
    import triton_device_identity  # noqa: PLC0415

    return triton_device_identity.record(record)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="artifact FILE path")
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--mutation-steps", type=int, default=MUTATION_STEPS)
    parser.add_argument("--block-sizes",
                        default=",".join(str(size) for size in BLOCK_SIZES),
                        help="comma-separated BLOCK constexprs for the S2 sweep")
    parser.add_argument("--corpus-root", default=None,
                        help="a MEEP python/ checkout; enables the OPTIONAL, "
                             "non-gating corpus-lift leg")
    parser.add_argument("--subnormal-policy", default="keep",
                        choices=("keep", "flush"),
                        help="the float32 subnormal policy to DRIVE EVERY EXECUTOR "
                             "TO before the first device compile. One process per "
                             "policy: CuPy's cache key sits ABOVE the -ftz strip, "
                             "so a shared cache serves flushed binaries under the "
                             "keeping record's name.")
    parser.add_argument("--no-device", action="store_true",
                        help="run only the legs that need neither CUDA nor Triton")
    parser.add_argument("--plant", choices=sorted(PLANTED), default=None,
                        help="run the whole gate against a defective SHIPPED "
                             "kernel; the verdict below MUST come back FAIL")
    args = parser.parse_args(argv)
    if args.steps < 1:
        raise SystemExit("--steps must be positive")
    block_sizes = tuple(int(part) for part in args.block_sizes.split(",") if part)
    if not block_sizes:
        raise SystemExit("--block-sizes named no size")

    started = time.perf_counter()
    payload: Dict[str, Any] = {
        "gate": "triton_folded_offdiag_fused_ade_chain",
        "product": "folded_offdiag_fused_ade_chain",
        "seam": "E->P",
        "cell": ["folded off-diagonal dispersive", "ADE update_P"],
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "steps": args.steps, "mutation_steps": args.mutation_steps,
        "block_sizes": list(block_sizes),
        "no_device": bool(args.no_device),
        "device_status": "NO_DEVICE" if args.no_device else "RUN",
        "planted_defect": args.plant,
        "verdict_flip_expected": bool(args.plant),
        "source_sha256": source_hashes(),
        "rows": [],
    }
    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    jsonl = os.path.splitext(out)[0] + ".jsonl"

    xp: Any = np
    if not args.no_device:
        # CUPY FIXES ITS REDUCTION-ACCELERATOR LIST AT IMPORT, and
        # ``set_reduction_accelerators([])`` does not move where a reduction is
        # dispatched: measured on device (subnormal_policy.py:1112-1120), ``cp.sum``
        # over one 2^-135 among 4095 zeros returned the KEPT word with the list
        # reported empty, because CUB is a binary built with CuPy and the dispatch
        # reads a list fixed at import. There is no in-process route to 'flush' for
        # reductions, so the variable is set BEFORE the import and recorded.
        if args.subnormal_policy == "flush":
            os.environ["CUPY_ACCELERATORS"] = ""
        if "cupy" in sys.modules:
            raise SystemExit(
                "cupy was imported before this gate could set CUPY_ACCELERATORS; "
                "the accelerator list is fixed at import and the requested policy "
                "would be a wish rather than a configuration")
        import cupy as cp  # noqa: PLC0415
        from meep_gpu import backends as _backends  # noqa: PLC0415
        from meep_gpu import subnormal_policy  # noqa: PLC0415

        xp = cp
        # THE HOST LEVER. ``MEEP``'s ``set_zero_subnormals`` is the ONLY exposure
        # of this process's FTZ/DAZ bits this package may use
        # (subnormal_policy.py:1911-1922 refuses without it), so a policy the host
        # is not already sitting on needs MEEP imported first. It is imported ONLY
        # when the measured host state disagrees with the request: importing it
        # initialises MPI as a side effect, and perturbing a configuration that
        # already satisfies the policy would change what the artifact describes.
        want_flush = args.subnormal_policy == "flush"
        before = bool(_backends.subnormals_flushed())
        lever: Dict[str, Any] = {"host_flushing_before": before,
                                 "meep_imported": False}
        if before != want_flush:
            import meep  # noqa: PLC0415, F401

            lever.update(meep_imported=True,
                         meep_version=getattr(meep, "__version__", "unknown"),
                         host_flushing_after_import=bool(
                             _backends.subnormals_flushed()))
        payload["host_subnormal_lever"] = lever
        payload["cupy_accelerators"] = os.environ.get("CUPY_ACCELERATORS")
        log(f"host subnormal lever: {json.dumps(lever, sort_keys=True)}")
        # STRICT, and BEFORE the first device compile: an artifact whose policy
        # field is a wish is worse than none. This also enforces the CUPY_CACHE_DIR
        # token rule (subnormal_policy.cupy_cache_reasons).
        subnormal_policy.install_subnormal_policy(
            args.subnormal_policy, cupy=cp, strict=True)
        payload["subnormal_policy"] = subnormal_policy.policy_stamp()
        payload["subnormal_policy_requested"] = args.subnormal_policy
        global _POLICY
        _POLICY = args.subnormal_policy
        log(f"subnormal policy installed: {args.subnormal_policy!r}")
    payload["environment"] = environment(xp)
    log(f"environment: {json.dumps(payload['environment'], sort_keys=True)}")

    from meep_gpu.triton_kernels import (  # noqa: PLC0415
        folded_offdiag_fused_ade_chain as family,
    )

    if args.plant:
        payload["planted"] = plant(family, args.plant)
        log(f"PLANTED DEFECT {args.plant!r}: the verdict below MUST be FAIL")

    value_cases: Tuple[Case, ...] = ()
    controls: Tuple[str, ...] = ()
    sweep_cases: Tuple[str, ...] = ()
    if not args.no_device:
        value_cases = tuple(
            [dataclasses.replace(_case(name), label=f"{name}:pm_zero_lattice",
                                 value_class="pm_zero_lattice")
             for name in ("one_state_all_components", "two_states_mixed_sigma")]
            + [dataclasses.replace(_case(name), label=f"{name}:subnormal_band",
                                   value_class="subnormal_band", scale=1e-34)
               for name in ("two_states_mixed_sigma",
                            "absorbed_power_density_shape")])
        controls = ("one_state_all_components", "two_states_mixed_sigma",
                    "metallic_terminated_fold", "absorbed_power_density_shape")
        sweep_cases = ("two_states_mixed_sigma", "absorbed_power_density_shape")

    static_legs: List[Tuple[str, str, Callable[[], Dict[str, Any]]]] = [
        ("transcription", "the_certified_body_and_the_certified_recurrence",
         leg_transcription),
        ("shape", "the_rotation_orbit_walked_to_closure", lambda: leg_shape(xp)),
        ("shape", "the_plan_refuses_a_broken_chain", lambda: leg_plan_refusals(xp)),
        ("refusals", "the_predicate_table", lambda: leg_refusals(xp)),
    ]
    if not args.no_device and args.plant is None:
        static_legs.append(("mutation", "no_armed_edit_lands_in_a_dead_slot",
                            lambda: leg_no_dead_mutations(xp)))

    total = (len(static_legs)
             + (0 if args.no_device else
                len(CASES) + len(controls) + len(sweep_cases)
                + len(value_cases) + 2))
    if not args.no_device and args.plant is None:
        total += len(mutation_table()) + len(HOST_MUTATIONS) + 1

    rows: List[Dict[str, Any]] = []
    index = 0
    with open(jsonl, "w", encoding="utf-8") as handle:
        for leg, label, runner in static_legs:
            index += 1
            clock = time.perf_counter()
            row = {"index": index, "total": total, "leg": leg, "label": label,
                   **runner()}
            row["elapsed_seconds"] = round(time.perf_counter() - clock, 2)
            rows.append(row)
            emit(handle, row)
            payload["rows"] = rows
            save(payload, out)

        if not args.no_device:
            for case in CASES:
                index += 1
                clock = time.perf_counter()
                row = {"index": index, "total": total, "leg": "product",
                       "label": case.label, "steps": args.steps,
                       **run_case(xp, case, "product", args.steps)}
                row["elapsed_seconds"] = round(time.perf_counter() - clock, 2)
                rows.append(row)
                emit(handle, row)
                payload["rows"] = rows
                save(payload, out)

            for name in controls:
                index += 1
                clock = time.perf_counter()
                row = {"index": index, "total": total, "leg": "separate_control",
                       "label": name, "steps": args.steps,
                       **run_separate_control(xp, _case(name), args.steps)}
                row["elapsed_seconds"] = round(time.perf_counter() - clock, 2)
                rows.append(row)
                emit(handle, row)
                payload["rows"] = rows
                save(payload, out)

            for name in sweep_cases:
                index += 1
                clock = time.perf_counter()
                row = {"index": index, "total": total, "leg": "block_sweep",
                       "label": f"{name}:S2",
                       **run_block_sweep(xp, _case(name), block_sizes, 12)}
                row["elapsed_seconds"] = round(time.perf_counter() - clock, 2)
                rows.append(row)
                emit(handle, row)
                payload["rows"] = rows
                save(payload, out)

            for case in value_cases:
                index += 1
                clock = time.perf_counter()
                row = {"index": index, "total": total, "leg": "value_classes",
                       "label": case.label, "steps": args.steps,
                       **run_case(xp, case, "value_classes", args.steps)}
                row["elapsed_seconds"] = round(time.perf_counter() - clock, 2)
                rows.append(row)
                emit(handle, row)
                payload["rows"] = rows
                save(payload, out)

            index += 1
            clock = time.perf_counter()
            row = {"index": index, "total": total, "leg": "value_classes",
                   "label": "the_subnormal_census_precondition",
                   **leg_precondition(xp, _case(MUTATION_CASE))}
            row["elapsed_seconds"] = round(time.perf_counter() - clock, 2)
            rows.append(row)
            emit(handle, row)
            payload["rows"] = rows
            save(payload, out)

            index += 1
            clock = time.perf_counter()
            row = {"index": index, "total": total, "leg": "corpus_lift",
                   "label": "absorbed_power_density_lifted",
                   **leg_corpus_lift(xp, args.corpus_root)}
            row["elapsed_seconds"] = round(time.perf_counter() - clock, 2)
            rows.append(row)
            emit(handle, row)
            payload["rows"] = rows
            save(payload, out)

        if not args.no_device and args.plant is None:
            pristine = kernel_ptx(family.folded_offdiag_fused_ade_chain_kernel())
            payload["shipped_ptx_specializations"] = len(pristine)
            for entry in run_mutations(xp, family, pristine, args.mutation_steps):
                index += 1
                rows.append({"index": index, "total": total, "leg": "mutation",
                             "label": entry["mutation"], **entry})
                emit(handle, rows[-1])
                payload["rows"] = rows
                save(payload, out)

            for name, (patch, expected) in HOST_MUTATIONS.items():
                index += 1
                case = _case(MUTATION_CASE)
                try:
                    result = run_case(xp, case, f"host:{name}", args.mutation_steps,
                                      patch=patch)
                    refused = None
                except AssertionError as exc:
                    result = {}
                    refused = f"the patch refused to arm: {exc}"[:300]
                finally:
                    restored = _drain_undo()
                caught = bool(result.get("error")) or (
                    result.get("first_divergence") is not None)
                outcome = "caught" if caught else "NULL CONFIRMED"
                rows.append({
                    "index": index, "total": total, "leg": "mutation",
                    "label": name, "host_defect": True, "expected": expected,
                    "outcome": outcome, "unarmed": refused,
                    "class_patches_restored": restored,
                    "passed": bool(refused is None
                                   and (outcome == "caught") == (expected == "caught")
                                   and (result.get("launches")
                                        or result.get("error"))),
                    "first_divergence": result.get("first_divergence"),
                    "differing_words": result.get("differing_words"),
                    "error": result.get("error"),
                    "patch": result.get("patch"),
                    "launches": result.get("launches")})
                emit(handle, rows[-1])
                payload["rows"] = rows
                save(payload, out)

            # THE DISARM CHECK. The identical harness, the identical case, the
            # SHIPPED bytes and no patch. A nonzero here would mean every "caught"
            # above is a harness that diverges on its own.
            index += 1
            result = run_case(xp, _case(MUTATION_CASE), "disarm",
                              args.mutation_steps)
            rows.append({"index": index, "total": total, "leg": "disarm",
                         "label": "shipped_bytes_on_the_mutation_case",
                         "differing_words": result.get("differing_words"),
                         "launches": result.get("launches"),
                         "written_arrays_that_never_moved": result.get(
                             "written_arrays_that_never_moved"),
                         "passed": bool(result.get("passed"))})
            emit(handle, rows[-1])
            payload["rows"] = rows
            save(payload, out)

    payload["rows"] = rows
    payload["counts"] = {
        "static": len(static_legs),
        "product": 0 if args.no_device else len(CASES),
        "separate_controls": len(controls),
        "block_sweeps": len(sweep_cases),
        "value_classes": len(value_cases),
        "mutations": sum(1 for row in rows if row["leg"] == "mutation"),
        "rows": len(rows),
    }
    gating = [row for row in rows if row["leg"] in RELEASE_GATING_LEGS]
    payload["failed"] = [row["label"] for row in gating if not row.get("passed")]
    payload["non_gating_failures"] = [
        row["label"] for row in rows
        if row["leg"] not in RELEASE_GATING_LEGS and not row.get("passed")]
    payload["verdict"] = "PASS" if not payload["failed"] else "FAIL"
    payload["passed"] = payload["verdict"] == "PASS"
    # THE GATE_BOUND SHAPE. ``build_triton_fusion_matrix``'s release-binding branch
    # reads exactly these three keys and refuses to credit a family whose artifact
    # is not `device_status == "RUN"`, `release.released` and `passed` with no
    # planted defect — which is what stops a ``--no-device`` run being read as a
    # release. The reasons are carried so a withheld credit says why.
    release_reasons: List[str] = []
    if args.no_device:
        release_reasons.append("no device: the byte legs did not run")
    if args.plant:
        release_reasons.append(f"planted defect {args.plant!r}")
    if payload["failed"]:
        release_reasons.append(f"{len(payload['failed'])} gating legs failed")
    payload["release"] = {"released": not release_reasons,
                          "reasons": release_reasons}
    payload["elapsed_seconds"] = round(time.perf_counter() - started, 2)
    payload["jsonl"] = jsonl
    save(payload, out)
    for name in _TEMPORARY:
        try:
            os.unlink(name)
        except OSError:
            pass
    log(f"VERDICT {payload['verdict']} in {payload['elapsed_seconds']:.2f}s; "
        f"device_status={payload['device_status']} "
        f"released={payload['release']['released']}; "
        f"failed={payload['failed']}; artifact {out}")
    return 0 if payload["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
